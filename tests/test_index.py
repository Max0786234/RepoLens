import os
import time

import pytest

from repolens.index import CodeIndex, fingerprint, reciprocal_rank_fusion


def test_index_skips_ignored_secret_and_binary_files(sample_index):
    paths = set(sample_index.file_map)
    assert "shop/auth.py" in paths and "web/server.js" in paths
    assert ".env" not in paths
    assert not any(p.startswith("node_modules/") for p in paths)
    assert "image.png" not in paths


def test_bm25_finds_function_by_natural_language(sample_index):
    hits = sample_index.search("how is the password hashed", top_k=3)
    assert hits[0].chunk.path == "shop/auth.py"
    assert hits[0].chunk.symbol == "hash_password"


def test_identifier_query_prefers_definition_over_test(sample_index):
    hits = sample_index.search("compute_total", top_k=3)
    assert hits[0].chunk.path == "shop/cart.py"
    assert hits[0].chunk.symbol == "compute_total"


def test_docs_penalised_unless_query_is_about_docs(sample_index):
    install = sample_index.search("installation instructions readme", top_k=2)
    assert install[0].chunk.path == "README.md"


def test_max_per_file_limits_duplicates(sample_index):
    hits = sample_index.search("login logout session user", top_k=5, max_per_file=1)
    assert len({h.chunk.path for h in hits}) == len(hits)


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert fused[1] == pytest.approx(fused[2])
    assert fused[1] > fused[3] and fused[1] > fused[4]


def test_vector_and_hybrid_search_with_fake_embedder(sample_repo, fake_embedder, tmp_path):
    index = CodeIndex.build(sample_repo, embedder=fake_embedder, cache_dir=tmp_path)
    assert index.has_embeddings
    vector = index.search("coupon discount", top_k=3, method="vector")
    assert vector[0].chunk.path == "shop/cart.py"
    hybrid = index.search("coupon discount", top_k=3, method="hybrid")
    assert hybrid[0].chunk.path == "shop/cart.py"
    assert "bm25" in hybrid[0].sources and "vector" in hybrid[0].sources


def test_vector_search_requires_embeddings(sample_index):
    with pytest.raises(ValueError):
        sample_index.search("x", method="vector")
    assert sample_index.search("password", method="auto")  # auto falls back to BM25


def test_embedding_cache_only_embeds_changed_chunks(sample_repo, fake_embedder, tmp_path):
    CodeIndex.build(sample_repo, embedder=fake_embedder, cache_dir=tmp_path)
    first = fake_embedder.document_calls
    CodeIndex.build(sample_repo, embedder=fake_embedder, cache_dir=tmp_path)
    assert fake_embedder.document_calls == first  # all cached
    (sample_repo / "shop" / "new.py").write_text("def brand_new():\n    return 42\n", encoding="utf-8")
    CodeIndex.build(sample_repo, embedder=fake_embedder, cache_dir=tmp_path)
    assert fake_embedder.document_calls == first + 1


def test_load_or_build_reuses_cache_until_files_change(sample_repo, tmp_path):
    first = CodeIndex.load_or_build(sample_repo, tmp_path)
    second = CodeIndex.load_or_build(sample_repo, tmp_path)
    assert second.built_at == first.built_at  # loaded from disk
    assert [c.id for c in second.chunks] == [c.id for c in first.chunks]
    before = fingerprint(sample_repo)
    target = sample_repo / "shop" / "cart.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    os.utime(target, (time.time() + 5, time.time() + 5))
    assert fingerprint(sample_repo) != before
    third = CodeIndex.load_or_build(sample_repo, tmp_path)
    assert third.built_at != first.built_at


def test_resolve_path_accepts_suffix_and_rejects_unknown(sample_index):
    assert sample_index.resolve_path("./shop/auth.py") == "shop/auth.py"
    assert sample_index.resolve_path("auth.py") == "shop/auth.py"
    assert sample_index.resolve_path("shop\\cart.py") == "shop/cart.py"
    assert sample_index.resolve_path("../../etc/passwd") is None
    with pytest.raises(FileNotFoundError):
        sample_index.read_lines(".env")


def test_stats(sample_index):
    stats = sample_index.stats()
    assert stats["files"] >= 5 and stats["symbols"] >= 6
    assert stats["languages"]["python"] >= 3
