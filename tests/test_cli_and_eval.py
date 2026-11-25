import json
from pathlib import Path

from repolens.baseline_v1 import BaselineV1
from repolens.cli import main
from repolens.evaluation import Expected, load_questions, mean_metrics, score_ranking

ROOT = Path(__file__).resolve().parent.parent


def test_cli_search_json(sample_repo, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REPOLENS_CACHE_DIR", str(tmp_path / "cache"))
    assert main(["search", str(sample_repo), "hash password", "--no-embeddings", "--json", "-k", "2"]) == 0
    hits = json.loads(capsys.readouterr().out)
    assert hits[0]["path"] == "shop/auth.py"


def test_cli_index_text(sample_repo, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REPOLENS_CACHE_DIR", str(tmp_path / "cache"))
    assert main(["index", str(sample_repo), "--no-embeddings"]) == 0
    assert "chunks" in capsys.readouterr().out


def test_cli_reports_bad_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REPOLENS_CACHE_DIR", str(tmp_path / "cache"))
    assert main(["search", str(tmp_path / "nope"), "x", "--no-embeddings"]) == 2
    assert "Source not found" in capsys.readouterr().err


def test_score_ranking_metrics():
    expected = [Expected("a.py", 10, 20)]
    results = [("b.py", 1, 5), ("a.py", 30, 40), ("a.py", 15, 18)]
    scores = score_ranking(results, expected)
    assert scores == {"file_hit@1": 0.0, "file_hit@5": 1.0, "mrr@10": 0.5, "span_hit@5": 1.0}
    assert score_ranking([], expected)["mrr@10"] == 0.0
    assert mean_metrics([scores, {k: 0.0 for k in scores}])["mrr@10"] == 0.25


def test_benchmark_files_are_well_formed():
    repos = json.loads((ROOT / "eval" / "repos.json").read_text(encoding="utf-8"))
    questions = load_questions(ROOT / "eval" / "questions.jsonl")
    assert len(questions) >= 50
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert q.repo in repos and q.expected and q.style in {"natural", "identifier"}
        for e in q.expected:
            assert e.start_line <= e.end_line
    for spec in repos.values():
        assert len(spec["ref"]) == 40  # pinned to a full commit sha


def test_v1_baseline_still_runs(sample_repo):
    results = BaselineV1(sample_repo).search("hash password", top_k=3)
    assert results and all("start_line" in r for r in results)
