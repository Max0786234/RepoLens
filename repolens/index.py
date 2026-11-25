"""The code index: files, structure-aware chunks, symbols, BM25 and embeddings.

Design notes:

* Build once, query many. v1 re-read and re-chunked the whole repository and
  reloaded the embedding model on *every question*. v2 builds an index once and
  caches it on disk, keyed by a fingerprint of file paths, sizes and mtimes.
* Embeddings are cached per chunk content hash, so editing one file only
  re-embeds the chunks that changed.
* Hybrid retrieval fuses BM25 and vector rankings with Reciprocal Rank Fusion;
  RRF needs no score calibration between the two very different scorers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np

from .chunking import Chunk, Symbol, chunk_file
from .languages import detect_language, is_doc, is_ignored_path, is_test_path
from .text import tokenize, tokenize_query

INDEX_VERSION = 3
MAX_FILE_BYTES = 1_000_000
MAX_FILES = 8_000

ProgressFn = Callable[[str, float], None]

_DOC_INTENT = {"readme", "doc", "docs", "documentation", "install", "setup", "usage", "guide", "tutorial", "changelog", "contributing", "license"}  # fmt: skip
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_TEST_INTENT = {"test", "tests", "testing", "spec", "fixture", "pytest", "unittest", "mock"}


@dataclass
class FileInfo:
    path: str
    language: str
    size: int
    lines: int
    sha1: str


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    rank: int
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {**self.chunk.to_dict(), "score": round(self.score, 5), "rank": self.rank, "sources": self.sources}


def _read_text(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw[:8192]:
        return None  # binary
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw.decode("latin-1")
        except UnicodeDecodeError:  # pragma: no cover - latin-1 decodes anything
            return None


def scan_files(root: Path) -> list[tuple[str, Path]]:
    """Indexable files below ``root`` as ``(posix_relative_path, absolute_path)``."""
    found: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if is_ignored_path(rel) or detect_language(rel) is None:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        found.append((rel, path))
        if len(found) >= MAX_FILES:
            break
    return found


def fingerprint(root: Path) -> str:
    digest = hashlib.sha1(f"v{INDEX_VERSION}".encode())
    for rel, path in scan_files(root):
        st = path.stat()
        digest.update(f"{rel}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def index_text(chunk: Chunk) -> str:
    """The text that represents a chunk for both BM25 and embeddings."""
    header = f"File: {chunk.path}"
    if chunk.symbol:
        header += f"\nSymbol: {chunk.symbol} ({chunk.kind})"
    return f"{header}\n{chunk.text}"


class BM25:
    """Okapi BM25 with field boosts for file path and symbol name."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.term_freqs: list[Counter[str]] = []
        self.lengths: list[int] = []
        for chunk in chunks:
            path = PurePosixPath(chunk.path)
            fields = tokenize(chunk.text)
            fields += tokenize(" ".join(path.parts)) * 2
            fields += tokenize(path.stem) * 2  # file name matters more than directories
            if chunk.symbol:
                fields += tokenize(chunk.symbol) * 3
            counts = Counter(fields)
            self.term_freqs.append(counts)
            self.lengths.append(sum(counts.values()))
        self.n = len(chunks)
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 1.0
        df: Counter[str] = Counter()
        for counts in self.term_freqs:
            df.update(counts.keys())
        self.idf = {term: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for term, f in df.items()}
        # Inverted index so scoring only touches chunks that contain a query term.
        self.postings: dict[str, list[int]] = {}
        for i, counts in enumerate(self.term_freqs):
            for term in counts:
                self.postings.setdefault(term, []).append(i)

    def scores(self, query_terms: list[str]) -> dict[int, float]:
        result: dict[int, float] = {}
        for term in set(query_terms):
            idf = self.idf.get(term)
            if idf is None:
                continue
            weight = query_terms.count(term)
            for i in self.postings[term]:
                tf = self.term_freqs[i][term]
                norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl))
                result[i] = result.get(i, 0.0) + weight * idf * norm
        return result


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Combine several rankings of chunk ids: ``score = sum(1 / (k + rank))``."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


class EmbeddingCache:
    """Persistent ``content_hash -> vector`` store, one file per embedding model."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.vectors: dict[str, np.ndarray] = {}
        if path and path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    keys, matrix = data["keys"], data["vectors"]
                self.vectors = {str(k): matrix[i] for i, k in enumerate(keys)}
            except (OSError, KeyError, ValueError):
                self.vectors = {}

    def save(self) -> None:
        if not self.path or not self.vectors:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.vectors.keys()))
        matrix = np.stack(list(self.vectors.values()))
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, vectors=matrix)
        tmp.replace(self.path)


class CodeIndex:
    """Searchable view of one repository."""

    def __init__(
        self,
        root: Path,
        files: list[FileInfo],
        chunks: list[Chunk],
        symbols: list[Symbol],
        embeddings: np.ndarray | None = None,
        embedder=None,
        built_at: float | None = None,
        fingerprint_value: str = "",
    ) -> None:
        self.root = Path(root).resolve()
        self.files = files
        self.file_map = {f.path: f for f in files}
        self.chunks = chunks
        self.symbols = symbols
        self.embeddings = embeddings
        self.embedder = embedder
        self.built_at = built_at or time.time()
        self.fingerprint = fingerprint_value
        self.bm25 = BM25(chunks)
        self._line_cache: dict[str, list[str]] = {}
        self._query_vectors: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ building

    @classmethod
    def build(
        cls,
        root: str | Path,
        embedder=None,
        cache_dir: Path | None = None,
        progress: ProgressFn | None = None,
    ) -> CodeIndex:
        root = Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Not a directory: {root}")
        report = progress or (lambda message, fraction: None)
        report("Scanning files", 0.0)
        entries = scan_files(root)
        files: list[FileInfo] = []
        chunks: list[Chunk] = []
        symbols: list[Symbol] = []
        for i, (rel, path) in enumerate(entries, start=1):
            text = _read_text(path)
            if text is None:
                continue
            language = detect_language(rel) or "text"
            file_chunks, file_symbols = chunk_file(rel, text, language)
            files.append(
                FileInfo(
                    rel, language, len(text.encode()), text.count("\n") + 1, hashlib.sha1(text.encode()).hexdigest()
                )
            )
            chunks.extend(file_chunks)
            symbols.extend(file_symbols)
            if i % 50 == 0:
                report(f"Chunking files ({i}/{len(entries)})", 0.3 * i / max(1, len(entries)))
        report(f"Chunked {len(files)} files into {len(chunks)} chunks", 0.3)
        embeddings = None
        if embedder is not None and chunks:
            embeddings = cls._embed_chunks(chunks, embedder, cache_dir, report)
        report("Index ready", 1.0)
        return cls(root, files, chunks, symbols, embeddings, embedder, fingerprint_value=fingerprint(root))

    @staticmethod
    def _embed_chunks(chunks: list[Chunk], embedder, cache_dir: Path | None, report: ProgressFn) -> np.ndarray:
        slug = hashlib.sha1(embedder.name.encode()).hexdigest()[:12]
        cache = EmbeddingCache(cache_dir / "embeddings" / f"{slug}.npz" if cache_dir else None)
        hashes = [c.content_hash for c in chunks]
        missing = [i for i, h in enumerate(hashes) if h not in cache.vectors]
        if missing:
            texts = [index_text(chunks[i]) for i in missing]

            def on_batch(done: int, total: int) -> None:
                report(f"Embedding chunks ({done}/{total})", 0.3 + 0.7 * done / max(1, total))

            vectors = embedder.embed_documents(texts, progress=on_batch)
            for i, vector in zip(missing, vectors):
                cache.vectors[hashes[i]] = vector
            cache.save()
        return np.stack([cache.vectors[h] for h in hashes]).astype(np.float32)

    # ------------------------------------------------------------ persistence

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": INDEX_VERSION,
            "root": str(self.root),
            "built_at": self.built_at,
            "fingerprint": self.fingerprint,
            "embedder": getattr(self.embedder, "name", None),
            "files": [f.__dict__ for f in self.files],
            "chunks": [c.to_dict() for c in self.chunks],
            "symbols": [s.to_dict() for s in self.symbols],
        }
        (directory / "index.json").write_text(json.dumps(meta), encoding="utf-8")
        if self.embeddings is not None:
            np.save(directory / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, directory: Path, embedder=None) -> CodeIndex | None:
        meta_path = directory / "index.json"
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("version") != INDEX_VERSION:
            return None
        embeddings = None
        if embedder is not None:
            if meta.get("embedder") != embedder.name or not (directory / "embeddings.npy").exists():
                return None
            embeddings = np.load(directory / "embeddings.npy")
        return cls(
            Path(meta["root"]),
            [FileInfo(**f) for f in meta["files"]],
            [Chunk(**c) for c in meta["chunks"]],
            [Symbol(**s) for s in meta["symbols"]],
            embeddings,
            embedder,
            meta["built_at"],
            meta["fingerprint"],
        )

    @classmethod
    def load_or_build(
        cls,
        root: str | Path,
        cache_dir: Path,
        embedder=None,
        progress: ProgressFn | None = None,
        force: bool = False,
    ) -> CodeIndex:
        """Reuse the on-disk index if the repository is unchanged, else rebuild it."""
        root = Path(root).resolve()
        key = hashlib.sha1(str(root).lower().encode()).hexdigest()[:16]
        directory = cache_dir / "indexes" / key
        if not force:
            cached = cls.load(directory, embedder)
            if cached is not None and cached.fingerprint == fingerprint(root):
                if progress:
                    progress("Loaded cached index", 1.0)
                return cached
        index = cls.build(root, embedder, cache_dir, progress)
        index.save(directory)
        return index

    # ------------------------------------------------------------ search

    @property
    def has_embeddings(self) -> bool:
        return self.embeddings is not None and self.embedder is not None and len(self.chunks) > 0

    def _prior(self, chunk: Chunk, query_terms: set[str]) -> float:
        """Mild multiplicative prior: prefer implementation code over docs and tests."""
        language = chunk.language
        factor = 1.0
        if is_doc(language) and not query_terms & _DOC_INTENT:
            factor *= 0.8
        if is_test_path(chunk.path) and not query_terms & _TEST_INTENT:
            factor *= 0.85
        return factor

    def bm25_ranking(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        terms = tokenize_query(query)
        raw_terms = {t.lower() for t in query.replace("?", " ").split()}
        identifiers = {m.lower() for m in _IDENT_RE.findall(query) if "_" in m or any(ch.isupper() for ch in m[1:])}
        scores = self.bm25.scores(terms)
        adjusted: dict[int, float] = {}
        for i, s in scores.items():
            chunk = self.chunks[i]
            s *= self._prior(chunk, raw_terms | set(terms))
            if chunk.symbol and identifiers:
                # A query naming an identifier should surface its definition first.
                name = chunk.symbol.split(" (part")[0].split(".")[-1].lower()
                if name in identifiers:
                    s *= 1.6
            adjusted[i] = s
        return sorted(adjusted.items(), key=lambda item: item[1], reverse=True)[:limit]

    def vector_ranking(self, query: str, limit: int = 50) -> list[tuple[int, float]]:
        if not self.has_embeddings:
            return []
        query_vector = self._query_vectors.get(query)
        if query_vector is None:
            # Query embedding can take seconds on CPU-only Ollama, and the agent often repeats queries.
            query_vector = self.embedder.embed_query(query)
            if len(self._query_vectors) < 256:
                self._query_vectors[query] = query_vector
        sims = self.embeddings @ query_vector
        order = np.argsort(-sims)[:limit]
        return [(int(i), float(sims[i])) for i in order]

    def search(
        self,
        query: str,
        top_k: int = 8,
        method: str = "auto",
        max_per_file: int | None = 3,
    ) -> list[SearchHit]:
        """Search chunks. ``method`` is ``bm25``, ``vector``, ``hybrid`` or ``auto``."""
        if method == "auto":
            method = "hybrid" if self.has_embeddings else "bm25"
        if method in {"vector", "hybrid"} and not self.has_embeddings:
            raise ValueError("This index was built without embeddings; use method='bm25'.")
        pool = max(50, top_k * 5)
        if method == "bm25":
            ranked = self.bm25_ranking(query, pool)
            sources = {i: ["bm25"] for i, _ in ranked}
        elif method == "vector":
            ranked = self.vector_ranking(query, pool)
            sources = {i: ["vector"] for i, _ in ranked}
        elif method == "hybrid":
            lexical = self.bm25_ranking(query, pool)
            semantic = self.vector_ranking(query, pool)
            fused = reciprocal_rank_fusion([[i for i, _ in lexical], [i for i, _ in semantic]])
            ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
            lexical_ids = {i for i, _ in lexical}
            semantic_ids = {i for i, _ in semantic}
            sources = {
                i: [name for name, ids in (("bm25", lexical_ids), ("vector", semantic_ids)) if i in ids]
                for i, _ in ranked
            }
        else:
            raise ValueError(f"Unknown search method: {method!r}")

        hits: list[SearchHit] = []
        per_file: Counter[str] = Counter()
        for chunk_id, score in ranked:
            chunk = self.chunks[chunk_id]
            if max_per_file and per_file[chunk.path] >= max_per_file:
                continue
            per_file[chunk.path] += 1
            hits.append(SearchHit(chunk, score, len(hits) + 1, sources.get(chunk_id, [])))
            if len(hits) >= top_k:
                break
        return hits

    # ------------------------------------------------------------ file access

    def resolve_path(self, path: str) -> str | None:
        """Map a user/model supplied path onto an indexed file path."""
        cleaned = path.strip().strip("`'\"").replace("\\", "/")
        while cleaned.startswith("./"):
            cleaned = cleaned[2:]
        cleaned = cleaned.lstrip("/")
        if cleaned in self.file_map:
            return cleaned
        # Models often drop leading directories: accept a unique suffix match.
        matches = [p for p in self.file_map if p.endswith("/" + cleaned)]
        if len(matches) == 1:
            return matches[0]
        lower = cleaned.lower()
        matches = [p for p in self.file_map if p.lower() == lower or p.lower().endswith("/" + lower)]
        return matches[0] if len(matches) == 1 else None

    def read_lines(self, path: str) -> list[str]:
        """Lines of an indexed file. Only indexed files are readable (sandbox boundary)."""
        resolved = self.resolve_path(path)
        if resolved is None:
            raise FileNotFoundError(path)
        cached = self._line_cache.get(resolved)
        if cached is not None:
            return cached
        full = (self.root / resolved).resolve()
        if self.root not in full.parents:
            raise PermissionError(path)
        text = _read_text(full)
        if text is None:
            raise FileNotFoundError(path)
        lines = text.splitlines()
        if len(self._line_cache) < 4_000:
            self._line_cache[resolved] = lines
        return lines

    # ------------------------------------------------------------ stats

    def stats(self) -> dict:
        languages = Counter(f.language for f in self.files)
        return {
            "files": len(self.files),
            "chunks": len(self.chunks),
            "symbols": len(self.symbols),
            "lines": sum(f.lines for f in self.files),
            "languages": dict(languages.most_common()),
            "embeddings": bool(self.has_embeddings),
            "embedder": getattr(self.embedder, "name", None),
        }
