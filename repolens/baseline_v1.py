"""The original v1 retriever, frozen for benchmarking.

This is the lexical path of ``retriever.py`` from RepoLens v1 (commit 4d510af),
reformatted but behaviourally unchanged: fixed 45-line windows with 8 lines of
overlap, plain ``[A-Za-z_]\\w*`` tokens, BM25 plus a +3 path bonus and a
README penalty. The evaluation harness runs it next to the v2 retrievers so the
README can report a real before/after comparison.

Do not "fix" this file - its value is that it reproduces v1.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")
IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "dist", "build"}
TEXT_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".h", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".html", ".css"}  # fmt: skip
CODE_FILE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".cpp", ".h", ".c", ".cs", ".php", ".rb", ".swift", ".kt"}  # fmt: skip


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def file_priority(path: str) -> float:
    lower = path.lower()
    if "readme" in lower or lower.endswith(".md"):
        return -1.5
    if Path(path).suffix.lower() in CODE_FILE_SUFFIXES:
        return 1.5
    return 0.0


def collect_chunks(root: str | Path, chunk_lines: int = 45, overlap: int = 8) -> list[dict[str, Any]]:
    root = Path(root).resolve()
    chunks = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES or IGNORE_DIRS.intersection(path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        step = max(1, chunk_lines - overlap)
        for start in range(0, len(lines), step):
            block = lines[start : start + chunk_lines]
            if not any(line.strip() for line in block):
                continue
            chunks.append(
                {
                    # v1 used str(); as_posix() only normalises Windows separators.
                    "path": path.relative_to(root).as_posix(),
                    "start_line": start + 1,
                    "end_line": start + len(block),
                    "text": "\n".join(block),
                }
            )
            if start + chunk_lines >= len(lines):
                break
    return chunks


def bm25_search(query: str, chunks: list[dict[str, Any]], top_k: int = 8) -> list[dict[str, Any]]:
    q = tokenize(query)
    if not q or not chunks:
        return []
    docs = []
    for c in chunks:
        path_tokens = tokenize(c["path"].replace("/", " "))
        filename = Path(c["path"]).stem.lower()
        file_hint_tokens = tokenize(filename.replace("_", " "))
        docs.append(
            tokenize(
                c["text"]
                + " "
                + c["path"].replace("/", " ")
                + " "
                + " ".join(file_hint_tokens)
                + " "
                + " ".join(path_tokens)
            )
        )
    df = Counter(token for doc in docs for token in set(doc))
    avgdl = sum(map(len, docs)) / len(docs) or 1
    k1, b = 1.5, 0.75
    scored = []
    for idx, doc in enumerate(docs):
        counts = Counter(doc)
        score = 0.0
        path_tokens = set(tokenize(chunks[idx]["path"].lower().replace("/", " ")))
        for term in q:
            freq = counts[term]
            if not freq:
                continue
            idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * len(doc) / avgdl))
            if term in path_tokens:
                score += 3.0
        score += file_priority(chunks[idx]["path"])
        if score > 0:
            scored.append((score, idx))
    scored.sort(reverse=True)
    return [{**chunks[i], "score": round(s, 5), "rank": r} for r, (s, i) in enumerate(scored[:top_k], 1)]


class BaselineV1:
    """Adapter exposing v1 search through the same shape the harness expects."""

    def __init__(self, root: str | Path) -> None:
        self.chunks = collect_chunks(root)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        lexical = bm25_search(query, self.chunks, top_k=max(top_k * 3, 20))
        return sorted(lexical[:top_k], key=lambda x: (x.get("score", 0), file_priority(x["path"])), reverse=True)
