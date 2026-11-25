"""Embedding backends for semantic search.

The default backend talks to the OpenAI-compatible ``/embeddings`` endpoint, so
the same code serves local Ollama (``embeddinggemma``, ``nomic-embed-text``) and
hosted providers. ``sentence-transformers`` is supported as an optional
in-process alternative.

Some embedding models are trained with task prefixes; using them measurably
improves retrieval, so they are applied automatically when known.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol

import numpy as np

# (query prefix, document prefix) per model family.
_PREFIXES: dict[str, tuple[str, str]] = {
    "embeddinggemma": ("task: code retrieval | query: ", "title: none | text: "),
    "nomic-embed": ("search_query: ", "search_document: "),
    "mxbai-embed": ("Represent this sentence for searching relevant passages: ", ""),
    "bge": ("Represent this sentence for searching relevant passages: ", ""),
    "e5": ("query: ", "passage: "),
}

MAX_EMBED_CHARS = 2_000  # keep within small embedding models' context windows


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _prefixes_for(model: str) -> tuple[str, str]:
    lower = model.lower()
    for key, value in _PREFIXES.items():
        if key in lower:
            return value
    return "", ""


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class OpenAICompatibleEmbedder:
    """Calls ``POST {base_url}/embeddings``. Works with Ollama, OpenAI and others."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        batch_size: int = 32,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout
        self.name = f"openai-compatible:{model}"
        self._query_prefix, self._doc_prefix = _prefixes_for(model)

    def _post(self, inputs: list[str]) -> list[list[float]]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = json.dumps({"model": self.model, "input": inputs}).encode()
        request = urllib.request.Request(f"{self.base_url}/embeddings", data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise EmbeddingError(f"Embedding request failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise EmbeddingError(
                f"Could not reach the embedding endpoint at {self.base_url}. "
                f"For Ollama run `ollama pull {self.model}` and make sure Ollama is running."
            ) from exc
        items = sorted(data["data"], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]

    def _embed(self, texts: list[str], progress=None) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[:MAX_EMBED_CHARS] for t in texts[i : i + self.batch_size]]
            vectors.extend(self._post(batch))
            if progress:
                progress(min(i + self.batch_size, len(texts)), len(texts))
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalise(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: list[str], progress=None) -> np.ndarray:
        return self._embed([self._doc_prefix + t for t in texts], progress)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self._query_prefix + text])[0]


class SentenceTransformerEmbedder:
    """In-process embeddings. Requires ``pip install repolens[st]``."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise EmbeddingError("sentence-transformers is not installed. Run: pip install 'repolens[st]'") from exc
        self.model = SentenceTransformer(model)
        self.name = f"sentence-transformers:{model}"
        self._query_prefix, self._doc_prefix = _prefixes_for(model)

    def embed_documents(self, texts: list[str], progress=None) -> np.ndarray:
        vectors = self.model.encode(
            [self._doc_prefix + t[:MAX_EMBED_CHARS] for t in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        if progress:
            progress(len(texts), len(texts))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vector = self.model.encode([self._query_prefix + text], normalize_embeddings=True)
        return np.asarray(vector, dtype=np.float32)[0]


def make_embedder(settings) -> Embedder | None:
    """Build the embedder described by :class:`repolens.config.Settings`."""
    backend = (settings.embed_backend or "none").lower()
    if backend in {"none", "off", "", "false"}:
        return None
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embed_model)
    return OpenAICompatibleEmbedder(
        settings.embed_model,
        settings.embed_base_url,
        api_key=settings.embed_api_key,
    )
