"""One call to go from "a source the user gave us" to a ready :class:`CodeIndex`.

Shared by the CLI, the Streamlit app and the evaluation harness so all three
behave identically.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .embeddings import EmbeddingError, make_embedder
from .index import CodeIndex, ProgressFn
from .ingest import RepoSource, load_source


@dataclass
class Workspace:
    source: RepoSource
    index: CodeIndex
    semantic_error: str | None = None  # why embeddings were unavailable, if requested


def open_index(
    source: RepoSource,
    settings: Settings,
    semantic: bool = True,
    progress: ProgressFn | None = None,
    force: bool = False,
) -> Workspace:
    """Index a repository on disk, falling back to BM25-only if embeddings fail."""
    embedder = None
    error = None
    if semantic:
        try:
            embedder = make_embedder(settings)
        except EmbeddingError as exc:
            error = str(exc)
    try:
        index = CodeIndex.load_or_build(source.root, settings.cache_dir, embedder, progress, force)
    except EmbeddingError as exc:
        # Keyword search still works without the embedding server: degrade, do not fail.
        error = str(exc)
        index = CodeIndex.load_or_build(source.root, settings.cache_dir, None, progress, force)
    return Workspace(source, index, error)


def open_workspace(
    source: str,
    settings: Settings,
    semantic: bool = True,
    progress: ProgressFn | None = None,
    force: bool = False,
) -> Workspace:
    repo = load_source(source, settings.cache_dir, token=settings.github_token)
    return open_index(repo, settings, semantic, progress, force)
