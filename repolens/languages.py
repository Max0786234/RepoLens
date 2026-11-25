"""File-type knowledge shared by ingestion, chunking and the UI.

Keeping this in one place fixes a v1 bug where the upload filter and the
retriever disagreed about which suffixes were indexable.
"""

from __future__ import annotations

from pathlib import PurePosixPath

LANGUAGE_BY_SUFFIX: dict[str, str] = {
    # code
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".dart": "dart",
    ".lua": "lua",
    ".r": "r",
    ".sh": "shell",
    ".bash": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".vue": "vue",
    ".svelte": "svelte",
    # docs
    ".md": "markdown",
    ".mdx": "markdown",
    ".rst": "rst",
    ".txt": "text",
    # config / markup
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".graphql": "graphql",
    ".proto": "protobuf",
    ".tf": "terraform",
}

LANGUAGE_BY_FILENAME: dict[str, str] = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "procfile": "text",
    "gemfile": "ruby",
    "rakefile": "ruby",
}

DOC_LANGUAGES = {"markdown", "rst", "text"}
CONFIG_LANGUAGES = {"json", "yaml", "toml", "ini", "xml", "dockerfile", "makefile", "terraform"}

# Directories that are almost never what a user wants to ask about.
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "bower_components",
    "dist",
    "build",
    "out",
    "target",
    ".next",
    ".nuxt",
    ".cache",
    ".idea",
    ".vscode",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".eggs",
    "site-packages",
    "coverage",
    "htmlcov",
    "vendor",
    ".repolens",
}

# Generated or lock files that add noise without answering questions.
IGNORED_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pipfile.lock",
    "cargo.lock",
    "composer.lock",
    "gemfile.lock",
    "go.sum",
    "uv.lock",
}

# Files the agent must never read, even if they look like text.
SECRET_FILENAMES = {".env", "id_rsa", "id_dsa", "id_ed25519", ".npmrc", ".pypirc", ".netrc"}
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".keystore"}


def detect_language(path: str) -> str | None:
    """Return the language name for a path, or ``None`` if it is not indexable."""
    p = PurePosixPath(path.replace("\\", "/"))
    name = p.name.lower()
    if name in LANGUAGE_BY_FILENAME:
        return LANGUAGE_BY_FILENAME[name]
    return LANGUAGE_BY_SUFFIX.get(p.suffix.lower())


def is_secret_path(path: str) -> bool:
    p = PurePosixPath(path.replace("\\", "/"))
    name = p.name.lower()
    if name in SECRET_FILENAMES or name.startswith(".env."):
        return True
    return p.suffix.lower() in SECRET_SUFFIXES


def is_ignored_path(path: str) -> bool:
    """True if any path component is an ignored directory or the file is noise."""
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if any(part in IGNORED_DIRS or part.endswith(".egg-info") for part in parts[:-1]):
        return True
    name = parts[-1].lower() if parts else ""
    if name in IGNORED_FILENAMES or ".min." in name or name.endswith(".map"):
        return True
    return is_secret_path(path)


def is_indexable(path: str) -> bool:
    return detect_language(path) is not None and not is_ignored_path(path)


def is_doc(language: str | None) -> bool:
    return language in DOC_LANGUAGES


def is_test_path(path: str) -> bool:
    lower = path.lower().replace("\\", "/")
    name = lower.rsplit("/", 1)[-1]
    return (
        "/tests/" in f"/{lower}"
        or "/test/" in f"/{lower}"
        or "/__tests__/" in f"/{lower}"
        or name.startswith("test_")
        or name.endswith(("_test.py", "_test.go", ".test.js", ".test.ts", ".spec.js", ".spec.ts"))
    )
