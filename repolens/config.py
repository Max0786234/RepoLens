"""Runtime settings, read from environment variables with local-first defaults.

Every LLM backend is reached through the OpenAI-compatible Chat Completions
API, so switching from local Ollama to a hosted provider is only a matter of
changing ``base_url``, ``model`` and ``api_key``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"
DEFAULT_CHAT_MODEL = "qwen3:4b-instruct"
DEFAULT_EMBED_MODEL = "embeddinggemma"


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    base_url: str
    default_model: str
    needs_api_key: bool
    api_key_env: str | None = None
    notes: str = ""


PROVIDERS: dict[str, ProviderPreset] = {
    "ollama": ProviderPreset(
        "Ollama (local)",
        DEFAULT_OLLAMA_URL,
        DEFAULT_CHAT_MODEL,
        needs_api_key=False,
        notes="Runs on your machine. `ollama pull qwen3:4b-instruct`.",
    ),
    "openai": ProviderPreset(
        "OpenAI",
        "https://api.openai.com/v1",
        "gpt-4.1-mini",
        needs_api_key=True,
        api_key_env="OPENAI_API_KEY",
    ),
    "groq": ProviderPreset(
        "Groq",
        "https://api.groq.com/openai/v1",
        "llama-3.3-70b-versatile",
        needs_api_key=True,
        api_key_env="GROQ_API_KEY",
        notes="Free tier available; good for a hosted demo.",
    ),
    "anthropic": ProviderPreset(
        "Anthropic (OpenAI-compatible)",
        "https://api.anthropic.com/v1",
        "claude-haiku-5-5",
        needs_api_key=True,
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "custom": ProviderPreset(
        "Custom OpenAI-compatible",
        DEFAULT_OLLAMA_URL,
        DEFAULT_CHAT_MODEL,
        needs_api_key=False,
    ),
}


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def default_cache_dir() -> Path:
    configured = os.environ.get("REPOLENS_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "repolens"


@dataclass
class Settings:
    """All tunables in one object. Construct with :meth:`from_env`."""

    llm_base_url: str = DEFAULT_OLLAMA_URL
    llm_model: str = DEFAULT_CHAT_MODEL
    llm_api_key: str | None = None
    llm_timeout: float = 600.0
    ollama_num_ctx: int = 12_288
    # "ollama" / "openai" use the OpenAI-compatible /embeddings endpoint,
    # "sentence-transformers" runs in-process, "none" disables semantic search.
    embed_backend: str = "ollama"
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_base_url: str = DEFAULT_OLLAMA_URL
    embed_api_key: str | None = None
    cache_dir: Path = field(default_factory=default_cache_dir)
    max_agent_steps: int = 6
    github_token: str | None = None

    @classmethod
    def from_env(cls) -> Settings:
        base_url = _env("REPOLENS_LLM_BASE_URL", "OPENAI_BASE_URL", default=DEFAULT_OLLAMA_URL)
        return cls(
            llm_base_url=base_url,
            llm_model=_env("REPOLENS_LLM_MODEL", "OPENAI_MODEL", default=DEFAULT_CHAT_MODEL),
            llm_api_key=_env("REPOLENS_LLM_API_KEY", "OPENAI_API_KEY"),
            llm_timeout=float(_env("REPOLENS_LLM_TIMEOUT", default="600")),
            ollama_num_ctx=int(_env("REPOLENS_OLLAMA_NUM_CTX", default="12288")),
            embed_backend=_env("REPOLENS_EMBED_BACKEND", default="ollama").lower(),
            embed_model=_env("REPOLENS_EMBED_MODEL", default=DEFAULT_EMBED_MODEL),
            embed_base_url=_env("REPOLENS_EMBED_BASE_URL", default=DEFAULT_OLLAMA_URL),
            embed_api_key=_env("REPOLENS_EMBED_API_KEY"),
            cache_dir=default_cache_dir(),
            max_agent_steps=int(_env("REPOLENS_MAX_STEPS", default="6")),
            github_token=_env("GITHUB_TOKEN", "GH_TOKEN"),
        )
