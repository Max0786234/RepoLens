"""Runtime settings, read from environment variables with local-first defaults.

Every LLM backend is reached through the OpenAI-compatible Chat Completions
API (plus Ollama's native API locally), so switching from local Ollama to a
hosted provider is only a matter of configuration:

    REPOLENS_PROVIDER=groq          # picks base URL, default model and budgets
    GROQ_API_KEY=...                # read from the environment or a local .env file

A ``.env`` file in the working directory is loaded automatically (it is
git-ignored); real environment variables always take precedence.
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
    # Token budgets for rate-limited tiers: the largest prompt the agent may send
    # and the cap on generated tokens per call. ``None`` means unlimited.
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None


PROVIDERS: dict[str, ProviderPreset] = {
    "ollama": ProviderPreset(
        "Ollama (local)",
        DEFAULT_OLLAMA_URL,
        DEFAULT_CHAT_MODEL,
        needs_api_key=False,
        notes="Runs on your machine. `ollama pull qwen3:4b-instruct`.",
    ),
    "groq": ProviderPreset(
        "Groq",
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
        needs_api_key=True,
        api_key_env="GROQ_API_KEY",
        notes="Free tier: 8k tokens/min per model. RepoLens paces requests and trims context to fit.",
        # One request must fit in the 8k tokens-per-minute window, output included.
        max_context_tokens=5_500,
        max_output_tokens=1_500,
    ),
    "openai": ProviderPreset(
        "OpenAI",
        "https://api.openai.com/v1",
        "gpt-4.1-mini",
        needs_api_key=True,
        api_key_env="OPENAI_API_KEY",
        max_output_tokens=2_000,
    ),
    "anthropic": ProviderPreset(
        "Anthropic (OpenAI-compatible)",
        "https://api.anthropic.com/v1",
        "claude-haiku-5-5",
        needs_api_key=True,
        api_key_env="ANTHROPIC_API_KEY",
        max_output_tokens=2_000,
    ),
    "custom": ProviderPreset(
        "Custom OpenAI-compatible",
        DEFAULT_OLLAMA_URL,
        DEFAULT_CHAT_MODEL,
        needs_api_key=False,
    ),
}


def load_dotenv(path: Path | str = ".env") -> None:
    """Minimal ``KEY=value`` loader (no dependency). Never overrides real env vars."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _int_env(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value) if value.lower() not in {"none", "0", "off"} else None


def default_cache_dir() -> Path:
    configured = os.environ.get("REPOLENS_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "repolens"


@dataclass
class Settings:
    """All tunables in one object. Construct with :meth:`from_env`."""

    provider: str = "ollama"
    llm_base_url: str = DEFAULT_OLLAMA_URL
    llm_model: str = DEFAULT_CHAT_MODEL
    llm_api_key: str | None = None
    llm_timeout: float = 600.0
    llm_max_context_tokens: int | None = None
    llm_max_output_tokens: int | None = None
    reasoning_effort: str | None = "low"
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

    def apply_provider(self, key: str) -> None:
        """Switch to a provider preset (base URL, default model, API key, budgets)."""
        preset = PROVIDERS[key]
        self.provider = key
        self.llm_base_url = preset.base_url
        self.llm_model = preset.default_model
        if preset.api_key_env:
            self.llm_api_key = os.environ.get(preset.api_key_env) or self.llm_api_key
        self.llm_max_context_tokens = preset.max_context_tokens
        self.llm_max_output_tokens = preset.max_output_tokens

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        settings = cls(
            llm_base_url=_env("OPENAI_BASE_URL", default=DEFAULT_OLLAMA_URL),
            llm_model=_env("OPENAI_MODEL", default=DEFAULT_CHAT_MODEL),
            llm_api_key=_env("OPENAI_API_KEY"),
            llm_timeout=float(_env("REPOLENS_LLM_TIMEOUT", default="600")),
            reasoning_effort=_env("REPOLENS_REASONING_EFFORT", default="low"),
            ollama_num_ctx=int(_env("REPOLENS_OLLAMA_NUM_CTX", default="12288")),
            embed_backend=_env("REPOLENS_EMBED_BACKEND", default="ollama").lower(),
            embed_model=_env("REPOLENS_EMBED_MODEL", default=DEFAULT_EMBED_MODEL),
            embed_base_url=_env("REPOLENS_EMBED_BASE_URL", default=DEFAULT_OLLAMA_URL),
            embed_api_key=_env("REPOLENS_EMBED_API_KEY"),
            cache_dir=default_cache_dir(),
            max_agent_steps=int(_env("REPOLENS_MAX_STEPS", default="6")),
            github_token=_env("GITHUB_TOKEN", "GH_TOKEN"),
        )
        provider = (_env("REPOLENS_PROVIDER") or "").lower()
        if provider in PROVIDERS and provider not in {"ollama", "custom"}:
            settings.apply_provider(provider)
        # Explicit REPOLENS_* values override any preset.
        settings.llm_base_url = _env("REPOLENS_LLM_BASE_URL", default=settings.llm_base_url)
        settings.llm_model = _env("REPOLENS_LLM_MODEL", default=settings.llm_model)
        settings.llm_api_key = _env("REPOLENS_LLM_API_KEY", default=settings.llm_api_key)
        settings.llm_max_context_tokens = _int_env("REPOLENS_MAX_CONTEXT_TOKENS", settings.llm_max_context_tokens)
        settings.llm_max_output_tokens = _int_env("REPOLENS_MAX_OUTPUT_TOKENS", settings.llm_max_output_tokens)
        return settings
