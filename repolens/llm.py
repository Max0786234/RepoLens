"""A dependency-free client for OpenAI-compatible Chat Completions APIs.

Works with Ollama (``http://localhost:11434/v1``), OpenAI, Groq, OpenRouter,
Anthropic's OpenAI-compatible endpoint and most self-hosted servers.

Small local models are not always well-behaved tool callers. The client
normalises the common failure modes so the agent loop does not have to:

* tool calls emitted as JSON in the message text (seen with qwen2.5-coder),
  optionally wrapped in code fences or ``<tool_call>`` tags,
* ``arguments`` returned as a dict instead of a JSON string, or invalid JSON,
* ``<think>...</think>`` reasoning blocks leaking into the answer.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import __version__


class LLMError(RuntimeError):
    """Any failure talking to the model, with a message fit for end users."""


class ToolCallParseError(LLMError):
    """The provider could not parse the model's tool call (Groq: ``output_parse_failed``).

    Recoverable: the agent tells the model and lets it try again.
    """

    def __init__(self, message: str, failed_generation: str = "") -> None:
        super().__init__(message)
        self.failed_generation = failed_generation


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_message(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None

    def to_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [call.to_message() for call in self.tool_calls]
        return message


class ChatClient(Protocol):
    model: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    ) -> ChatResponse: ...


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json|tool_call)?\s*(.*?)\s*```$", re.DOTALL)
_TOOL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def strip_thinking(text: str | None) -> str:
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # some servers drop the opening tag
        text = text.split("</think>", 1)[1]
    return text.strip()


def parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {"_raw": str(raw)}
    return value if isinstance(value, dict) else {"_raw": str(raw)}


def extract_text_tool_calls(content: str, tool_names: set[str]) -> list[ToolCall]:
    """Recover tool calls that a model wrote as JSON text instead of using the API field."""
    candidates = _TOOL_TAG_RE.findall(content)
    if not candidates:
        stripped = content.strip()
        fence = _FENCE_RE.match(stripped)
        candidates = [fence.group(1)] if fence else [stripped]
    calls: list[ToolCall] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate.startswith(("{", "[")):
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, dict):
                continue
            if "function" in item and isinstance(item["function"], dict):
                item = item["function"]
            name = item.get("name")
            if name not in tool_names:
                continue
            args = item.get("arguments", item.get("parameters", {}))
            calls.append(ToolCall(f"text-{uuid.uuid4().hex[:8]}", name, parse_arguments(args)))
    return calls


def parse_response(data: dict[str, Any], tool_names: set[str] | None = None) -> ChatResponse:
    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected response from the model server: {str(data)[:300]}") from exc
    content = strip_thinking(message.get("content"))
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        name = function.get("name")
        if not name:
            continue
        calls.append(
            ToolCall(raw.get("id") or f"call-{uuid.uuid4().hex[:8]}", name, parse_arguments(function.get("arguments")))
        )
    if not calls and tool_names and content:
        calls = extract_text_tool_calls(content, tool_names)
        if calls:
            content = ""
    usage = {k: v for k, v in (data.get("usage") or {}).items() if isinstance(v, int)}
    return ChatResponse(content, calls, usage, choice.get("finish_reason"))


def _is_local(url: str) -> bool:
    return any(host in url for host in ("localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"))


_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def parse_duration(value: str | None) -> float | None:
    """Parse rate-limit reset values such as ``"7.66s"``, ``"1m2.5s"``, ``"120ms"`` or ``"12"``."""
    if not value:
        return None
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        pass
    parts = _DURATION_RE.findall(value)
    if not parts:
        return None
    scale = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    return sum(float(number) * scale[unit] for number, unit in parts)


def estimate_tokens(payload: Any) -> int:
    """Cheap, deliberately pessimistic token estimate (~3 characters per token for code/JSON)."""
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return len(text) // 3 + 1


class RateLimiter:
    """Pace requests using the provider's ``x-ratelimit-*`` headers.

    Free tiers (e.g. Groq) limit tokens per minute. Instead of firing a request
    that will be rejected, wait until the token window resets. Waits longer than
    ``max_wait`` (a daily quota, typically) raise an :class:`LLMError` instead.
    """

    def __init__(self, max_wait: float = 90.0, sleep=time.sleep, clock=time.monotonic) -> None:
        self.max_wait = max_wait
        self.sleep = sleep
        self.clock = clock
        self.remaining_tokens: int | None = None
        self.reset_at: float | None = None
        self.on_wait: Any = None  # optional callback(seconds, reason) for UIs

    def update(self, headers) -> None:
        if headers is None:
            return
        remaining = headers.get("x-ratelimit-remaining-tokens")
        reset = parse_duration(headers.get("x-ratelimit-reset-tokens"))
        if remaining is not None:
            try:
                self.remaining_tokens = int(float(remaining))
            except ValueError:
                self.remaining_tokens = None
        if reset is not None:
            self.reset_at = self.clock() + reset

    def wait(self, seconds: float, reason: str) -> None:
        if seconds > self.max_wait:
            raise LLMError(f"{reason} The limit resets in about {seconds / 60:.0f} minutes.")
        if self.on_wait:
            self.on_wait(seconds, reason)
        self.sleep(seconds)

    def before_request(self, estimated_tokens: int) -> None:
        if self.remaining_tokens is None or self.reset_at is None:
            return
        if estimated_tokens > self.remaining_tokens:
            pause = self.reset_at - self.clock()
            if pause > 0:
                self.wait(pause + 0.5, "Pausing for the provider's tokens-per-minute limit.")
            self.remaining_tokens = None


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        max_retries: int = 3,
        max_wait: float = 90.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[: -len("/chat/completions")]
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.limiter = RateLimiter(max_wait=max_wait)

    # ------------------------------------------------------------ transport

    def _headers(self) -> dict[str, str]:
        # Some providers sit behind Cloudflare, which rejects urllib's default
        # "Python-urllib" User-Agent with 403 / error 1010.
        headers = {"Content-Type": "application/json", "User-Agent": f"RepoLens/{__version__}"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _hint(self) -> str:
        if _is_local(self.base_url):
            return f" Is Ollama running (`ollama serve`) and is the model pulled (`ollama pull {self.model}`)?"
        return " Check the base URL, model name and API key."

    def _open(self, path: str, body: dict[str, Any] | None = None, timeout: float | None = None):
        data = json.dumps(body).encode() if body is not None else None
        if data is not None:
            self.limiter.before_request(len(data) // 3 + int((body or {}).get("max_tokens") or 1000))
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}{path}", data=data, headers=self._headers(), method="POST" if data else "GET"
            )
            try:
                response = urllib.request.urlopen(request, timeout=timeout or self.timeout)
                self.limiter.update(getattr(response, "headers", None))
                return response
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:500]
                if exc.code == 401:
                    raise LLMError("The model server rejected the API key (401)." + self._hint()) from exc
                if exc.code == 404:
                    raise LLMError(f"Model or endpoint not found (404): {detail}." + self._hint()) from exc
                if exc.code == 413:
                    raise LLMError(
                        "The request is larger than the provider allows (413). Lower the context budget "
                        "(REPOLENS_MAX_CONTEXT_TOKENS) or ask a narrower question."
                    ) from exc
                if exc.code == 429:
                    headers = exc.headers or {}
                    wait = parse_duration(headers.get("retry-after")) or parse_duration(
                        headers.get("x-ratelimit-reset-tokens")
                    )
                    daily = "per day" in detail.lower() or "(rpd)" in detail.lower() or "(tpd)" in detail.lower()
                    if daily:
                        raise LLMError(f"Daily free-tier quota reached for {self.model}. {detail[:200]}") from exc
                    if attempt < self.max_retries:
                        self.limiter.wait((wait or 2.0 * (attempt + 1)) + 0.5, "Rate limited (429), retrying.")
                        continue
                    raise LLMError("Rate limited by the model provider (429). Wait a moment and retry.") from exc
                if exc.code >= 500 and attempt < self.max_retries:
                    self.limiter.wait(2.0 * (attempt + 1), f"Provider error {exc.code}, retrying.")
                    continue
                if exc.code == 400 and "output_parse_failed" in detail:
                    try:
                        failed = json.loads(detail).get("error", {}).get("failed_generation", "")
                    except (json.JSONDecodeError, AttributeError):
                        failed = ""
                    raise ToolCallParseError("The model produced an unparseable tool call.", failed) from exc
                raise LLMError(f"Model request failed ({exc.code}): {detail}") from exc
            except (urllib.error.URLError, ConnectionError) as exc:
                reason = getattr(exc, "reason", exc)
                raise LLMError(
                    f"Could not reach the model server at {self.base_url} ({reason})." + self._hint()
                ) from exc
            except TimeoutError as exc:
                raise LLMError(f"The model did not respond within {self.timeout:.0f}s.") from exc
        raise LLMError("Model request failed after retries.")  # pragma: no cover - loop always returns/raises

    # ------------------------------------------------------------ API

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    ) -> ChatResponse:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature, "stream": False}
        if tools:
            body["tools"] = tools
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        body.update(self.extra_body)
        with self._open("/chat/completions", body) as response:
            try:
                data = json.load(response)
            except json.JSONDecodeError as exc:
                raise LLMError("The model server returned invalid JSON.") from exc
        names = {t["function"]["name"] for t in tools} if tools else None
        return parse_response(data, names)

    def stream(self, messages: list[dict[str, Any]], temperature: float = 0.1) -> Iterator[str]:
        """Yield answer text as it is generated (server-sent events)."""
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature, "stream": True}
        if self.max_tokens:
            body["max_tokens"] = self.max_tokens
        body.update(self.extra_body)
        in_think = False
        with self._open("/chat/completions", body) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                piece = delta.get("content") or ""
                if "<think>" in piece:
                    in_think = True
                    piece = piece.split("<think>", 1)[0]
                if in_think:
                    if "</think>" in piece:
                        in_think = False
                        piece = piece.split("</think>", 1)[1]
                    else:
                        continue
                if piece:
                    yield piece

    def list_models(self) -> list[str]:
        """Model ids advertised by the server (used for health checks in the UI)."""
        with self._open("/models", timeout=5) as response:
            data = json.load(response)
        return sorted(item.get("id", "") for item in data.get("data", []) if item.get("id"))


class OllamaClient(OpenAICompatibleClient):
    """Ollama's native ``/api/chat`` endpoint.

    Ollama's OpenAI-compatible endpoint cannot set the context window, and its
    default is small; once a conversation exceeds it Ollama silently drops the
    oldest messages - including the system prompt. The native API accepts
    ``options.num_ctx``, so an agent with several tool results stays coherent.
    """

    def __init__(
        self, base_url: str, model: str, api_key: str | None = None, timeout: float = 600.0, num_ctx: int = 12_288
    ) -> None:
        root = base_url.rstrip("/")
        for suffix in ("/chat/completions", "/v1", "/api"):
            if root.endswith(suffix):
                root = root[: -len(suffix)]
        super().__init__(root, model, api_key, timeout)
        self.num_ctx = num_ctx

    @staticmethod
    def _to_native(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        native = []
        for message in messages:
            item = {"role": message["role"], "content": message.get("content") or ""}
            if message.get("tool_calls"):
                item["tool_calls"] = [
                    {
                        "function": {
                            "name": c["function"]["name"],
                            "arguments": parse_arguments(c["function"].get("arguments")),
                        }
                    }
                    for c in message["tool_calls"]
                ]
            if message["role"] == "tool" and message.get("name"):
                item["tool_name"] = message["name"]
            native.append(item)
        return native

    def _options(self, temperature: float) -> dict[str, Any]:
        return {"temperature": temperature, "num_ctx": self.num_ctx}

    def chat(self, messages, tools=None, temperature: float = 0.1) -> ChatResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_native(messages),
            "stream": False,
            "options": self._options(temperature),
        }
        if tools:
            body["tools"] = tools
        with self._open("/api/chat", body) as response:
            data = json.load(response)
        message = data.get("message") or {}
        openai_shape = {
            "choices": [
                {
                    "message": {
                        "content": message.get("content"),
                        "tool_calls": [
                            {"id": f"call-{uuid.uuid4().hex[:8]}", "function": c.get("function", {})}
                            for c in message.get("tool_calls") or []
                        ],
                    },
                    "finish_reason": data.get("done_reason"),
                }
            ],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        }
        names = {t["function"]["name"] for t in tools} if tools else None
        return parse_response(openai_shape, names)

    def stream(self, messages, temperature: float = 0.1) -> Iterator[str]:
        body = {
            "model": self.model,
            "messages": self._to_native(messages),
            "stream": True,
            "options": self._options(temperature),
        }
        in_think = False
        with self._open("/api/chat", body) as response:
            for raw_line in response:
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                piece = (chunk.get("message") or {}).get("content") or ""
                if "<think>" in piece:
                    in_think, piece = True, piece.split("<think>", 1)[0]
                if in_think:
                    if "</think>" not in piece:
                        continue
                    in_think, piece = False, piece.split("</think>", 1)[1]
                if piece:
                    yield piece
                if chunk.get("done"):
                    break

    def list_models(self) -> list[str]:
        with self._open("/api/tags", timeout=5) as response:
            data = json.load(response)
        return sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))


def looks_like_ollama(base_url: str) -> bool:
    return ":11434" in base_url or "ollama" in base_url.lower()


def model_options(model: str, reasoning_effort: str | None = "low") -> dict[str, Any]:
    """Provider/model-specific request fields that save tokens.

    gpt-oss models are reasoning models; ``reasoning_effort=low`` keeps hidden
    reasoning short, which matters on token-limited free tiers.
    """
    if "gpt-oss" in model and reasoning_effort:
        return {"reasoning_effort": reasoning_effort}
    if "qwen3" in model.lower() and reasoning_effort:
        # Hosted Qwen3 accepts "none" | "default"; "low" (our frugal default) maps to no hidden thinking.
        return {"reasoning_effort": "none" if reasoning_effort in {"low", "none"} else "default"}
    return {}


def make_chat_client(
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout: float = 600.0,
    num_ctx: int = 12_288,
    max_tokens: int | None = None,
    reasoning_effort: str | None = "low",
) -> ChatClient:
    """Pick the native Ollama client for Ollama URLs, the OpenAI-compatible one otherwise."""
    if looks_like_ollama(base_url) and not api_key:
        return OllamaClient(base_url, model, timeout=timeout, num_ctx=num_ctx)
    return OpenAICompatibleClient(
        base_url, model, api_key, timeout, extra_body=model_options(model, reasoning_effort), max_tokens=max_tokens
    )


def client_from_settings(settings) -> ChatClient:
    return make_chat_client(
        settings.llm_base_url,
        settings.llm_model,
        settings.llm_api_key,
        settings.llm_timeout,
        settings.ollama_num_ctx,
        settings.llm_max_output_tokens,
        settings.reasoning_effort,
    )


class CachedClient:
    """Disk cache of chat responses keyed by the exact request.

    Re-running a benchmark (or asking an identical question) costs zero tokens,
    which keeps free-tier quotas for new work. Opt-in: used by ``repolens eval``.
    """

    def __init__(self, inner: ChatClient, directory: Path, enabled: bool = True) -> None:
        self.inner = inner
        self.model = inner.model
        self.directory = Path(directory)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.tokens_total = 0  # tokens of all responses, cached or not
        self.tokens_spent = 0  # tokens actually billed by the provider in this session

    @staticmethod
    def _tokens(usage: dict[str, int]) -> int:
        return int(usage.get("prompt_tokens", 0)) + int(usage.get("completion_tokens", 0))

    def _key(self, messages, tools, temperature) -> str:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "tools": tools, "temperature": temperature},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def chat(self, messages, tools=None, temperature: float = 0.1) -> ChatResponse:
        key = self._key(messages, tools, temperature)
        path = self.directory / key[:2] / f"{key}.json"
        if self.enabled and path.exists():
            self.hits += 1
            data = json.loads(path.read_text(encoding="utf-8"))
            calls = [ToolCall(c["id"], c["name"], c["arguments"]) for c in data["tool_calls"]]
            self.tokens_total += self._tokens(data["usage"])
            return ChatResponse(data["content"], calls, {**data["usage"], "cached": 1}, data.get("finish_reason"))
        self.misses += 1
        response = self.inner.chat(messages, tools=tools, temperature=temperature)
        spent = self._tokens(response.usage)
        self.tokens_total += spent
        self.tokens_spent += spent
        if not self.enabled:
            return response
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "content": response.content,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in response.tool_calls],
            "usage": response.usage,
            "finish_reason": response.finish_reason,
        }
        path.write_text(json.dumps(record), encoding="utf-8")
        return response

    def stream(self, messages, temperature: float = 0.1) -> Iterator[str]:
        yield self.chat(messages, temperature=temperature).content

    def list_models(self) -> list[str]:
        return self.inner.list_models()


class ScriptedClient:
    """A deterministic stand-in for a model, used by tests and offline demos.

    ``script`` is a list of :class:`ChatResponse` objects (or callables that take
    the message list and return one) consumed in order.
    """

    def __init__(self, script: list[Any], model: str = "scripted") -> None:
        self.script = list(script)
        self.model = model
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools=None, temperature: float = 0.1) -> ChatResponse:
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            return ChatResponse("I have no further scripted responses.")
        item = self.script.pop(0)
        return item(messages) if callable(item) else item

    def stream(self, messages, temperature: float = 0.1) -> Iterator[str]:
        response = self.chat(messages)
        for word in response.content.split(" "):
            yield word + " "

    def list_models(self) -> list[str]:
        return [self.model]
