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

import json
import re
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    """Any failure talking to the model, with a message fit for end users."""


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


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[: -len("/chat/completions")]
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.extra_body = extra_body or {}

    # ------------------------------------------------------------ transport

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _hint(self) -> str:
        if _is_local(self.base_url):
            return f" Is Ollama running (`ollama serve`) and is the model pulled (`ollama pull {self.model}`)?"
        return " Check the base URL, model name and API key."

    def _open(self, path: str, body: dict[str, Any] | None = None, timeout: float | None = None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=self._headers(), method="POST" if data else "GET"
        )
        try:
            return urllib.request.urlopen(request, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            if exc.code == 401:
                raise LLMError("The model server rejected the API key (401)." + self._hint()) from exc
            if exc.code == 404:
                raise LLMError(f"Model or endpoint not found (404): {detail}." + self._hint()) from exc
            if exc.code == 429:
                raise LLMError("Rate limited by the model provider (429). Wait a moment and retry.") from exc
            raise LLMError(f"Model request failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, ConnectionError) as exc:
            reason = getattr(exc, "reason", exc)
            raise LLMError(f"Could not reach the model server at {self.base_url} ({reason})." + self._hint()) from exc
        except TimeoutError as exc:
            raise LLMError(f"The model did not respond within {self.timeout:.0f}s.") from exc

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
        body = {"model": self.model, "messages": messages, "temperature": temperature, "stream": True}
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


def make_chat_client(
    base_url: str,
    model: str,
    api_key: str | None = None,
    timeout: float = 600.0,
    num_ctx: int = 12_288,
) -> ChatClient:
    """Pick the native Ollama client for Ollama URLs, the OpenAI-compatible one otherwise."""
    if looks_like_ollama(base_url) and not api_key:
        return OllamaClient(base_url, model, timeout=timeout, num_ctx=num_ctx)
    return OpenAICompatibleClient(base_url, model, api_key, timeout)


def client_from_settings(settings) -> ChatClient:
    return make_chat_client(
        settings.llm_base_url, settings.llm_model, settings.llm_api_key, settings.llm_timeout, settings.ollama_num_ctx
    )


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
