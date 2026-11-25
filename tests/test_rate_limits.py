"""Free-tier friendliness: pacing, retries, context budget and the response cache."""

import io
import json
import urllib.error

import pytest

from repolens import llm
from repolens.agent import TRIMMED_NOTE, Agent, AgentConfig, fit_context
from repolens.config import PROVIDERS, Settings
from repolens.llm import (
    CachedClient,
    ChatResponse,
    LLMError,
    OpenAICompatibleClient,
    RateLimiter,
    ScriptedClient,
    ToolCall,
    make_chat_client,
    parse_duration,
)


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, headers=None):
        super().__init__(data)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


OK_BODY = json.dumps(
    {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("7.66s", 7.66), ("1m2.5s", 62.5), ("120ms", 0.12), ("12", 12.0), ("2h", 7200.0), (None, None), ("??", None)],
)
def test_parse_duration(text, seconds):
    assert parse_duration(text) == (pytest.approx(seconds) if seconds is not None else None)


def test_rate_limiter_waits_for_token_window():
    waits = []
    clock = [100.0]
    limiter = RateLimiter(sleep=waits.append, clock=lambda: clock[0])
    limiter.update({"x-ratelimit-remaining-tokens": "500", "x-ratelimit-reset-tokens": "6s"})
    limiter.before_request(400)  # fits: no wait
    assert waits == []
    limiter.before_request(2_000)  # does not fit: wait for reset
    assert waits and waits[0] == pytest.approx(6.5)


def test_rate_limiter_refuses_very_long_waits():
    limiter = RateLimiter(max_wait=30, sleep=lambda s: None)
    with pytest.raises(LLMError, match="resets in about"):
        limiter.wait(3_600, "Daily limit.")


def test_429_is_retried_after_retry_after(monkeypatch):
    calls = {"n": 0}
    slept = []

    def fake_urlopen(request, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many", {"retry-after": "3"}, io.BytesIO(b"tpm"))
        return FakeResponse(OK_BODY.encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient("https://api.groq.com/openai/v1", "m", "key")
    client.limiter.sleep = slept.append
    assert client.chat([{"role": "user", "content": "hi"}]).content == "ok"
    assert calls["n"] == 2 and slept == [3.5]


def test_daily_quota_429_fails_fast_with_clear_message(monkeypatch):
    body = b'{"error": {"message": "Rate limit reached on tokens per day (TPD): Limit 200000"}}'

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many", {"retry-after": "600"}, io.BytesIO(body))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMError, match="Daily free-tier quota"):
        OpenAICompatibleClient("https://api.groq.com/openai/v1", "m", "key").chat([])


def test_gpt_oss_requests_low_reasoning_and_output_cap(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return FakeResponse(OK_BODY.encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    client = make_chat_client("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "key", max_tokens=1500)
    client.chat([{"role": "user", "content": "hi"}])
    assert captured["reasoning_effort"] == "low" and captured["max_tokens"] == 1500


def test_groq_preset_and_provider_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("GROQ_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("REPOLENS_PROVIDER", "groq")
    settings = Settings.from_env()
    preset = PROVIDERS["groq"]
    assert settings.llm_base_url == preset.base_url and settings.llm_model == preset.default_model
    assert settings.llm_api_key == "from-dotenv"
    assert settings.llm_max_context_tokens == preset.max_context_tokens
    monkeypatch.setenv("REPOLENS_MAX_CONTEXT_TOKENS", "4000")
    assert Settings.from_env().llm_max_context_tokens == 4000


def test_fit_context_trims_oldest_tool_output_first():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "Question: q\n\nInitial search results:\n" + "x" * 3000},
        {"role": "tool", "content": "a" * 6000},
        {"role": "tool", "content": "b" * 6000},
    ]
    trimmed = fit_context(messages, None, budget=3500)
    assert trimmed >= 1
    assert messages[2]["content"].endswith(TRIMMED_NOTE)
    assert messages[3]["content"] == "b" * 6000  # the latest result is kept intact
    assert fit_context(messages, None, budget=None) == 0


def test_agent_respects_context_budget(sample_index):
    big = ToolCall("1", "read_file", {"path": "shop/auth.py"})
    big2 = ToolCall("2", "grep", {"pattern": "def"})
    client = ScriptedClient([ChatResponse("", [big]), ChatResponse("", [big2]), ChatResponse("done")])
    Agent(sample_index, client, AgentConfig(max_context_tokens=900)).ask("How are passwords hashed?")
    last = client.calls[-1]
    assert sum(len(m.get("content") or "") for m in last) // 3 < 2_500  # bounded despite tool output


def test_cached_client_replays_without_spending(tmp_path):
    inner = ScriptedClient([ChatResponse("first", usage={"prompt_tokens": 100, "completion_tokens": 20})])
    cache = CachedClient(inner, tmp_path)
    messages = [{"role": "user", "content": "q"}]
    assert cache.chat(messages).content == "first"
    again = CachedClient(ScriptedClient([]), tmp_path)  # a new run: nothing scripted, must hit disk
    response = again.chat(messages)
    assert response.content == "first" and response.usage["cached"] == 1
    assert again.hits == 1 and again.tokens_spent == 0 and again.tokens_total == 120
    assert cache.tokens_spent == 120


def test_cached_client_disabled_is_pass_through(tmp_path):
    inner = ScriptedClient([ChatResponse("a"), ChatResponse("b")])
    cache = CachedClient(inner, tmp_path, enabled=False)
    messages = [{"role": "user", "content": "q"}]
    assert [cache.chat(messages).content, cache.chat(messages).content] == ["a", "b"]
    assert not any(tmp_path.iterdir())


def test_unparseable_tool_call_is_retried_by_the_agent(sample_index):
    from repolens.llm import ToolCallParseError

    def broken(messages):
        raise ToolCallParseError("unparseable", "thinking about DEFAULT_RETRIES")

    client = ScriptedClient([broken, ChatResponse("Hashing is in shop/auth.py:6-8.")])
    result = Agent(sample_index, client).ask("How are passwords hashed?")
    assert result.stop_reason == "answered" and "shop/auth.py" in result.answer
    assert any(e.kind == "warning" and "could not be parsed" in e.message for e in result.events)
    assert "could not be parsed" in client.calls[-1][-1]["content"]


def test_groq_output_parse_failed_maps_to_recoverable_error(monkeypatch):
    from repolens.llm import ToolCallParseError

    body = json.dumps({"error": {"code": "output_parse_failed", "failed_generation": "We need..."}}).encode()

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(body))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ToolCallParseError) as info:
        OpenAICompatibleClient("https://api.groq.com/openai/v1", "m", "key").chat([])
    assert info.value.failed_generation == "We need..."
