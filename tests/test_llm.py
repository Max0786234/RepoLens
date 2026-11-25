import io
import json
import urllib.error

import pytest

from repolens import llm
from repolens.llm import (
    LLMError,
    OllamaClient,
    OpenAICompatibleClient,
    extract_text_tool_calls,
    make_chat_client,
    parse_response,
    strip_thinking,
)

TOOLS = {"search_code", "read_file"}


def test_strip_thinking():
    assert strip_thinking("<think>hmm</think>\nAnswer") == "Answer"
    assert strip_thinking("reasoning...</think>Answer") == "Answer"
    assert strip_thinking(None) == ""


def test_parse_native_tool_calls_with_string_or_dict_arguments():
    data = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {"id": "1", "function": {"name": "search_code", "arguments": '{"query": "auth"}'}},
                        {"id": "2", "function": {"name": "read_file", "arguments": {"path": "a.py"}}},
                        {"id": "3", "function": {"name": "read_file", "arguments": "{broken"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    response = parse_response(data, TOOLS)
    assert [c.arguments for c in response.tool_calls] == [{"query": "auth"}, {"path": "a.py"}, {"_raw": "{broken"}]
    assert response.usage == {"prompt_tokens": 10, "completion_tokens": 2}


@pytest.mark.parametrize(
    "content",
    [
        '{"name": "search_code", "arguments": {"query": "retry"}}',
        '```json\n{"name": "search_code", "parameters": {"query": "retry"}}\n```',
        '<tool_call>\n{"name": "search_code", "arguments": {"query": "retry"}}\n</tool_call>',
    ],
)
def test_tool_calls_written_as_text_are_recovered(content):
    response = parse_response({"choices": [{"message": {"content": content}}]}, TOOLS)
    assert response.tool_calls[0].name == "search_code"
    assert response.tool_calls[0].arguments == {"query": "retry"}
    assert response.content == ""


def test_text_json_for_unknown_tool_is_left_as_content():
    content = '{"name": "delete_everything", "arguments": {}}'
    assert extract_text_tool_calls(content, TOOLS) == []
    assert parse_response({"choices": [{"message": {"content": content}}]}, TOOLS).content == content


def test_malformed_response_raises():
    with pytest.raises(LLMError):
        parse_response({"error": "boom"})


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_openai_client_sends_tools_and_auth(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data)
        return FakeResponse(json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleClient("https://api.example.com/v1/", "m", api_key="sk-test")
    tools = [{"type": "function", "function": {"name": "search_code", "parameters": {}}}]
    assert client.chat([{"role": "user", "content": "x"}], tools=tools).content == "hi"
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["body"]["tools"] == tools


def test_ollama_client_uses_native_api_with_num_ctx(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data)
        reply = {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "search_code", "arguments": {"query": "q"}}}],
            },
            "done_reason": "stop",
            "prompt_eval_count": 50,
            "eval_count": 5,
        }
        return FakeResponse(json.dumps(reply).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    client = make_chat_client("http://localhost:11434/v1", "qwen3:4b-instruct", num_ctx=8192)
    assert isinstance(client, OllamaClient)
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "1", "name": "read_file", "content": "..."},
    ]
    response = client.chat(messages, tools=[{"type": "function", "function": {"name": "search_code"}}])
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["body"]["options"]["num_ctx"] == 8192
    assert captured["body"]["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"path": "a.py"}
    assert captured["body"]["messages"][1]["tool_name"] == "read_file"
    assert response.tool_calls[0].arguments == {"query": "q"}
    assert response.usage == {"prompt_tokens": 50, "completion_tokens": 5}


def test_connection_error_has_actionable_hint(monkeypatch):
    def refuse(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm.urllib.request, "urlopen", refuse)
    with pytest.raises(LLMError, match="ollama pull qwen3"):
        make_chat_client("http://localhost:11434/v1", "qwen3:4b-instruct").chat([])
    with pytest.raises(LLMError, match="API key"):
        OpenAICompatibleClient("https://api.example.com/v1", "m", "k").chat([])


def test_http_errors_are_translated(monkeypatch):
    def unauthorized(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr(llm.urllib.request, "urlopen", unauthorized)
    with pytest.raises(LLMError, match="401"):
        OpenAICompatibleClient("https://api.example.com/v1", "m", "k").chat([])


def test_streaming_skips_thinking(monkeypatch):
    events = [
        {"choices": [{"delta": {"content": "<think>secret"}}]},
        {"choices": [{"delta": {"content": " plan</think>Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
    ]
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"

    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda request, timeout: FakeResponse(body.encode()))
    client = OpenAICompatibleClient("https://api.example.com/v1", "m", "k")
    assert "".join(client.stream([])) == "Hello world"


def test_hosted_url_or_key_uses_openai_client():
    assert isinstance(make_chat_client("https://api.groq.com/openai/v1", "m", "key"), OpenAICompatibleClient)
    assert not isinstance(make_chat_client("https://api.groq.com/openai/v1", "m", "key"), OllamaClient)
