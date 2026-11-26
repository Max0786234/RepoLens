"""Smoke tests that drive the real Streamlit script with AppTest (no browser, no model)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("REPOLENS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("REPOLENS_EMBED_BACKEND", "none")
    # An unroutable port: the app must cope with no model server at all.
    monkeypatch.setenv("REPOLENS_LLM_BASE_URL", "http://127.0.0.1:9/v1")


def test_landing_page_renders_without_a_model_server():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("Understand any codebase" in m.value for m in at.markdown)
    assert [t.label for t in at.tabs] == ["GitHub URL", "Pull request", "Upload", "Examples"]


def test_index_sample_and_search_without_llm():
    at = AppTest.from_file(APP, default_timeout=120).run()
    next(b for b in at.button if b.label == "RepoLens (this app)").click().run()
    assert not at.exception
    assert at.session_state.workspace is not None
    at.sidebar.radio[0].set_value("search").run()
    at.chat_input[0].set_value("reciprocal rank fusion").run()
    assert not at.exception
    messages = at.session_state.messages
    assert messages[-1]["mode"] == "search"
    assert any(h["path"] == "repolens/index.py" for h in messages[-1]["hits"])


def _scripted(monkeypatch, responses):
    import repolens.llm as llm_module
    from repolens.llm import ScriptedClient

    client = ScriptedClient(list(responses))
    monkeypatch.setattr(llm_module, "make_chat_client", lambda *args, **kwargs: client)
    return client


def _indexed_app():
    at = AppTest.from_file(APP, default_timeout=120).run()
    next(b for b in at.button if b.label == "RepoLens (this app)").click().run()
    return at


def test_agent_mode_renders_answer_and_grounding(monkeypatch):
    from repolens.llm import ChatResponse, ToolCall

    _scripted(
        monkeypatch,
        [
            ChatResponse("", [ToolCall("1", "find_symbol", {"name": "reciprocal_rank_fusion"})]),
            ChatResponse("RRF is implemented in repolens/index.py:1-5 and fake/file.py:1-2."),
        ],
    )
    at = _indexed_app()
    at.chat_input[0].set_value("Where is RRF implemented?").run()
    assert not at.exception
    message = at.session_state.messages[-1]
    assert message["mode"] == "agent"
    assert message["grounding"]["invalid"] == 1  # fake/file.py does not exist
    assert any(e["data"].get("tool") == "find_symbol" for e in message["trace"] if e["kind"] == "tool_call")


def test_quick_mode_streams_answer(monkeypatch):
    from repolens.llm import ChatResponse

    _scripted(monkeypatch, [ChatResponse("Fusion happens in repolens/index.py:1-3.")])
    at = _indexed_app()
    at.sidebar.radio[0].set_value("quick").run()
    at.chat_input[0].set_value("reciprocal rank fusion").run()
    assert not at.exception
    message = at.session_state.messages[-1]
    assert message["mode"] == "quick" and "Fusion happens" in message["content"]
    assert message["hits"]


def test_server_api_key_is_never_sent_to_the_browser(monkeypatch):
    monkeypatch.setenv("REPOLENS_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_SERVER_SECRET_123")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    values = [str(w.value) for w in at.text_input] + [str(w.placeholder or "") for w in at.text_input]
    assert all("gsk_SERVER_SECRET_123" not in v for v in values)
    key_box = next(w for w in at.text_input if w.label == "API key")
    assert key_box.value == "" and "server" in (key_box.placeholder or "")


def test_public_demo_hides_settings_and_caps_questions(monkeypatch):
    from repolens.llm import ChatResponse

    monkeypatch.setenv("REPOLENS_PUBLIC_DEMO", "1")
    monkeypatch.setenv("REPOLENS_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_SERVER_SECRET_123")
    monkeypatch.setenv("REPOLENS_MAX_QUESTIONS", "1")
    client = _scripted(monkeypatch, [ChatResponse("Fusion is in repolens/index.py:1-3.")] * 3)
    at = _indexed_app()
    assert not any(w.label in {"API key", "Base URL"} for w in at.text_input)
    at.sidebar.radio[0].set_value("quick").run()
    at.chat_input[0].set_value("reciprocal rank fusion").run()
    at.chat_input[0].set_value("and the BM25 part?").run()
    assert not at.exception
    answers = [m for m in at.session_state.messages if m["role"] == "assistant"]
    assert len(answers) == 1  # the second question was refused
    assert any("public demo allows 1" in w.value for w in at.warning)
    assert len(client.calls) == 1


def _text_py_diff():
    """A one-line change inside repolens/text.py:stem (the sample app indexes RepoLens itself)."""
    lines = (Path(APP).parent / "repolens" / "text.py").read_text(encoding="utf-8").splitlines()
    number = next(i for i, line in enumerate(lines, start=1) if line.startswith("def stem("))
    old = lines[number]
    return number + 1, (
        "--- a/repolens/text.py\n+++ b/repolens/text.py\n"
        f"@@ -{number + 1} +{number + 1} @@\n-{old}\n+{old}  # reviewed\n"
    )


def test_explain_change_panel_reviews_a_pasted_diff(monkeypatch):
    from repolens.llm import ChatResponse

    number, diff = _text_py_diff()
    client = _scripted(monkeypatch, [ChatResponse(f"**Summary** A comment in repolens/text.py:{number}.")])
    at = _indexed_app()
    at.text_area(key="change_diff").set_value(diff).run()
    next(b for b in at.button if b.label == "Explain change").click().run()
    assert not at.exception
    message = at.session_state.messages[-1]
    assert message["mode"] == "change"
    assert "stem" in message["report"]  # the changed function is identified
    assert message["grounding"]["verified"] == 1
    assert "Changed code after the change" in client.calls[0][-1]["content"]


def test_explain_change_in_search_mode_needs_no_model(monkeypatch):
    _, diff = _text_py_diff()
    client = _scripted(monkeypatch, [])
    at = _indexed_app()
    at.sidebar.radio[0].set_value("search").run()
    at.text_area(key="change_diff").set_value(diff).run()
    next(b for b in at.button if b.label == "Explain change").click().run()
    assert not at.exception
    assert at.session_state.messages[-1]["mode"] == "change-report"
    assert client.calls == []
