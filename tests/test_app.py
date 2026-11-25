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
    assert any("RepoLens" in t.value for t in at.title)
    assert len(at.tabs) == 3


def test_index_sample_and_search_without_llm():
    at = AppTest.from_file(APP, default_timeout=120).run()
    next(b for b in at.button if b.label == "Index RepoLens").click().run()
    assert not at.exception
    assert at.session_state.workspace is not None
    at.sidebar.radio[0].set_value("search").run()
    at.chat_input[0].set_value("reciprocal rank fusion").run()
    assert not at.exception
    messages = at.session_state.messages
    assert messages[-1]["mode"] == "search"
    assert any(h["path"] == "repolens/index.py" for h in messages[-1]["hits"])
