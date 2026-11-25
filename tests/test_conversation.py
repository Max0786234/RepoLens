import pytest

from repolens.agent import Agent
from repolens.conversation import ConversationMemory, contextual_query, is_follow_up
from repolens.llm import ChatResponse, ScriptedClient
from repolens.rag import retrieve
from repolens.tools import EvidenceRange


@pytest.mark.parametrize(
    "question",
    [
        "Are there tests for it?",
        "why?",
        "and the logout?",
        "What about errors?",
        "Where is that called?",
        "show me more",
    ],
)
def test_follow_ups_are_detected(question):
    assert is_follow_up(question)


@pytest.mark.parametrize(
    "question",
    [
        "How does the login manager hash passwords before comparing them?",
        "Where is compute_total defined and how is the discount applied?",
        "Explain the HTTP server routes in web/server.js",
    ],
)
def test_standalone_questions_are_not_follow_ups(question):
    assert not is_follow_up(question)


HISTORY = [
    {"role": "user", "content": "How are passwords hashed?"},
    {"role": "assistant", "content": "With hash_password in shop/auth.py:6-8."},
]


def test_contextual_query_adds_previous_question_only_for_follow_ups():
    assert contextual_query("Where is it called?", HISTORY) == "How are passwords hashed? Where is it called?"
    standalone = "How is the cart total computed with discounts?"
    assert contextual_query(standalone, HISTORY) == standalone
    assert contextual_query("Where is it called?", []) == "Where is it called?"


def test_follow_up_retrieval_keeps_the_topic(sample_index):
    without = retrieve(sample_index, "Where is it called?")
    with_history = retrieve(sample_index, "Where is it called?", history=HISTORY)
    assert with_history.hits[0].chunk.path == "shop/auth.py"
    assert [h.chunk.id for h in with_history.hits] != [h.chunk.id for h in without.hits]


def test_agent_seed_uses_context_and_mentions_earlier_files(sample_index):
    llm = ScriptedClient([ChatResponse("It is called by LoginManager.login (shop/auth.py:18-22).")])
    memory = ConversationMemory()
    memory.add_turn([EvidenceRange("shop/auth.py", 6, 8, "read_file")])
    result = Agent(sample_index, llm).ask("Where is it called?", history=HISTORY, memory=memory)
    seed_event = result.events[0]
    assert "passwords hashed" in seed_event.data["arguments"]["query"]
    first_user_message = llm.calls[0][-1]["content"]
    assert "Files examined earlier in this conversation" in first_user_message and "shop/auth.py" in first_user_message


def test_citations_to_code_seen_in_earlier_turns_are_verified(sample_index):
    memory = ConversationMemory()
    memory.add_turn([EvidenceRange("web/server.js", 1, 12, "read_file")])
    llm = ScriptedClient([ChatResponse("As shown before, the server is created in web/server.js:7-12.")])
    result = Agent(sample_index, llm).ask("and the tests?", history=HISTORY, memory=memory)
    assert result.grounding.count("verified") == 1
    # The turn's own evidence does not include the earlier range (so memory does not snowball).
    assert all(r.tool != "earlier_turn" for r in result.evidence.ranges)


def test_memory_files_are_recent_first_and_serialisable():
    memory = ConversationMemory()
    memory.add_turn([EvidenceRange("a.py", 1, 2, "x"), EvidenceRange("b.py", 1, 2, "x")])
    memory.add_turn([EvidenceRange("a.py", 5, 9, "x")])
    assert memory.files() == ["a.py", "b.py"]
    restored = ConversationMemory.from_list(memory.to_list())
    assert [r.path for r in restored.ranges] == ["a.py", "b.py", "a.py"]


def test_cli_chat_session(sample_repo, tmp_path, monkeypatch, capsys):
    import repolens.cli as cli

    monkeypatch.setenv("REPOLENS_CACHE_DIR", str(tmp_path / "cache"))
    client = ScriptedClient(
        [
            ChatResponse("Passwords are hashed in shop/auth.py:6-8."),
            ChatResponse("LoginManager.login calls it (shop/auth.py:18-22)."),
        ]
    )
    monkeypatch.setattr(cli, "client_from_settings", lambda settings: client)
    inputs = iter(["How are passwords hashed?", "Where is it called?", "/files", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert cli.main(["chat", str(sample_repo), "--no-embeddings", "--mode", "quick"]) == 0
    out = capsys.readouterr().out
    assert "follow-up" in out
    assert "citations verified" in out
    assert "  shop/auth.py" in out  # printed by /files: the file examined in earlier turns
    # The follow-up's prompt to the model included the earlier conversation.
    assert any("How are passwords hashed?" in m["content"] for m in client.calls[1])
