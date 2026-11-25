from repolens.agent import Agent, AgentConfig
from repolens.llm import ChatResponse, LLMError, ScriptedClient, ToolCall
from repolens.rag import answer_question, build_context


def call(tool_name, **arguments):
    return ToolCall(f"id-{tool_name}", tool_name, arguments)


def test_agent_seeds_with_search_and_answers_with_verified_citation(sample_index):
    llm = ScriptedClient(
        [
            ChatResponse("", [call("read_file", path="shop/cart.py", start_line=4, end_line=7)]),
            ChatResponse("`compute_total` sums price x qty and applies the discount (shop/cart.py:4-7)."),
        ]
    )
    result = Agent(sample_index, llm).ask("How is the cart total computed?")
    assert result.stop_reason == "answered"
    assert result.grounding.count("verified") == 1
    assert result.tool_calls == 1 and result.steps == 2
    first_prompt = llm.calls[0]
    assert first_prompt[0]["role"] == "system" and "RepoLens" in first_prompt[0]["content"]
    assert "Initial search results" in first_prompt[-1]["content"]  # seeded before the first model call
    assert "shop/cart.py" in first_prompt[-1]["content"]
    kinds = [e.kind for e in result.events]
    assert kinds[0] == "tool_call" and kinds[-1] == "answer"


def test_tool_results_are_sent_back_to_the_model(sample_index):
    llm = ScriptedClient(
        [
            ChatResponse("", [call("find_symbol", name="LoginManager")]),
            ChatResponse("Done."),
        ]
    )
    Agent(sample_index, llm).ask("Where is LoginManager?")
    second = llm.calls[1]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"]
    assert second[-1]["role"] == "tool" and "shop/auth.py" in second[-1]["content"]


def test_budget_exhaustion_forces_a_final_answer(sample_index):
    looping = [ChatResponse("", [call("grep", pattern=f"term{i}")]) for i in range(3)]
    llm = ScriptedClient([*looping, ChatResponse("Final answer from evidence.")])
    result = Agent(sample_index, llm, AgentConfig(max_steps=3)).ask("Anything?")
    assert result.stop_reason == "budget"
    assert result.answer == "Final answer from evidence."
    assert "investigation budget" in llm.calls[-1][-1]["content"]


def test_duplicate_calls_are_short_circuited(sample_index):
    same = call("grep", pattern="hash")
    llm = ScriptedClient([ChatResponse("", [same]), ChatResponse("", [same]), ChatResponse("ok")])
    result = Agent(sample_index, llm).ask("q")
    assert result.tool_calls == 1  # the repeat was not executed again
    assert "already made this exact call" in llm.calls[2][-1]["content"]


def test_unknown_tools_and_bad_arguments_do_not_crash(sample_index):
    llm = ScriptedClient(
        [
            ChatResponse("", [call("delete_repo"), ToolCall("x", "read_file", {"_raw": "{oops"})]),
            ChatResponse("Recovered."),
        ]
    )
    result = Agent(sample_index, llm).ask("q")
    assert result.answer == "Recovered."
    errors = [e for e in result.events if e.kind == "tool_result" and e.data.get("is_error")]
    assert len(errors) == 2


def test_empty_reply_gets_one_nudge(sample_index):
    llm = ScriptedClient([ChatResponse(""), ChatResponse("Answer after nudge.")])
    assert Agent(sample_index, llm).ask("q").answer == "Answer after nudge."


def test_llm_errors_become_a_readable_answer(sample_index):
    def fail(messages):
        raise LLMError("Could not reach the model server")

    result = Agent(sample_index, ScriptedClient([fail])).ask("q")
    assert result.stop_reason == "error"
    assert "Could not reach" in result.answer
    assert any(e.kind == "error" for e in result.events)


def test_hallucinated_citation_produces_warning(sample_index):
    llm = ScriptedClient([ChatResponse("It is in shop/payments.py:1-10.")])
    result = Agent(sample_index, llm).ask("Where are payments?")
    assert result.grounding.count("invalid") == 1
    assert any(e.kind == "warning" for e in result.events)


def test_history_is_included(sample_index):
    llm = ScriptedClient([ChatResponse("ok")])
    history = [{"role": "user", "content": "earlier question"}, {"role": "assistant", "content": "earlier answer"}]
    Agent(sample_index, llm).ask("follow up", history=history)
    contents = [m["content"] for m in llm.calls[0]]
    assert "earlier question" in contents and "earlier answer" in contents


def test_result_serialises(sample_index):
    result = Agent(sample_index, ScriptedClient([ChatResponse("shop/cart.py:4")])).ask("compute the cart total")
    data = result.to_dict()
    assert data["grounding"]["verified"] == 1  # seed search showed cart.py
    assert data["trace"][0]["data"]["tool"] == "search_code"


def test_quick_rag_answer_is_grounded_in_retrieved_chunks(sample_index):
    llm = ScriptedClient([ChatResponse("Passwords are hashed in shop/auth.py:6-8.")])
    rag = answer_question(llm, sample_index, "How are passwords hashed?")
    assert rag.hits and rag.hits[0].chunk.path == "shop/auth.py"
    assert rag.grounding.count("verified") == 1
    assert "Source excerpts" in llm.calls[0][-1]["content"]


def test_build_context_respects_budget(sample_index):
    hits = sample_index.search("password login total server", top_k=6)
    context = build_context(hits, max_chars=300)
    assert context.count("---") <= 1


def test_repeated_duplicates_stop_the_investigation_early(sample_index):
    same = call("grep", pattern="hash")
    llm = ScriptedClient([ChatResponse("", [same])] * 3 + [ChatResponse("Answer from evidence.")])
    result = Agent(sample_index, llm, AgentConfig(max_steps=6)).ask("q")
    assert result.stop_reason == "stalled"
    assert result.answer == "Answer from evidence."
    assert result.steps == 3  # did not burn the remaining steps
