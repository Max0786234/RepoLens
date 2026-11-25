"""The RepoLens investigation agent.

Loop (see the README's "Design decisions"):

1. **Seed** - run ``search_code`` on the question before the model's first
   turn and put the results in the prompt. Small local models answer far more
   reliably when they start from evidence instead of having to decide to search.
2. **Investigate** - the model may call tools (search, read, grep, symbols,
   references) for up to ``max_steps`` turns. Identical repeated calls are
   short-circuited, and tool errors are returned to the model as text.
3. **Answer** - when the model stops calling tools, or the budget runs out, it
   must answer from the evidence gathered so far (it never ends in a crash).
4. **Verify** - every ``path:line`` citation in the answer is checked against
   the evidence log (:mod:`repolens.citations`).

``Agent.run`` is a generator of :class:`AgentEvent` objects so the CLI and the
Streamlit UI can show the investigation live.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .citations import GroundingReport, verify_citations
from .conversation import ConversationMemory, contextual_query
from .index import CodeIndex
from .llm import ChatClient, ChatResponse, LLMError, ToolCallParseError, estimate_tokens
from .tools import EvidenceLog, ToolBox, overview_text, tool_call_summary

SYSTEM_PROMPT = """You are RepoLens, a senior software engineer who answers questions about a code repository by investigating its source code.

Repository: {label}
{overview}

How to work:
- Initial search results for the question are provided. If they already answer it, answer immediately.
- Otherwise use tools to gather evidence: search_code to locate code, read_file to see exact lines, find_symbol and find_references to follow definitions and call sites, grep for exact strings, list_files for structure.
- Make focused tool calls and never repeat an identical call. Stop investigating as soon as the evidence is sufficient.

Answer rules:
- Start with a direct answer, then explain briefly (short paragraphs or bullet points).
- Back every claim with a citation in the form `path:start-end`, for example `src/app.py:10-24`. Only cite line ranges that appeared in tool results.
- Quote small code excerpts only when they help.
- If the evidence is insufficient, say exactly what is missing. Never invent files, functions or behaviour."""

PARSE_RETRY_PROMPT = (
    "Your previous reply could not be parsed. Either call exactly one tool with valid JSON arguments, "
    "or write your final answer as plain text with `path:start-end` citations."
)

FINAL_ANSWER_PROMPT = (
    "You have used your investigation budget. Using only the evidence gathered above, "
    "write your final answer now, with `path:start-end` citations. Do not call tools."
)


@dataclass
class AgentConfig:
    max_steps: int = 6
    seed_top_k: int = 4
    seed_snippet_chars: int = 900
    max_calls_per_step: int = 4
    max_duplicate_calls: int = 2  # stop investigating once the model repeats itself this often
    temperature: float = 0.1
    history_turns: int = 3
    # Largest prompt (messages + tool schemas, estimated tokens) sent in one call.
    # ``None`` = unlimited (local models). Free hosted tiers need a budget.
    max_context_tokens: int | None = None

    @classmethod
    def from_settings(cls, settings, **overrides) -> AgentConfig:
        values = {"max_steps": settings.max_agent_steps, "max_context_tokens": settings.llm_max_context_tokens}
        values.update(overrides)
        return cls(**values)


def call_key(name: str, arguments: dict[str, Any]) -> str:
    """Identity of a tool call for duplicate detection (search_code ignores top_k and case)."""
    if name == "search_code":
        return f"search_code:{str(arguments.get('query', '')).strip().lower()}"
    return f"{name}:{json.dumps(arguments, sort_keys=True)}"


TRIMMED_NOTE = "\n[... older tool output trimmed to fit the context budget; call the tool again if needed]"


def fit_context(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, budget: int | None) -> int:
    """Shrink the oldest tool outputs until the request fits ``budget`` tokens.

    The system prompt, the question and the most recent tool result are kept.
    Returns the number of messages that were trimmed.
    """
    if not budget:
        return 0
    fixed = estimate_tokens(tools) if tools else 0
    trimmed = 0

    def total() -> int:
        return fixed + sum(estimate_tokens(m.get("content") or "") + 8 for m in messages)

    candidates = [i for i, m in enumerate(messages) if m["role"] == "tool"][:-1]
    # The seed results live in the first user message; they are the last resort.
    first_user = next((i for i, m in enumerate(messages) if m["role"] == "user" and "Initial search" in (m.get("content") or "")), None)  # fmt: skip
    if first_user is not None:
        candidates.append(first_user)
    for i in candidates:
        if total() <= budget:
            break
        content = messages[i].get("content") or ""
        if content.endswith(TRIMMED_NOTE) or len(content) < 600:
            continue
        keep = 400 if messages[i]["role"] == "tool" else 1_200
        messages[i] = {**messages[i], "content": content[:keep] + TRIMMED_NOTE}
        trimmed += 1
    return trimmed


@dataclass
class AgentEvent:
    kind: str  # status | tool_call | tool_result | answer | warning | error
    step: int
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "step": self.step, "message": self.message, "data": self.data}


@dataclass
class AgentResult:
    question: str
    answer: str
    grounding: GroundingReport
    events: list[AgentEvent]
    evidence: EvidenceLog
    steps: int
    tool_calls: int
    elapsed: float
    usage: dict[str, int]
    stop_reason: str  # answered | budget | stalled | error

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "grounding": self.grounding.to_dict(),
            "files_consulted": self.evidence.files(),
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "elapsed_seconds": round(self.elapsed, 2),
            "usage": self.usage,
            "stop_reason": self.stop_reason,
            "trace": [e.to_dict() for e in self.events if e.kind in {"tool_call", "tool_result", "warning"}],
        }


class Agent:
    def __init__(
        self,
        index: CodeIndex,
        llm: ChatClient,
        config: AgentConfig | None = None,
        label: str | None = None,
        search_method: str = "auto",
    ) -> None:
        self.index = index
        self.llm = llm
        self.config = config or AgentConfig()
        self.label = label or index.root.name
        self.search_method = search_method

    def _system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(label=self.label, overview=overview_text(self.index, readme_lines=0))

    def run(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        memory: ConversationMemory | None = None,
    ) -> Iterator[AgentEvent]:
        """Investigate ``question``. The last event has ``kind == 'answer'`` and carries the result.

        ``history`` holds earlier user/assistant messages; ``memory`` the evidence
        gathered in earlier turns of the same conversation.
        """
        started = time.perf_counter()
        evidence = EvidenceLog()
        toolbox = ToolBox(self.index, evidence, self.search_method)
        seed_box = ToolBox(self.index, evidence, self.search_method, snippet_chars=self.config.seed_snippet_chars)
        schemas = toolbox.schemas()
        events: list[AgentEvent] = []
        usage: dict[str, int] = {}
        tool_calls = 0
        seen_calls: dict[str, str] = {}
        duplicates = 0

        def emit(event: AgentEvent) -> AgentEvent:
            events.append(event)
            return event

        def add_usage(response: ChatResponse) -> None:
            for key, value in response.usage.items():
                usage[key] = usage.get(key, 0) + value

        # 1. Seed with an automatic search (follow-ups also use the previous question).
        seed_query = contextual_query(question, history)
        seed_args = {"query": seed_query, "top_k": self.config.seed_top_k}
        yield emit(
            AgentEvent(
                "tool_call",
                0,
                tool_call_summary("search_code", seed_args),
                {"tool": "search_code", "arguments": seed_args, "auto": True},
            )
        )
        seed = seed_box.call("search_code", seed_args)
        hits = seed.data if isinstance(seed.data, list) else []
        yield emit(
            AgentEvent(
                "tool_result",
                0,
                f"{len(hits)} results",
                {"tool": "search_code", "preview": seed.text[:1500], "hits": hits},
            )
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]
        for turn in (history or [])[-2 * self.config.history_turns :]:
            if turn.get("role") in {"user", "assistant"} and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"][:2000]})
        earlier = ""
        if memory and memory.files():
            earlier = (
                "Files examined earlier in this conversation (revisit with read_file if needed): "
                + ", ".join(memory.files())
                + "\n\n"
            )
        messages.append(
            {
                "role": "user",
                "content": f"Question: {question}\n\n{earlier}Initial search results (from search_code):\n{seed.text}",
            }
        )

        answer = ""
        stop_reason = "answered"
        step = 0
        try:
            for step in range(1, self.config.max_steps + 1):
                yield emit(AgentEvent("status", step, "Thinking..."))
                if fit_context(messages, schemas, self.config.max_context_tokens):
                    yield emit(AgentEvent("status", step, "Trimmed older tool output to fit the token budget"))
                try:
                    response = self.llm.chat(messages, tools=schemas, temperature=self.config.temperature)
                except ToolCallParseError:
                    # Seen with gpt-oss on Groq: reasoning text leaks into the tool-call channel.
                    yield emit(AgentEvent("warning", step, "The model's tool call could not be parsed; retrying."))
                    messages.append({"role": "user", "content": PARSE_RETRY_PROMPT})
                    continue
                add_usage(response)
                if not response.tool_calls:
                    answer = response.content.strip()
                    if answer:
                        break
                    # Empty reply: nudge once, then give up on tools.
                    messages.append(
                        {"role": "user", "content": "Please answer the question now based on the evidence."}
                    )
                    continue
                calls = response.tool_calls[: self.config.max_calls_per_step]
                response.tool_calls = calls
                messages.append(response.to_message())
                for call in calls:
                    key = call_key(call.name, call.arguments)
                    summary = tool_call_summary(call.name, call.arguments)
                    yield emit(AgentEvent("tool_call", step, summary, {"tool": call.name, "arguments": call.arguments}))
                    if key in seen_calls:
                        duplicates += 1
                        text = (
                            "You already made this exact call. Its result is above; use it or try something different."
                        )
                        is_error = True
                    else:
                        result = toolbox.call(call.name, call.arguments)
                        text, is_error = result.text, result.is_error
                        seen_calls[key] = text
                        tool_calls += 1
                    preview = text if len(text) <= 1500 else text[:1500] + "\n..."
                    yield emit(
                        AgentEvent(
                            "tool_result",
                            step,
                            "error" if is_error else "ok",
                            {"tool": call.name, "preview": preview, "is_error": is_error},
                        )
                    )
                    messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": text})
                if duplicates >= self.config.max_duplicate_calls:
                    # The model is going in circles; more steps only burn tokens.
                    stop_reason = "stalled"
                    yield emit(AgentEvent("status", step, "Repeated tool calls; answering from the evidence so far"))
                    break
            else:
                stop_reason = "budget"
            if not answer:
                if stop_reason != "stalled":
                    stop_reason = "budget"
                yield emit(AgentEvent("status", step, "Writing the final answer..."))
                messages.append({"role": "user", "content": FINAL_ANSWER_PROMPT})
                fit_context(messages, None, self.config.max_context_tokens)
                response = self.llm.chat(messages, tools=None, temperature=self.config.temperature)
                add_usage(response)
                answer = response.content.strip() or "I could not produce an answer from the gathered evidence."
        except LLMError as exc:
            stop_reason = "error"
            yield emit(AgentEvent("error", step, str(exc)))
            answer = f"**The model could not be reached:** {exc}"

        # Code shown in earlier turns of the conversation also counts as seen.
        checked = EvidenceLog(list(memory.ranges) + evidence.ranges) if memory else evidence
        grounding = verify_citations(answer, self.index, checked)
        if grounding.count("invalid"):
            yield emit(
                AgentEvent(
                    "warning",
                    step,
                    f"{grounding.count('invalid')} citation(s) point to files or lines that do not exist.",
                )
            )
        result = AgentResult(
            question=question,
            answer=answer,
            grounding=grounding,
            events=events,
            evidence=evidence,
            steps=step,
            tool_calls=tool_calls,
            elapsed=time.perf_counter() - started,
            usage=usage,
            stop_reason=stop_reason,
        )
        yield emit(AgentEvent("answer", step, answer, {"result": result}))

    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        memory: ConversationMemory | None = None,
    ) -> AgentResult:
        """Run to completion and return the :class:`AgentResult`."""
        final = None
        for event in self.run(question, history, memory):
            final = event
        assert final is not None and final.kind == "answer"
        return final.data["result"]
