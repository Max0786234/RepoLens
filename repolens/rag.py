"""Single-shot retrieval-augmented answering ("Quick" mode).

One search, one model call, streamed. Much faster than the agent on CPU-only
machines and a useful baseline in evaluations. Citations are verified against
the retrieved chunks exactly like agent answers.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from .citations import GroundingReport, verify_citations
from .conversation import ConversationMemory, contextual_query
from .index import CodeIndex, SearchHit
from .tools import EvidenceLog

RAG_SYSTEM_PROMPT = (
    "You are RepoLens, a careful software engineer answering questions about a repository. "
    "Answer using only the provided source excerpts. Start with the direct answer, then explain briefly. "
    "Cite every claim as `path:start-end` using the line numbers shown; always write the file path, "
    "never numbered references or markers such as [1] or 【1†L10-L20】. "
    "If the excerpts do not contain the answer, say what is missing instead of guessing."
)

MAX_CONTEXT_CHARS = 14_000
MAX_CHUNK_CHARS = 2_500


@dataclass
class RagAnswer:
    question: str
    hits: list[SearchHit]
    evidence: EvidenceLog
    answer: str = ""
    grounding: GroundingReport = field(default_factory=GroundingReport)


def build_context(hits: list[SearchHit], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts: list[str] = []
    used = 0
    for hit in hits:
        c = hit.chunk
        lines = c.text.splitlines()
        body = "\n".join(f"{c.start_line + n} | {line}" for n, line in enumerate(lines))
        if len(body) > MAX_CHUNK_CHARS:
            body = body[:MAX_CHUNK_CHARS] + "\n..."
        label = f" ({c.kind} {c.symbol})" if c.symbol else ""
        block = f"--- {c.path}:{c.start_line}-{c.end_line}{label}\n{body}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts) if parts else "No matching code was found."


def build_messages(question: str, context: str, history: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:
        if turn.get("role") in {"user", "assistant"} and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"][:2000]})
    messages.append({"role": "user", "content": f"Question: {question}\n\nSource excerpts:\n{context}"})
    return messages


def retrieve(index: CodeIndex, question: str, top_k: int = 6, method: str = "auto", history=None) -> RagAnswer:
    """Retrieve chunks for ``question``; follow-ups are searched together with the previous question."""
    hits = index.search(contextual_query(question, history), top_k=top_k, method=method)
    evidence = EvidenceLog()
    for hit in hits:
        evidence.add(hit.chunk.path, hit.chunk.start_line, hit.chunk.end_line, "search_code")
    return RagAnswer(question, hits, evidence)


def stream_answer(
    llm,
    index: CodeIndex,
    rag: RagAnswer,
    history=None,
    memory: ConversationMemory | None = None,
    max_context_tokens: int | None = None,
) -> Iterator[str]:
    """Stream answer text; fills ``rag.answer`` and ``rag.grounding`` when done.

    ``max_context_tokens`` shrinks the excerpt budget for rate-limited providers.
    """
    max_chars = MAX_CONTEXT_CHARS
    if max_context_tokens:
        # ~3 characters per token, minus room for the prompt, history and the question.
        max_chars = max(2_000, min(MAX_CONTEXT_CHARS, (max_context_tokens - 1_200) * 3))
        history = (history or [])[-2:]
    messages = build_messages(rag.question, build_context(rag.hits, max_chars), history)
    pieces: list[str] = []
    if hasattr(llm, "stream"):
        for piece in llm.stream(messages):
            pieces.append(piece)
            yield piece
    else:
        text = llm.chat(messages).content
        pieces.append(text)
        yield text
    rag.answer = "".join(pieces).strip()
    checked = EvidenceLog(list(memory.ranges) + rag.evidence.ranges) if memory else rag.evidence
    rag.grounding = verify_citations(rag.answer, index, checked)


def answer_question(
    llm,
    index: CodeIndex,
    question: str,
    top_k: int = 6,
    method: str = "auto",
    history=None,
    max_context_tokens: int | None = None,
) -> RagAnswer:
    rag = retrieve(index, question, top_k, method, history)
    for _ in stream_answer(llm, index, rag, history, max_context_tokens=max_context_tokens):
        pass
    return rag
