"""Multi-turn support: follow-up detection and conversation memory.

RepoLens is meant to be *discussed with*, not just queried. Two problems appear
in a conversation that single questions never hit:

1. **Follow-ups lose the topic.** "Are there tests for it?" contains no useful
   search terms, so the automatic seed search retrieves noise. When a message
   looks like a follow-up, the seed query is combined with the previous user
   question (see :func:`contextual_query`).
2. **Earlier evidence is forgotten.** Files and line ranges inspected in
   earlier turns are carried forward (:class:`ConversationMemory`), so the agent
   is told which files it already examined and citations to code shown earlier
   in the conversation still count as verified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import tokenize_query
from .tools import EvidenceRange

# Words that usually point back at something said earlier.
_REFERENCE_WORDS = {
    "it", "its", "this", "that", "these", "those", "they", "them", "their", "there",
    "above", "previous", "earlier", "same", "former", "latter", "also", "else", "more",
}  # fmt: skip
_FOLLOW_UP_OPENERS = re.compile(
    r"^\s*(and|also|but|so|then|what about|how about|why|what if|ok|okay|can you|could you|explain more|show me)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z']+")

MAX_MEMORY_RANGES = 400


def is_follow_up(question: str) -> bool:
    """Heuristic: does this message depend on earlier turns to make sense?"""
    words = [w.lower() for w in _WORD_RE.findall(question)]
    if not words:
        return False
    content_terms = tokenize_query(question)
    has_reference = any(w in _REFERENCE_WORDS for w in words)
    # Short messages with few searchable terms ("why?", "and the tests?") lean on context.
    if len(content_terms) <= 2 and len(words) <= 8:
        return True
    if has_reference and len(content_terms) <= 5:
        return True
    return bool(_FOLLOW_UP_OPENERS.match(question)) and len(content_terms) <= 4


def previous_user_question(history: list[dict[str, str]] | None) -> str | None:
    for turn in reversed(history or []):
        if turn.get("role") == "user" and turn.get("content", "").strip():
            return turn["content"].strip()
    return None


def contextual_query(question: str, history: list[dict[str, str]] | None) -> str:
    """The query used for retrieval: the question, plus the previous one if this is a follow-up."""
    previous = previous_user_question(history)
    if previous and is_follow_up(question):
        return f"{previous} {question}"
    return question


@dataclass
class ConversationMemory:
    """Evidence gathered across the turns of one conversation."""

    ranges: list[EvidenceRange] = field(default_factory=list)

    def add_turn(self, ranges: list[EvidenceRange]) -> None:
        for r in ranges:
            self.ranges.append(EvidenceRange(r.path, r.start_line, r.end_line, "earlier_turn"))
        if len(self.ranges) > MAX_MEMORY_RANGES:
            self.ranges = self.ranges[-MAX_MEMORY_RANGES:]

    def files(self, limit: int = 12) -> list[str]:
        """Most recently examined files first, without duplicates."""
        seen: dict[str, None] = {}
        for r in reversed(self.ranges):
            seen.setdefault(r.path, None)
        return list(seen)[:limit]

    def to_list(self) -> list[dict]:
        return [r.__dict__ for r in self.ranges]

    @classmethod
    def from_list(cls, items: list[dict] | None) -> ConversationMemory:
        memory = cls()
        for item in items or []:
            memory.ranges.append(EvidenceRange(item["path"], item["start_line"], item["end_line"], "earlier_turn"))
        return memory
