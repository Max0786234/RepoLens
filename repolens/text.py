"""Code-aware tokenisation for lexical search.

Identifiers are split on snake_case and camelCase boundaries so that a natural
language query such as "parse header value" matches ``parseHeaderValue`` and
``parse_header_value``. The full identifier is kept as well so exact symbol
queries still score highest.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

# Words that carry no signal in questions about code. Only removed from queries:
# document statistics (IDF) already down-weight common words in code.
QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "can", "code", "codebase",
    "could", "do", "does", "doing", "done", "explain", "file", "files", "find", "for",
    "from", "function", "functions", "get", "give", "happen", "happens", "has", "have",
    "how", "i", "implement", "implementation", "implemented", "in", "inside", "into",
    "is", "it", "its", "logic", "me", "method", "of", "on", "or", "please", "project",
    "repo", "repository", "show", "so", "tell", "that", "the", "their", "them", "there",
    "these", "this", "to", "used", "uses", "using", "was", "we", "what", "when", "where",
    "which", "who", "why", "will", "with", "work", "works", "would", "you", "your",
}  # fmt: skip


def stem(token: str) -> str:
    """A deliberately tiny stemmer: enough to match plurals and -ing/-ed forms."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed") and not token.endswith("eed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def split_identifier(word: str) -> list[str]:
    """``getHTTPResponse_code`` -> ``['get', 'http', 'response', 'code']``."""
    parts: list[str] = []
    for piece in word.split("_"):
        if piece:
            parts.extend(m.group(0).lower() for m in _CAMEL_RE.finditer(piece))
    return parts


def tokenize(text: str) -> list[str]:
    """Tokenise code or prose into stemmed, lower-case terms."""
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        lower = word.lower()
        parts = split_identifier(word)
        if len(parts) > 1:
            tokens.append(lower.strip("_"))
            tokens.extend(stem(p) for p in parts if len(p) > 1)
        elif len(lower) > 1 or lower.isdigit():
            tokens.append(stem(lower.strip("_") or lower))
    return tokens


def tokenize_query(text: str) -> list[str]:
    """Tokenise a user question, dropping question words that match everything."""
    kept: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        if word.lower() in QUERY_STOPWORDS:
            continue
        kept.append(word)
    tokens = tokenize(" ".join(kept))
    return tokens or tokenize(text)
