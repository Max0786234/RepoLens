"""A lightweight, language-agnostic call graph built from the symbol table.

For every function or method the index knows (from ``ast`` or tree-sitter), the
lines of its body are scanned for call sites (``name(`` and ``obj.name(``). Calls
to names that are defined somewhere in the repository become edges. This answers
"what calls X?" and "what does X call?" in one tool call instead of several
searches - fewer agent steps, fewer tokens.

It is deliberately approximate (no type resolution): ``a.save()`` links to every
repository function named ``save``. Tool output says so, and the agent is shown
the exact call lines, so it can check before relying on an edge.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .chunking import Symbol

_CALL_RE = re.compile(r"(?<![\w$])([A-Za-z_$][\w$]*)\s*\(")
_STRING_RE = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`[^`]*`")
_COMMENT_RE = re.compile(r"(#|//).*$")
_NOT_CALLS = {
    "if", "for", "while", "switch", "catch", "return", "function", "def", "class", "elif", "except", "with",
    "and", "or", "not", "in", "is", "lambda", "sizeof", "typeof", "await", "yield", "match", "case", "fn",
    "func", "super", "self", "this", "print", "len", "str", "int", "isinstance", "require", "assert",
}  # fmt: skip
CALLABLE_KINDS = {"function", "method"}


@dataclass(frozen=True)
class CallSite:
    caller: Symbol
    callee_name: str
    line: int
    text: str
    attribute: bool = False  # obj.name(...) rather than name(...)


class CallGraph:
    def __init__(self, symbols: list[Symbol], read_lines) -> None:
        self.by_name: dict[str, list[Symbol]] = defaultdict(list)
        for symbol in symbols:
            self.by_name[symbol.name].append(symbol)
        self.calls_from: dict[tuple[str, str, int], list[CallSite]] = defaultdict(list)
        self.calls_to: dict[str, list[CallSite]] = defaultdict(list)
        callables = sorted((s for s in symbols if s.kind in CALLABLE_KINDS), key=lambda s: (s.path, s.line))
        by_file: dict[str, list[Symbol]] = defaultdict(list)
        for symbol in callables:
            by_file[symbol.path].append(symbol)
        for path, file_symbols in by_file.items():
            try:
                lines = read_lines(path)
            except (FileNotFoundError, PermissionError):
                continue
            owner = self._innermost_owner(file_symbols, len(lines))
            for number in range(1, len(lines) + 1):
                caller = owner.get(number)
                if caller is None or number == caller.line:
                    continue  # outside any function, or the definition line itself
                raw = lines[number - 1]
                code = _COMMENT_RE.sub("", _STRING_RE.sub('""', raw))
                for match in _CALL_RE.finditer(code):
                    name = match.group(1)
                    if name in _NOT_CALLS or name not in self.by_name:
                        continue
                    before = code[: match.start()].rstrip()
                    site = CallSite(
                        caller, name, number, raw.strip()[:160], attribute=before.endswith((".", "->", "::"))
                    )
                    self.calls_from[self._key(caller)].append(site)
                    self.calls_to[name].append(site)

    @staticmethod
    def _key(symbol: Symbol) -> tuple[str, str, int]:
        return (symbol.path, symbol.qualname, symbol.line)

    @staticmethod
    def _innermost_owner(symbols: list[Symbol], total: int) -> dict[int, Symbol]:
        """Map each line to the smallest function/method that contains it."""
        owner: dict[int, Symbol] = {}
        for symbol in sorted(symbols, key=lambda s: s.end_line - s.line, reverse=True):
            for number in range(symbol.line, min(symbol.end_line, total) + 1):
                owner[number] = symbol  # smaller (inner) definitions overwrite larger ones
        return owner

    def definitions(self, name: str) -> list[Symbol]:
        short = name.strip().strip("()").split(".")[-1]
        matches = [s for s in self.by_name.get(short, []) if s.kind in CALLABLE_KINDS]
        if "." in name:
            exact = [s for s in matches if s.qualname.lower().endswith(name.lower())]
            matches = exact or matches
        return matches

    def callers(self, name: str) -> list[CallSite]:
        short = name.strip().strip("()").split(".")[-1]
        return list(self.calls_to.get(short, []))

    def callees(self, symbol: Symbol) -> list[CallSite]:
        return self.calls_from.get(self._key(symbol), [])
