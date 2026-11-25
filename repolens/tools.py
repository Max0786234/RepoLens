"""Tools the agent can call, plus the evidence log used to verify citations.

Every tool:

* is read-only and confined to files in the index (the sandbox boundary),
* validates its arguments and reports problems as an ``ERROR:`` string instead
  of raising, so a confused model can recover,
* returns compact, line-numbered text (small local models handle this better
  than nested JSON),
* records which line ranges it showed the model in the :class:`EvidenceLog`.
"""

from __future__ import annotations

import fnmatch
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .index import CodeIndex

MAX_READ_LINES = 120
MAX_SNIPPET_CHARS = 1_200
MAX_TOOL_CHARS = 5_000


@dataclass
class EvidenceRange:
    path: str
    start_line: int
    end_line: int
    tool: str


@dataclass
class EvidenceLog:
    """All line ranges the model has been shown during one investigation."""

    ranges: list[EvidenceRange] = field(default_factory=list)

    def add(self, path: str, start: int, end: int, tool: str) -> None:
        self.ranges.append(EvidenceRange(path, max(1, start), max(start, end), tool))

    def covers(self, path: str, start: int, end: int) -> bool:
        return any(r.path == path and r.start_line <= end and start <= r.end_line for r in self.ranges)

    def files(self) -> list[str]:
        return list(dict.fromkeys(r.path for r in self.ranges))


@dataclass
class ToolResult:
    text: str
    is_error: bool = False
    data: Any = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., ToolResult]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


def _error(message: str) -> ToolResult:
    return ToolResult(f"ERROR: {message}", is_error=True)


def _clip(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} characters]"


def _number(lines: list[str], start: int) -> str:
    width = len(str(start + len(lines)))
    return "\n".join(f"{start + i:>{width}} | {line}" for i, line in enumerate(lines))


def _int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


class ToolBox:
    """The registry of tools bound to one index and one evidence log."""

    def __init__(
        self,
        index: CodeIndex,
        evidence: EvidenceLog | None = None,
        search_method: str = "auto",
        snippet_chars: int = MAX_SNIPPET_CHARS,
    ) -> None:
        self.index = index
        self.snippet_chars = snippet_chars
        self.evidence = evidence or EvidenceLog()
        self.search_method = search_method
        self.tools: dict[str, Tool] = {}
        self._register_all()

    # ------------------------------------------------------------ registry

    def register(self, name: str, description: str, properties: dict[str, Any], required: list[str]):
        def decorator(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
            self.tools[name] = Tool(
                name,
                description,
                {"type": "object", "properties": properties, "required": required},
                func,
            )
            return func

        return decorator

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def call(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return _error(f"unknown tool {name!r}. Available tools: {', '.join(self.tools)}.")
        arguments = arguments or {}
        if "_raw" in arguments:
            return _error(f"arguments were not valid JSON: {arguments['_raw'][:200]!r}")
        allowed = set(tool.parameters["properties"])
        clean = {k: v for k, v in arguments.items() if k in allowed}
        missing = [k for k in tool.parameters.get("required", []) if clean.get(k) in (None, "")]
        if missing:
            return _error(f"{name} requires argument(s): {', '.join(missing)}.")
        try:
            result = tool.func(**clean)
        except Exception as exc:  # a tool bug must not crash the agent loop
            return _error(f"{name} failed: {type(exc).__name__}: {exc}")
        result.text = _clip(result.text)
        return result

    # ------------------------------------------------------------ tools

    def _register_all(self) -> None:
        index = self.index
        evidence = self.evidence

        @self.register(
            "search_code",
            "Hybrid keyword + semantic search over the repository. Returns the most relevant "
            "code chunks with file paths and line ranges. Use it first to locate code.",
            {
                "query": {"type": "string", "description": "What to look for, in words or identifiers."},
                "top_k": {"type": "integer", "description": "Number of results (1-10, default 6)."},
            },
            ["query"],
        )
        def search_code(query: str, top_k: int = 6) -> ToolResult:
            hits = index.search(str(query), top_k=_int(top_k, 6, 1, 10), method=self.search_method)
            if not hits:
                return ToolResult(f"No results for {query!r}. Try different keywords or grep.", data=[])
            blocks = []
            for hit in hits:
                c = hit.chunk
                label = f" ({c.kind} {c.symbol})" if c.symbol else ""
                lines = c.text.splitlines()
                truncated = len(c.text) > self.snippet_chars
                if truncated:
                    # Keep whole lines only, and record as evidence just the lines actually shown.
                    lines = c.text[: self.snippet_chars].splitlines()[:-1] or lines[:1]
                shown_end = c.start_line + len(lines) - 1
                evidence.add(c.path, c.start_line, shown_end, "search_code")
                more = f"\n... (chunk continues to line {c.end_line}; use read_file to see more)" if truncated else ""
                blocks.append(
                    f"[{hit.rank}] {c.path}:{c.start_line}-{c.end_line}{label}\n{_number(lines, c.start_line)}{more}"
                )
            return ToolResult("\n\n".join(blocks), data=[h.to_dict() for h in hits])

        @self.register(
            "read_file",
            f"Read a file (or a line range of it) with line numbers. At most {MAX_READ_LINES} lines per call.",
            {
                "path": {"type": "string", "description": "File path relative to the repository root."},
                "start_line": {"type": "integer", "description": "First line to read (1-based, default 1)."},
                "end_line": {"type": "integer", "description": "Last line to read (inclusive)."},
            },
            ["path"],
        )
        def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> ToolResult:
            resolved = index.resolve_path(str(path))
            if resolved is None:
                suggestions = [p for p in index.file_map if PurePosixPath(p).name == PurePosixPath(str(path)).name][:5]
                hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else " Use list_files or search_code."
                return _error(f"file {path!r} is not in the indexed repository.{hint}")
            lines = index.read_lines(resolved)
            total = len(lines)
            start = _int(start_line, 1, 1, max(1, total))
            default_end = start + MAX_READ_LINES - 1
            end = _int(end_line if end_line is not None else default_end, default_end, start, total)
            end = min(end, start + MAX_READ_LINES - 1, total)
            if total == 0:
                return ToolResult(f"{resolved} is empty.")
            evidence.add(resolved, start, end, "read_file")
            more = f" (file has {total} lines; read more with start_line={end + 1})" if end < total else ""
            header = f"{resolved} lines {start}-{end} of {total}{more}"
            return ToolResult(
                f"{header}\n{_number(lines[start - 1 : end], start)}",
                data={"path": resolved, "start": start, "end": end},
            )

        @self.register(
            "grep",
            "Regex search (case-insensitive) across indexed files. Good for exact identifiers, "
            "error messages and config keys.",
            {
                "pattern": {"type": "string", "description": "Regular expression or plain text."},
                "path_glob": {"type": "string", "description": "Optional glob filter, e.g. '*.py' or 'src/**'."},
            },
            ["pattern"],
        )
        def grep(pattern: str, path_glob: str | None = None, max_results: int = 40) -> ToolResult:
            pattern = str(pattern)
            if len(pattern) > 200:
                return _error("pattern is too long (max 200 characters).")
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error:
                regex = re.compile(re.escape(pattern), re.IGNORECASE)
            hits: list[str] = []
            per_file: Counter[str] = Counter()
            for info in index.files:
                if path_glob and not (
                    fnmatch.fnmatch(info.path, path_glob) or fnmatch.fnmatch(PurePosixPath(info.path).name, path_glob)
                ):
                    continue
                try:
                    lines = index.read_lines(info.path)
                except (FileNotFoundError, PermissionError):
                    continue
                for number, line in enumerate(lines, start=1):
                    if regex.search(line):
                        per_file[info.path] += 1
                        evidence.add(info.path, number, number, "grep")
                        hits.append(f"{info.path}:{number}: {line.strip()[:200]}")
                        if len(hits) >= max_results:
                            break
                if len(hits) >= max_results:
                    hits.append(f"... stopped after {max_results} matches; narrow the pattern or use path_glob.")
                    break
            if not hits:
                return ToolResult(f"No matches for /{pattern}/.", data=[])
            return ToolResult("\n".join(hits), data=dict(per_file))

        @self.register(
            "list_files",
            "List indexed files under a directory as a tree (with line counts).",
            {
                "directory": {"type": "string", "description": "Directory relative to the root, default '.'."},
                "max_depth": {"type": "integer", "description": "How many levels to expand (default 2)."},
            },
            [],
        )
        def list_files(directory: str = ".", max_depth: int = 2) -> ToolResult:
            base = str(directory or ".").replace("\\", "/").strip("/")
            base = "" if base in {".", ""} else base
            depth = _int(max_depth, 2, 1, 6)
            prefix = f"{base}/" if base else ""
            entries = [f for f in index.files if f.path.startswith(prefix)]
            if not entries:
                return _error(f"no indexed files under {directory!r}.")
            dirs: Counter[str] = Counter()
            shown: list[str] = []
            for info in entries:
                rest = info.path[len(prefix) :].split("/")
                if len(rest) <= depth:
                    shown.append(f"{info.path} ({info.lines} lines)")
                else:
                    dirs[prefix + "/".join(rest[:depth]) + "/"] += 1
            lines = shown[:200] + [f"{d} ... {n} more files" for d, n in sorted(dirs.items())][:100]
            return ToolResult("\n".join(lines), data={"files": len(entries)})

        @self.register(
            "find_symbol",
            "Find where a function, class or method is defined. Accepts 'name' or 'Class.method'.",
            {"name": {"type": "string", "description": "Symbol name, e.g. 'parse_args' or 'Session.send'."}},
            ["name"],
        )
        def find_symbol(name: str) -> ToolResult:
            query = str(name).strip().strip("()")
            lower = query.lower()
            exact = [s for s in index.symbols if s.qualname == query or s.name == query]
            if not exact:
                exact = [s for s in index.symbols if s.qualname.lower().endswith(lower) or s.name.lower() == lower]
            if not exact:
                exact = [s for s in index.symbols if lower in s.qualname.lower()][:15]
            if not exact:
                return ToolResult(f"No definition found for {name!r}. Try grep or search_code.", data=[])
            lines = []
            for s in exact[:15]:
                evidence.add(s.path, s.line, s.line, "find_symbol")
                lines.append(f"{s.path}:{s.line}-{s.end_line} {s.kind} {s.qualname}\n    {s.signature}")
            return ToolResult("\n".join(lines), data=[s.to_dict() for s in exact[:15]])

        @self.register(
            "find_references",
            "Find lines that use a symbol (whole-word match), excluding its definition.",
            {"name": {"type": "string", "description": "Identifier to look for."}},
            ["name"],
        )
        def find_references(name: str) -> ToolResult:
            ident = str(name).strip().split(".")[-1].strip("()")
            if not re.fullmatch(r"[A-Za-z_$][\w$]*", ident):
                return _error("name must be a single identifier.")
            regex = re.compile(rf"(?<![\w$]){re.escape(ident)}(?![\w$])")
            definitions = {(s.path, s.line) for s in index.symbols if s.name == ident}
            hits: list[str] = []
            files: Counter[str] = Counter()
            for info in index.files:
                try:
                    lines = index.read_lines(info.path)
                except (FileNotFoundError, PermissionError):
                    continue
                for number, line in enumerate(lines, start=1):
                    if (info.path, number) in definitions or not regex.search(line):
                        continue
                    files[info.path] += 1
                    if len(hits) < 40:
                        evidence.add(info.path, number, number, "find_references")
                        hits.append(f"{info.path}:{number}: {line.strip()[:200]}")
            if not hits:
                return ToolResult(f"No references to {ident!r} found.", data={})
            summary = f"{sum(files.values())} references in {len(files)} files."
            return ToolResult(summary + "\n" + "\n".join(hits), data=dict(files))

        @self.register(
            "repo_overview",
            "Summary of the repository: languages, top-level layout, key files and the README start.",
            {},
            [],
        )
        def repo_overview() -> ToolResult:
            return ToolResult(overview_text(index))


def overview_text(index: CodeIndex, readme_lines: int = 25) -> str:
    stats = index.stats()
    top: dict[str, int] = defaultdict(int)
    for info in index.files:
        head = info.path.split("/", 1)[0] + ("/" if "/" in info.path else "")
        top[head] += 1
    langs = ", ".join(f"{k} ({v})" for k, v in list(stats["languages"].items())[:8])
    layout = ", ".join(f"{k} [{v}]" if k.endswith("/") else k for k, v in sorted(top.items())[:40])
    parts = [
        f"{stats['files']} files, {stats['lines']} lines, {stats['symbols']} symbols.",
        f"Languages: {langs}.",
        f"Top level: {layout}.",
    ]
    key_names = {"main.py", "app.py", "cli.py", "__main__.py", "index.js", "index.ts", "main.go", "main.rs",
                 "package.json", "pyproject.toml", "setup.py", "cargo.toml", "go.mod", "dockerfile"}  # fmt: skip
    keys = [f.path for f in index.files if PurePosixPath(f.path).name.lower() in key_names][:12]
    if keys:
        parts.append("Key files: " + ", ".join(keys) + ".")
    readme = next(
        (f.path for f in index.files if PurePosixPath(f.path).name.lower().startswith("readme") and "/" not in f.path),
        None,
    )
    if readme and readme_lines > 0:
        lines = index.read_lines(readme)[:readme_lines]
        parts.append(f"{readme} (first {len(lines)} lines):\n" + "\n".join(lines))
    return "\n".join(parts)


def tool_call_summary(name: str, arguments: dict[str, Any]) -> str:
    """Human-readable one-liner, e.g. ``search_code(query='retry')``."""
    args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in arguments.items())
    return f"{name}({args})"
