"""Split source files into retrievable chunks that follow code structure.

v1 cut every file into fixed 45-line windows, which routinely split a function
in half. v2 chunks along definitions instead:

* **Python** - exact boundaries from the ``ast`` module (functions, classes,
  methods, including decorators).
* **Other languages** - regex-detected definition lines for JS/TS, Go, Rust,
  Java/Kotlin/C#/Swift/Scala, C/C++, Ruby and PHP, with indentation used to
  qualify methods with their class name.
* **Markdown** - one chunk per heading section.
* **Everything else** - overlapping line windows.

Oversized definitions are split into windows that keep the symbol name, and the
same pass produces the symbol table used by ``find_symbol``.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import asdict, dataclass

MAX_CHUNK_LINES = 90
WINDOW_LINES = 60
WINDOW_OVERLAP = 10
MIN_SEGMENT_LINES = 4


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    text: str
    language: str
    kind: str  # function | class | method | module | section | window
    symbol: str | None = None

    @property
    def id(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha1(f"{self.path}\n{self.text}".encode()).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Symbol:
    name: str
    qualname: str
    kind: str  # function | class | method | type | ...
    path: str
    line: int
    end_line: int
    signature: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- helpers


def _windows(
    path: str,
    lines: list[str],
    start: int,
    end: int,
    language: str,
    kind: str,
    symbol: str | None,
) -> list[Chunk]:
    """Chunks for the 1-based inclusive range ``start..end`` as overlapping windows."""
    chunks: list[Chunk] = []
    if end - start + 1 <= MAX_CHUNK_LINES:
        text = "\n".join(lines[start - 1 : end])
        if text.strip():
            chunks.append(Chunk(path, start, end, text, language, kind, symbol))
        return chunks
    step = WINDOW_LINES - WINDOW_OVERLAP
    cursor = start
    part = 1
    while cursor <= end:
        stop = min(cursor + WINDOW_LINES - 1, end)
        text = "\n".join(lines[cursor - 1 : stop])
        if text.strip():
            label = f"{symbol} (part {part})" if symbol else None
            chunks.append(Chunk(path, cursor, stop, text, language, kind, label))
            part += 1
        if stop >= end:
            break
        cursor += step
    return chunks


def _segments_to_chunks(
    path: str,
    lines: list[str],
    segments: list[tuple[int, int, str, str | None]],
    language: str,
) -> list[Chunk]:
    """Fill gaps between definition segments with module-level chunks."""
    chunks: list[Chunk] = []
    cursor = 1
    total = len(lines)
    for start, end, kind, symbol in sorted(segments):
        if start > cursor:
            gap = "\n".join(lines[cursor - 1 : start - 1])
            if gap.strip():
                chunks.extend(_windows(path, lines, cursor, start - 1, language, "module", None))
        start = max(start, cursor)
        if end >= start:
            chunks.extend(_windows(path, lines, start, end, language, kind, symbol))
        cursor = max(cursor, end + 1)
    if cursor <= total and "\n".join(lines[cursor - 1 :]).strip():
        chunks.extend(_windows(path, lines, cursor, total, language, "module", None))
    return _merge_tiny(chunks)


def _merge_tiny(chunks: list[Chunk]) -> list[Chunk]:
    """Glue very small module-level fragments (imports, constants) onto a neighbour."""
    merged: list[Chunk] = []
    for chunk in chunks:
        size = chunk.end_line - chunk.start_line + 1
        prev = merged[-1] if merged else None
        if (
            prev
            and chunk.kind == "module"
            and prev.kind == "module"
            and size + (prev.end_line - prev.start_line + 1) <= MAX_CHUNK_LINES
            and prev.end_line + 1 >= chunk.start_line - 1
        ):
            prev.text = prev.text + "\n" + chunk.text
            prev.end_line = chunk.end_line
            continue
        # A dangling line or two after a definition is not worth its own chunk.
        if (
            prev
            and size < MIN_SEGMENT_LINES
            and chunk.kind == "module"
            and prev.kind != "module"
            and prev.end_line + 1 == chunk.start_line
            and (prev.end_line - prev.start_line) < MAX_CHUNK_LINES
        ):
            prev.text = prev.text + "\n" + chunk.text
            prev.end_line = chunk.end_line
            continue
        merged.append(chunk)
    return merged


def _signature(line: str) -> str:
    return line.strip()[:200]


# ---------------------------------------------------------------- python


def _chunk_python(path: str, text: str, lines: list[str]) -> tuple[list[Chunk], list[Symbol]] | None:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    symbols: list[Symbol] = []
    segments: list[tuple[int, int, str, str | None]] = []

    def start_of(node: ast.AST) -> int:
        decorators = getattr(node, "decorator_list", [])
        return min([node.lineno] + [d.lineno for d in decorators])

    def visit_class(node: ast.ClassDef, prefix: str = "") -> None:
        qual = f"{prefix}{node.name}"
        start, end = start_of(node), node.end_lineno or node.lineno
        symbols.append(Symbol(node.name, qual, "class", path, start, end, _signature(lines[node.lineno - 1])))
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        for child in methods:
            if isinstance(child, ast.ClassDef):
                symbols.append(
                    Symbol(
                        child.name,
                        f"{qual}.{child.name}",
                        "class",
                        path,
                        start_of(child),
                        child.end_lineno or child.lineno,
                        _signature(lines[child.lineno - 1]),
                    )
                )
            else:
                symbols.append(
                    Symbol(
                        child.name,
                        f"{qual}.{child.name}",
                        "method",
                        path,
                        start_of(child),
                        child.end_lineno or child.lineno,
                        _signature(lines[child.lineno - 1]),
                    )
                )
        if end - start + 1 <= MAX_CHUNK_LINES or not methods:
            segments.append((start, end, "class", qual))
            return
        # Large class: header (docstring, attributes) + one chunk per method.
        first_method = start_of(methods[0])
        if first_method > start:
            segments.append((start, first_method - 1, "class", qual))
        for child in methods:
            c_start, c_end = start_of(child), child.end_lineno or child.lineno
            kind = "class" if isinstance(child, ast.ClassDef) else "method"
            segments.append((c_start, c_end, kind, f"{qual}.{child.name}"))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = start_of(node), node.end_lineno or node.lineno
            symbols.append(
                Symbol(node.name, node.name, "function", path, start, end, _signature(lines[node.lineno - 1]))
            )
            segments.append((start, end, "function", node.name))
        elif isinstance(node, ast.ClassDef):
            visit_class(node)
    return _segments_to_chunks(path, lines, segments, "python"), symbols


# ---------------------------------------------------------------- regex languages

_CONTROL_WORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "else",
    "do",
    "try",
    "with",
    "elif",
    "sizeof",
    "new",
}

_JS_PATTERNS = [
    (
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"),
        "class",
    ),
    (re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?(?:interface|type|enum)\s+([A-Za-z_$][\w$]*)"), "type"),
    (
        re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*[<(]"),
        "function",
    ),
    (
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|[A-Za-z_$][\w$]*\s*=>)"
        ),
        "function",
    ),
    # res.status = function status(code) {  /  Foo.prototype.bar = function () {
    (
        re.compile(
            r"^\s*(?:([\w$]+)\.)?(?:[\w$]+\.)*?(?:prototype\.)?([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?function\b"
        ),
        "function",
    ),
    # defineGetter(req, 'ip', function ip() {  /  Object.defineProperty(obj, 'x', {
    (re.compile(r"^\s*(?:Object\.)?define\w*\(\s*([\w$.]+)\s*,\s*['\"]([\w$]+)['\"]"), "function"),
    (
        re.compile(
            r"^\s+(?:(?:public|private|protected|static|async|readonly|override|get|set)\s+)*\*?([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*(?::\s*[^{]+)?\{\s*$"
        ),
        "method",
    ),
]
_GO_PATTERNS = [
    (re.compile(r"^func\s+\(\s*\w+\s+\*?(\w+)[^)]*\)\s*(\w+)\s*[\[(]"), "method"),
    (re.compile(r"^func\s+(\w+)\s*[\[(]"), "function"),
    (re.compile(r"^type\s+(\w+)\s+(?:struct|interface)\b"), "type"),
]
_RUST_PATTERNS = [
    (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+(\w+)"), "function"),
    (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|union)\s+(\w+)"), "type"),
    (re.compile(r"^\s*impl(?:<[^>]*>)?\s+(?:[\w:<>, ]+\s+for\s+)?(\w+)"), "impl"),
]
_JVM_MODIFIERS = r"(?:(?:public|private|protected|internal|static|final|abstract|override|open|sealed|data|async|virtual|partial|suspend|inline|synchronized|fileprivate)\s+)*"
_JVM_PATTERNS = [
    (
        re.compile(
            rf"^\s*{_JVM_MODIFIERS}(?:class|interface|enum|record|struct|object|protocol|extension|trait)\s+(\w+)"
        ),
        "class",
    ),
    (re.compile(rf"^\s*{_JVM_MODIFIERS}(?:fun|func|def)\s+(?:<[^>]*>\s*)?(?:[\w.]+\.)?(\w+)\s*[(<\[]"), "function"),
    (re.compile(rf"^\s*{_JVM_MODIFIERS}(?:[\w<>\[\],.?]+\s+)+(\w+)\s*\([^;]*$"), "method"),
]
_C_PATTERNS = [
    (re.compile(r"^(?:class|struct|namespace)\s+(\w+)[^;]*$"), "class"),
    (re.compile(r"^(?:[\w*&:<>,~]+\s+)+\**&?([\w:~]+)\s*\([^;]*$"), "function"),
]
_RUBY_PATTERNS = [
    (re.compile(r"^\s*(?:class|module)\s+([\w:]+)"), "class"),
    (re.compile(r"^\s*def\s+(?:self\.)?([\w?!=]+)"), "function"),
]
_PHP_PATTERNS = [
    (re.compile(r"^\s*(?:abstract\s+|final\s+)?(?:class|interface|trait|enum)\s+(\w+)"), "class"),
    (re.compile(r"^\s*(?:(?:public|private|protected|static|abstract|final)\s+)*function\s+&?(\w+)\s*\("), "function"),
]

PATTERNS_BY_LANGUAGE = {
    "javascript": _JS_PATTERNS,
    "typescript": _JS_PATTERNS,
    "vue": _JS_PATTERNS,
    "svelte": _JS_PATTERNS,
    "go": _GO_PATTERNS,
    "rust": _RUST_PATTERNS,
    "java": _JVM_PATTERNS,
    "kotlin": _JVM_PATTERNS,
    "scala": _JVM_PATTERNS,
    "csharp": _JVM_PATTERNS,
    "swift": _JVM_PATTERNS,
    "dart": _JVM_PATTERNS,
    "c": _C_PATTERNS,
    "cpp": _C_PATTERNS,
    "ruby": _RUBY_PATTERNS,
    "php": _PHP_PATTERNS,
}

_CONTAINER_KINDS = {"class", "impl", "type"}


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t").replace("\t", "    "))


def _leading_comment_start(lines: list[str], index: int) -> int:
    """Move a definition start upwards over doc comments, decorators and annotations."""
    i = index
    while i > 0:
        prev = lines[i - 1].strip()
        if prev.startswith(("//", "/*", "*", "#", "@", "///", "--", "#[")) and not prev.startswith(
            ("#include", "#define")
        ):
            i -= 1
            continue
        # JSDoc/Javadoc blocks are often separated from the definition by one blank line.
        if not prev and i > 1 and lines[i - 2].strip().endswith("*/") and i == index:
            i -= 1
            continue
        break
    return i


def _detect_definitions(lines: list[str], language: str) -> list[tuple[int, int, str, str]]:
    """Return ``(line_index, indent, kind, name)`` for every definition line."""
    patterns = PATTERNS_BY_LANGUAGE.get(language, [])
    found: list[tuple[int, int, str, str]] = []
    for i, line in enumerate(lines):
        if not line.strip() or len(line) > 400:
            continue
        for pattern, kind in patterns:
            match = pattern.match(line)
            if not match:
                continue
            groups = [g for g in match.groups() if g]
            if not groups:
                break
            # Two captures mean "owner.member" (Go receivers, res.status = function, defineGetter).
            name = ".".join(groups[-2:]) if len(match.groups()) >= 2 and len(groups) >= 2 else groups[-1]
            if name.split(".")[-1] in _CONTROL_WORDS:
                break
            found.append((i, _indent(line), kind, name))
            break
    return found


def _block_end(lines: list[str], start: int, indent: int, language: str) -> int:
    """Best-effort end (0-based, inclusive) of a block starting at ``start``."""
    if language in {"ruby"}:
        for j in range(start + 1, len(lines)):
            stripped = lines[j].strip()
            if stripped == "end" and _indent(lines[j]) <= indent:
                return j
        return len(lines) - 1
    depth = 0
    seen_brace = False
    for j in range(start, len(lines)):
        code = re.sub(r"(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`[^`]*`|//.*$)", "", lines[j])
        opens, closes = code.count("{"), code.count("}")
        if opens:
            seen_brace = True
        depth += opens - closes
        if seen_brace and depth <= 0:
            return j
        if not seen_brace and j > start and code.rstrip().endswith(";"):
            return j  # declaration without a body
        if not seen_brace and j - start > 8:
            break
    return start


def _chunk_regex(path: str, lines: list[str], language: str) -> tuple[list[Chunk], list[Symbol]]:
    definitions = _detect_definitions(lines, language)
    symbols: list[Symbol] = []
    containers: list[tuple[int, str, int]] = []  # (indent, name, end_index)
    spans: list[tuple[int, int, str, str, int]] = []  # start, end, kind, qualname, indent
    for index, indent, kind, name in definitions:
        while containers and (indent <= containers[-1][0] or index > containers[-1][2]):
            containers.pop()
        if kind == "method" and "." not in name and not containers:
            continue  # "name(...) {" outside a class is usually a call with a callback
        end = _block_end(lines, index, indent, language)
        qual = f"{containers[-1][1]}.{name}" if containers and "." not in name else name
        if containers and kind == "function":
            kind = "method"
        symbols.append(Symbol(name.split(".")[-1], qual, kind, path, index + 1, end + 1, _signature(lines[index])))
        spans.append((index, end, kind, qual, indent))
        if kind in _CONTAINER_KINDS and end > index:
            containers.append((indent, name, end))

    # Choose chunk segments: outermost definitions, except that large containers
    # are broken into their member definitions.
    segments: list[tuple[int, int, str, str | None]] = []
    covered_until = -1
    for i, (start, end, kind, qual, _indent_level) in enumerate(spans):
        if start <= covered_until:
            continue
        size = end - start + 1
        children = [s for s in spans[i + 1 :] if start < s[0] <= end]
        if kind in _CONTAINER_KINDS and size > MAX_CHUNK_LINES and children:
            first_child = _leading_comment_start(lines, children[0][0])
            seg_start = _leading_comment_start(lines, start)
            if first_child > seg_start:
                segments.append((seg_start + 1, first_child, kind, qual))
            continue  # children become their own segments
        seg_start = _leading_comment_start(lines, start)
        segments.append((seg_start + 1, end + 1, kind, qual))
        covered_until = end
    # Remove overlaps created by comment back-tracking.
    cleaned: list[tuple[int, int, str, str | None]] = []
    last_end = 0
    for start, end, kind, qual in sorted(segments):
        start = max(start, last_end + 1)
        if end >= start:
            cleaned.append((start, end, kind, qual))
            last_end = end
    return _segments_to_chunks(path, lines, cleaned, language), symbols


# ---------------------------------------------------------------- markdown

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _chunk_markdown(path: str, lines: list[str], language: str) -> tuple[list[Chunk], list[Symbol]]:
    segments: list[tuple[int, int, str, str | None]] = []
    heads: list[tuple[int, str]] = []
    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if match:
            heads.append((i + 1, match.group(2)))
    for n, (line_no, title) in enumerate(heads):
        end = heads[n + 1][0] - 1 if n + 1 < len(heads) else len(lines)
        segments.append((line_no, end, "section", title[:120]))
    return _segments_to_chunks(path, lines, segments, language), []


# ---------------------------------------------------------------- entry point


def chunk_file(path: str, text: str, language: str) -> tuple[list[Chunk], list[Symbol]]:
    """Chunk one file. ``path`` must be a POSIX-style path relative to the repo root."""
    lines = text.splitlines()
    if not lines:
        return [], []
    if language == "python":
        result = _chunk_python(path, text, lines)
        if result is not None:
            return result
    if language in PATTERNS_BY_LANGUAGE:
        return _chunk_regex(path, lines, language)
    if language in {"markdown"}:
        return _chunk_markdown(path, lines, language)
    return _windows(path, lines, 1, len(lines), language, "window", None), []
