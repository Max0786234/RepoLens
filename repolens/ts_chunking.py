"""Syntax-tree chunking with tree-sitter (optional, more precise than regexes).

Used for JavaScript/TypeScript, Go, Rust, Java, C, C++, C#, Ruby and PHP when the
corresponding grammar package is installed (``pip install "repolens[treesitter]"``).
Python keeps using the standard-library ``ast`` module. If a grammar is missing
or a file fails to parse, :func:`chunk_with_tree_sitter` returns ``None`` and the
caller falls back to the regex chunker, so tree-sitter is never a hard dependency.

Per-language grammar packages are used instead of a bundled "language pack":
they are small, and the bundled native library is blocked by some Windows
application-control policies.
"""

from __future__ import annotations

import importlib
from functools import cache

# language -> (python module, function returning the language pointer)
GRAMMARS: dict[str, tuple[str, str]] = {
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "java": ("tree_sitter_java", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "csharp": ("tree_sitter_c_sharp", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "php": ("tree_sitter_php", "language_php"),
}

# node type -> kind. "container" kinds hold members that become their own chunks when large.
DEFINITIONS: dict[str, dict[str, str]] = {
    "javascript": {
        "function_declaration": "function",
        "generator_function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
    },
    "go": {"function_declaration": "function", "method_declaration": "method", "type_spec": "type"},
    "rust": {
        "function_item": "function",
        "struct_item": "type",
        "enum_item": "type",
        "trait_item": "type",
        "impl_item": "impl",
        "mod_item": "module",
    },
    "java": {
        "class_declaration": "class",
        "interface_declaration": "class",
        "enum_declaration": "class",
        "record_declaration": "class",
        "method_declaration": "method",
        "constructor_declaration": "method",
    },
    "c": {"function_definition": "function", "struct_specifier": "type"},
    "cpp": {
        "function_definition": "function",
        "class_specifier": "class",
        "struct_specifier": "class",
        "namespace_definition": "module",
    },
    "csharp": {
        "class_declaration": "class",
        "interface_declaration": "class",
        "struct_declaration": "class",
        "enum_declaration": "class",
        "record_declaration": "class",
        "method_declaration": "method",
        "constructor_declaration": "method",
        "namespace_declaration": "module",
    },
    "ruby": {"class": "class", "module": "module", "method": "method", "singleton_method": "method"},
    "php": {
        "function_definition": "function",
        "class_declaration": "class",
        "interface_declaration": "class",
        "trait_declaration": "class",
        "method_declaration": "method",
    },
}
DEFINITIONS["typescript"] = {
    **DEFINITIONS["javascript"],
    "abstract_class_declaration": "class",
    "interface_declaration": "type",
    "type_alias_declaration": "type",
    "enum_declaration": "type",
}
DEFINITIONS["tsx"] = DEFINITIONS["typescript"]

CONTAINERS = {"class", "impl", "module", "type"}
FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}
COMMENT_TYPES = {"comment", "line_comment", "block_comment", "doc_comment"}


@cache
def get_parser(language: str):
    """A tree-sitter parser for ``language``, or ``None`` if unavailable."""
    spec = GRAMMARS.get(language)
    if spec is None:
        return None
    try:
        import tree_sitter

        module = importlib.import_module(spec[0])
        return tree_sitter.Parser(tree_sitter.Language(getattr(module, spec[1])()))
    except Exception:  # missing package, blocked native library, ABI mismatch...
        return None


def available_languages() -> list[str]:
    return [language for language in GRAMMARS if get_parser(language) is not None]


def _text(node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _declarator_name(node, source: bytes) -> str | None:
    """Dig through C/C++ declarators (pointer, function, reference...) to the identifier."""
    current = node
    for _ in range(8):
        if current is None:
            return None
        if current.type in {
            "identifier",
            "field_identifier",
            "qualified_identifier",
            "destructor_name",
            "operator_name",
        }:
            return _text(current, source)
        current = current.child_by_field_name("declarator") or (
            current.named_children[0] if current.named_children else None
        )
    return None


def _name(node, language: str, source: bytes) -> str | None:
    kind = node.type
    if language in {"c", "cpp"} and kind == "function_definition":
        return _declarator_name(node.child_by_field_name("declarator"), source)
    if language == "rust" and kind == "impl_item":
        type_node = node.child_by_field_name("type")
        trait = node.child_by_field_name("trait")
        name = _text(type_node, source) if type_node else None
        return f"{name} ({_text(trait, source)})" if name and trait else name
    if language == "go" and kind == "method_declaration":
        receiver = node.child_by_field_name("receiver")
        name = node.child_by_field_name("name")
        receiver_type = ""
        if receiver is not None:
            words = [w.strip("*()") for w in _text(receiver, source).replace("*", " ").split()]
            receiver_type = words[-1] if words else ""
        return (
            f"{receiver_type}.{_text(name, source)}"
            if receiver_type and name
            else (_text(name, source) if name else None)
        )
    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return _text(name_node, source)
    return None


def _js_extra(node, source: bytes) -> tuple[str, str] | None:
    """JS/TS definitions that are not declarations: ``const f = () => {}``,
    ``res.status = function () {}`` and ``defineGetter(req, 'ip', function () {})``."""
    if node.type in {"lexical_declaration", "variable_declaration"}:
        for declarator in node.named_children:
            if declarator.type != "variable_declarator":
                continue
            value = declarator.child_by_field_name("value")
            name = declarator.child_by_field_name("name")
            if value is not None and name is not None and value.type in FUNCTION_VALUES:
                return _text(name, source), "function"
            if value is not None and name is not None and value.type == "class":
                return _text(name, source), "class"
    if node.type == "expression_statement" and node.named_children:
        expr = node.named_children[0]
        if expr.type == "assignment_expression":
            left, right = expr.child_by_field_name("left"), expr.child_by_field_name("right")
            # Chained aliases: res.contentType = res.type = function () {...}
            while right is not None and right.type == "assignment_expression":
                left, right = right.child_by_field_name("left"), right.child_by_field_name("right")
            if left is not None and right is not None and right.type in FUNCTION_VALUES:
                name = _text(left, source).replace(".prototype.", ".")
                parts = name.split(".")
                return (".".join(parts[-2:]) if len(parts) > 1 else name), "function"
        if expr.type == "call_expression":
            callee = expr.child_by_field_name("function")
            args = expr.child_by_field_name("arguments")
            if callee is not None and args is not None and "define" in _text(callee, source).lower():
                named = args.named_children
                if len(named) >= 3 and named[1].type == "string" and named[-1].type in FUNCTION_VALUES:
                    return f"{_text(named[0], source)}.{_text(named[1], source).strip(chr(34) + chr(39))}", "function"
    if node.type == "export_statement":
        for child in node.named_children:
            found = _js_extra(child, source)
            if found:
                return found
    return None


def _start_with_comments(node) -> int:
    """First row of a definition, extended upwards over adjacent doc comments."""
    start = node.start_point[0]
    sibling = node.prev_named_sibling
    while sibling is not None and sibling.type in COMMENT_TYPES and start - sibling.end_point[0] <= 2:
        start = sibling.start_point[0]
        sibling = sibling.prev_named_sibling
    return start


def collect_definitions(text: str, language: str) -> list[tuple[int, int, int, str, str]] | None:
    """``(start_row, definition_row, end_row, kind, qualname)`` per definition (0-based rows), or ``None``.

    ``start_row`` includes leading doc comments; ``definition_row`` is the line of the definition itself.
    """
    parser = get_parser(language)
    if parser is None:
        return None
    source = text.encode("utf-8")
    try:
        tree = parser.parse(source)
    except Exception:
        return None
    if tree.root_node.has_error and tree.root_node.child_count == 0:
        return None
    definitions = DEFINITIONS.get(language, {})
    is_js = language in {"javascript", "typescript", "tsx"}
    found: list[tuple[int, int, int, str, str]] = []

    def visit(node, prefix: str, depth: int) -> None:
        for child in node.named_children:
            kind = definitions.get(child.type)
            name = None
            target = child
            if kind:
                name = _name(child, language, source)
            elif is_js:
                extra = _js_extra(child, source)
                if extra:
                    name, kind = extra
            if child.type == "export_statement" and not kind:
                visit(child, prefix, depth)  # export class X {...} / export function f() {}
                continue
            if not (kind and name):
                # Descend into wrappers (declaration lists, class bodies, blocks of namespaces).
                if child.type in {"declaration_list", "class_body", "body_statement", "field_declaration_list",
                                  "enum_body", "interface_body", "block", "compound_statement", "template_declaration",
                                  "statement_block", "program", "source_file", "translation_unit"} or child.type == "type_declaration":  # fmt: skip
                    visit(child, prefix, depth)
                continue
            name = name.strip()
            if "." in name or "::" in name or not prefix:
                qualname = name
            else:
                qualname = f"{prefix}.{name}"
            if prefix and kind == "function":
                kind = "method"
            start = _start_with_comments(target)
            found.append((start, target.start_point[0], target.end_point[0], kind, qualname))
            if kind in CONTAINERS:
                visit(child, qualname.split(" (")[0], depth + 1)

    visit(tree.root_node, "", 0)
    return found
