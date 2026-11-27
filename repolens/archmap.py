"""Architecture map: which parts of a repository depend on which.

Built from the call graph (:mod:`repolens.callgraph`): an arrow ``A -> B`` means
code in module A calls functions or classes defined in module B, weighted by
the number of call sites. To stay precise rather than noisy:

* calls to a name defined in more than one file are **skipped** (they cannot be
  attributed without type information) and counted, so the map says how much it
  left out;
* small projects are shown file by file, larger ones by folder, so the diagram
  stays readable (``max_nodes``);
* tests are excluded by default (they depend on everything and hide the
  architecture).

Outputs: plain text (for models and the CLI), Mermaid (renders on GitHub) and
Graphviz DOT (rendered in the browser by the web app).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .index import CodeIndex
from .languages import is_test_path

CODE_KINDS = {"function", "method", "class", "type", "impl", "module"}


@dataclass
class ModuleInfo:
    name: str
    files: int = 0
    lines: int = 0
    symbols: int = 0


@dataclass
class ArchitectureMap:
    level: str  # "file" | "folder"
    modules: dict[str, ModuleInfo]
    edges: dict[tuple[str, str], int]
    skipped_ambiguous: int = 0
    merged: list[str] = field(default_factory=list)  # modules folded into "(other)"

    def incoming(self, name: str) -> int:
        return sum(count for (_src, dst), count in self.edges.items() if dst == name)

    def outgoing(self, name: str) -> int:
        return sum(count for (src, _dst), count in self.edges.items() if src == name)

    @property
    def common_prefix(self) -> str:
        """Folder shared by every module (e.g. ``src/requests/``), shown once instead of on every label."""
        names = [m for m in self.modules if not m.startswith("(")]
        if len(names) < 2:
            return ""
        prefix = names[0]
        for name in names[1:]:
            while not name.startswith(prefix):
                prefix = prefix[: prefix.rstrip("/").rfind("/") + 1] if "/" in prefix.rstrip("/") else ""
        return prefix if prefix.endswith("/") else ""

    def short(self, name: str) -> str:
        prefix = self.common_prefix
        return name[len(prefix) :] or name if prefix and name.startswith(prefix) else name

    # ------------------------------------------------------------ renderers

    def summary_text(self, max_edges: int = 30) -> str:
        lines = [
            f"Architecture map ({self.level} level): {len(self.modules)} modules, "
            f"{len(self.edges)} dependencies from {sum(self.edges.values())} call sites."
        ]
        if self.common_prefix:
            lines.append(f"All modules are under {self.common_prefix}")
        if self.skipped_ambiguous:
            lines.append(f"{self.skipped_ambiguous} calls to names defined in several files were left out (ambiguous).")
        if self.merged:
            lines.append(f"Small modules grouped as (other): {', '.join(self.merged[:12])}")
        core = sorted(self.modules, key=lambda m: (-self.incoming(m), m))
        lines.append("\nMost depended-on modules:")
        for name in core[:6]:
            if self.incoming(name):
                lines.append(f"  {self.short(name)}: used by {self.incoming(name)} call sites")
        entry = [m for m in self.modules if self.outgoing(m) and not self.incoming(m)]
        if entry:
            lines.append(
                "\nEntry points (call others, nobody calls them): "
                + ", ".join(self.short(m) for m in sorted(entry)[:10])
            )
        lines.append("\nDependencies (caller -> callee: call sites):")
        for (src, dst), count in sorted(self.edges.items(), key=lambda item: -item[1])[:max_edges]:
            lines.append(f"  {self.short(src)} -> {self.short(dst)}: {count}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        ids = {name: f"m{i}" for i, name in enumerate(sorted(self.modules))}
        lines = ["flowchart LR"]
        for name, info in sorted(self.modules.items()):
            shown = self.short(name)
            label = (
                f"{shown}<br/>{info.files} files · {info.lines:,} lines"
                if self.level == "folder"
                else f"{shown}<br/>{info.lines:,} lines"
            )
            lines.append(f'    {ids[name]}["{_mermaid_escape(label)}"]')
        for (src, dst), count in sorted(self.edges.items()):
            lines.append(f"    {ids[src]} -->|{count}| {ids[dst]}")
        return "\n".join(lines)

    def strongest_edges(self, max_edges: int | None) -> dict[tuple[str, str], int]:
        """The ``max_edges`` heaviest dependencies (all of them if ``None``)."""
        ranked = sorted(self.edges.items(), key=lambda item: (-item[1], item[0]))
        return dict(ranked if max_edges is None else ranked[:max_edges])

    def to_dot(self, max_edges: int | None = 18) -> str:
        """Graphviz DOT. Only the strongest dependencies are drawn so the diagram stays readable."""
        edges = self.strongest_edges(max_edges)
        shown = {name for pair in edges for name in pair}
        top = max(edges.values(), default=1)
        lines = [
            "digraph architecture {",
            '  rankdir=LR; bgcolor="transparent"; concentrate=true; nodesep=0.35; ranksep=0.7;',
            '  node [shape=box, style="rounded,filled", fillcolor="#eef3ff", color="#3b6ff5", fontname="Helvetica", fontsize=11];',
            '  edge [color="#8a94a6", fontname="Helvetica", fontsize=9, fontcolor="#5b6474", arrowsize=0.7];',
        ]
        for name, info in sorted(self.modules.items()):
            if name not in shown:
                continue
            detail = f"{info.files} files, {info.lines:,} lines" if self.level == "folder" else f"{info.lines:,} lines"
            lines.append(f'  "{_dot_escape(name)}" [label="{_dot_escape(self.short(name))}\\n{detail}"];')
        for (src, dst), count in sorted(edges.items()):
            width = 1 + 3 * count / top
            lines.append(f'  "{_dot_escape(src)}" -> "{_dot_escape(dst)}" [label="{count}", penwidth={width:.1f}];')
        lines.append("}")
        return "\n".join(lines)


def _mermaid_escape(text: str) -> str:
    return text.replace('"', "'")


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


EXAMPLE_DIRS = {"example", "examples", "sample", "samples", "demo", "demos", "docs", "doc", "benchmarks", "bench"}


def is_example_path(path: str) -> bool:
    return any(part.lower() in EXAMPLE_DIRS for part in PurePosixPath(path).parts[:-1])


def _folder_key(path: str, depth: int) -> str:
    parts = PurePosixPath(path).parts[:-1]
    return "/".join(parts[:depth]) + "/" if parts else "(root)"


def build_architecture_map(
    index: CodeIndex,
    max_nodes: int = 16,
    level: str = "auto",
    include_tests: bool = False,
    include_examples: bool = False,
) -> ArchitectureMap:
    """Aggregate the call graph into a module-level dependency map.

    Tests and example/demo/docs code are excluded by default: they show how a project is
    used, not how it is built, and in small libraries they would dominate the picture.
    """
    graph = index.call_graph()
    symbols_by_path: Counter[str] = Counter(s.path for s in index.symbols if s.kind in CODE_KINDS)
    code_files = [
        f
        for f in index.files
        if symbols_by_path.get(f.path)
        and (include_tests or not is_test_path(f.path))
        and (include_examples or not is_example_path(f.path))
    ]

    # Choose the grouping: files if few enough, otherwise the shallowest-informative folder depth.
    depth, groups = 1, {_folder_key(f.path, 1) for f in code_files}  # top-level folders are the baseline
    for candidate in (2, 3):
        candidate_groups = {_folder_key(f.path, candidate) for f in code_files}
        if len(candidate_groups) <= max_nodes:
            depth, groups = candidate, candidate_groups
    # Folders only help when they actually separate the code; a project whose code sits in one
    # package (most small libraries) is clearer file by file, with small files folded together.
    use_files = level == "file" or (level == "auto" and (len(code_files) <= max_nodes or len(groups) < 5))
    if use_files:
        chosen, key = "file", (lambda path: path)
    else:
        chosen, key = "folder", (lambda path, d=depth: _folder_key(path, d))

    modules: dict[str, ModuleInfo] = {}
    module_of: dict[str, str] = {}
    for f in code_files:
        name = key(f.path)
        module_of[f.path] = name
        info = modules.setdefault(name, ModuleInfo(name))
        info.files += 1
        info.lines += f.lines
        info.symbols += symbols_by_path[f.path]

    # Names defined in exactly one (included) file can be attributed safely. Method calls
    # (obj.name()) may only resolve to methods / owner-qualified functions (res.json = function),
    # plain calls (name()) only to top-level functions and classes - otherwise a library calling
    # self.get_db() would be "linked" to an example app's top-level get_db().
    member_defs: dict[str, set[str]] = defaultdict(set)
    plain_defs: dict[str, set[str]] = defaultdict(set)
    for symbol in index.symbols:
        if symbol.path not in module_of:
            continue
        if symbol.kind == "method" or "." in symbol.qualname:
            member_defs[symbol.name].add(symbol.path)
        else:
            plain_defs[symbol.name].add(symbol.path)

    edges: Counter[tuple[str, str]] = Counter()
    skipped = 0
    for sites in graph.calls_from.values():
        for site in sites:
            caller_module = module_of.get(site.caller.path)
            if caller_module is None:
                continue
            targets = (member_defs if site.attribute else plain_defs).get(site.callee_name, set())
            if len(targets) != 1:
                skipped += len(targets) > 1
                continue
            callee_module = module_of[next(iter(targets))]
            if callee_module != caller_module:
                edges[(caller_module, callee_module)] += 1

    # Keep the map readable: fold the least-connected modules into "(other)".
    merged: list[str] = []
    if len(modules) > max_nodes:
        degree = Counter()
        for (src, dst), count in edges.items():
            degree[src] += count
            degree[dst] += count
        ranked = sorted(modules, key=lambda m: (-degree[m], -modules[m].lines, m))
        keep = set(ranked[: max_nodes - 1])
        merged = sorted(m for m in modules if m not in keep)
        other = ModuleInfo("(other)")
        for name in merged:
            info = modules.pop(name)
            other.files += info.files
            other.lines += info.lines
            other.symbols += info.symbols
        modules["(other)"] = other
        folded: Counter[tuple[str, str]] = Counter()
        for (src, dst), count in edges.items():
            src = src if src in keep else "(other)"
            dst = dst if dst in keep else "(other)"
            if src != dst:
                folded[(src, dst)] += count
        edges = folded

    return ArchitectureMap(chosen, modules, dict(edges), skipped, merged)


_SAFE_ID = re.compile(r"[^A-Za-z0-9_]")
