"""Explain a code change: what changed, what it affects, and what to test.

Input is a unified diff (``git diff`` output, a ``.diff``/``.patch`` file) or a
GitHub pull-request URL. The repository index is built from the *new* version
of the code, so:

1. :func:`parse_unified_diff` turns the diff into files and hunks with exact
   old/new line numbers;
2. :func:`analyze_change` maps added/modified lines onto the functions and
   classes they fall in, follows the call graph to find **callers that may be
   affected**, and finds **tests that exercise** the changed code;
3. :func:`explain_change` asks the model for a review-style explanation whose
   ``path:line`` citations are verified against the evidence it was shown, like
   every other RepoLens answer.

Steps 1-2 need no model; the report is useful on its own (and is what the MCP
``analyze_change`` tool returns to a client's model).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .chunking import Symbol
from .citations import GroundingReport, verify_citations
from .index import CodeIndex
from .languages import is_test_path
from .tools import EvidenceLog

MAX_DIFF_BYTES = 400_000
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_PR_RE = re.compile(r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+)/pull/(\d+)")


class ChangeError(RuntimeError):
    pass


@dataclass
class Hunk:
    old_start: int
    new_start: int
    header: str
    added: list[tuple[int, str]] = field(default_factory=list)  # (new line number, text)
    removed: list[tuple[int, str]] = field(default_factory=list)  # (old line number, text)


@dataclass
class FileChange:
    path: str  # new path (old path for deletions)
    old_path: str | None
    status: str  # added | modified | deleted | renamed
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def additions(self) -> int:
        return sum(len(h.added) for h in self.hunks)

    @property
    def deletions(self) -> int:
        return sum(len(h.removed) for h in self.hunks)

    def touched_new_lines(self) -> list[int]:
        """New-file line numbers that were added, plus the anchor line of pure deletions."""
        lines: list[int] = []
        for hunk in self.hunks:
            lines.extend(n for n, _ in hunk.added)
            if hunk.removed and not hunk.added:
                lines.append(max(1, hunk.new_start))
        return sorted(set(lines))


def _strip_prefix(path: str) -> str | None:
    path = path.strip().split("\t")[0]
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def parse_unified_diff(text: str) -> list[FileChange]:
    """Parse ``git diff`` / unified diff text. Binary files and noise lines are skipped."""
    if len(text.encode("utf-8", errors="ignore")) > MAX_DIFF_BYTES:
        raise ChangeError(f"Diff is larger than {MAX_DIFF_BYTES // 1000} kB; explain a smaller change.")
    changes: list[FileChange] = []
    current: FileChange | None = None
    hunk: Hunk | None = None
    old_line = new_line = 0
    pending_old: str | None = None
    renamed_from: str | None = None
    all_lines = text.splitlines()
    for i, raw in enumerate(all_lines):
        if raw.startswith("diff --git "):
            current, hunk, pending_old, renamed_from = None, None, None, None
            continue
        if raw.startswith("rename from "):
            renamed_from = raw[len("rename from ") :].strip()
            continue
        next_line = all_lines[i + 1] if i + 1 < len(all_lines) else ""
        if raw.startswith("--- ") and next_line.startswith("+++ "):
            # A file header (also without a preceding 'diff --git' line, as in plain `diff -u`).
            pending_old, hunk = raw[4:], None
            continue
        if raw.startswith("+++ ") and pending_old is not None:
            old_path = _strip_prefix(pending_old)
            new_path = _strip_prefix(raw[4:])
            if old_path is None and new_path is None:
                continue
            status = "added" if old_path is None else "deleted" if new_path is None else "modified"
            if status == "modified" and (renamed_from or old_path != new_path):
                status = "renamed"
            current = FileChange(new_path or old_path, old_path, status)
            changes.append(current)
            pending_old, hunk = None, None
            continue
        match = _HUNK_RE.match(raw)
        if match and current is not None:
            old_line, new_line = int(match.group(1)), int(match.group(3))
            hunk = Hunk(old_line, new_line, match.group(5).strip())
            current.hunks.append(hunk)
            continue
        if hunk is None or current is None:
            continue
        if raw.startswith("+"):
            hunk.added.append((new_line, raw[1:]))
            new_line += 1
        elif raw.startswith("-"):
            hunk.removed.append((old_line, raw[1:]))
            old_line += 1
        elif raw.startswith("\\"):
            continue  # "\ No newline at end of file"
        else:
            old_line += 1
            new_line += 1
    return [c for c in changes if c.hunks or c.status in {"added", "deleted", "renamed"}]


@dataclass
class ChangedSymbol:
    symbol: Symbol
    changed_lines: list[int]


@dataclass
class ImpactSite:
    path: str
    line: int
    caller: str
    callee: str
    text: str


@dataclass
class ChangeReport:
    files: list[FileChange]
    changed: list[ChangedSymbol]
    callers: list[ImpactSite]
    tests: list[ImpactSite]
    missing_files: list[str]  # changed files not present in the indexed (new) version
    evidence: EvidenceLog = field(default_factory=EvidenceLog)
    title: str = ""
    description: str = ""

    def summary_text(self, max_items: int = 25) -> str:
        """Deterministic, model-free report (also returned by the MCP tool)."""
        lines = []
        if self.title:
            lines.append(f"Change: {self.title}")
        adds = sum(f.additions for f in self.files)
        dels = sum(f.deletions for f in self.files)
        lines.append(f"{len(self.files)} files changed, +{adds} / -{dels} lines.")
        lines.append("\nFiles:")
        for f in self.files[:max_items]:
            rename = f" (from {f.old_path})" if f.status == "renamed" and f.old_path else ""
            lines.append(f"  {f.status:8} {f.path}{rename}  +{f.additions} -{f.deletions}")
        lines.append("\nChanged functions and classes:")
        if not self.changed:
            lines.append("  (none detected - changes are outside functions, or in files without symbols)")
        for item in self.changed[:max_items]:
            s = item.symbol
            span = f"{item.changed_lines[0]}-{item.changed_lines[-1]}" if item.changed_lines else "?"
            lines.append(f"  {s.path}:{s.line}-{s.end_line} {s.kind} {s.qualname} (changed lines {span})")
        lines.append("\nCallers outside the change that may be affected (matched by name):")
        if not self.callers:
            lines.append("  (none found)")
        for site in self.callers[:max_items]:
            lines.append(f"  {site.path}:{site.line} {site.caller} calls {site.callee}: {site.text}")
        lines.append("\nTests that exercise the changed code:")
        if not self.tests:
            lines.append("  (none found - consider adding tests)")
        for site in self.tests[:max_items]:
            lines.append(f"  {site.path}:{site.line} {site.caller} -> {site.callee}")
        if self.missing_files:
            lines.append("\nNot in the indexed snapshot (deleted or non-text): " + ", ".join(self.missing_files[:10]))
        return "\n".join(lines)


def analyze_change(index: CodeIndex, changes: list[FileChange], title: str = "", description: str = "") -> ChangeReport:
    """Map a diff onto the (new) code: changed symbols, affected callers, related tests."""
    evidence = EvidenceLog()
    graph = index.call_graph()
    symbols_by_file: dict[str, list[Symbol]] = {}
    for symbol in index.symbols:
        symbols_by_file.setdefault(symbol.path, []).append(symbol)

    changed: list[ChangedSymbol] = []
    missing: list[str] = []
    for change in changes:
        if change.status == "deleted":
            missing.append(change.path)
            continue
        resolved = index.resolve_path(change.path)
        if resolved is None:
            missing.append(change.path)
            continue
        change.path = resolved
        touched = change.touched_new_lines()
        for number in touched:
            evidence.add(resolved, number, number, "diff")
        # Innermost definition containing each touched line.
        owners: dict[tuple[str, int], ChangedSymbol] = {}
        for number in touched:
            candidates = [s for s in symbols_by_file.get(resolved, []) if s.line <= number <= s.end_line]
            if not candidates:
                continue
            owner = min(candidates, key=lambda s: s.end_line - s.line)
            key = (owner.qualname, owner.line)
            owners.setdefault(key, ChangedSymbol(owner, [])).changed_lines.append(number)
        changed.extend(owners.values())

    changed_keys = {(c.symbol.path, c.symbol.qualname) for c in changed}
    callers: list[ImpactSite] = []
    tests: list[ImpactSite] = []
    seen: set[tuple[str, int]] = set()
    for item in changed:
        if item.symbol.kind not in {"function", "method"}:
            continue
        for site in graph.callers(item.symbol.name):
            key = (site.caller.path, site.line)
            if key in seen or (site.caller.path, site.caller.qualname) in changed_keys:
                continue
            seen.add(key)
            impact = ImpactSite(site.caller.path, site.line, site.caller.qualname, item.symbol.qualname, site.text)
            evidence.add(site.caller.path, site.line, site.line, "callers")
            (tests if is_test_path(site.caller.path) else callers).append(impact)
    return ChangeReport(changes, changed, callers, tests, missing, evidence, title, description)


# ---------------------------------------------------------------- pull requests


def parse_pr_url(url: str) -> tuple[str, str, int] | None:
    match = _PR_RE.match(url.strip())
    return (match.group(1), match.group(2), int(match.group(3))) if match else None


def _github_get(url: str, accept: str, token: str | None) -> bytes:
    headers = {"User-Agent": "RepoLens", "Accept": accept}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
            return response.read(MAX_DIFF_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ChangeError("Pull request not found (or the repository is private).") from exc
        if exc.code == 403:
            raise ChangeError("GitHub rate limit reached. Set GITHUB_TOKEN and retry.") from exc
        raise ChangeError(f"GitHub request failed ({exc.code}).") from exc
    except urllib.error.URLError as exc:
        raise ChangeError(f"Could not reach GitHub: {exc.reason}") from exc


@dataclass
class PullRequest:
    owner: str
    repo: str
    number: int
    title: str
    body: str
    head_sha: str
    diff: str

    @property
    def snapshot_url(self) -> str:
        """The repository at the PR's head commit (reachable from the base repo via refs/pull)."""
        return f"https://github.com/{self.owner}/{self.repo}/tree/{self.head_sha}"


def fetch_pull_request(url: str, token: str | None = None) -> PullRequest:
    parsed = parse_pr_url(url)
    if not parsed:
        raise ChangeError(f"Not a GitHub pull request URL: {url!r}")
    owner, repo, number = parsed
    api = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    meta = json.loads(_github_get(api, "application/vnd.github+json", token))
    diff = _github_get(api, "application/vnd.github.v3.diff", token).decode("utf-8", errors="replace")
    return PullRequest(owner, repo, number, meta.get("title", ""), meta.get("body") or "", meta["head"]["sha"], diff)


# ---------------------------------------------------------------- explanation

EXPLAIN_PROMPT = """You are RepoLens, a senior engineer reviewing a code change for a teammate.
You get the diff, the changed functions as they look AFTER the change, and callers/tests found via the call graph.

Write a concise review with these sections:
1. **Summary** - what the change does and why it likely exists (one short paragraph).
2. **What changed** - per file or function, the behaviour before vs after.
3. **Impact** - which callers or features may behave differently; say "none found" if the evidence shows none.
4. **Risks and what to test** - edge cases, compatibility concerns, missing tests.

Rules: cite code as `path:start-end` using the line numbers shown (they refer to the new version); always write the
file path, never numbered references. Describe only code you were shown; if something is not visible, say so."""


def build_change_context(index: CodeIndex, report: ChangeReport, max_chars: int = 14_000) -> str:
    parts = []
    if report.title:
        parts.append(f"Title: {report.title}")
    if report.description:
        parts.append(f"Description: {report.description[:1500]}")
    parts.append(report.summary_text())
    diff_lines = []
    for f in report.files:
        diff_lines.append(f"### {f.status} {f.path}")
        for hunk in f.hunks:
            diff_lines.append(f"@@ new line {hunk.new_start} @@ {hunk.header}")
            diff_lines.extend(f"- {text}" for _, text in hunk.removed[:60])
            diff_lines.extend(f"+ {n:>5} | {text}" for n, text in hunk.added[:80])
    parts.append("Diff (added lines carry their new line numbers):\n" + "\n".join(diff_lines))
    code_blocks = []
    for item in report.changed[:8]:
        s = item.symbol
        try:
            lines = index.read_lines(s.path)
        except (FileNotFoundError, PermissionError):
            continue
        end = min(s.end_line, s.line + 70)
        report.evidence.add(s.path, s.line, end, "changed_symbol")
        body = "\n".join(f"{n:>5} | {lines[n - 1]}" for n in range(s.line, end + 1) if n - 1 < len(lines))
        code_blocks.append(f"--- {s.path}:{s.line}-{end} ({s.kind} {s.qualname}, after the change)\n{body}")
    if code_blocks:
        parts.append("Changed code after the change:\n" + "\n\n".join(code_blocks))
    context = "\n\n".join(parts)
    return context if len(context) <= max_chars else context[:max_chars] + "\n... [truncated]"


@dataclass
class ChangeExplanation:
    report: ChangeReport
    answer: str
    grounding: GroundingReport


def explain_change(
    llm, index: CodeIndex, report: ChangeReport, max_context_tokens: int | None = None
) -> ChangeExplanation:
    max_chars = 14_000 if not max_context_tokens else max(3_000, min(14_000, (max_context_tokens - 1_000) * 3))
    context = build_change_context(index, report, max_chars)
    messages = [{"role": "system", "content": EXPLAIN_PROMPT}, {"role": "user", "content": context}]
    answer = llm.chat(messages).content.strip()
    return ChangeExplanation(report, answer, verify_citations(answer, index, report.evidence))
