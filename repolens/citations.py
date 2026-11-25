"""Extract ``path:line`` citations from an answer and check them against evidence.

Each citation gets one of three statuses:

* ``verified`` - the file exists, the lines exist, and the agent was actually
  shown those lines during the investigation;
* ``unverified`` - the file and lines exist, but the agent never looked at them
  (the claim may be right, but it is not backed by gathered evidence);
* ``invalid`` - the file is not in the repository or the line range is out of
  bounds: a hallucinated reference.

This turns "grounded answers" from a promise in the README into something the
UI can measure and display for every response.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .index import CodeIndex
from .languages import detect_language
from .tools import EvidenceLog


def looks_like_file(path: str) -> bool:
    """A citation path must have a directory or a known source/doc extension.

    Without this, prose such as ``socket.gaierror (lines 861-868)`` was read as a
    citation of a file called ``socket.gaierror`` and counted as hallucinated.
    """
    return "/" in path or detect_language(path) is not None


_PATH = r"(?P<path>(?:[\w.\-]+/)*[\w.\-]+\.[A-Za-z0-9]{1,8})"
_CITATION_RES = [
    # path:12-34, path:12–34, path:L12-L34, path:12
    re.compile(_PATH + r":L?(?P<start>\d+)(?:\s*[-–—]\s*L?(?P<end>\d+))?"),
    # path#L12-L34 (GitHub style)
    re.compile(_PATH + r"#L(?P<start>\d+)(?:-L(?P<end>\d+))?"),
    # path (lines 12-34) / path, lines 12–34 / path line 12
    re.compile(_PATH + r"`?,?\s*\(?\s*lines?\s+(?P<start>\d+)(?:\s*(?:[-–—]|to)\s*(?P<end>\d+))?"),
]


@dataclass
class Citation:
    raw: str
    path: str
    start_line: int
    end_line: int
    status: str = "invalid"  # verified | unverified | invalid
    resolved_path: str | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroundingReport:
    citations: list[Citation] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.citations)

    def count(self, status: str) -> int:
        return sum(1 for c in self.citations if c.status == status)

    @property
    def score(self) -> float | None:
        """Share of citations backed by evidence, or ``None`` if nothing was cited."""
        return self.count("verified") / self.total if self.total else None

    @property
    def label(self) -> str:
        if not self.total:
            return "no citations"
        return f"{self.count('verified')}/{self.total} citations verified"

    def to_dict(self) -> dict:
        return {
            "citations": [c.to_dict() for c in self.citations],
            "verified": self.count("verified"),
            "unverified": self.count("unverified"),
            "invalid": self.count("invalid"),
            "score": self.score,
        }


def extract_citations(text: str) -> list[Citation]:
    found: dict[tuple[str, int, int], Citation] = {}
    spans: list[tuple[int, int]] = []
    for pattern in _CITATION_RES:
        for match in pattern.finditer(text):
            if any(s <= match.start() < e for s, e in spans):
                continue  # already captured by an earlier, more specific pattern
            path = match["path"].removeprefix("./")
            if re.fullmatch(r"[\d.]+", path):
                continue  # version numbers like 1.2.3
            if not looks_like_file(path):
                continue  # hostnames, exceptions, attributes: sub.example.com, socket.gaierror
            start = int(match["start"])
            end = int(match["end"]) if match["end"] else start
            if end < start:
                start, end = end, start
            key = (path, start, end)
            spans.append((match.start(), match.end()))
            if key not in found:
                found[key] = Citation(match.group(0), path, start, end)
    return list(found.values())


def linkify_citations(text: str, url_for) -> str:
    """Turn citations into Markdown links, leaving fenced code blocks untouched.

    ``url_for(path, start, end)`` returns a URL or ``None``. Works whatever wraps the
    citation - spaces, parentheses, backticks or the 【】 brackets gpt-oss likes.
    """
    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    for i, part in enumerate(parts):
        if part.startswith("```"):
            continue
        # Longest first, so "a.py:1-20" is linked before a shorter "a.py:1" could match inside it.
        for citation in sorted(extract_citations(part), key=lambda c: -len(c.raw)):
            url = url_for(citation.path, citation.start_line, citation.end_line)
            if not url:
                continue
            link = f"[`{citation.raw}`]({url})"
            raw = re.escape(citation.raw)
            part = re.sub(rf"(?<!\[)`{raw}`", lambda _m, link=link: link, part)
            part = re.sub(rf"(?<![\[`\w/.]){raw}(?![\w`-])", lambda _m, link=link: link, part)
        parts[i] = part
    return "".join(parts)


def verify_citations(text: str, index: CodeIndex, evidence: EvidenceLog) -> GroundingReport:
    report = GroundingReport()
    for citation in extract_citations(text):
        resolved = index.resolve_path(citation.path)
        if resolved is None:
            citation.status, citation.reason = "invalid", "file not in repository"
        else:
            citation.resolved_path = resolved
            total = index.file_map[resolved].lines
            if citation.start_line < 1 or citation.start_line > total:
                citation.status, citation.reason = "invalid", f"line out of range (file has {total} lines)"
            elif evidence.covers(resolved, citation.start_line, min(citation.end_line, total)):
                citation.status = "verified"
            else:
                citation.status, citation.reason = "unverified", "lines were not inspected by the agent"
        report.citations.append(citation)
    return report
