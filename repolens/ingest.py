"""Turn a source (local folder, ZIP, GitHub URL or uploaded files) into a folder on disk.

Untrusted archives are extracted defensively:

* absolute paths, ``..`` components and symlinks are rejected,
* only indexable text files are written,
* size limits are enforced on the bytes actually decompressed, not on the
  sizes the archive claims (which a zip bomb can lie about).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import stat
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from .languages import is_indexable


class IngestError(RuntimeError):
    """Raised when a source cannot be loaded safely."""


@dataclass(frozen=True)
class Limits:
    max_file_bytes: int = 1_000_000
    max_total_bytes: int = 60_000_000
    max_files: int = 6_000
    max_download_bytes: int = 150_000_000


DEFAULT_LIMITS = Limits()


@dataclass
class RepoSource:
    """A repository that is ready on disk."""

    root: Path
    label: str
    kind: str  # "local" | "zip" | "github" | "upload"
    github: dict[str, str] | None = None  # owner, repo, ref (commit sha when known)
    skipped: dict[str, int] = field(default_factory=dict)

    def web_url(self, path: str, start: int | None = None, end: int | None = None) -> str | None:
        """Link to a file (and line range) on GitHub, if this source came from GitHub."""
        if not self.github:
            return None
        g = self.github
        url = f"https://github.com/{g['owner']}/{g['repo']}/blob/{g.get('ref') or 'HEAD'}/{path}"
        if start:
            url += f"#L{start}" + (f"-L{end}" if end and end != start else "")
        return url


def _safe_member_path(name: str) -> PurePosixPath | None:
    name = name.replace("\\", "/")
    if not name or name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        return None
    path = PurePosixPath(name)
    if any(part in ("..", "") for part in path.parts):
        return None
    return path


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def _common_top_dir(names: Iterable[str]) -> str | None:
    tops = {n.replace("\\", "/").split("/", 1)[0] for n in names if n.strip("/")}
    roots_with_children = [n for n in names if "/" in n.replace("\\", "/").strip("/")]
    if len(tops) == 1 and roots_with_children:
        return tops.pop()
    return None


def _copy_limited(src: BinaryIO, dst: BinaryIO, limit: int) -> int:
    written = 0
    while True:
        block = src.read(64 * 1024)
        if not block:
            return written
        written += len(block)
        if written > limit:
            raise _TooLarge
        dst.write(block)


class _TooLarge(Exception):
    pass


def extract_zip(
    archive: bytes | str | Path | BinaryIO,
    dest: Path,
    limits: Limits = DEFAULT_LIMITS,
    strip_top_dir: bool = True,
) -> dict[str, int]:
    """Extract indexable files from a ZIP into ``dest``. Returns skip statistics."""
    if isinstance(archive, bytes):
        archive = io.BytesIO(archive)
    dest.mkdir(parents=True, exist_ok=True)
    stats = {"written": 0, "unsafe": 0, "unsupported": 0, "too_large": 0}
    total = 0
    try:
        zf = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as exc:
        raise IngestError("The file is not a valid ZIP archive.") from exc
    with zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        top = _common_top_dir([m.filename for m in members]) if strip_top_dir else None
        for member in members:
            rel = _safe_member_path(member.filename)
            if rel is None or _is_symlink(member):
                stats["unsafe"] += 1
                continue
            if top and rel.parts[0] == top:
                rel = PurePosixPath(*rel.parts[1:])
            if not rel.parts or not is_indexable(str(rel)):
                stats["unsupported"] += 1
                continue
            if member.file_size > limits.max_file_bytes:
                stats["too_large"] += 1
                continue
            if stats["written"] >= limits.max_files:
                raise IngestError(f"Archive has more than {limits.max_files} indexable files.")
            target = dest.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(member) as src, open(target, "wb") as dst:
                    written = _copy_limited(src, dst, limits.max_file_bytes)
            except _TooLarge:
                # The header understated the size: treat as hostile, drop the file.
                target.unlink(missing_ok=True)
                stats["too_large"] += 1
                continue
            total += written
            if total > limits.max_total_bytes:
                raise IngestError(f"Extracted text exceeds the {limits.max_total_bytes // 1_000_000} MB limit.")
            stats["written"] += 1
    if stats["written"] == 0:
        raise IngestError("No supported source files were found in the archive.")
    return stats


_GITHUB_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com[/:](?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)"
    r"(?:\.git)?(?:/(?:tree|commit|blob)/(?P<ref>[^?#]+?))?/?(?:[?#].*)?$"
)
_GITHUB_SSH_RE = re.compile(r"^git@github\.com:(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?$")


def parse_github_url(url: str) -> tuple[str, str, str | None] | None:
    """Return ``(owner, repo, ref)`` for a GitHub URL, or ``None``."""
    url = url.strip()
    match = _GITHUB_SSH_RE.match(url) or _GITHUB_RE.match(url)
    if not match:
        return None
    ref = match.groupdict().get("ref")
    return match["owner"], match["repo"], ref or None


def is_github_url(value: str) -> bool:
    return parse_github_url(value) is not None


def download_github_repo(
    url: str,
    cache_dir: Path,
    limits: Limits = DEFAULT_LIMITS,
    token: str | None = None,
    refresh: bool = False,
) -> RepoSource:
    """Download a public GitHub repository snapshot (no ``git`` binary required)."""
    parsed = parse_github_url(url)
    if not parsed:
        raise IngestError(f"Not a GitHub repository URL: {url!r}")
    owner, repo, ref = parsed
    slug = f"{owner}-{repo}-{ref or 'default'}".replace("/", "_")
    dest = cache_dir / "repos" / slug
    meta_path = dest.parent / f"{slug}.json"
    if dest.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return RepoSource(dest, f"{owner}/{repo}", "github", github=meta)

    api = f"https://api.github.com/repos/{owner}/{repo}/zipball" + (f"/{ref}" if ref else "")
    headers = {"User-Agent": "RepoLens", "Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(api, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            buffer = io.BytesIO()
            _copy_limited(response, buffer, limits.max_download_bytes)
    except _TooLarge as exc:
        raise IngestError("Repository archive is too large to download.") from exc
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise IngestError(f"Repository {owner}/{repo} not found (or it is private).") from exc
        if exc.code == 403:
            raise IngestError("GitHub rate limit reached. Set GITHUB_TOKEN and retry.") from exc
        raise IngestError(f"GitHub download failed ({exc.code}).") from exc
    except urllib.error.URLError as exc:
        raise IngestError(f"Could not reach GitHub: {exc.reason}") from exc

    # Zipballs contain a single top folder named "<owner>-<repo>-<sha7>".
    buffer.seek(0)
    sha = None
    with zipfile.ZipFile(buffer) as zf:
        names = zf.namelist()
        top = _common_top_dir(names)
        if top:
            sha = top.rsplit("-", 1)[-1]
    if dest.exists():
        shutil.rmtree(dest)
    buffer.seek(0)
    stats = extract_zip(buffer, dest, limits)
    meta = {"owner": owner, "repo": repo, "ref": ref or sha or "HEAD"}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    return RepoSource(dest, f"{owner}/{repo}", "github", github=meta, skipped=stats)


def save_uploads(
    files: Iterable[tuple[str, bytes]],
    cache_dir: Path,
    limits: Limits = DEFAULT_LIMITS,
) -> RepoSource:
    """Persist uploaded files (ZIPs and/or individual sources) into a content-addressed folder."""
    files = list(files)
    digest = hashlib.sha1()
    for name, data in files:
        digest.update(name.encode())
        digest.update(hashlib.sha1(data).digest())
    dest = cache_dir / "uploads" / digest.hexdigest()[:16]
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    total = 0
    written = 0
    stats: dict[str, int] = {}
    zip_names = [n for n, _ in files if n.lower().endswith(".zip")]
    for name, data in files:
        if name.lower().endswith(".zip"):
            sub = dest if len(zip_names) == 1 else dest / Path(name).stem
            result = extract_zip(data, sub, limits)
            written += result["written"]
            for key, value in result.items():
                stats[key] = stats.get(key, 0) + value
            continue
        rel = _safe_member_path(name)
        if rel is None or not is_indexable(str(rel)) or len(data) > limits.max_file_bytes:
            stats["unsupported"] = stats.get("unsupported", 0) + 1
            continue
        total += len(data)
        if total > limits.max_total_bytes:
            raise IngestError("Uploaded files exceed the size limit.")
        target = dest.joinpath(*rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written += 1
    if written == 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise IngestError("No supported source files were found in the upload.")
    label = Path(zip_names[0]).stem if len(zip_names) == 1 else f"{written} uploaded files"
    return RepoSource(dest, label, "upload", skipped=stats)


def load_source(
    source: str | Path,
    cache_dir: Path,
    limits: Limits = DEFAULT_LIMITS,
    token: str | None = None,
) -> RepoSource:
    """Resolve a CLI-style source string: a local folder, a ``.zip`` file or a GitHub URL."""
    text = str(source)
    if is_github_url(text):
        return download_github_repo(text, cache_dir, limits, token=token)
    path = Path(text).expanduser()
    if path.is_dir():
        return RepoSource(path.resolve(), path.resolve().name, "local")
    if path.is_file() and path.suffix.lower() == ".zip":
        return save_uploads([(path.name, path.read_bytes())], cache_dir, limits)
    raise IngestError(f"Source not found: {text!r} (expected a folder, a .zip or a GitHub URL).")
