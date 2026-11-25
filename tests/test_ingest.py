import io
import stat
import zipfile

import pytest

from repolens.ingest import (
    IngestError,
    Limits,
    extract_zip,
    load_source,
    parse_github_url,
    save_uploads,
)


def make_zip(entries: dict[str, bytes], symlink: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, "/etc/passwd")
    return buffer.getvalue()


def test_extract_zip_rejects_traversal_absolute_and_symlinks(tmp_path):
    data = make_zip(
        {"../evil.py": b"x", "/abs.py": b"x", "C:/win.py": b"x", "ok/app.py": b"print(1)"},
        symlink="ok/link.py",
    )
    stats = extract_zip(data, tmp_path / "out", strip_top_dir=False)
    assert (tmp_path / "out" / "ok" / "app.py").exists()
    assert not (tmp_path / "evil.py").exists()
    assert not (tmp_path / "out" / "ok" / "link.py").exists()
    assert stats["unsafe"] == 4 and stats["written"] == 1


def test_extract_zip_strips_single_top_folder_and_skips_unsupported(tmp_path):
    data = make_zip(
        {
            "proj-main/src/a.py": b"a = 1",
            "proj-main/logo.png": b"\x89PNG",
            "proj-main/node_modules/x.js": b"1",
            "proj-main/.env": b"SECRET=1",
        }
    )
    stats = extract_zip(data, tmp_path / "out")
    assert (tmp_path / "out" / "src" / "a.py").exists()
    assert stats["written"] == 1 and stats["unsupported"] == 3


def test_extract_zip_enforces_real_decompressed_size(tmp_path):
    # A highly compressible file larger than the per-file limit is dropped.
    data = make_zip({"big.py": b"a" * 5_000, "small.py": b"b = 2"})
    stats = extract_zip(data, tmp_path / "out", Limits(max_file_bytes=1_000))
    assert stats["too_large"] == 1 and stats["written"] == 1


def test_extract_zip_total_limit(tmp_path):
    data = make_zip({f"f{i}.py": b"x" * 600 for i in range(5)})
    with pytest.raises(IngestError, match="limit"):
        extract_zip(data, tmp_path / "out", Limits(max_file_bytes=1_000, max_total_bytes=1_500))


def test_bad_zip_raises_friendly_error(tmp_path):
    with pytest.raises(IngestError, match="not a valid ZIP"):
        extract_zip(b"definitely not a zip", tmp_path / "out")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/psf/requests", ("psf", "requests", None)),
        ("github.com/psf/requests.git", ("psf", "requests", None)),
        ("https://github.com/pallets/flask/tree/main", ("pallets", "flask", "main")),
        ("https://github.com/a/b/tree/feature/x", ("a", "b", "feature/x")),
        ("git@github.com:owner/repo.git", ("owner", "repo", None)),
        ("https://gitlab.com/a/b", None),
        ("not a url", None),
    ],
)
def test_parse_github_url(url, expected):
    assert parse_github_url(url) == expected


def test_save_uploads_mixes_zip_and_files(tmp_path):
    files = [
        ("project.zip", make_zip({"pkg/a.py": b"a = 1", "setup.py": b""})),
        ("notes.md", b"# hi"),
        ("../x.py", b"bad"),
    ]
    source = save_uploads(files, tmp_path)
    assert (source.root / "pkg" / "a.py").exists()
    assert (source.root / "notes.md").exists()
    assert source.kind == "upload"


def test_save_uploads_without_supported_files_fails(tmp_path):
    with pytest.raises(IngestError):
        save_uploads([("image.png", b"\x89PNG")], tmp_path)


def test_load_source_local_folder_and_missing(tmp_path, sample_repo):
    source = load_source(sample_repo, tmp_path)
    assert source.kind == "local" and source.root == sample_repo.resolve()
    with pytest.raises(IngestError, match="not found"):
        load_source(tmp_path / "missing", tmp_path)


def test_web_url_for_github_sources(tmp_path):
    from repolens.ingest import RepoSource

    source = RepoSource(tmp_path, "o/r", "github", github={"owner": "o", "repo": "r", "ref": "abc"})
    assert source.web_url("src/a.py", 3, 9) == "https://github.com/o/r/blob/abc/src/a.py#L3-L9"
    assert RepoSource(tmp_path, "x", "local").web_url("a.py") is None
