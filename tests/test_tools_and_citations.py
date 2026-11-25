from repolens.citations import extract_citations, verify_citations
from repolens.tools import EvidenceLog, ToolBox


def make_box(index):
    return ToolBox(index, EvidenceLog())


def test_search_code_records_evidence(sample_index):
    box = make_box(sample_index)
    result = box.call("search_code", {"query": "hash password", "top_k": 2})
    assert not result.is_error
    assert "shop/auth.py" in result.text
    assert box.evidence.files()[0] == "shop/auth.py"


def test_read_file_returns_numbered_range(sample_index):
    box = make_box(sample_index)
    result = box.call("read_file", {"path": "shop/cart.py", "start_line": 4, "end_line": 7})
    assert result.text.splitlines()[0].startswith("shop/cart.py lines 4-7")
    assert "4 | def compute_total" in result.text
    assert box.evidence.covers("shop/cart.py", 5, 5)
    assert not box.evidence.covers("shop/cart.py", 10, 12)


def test_read_file_sandbox(sample_index):
    box = make_box(sample_index)
    for path in ["../../etc/passwd", ".env", "node_modules/lib/index.js", "/etc/hosts"]:
        result = box.call("read_file", {"path": path})
        assert result.is_error, path
    assert box.evidence.ranges == []


def test_read_file_suggests_similar_paths(sample_index):
    result = make_box(sample_index).call("read_file", {"path": "lib/cart.py"})
    assert result.is_error and "shop/cart.py" in result.text


def test_grep_regex_and_glob(sample_index):
    box = make_box(sample_index)
    result = box.call("grep", {"pattern": r"def \w+_total", "path_glob": "*.py"})
    assert "shop/cart.py:4:" in result.text
    js_only = box.call("grep", {"pattern": "return", "path_glob": "*.js"})
    assert "web/server.js" in js_only.text and ".py" not in js_only.text
    invalid_regex = box.call("grep", {"pattern": "compute_total("})  # falls back to a literal search
    assert "compute_total(" in invalid_regex.text


def test_find_symbol_and_references(sample_index):
    box = make_box(sample_index)
    symbol = box.call("find_symbol", {"name": "LoginManager.login"})
    assert "shop/auth.py" in symbol.text and "method LoginManager.login" in symbol.text
    refs = box.call("find_references", {"name": "hash_password"})
    assert "shop/auth.py" in refs.text and "shop/cart.py:1:" in refs.text
    assert "def hash_password" not in refs.text  # definition excluded


def test_list_files_and_overview(sample_index):
    box = make_box(sample_index)
    listing = box.call("list_files", {"directory": "shop"})
    assert "shop/auth.py" in listing.text and "web/" not in listing.text
    overview = box.call("repo_overview", {})
    assert "Languages:" in overview.text and "README.md" in overview.text


def test_tool_errors_never_raise(sample_index):
    box = make_box(sample_index)
    assert box.call("run_shell", {"cmd": "rm -rf /"}).is_error
    assert box.call("read_file", {}).is_error  # missing required argument
    assert box.call("read_file", {"_raw": "{not json"}).is_error
    assert not box.call("read_file", {"path": "shop/cart.py", "start_line": "abc"}).is_error
    assert len(box.schemas()) == 7


def test_extract_citations_formats():
    text = (
        "See `shop/auth.py:5-9` and shop/cart.py:12, also web/server.js#L3-L8, "
        "plus shop/cart.py (lines 4-7). Version 1.2.3 is not a citation."
    )
    found = {(c.path, c.start_line, c.end_line) for c in extract_citations(text)}
    assert ("shop/auth.py", 5, 9) in found
    assert ("shop/cart.py", 12, 12) in found
    assert ("web/server.js", 3, 8) in found
    assert ("shop/cart.py", 4, 7) in found
    assert len(found) == 4


def test_verify_citations_statuses(sample_index):
    evidence = EvidenceLog()
    evidence.add("shop/cart.py", 4, 7, "read_file")
    answer = "Totals: shop/cart.py:4-7. Coupons: shop/cart.py:10-13. Fake: shop/payments.py:1-5. Out: shop/cart.py:900."
    report = verify_citations(answer, sample_index, evidence)
    statuses = {c.path + ":" + str(c.start_line): c.status for c in report.citations}
    assert statuses == {
        "shop/cart.py:4": "verified",
        "shop/cart.py:10": "unverified",
        "shop/payments.py:1": "invalid",
        "shop/cart.py:900": "invalid",
    }
    assert report.score == 0.25
    assert report.label == "1/4 citations verified"


def test_citation_with_shortened_path_resolves(sample_index):
    evidence = EvidenceLog()
    evidence.add("shop/auth.py", 1, 30, "read_file")
    report = verify_citations("In auth.py:6-8 the hash is computed.", sample_index, evidence)
    assert report.citations[0].status == "verified"
    assert report.citations[0].resolved_path == "shop/auth.py"


def test_truncated_search_snippets_only_count_shown_lines_as_evidence(sample_index):
    box = ToolBox(sample_index, EvidenceLog(), snippet_chars=60)
    result = box.call("search_code", {"query": "LoginManager login logout", "top_k": 1})
    chunk_range = next(r for r in box.evidence.ranges)
    assert "chunk continues to line" in result.text
    assert chunk_range.end_line - chunk_range.start_line + 1 == len(
        [line for line in result.text.splitlines() if " | " in line]
    )
