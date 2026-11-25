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
    assert len(box.schemas()) == 9


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
    shown = box.evidence.ranges[0]
    assert "showing lines" in result.text
    numbered = [line for line in result.text.splitlines() if " | " in line]
    assert shown.end_line - shown.start_line + 1 == len(numbered)


def test_query_snippet_centres_on_the_match():
    from repolens.tools import query_snippet

    text = "\n".join([f"filler_line_{i} = {i}" for i in range(80)] + ["DEFAULT_RETRIES = 0"])
    lines, offset, truncated = query_snippet(text, "DEFAULT_RETRIES", 300)
    assert truncated and "DEFAULT_RETRIES = 0" in lines
    assert offset > 0  # not simply the head of the chunk
    assert query_snippet("short", "x", 300) == (["short"], 0, False)


def test_search_code_duplicates_ignore_top_k():
    from repolens.agent import call_key

    assert call_key("search_code", {"query": "Retry ", "top_k": 5}) == call_key(
        "search_code", {"query": "retry", "top_k": 20}
    )
    assert call_key("grep", {"pattern": "a"}) != call_key("grep", {"pattern": "b"})


def test_linkify_citations_handles_any_wrapping_and_skips_code():
    from repolens.citations import linkify_citations

    def url_for(path, start, end):
        return None if path == "missing.py" else f"https://x/{path}#L{start}-L{end}"

    text = (
        "See (src/a.py:1-3), 【src/a.py:5-9】, `src/b.py:4` and src/a.py:1 plus missing.py:2.\n"
        "```python\nprint('src/a.py:1-3')\n```"
    )
    out = linkify_citations(text, url_for)
    assert "([`src/a.py:1-3`](https://x/src/a.py#L1-L3))" in out
    assert "【[`src/a.py:5-9`](https://x/src/a.py#L5-L9)】" in out
    assert "[`src/b.py:4`](https://x/src/b.py#L4-L4)" in out
    assert "[`src/a.py:1`](https://x/src/a.py#L1-L1)" in out
    assert "missing.py:2" in out and "x/missing.py" not in out
    assert "print('src/a.py:1-3')" in out  # code blocks untouched
    assert out.count("](https://x/src/a.py#L1-L3)") == 1  # no nested links


def test_find_callers_and_callees(sample_index):
    box = make_box(sample_index)
    callers = box.call("find_callers", {"name": "hash_password"})
    assert "LoginManager.login" in callers.text  # shop/auth.py calls it inside login()
    assert box.evidence.covers("shop/auth.py", 19, 19)  # the call line inside login()
    callees = box.call("find_callees", {"name": "LoginManager.login"})
    assert "hash_password() -> defined at shop/auth.py:" in callees.text
    missing = box.call("find_callees", {"name": "does_not_exist"})
    assert "No function or method" in missing.text


def test_call_graph_ignores_strings_comments_and_definition_lines(tmp_path):
    from repolens.index import CodeIndex

    (tmp_path / "m.py").write_text(
        "def target():\n    return 1\n\n\ndef user():\n    # target() in a comment\n    s = 'target()'\n    return target()\n",
        encoding="utf-8",
    )
    graph = CodeIndex.build(tmp_path).call_graph()
    sites = graph.callers("target")
    assert [(s.caller.qualname, s.line) for s in sites] == [("user", 8)]


def test_prose_with_dots_is_not_a_citation():
    text = (
        "It resolves `sub.example.com` (lines 848-859) and catches socket.gaierror (lines 861-868); "
        "see src/requests/utils.py:819-820 and setup.cfg:3."
    )
    found = {(c.path, c.start_line) for c in extract_citations(text)}
    assert found == {("src/requests/utils.py", 819), ("setup.cfg", 3)}
