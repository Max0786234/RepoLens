"""'Explain this change': diff parsing, impact analysis and the explanation (no network, no model)."""

import json

import pytest

from repolens import change as change_mod
from repolens.change import ChangeError, analyze_change, explain_change, parse_pr_url, parse_unified_diff
from repolens.llm import ChatResponse, ScriptedClient


def line_of(path, needle):
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return number
    raise AssertionError(needle)


def modify_line_diff(rel, number, old, new):
    return (
        f"diff --git a/{rel} b/{rel}\n--- a/{rel}\n+++ b/{rel}\n"
        f"@@ -{number},1 +{number},1 @@ def context\n-{old}\n+{new}\n"
    )


def test_parse_unified_diff_statuses_and_line_numbers():
    diff = (
        "diff --git a/src/a.py b/src/a.py\n--- a/src/a.py\n+++ b/src/a.py\n"
        "@@ -10,3 +10,4 @@ def f():\n     keep\n-    old\n+    new\n+    extra\n     keep\n"
        "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n"
        "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-z = 3\n"
        "diff --git a/old.py b/moved.py\nrename from old.py\nrename to moved.py\n--- a/old.py\n+++ b/moved.py\n"
        "@@ -1 +1 @@\n-a = 1\n+a = 2\n\\ No newline at end of file\n"
    )
    files = {f.path: f for f in parse_unified_diff(diff)}
    a = files["src/a.py"]
    assert a.status == "modified" and a.additions == 2 and a.deletions == 1
    assert a.hunks[0].added == [(11, "    new"), (12, "    extra")]
    assert a.hunks[0].removed == [(11, "    old")]
    assert files["new.py"].status == "added" and files["new.py"].touched_new_lines() == [1, 2]
    assert files["gone.py"].status == "deleted"
    assert files["moved.py"].status == "renamed" and files["moved.py"].old_path == "old.py"


def test_plain_diff_u_without_git_headers():
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n--- a/y.py\n+++ b/y.py\n@@ -2 +2 @@\n-c\n+d\n"
    assert [f.path for f in parse_unified_diff(diff)] == ["x.py", "y.py"]


def test_too_large_diff_is_refused(monkeypatch):
    monkeypatch.setattr(change_mod, "MAX_DIFF_BYTES", 10)
    with pytest.raises(ChangeError, match="larger than"):
        parse_unified_diff("--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n")


def test_analyze_finds_changed_function_callers_and_tests(sample_index, sample_repo):
    cart = sample_repo / "shop" / "cart.py"
    number = line_of(cart, "subtotal = sum(")
    old = cart.read_text(encoding="utf-8").splitlines()[number - 1]
    diff = modify_line_diff("shop/cart.py", number, old, old.replace('item["qty"]', 'max(item["qty"], 0)'))
    report = analyze_change(sample_index, parse_unified_diff(diff), title="Clamp negative quantities")
    assert [c.symbol.qualname for c in report.changed] == ["compute_total"]
    assert [t.caller for t in report.tests] == ["test_compute_total"]  # tests/test_cart.py calls it
    assert report.evidence.covers("shop/cart.py", number, number)
    text = report.summary_text()
    assert "compute_total" in text and "tests/test_cart.py" in text and "Clamp negative quantities" in text


def test_callers_outside_the_change_are_reported(sample_index, sample_repo):
    auth = sample_repo / "shop" / "auth.py"
    number = line_of(auth, "return hashlib.sha256")
    old = auth.read_text(encoding="utf-8").splitlines()[number - 1]
    report = analyze_change(
        sample_index, parse_unified_diff(modify_line_diff("shop/auth.py", number, old, old.replace("sha256", "sha512")))
    )
    assert [c.symbol.qualname for c in report.changed] == ["hash_password"]
    assert ("LoginManager.login", "hash_password") in {(s.caller, s.callee) for s in report.callers}


def test_unknown_and_deleted_files_are_listed(sample_index):
    diff = (
        "--- a/nowhere.py\n+++ b/nowhere.py\n@@ -1 +1 @@\n-a\n+b\n--- a/shop/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    )
    report = analyze_change(sample_index, parse_unified_diff(diff))
    assert set(report.missing_files) == {"nowhere.py", "shop/old.py"}
    assert report.changed == []


def test_explanation_citations_are_verified(sample_index, sample_repo):
    cart = sample_repo / "shop" / "cart.py"
    number = line_of(cart, "subtotal = sum(")
    old = cart.read_text(encoding="utf-8").splitlines()[number - 1]
    report = analyze_change(
        sample_index, parse_unified_diff(modify_line_diff("shop/cart.py", number, old, old + "  # x"))
    )
    llm = ScriptedClient(
        [ChatResponse(f"**Summary** Comment added in shop/cart.py:{number}. Unrelated: shop/auth.py:1-2.")]
    )
    result = explain_change(llm, sample_index, report)
    statuses = {c.path: c.status for c in result.grounding.citations}
    assert statuses == {"shop/cart.py": "verified", "shop/auth.py": "unverified"}
    prompt = llm.calls[0][-1]["content"]
    assert "Changed code after the change" in prompt and "compute_total" in prompt


def test_parse_pr_url():
    assert parse_pr_url("https://github.com/psf/requests/pull/6767") == ("psf", "requests", 6767)
    assert parse_pr_url("github.com/a/b/pull/1/files") == ("a", "b", 1)
    assert parse_pr_url("https://github.com/a/b/issues/1") is None


def test_fetch_pull_request_uses_api_metadata_and_diff(monkeypatch):
    calls = []

    def fake_get(url, accept, token):
        calls.append(accept)
        if accept.endswith("diff"):
            return b"--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        return json.dumps({"title": "Fix x", "body": None, "head": {"sha": "abc123"}}).encode()

    monkeypatch.setattr(change_mod, "_github_get", fake_get)
    pr = change_mod.fetch_pull_request("https://github.com/o/r/pull/5")
    assert (pr.title, pr.head_sha, pr.body) == ("Fix x", "abc123", "")
    assert pr.snapshot_url == "https://github.com/o/r/tree/abc123"
    assert parse_unified_diff(pr.diff)[0].path == "x.py"
    with pytest.raises(ChangeError):
        change_mod.fetch_pull_request("https://example.com/not-a-pr")
