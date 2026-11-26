"""MCP server: JSON-RPC handling, tools and the stdio loop (no network, no model)."""

import io
import json

import pytest

from repolens.mcp_server import LATEST_VERSION, RepoLensMCPServer, serve


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("REPOLENS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("REPOLENS_EMBED_BACKEND", "none")
    return RepoLensMCPServer()


def call(server, method, params=None, msg_id=1):
    message = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        message["params"] = params
    return server.handle(message)


def tool(server, name, **arguments):
    response = call(server, "tools/call", {"name": name, "arguments": arguments})
    result = response["result"]
    return result["content"][0]["text"], result["isError"]


def test_initialize_negotiates_protocol_version(server):
    result = call(server, "initialize", {"protocolVersion": "2025-03-26", "capabilities": {}})["result"]
    assert result["protocolVersion"] == "2025-03-26"
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert result["serverInfo"]["name"] == "repolens" and "verify_citations" in result["instructions"]
    unknown = call(server, "initialize", {"protocolVersion": "1999-01-01"})["result"]
    assert unknown["protocolVersion"] == LATEST_VERSION


def test_tools_list_has_input_schemas(server):
    tools = call(server, "tools/list")["result"]["tools"]
    names = [t["name"] for t in tools]
    assert names[0] == "open_repository" and names[-1] == "verify_citations"
    assert "analyze_change" in names
    assert {"search_code", "read_file", "find_callers", "find_callees"} <= set(names)
    assert all(t["inputSchema"]["type"] == "object" and t["description"] for t in tools)


def test_tools_require_an_open_repository(server):
    text, is_error = tool(server, "search_code", query="anything")
    assert is_error and "open_repository" in text


def test_open_search_read_and_verify(server, sample_repo):
    text, is_error = tool(server, "open_repository", source=str(sample_repo))
    assert not is_error and "files" in text
    text, is_error = tool(server, "read_file", path="shop/cart.py", start_line=4, end_line=7)
    assert not is_error and "compute_total" in text
    report, _ = tool(
        server, "verify_citations", text="Totals in shop/cart.py:4-7; also shop/cart.py:10-13 and nope.py:1."
    )
    assert "1/3 citations verified" in report
    assert "VERIFIED: shop/cart.py:4-7" in report
    assert "NOT INSPECTED: shop/cart.py:10-13" in report and "NOT FOUND: nope.py:1-1" in report


def test_opening_a_new_repository_resets_evidence(server, sample_repo):
    tool(server, "open_repository", source=str(sample_repo))
    tool(server, "read_file", path="shop/cart.py")
    tool(server, "open_repository", source=str(sample_repo))
    report, _ = tool(server, "verify_citations", text="shop/cart.py:4-7")
    assert "NOT INSPECTED" in report


def test_bad_source_and_unknown_tool(server, tmp_path):
    text, is_error = tool(server, "open_repository", source=str(tmp_path / "missing"))
    assert is_error and "Could not open" in text
    response = call(server, "tools/call", {"name": "rm_rf", "arguments": {}})
    assert response["error"]["code"] == -32602


def test_notifications_get_no_response_and_unknown_methods_error(server):
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert call(server, "ping")["result"] == {}
    assert call(server, "resources/list")["error"]["code"] == -32601


def test_stdio_loop_handles_parse_errors_and_batches(server):
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}),
        "this is not json",
        json.dumps([{"jsonrpc": "2.0", "id": 2, "method": "ping"}, {"jsonrpc": "2.0", "method": "notifications/x"}]),
        "",
    ]
    stdin = io.BytesIO(("\n".join(lines) + "\n").encode())
    stdout = io.BytesIO()
    serve(server, stdin, stdout)
    replies = [json.loads(line) for line in stdout.getvalue().decode().splitlines()]
    assert [r.get("id") for r in replies] == [1, None, 2]
    assert replies[1]["error"]["code"] == -32700


def test_analyze_change_tool(server, sample_repo):
    tool(server, "open_repository", source=str(sample_repo))
    lines = (sample_repo / "shop" / "cart.py").read_text(encoding="utf-8").splitlines()
    number = next(i for i, line in enumerate(lines, start=1) if "subtotal = sum(" in line)
    diff = (
        f"--- a/shop/cart.py\n+++ b/shop/cart.py\n@@ -{number} +{number} @@\n"
        f"-{lines[number - 1]}\n+{lines[number - 1]}  # x\n"
    )
    text, is_error = tool(server, "analyze_change", diff=diff)
    assert not is_error and "compute_total" in text and "tests/test_cart.py" in text
    report, _ = tool(server, "verify_citations", text=f"See shop/cart.py:{number}")
    assert "VERIFIED" in report  # lines from the change report count as evidence
    _, is_error = tool(server, "analyze_change", diff="not a diff")
    assert is_error
