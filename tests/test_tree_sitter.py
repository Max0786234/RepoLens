"""Tree-sitter chunking, and the regex fallback that must keep working without it."""

import textwrap

import pytest

from repolens import chunking, ts_chunking
from repolens.chunking import chunk_file

JS_SOURCE = """
/**
 * Make a server.
 */

function createServer(port) {
  setImmediate(function () {
    run();
  });
  return port;
}

class Limiter {
  allow(id) {
    return true;
  }
}

res.contentType = res.type = function contentType(type) {
  return this.set('Content-Type', type);
};

defineGetter(req, 'ip', function ip() {
  return this.socket.remoteAddress;
});
"""


def chunks_for(path, source, language):
    return chunk_file(path, textwrap.dedent(source).strip("\n") + "\n", language)


@pytest.fixture(params=["tree-sitter", "regex"])
def backend(request, monkeypatch):
    if request.param == "tree-sitter":
        pytest.importorskip("tree_sitter_javascript")
        monkeypatch.setattr(chunking, "USE_TREE_SITTER", True)
    else:
        monkeypatch.setattr(chunking, "USE_TREE_SITTER", False)
    return request.param


def test_javascript_definitions_with_both_backends(backend):
    chunks, symbols = chunks_for("s.js", JS_SOURCE, "javascript")
    quals = {s.qualname for s in symbols}
    assert {"createServer", "Limiter", "Limiter.allow", "req.ip"} <= quals
    if backend == "tree-sitter":
        # One-line chained alias (res.contentType = res.type = function ...) is beyond the regex parser.
        assert "res.type" in quals
    assert "setImmediate" not in quals  # a call with a callback is not a definition
    create = next(c for c in chunks if c.symbol == "createServer")
    assert create.start_line == 1 and "Make a server" in create.text  # JSDoc attached


def test_tree_sitter_skips_nested_callbacks_that_regex_reports():
    pytest.importorskip("tree_sitter_javascript")
    source = "function outer() {\n  function onerror(err) {\n    return err;\n  }\n  return onerror;\n}\n"
    _, symbols = chunk_file("n.js", source, "javascript")
    assert [s.qualname for s in symbols] == ["outer"]


@pytest.mark.parametrize(
    ("path", "language", "source", "expected"),
    [
        (
            "main.go",
            "go",
            "package main\n\ntype Server struct{}\n\nfunc (s *Server) Start() error { return nil }\n\nfunc main() {}\n",
            {"Server", "Server.Start", "main"},
        ),
        (
            "lib.rs",
            "rust",
            "struct Point { x: i32 }\nimpl Point {\n    fn new() -> Self { Point { x: 0 } }\n}\nfn free() {}\n",
            {"Point", "Point.new", "free"},
        ),
        (
            "Shop.java",
            "java",
            "public class Shop {\n  public int total(int a) { return a; }\n  public Shop() {}\n}\n",
            {"Shop", "Shop.total", "Shop.Shop"},
        ),
        (
            "repo.ts",
            "typescript",
            "export interface User { id: number }\nexport class Repo {\n  find(id: number) { return id; }\n}\n"
            "export const handler = async (req: Req) => 1;\n",
            {"User", "Repo", "Repo.find", "handler"},
        ),
        (
            "app.tsx",
            "typescript",
            "export function App() { return <div/>; }\nconst Button = ({x}: P) => <button>{x}</button>;\n",
            {"App", "Button"},
        ),
        (
            "net.cpp",
            "cpp",
            "namespace net {\nclass Socket {\n public:\n  int send() { return 0; }\n};\n}\nstatic int helper(int a) { return a; }\n",
            {"net", "net.Socket", "net.Socket.send", "helper"},
        ),
        ("add.c", "c", "struct node { int v; };\nint add(int a, int b) { return a + b; }\n", {"node", "add"}),
        (
            "Cart.cs",
            "csharp",
            "namespace App {\n  public class Cart {\n    public decimal Total() { return 0; }\n  }\n}\n",
            {"App", "App.Cart", "App.Cart.Total"},
        ),
        (
            "cart.rb",
            "ruby",
            "module Shop\n  class Cart\n    def total; 0; end\n  end\nend\n",
            {"Shop", "Shop.Cart", "Shop.Cart.total"},
        ),
        (
            "cart.php",
            "php",
            "<?php\nclass Cart {\n  public function total() { return 0; }\n}\nfunction helper() {}\n",
            {"Cart", "Cart.total", "helper"},
        ),
    ],
)
def test_tree_sitter_symbols_per_language(path, language, source, expected):
    grammar = "tsx" if path.endswith(".tsx") else language
    if ts_chunking.get_parser(grammar) is None:
        pytest.skip(f"tree-sitter grammar for {grammar} not installed")
    _, symbols = chunk_file(path, source, language)
    assert expected <= {s.qualname for s in symbols}


def test_missing_grammar_falls_back_to_regex(monkeypatch):
    monkeypatch.setattr(ts_chunking, "get_parser", lambda language: None)
    _, symbols = chunks_for("s.js", JS_SOURCE, "javascript")
    assert "createServer" in {s.qualname for s in symbols}


def test_large_class_is_split_into_members_with_tree_sitter():
    pytest.importorskip("tree_sitter_java")
    methods = "\n".join(
        f"  public int m{i}() {{\n" + "".join(f"    int x{j} = {j};\n" for j in range(12)) + "    return 0;\n  }"
        for i in range(8)
    )
    source = f"public class Big {{\n  private int field;\n{methods}\n}}\n"
    chunks, _ = chunk_file("Big.java", source, "java")
    assert {"Big", "Big.m0", "Big.m7"} <= {c.symbol for c in chunks}
    assert all(c.end_line - c.start_line + 1 <= chunking.MAX_CHUNK_LINES for c in chunks)
