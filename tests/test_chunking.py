import textwrap

from repolens.chunking import MAX_CHUNK_LINES, chunk_file


def chunks_for(path, source, language):
    return chunk_file(path, textwrap.dedent(source).lstrip("\n"), language)


def test_python_functions_and_classes_become_chunks():
    chunks, symbols = chunks_for(
        "m.py",
        """
        import os


        @decorator
        def first(a):
            return a


        class Thing:
            def method(self):
                return 1
        """,
        "python",
    )
    by_symbol = {c.symbol: c for c in chunks}
    assert by_symbol["first"].start_line == 4  # decorator included
    assert by_symbol["first"].kind == "function"
    assert by_symbol["Thing"].kind == "class"
    assert {s.qualname for s in symbols} == {"first", "Thing", "Thing.method"}


def test_large_python_class_is_split_into_methods():
    methods = "\n".join(f"    def m{i}(self):\n" + "".join(f"        x = {j}\n" for j in range(12)) for i in range(10))
    source = "class Big:\n    '''Doc.'''\n\n" + methods
    chunks, _ = chunk_file("big.py", source, "python")
    symbols = [c.symbol for c in chunks]
    assert "Big" in symbols  # header chunk
    assert "Big.m0" in symbols and "Big.m9" in symbols
    assert all(c.end_line - c.start_line + 1 <= MAX_CHUNK_LINES for c in chunks)


def test_oversized_function_is_windowed_with_symbol_name():
    body = "".join(f"    x{i} = {i}\n" for i in range(200))
    chunks, _ = chunk_file("long.py", "def long_one():\n" + body, "python")
    assert len(chunks) > 1
    assert all((c.symbol or "").startswith("long_one") for c in chunks)
    assert chunks[0].start_line == 1 and chunks[-1].end_line == 201


def test_invalid_python_falls_back_to_windows():
    chunks, symbols = chunk_file("broken.py", "def oops(:\n    pass\n", "python")
    assert chunks and not symbols


def test_javascript_functions_classes_and_methods():
    chunks, symbols = chunks_for(
        "s.js",
        """
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

        res.status = function status(code) {
          this.statusCode = code;
        };
        """,
        "javascript",
    )
    quals = {s.qualname for s in symbols}
    assert {"createServer", "Limiter", "Limiter.allow", "res.status"} <= quals
    assert "setImmediate" not in quals  # a call with a callback is not a definition
    create = next(c for c in chunks if c.symbol == "createServer")
    assert create.start_line == 1  # JSDoc attached despite the blank line
    assert "Make a server" in create.text


def test_go_methods_are_qualified_by_receiver():
    _, symbols = chunks_for(
        "main.go",
        """
        package main

        type Server struct {
            port int
        }

        func (s *Server) Start() error {
            return nil
        }

        func main() {
        }
        """,
        "go",
    )
    quals = {s.qualname for s in symbols}
    assert {"Server", "Server.Start", "main"} <= quals


def test_markdown_splits_by_heading():
    chunks, _ = chunks_for(
        "README.md",
        """
        # Title
        intro

        ## Install
        pip install x

        ```bash
        # not a heading
        ```
        """,
        "markdown",
    )
    assert [c.symbol for c in chunks] == ["Title", "Install"]


def test_chunks_cover_every_nonblank_line():
    source = "import a\n\nX = 1\n\ndef f():\n    return X\n\nprint(f())\n"
    chunks, _ = chunk_file("cov.py", source, "python")
    covered = set()
    for c in chunks:
        covered.update(range(c.start_line, c.end_line + 1))
    for number, line in enumerate(source.splitlines(), start=1):
        if line.strip():
            assert number in covered
