"""Architecture map: module dependencies from the call graph."""

import textwrap

from repolens.archmap import build_architecture_map, is_example_path
from repolens.index import CodeIndex


def write(root, files):
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return root


LAYERED = {
    "app/main.py": """
        from app.service import place_order


        def run():
            return place_order(3)
    """,
    "app/service.py": """
        from app.db import save, load


        def place_order(qty):
            save(qty)
            return load()
    """,
    "app/db.py": """
        def save(value):
            return value


        def load():
            return 1
    """,
    "tests/test_service.py": """
        from app.service import place_order


        def test_place_order():
            assert place_order(1) == 1
    """,
    "examples/demo.py": """
        from app.main import run


        def demo():
            return run()
    """,
}


def test_layered_dependencies_at_file_level(tmp_path):
    index = CodeIndex.build(write(tmp_path, LAYERED))
    amap = build_architecture_map(index)
    assert amap.level == "file"
    assert set(amap.modules) == {"app/main.py", "app/service.py", "app/db.py"}  # tests/examples excluded
    assert amap.edges == {("app/main.py", "app/service.py"): 1, ("app/service.py", "app/db.py"): 2}
    text = amap.summary_text()
    assert "All modules are under app/" in text
    assert "Entry points (call others, nobody calls them): main.py" in text
    assert "db.py: used by 2 call sites" in text


def test_tests_and_examples_can_be_included(tmp_path):
    index = CodeIndex.build(write(tmp_path, LAYERED))
    amap = build_architecture_map(index, include_tests=True, include_examples=True)
    assert ("tests/test_service.py", "app/service.py") in amap.edges
    assert ("examples/demo.py", "app/main.py") in amap.edges


def test_ambiguous_names_are_skipped_and_counted(tmp_path):
    files = {
        "a.py": "def helper():\n    return 1\n",
        "b.py": "def helper():\n    return 2\n",
        "c.py": "def use():\n    return helper()\n",
    }
    amap = build_architecture_map(CodeIndex.build(write(tmp_path, files)))
    assert amap.edges == {} and amap.skipped_ambiguous == 1
    assert "left out (ambiguous)" in amap.summary_text()


def test_method_calls_do_not_link_to_top_level_functions(tmp_path):
    files = {
        "lib/core.py": "def handle(app):\n    return app.get_db()\n",
        "lib/other.py": "def get_db():\n    return None\n",
    }
    amap = build_architecture_map(CodeIndex.build(write(tmp_path, files)))
    assert amap.edges == {}  # app.get_db() is a method call, not a call to other.get_db


def test_large_projects_are_grouped_and_folded(tmp_path):
    files = {}
    for pkg in range(8):
        for mod in range(3):
            files[f"pkg{pkg}/m{mod}.py"] = f"def f{pkg}_{mod}():\n    return f{(pkg + 1) % 8}_0()\n"
    amap = build_architecture_map(CodeIndex.build(write(tmp_path, files)), max_nodes=6)
    assert amap.level == "folder"
    assert len(amap.modules) <= 6 and "(other)" in amap.modules and amap.merged
    assert all(src != dst for src, dst in amap.edges)


def test_renderers(tmp_path):
    amap = build_architecture_map(CodeIndex.build(write(tmp_path, LAYERED)))
    mermaid = amap.to_mermaid()
    assert mermaid.startswith("flowchart LR") and "-->|2|" in mermaid
    dot = amap.to_dot()
    assert dot.startswith("digraph architecture {") and '"app/service.py" -> "app/db.py"' in dot


def test_example_path_detection():
    assert is_example_path("examples/auth/index.js") and is_example_path("docs/conf.py")
    assert not is_example_path("src/examples_helper.py") and not is_example_path("lib/app.py")


def test_common_prefix_is_shown_once(tmp_path):
    files = {
        f"src/pkg/{name}": content
        for name, content in {k.split("/")[-1]: v for k, v in LAYERED.items() if k.startswith("app/")}.items()
    }
    amap = build_architecture_map(CodeIndex.build(write(tmp_path, files)))
    assert amap.common_prefix == "src/pkg/"
    assert '"main.py<br/>' in amap.to_mermaid() and "All modules are under src/pkg/" in amap.summary_text()


def test_dot_draws_only_the_strongest_edges(tmp_path):
    amap = build_architecture_map(CodeIndex.build(write(tmp_path, LAYERED)))
    dot = amap.to_dot(max_edges=1)
    assert '"app/service.py" -> "app/db.py"' in dot and '"app/main.py" -> ' not in dot
    assert '"app/main.py" [label' not in dot  # nodes without drawn edges are left out
