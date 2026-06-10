#!/usr/bin/env python3
"""
The multi-file run model: positional targets accept files/directories/globs (expanded to a deduped, ordered source
list); a `--reference` set is captured read-only; the project blurb and reference one-liners compose one context
string. Also guards the CLI rules that fail fast (before the model loads): no matches, -o with multiple targets, and
single-target-only manifest phases.

Model-free: the helper tests are pure, and the main() guard tests return before any LLM is loaded.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_project  # noqa: E402
from scale import _parse_args, main, _build_call_graph, _symbol_provider_for  # noqa: E402


def test_gather_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "a.c").write_text("int a;", encoding="utf-8")
        (root / "b.py").write_text("b = 1", encoding="utf-8")
        (root / "README.md").write_text("# doc", encoding="utf-8")
        (root / "data.txt").write_text("data", encoding="utf-8")
        sub = root / "sub"
        sub.mkdir()
        (sub / "c.c").write_text("int c;", encoding="utf-8")

        # A directory expands recursively to source files only (no .md/.txt).
        names = {p.name for p in scale_project.gather_files([str(root)])}
        assert names == {"a.c", "b.py", "c.c"}, names

        # An explicit non-source file is taken as-is (extension ignored for explicit paths).
        assert scale_project.gather_files([str(root / "README.md")]) == [root / "README.md"]

        # A glob matches by pattern; results are deduped across overlapping patterns and sorted.
        got = scale_project.gather_files([str(root / "*.c"), str(root / "a.c")])
        assert [p.name for p in got] == ["a.c"], got


def test_compose_project_context():
    assert isinstance(scale_project.MAX_REFERENCE_FILES, int) and scale_project.MAX_REFERENCE_FILES > 0
    assert scale_project.compose_project_context("", []) == ""
    assert scale_project.compose_project_context("Blurb here.", []) == "Blurb here."
    only_related = scale_project.compose_project_context("", [("err.h", "error codes")])
    assert "Related files" in only_related and "- err.h: error codes" in only_related
    both = scale_project.compose_project_context("Blurb.", [("err.h", "error codes")])
    assert both.startswith("Blurb.") and "- err.h: error codes" in both


def test_parse_args_multi_target_and_reference():
    args = _parse_args(["a.c", "b.c", "-c", "--reference", "inc/", "--reference", "x.h"])
    assert args.source == ["a.c", "b.c"]
    assert args.reference == ["inc/", "x.h"]


def test_main_guards_fail_fast():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        f1 = root / "a.py"
        f2 = root / "b.py"
        f1.write_text("x = 1", encoding="utf-8")
        f2.write_text("y = 2", encoding="utf-8")

        # No matches -> error (before any model load).
        assert main([str(root / "nope.zzz"), "-c"]) == 1

        # -o with multiple targets is rejected.
        assert main([str(f1), str(f2), "-c", "-o", str(root / "out.py")]) == 1

        # Manifest phases are single-target only.
        assert main([str(f1), str(f2), "-c", "--emit-manifest", str(root / "m.json")]) == 1
        assert main([str(f1), str(f2), "--apply-manifest", str(root / "m.json")]) == 1


def test_build_call_graph():
    # The model-free call-graph pre-pass: parses targets ∪ references into a graph + seeded store, links a cross-file
    # call, and orders the callee's file first. A reference seeds a contract but is not itself a documentation target.
    assert _symbol_provider_for("python") is not None
    assert _symbol_provider_for("ruby") is None     # unsupported language -> skipped, not an error

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        util = root / "util.py"
        core = root / "core.py"
        util.write_text('def clamp(x):\n    """Clamp x to range."""\n    return x\n', encoding="utf-8")
        core.write_text("def run(xs):\n    return [clamp(x) for x in xs]\n", encoding="utf-8")

        graph, store = _build_call_graph([core, util], [], "python")
        assert graph is not None and store is not None

        uk, ck = str(util.resolve()), str(core.resolve())
        # The cross-file free call resolves (run-wide unique) and the callee's file is ordered first.
        assert graph.edges[(ck, "run")] == [(uk, "clamp")]
        assert graph.file_order([ck, uk]) == [uk, ck]
        # The store is seeded from clamp's existing docstring, so run already has its callee's contract.
        assert "clamp: Clamp x to range." in store.callee_notes(ck, "run")


def main_():
    test_gather_files()
    test_compose_project_context()
    test_parse_args_multi_target_and_reference()
    test_main_guards_fail_fast()
    test_build_call_graph()
    print("PASS: target/reference expansion, project-context compose, the multi-target CLI guards, and the call-graph pre-pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())
