#!/usr/bin/env python3
"""
The multi-file run model: positional targets accept files/directories/globs (expanded to a deduped, ordered source
list); a `--reference` set is captured read-only; `scan_run_files` loads and parses each run file exactly once into
the retained store the pre-passes share (references are parsed, never summarised). Also guards the CLI rules that
fail fast (before the model loads): no matches, -o with multiple targets, and single-target-only manifest phases.

Model-free: the helper tests are pure, and the main() guard tests return before any LLM is loaded.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_project  # noqa: E402
from scale import _parse_args, main, _scan_run_files, _build_call_graph, _symbol_provider_for  # noqa: E402


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


def test_scan_run_files():
    # The merged pre-pass: every run file (target ∪ reference) is loaded and parsed exactly once into the retained
    # store - source text, language, target/reference flag, and symbols all captured per file.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        util = root / "util.py"
        core = root / "core.py"
        util.write_text('def clamp(x):\n    """Clamp x to range."""\n    return x\n', encoding="utf-8")
        core.write_text("def run(xs):\n    return [clamp(x) for x in xs]\n", encoding="utf-8")

        loads: list = []
        from scale import load_source

        def counting_load(p):
            loads.append(str(p))
            return load_source(p, "python")

        run_files = scale_project.scan_run_files([core], [util], counting_load, _symbol_provider_for)
        uk, ck = str(util.resolve()), str(core.resolve())

        # One load per file, no re-reads; targets and references both retained, flagged correctly.
        assert sorted(loads) == sorted([str(core), str(util)])
        assert run_files[ck].is_target and not run_files[uk].is_target
        assert run_files[ck].language == "python" and run_files[ck].source_lines[0].startswith("def run")
        # Symbols are parsed in (with the full span, for the lazy one-liner generator).
        clamp = next(s for s in run_files[uk].symbols if s.qualname == "clamp")
        assert clamp.end >= clamp.start > 0

        # A duplicate path (target repeated as reference) is scanned once.
        again: list = []
        run_files2 = scale_project.scan_run_files([core], [core], lambda p: (again.append(p), load_source(p, "python"))[1],
                                                  _symbol_provider_for)
        assert len(again) == 1 and list(run_files2) == [ck]


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

        run_files = _scan_run_files([core, util], [], "python")
        graph, store = _build_call_graph(run_files)
        assert graph is not None and store is not None

        uk, ck = str(util.resolve()), str(core.resolve())
        # The cross-file free call resolves (run-wide unique) and the callee's file is ordered first.
        assert graph.edges[(ck, "run")] == [(uk, "clamp")]
        assert graph.file_order([ck, uk]) == [uk, ck]
        # The store is seeded from clamp's existing docstring, so run already has its callee's contract.
        assert "clamp: Clamp x to range." in store.callee_notes(ck, "run")


def main_():
    test_gather_files()
    test_scan_run_files()
    test_parse_args_multi_target_and_reference()
    test_main_guards_fail_fast()
    test_build_call_graph()
    print("PASS: target/reference expansion, the retained run-file scan, the multi-target CLI guards, and the call-graph pre-pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())
