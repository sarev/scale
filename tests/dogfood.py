#!/usr/bin/env python3
"""
The dogfood harness: SCALE annotating its own source from a cold start.

Every `scale*.py` module is copied into a scratch area and "walled" - module/class/function docstrings, full-line
`#` comments, and blank lines *inside* function/method bodies are stripped (blank lines between routines and at
class/module level are kept, so the file shape survives). SCALE is then run over the walled copies so its output can
be judged against the hand-reviewed originals.

Two scratch directories are produced under `temp/dogfood/` and both are left in place for review:

    walled/     the stripped inputs, untouched by the run
    annotated/  the same files after SCALE has annotated them in place

Modes:
    --mode offline      one local-model run: definition docs + block comments + file descriptions. No network, no
                        stronger model - what a fully offline user gets.
    --mode escalation   the same def/block pass but with complex routines deferred into a run manifest
                        (`temp/dogfood/scale-manifest.json`) for a stronger model to fill. The harness stops after the
                        emit phase and prints the remaining loop commands; in Claude Code, the /scale skill drives
                        that loop end-to-end.

After the run (offline) or the emit phase (escalation) the harness verifies, deterministically, that re-walling each
annotated file reproduces the walled input's code lines byte-for-byte - SCALE's code-preservation guarantee checked
from the outside - and that every output still parses.

Usage:
    python tests/dogfood.py [--mode offline|escalation] [--files NAME ...] [--wall-only] [--cutoff N] [-- EXTRA...]

Anything after `--` is passed through to scale.py (e.g. `-- -m /path/model.gguf --n-ctx 12288`). This loads a real
GGUF and is SLOW on the full source set; use `--files` to dogfood a subset (e.g. `--files scale_text.py scale_log.py`).
"""
from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOGFOOD = ROOT / "temp" / "dogfood"
WALLED = DOGFOOD / "walled"
ANNOTATED = DOGFOOD / "annotated"
MANIFEST = DOGFOOD / "scale-manifest.json"
REWORD = DOGFOOD / "scale-reword.json"


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Return the line numbers of every module/class/function docstring that can be removed safely.

    A docstring is only reported when its owner has at least one further statement, so stripping it never leaves an
    empty (syntactically invalid) suite.
    """
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if not body or len(body) < 2:
                continue
            first = body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def wall(src: str, strip_all_blanks: bool = False) -> str:
    """
    Strip `src` to a dogfood wall: no docstrings, no full-line comments, no blank lines inside routine bodies.

    Blank lines *between* routines (and at class/module level) are kept so the file keeps its overall shape; pass
    `strip_all_blanks=True` to drop those too, which reduces a file to its code lines only (used for the
    code-preservation comparison). Blank/`#` lines inside non-docstring string literals are always protected, and a
    leading shebang is kept.
    """

    lines = src.split("\n")
    tree = ast.parse(src)
    doc_lines = _docstring_lines(tree)

    # Every line inside a function/method (including nested defs) - blanks here are stripped.
    in_routine: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            in_routine.update(range(node.lineno, node.end_lineno + 1))

    # Protect lines inside real (non-docstring) string literals from blank/comment stripping.
    protected: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and getattr(node, "lineno", None):
            protected.update(range(node.lineno, node.end_lineno + 1))
    protected -= doc_lines

    kept: list[tuple[int, str]] = []
    for i, line in enumerate(lines, start=1):
        if i == 1 and line.startswith("#!"):
            kept.append((i, line))
            continue
        if i in doc_lines:
            continue
        if i in protected:
            kept.append((i, line))
            continue
        s = line.strip()
        if s.startswith("#"):
            continue
        if s == "" and (strip_all_blanks or i in in_routine):
            continue
        kept.append((i, line))

    # Comment removal can leave stacked blank lines at module/class level; collapse runs beyond the usual two.
    # Protected (in-string) lines never count towards a run.
    out: list[str] = []
    blanks = 0
    for i, line in kept:
        if line.strip() == "" and i not in protected:
            blanks += 1
            if blanks > 2:
                continue
        else:
            blanks = 0
        out.append(line)
    while out and out[0].strip() == "":
        out.pop(0)

    text = "\n".join(out)
    if not text.endswith("\n"):
        text += "\n"
    return text


def code_lines(src: str) -> list[str]:
    """Reduce `src` to its code lines only (no docstrings/comments/blanks) for preservation comparison."""
    return [ln for ln in wall(src, strip_all_blanks=True).split("\n") if ln.strip()]


def prepare(files: list[Path]) -> None:
    """Build temp/dogfood: walled copies of `files` plus an identical annotated/ set for SCALE to write into."""
    if DOGFOOD.exists():
        shutil.rmtree(DOGFOOD)
    WALLED.mkdir(parents=True)
    ANNOTATED.mkdir(parents=True)

    print(f"Walling {len(files)} files into {WALLED.relative_to(ROOT)}/ ...")
    for f in files:
        src = f.read_text(encoding="utf-8")
        walled = wall(src)
        ast.parse(walled)  # the wall must still be valid Python
        for dest in (WALLED / f.name, ANNOTATED / f.name):
            dest.write_text(walled, encoding="utf-8", newline="\n")
        print(f"  {f.name}: {len(src.splitlines())} -> {len(walled.splitlines())} lines")


def run_scale(args: list[str]) -> int:
    """Run scale.py from the project root (so the default model path and scale-cfg resolve) and return its exit code."""
    cmd = [sys.executable, str(ROOT / "scale.py")] + args
    print("\n$ " + " ".join(args))
    return subprocess.run(cmd, cwd=ROOT).returncode


def verify(files: list[Path]) -> bool:
    """Check every annotated file still parses and that its code lines match the walled input exactly."""
    print("\n===== verification =====")
    ok = True
    for f in files:
        name = f.name
        walled_text = (WALLED / name).read_text(encoding="utf-8")
        annotated_text = (ANNOTATED / name).read_text(encoding="utf-8")
        try:
            ast.parse(annotated_text)
        except SyntaxError as exc:
            print(f"  {name}: FAIL - annotated output does not parse ({exc})")
            ok = False
            continue
        if code_lines(annotated_text) != code_lines(walled_text):
            print(f"  {name}: FAIL - code lines differ from the walled input")
            ok = False
            continue
        added = len(annotated_text.splitlines()) - len(walled_text.splitlines())
        print(f"  {name}: code preserved, +{added} lines of documentation")
    print("Verification " + ("PASSED" if ok else "FAILED"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Dogfood SCALE against its own (walled) source.")
    ap.add_argument("--mode", choices=("offline", "escalation"), default="offline",
                    help="offline = full local run; escalation = defer complex routines into a manifest")
    ap.add_argument("--files", nargs="+", metavar="NAME", default=None,
                    help="subset of source file names to dogfood (default: every scale*.py)")
    ap.add_argument("--wall-only", action="store_true", help="prepare the walled/annotated dirs and stop")
    ap.add_argument("--cutoff", type=int, default=10, help="escalation cognitive-complexity threshold (default 10)")
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    sources = sorted(ROOT.glob("scale*.py"))
    if args.files:
        wanted = set(args.files)
        sources = [f for f in sources if f.name in wanted]
        missing = wanted - {f.name for f in sources}
        if missing:
            ap.error(f"unknown source files: {', '.join(sorted(missing))}")

    prepare(sources)
    if args.wall_only:
        print(f"\nWall-only run complete; inputs are in {WALLED.relative_to(ROOT)}/")
        return 0

    targets = [str((ANNOTATED / f.name).relative_to(ROOT)) for f in sources]
    # Multiple targets are annotated in place; a single target without -o would print to stdout instead.
    out_flag = ["-o", targets[0]] if len(targets) == 1 else []
    base = ["-c", "--block-comments", "medium", "-l", "python",
            "--project-doc", "README.md", "-v"] + extra

    if args.mode == "offline":
        rc = run_scale(base + ["--file-doc"] + targets + out_flag)
        if rc != 0:
            print(f"\nSCALE exited with {rc}")
            return rc
        ok = verify(sources)
        print(f"\nReview the results:\n  walled inputs:     {WALLED.relative_to(ROOT)}/"
              f"\n  annotated outputs: {ANNOTATED.relative_to(ROOT)}/"
              f"\n  diff:              git diff --no-index {WALLED.relative_to(ROOT)} {ANNOTATED.relative_to(ROOT)}")
        return 0 if ok else 1

    # Escalation: emit phase only - the manifest needs a stronger model before the loop can finish.
    rc = run_scale(base + ["--escalate-cognitive", str(args.cutoff),
                           "--emit-manifest", str(MANIFEST.relative_to(ROOT))] + targets + out_flag)
    if rc != 0:
        print(f"\nSCALE exited with {rc}")
        return rc
    ok = verify(sources)
    run_scale(["--check-manifest", str(MANIFEST.relative_to(ROOT))])  # informational: how much was deferred

    m, r, t = MANIFEST.relative_to(ROOT), REWORD.relative_to(ROOT), " ".join(targets + out_flag)
    print(f"\nEmit phase complete. Finish the loop with a stronger model filling the manifest answers, then:\n"
          f"  python scale.py --check-manifest {m}\n"
          f"  python scale.py -l python --apply-manifest {m} {t}\n"
          f"  python scale.py --file-doc -l python --project-doc README.md --emit-reword {r} {t}\n"
          f"  ...fill the reword answers...\n"
          f"  python scale.py -l python --apply-reword {r} {t}\n"
          f"In Claude Code, the /scale skill drives this whole loop.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
