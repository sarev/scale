#!/usr/bin/env python3
"""
The dogfood harness: SCALE annotating its own source from a cold start.

Every `scale*.py` module is copied into a scratch area and "walled" - module/class/function docstrings, full-line
`#` comments, and blank lines *inside* function/method bodies are stripped (blank lines between routines and at
class/module level are kept, so the file shape survives). Legal boilerplate inside the module docstring is the one
thing the wall keeps: stripping it would both lose the license from the output and bypass the file-doc pass's
preservation path, which is exactly what the dogfood run should exercise. SCALE is then run over the walled copies so
its output can be judged against the hand-reviewed originals.

Two scratch directories are produced under `temp/dogfood/` and both are left in place for review:

    walled/     the stripped inputs, untouched by the run
    annotated/  the same files after SCALE has annotated them in place

Modes:
    --mode offline      one local-model run: definition docs + block comments + file descriptions. No network, no
                        stronger model - what a fully offline user gets.
    --mode online       every routine's comments deferred into a run manifest (`temp/dogfood/scale-manifest.json`)
                        for a stronger model to fill. The emit is model-free and completes in seconds; the harness
                        stops after it and prints the remaining loop commands; in Claude Code, the /scale skill
                        drives that loop end-to-end.

After the run (offline) or the emit phase (online) the harness verifies, deterministically, that re-walling each
annotated file reproduces the walled input's code lines byte-for-byte - SCALE's code-preservation guarantee checked
from the outside - and that every output still parses.

Usage:
    python tests/dogfood.py [--mode offline|online] [--files NAME ...] [--wall-only] [-- EXTRA...]

Anything after `--` is passed through to scale.py (e.g. `-- -m /path/model.gguf --n-ctx 12288`). The offline mode
loads a real GGUF and is SLOW on the full source set; use `--files` to dogfood a subset (e.g. `--files scale_text.py
scale_log.py`). The online emit loads no model at all.
"""
from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scale_filedoc import looks_legal  # noqa: E402  (the same legal-text veto the file-doc pass applies)

DOGFOOD = ROOT / "temp" / "dogfood"
WALLED = DOGFOOD / "walled"
ANNOTATED = DOGFOOD / "annotated"
MANIFEST = DOGFOOD / "scale-manifest.json"
FILEDOC = DOGFOOD / "scale-filedoc.json"


def _docstring_lines(tree: ast.AST, src_lines: list[str]) -> set[int]:
    """Return the line numbers of every module/class/function docstring that can be removed safely.

    A docstring is only reported when its owner has at least one further statement, so stripping it never leaves an
    empty (syntactically invalid) suite. A module docstring containing legal boilerplate (per the file-doc pass's
    `looks_legal` veto) is only stripped below the paragraph holding its last legal line: the license must survive
    the wall, both so the annotated output keeps it and so the run exercises the file-doc preservation path rather
    than starting bare.
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
                start, end = first.lineno, first.end_lineno
                if isinstance(node, ast.Module):
                    legal = [i for i in range(start, end + 1) if looks_legal(src_lines[i - 1])]
                    if legal:
                        # Keep the boilerplate through the end of the paragraph holding the last legal line
                        # (license continuation lines carry no marker themselves), plus the closing quotes;
                        # strip only the description below.
                        keep = legal[-1]
                        while keep + 1 < end and src_lines[keep].strip():
                            keep += 1
                        start = keep + 1
                        end = end - 1
                        if start > end:
                            continue
                lines.update(range(start, end + 1))
    return lines


def wall(src: str, strip_all_blanks: bool = False) -> str:
    """
    Strip `src` to a dogfood wall: no docstrings, no full-line comments, no blank lines inside routine bodies. The
    one exception is legal boilerplate in the module docstring, which is kept (see `_docstring_lines`).

    Blank lines *between* routines (and at class/module level) are kept so the file keeps its overall shape; pass
    `strip_all_blanks=True` to drop those too, which reduces a file to its code lines only (used for the
    code-preservation comparison). Blank/`#` lines inside non-docstring string literals are always protected, and a
    leading shebang is kept.
    """

    lines = src.split("\n")
    tree = ast.parse(src)
    doc_lines = _docstring_lines(tree, lines)

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
    ap.add_argument("--mode", choices=("offline", "online"), default="offline",
                    help="offline = full local run; online = defer every routine into a manifest (model-free)")
    ap.add_argument("--files", nargs="+", metavar="NAME", default=None,
                    help="subset of source file names to dogfood (default: every scale*.py)")
    ap.add_argument("--wall-only", action="store_true", help="prepare the walled/annotated dirs and stop")
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

    # Online: model-free emit only - the manifest needs a stronger model before the loop can finish.
    rc = run_scale(["-c", "--block-comments", "medium", "-l", "python", "-v", "--online",
                    "--emit-manifest", str(MANIFEST.relative_to(ROOT))] + extra + targets + out_flag)
    if rc != 0:
        print(f"\nSCALE exited with {rc}")
        return rc
    ok = verify(sources)
    run_scale(["--check-manifest", str(MANIFEST.relative_to(ROOT))])  # informational: how much was deferred

    m, fd, t = MANIFEST.relative_to(ROOT), FILEDOC.relative_to(ROOT), " ".join(targets + out_flag)
    print(f"\nEmit phase complete. Finish the loop with a stronger model filling the manifest answers, then:\n"
          f"  python scale.py --check-manifest {m}\n"
          f"  python scale.py -l python --apply-manifest {m} {t}\n"
          f"  python scale.py -l python --emit-filedoc {fd} {t}\n"
          f"  ...fill each file's range + description answer...\n"
          f"  python scale.py -l python --apply-filedoc {fd} {t}\n"
          f"In Claude Code, the /scale skill drives this whole loop.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
