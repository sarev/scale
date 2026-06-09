#!/usr/bin/env python3
"""
Strip a source file down to a "wall of statements" - a worst-case input for the block/definition passes, so we can see
whether SCALE usefully re-paragraphs and (re)comments it. Always removes blank lines and full-line `#` comments; with
`--strip-docstrings` it also removes module/class/function docstrings, leaving a file with **no comments at all** apart
from a leading shebang. Blank/`#` lines inside a *non-docstring* string literal are protected (never stripped), and a
docstring is only removed when its routine has other statements (so the body never becomes empty / invalid).

Usage:
    python tests/block_eval/make_wall.py [--strip-docstrings] <src.py> <out.py>

Note: this removes *all* blank lines, including the PEP 8 spacing between top-level defs, so functions abut in the
output. That is cosmetic and only an artefact of the stripping - the block pass operates inside routine bodies and
never touches the gaps between top-level definitions. A leading `#!` shebang is always kept.
"""
import argparse
import ast
import io
import sys
from pathlib import Path


def _docstring_line_range(node: ast.AST):
    """Return the inclusive line range of `node`'s own docstring, or None if it has none / it is its only statement.

    A docstring is the first statement when it is a bare string constant. It is only reported for removal when the body
    holds at least one further statement, so stripping it can never leave an empty (syntactically invalid) suite.
    """
    body = getattr(node, "body", None)
    if not body or len(body) < 2:
        return None
    first = body[0]
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return getattr(first, "lineno", None), getattr(first, "end_lineno", getattr(first, "lineno", None))
    return None


def strip_to_wall(src: str, strip_docstrings: bool = False) -> str:
    """
    Return `src` with blank lines and full-line comments removed (and, optionally, docstrings too).

    Parameters:
    - `src`: The source text.
    - `strip_docstrings`: Also remove module/class/function docstrings (leaving no comments bar a leading shebang).

    Returns:
    - The stripped source text.
    """

    lines = src.split("\n")
    tree = ast.parse(src)

    # Lines that belong to docstrings we intend to delete (so they are neither kept nor protected).
    docstring_lines: set[int] = set()
    if strip_docstrings:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                rng = _docstring_line_range(node)
                if rng and rng[0]:
                    docstring_lines.update(range(rng[0], rng[1] + 1))

    # Protect every line inside a string literal so blank/`#` lines within real data strings (and any kept docstrings)
    # survive the strip. Docstrings being removed are excluded from this protection.
    protected: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lo = getattr(node, "lineno", None)
            hi = getattr(node, "end_lineno", lo)
            if lo:
                protected.update(range(lo, hi + 1))
    protected -= docstring_lines

    kept = []
    for i, line in enumerate(lines, start=1):
        if i == 1 and line.startswith("#!"):
            kept.append(line)                 # keep a leading shebang
            continue
        if i in docstring_lines:
            continue                          # drop docstrings when requested
        if i in protected:
            kept.append(line)
            continue
        s = line.strip()
        if s == "" or s.startswith("#"):
            continue                          # drop blank lines and full-line comments
        kept.append(line)
    return "\n".join(kept)


def main() -> int:
    ap = argparse.ArgumentParser(description="Strip a source file to a wall of statements.")
    ap.add_argument("src", type=Path, help="source file to strip")
    ap.add_argument("out", type=Path, help="destination for the stripped wall")
    ap.add_argument("--strip-docstrings", action="store_true",
                    help="also remove module/class/function docstrings (no comments left bar a shebang)")
    args = ap.parse_args()

    src = io.open(args.src, encoding="utf-8").read()
    wall = strip_to_wall(src, strip_docstrings=args.strip_docstrings)
    io.open(args.out, "w", encoding="utf-8", newline="\n").write(wall)
    what = "blanks + comments + docstrings" if args.strip_docstrings else "blanks + comments"
    print(f"stripped {what}: {len(src.splitlines())} -> {len(wall.splitlines())} lines  ({args.src} -> {args.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
