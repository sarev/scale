#!/usr/bin/env python3
"""
Strip blank lines and full-line `#` comments from a source file to produce a "wall of statements" (docstrings kept) -
a worst-case input for the within-function block pass, so we can see whether SCALE usefully re-paragraphs and comments
it. Blank/`#` lines that fall inside a string literal or docstring are protected (never stripped).

Usage:
    python tests/block_eval/make_wall.py <src.py> <out.py>

Note: this removes *all* blank lines, including the PEP 8 spacing between top-level defs, so functions will abut in the
output. That is cosmetic and only an artefact of the stripping - the block pass operates inside routine bodies and
never touches the gaps between top-level definitions.
"""
import ast
import io
import sys
from pathlib import Path


def strip_to_wall(src: str) -> str:
    """
    Return `src` with blank lines and full-line comments removed, except where they fall inside a string literal.

    Parameters:
    - `src`: The source text.

    Returns:
    - The stripped source text.
    """

    lines = src.split("\n")

    # Protect every line inside a string literal so docstring blanks / '#' lines survive.
    protected: set[int] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lo = getattr(node, "lineno", None)
            hi = getattr(node, "end_lineno", lo)
            if lo:
                protected.update(range(lo, hi + 1))

    kept = []
    for i, line in enumerate(lines, start=1):
        if i in protected:
            kept.append(line)
            continue
        s = line.strip()
        if s == "" or s.startswith("#"):
            continue
        kept.append(line)
    return "\n".join(kept)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    src = io.open(src_path, encoding="utf-8").read()
    wall = strip_to_wall(src)
    io.open(out_path, "w", encoding="utf-8", newline="\n").write(wall)
    print(f"stripped {len(src.splitlines())} -> {len(wall.splitlines())} lines  ({src_path} -> {out_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
