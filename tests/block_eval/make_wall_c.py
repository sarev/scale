#!/usr/bin/env python3
"""
C counterpart of `make_wall.py`: strip a C source file down to a "wall" so we can see whether SCALE's block pass
usefully re-paragraphs and (re)comments it. Removes **all** comments (`//` and `/* */`, found via tree-sitter so
comment-like text inside string literals is never touched) and every blank line that falls *inside a function body*.
Blank lines outside functions (the file header, the gaps between top-level definitions) are kept, so a diff against
the original stays focused on what the block pass actually changes - the paragraphing and comments inside bodies.

Usage:
    python tests/block_eval/make_wall_c.py <src.c> <out.c>
"""
import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scale_c import _parse_c, iter_defs_with_info_c, _row_of, _to_1based  # noqa: E402


def _comment_spans(tree):
    """Return the (start_byte, end_byte) of every comment node in the tree, sorted."""
    spans = []
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.type == "comment":
            spans.append((n.start_byte, n.end_byte))
        for i in range(n.named_child_count):
            stack.append(n.named_child(i))
    return sorted(spans)


def strip_to_wall(src: str) -> str:
    """
    Return `src` with all comments removed and all blank lines inside function bodies removed.

    Parameters:
    - `src`: The C source text.

    Returns:
    - The stripped wall text (normalised to `\\n` line endings).
    """

    norm = src.replace("\r\n", "\n").replace("\r", "\n")

    # 1. Remove every comment by splicing out its byte span (string literals are unaffected - they are not comments).
    tree, src_bytes = _parse_c(norm)
    out = bytearray()
    prev = 0
    for s, e in _comment_spans(tree):
        out += src_bytes[prev:s]
        prev = e
    out += src_bytes[prev:]
    stripped = out.decode("utf-8", errors="replace")

    # 2. Re-parse and drop blank lines strictly inside any function body (between the header and the closing brace).
    tree2, src_bytes2 = _parse_c(stripped)
    inside: set = set()
    for info in iter_defs_with_info_c(tree2, src_bytes2):
        body = info.node.child_by_field_name("body")
        if body is None or body.type != "compound_statement":
            continue
        first = _to_1based(_row_of(body.start_point))   # the `{` line
        last = _to_1based(_row_of(body.end_point))      # the `}` line
        inside.update(range(first + 1, last))           # strictly between the braces

    kept = []
    for i, line in enumerate(stripped.split("\n"), start=1):
        if i in inside and line.strip() == "":
            continue                                    # drop an intra-body blank line
        kept.append(line.rstrip())                      # trim trailing whitespace left by removed trailing comments
    return "\n".join(kept)


def main() -> int:
    ap = argparse.ArgumentParser(description="Strip a C source file to a wall (no comments, no intra-body blanks).")
    ap.add_argument("src", type=Path, help="source file to strip")
    ap.add_argument("out", type=Path, help="destination for the stripped wall")
    args = ap.parse_args()

    src = io.open(args.src, encoding="utf-8").read()
    wall = strip_to_wall(src)
    io.open(args.out, "w", encoding="utf-8", newline="\n").write(wall)
    print(f"stripped comments + intra-body blanks: {len(src.splitlines())} -> {len(wall.splitlines())} lines  "
          f"({args.src} -> {args.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
