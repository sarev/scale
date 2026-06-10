#!/usr/bin/env python3
"""
The deterministic structural segmenter for C (the tree-sitter provider feeding `scale_blocks.structural_breaks`
with the brace-language flags: no in-body docstring, no nested-def rule). On a crafted function it must:

- break before a compound block of >= 3 source lines (before_compound), but not before a tiny inline one,
- break before a `return` whose previous statement is at the same depth (before_return),
- break when resuming after a substantial block closes (dedent),
- never break above the routine's first statement (first_in_scope is disabled for brace languages).

No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_c import iter_block_targets_c  # noqa: E402

SRC = (
    "int g(int a) {\n"        # 1
    "    int x = a;\n"        # 2  first statement - never gets a break above it
    "    if (x) x = 1;\n"     # 3  tiny inline block - no before_compound break
    "    if (x > 0) {\n"      # 4  opens a 4-line block (4-7) -> before_compound break
    "        x = 2;\n"        # 5
    "        x = 3;\n"        # 6
    "    }\n"                 # 7
    "    x = 4;\n"            # 8  resumes after the block -> dedent break
    "    return x;\n"         # 9  return after a same-depth statement -> before_return break
    "}\n"                     # 10
)


def main():
    lines = SRC.split("\n")
    t = iter_block_targets_c(SRC, lines)[0]

    starts = [s for s, _ in t.segments]
    assert starts == [4, 8, 9], f"unexpected segment starts: {starts} (segments {t.segments})"

    assert 2 not in starts, "the first body statement must never get a paragraph break above it"
    assert 3 not in starts, "a tiny inline block must not earn a before-compound break"
    assert t.segments[0] == (4, 7) and t.segments[-1][0] == 9, t.segments

    # Ranges are well-formed: sorted, non-overlapping, legal starts, clamped to the body.
    bl = set(t.boundary_lines)
    prev_end = -1
    for s, e in t.segments:
        assert s in bl and t.body_start <= s <= e <= t.body_end and s > prev_end, (s, e)
        prev_end = e

    print("PASS: C structural segmenter fires before-compound/return/dedent and skips first-in-scope and tiny blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
