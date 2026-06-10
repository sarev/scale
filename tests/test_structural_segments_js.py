#!/usr/bin/env python3
"""
The deterministic structural segmenter for JavaScript (the tree-sitter provider feeding
`scale_blocks.structural_breaks` with the brace-language flags - no in-body docstring - but, unlike C, WITH the
nested-def rule, since JS has closures). On a crafted function it must:

- break before a substantial nested definition (before_compound on the def's span),
- break on the statement that follows a nested definition (after_def - the JS-specific rule),
- break before a `return` whose previous statement is at the same depth (before_return),
- never break above the routine's first statement (first_in_scope is disabled for brace languages).

No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_javascript import iter_block_targets_js  # noqa: E402

SRC = (
    "function h(a) {\n"        # 1
    "  let x = a;\n"           # 2  first statement - never gets a break above it
    "  function g() {\n"       # 3  nested def opening a 4-line block (3-6) -> before_compound break
    "    let t = 1;\n"         # 4
    "    return t;\n"          # 5
    "  }\n"                    # 6
    "  let y = g();\n"         # 7  statement after a nested def -> after_def break
    "  return y;\n"            # 8  return after a same-depth statement -> before_return break
    "}\n"                      # 9
)


def main():
    lines = SRC.split("\n")
    h = {t.qualname: t for t in iter_block_targets_js(SRC, lines)}["h"]

    starts = [s for s, _ in h.segments]
    assert starts == [3, 7, 8], f"unexpected segment starts: {starts} (segments {h.segments})"

    assert 2 not in starts, "the first body statement must never get a paragraph break above it"
    assert 7 in starts, "the statement after a nested definition must start a new paragraph (after_def)"

    # Ranges are well-formed: sorted, non-overlapping, legal starts, clamped to the body.
    bl = set(h.boundary_lines)
    prev_end = -1
    for s, e in h.segments:
        assert s in bl and h.body_start <= s <= e <= h.body_end and s > prev_end, (s, e)
        prev_end = e

    print("PASS: JS structural segmenter fires before-compound/after-def/before-return and skips first-in-scope")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
