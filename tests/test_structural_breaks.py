#!/usr/bin/env python3
"""
The shared structural-paragraph rule engine (`scale_blocks.structural_breaks`), tested in isolation from any
language or parser. It operates purely on normalised `SegStatement` records, so hand-built records can pin each
rule precisely:

- `first_in_scope` only breaks when the body has a docstring AND that rule is enabled (off for brace languages);
- `after_def` breaks on the statement following a nested def, and is gated by `allow_after_def` (off for C);
- `before_return` breaks before a `return` whose previous statement is at the same depth;
- `before_compound` breaks before a block of >= min_block_lines, and not before a smaller one;
- `dedent` breaks when resuming after a substantial (or unidentified) closed block, but not a small one;
- a break is never placed at a line that is not a legal boundary;
- the returned ranges are sorted, non-overlapping, and clamped to body_end.

No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_blocks import SegStatement, structural_breaks  # noqa: E402


def S(start, depth, *, end=None, is_return=False, is_def=False, opens_block=0,
      first_in_scope=False, closed_block=0, merge_anchor=None):
    """Build a SegStatement with test-friendly defaults (end defaults to start)."""
    return SegStatement(
        start=start, end=end if end is not None else start, depth=depth,
        is_return=is_return, is_def=is_def, opens_block=opens_block,
        first_in_scope=first_in_scope, closed_block=closed_block, merge_anchor=merge_anchor,
    )


def _starts(ranges):
    return [a for a, _ in ranges]


def test_compound_return_dedent_and_ranges():
    # 2 first stmt; 3 simple; 4 opens a 4-line if (4-7); 6 inside it; 8 resumes (dedent); 9 trailing return.
    stmts = [
        S(2, 0, first_in_scope=True),
        S(3, 0),
        S(4, 0, end=7, opens_block=4),
        S(6, 1),
        S(8, 0, closed_block=4),
        S(9, 0, is_return=True),
    ]
    boundary = (2, 3, 4, 6, 8, 9)

    # No docstring: the opening paragraph (line 2) is its own chunk; compound, dedent and return break the rest.
    ranges = structural_breaks(stmts, has_doc=False, boundary_lines=boundary, body_end=9)
    assert ranges == [(2, 3), (4, 7), (8, 8), (9, 9)], ranges

    # With a docstring (and the rule enabled), the first statement is paragraphed off too (line 2 already a start).
    ranges_doc = structural_breaks(stmts, has_doc=True, boundary_lines=boundary, body_end=9)
    assert _starts(ranges_doc) == [2, 4, 8, 9], ranges_doc

    # Brace-language config: has_doc is irrelevant; the opening chunk is still emitted (line 2).
    ranges_brace = structural_breaks(stmts, has_doc=True, boundary_lines=boundary, body_end=9,
                                     allow_first_in_scope=False)
    assert _starts(ranges_brace) == [2, 4, 8, 9], ranges_brace


def test_after_def_gating_small_block_and_boundary_gate():
    # 2 nested def (first stmt); 5 resumes after it; 6 opens a tiny 2-line block; 7 a return but NOT a boundary.
    stmts = [
        S(2, 0, end=4, is_def=True, opens_block=3, first_in_scope=True),
        S(5, 0),
        S(6, 0, end=7, opens_block=2),
        S(7, 0, is_return=True),
    ]
    boundary = (2, 5, 6)  # note: 7 is deliberately not a legal boundary

    # after_def on: the opening chunk (line 2) plus the statement after the def; the tiny block / non-boundary
    # return do not break.
    on = structural_breaks(stmts, has_doc=False, boundary_lines=boundary, body_end=8)
    assert _starts(on) == [2, 5], on

    # after_def off (C): only the opening chunk remains (no rule fires on the body).
    off = structural_breaks(stmts, has_doc=False, boundary_lines=boundary, body_end=8,
                            allow_after_def=False)
    assert _starts(off) == [2], off


def test_dedent_unknown_block_breaks_small_block_does_not():
    stmts = [
        S(2, 0, first_in_scope=True),
        S(3, 2),
        S(4, 0, closed_block=0),   # resume, no block identified -> still a break
        S(5, 2),
        S(6, 0, closed_block=2),   # resume after a 2-line block -> too small, no break
    ]
    boundary = (2, 3, 4, 5, 6)
    ranges = structural_breaks(stmts, has_doc=False, boundary_lines=boundary, body_end=6)
    assert _starts(ranges) == [2, 4], ranges  # opening chunk (2) + the dedent resume (4)


def test_merge_anchor_redirects_a_trailing_return():
    # A `[stmt, return]` suite inside an if: the return carries merge_anchor=5, so the break lands on the anchor
    # (line 5, the preceding statement) and NOT on the return (line 6) - the two share one paragraph.
    stmts = [
        S(2, 0, first_in_scope=True),
        S(3, 0, end=7, opens_block=4),               # an `if` opening a 5-line block (before_compound)
        S(5, 1),                                     # the suite's first statement (the anchor)
        S(6, 1, is_return=True, merge_anchor=5),     # return -> merge up to line 5
    ]
    boundary = (2, 3, 5, 6)
    starts = _starts(structural_breaks(stmts, has_doc=False, boundary_lines=boundary, body_end=7))
    assert 5 in starts, f"the merge anchor must start a paragraph: {starts}"
    assert 6 not in starts, f"a merged return must NOT get its own break: {starts}"
    assert starts == [2, 3, 5], starts  # opening chunk (2) + the `if` (3) + the merge anchor (5)


def test_opening_paragraph_is_always_a_chunk():
    # Even when no rule fires, the opening paragraph (first body statement -> first break) is emitted as a chunk,
    # so it is summarised/scored rather than silently dropped. Here nothing breaks, so the whole body is one chunk.
    stmts = [S(2, 0, first_in_scope=True), S(3, 0), S(4, 0)]
    ranges = structural_breaks(stmts, has_doc=False, boundary_lines=(2, 3, 4), body_end=4)
    assert ranges == [(2, 4)], ranges
    # But it is only emitted when the first statement is a legal boundary (e.g. not a multi-statement line).
    ranges2 = structural_breaks(stmts, has_doc=False, boundary_lines=(3, 4), body_end=4)
    assert ranges2 == [], ranges2


def main():
    test_compound_return_dedent_and_ranges()
    test_after_def_gating_small_block_and_boundary_gate()
    test_dedent_unknown_block_breaks_small_block_does_not()
    test_merge_anchor_redirects_a_trailing_return()
    test_opening_paragraph_is_always_a_chunk()
    print("PASS: structural_breaks fires each paragraph rule correctly and respects boundary/range constraints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
