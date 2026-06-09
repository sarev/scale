#!/usr/bin/env python3
"""
Within-function block insertion patcher: edits must be insertion-only (blank/comment lines), so executable code is
preserved byte-for-byte. This guards that the patcher:

- inserts a blank line plus a comment above a chosen boundary,
- replaces an existing same-indent comment block above a boundary,
- paragraphs every chunk with a blank line (even when the comment is NONE), and on NONE preserves any existing comment,
- preserves the code signature (the safety guard passes for valid edits), and
- abandons a routine and keeps the original when a (forced) edit would alter code.

No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_blocks  # noqa: E402
from scale_blocks import BlockTarget, PYTHON_STYLE, apply_blocks, code_preserved  # noqa: E402


def _target(boundary_lines, body_end):
    """Build a single-function BlockTarget with 4-space indentation on every boundary."""
    return BlockTarget(
        qualname="f",
        kind="def",
        header_start=1,
        header_end=1,
        body_start=2,
        body_end=body_end,
        boundary_lines=tuple(boundary_lines),
        indent_of={ln: "    " for ln in boundary_lines},
        depth=0,
    )


def _code(lines):
    return [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def main():
    # ---- 1. Insert a blank + comment above a chosen boundary ----
    src = ["def f():", "    a = 1", "    b = 2", "    c = 3"]
    target = _target([2, 3, 4], body_end=4)
    out = apply_blocks(src, target, [(3, "second step")], PYTHON_STYLE)

    assert "    # second step" in out, "comment should be inserted at the boundary indent"
    ci = out.index("    # second step")
    assert out[ci - 1] == "", "a blank line must separate the block from what precedes it"
    assert out[ci + 1] == "    b = 2", "the comment sits directly above its statement"
    assert _code(out) == _code(src), "code lines must be unchanged"

    # ---- 2. Replace an existing same-indent comment ----
    src2 = ["def f():", "    a = 1", "    # old comment", "    b = 2", "    c = 3"]
    target2 = _target([4], body_end=5)  # b = 2 is line 4
    out2 = apply_blocks(src2, target2, [(4, "new comment")], PYTHON_STYLE)
    assert "    # new comment" in out2, "comment should be rewritten"
    assert "    # old comment" not in out2, "the previous comment must be replaced, not duplicated"
    assert _code(out2) == _code(src2), "code lines must be unchanged"

    # ---- 3. NONE preserves an existing comment (never deletes) while still paragraphing the chunk ----
    out3 = apply_blocks(src2, target2, [(4, None)], PYTHON_STYLE)
    assert "    # old comment" in out3, "NONE must NOT delete an existing comment"
    ci3 = out3.index("    # old comment")
    assert out3[ci3 - 1] == "", "a blank line should separate the kept comment from what precedes it"
    assert out3[ci3 + 1] == "    b = 2", "the kept comment stays directly above its statement"
    assert _code(out3) == _code(src2), "code lines must be unchanged"

    # ---- 3b. NONE with no existing comment still paragraphs the chunk (blank only, no invented comment) ----
    out3b = apply_blocks(src, target, [(3, None)], PYTHON_STYLE)
    bi = out3b.index("    b = 2")
    assert out3b[bi - 1] == "", "a blank line should paragraph the chunk"
    assert not any(ln.lstrip().startswith("#") for ln in out3b), "NONE must not invent a comment"
    assert _code(out3b) == _code(src), "code lines must be unchanged"

    # ---- 4. Multiple boundaries apply correctly (reverse-order stability) ----
    out4 = apply_blocks(src, target, [(2, "first"), (4, "third")], PYTHON_STYLE)
    assert "    # first" in out4 and "    # third" in out4, "all chosen boundaries get comments"
    assert _code(out4) == _code(src), "code lines must be unchanged across multiple edits"

    # ---- 5. Safety guard: code-altering edit is rejected, original kept ----
    assert code_preserved(src, out, PYTHON_STYLE), "valid edit must pass the guard"
    assert not code_preserved(src, ["def f():", "    a = 1"], PYTHON_STYLE), \
        "dropping a code line must fail the guard"

    # Force the low-level patcher to corrupt code and confirm apply_blocks keeps the original.
    original_apply = scale_blocks._apply_edits
    try:
        scale_blocks._apply_edits = lambda *_a, **_k: ["def f():", "    a = 1"]  # drops b, c
        kept = apply_blocks(src, target, [(3, "x")], PYTHON_STYLE)
        assert kept == src, "a code-altering edit must abort the routine and keep the original"
    finally:
        scale_blocks._apply_edits = original_apply

    print("PASS: block insertion is insertion-only, replaces/drops comments, and the guard protects code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
