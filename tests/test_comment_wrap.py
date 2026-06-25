#!/usr/bin/env python3
"""
Opt-in line-length wrapping of inserted block comments (known-issue #3).

A one-line answer that looked short overflowed the project's column limit once indented inside a method, because the
writer cannot anticipate the final indentation - so the patcher enforces the budget. `render_comment_lines(width=N)`
re-wraps each logical line to fit N columns (indent + prefix included), continuing on fresh comment lines; `width=0`
(the default) leaves text untouched. Identifiers are never split, and a deep indent that leaves too little room
disables wrapping rather than shredding the text. No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_blocks import render_comment_lines, PYTHON_STYLE, SLASH_BLOCK_STYLE, _MIN_WRAP_WIDTH  # noqa: E402

LONG = ("Retry the request up to three times with exponential backoff before surfacing the timeout to the caller, "
        "because the upstream service is known to drop the first connection after an idle period.")


def main():
    indent = " " * 8   # a method body, two levels deep

    # ---- 1. width=0 is a no-op: one comment line, however long ----
    off = render_comment_lines(LONG, indent, PYTHON_STYLE, width=0)
    assert off == [f"{indent}# {LONG}"], "width=0 must leave the comment unwrapped"
    assert len(off) == 1 and len(off[0]) > 120, "the unwrapped line is the overflow the issue describes"

    # ---- 2. width=99 wraps to fit, every line within budget, as line comments ----
    wrapped = render_comment_lines(LONG, indent, PYTHON_STYLE, width=99)
    assert len(wrapped) > 1, "a long comment must wrap onto several lines"
    for ln in wrapped:
        assert len(ln) <= 99, f"wrapped line overruns the budget ({len(ln)}): {ln!r}"
        assert ln.startswith(f"{indent}# "), f"each wrapped line stays a comment at the right indent: {ln!r}"
    # No words lost or duplicated across the wrap.
    assert " ".join(l[len(indent) + 2:] for l in wrapped).split() == LONG.split(), "wrap must preserve the words"

    # ---- 3. identifiers are never split mid-token ----
    ident = "the_callback_registry_dispatch_table_lookup_helper handles this case"
    pieces = render_comment_lines(ident, "", PYTHON_STYLE, width=20)
    assert any("the_callback_registry_dispatch_table_lookup_helper" in p for p in pieces), \
        "a long identifier must survive whole even when it exceeds the budget"

    # ---- 4. a degenerate budget (deep indent, little room) disables wrapping ----
    deep = " " * 40
    avail = 44 - len(deep) - len(PYTHON_STYLE.line_prefix)   # well under _MIN_WRAP_WIDTH
    assert avail < _MIN_WRAP_WIDTH
    nowrap = render_comment_lines(LONG, deep, PYTHON_STYLE, width=44)
    assert nowrap == [f"{deep}# {LONG}"], "too little room must leave the comment unwrapped rather than shred it"

    # ---- 5. block-delimited style: a wrapped long line becomes a /* */ block ----
    cblock = render_comment_lines(LONG, "    ", SLASH_BLOCK_STYLE, width=60)
    assert cblock[0].strip() == "/*" and cblock[-1].strip() == "*/", "block style must wrap in /* */ delimiters"
    for ln in cblock[1:-1]:
        assert len(ln) <= 60, f"block continuation overruns the budget: {ln!r}"

    print("PASS: block comments wrap to the line-length budget when set, are untouched at width=0, never split idents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
