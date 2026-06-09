#!/usr/bin/env python3
"""
Boundary-pass numbered view + reply sanitiser:

- only legal boundary lines carry a number; an over-long run of non-boundary lines collapses to an elision band that
  carries no number (so the model cannot pick a line inside it), while a short run is shown verbatim;
- the segment parser keeps only ranges whose start is a legal boundary, clamps ends within the body and to the next
  chunk's start, and falls back to bare numbers (as starts) when the model gives no ranges.

No GGUF model required: only the pure rendering / parsing helpers run.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_blocks import (  # noqa: E402
    BlockTarget,
    render_numbered_body,
    _parse_segments,
    MAX_CONTEXT_RUN,
)


def _target(source_lines, boundary_lines):
    return BlockTarget(
        qualname="f",
        kind="def",
        header_start=1,
        header_end=1,
        body_start=2,
        body_end=len(source_lines),
        boundary_lines=tuple(boundary_lines),
        indent_of={ln: "    " for ln in boundary_lines},
        depth=0,
    )


def main():
    # A long (> MAX_CONTEXT_RUN) run of non-boundary lines between two boundaries must collapse to a band.
    long_run = MAX_CONTEXT_RUN + 2
    lines = ["def f():", "    x = start()"]
    lines += [f"    filler_{i}()" for i in range(long_run)]
    lines += ["    return x"]
    boundary = [2, len(lines)]  # first statement and the return
    view = render_numbered_body(lines, _target(lines, boundary))

    assert f"« {long_run} lines elided »" in view, "an over-long non-boundary run must become an elision band"
    assert "filler_0()" not in view, "elided lines must not appear in the view"

    # Boundary lines carry their number in the gutter; non-boundary content does not.
    assert re.search(r"^\s*2\|\s+x = start\(\)$", view, flags=re.M), "boundary line should be numbered"
    band_line = next(ln for ln in view.split("\n") if "elided" in ln)
    assert not re.search(r"\d", band_line.split("|")[0]), "the elision band must carry no line number"

    # A short run (<= MAX_CONTEXT_RUN) is shown verbatim rather than elided.
    short = ["def g():", "    x = start()", "    a()", "    b()", "    return x"]
    sview = render_numbered_body(short, _target(short, [2, len(short)]))
    assert "elided" not in sview, "a short non-boundary run should not be elided"
    assert "a()" in sview and "b()" in sview, "short-run lines should be shown verbatim"

    # ---- Segment parser ----
    allowed = (2, 5, 10)
    body_end = 20

    # Ranges: legal starts kept, ends clamped within body and to the next chunk's start.
    assert _parse_segments("2-4, 5-9, 10-15", allowed, body_end) == [(2, 4), (5, 9), (10, 15)], \
        "must parse start-end ranges with legal starts"
    assert _parse_segments("3-7, 10-12", allowed, body_end) == [(10, 12)], \
        "a range whose start is not a legal boundary must be dropped"
    assert _parse_segments("2-99, 10-12", allowed, body_end) == [(2, 9), (10, 12)], \
        "an over-long end must be clamped so chunks do not overlap"
    assert _parse_segments("10-12, 2-4", allowed, body_end) == [(2, 4), (10, 12)], \
        "ranges must come back sorted by start"

    # Fallback: bare numbers (no ranges) become starts, ends inferred from the next start / body end.
    assert _parse_segments("2, 5", allowed, body_end) == [(2, 4), (5, body_end)], \
        "bare numbers fall back to starts with inferred ends"
    assert _parse_segments("no numbers here", allowed, body_end) == [], "no numbers -> no chunks"
    assert _parse_segments("100 200", allowed, body_end) == [], "out-of-set numbers -> no chunks"

    print("PASS: numbered view elides over-long runs without numbers; segment parser keeps legal, clamped ranges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
