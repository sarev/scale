#!/usr/bin/env python3
"""
Snippet elision: an oversized routine must be reduced to fit a token budget while always preserving its
signature/header, and reporting how many body lines were omitted.

Uses a trivial deterministic token estimate (1 "token" per character) so no GGUF model is required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_text import elide_to_budget  # noqa: E402

ESTIMATE = len  # treat each character as one token


def main():
    header = ["def big(a, b):", "    # signature/header kept verbatim"]
    body = [f"    line_{i} = {i}" for i in range(60)]
    snippet = header + body
    header_count = len(header)
    marker = "# ... {n} lines omitted ..."

    # 1. Comfortable budget: returned unchanged.
    lines, omitted = elide_to_budget(snippet, header_count, 10_000, ESTIMATE, marker)
    assert omitted == 0 and lines == snippet, "should not elide when it already fits"

    # 2. Tight budget: must elide the middle, keep header + a marker + some head/tail.
    budget = ESTIMATE("\n".join(header)) + 200
    lines, omitted = elide_to_budget(snippet, header_count, budget, ESTIMATE, marker)
    assert lines[:header_count] == header, "header must be preserved verbatim"
    assert omitted > 0, "expected some body lines to be omitted"
    marker_lines = [ln for ln in lines if "lines omitted" in ln]
    assert len(marker_lines) == 1, "exactly one elision marker expected"
    assert f"{omitted} lines omitted" in marker_lines[0], "marker must report the omitted count"
    assert ESTIMATE("\n".join(lines)) <= budget, "elided snippet must fit the budget"
    # Some head and some tail of the body should survive (marker is not adjacent to header on both sides).
    assert lines[header_count] == body[0], "first body line (head) should be kept"
    assert lines[-1] == body[-1], "last body line (tail) should be kept"

    # 3. Budget that only fits the header: keep header + a single marker for the whole body.
    tiny = ESTIMATE("\n".join(header))
    lines, omitted = elide_to_budget(snippet, header_count, tiny, ESTIMATE, marker)
    assert lines[:header_count] == header
    assert omitted == len(body), "whole body should be reported omitted"
    assert lines[header_count:] == [marker.format(n=len(body))]

    print("PASS: elision preserves the header, fits the budget, and reports omissions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
