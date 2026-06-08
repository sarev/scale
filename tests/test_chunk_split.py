#!/usr/bin/env python3
"""
Source chunk splitter (used by the map-reduce summary): chunks must cover every source line exactly once
(so they rejoin to the original) and each chunk must fit the token budget. Over-long single lines are
hard-split into budget-sized pieces.

Uses a trivial deterministic token estimate (1 "token" per character) so no GGUF model is required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale import _split_source  # noqa: E402

ESTIMATE = len  # treat each character as one token


def main():
    # A realistic-ish source with blank lines between blocks (no over-long lines).
    blocks = []
    for b in range(8):
        blocks.append(f"def func_{b}(x):")
        for i in range(6):
            blocks.append(f"    step_{i} = x + {i}")
        blocks.append("")  # blank line between blocks
    source = "\n".join(blocks)

    budget = 120
    chunks = _split_source(source, budget, ESTIMATE)

    assert len(chunks) > 1, "expected the source to be split into multiple chunks"
    assert "\n".join(chunks) == source, "chunks must rejoin exactly to the original source"
    for c in chunks:
        assert ESTIMATE(c) <= budget, f"chunk exceeds budget: {ESTIMATE(c)} > {budget}"

    # Over-long single line: must be hard-split into pieces that each fit the budget.
    giant = "x" * 1000
    pieces = _split_source(giant, 100, ESTIMATE)
    assert len(pieces) > 1, "an over-long line should be hard-split"
    for p in pieces:
        assert ESTIMATE(p) <= 100, "hard-split pieces must fit the budget"
    assert "".join(pieces) == giant, "hard-split pieces must reconstruct the original line"

    print("PASS: chunk splitter covers all lines, respects the budget, and hard-splits giant lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
