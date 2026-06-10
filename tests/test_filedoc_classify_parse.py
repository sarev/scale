#!/usr/bin/env python3
"""
`_parse_classify_range` turns the classify turn's reply into a 1-based inclusive range over the eligible header
entries. It must accept `START-END`, a single number, and NONE/empty, clamp to `[1, n]`, and reject an inverted or
out-of-range answer (so a confused model can never point the patcher outside the header).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_filedoc import _parse_classify_range  # noqa: E402


def main():
    n = 5
    assert _parse_classify_range("2-4", n) == (2, 4)
    assert _parse_classify_range("  3 - 5 ", n) == (3, 5)
    assert _parse_classify_range("range: 1 to 2", n) == (1, 2)
    assert _parse_classify_range("3", n) == (3, 3)            # single line
    assert _parse_classify_range("NONE", n) is None
    assert _parse_classify_range("none.", n) is None
    assert _parse_classify_range("", n) is None

    # Clamping and rejection.
    assert _parse_classify_range("4-99", n) == (4, 5), "end clamps to n"
    assert _parse_classify_range("0-2", n) == (1, 2), "start clamps to 1"
    assert _parse_classify_range("5-2", n) is None, "inverted range rejected"
    assert _parse_classify_range("3", 0) is None, "no entries -> nothing usable"

    print("PASS: _parse_classify_range parses, clamps, and rejects header description ranges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
