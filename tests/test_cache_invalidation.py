#!/usr/bin/env python3
"""
SummaryCache must reuse a summary on identical content and invalidate it when
the source changes (keyed on a content hash, not just the path).

No GGUF model required.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale import SummaryCache  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        SummaryCache._CACHE_DIR = Path(tmp)
        SummaryCache._CACHE_INDEX = Path(tmp) / "index.pkl"
        p = Path("virtual_source.py")

        c1 = SummaryCache(p, "print('A')")
        assert c1.summary is None, "expected no summary on first sight"
        c1.summary = "SUMMARY FOR A"

        assert SummaryCache(p, "print('A')").summary == "SUMMARY FOR A", "identical content should reuse"
        assert SummaryCache(p, "print('B')").summary is None, "changed content must invalidate"
        assert SummaryCache(p, "print('A')").summary == "SUMMARY FOR A", "reverted content should be valid again"

    print("PASS: cache reuses on identical content and invalidates on change")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
