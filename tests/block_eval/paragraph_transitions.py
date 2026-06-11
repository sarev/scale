#!/usr/bin/env python3
"""
Transition / confidence model over anonymised statement shapes (companion to `anon_paragraphs.py`).

Where `anon_paragraphs.py` tallies the *opener* shape of each comment-led paragraph, this looks at every adjacent pair
of code lines and asks: after a line of shape S, did the author continue the same paragraph, or place a break (a
comment or blank line, or a dedent out of the block)? Aggregated per shape, that yields:

- `end%`  - how often a line of shape S is the LAST line of its paragraph (a high value means S is a natural closer).
- and, read the other way via `anon_paragraphs.py`, which shapes are natural openers.

A paragraph boundary between adjacent lines A -> B is then predictable when A is a high-`end%` shape or B is a high-
open% shape, and each *continuation* that matches the common "stay together" pairs boosts confidence we are still in
one paragraph. This is the empirical basis for the deterministic segmenter's break rules and for attacking the
simple->simple mid-run breaks that pure structure cannot see.

Usage:
    ../.llm-venv/Scripts/python.exe tests/block_eval/paragraph_transitions.py [PATH ...] [--min-count N] [--top N]
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anon_paragraphs import _anonymise_by_row, ROOT, _iter_paths  # noqa: E402


def _classify_transitions(src: str) -> List[Tuple[str, str]]:
    """
    Return one `(shape, outcome)` pair per code line, describing what follows it.

    `outcome` is one of: `continue` (the next code line is at the same-or-deeper indent with nothing between),
    `break-marked` (a comment and/or blank line intervenes before the next code line), `break-dedent` (the next code
    line is indented less - the block closed), or `end` (no further code line). The shape is the line's anonymised
    token string.
    """
    lines = src.split("\n")
    by_row = _anonymise_by_row(src)

    def is_code(idx: int) -> bool:
        s = lines[idx].strip()
        return s != "" and not s.startswith("#")

    code_idx = [i for i in range(len(lines)) if is_code(i)]
    out: List[Tuple[str, str]] = []

    for pos, i in enumerate(code_idx):
        shape = " ".join(by_row.get(i + 1, []))
        if pos + 1 >= len(code_idx):
            out.append((shape, "end"))
            continue
        j = code_idx[pos + 1]

        indent_i = len(lines[i]) - len(lines[i].lstrip())
        indent_j = len(lines[j]) - len(lines[j].lstrip())
        saw_marker = any(lines[k].strip() == "" or lines[k].lstrip().startswith("#") for k in range(i + 1, j))

        if indent_j < indent_i:
            outcome = "break-dedent"
        elif saw_marker:
            outcome = "break-marked"
        else:
            outcome = "continue"
        out.append((shape, outcome))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-shape paragraph-break rates from anonymised statement transitions.")
    ap.add_argument("paths", nargs="*", type=Path, help="source files/dirs (default: repo scale*.py)")
    ap.add_argument("--min-count", type=int, default=4, help="only show shapes seen at least this many times")
    ap.add_argument("--top", type=int, default=30, help="rows to show")
    args = ap.parse_args()

    files = _iter_paths(args.paths) if args.paths else sorted(ROOT.glob("scale*.py"))

    cont: Counter = Counter()
    brk_marked: Counter = Counter()
    brk_dedent: Counter = Counter()
    total: Counter = Counter()
    for f in files:
        for shape, outcome in _classify_transitions(f.read_text(encoding="utf-8")):
            total[shape] += 1
            if outcome == "continue":
                cont[shape] += 1
            elif outcome == "break-marked":
                brk_marked[shape] += 1
            elif outcome == "break-dedent":
                brk_dedent[shape] += 1
            # 'end' counts toward total only

    rows = []
    for shape, n in total.items():
        if n < args.min_count:
            continue
        breaks = brk_marked[shape] + brk_dedent[shape]
        decided = cont[shape] + breaks                # exclude 'end' (no following code)
        if decided == 0:
            continue
        rows.append((breaks / decided, n, cont[shape], brk_marked[shape], brk_dedent[shape], shape))

    rows.sort(key=lambda r: (-r[0], -r[1]))
    print(f"# shapes seen >= {args.min_count} times, by P(a paragraph break follows this shape)\n")
    print(f"{'break%':>7} {'n':>4} {'cont':>5} {'mark':>5} {'dedent':>6}  shape")
    for rate, n, c, m, d, shape in rows[: args.top]:
        print(f"{rate*100:6.0f}% {n:4d} {c:5d} {m:5d} {d:6d}  {shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
