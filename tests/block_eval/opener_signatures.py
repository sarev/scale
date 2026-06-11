#!/usr/bin/env python3
"""
Opener-signature model: P(a line begins a new paragraph | the anonymised shape of it and the lines that follow).

This is the forward-looking face of the corpus (companion to `paragraph_transitions.py`, which measures the backward
"break after this shape" face - the two are complementary evidence for the same boundary and are meant to be combined,
not chosen between).

Every line in the sources is labelled an *opener* (a paragraph boundary sits immediately before it: a comment or blank
line, an indent change, or the start of scope) or a *continuation* (same-indent code directly above, nothing between).
For each anonymised n-gram *starting* at a line - the line's own shape (n=1), plus the next one (n=2) or two (n=3) code
lines - we report the fraction of its occurrences that are openers. As n grows the signature gets more specific, so a
recurring multi-line shape sharpens the opener probability of its first line: that is the "three lines boost our
confidence the first line started a new paragraph" effect, made quantitative.

Features come from code lines only (as the segmenter will see a comment-stripped body); the opener label comes from the
human paragraphing in the original source.

Usage:
    ../.llm-venv/Scripts/python.exe tests/block_eval/opener_signatures.py [PATH ...] [--min-count N] [--top N]
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anon_paragraphs import _anonymise_by_row, ROOT, _iter_paths  # noqa: E402


def _labelled_code_lines(src: str) -> List[Tuple[str, bool]]:
    """
    Return `(shape, is_opener)` for every code line in `src`, in source order.

    A line is an opener when a paragraph boundary sits immediately before it: it is the first code line, a comment or
    blank line precedes it, or its indent differs from the previous code line (a dedent out of a block, or the first
    line of a deeper suite). Otherwise it is a continuation of the current paragraph.
    """
    lines = src.split("\n")
    by_row = _anonymise_by_row(src)
    code_idx = [i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#")]

    recs: List[Tuple[str, bool]] = []
    for pos, i in enumerate(code_idx):
        shape = " ".join(by_row.get(i + 1, []))
        if pos == 0:
            opener = True
        else:
            p = code_idx[pos - 1]
            indent = len(lines[i]) - len(lines[i].lstrip())
            prev_indent = len(lines[p]) - len(lines[p].lstrip())
            marker = any(lines[k].strip() == "" or lines[k].lstrip().startswith("#") for k in range(p + 1, i))
            opener = marker or indent != prev_indent
        recs.append((shape, opener))
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description="P(line opens a paragraph | its forward n-line anonymised shape).")
    ap.add_argument("paths", nargs="*", type=Path, help="source files/dirs (default: repo scale*.py)")
    ap.add_argument("--min-count", type=int, default=4, help="only show n-grams seen at least this many times")
    ap.add_argument("--top", type=int, default=25, help="rows per n")
    args = ap.parse_args()

    files = _iter_paths(args.paths) if args.paths else sorted(ROOT.glob("scale*.py"))
    recs: List[Tuple[str, bool]] = []
    for f in files:
        recs.extend(_labelled_code_lines(f.read_text(encoding="utf-8")))

    base = sum(1 for _, o in recs if o) / max(1, len(recs))
    print(f"# {len(recs)} code lines; base opener rate = {base*100:.0f}%  (lift = opener% / base)\n")

    for n in (1, 2, 3):
        total: Counter = Counter()
        opens: Counter = Counter()
        for s in range(len(recs) - n + 1):
            gram = " : ".join(recs[s + k][0] for k in range(n))
            total[gram] += 1
            if recs[s][1]:
                opens[gram] += 1

        rows = [(opens[g] / total[g], total[g], g) for g in total if total[g] >= args.min_count]
        rows.sort(key=lambda r: (-r[0], -r[1]))
        print(f"## n={n}  (>= {args.min_count} occurrences)")
        print(f"{'open%':>6} {'lift':>5} {'n':>4}  shape")
        for rate, cnt, gram in rows[: args.top]:
            print(f"{rate*100:5.0f}% {rate/base:5.1f} {cnt:4d}  {gram}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
