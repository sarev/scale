#!/usr/bin/env python3
"""
Golden-reference miner for the structural paragraph segmenter.

For every *full-line* comment in the given Python sources, capture the first few code lines it introduces (stopping at a
blank line, a dedent below the comment's own indent, or the next comment), then **anonymise their syntax**: identifiers
become `v`, anything called becomes `f`, literals become `0`, expression operators become `+`, while keywords and
structural punctuation (`(`, `)`, `:`, `,`, `.`, `=`, ...) are kept verbatim. The anonymised lines are joined with `:`
into one `stmt:stmt:stmt` shape per comment, and the shapes are tallied across all sources.

The result is a frequency table of the statement *shapes* a human (or an earlier SCALE) chose to prefix with a comment -
i.e. empirical evidence for what a "paragraph opener" looks like. That is the reference we tune the deterministic
segmenter's heuristics against, instead of guessing the rules.

Usage:
    ../.llm-venv/Scripts/python.exe tests/block_eval/anon_paragraphs.py [PATH ...] [--max-lines N] [--min-count N]

With no PATH, defaults to the repo's `scale*.py` modules. A PATH may be a file or a directory (walked for `*.py`).
"""

import argparse
import keyword
import tokenize
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

# OP tokens kept verbatim because they carry statement *shape*, not semantics: grouping/subscript/dict braces, the
# call/slice colon, argument and element separators, attribute dots, the statement terminator, plain assignment, the
# return-annotation arrow, and the decorator marker. Every other OP (arithmetic, comparison, boolean-symbol, augmented
# assignment, walrus, ...) is an "expression operator" and collapses to `+`.
KEEP_OPS = {"(", ")", "[", "]", "{", "}", ":", ",", ".", ";", "=", "->", "@"}

# Keyword constants that are semantically literals - anonymised like any other literal.
LITERAL_KEYWORDS = {"True", "False", "None"}

_SKIP_TOKEN_TYPES = {
    tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT,
    tokenize.COMMENT, tokenize.ENCODING, tokenize.ENDMARKER,
}
_HAS_FSTRING = hasattr(tokenize, "FSTRING_START")  # Python 3.12+ tokenises f-strings into pieces


def _next_significant(toks: List[tokenize.TokenInfo], i: int) -> Optional[tokenize.TokenInfo]:
    """Return the next token after index `i` that carries content (skipping layout tokens), or None."""
    for j in range(i + 1, len(toks)):
        if toks[j].type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            return toks[j]
    return None


def _anon_token(tok: tokenize.TokenInfo, toks: List[tokenize.TokenInfo], i: int) -> Optional[str]:
    """
    Map one token to its anonymised form, or None when it contributes nothing to the shape.

    A NAME is a kept keyword, a literal `0` (for True/False/None), `f` when immediately called (next token is `(`), or
    `v` otherwise. NUMBER/STRING and the ellipsis are literals (`0`). A kept structural OP is returned verbatim; every
    other OP is an expression operator and becomes `+`.
    """
    ttype, tval = tok.type, tok.string
    if ttype == tokenize.NAME:
        if tval in LITERAL_KEYWORDS:
            return "0"
        if keyword.iskeyword(tval):
            return tval
        nxt = _next_significant(toks, i)
        if nxt is not None and nxt.type == tokenize.OP and nxt.string == "(":
            return "f"
        return "v"
    if ttype in (tokenize.NUMBER, tokenize.STRING):
        return "0"
    if ttype == tokenize.OP:
        if tval == "...":
            return "0"
        return tval if tval in KEEP_OPS else "+"
    return None


def _anonymise_by_row(src: str) -> Dict[int, List[str]]:
    """
    Tokenise `src` and return a map from 1-based line number to the anonymised tokens that start on that line.

    f-string interiors (Python 3.12+, where an f-string is several tokens) are collapsed to a single literal `0`, so a
    formatted string reads as one literal regardless of tokeniser version.
    """
    by_row: Dict[int, List[str]] = {}
    toks = list(tokenize.generate_tokens(StringIO(src).readline))
    fstring_depth = 0
    for i, tok in enumerate(toks):
        row = tok.start[0]

        # Collapse a whole f-string to one literal: emit `0` at its start, then swallow its inner tokens.
        if _HAS_FSTRING and tok.type == tokenize.FSTRING_START:
            by_row.setdefault(row, []).append("0")
            fstring_depth += 1
            continue
        if fstring_depth:
            if _HAS_FSTRING and tok.type == tokenize.FSTRING_END:
                fstring_depth -= 1
            continue

        if tok.type in _SKIP_TOKEN_TYPES:
            continue
        piece = _anon_token(tok, toks, i)
        if piece is not None:
            by_row.setdefault(row, []).append(piece)
    return by_row


def extract_patterns(src: str, max_lines: int) -> List[str]:
    """
    Return one `stmt:stmt:...` anonymised shape per full-line comment in `src`.

    For each line whose first non-blank character is `#`, the following code lines are gathered (up to `max_lines`),
    stopping at the first blank line, a line indented less than the comment (a dedent out of the paragraph), or the next
    comment line. Each gathered line is rendered from its anonymised tokens and the lines are joined with `:`. Comments
    that introduce no code (e.g. an interior line of a multi-line comment block) yield nothing.
    """
    lines = src.split("\n")
    by_row = _anonymise_by_row(src)
    patterns: List[str] = []

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            continue
        base_indent = len(line) - len(stripped)

        collected: List[str] = []
        j = i + 1
        while len(collected) < max_lines and j < len(lines):
            cur = lines[j]
            if cur.strip() == "":
                break                                   # blank line ends the paragraph
            if len(cur) - len(cur.lstrip()) < base_indent:
                break                                   # dedent below the comment ends the paragraph
            if cur.lstrip().startswith("#"):
                break                                   # next comment starts a new paragraph
            collected.append(" ".join(by_row.get(j + 1, [])))
            j += 1

        if collected:
            patterns.append(":".join(collected))
    return patterns


def _iter_paths(paths: List[Path]) -> List[Path]:
    """Expand the given paths into a flat list of `*.py` files (directories are walked)."""
    out: List[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(p.rglob("*.py")))
        else:
            out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine anonymised paragraph-opener shapes from comment sites.")
    ap.add_argument("paths", nargs="*", type=Path, help="source files/dirs (default: repo scale*.py)")
    ap.add_argument("--max-lines", type=int, default=3, help="lines to capture after each comment (default 3)")
    ap.add_argument("--min-count", type=int, default=1, help="only show shapes seen at least this many times")
    args = ap.parse_args()

    files = _iter_paths(args.paths) if args.paths else sorted(ROOT.glob("scale*.py"))

    tally: Counter = Counter()
    scanned = 0
    for f in files:
        try:
            src = f.read_text(encoding="utf-8")
            tally.update(extract_patterns(src, args.max_lines))
            scanned += 1
        except (SyntaxError, tokenize.TokenError) as exc:
            print(f"# skipped {f}: {exc}")

    total = sum(tally.values())
    shown = [(c, p) for p, c in tally.items() if c >= args.min_count]
    shown.sort(key=lambda cp: (-cp[0], cp[1]))

    print(f"# {scanned} file(s), {total} comment-led paragraphs, {len(tally)} distinct shapes "
          f"({len(shown)} shown at count >= {args.min_count})\n")
    for count, pattern in shown:
        print(f"{count:4d}  {pattern}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
