#!/usr/bin/env python3
"""
Stand-alone development harness for the deterministic structural paragraph segmenter.

The eventual feature replaces the block pass's LLM segmentation with simple, reproducible structural rules. This script
lets us iterate those rules with no model and no SCALE import, via one monolithic round-trip:

    1. concatenate the corpus (default: the repo's `scale*.py`) into one file  -> temp/corpus.py        (ground truth)
    2. strip it to a "wall" - remove blank lines AND comments, the two signals  -> temp/corpus.wall.py    (input)
       that reveal the human paragraphing (comments sit at paragraph heads)
    3. render the wall with a blank line above each break the *humans* chose    -> temp/corpus.ref.py     (reference)
    4. render the wall with a blank above each break the segmenter predicts    -> temp/corpus.candidate.py (candidate)

`ref` and `candidate` are identical comment-free walls that differ ONLY in blank-line placement, so a diff between them
shows exactly where the structural rules over- or under-break. The script prints that diff plus precision/recall/F1.

Segmentation rules (toggleable in RULES, so we can ablate them):
- `first_in_scope`  - blank after a routine's docstring, only when it has one (the convention puts no blank at the
                      first line of a new indentation, and a docstring-less body's first statement is exactly that).
- `after_def`       - blank after a nested def/class (the statement following one ends/starts a paragraph). Ungated.
- `before_return`   - break above a `return` whose preceding statement is at the same indentation (paragraph the
                      trailing return off from the body above it). Ungated.
- `before_compound` - break above a compound statement (`if`/`for`/`while`/`try`/`with`/`match`) or a nested def/class.
- `dedent`          - break above a statement that resumes a shallower indent after a nested block closed.

Breaks are only ever placed at *legal* positions - statement starts, excluding the first statement of an inner suite
(a blank there reads badly) - exactly as SCALE's block provider allows.

Usage:
    env/Scripts/python.exe tests/block_eval/segment_eval.py [PATH ...] [--maps] [--out DIR] [--width N]
"""

import argparse
import ast
import difflib
import sys
from pathlib import Path
from typing import List, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Rule toggles - flip to ablate a rule and watch precision/recall move. `merge_singleline` is a post-pass that drops
# the interior breaks of any run of two or more consecutive single-statement paragraphs (a wall of one-line paragraphs
# reads worse than one grouped paragraph).
RULES = {"first_in_scope": True, "after_def": True, "before_return": True, "before_compound": True, "dedent": True,
         "merge_singleline": False}

# Triviality gate for the before_compound / dedent rules: a block only earns a paragraph break when its significance
# reaches `k`. `measure` is one of "none" (no gate), "stmts" (recursive statement count), or "lines" (span). Tune via
# the ablation sweep - "lines" (raw size) discriminates best: a trivial guard / one-line block is not a paragraph, a
# substantial block is, and size beats cognitive complexity (which is nesting-dominated and wrongly demotes
# long-but-flat blocks; the sweep is why SCALE dropped that metric). first_in_scope is not size-gated: it fires iff
# the body has a docstring (the blank separates docstring from code), which matches the convention.
GATE = {"measure": "lines", "k": 3}

_DEF = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_COMPOUND = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith) + _DEF
if hasattr(ast, "Match"):
    _COMPOUND = _COMPOUND + (ast.Match,)


# ---------------------------- structural facts ----------------------------


def _indent(lines: List[str], lineno: int) -> int:
    """Return the leading-whitespace width of the 1-based source line."""
    text = lines[lineno - 1]
    return len(text) - len(text.lstrip())


def _stmt_start(node: ast.AST) -> int:
    """Return the line a paragraph break would sit above: the first decorator line for a def, else the node line."""
    if isinstance(node, _DEF) and getattr(node, "decorator_list", None):
        return min(d.lineno for d in node.decorator_list)
    return node.lineno


def _child_suites(node: ast.AST) -> List[List[ast.AST]]:
    """Return the statement-list suites nested directly inside a compound statement (bodies, else, except, finally)."""
    suites: List[List[ast.AST]] = []
    for attr in ("body", "orelse", "finalbody"):
        v = getattr(node, attr, None)
        if v:
            suites.append(v)
    for h in getattr(node, "handlers", []):           # try/except handlers
        if getattr(h, "body", None):
            suites.append(h.body)
    for c in getattr(node, "cases", []):              # match cases
        if getattr(c, "body", None):
            suites.append(c.body)
    return suites


def _is_docstring(stmt: ast.AST) -> bool:
    """Report whether a statement is a bare string-literal expression (a docstring)."""
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def _iter_targets(tree: ast.AST) -> List[Tuple[str, ast.AST]]:
    """Return `(qualname, node)` for every function/class, in source order (mirrors SCALE's routine set)."""
    out: List[Tuple[str, ast.AST]] = []

    def rec(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _DEF):
                q = f"{prefix}{child.name}"
                out.append((q, child))
                rec(child, q + ".")
            else:
                rec(child, prefix)

    rec(tree, "")
    return out


class Record:
    """One statement start in a routine body, with the structural facts the segmenter reasons over."""

    __slots__ = ("start", "indent", "node", "first_of_suite", "first_in_scope", "legal")

    def __init__(self, start, indent, node, first_of_suite, first_in_scope):
        self.start = start
        self.indent = indent
        self.node = node
        self.first_of_suite = first_of_suite
        self.first_in_scope = first_in_scope
        self.legal = not first_of_suite          # a blank as the first line of an inner suite reads badly


def _collect(node: ast.AST, lines: List[str]) -> List[Record]:
    """Collect the statement-start records of a routine body, not descending into nested definitions."""
    records: List[Record] = []

    def walk(stmts: List[ast.AST], is_top: bool) -> None:
        for idx, stmt in enumerate(stmts):
            start = _stmt_start(stmt)
            records.append(Record(start, _indent(lines, start), stmt,
                                   first_of_suite=(not is_top) and idx == 0,
                                   first_in_scope=is_top and idx == 0))
            if not isinstance(stmt, _DEF):
                for sub in _child_suites(stmt):
                    walk(sub, False)

    body = list(getattr(node, "body", []))
    if body and _is_docstring(body[0]):
        body = body[1:]
    walk(body, True)
    records.sort(key=lambda r: r.start)
    return records


# ---------------------------- the rules under test ----------------------------


def _parent_map(tree: ast.AST) -> dict:
    """Return a child-node -> parent-node map for the whole tree (for ancestor walks)."""
    parents: dict = {}
    for n in ast.walk(tree):
        for c in ast.iter_child_nodes(n):
            parents[c] = n
    return parents


def _stmt_count(node: ast.AST) -> int:
    """Count statements inside a compound's suites, recursively, treating a nested def/class as one opaque statement."""
    total = 0
    for suite in _child_suites(node):
        for s in suite:
            total += 1
            if not isinstance(s, _DEF):
                total += _stmt_count(s)
    return total


def _significance(node: ast.AST) -> int:
    """Return the configured significance measure of a block: statement count or line span."""
    measure = GATE["measure"]
    if measure == "stmts":
        return _stmt_count(node)
    if measure == "lines":
        return getattr(node, "end_lineno", node.lineno) - _stmt_start(node) + 1
    return 1 << 30                                # "none": always significant


def _closed_block(prev_node: ast.AST, resume_start: int, parents: dict, target: ast.AST):
    """Return the outermost compound that closed between `prev_node` and a dedent resuming at `resume_start`."""
    best = None
    n = parents.get(prev_node)
    while n is not None and n is not target:
        if isinstance(n, _COMPOUND) and getattr(n, "end_lineno", 0) < resume_start:
            best = n                             # keep climbing to reach the outermost closed block
        n = parents.get(n)
    return best


def predict_breaks(records: List[Record], parents: dict, target: ast.AST) -> Set[int]:
    """
    Return the set of legal statement-start lines the segmenter would prefix with a paragraph break.

    A position breaks when it is the body's first statement (`first_in_scope`), starts a compound statement / nested
    definition (`before_compound`), or resumes a shallower indent after a nested block closed (`dedent`). The latter
    two are gated by GATE: the block (the compound being entered, or the block that just closed) must reach the
    significance threshold, so a trivial guard or one-line block is not paragraphed off. Only legal positions (not the
    first line of an inner suite) are ever returned.
    """
    # first_in_scope fires only when the body has a docstring (the blank separates docstring from code); a docstring-
    # less body's first statement is itself the first line of a new indentation, which takes no blank.
    target_body = list(getattr(target, "body", []))
    has_doc = bool(target_body and _is_docstring(target_body[0]))

    breaks: Set[int] = set()
    for i, r in enumerate(records):
        if not r.legal:
            continue
        prev = records[i - 1] if i > 0 else None
        if RULES["first_in_scope"] and r.first_in_scope:
            if has_doc:
                breaks.add(r.start)
        elif RULES["after_def"] and prev is not None and isinstance(prev.node, _DEF):
            breaks.add(r.start)                  # blank after a nested def/method - it clearly ends a paragraph
        elif RULES["before_return"] and isinstance(r.node, ast.Return) and prev is not None and prev.indent == r.indent:
            breaks.add(r.start)                  # always paragraph a trailing return off from the body above it
        elif RULES["before_compound"] and isinstance(r.node, _COMPOUND):
            if _significance(r.node) >= GATE["k"]:
                breaks.add(r.start)
        elif RULES["dedent"] and prev is not None and prev.indent > r.indent:
            blk = _closed_block(prev.node, r.start, parents, target)
            if blk is None or _significance(blk) >= GATE["k"]:
                breaks.add(r.start)

    if RULES.get("merge_singleline"):
        breaks = _merge_singleline(breaks, records, getattr(target, "end_lineno", 1 << 30))
    return breaks


def _merge_singleline(breaks: Set[int], records: List[Record], body_end: int) -> Set[int]:
    """
    Drop the interior breaks of any run of two or more consecutive single-statement paragraphs.

    A paragraph spanning a single source line (a one-liner, e.g. `x = f()` or `if x: return`) is "single-line";
    several in a row read as noise, so a maximal run of them keeps only its opening break (separating it from the prior
    paragraph) and the rest are merged into one paragraph. An isolated single-line paragraph between larger ones is
    left alone. Span is measured in source lines, not AST records, so an inline-compound one-liner still counts as one.
    """
    starts = sorted(breaks)
    if len(starts) < 2:
        return breaks
    bounds = starts + [body_end + 1]

    def span(lo: int, hi: int) -> int:
        """Source-line span of the paragraph occupying [lo, hi): last statement end minus lo, or 0 if empty."""
        ends = [getattr(r.node, "end_lineno", r.start) for r in records if lo <= r.start < hi]
        return max(ends) - lo + 1 if ends else 0

    sizes = [span(bounds[i], bounds[i + 1]) for i in range(len(starts))]

    keep = set(starts)
    i = 0
    while i < len(starts):
        if sizes[i] == 1:
            j = i
            while j + 1 < len(starts) and sizes[j + 1] == 1:
                j += 1
            for k in range(i + 1, j + 1):       # j > i means a run of >= 2 single-line paragraphs
                keep.discard(starts[k])
            i = j + 1
        else:
            i += 1
    return keep


def _is_comment(line: str) -> bool:
    """Report whether a source line is a full-line comment (ignoring leading whitespace)."""
    return line.lstrip().startswith("#")


def human_breaks(records: List[Record], lines: List[str]) -> Set[int]:
    """
    Return the legal positions that have an existing paragraph break above them in the source.

    A break exists when, skipping any contiguous comment lines directly above the statement, the next line up is blank -
    i.e. the original paragraphed this statement off from the one before it.
    """
    truth: Set[int] = set()
    for r in records:
        if not r.legal:
            continue
        k = r.start - 2                         # 0-based index of the line directly above
        while k >= 0 and _is_comment(lines[k]):
            k -= 1
        if k >= 0 and lines[k].strip() == "":
            truth.add(r.start)
    return truth


# ---------------------------- corpus round-trip ----------------------------


def build_corpus(files: List[Path]) -> str:
    """
    Concatenate the sources into one parseable module: hoist a single `from __future__` line, drop the rest.

    `from __future__` imports must be the first statement of a module, so the per-file copies are removed and one is
    re-emitted at the top. They are module-level only, so this never affects any routine's intra-body paragraphing.
    """
    future = False
    body: List[str] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").split("\n"):
            if line.strip().startswith("from __future__ import"):
                future = True
            else:
                body.append(line)
        body.append("")                          # blank gap between files (stripped by the wall)
    header = ["from __future__ import annotations", ""] if future else []
    return "\n".join(header + body)


def _kept_linenos(src: str) -> List[int]:
    """
    Return the 1-based line numbers that survive walling: everything except blank lines and full-line comments.

    Lines inside a string literal (docstrings, multi-line data) are protected so their blanks/`#` survive, and a
    leading shebang is kept - mirroring `make_wall.strip_to_wall`, but reporting line numbers so breaks can be mapped.
    """
    lines = src.split("\n")
    protected: Set[int] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lo = getattr(node, "lineno", None)
            hi = getattr(node, "end_lineno", lo)
            if lo:
                protected.update(range(lo, hi + 1))

    kept: List[int] = []
    for i, line in enumerate(lines, start=1):
        if i == 1 and line.startswith("#!"):
            kept.append(i)
        elif i in protected:
            kept.append(i)
        elif line.strip() != "" and not line.lstrip().startswith("#"):
            kept.append(i)
    return kept


def _render(kept: List[int], lines: List[str], breaks: Set[int]) -> str:
    """Render the wall (kept lines, in order) with a blank line inserted above each line whose number is in `breaks`."""
    out: List[str] = []
    for gt in kept:
        if gt in breaks and out and out[-1].strip() != "":
            out.append("")
        out.append(lines[gt - 1])
    return "\n".join(out) + "\n"


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Return (precision, recall, F1) from a TP/FP/FN triple, treating 0/0 as 1.0."""
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def show_routine(qualname: str, records: List[Record], pred: Set[int], truth: Set[int],
                 lines: List[str], width: int) -> None:
    """Print a paragraph map of one routine, marking each legal position as a correct / missed / extra break."""
    fp, fn = len(pred - truth), len(truth - pred)
    if not (fp or fn):
        return
    print(f"\n### {qualname}   (+{fp} extra, -{fn} missed)")
    for r_ in records:
        text = lines[r_.start - 1].rstrip()
        if len(text) > width:
            text = text[: width - 1] + "…"
        mark = "        "
        if r_.legal:
            in_p, in_t = r_.start in pred, r_.start in truth
            mark = "  ✓     " if in_p and in_t else "  ✗ MISS" if in_t else "  ✗ EXTR" if in_p else mark
        print(f"{mark}  {text}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Develop/score the deterministic structural paragraph segmenter.")
    ap.add_argument("paths", nargs="*", type=Path, help="source files/dirs (default: repo scale*.py)")
    ap.add_argument("--maps", action="store_true", help="also print per-routine break maps for routines that diverge")
    ap.add_argument("--out", type=Path, default=ROOT / "temp", help="directory for the corpus artifacts")
    ap.add_argument("--width", type=int, default=96, help="max displayed line width in maps")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")     # the maps/diff carry ✓/✗ and « » glyphs
    except AttributeError:
        pass

    files = list(args.paths) if args.paths else sorted(ROOT.glob("scale*.py"))
    corpus = build_corpus(files)
    lines = corpus.split("\n")
    tree = ast.parse(corpus)

    parents = _parent_map(tree)
    pred_all: Set[int] = set()
    truth_all: Set[int] = set()
    legal_total = 0
    per_routine = []
    for qualname, node in _iter_targets(tree):
        records = _collect(node, lines)
        if not any(r.legal for r in records):
            continue
        pred = predict_breaks(records, parents, node)
        truth = human_breaks(records, lines)
        pred_all |= pred
        truth_all |= truth
        legal_total += sum(1 for r in records if r.legal)
        per_routine.append((qualname, records, pred, truth))

    # Build the four artifacts; ref and candidate differ only in blank placement.
    args.out.mkdir(parents=True, exist_ok=True)
    kept = _kept_linenos(corpus)
    (args.out / "corpus.py").write_text(corpus, encoding="utf-8", newline="\n")
    (args.out / "corpus.wall.py").write_text(_render(kept, lines, set()), encoding="utf-8", newline="\n")
    ref = _render(kept, lines, truth_all)
    cand = _render(kept, lines, pred_all)
    (args.out / "corpus.ref.py").write_text(ref, encoding="utf-8", newline="\n")
    (args.out / "corpus.candidate.py").write_text(cand, encoding="utf-8", newline="\n")

    if args.maps:
        for qualname, records, pred, truth in per_routine:
            show_routine(qualname, records, pred, truth, lines, args.width)

    tp = len(pred_all & truth_all)
    fp, fn = len(pred_all - truth_all), len(truth_all - pred_all)
    p, r, fl = _prf(tp, fp, fn)

    diff = list(difflib.unified_diff(ref.split("\n"), cand.split("\n"),
                                     "corpus.ref.py", "corpus.candidate.py", lineterm=""))
    print("\n".join(diff))
    print(f"\n#### {len(files)} files, {legal_total} legal positions, {len(truth_all)} human breaks")
    print(f"#### Structural: P={p*100:.0f}% R={r*100:.0f}% F1={fl*100:.0f}%  (tp={tp} fp={fp} fn={fn})  rules={RULES}")
    print(f"#### artifacts in {args.out}\\corpus.*.py  -  e.g.  git --no-pager diff --no-index "
          f"{args.out}\\corpus.ref.py {args.out}\\corpus.candidate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
