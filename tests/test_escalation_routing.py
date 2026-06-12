#!/usr/bin/env python3
"""
Online-mode emit routing: the model-free collectors defer EVERY routine - `collect_def_requests` records each
definition's docstring slot, `defer_block_targets` records each routine's chunk recipe, both into ONE per-routine
request - while the emitted text is left byte-for-byte untouched. The structural routine signature (`node_sig`) stays
invariant to the line shifts, inserted comments, and docstring changes that happen between the emit and apply phases
(how requests are re-bound).

No GGUF model required.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_python import collect_def_requests, iter_block_targets, node_sig  # noqa: E402
from scale_blocks import defer_block_targets  # noqa: E402
from scale_escalate import Escalation, unfilled_answers  # noqa: E402


SRC = (
    "def simple():\n"
    "    a = 1\n"
    "    return a\n"
    "\n"
    "\n"
    "class Box:\n"
    "    def get(self):\n"
    "        return self.v\n"
)


def main():
    lines = SRC.split("\n")

    # ---- 1. Online defers everything: every definition gets a def request, every routine body a block recipe ----
    esc = Escalation(doc_style="style")
    n_defs = collect_def_requests(SRC, lines, esc)
    assert n_defs == 3, f"every definition must be requested, got {n_defs}"
    assert [r["qualname"] for r in esc.requests] == ["simple", "Box", "Box.get"], \
        f"unexpected request set: {[r['qualname'] for r in esc.requests]}"
    assert all(r.get("def") is not None for r in esc.requests), "every request carries a def slot"
    assert esc.requests[0]["snippet"].startswith("def simple():"), "the snippet is the routine's verbatim span"

    targets = iter_block_targets(SRC, lines)
    n_blocks = defer_block_targets(esc, lines, targets)
    assert n_blocks >= 1, "routines with bodies must contribute block recipes"

    # ---- 2. Def + block merge into ONE per-routine request, sharing one snippet ----
    simple = [r for r in esc.requests if r["qualname"] == "simple"]
    assert len(simple) == 1, "both collectors must record into one request per routine"
    assert simple[0].get("def") is not None and simple[0].get("blocks") is not None
    assert simple[0]["snippet"].startswith("def simple():")
    for chunk in simple[0]["blocks"]["chunks"]:
        a, b = chunk["lines"]
        snip = simple[0]["snippet"].split("\n")
        assert 1 <= a <= b <= len(snip), "chunk ranges index into the merged snippet"

    # Every slot starts unfilled (the counter is what drives the fill loop).
    manifest = esc.to_manifest("test.py", "python", "\n")
    assert len(unfilled_answers(manifest)) >= n_defs, "every recorded slot must start unfilled"

    # ---- 3. The emitted text is untouched: the collectors only read ----
    assert lines == SRC.split("\n"), "emit must never modify the source"

    # ---- 4. node_sig is invariant to position, comments and docstrings, but not to code changes ----
    def fn(src):
        return ast.parse(src).body[0]

    base = node_sig(fn("def target(x):\n    if x:\n        return 1\n    return 0\n"))
    # Same routine with an added docstring and an inserted block comment - all invisible to it.
    decorated = node_sig(fn(
        "def target(x):\n"
        '    """A docstring the apply phase might add."""\n'
        "    # an inserted block comment\n"
        "    if x:\n"
        "        return 1\n"
        "    return 0\n"
    ))
    assert base == decorated, "node_sig must ignore docstrings and comments (only structure matters)"

    changed = node_sig(fn("def target(x):\n    if x:\n        return 2\n    return 0\n"))  # body changed
    assert changed != base, "a code change must change the structural signature"

    print("PASS: online emit defers every def + block recipe into merged untouched-source requests; node_sig invariant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
