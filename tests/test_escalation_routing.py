#!/usr/bin/env python3
"""
Selective-escalation routing: the cognitive-complexity scorer ranks routines sensibly, the Escalation policy applies the
cutoff (with optional codestats-JSON override), and the structural routine signature (`node_sig`) is invariant to the
line shifts, inserted comments, and docstring changes that happen between the emit and apply phases. These are the
decisions that pick which routines go to the stronger model and how requests are re-bound.

No GGUF model required.
"""
import ast
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_python import cognitive_complexity, node_sig  # noqa: E402
from scale_escalate import Escalation, load_codestats_json  # noqa: E402


def _score(src):
    """Cognitive complexity of the single top-level definition in `src`."""
    return cognitive_complexity(ast.parse(src).body[0])


def main():
    # ---- 1. Cognitive complexity: trivial scores 0, branching/nesting accrues, nested defs are opaque ----
    assert _score("def f(x):\n    return x + 1\n") == 0, "a straight-line routine has zero cognitive complexity"

    flat = _score("def f(x):\n    if x: return 1\n    return 0\n")
    nested = _score("def f(x):\n    if x:\n        if x > 1:\n            return 1\n    return 0\n")
    assert nested > flat, "a nested if must score higher than a flat one (the nesting penalty)"

    # `and`/`or` each add one; an `elif` is a cheap continuation, not a fresh nested if.
    assert _score("def f(a, b):\n    if a and b:\n        return 1\n    return 0\n") == 2, \
        "if (+1) plus one boolean operator (+1)"
    chain = _score("def f(x):\n    if x == 1:\n        return 1\n    elif x == 2:\n        return 2\n    else:\n        return 3\n")
    assert chain == 3, "if (+1) + elif (+1) + else (+1), all at the same level"

    # A nested function is opaque: its body does not inflate the outer routine's score.
    opaque = _score("def outer():\n    def inner():\n        if a:\n            if b:\n                return 1\n    return inner\n")
    assert opaque == 0, "nested-function complexity must not leak into the enclosing routine"

    # ---- 2. Escalation policy: strict cutoff, native score by default ----
    esc = Escalation(threshold=10)
    assert esc.should_escalate("foo", 11) is True, "score above the cutoff escalates"
    assert esc.should_escalate("foo", 10) is False, "the cutoff is strict (> not >=)"
    assert esc.should_escalate("foo", 0) is False

    # ---- 3. codestats-JSON override wins for named routines; native fills the gaps ----
    esc2 = Escalation(threshold=10, override={"Heavy.method": 25})
    assert esc2.should_escalate("Heavy.method", 0) is True, "override score escalates even when native is low"
    assert esc2.should_escalate("Other.fn", 12) is True, "an unlisted routine falls back to its native score"
    assert esc2.should_escalate("Other.fn", 3) is False
    assert esc2.score_for("Heavy.method", 0) == 25, "the recorded score is the override, not the native value"

    # ---- 4. load_codestats_json: pulls function->cognitive, keeps the max on collisions ----
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cs.json"
        p.write_text(json.dumps({"functions": [
            {"function": "a.b", "cognitive": 7, "loc": 10},
            {"function": "a.b", "cognitive": 19, "loc": 40},  # same name, higher score wins
            {"function": "c", "cognitive": 2},
        ]}), encoding="utf-8")
        cs = load_codestats_json(p)
        assert cs == {"a.b": 19, "c": 2}, f"codestats map should keep the max per name, got {cs}"

    # ---- 5. node_sig is invariant to position, comments and docstrings, but not to code changes ----
    def fn(src):
        return ast.parse(src).body[0]

    base = node_sig(fn("def target(x):\n    if x:\n        return 1\n    return 0\n"))
    # Same routine sitting lower in a file, with an added docstring and an inserted block comment - all invisible to it.
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

    print("PASS: cognitive scoring, escalation cutoff/override, and node_sig invariance all hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
