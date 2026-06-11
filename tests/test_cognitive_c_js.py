#!/usr/bin/env python3
"""
Native cognitive complexity for C and JS (`scale_c.cognitive_complexity_c` / `scale_javascript.cognitive_complexity_js`):
the tree-sitter scorers mirror `scale_python.cognitive_complexity` (nesting penalty, else-if as a cheap continuation,
run-collapsed boolean sequences, opaque nested defs), the block providers stamp the score onto each `BlockTarget`, and
the C def pass routes a complex routine to the manifest on its NATIVE score - no `--codestats-json` report needed.

No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_c  # noqa: E402
import scale_javascript  # noqa: E402
from scale_c import cognitive_complexity_c  # noqa: E402
from scale_escalate import Escalation  # noqa: E402
from scale_javascript import cognitive_complexity_js  # noqa: E402
from scale_llm import GenerationConfig  # noqa: E402


def _score_c(src):
    """Cognitive complexity of each top-level C function in `src`, by name."""
    tree, sb = scale_c._parse_c(src)
    return {d.qualname: cognitive_complexity_c(d.node) for d in scale_c.iter_defs_with_info_c(tree, sb)}


def _score_js(src):
    """Cognitive complexity of each routine in `src`, by qualname (covers every DefInfoJS node shape)."""
    tree, sb = scale_javascript._parse_js(src)
    return {d.qualname: cognitive_complexity_js(d.node) for d in scale_javascript.iter_defs_with_info_js(tree, sb)}


class StubLLM:
    """Canned usable C doc comment; generous budget."""
    n_ctx = 8192
    ctx_margin = 256

    def estimate_tokens(self, text):
        return max(1, len(text) // 4)

    def count_tokens(self, messages):
        return sum(self.estimate_tokens(m["content"]) for m in messages)

    def snippet_budget(self, messages, cfg, **kwargs):
        return 4000

    def generate(self, messages, cfg=None, stop=None):
        return "/*\n * A stub doc comment.\n */"


def main():
    # ---- 1. C: trivial scores 0, branching/nesting accrues, parity with the Python scorer's rules ----
    c_scores = _score_c(
        "int flat(int x) { return x + 1; }\n"
        "int one_if(int x) { if (x) { return 1; } return 0; }\n"
        "int nested(int x) { if (x) { if (x > 1) { return 1; } } return 0; }\n"
        "int boolean(int a, int b) { if (a && b) { return 1; } return 0; }\n"
        "int chain(int x) { if (x == 1) { return 1; } else if (x == 2) { return 2; } else { return 3; } }\n"
        "int run(int a, int b, int c) { return (a && b && c) ? 1 : 0; }\n"
        "int mixed(int a, int b, int c) { return (a && b || c) ? 1 : 0; }\n"
        "int deep(int n) { while (n) { switch (n) { case 1: n = 0; break; default: n--; } } return n; }\n"
    )
    assert c_scores["flat"] == 0, "a straight-line routine has zero cognitive complexity"
    assert c_scores["nested"] > c_scores["one_if"], "a nested if must score higher than a flat one (nesting penalty)"
    assert c_scores["boolean"] == 2, "if (+1) plus one boolean-operator sequence (+1)"
    assert c_scores["chain"] == 3, "if (+1) + else-if (+1) + else (+1): continuations, not fresh nested ifs"
    assert c_scores["run"] == 2, "ternary (+1) + ONE sequence for `a && b && c` (run-collapsed, like Python's BoolOp)"
    assert c_scores["mixed"] == 3, "ternary (+1) + `&&` (+1) + `||` (+1): a changed operator starts a new sequence"
    assert c_scores["deep"] == 3, "while (+1) + switch (+1 +1 nesting)"

    # A prototype declaration record has no body: it scores 0 (the doc-site rule, not complexity, defers it).
    tree, sb = scale_c._parse_c("int helper(int x);\n")
    decl = scale_c.iter_decls_with_info_c(tree, sb)[0]
    assert cognitive_complexity_c(decl.node) == 0, "a bodyless declaration scores 0"

    # ---- 2. JS: same rules, every def-record shape unwraps, nested defs are opaque ----
    js_scores = _score_js(
        "function flat(x) { return x + 1; }\n"
        "function chain(x) { if (x == 1) { return 1; } else if (x == 2) { return 2; } else { return 3; } }\n"
        "const arrow = (a, b, c) => (a && b && c) ? 1 : 0;\n"
        "function opaque() { function inner(a) { if (a) { if (a > 1) { return 1; } } } return inner; }\n"
        "function tryer(x) { try { x(); } catch (e) { return 1; } finally { x = 0; } return 0; }\n"
        "function forof(xs) { for (const x of xs) { if (x) { return x; } } return null; }\n"
    )
    assert js_scores["flat"] == 0, "a straight-line routine has zero cognitive complexity"
    assert js_scores["chain"] == 3, "if (+1) + else-if (+1) + else (+1), folded as continuations"
    assert js_scores["arrow"] == 2, "an expression-bodied arrow scores its expression: ternary (+1) + one && run (+1)"
    assert js_scores["opaque"] == 0, "nested-function complexity must not leak into the enclosing routine"
    assert js_scores["opaque.inner"] == 3, "the nested function is scored separately as its own routine"
    assert js_scores["tryer"] == 3, "try (+1) + catch (+1) + finally (+1), mirroring Python's try/except/finally"
    assert js_scores["forof"] == 3, "for-of (+1) + nested if (+1 +1 nesting)"

    # ---- 3. The block providers stamp the native score onto each BlockTarget ----
    c_src = (
        "int busy(int x)\n"
        "{\n"
        "    int a = 0;\n"
        "    if (x) {\n"
        "        if (x > 1) {\n"
        "            a = 1;\n"
        "        }\n"
        "    }\n"
        "    return a;\n"
        "}\n"
    )
    c_targets = scale_c.iter_block_targets_c(c_src, c_src.split("\n"))
    assert c_targets and c_targets[0].cognitive == 3, "the C BlockTarget must carry the native cognitive score"

    js_src = (
        "function busy(x) {\n"
        "    let a = 0;\n"
        "    if (x) {\n"
        "        if (x > 1) {\n"
        "            a = 1;\n"
        "        }\n"
        "    }\n"
        "    return a;\n"
        "}\n"
    )
    js_targets = scale_javascript.iter_block_targets_js(js_src, js_src.split("\n"))
    assert js_targets and js_targets[0].cognitive == 3, "the JS BlockTarget must carry the native cognitive score"

    # ---- 4. The C def pass escalates on the native score alone (no codestats override, no doc-site plan) ----
    src = (
        "static int simple(int x)\n"
        "{\n"
        "    return x + 1;\n"
        "}\n"
        "\n"
        "static int complex_fn(int x)\n"
        "{\n"
        "    int a = 0;\n"
        "    if (x) {\n"
        "        if (x > 1) {\n"
        "            if (x > 2) {\n"
        "                a = 1;\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "    return a;\n"
        "}\n"
    )
    lines = src.split("\n")
    tree, sb = scale_c._parse_c(src)
    defs = scale_c.iter_defs_with_info_c(tree, sb)
    native = {d.qualname: cognitive_complexity_c(d.node) for d in defs}
    assert native["complex_fn"] > 3 >= native["simple"], "the fixture must straddle the cutoff"

    esc = Escalation(threshold=3)
    doc_map = scale_c.generate_comments_c(StubLLM(), GenerationConfig(max_new_tokens=256), [], defs, src, lines,
                                          escalation=esc)
    deferred = [r for r in esc.requests if r.get("def") is not None]
    assert [r["qualname"] for r in deferred] == ["complex_fn"], "only the complex routine defers on its native score"
    assert deferred[0]["cognitive"] == native["complex_fn"], "the recorded score is the native cognitive complexity"
    assert len(doc_map) == 1, "the simple routine is still documented locally"

    print("OK - C/JS native cognitive complexity: scorer parity, provider wiring, and def-pass routing verified")


if __name__ == "__main__":
    main()
