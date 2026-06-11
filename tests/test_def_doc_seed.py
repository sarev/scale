#!/usr/bin/env python3
"""
The definition pass's ingest-and-update seed: a routine that already has documentation is shown that documentation in
its generation turn, so the model refreshes the existing contract instead of re-deriving it blind (the routine-level
analogue of the `--file-doc` description seed). For C/JS the doc comment sits ABOVE the header - outside the assembled
snippet - so the worker surfaces it explicitly; for Python the docstring is inside the body, so the assembled snippet
carries it natively. A routine with no existing doc gets no seed text.

Model-free: a fake LLM records each generation prompt.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_python import generate_docstrings, iter_defs_with_info  # noqa: E402
import scale_c as sc            # noqa: E402
import scale_javascript as sjs  # noqa: E402


class FakeLLM:
    """Records each generation prompt; returns a canned doc-comment block."""

    def __init__(self, reply):
        self.prompts = []
        self._reply = reply

    def snippet_budget(self, messages, cfg):
        return 100000      # never elide

    def estimate_tokens(self, text):
        return 1

    def generate(self, messages, cfg=None):
        self.prompts.append(messages[-1]["content"])
        return self._reply


SEED_MARK = "already documented"


def test_c_seed():
    src = ("/* Clamp v into [lo, hi]. */\n"
           "int clamp(int v, int lo, int hi) { return v < lo ? lo : v; }\n"
           "int bare(void) { return 1; }\n")
    tree, sb = sc._parse_c(src)
    defs = sc.iter_defs_with_info_c(tree, sb)
    llm = FakeLLM("/*\n * A function.\n */")
    sc.generate_comments_c(llm, object(), [], defs, src, src.split("\n"))

    clamp_prompt = next(p for p in llm.prompts if "int clamp" in p)
    bare_prompt = next(p for p in llm.prompts if "int bare" in p)
    assert SEED_MARK in clamp_prompt and "Clamp v into [lo, hi]." in clamp_prompt
    assert SEED_MARK not in bare_prompt                  # no existing doc -> no seed text


def test_js_seed():
    src = ("/**\n * Clamp x into range.\n */\n"
           "function clamp(x) { return x; }\n"
           "function bare() { return 1; }\n")
    tree, sb = sjs._parse_js(src)
    defs = sjs.iter_defs_with_info_js(tree, sb)
    llm = FakeLLM("/**\n * A function.\n */")
    sjs.generate_comments_js(llm, object(), [], defs, src, src.split("\n"))

    clamp_prompt = next(p for p in llm.prompts if "function clamp" in p)
    bare_prompt = next(p for p in llm.prompts if "function bare" in p)
    assert SEED_MARK in clamp_prompt and "Clamp x into range." in clamp_prompt
    assert SEED_MARK not in bare_prompt


def test_python_snippet_carries_docstring_natively():
    # Python's docstring lives inside the body, so the assembled snippet already shows it (no separate seed text);
    # the prompt asks the model to update "any existing comment".
    src = 'def clamp(x):\n    """Clamp x into range."""\n    return x\n'
    defs = iter_defs_with_info(ast.parse(src))
    llm = FakeLLM('"""A docstring."""')
    generate_docstrings(llm, object(), [], defs, src, src.split("\n"))
    assert "Clamp x into range." in llm.prompts[0]


def main():
    test_c_seed()
    test_js_seed()
    test_python_snippet_carries_docstring_natively()
    print("PASS: an existing routine doc is surfaced as an ingest-and-update seed (C/JS explicit, Python via the snippet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
