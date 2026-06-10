#!/usr/bin/env python3
"""
The definition-pass call-graph hooks in the three workers, driven by a fake LLM: routines are visited in the supplied
`doc_order`, each generation turn carries its callees' contract notes (`callee_context`), and `on_doc` fires after each
routine so the contract store can update. Absent hooks, behaviour is unchanged (deepest-first, no injected notes).

Model-free: the LLM is stubbed and only the parsing/snippet/patch logic runs.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_python import generate_docstrings, iter_defs_with_info  # noqa: E402
import scale_c as sc          # noqa: E402
import scale_javascript as sjs  # noqa: E402


class FakeLLM:
    """Records each generation prompt; returns a canned doc-comment block for any language's extractor."""
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


def test_python_wiring():
    src = "def helper():\n    return 1\n\ndef caller():\n    return helper()\n"
    defs = iter_defs_with_info(ast.parse(src))
    llm = FakeLLM('"""A docstring."""')

    visited, notes_seen = [], {}

    def callee_context(q):
        return "Functions/methods this routine calls:\n- helper: Return one." if q == "caller" else ""

    def on_doc(q, doc):
        visited.append(q)
        notes_seen[q] = doc

    generate_docstrings(llm, object(), [], defs, src, src.split("\n"),
                        doc_order=["helper", "caller"], callee_context=callee_context, on_doc=on_doc)

    assert visited == ["helper", "caller"], visited                  # doc_order honoured
    # The caller's prompt carries its callee contract; the leaf's does not.
    helper_prompt = next(p for p in llm.prompts if "def helper" in p)
    caller_prompt = next(p for p in llm.prompts if "def caller" in p)
    assert "helper: Return one." in caller_prompt
    assert "helper: Return one." not in helper_prompt
    assert notes_seen["caller"] == '"""A docstring."""'              # on_doc got the generated docstring


def test_python_absent_hooks_unchanged():
    # Without doc_order, the loop is deepest-first; a nested child is documented before its parent (so the parent sees
    # the child's stub). No notes are injected. This is the pre-feature behaviour.
    src = "def outer():\n    def inner():\n        return 1\n    return inner()\n"
    defs = iter_defs_with_info(ast.parse(src))
    llm = FakeLLM('"""Doc."""')
    doc_map = generate_docstrings(llm, object(), [], defs, src, src.split("\n"))
    assert len(doc_map) == 2                                          # both routines documented
    inner_prompt = next(p for p in llm.prompts if "def inner" in p and "return 1" in p)
    assert "Functions/methods this routine calls:" not in inner_prompt


def test_c_wiring():
    src = ("int helper(void) { return 1; }\n"
           "int caller(void) { return helper(); }\n")
    tree, sb = sc._parse_c(src)
    defs = sc.iter_defs_with_info_c(tree, sb)
    llm = FakeLLM("/*\n * A function.\n */")
    visited = []
    cc = lambda q: "Functions/methods this routine calls:\n- helper: Return one." if q == "caller" else ""
    sc.generate_comments_c(llm, object(), [], defs, src, src.split("\n"),
                           doc_order=["helper", "caller"], callee_context=cc,
                           on_doc=lambda q, d: visited.append(q))
    assert visited == ["helper", "caller"], visited
    caller_prompt = next(p for p in llm.prompts if "caller" in p and "return helper" in p)
    assert "helper: Return one." in caller_prompt


def test_js_wiring():
    src = ("function helper() { return 1; }\n"
           "function caller() { return helper(); }\n")
    tree, sb = sjs._parse_js(src)
    defs = sjs.iter_defs_with_info_js(tree, sb)
    llm = FakeLLM("/**\n * A function.\n */")
    visited = []
    cc = lambda q: "Functions/methods this routine calls:\n- helper: Return one." if q == "caller" else ""
    sjs.generate_comments_js(llm, object(), [], defs, src, src.split("\n"),
                             doc_order=["helper", "caller"], callee_context=cc,
                             on_doc=lambda q, d: visited.append(q))
    assert visited == ["helper", "caller"], visited
    caller_prompt = next(p for p in llm.prompts if "function caller" in p)
    assert "helper: Return one." in caller_prompt


def main():
    test_python_wiring()
    test_python_absent_hooks_unchanged()
    test_c_wiring()
    test_js_wiring()
    print("PASS: def-pass honours doc_order, injects callee notes, and fires on_doc (Py/C/JS); absent hooks unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
