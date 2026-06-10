#!/usr/bin/env python3
"""
The Python definition pass assembles each routine's snippet with a cursor over source lines (not statement by
statement), so the author's **standalone in-body comments survive into the snippet the model sees** - matching the
C/JS workers, which send the whole body verbatim. This used to drop comments sitting in the gaps between statements,
leaving the docstring model blind to the very comments that explain intent.

Drives the real `generate_docstrings` with a fake model and inspects the assembled snippet. No GGUF required.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_python import generate_docstrings, iter_defs_with_info  # noqa: E402

SRC = (
    "def f(a):\n"
    "    # leading explanation\n"
    "    x = a + 1   # trailing note\n"
    "    # a standalone comment BETWEEN statements\n"
    "    y = x * 2\n"
    "    return y\n"
)


class _FakeLLM:
    """Captures the snippet it is asked to document; returns a canned docstring."""
    def __init__(self):
        self.prompts = []

    def snippet_budget(self, messages, cfg):
        return 100000          # huge: never elide, so we see the full assembled snippet

    def estimate_tokens(self, text):
        return 1

    def generate(self, messages, cfg=None):
        self.prompts.append(messages[-1]["content"])
        return '"""Does a thing."""'


def main():
    llm = _FakeLLM()
    defs = iter_defs_with_info(ast.parse(SRC))
    generate_docstrings(llm, object(), [], defs, SRC, SRC.split("\n"))
    snippet = llm.prompts[-1]

    assert "leading explanation" in snippet, "a comment right after the signature must be present"
    assert "trailing note" in snippet, "a trailing/inline comment must be present"
    assert "BETWEEN statements" in snippet, \
        "a standalone comment between statements must reach the model (the bug this guards)"

    print("PASS: the Python def-pass snippet preserves standalone in-body comments for the model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
