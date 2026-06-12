#!/usr/bin/env python3
"""
A fresh docstring inserted into a class whose first member is decorated must land above the
decorator, not between the decorator and its `def`. The patched output must remain valid Python.

Guards the bug where `_header_span` ended a class header at `body[0].lineno - 1` — for a decorated
first method that line is the decorator itself, so the inserted class docstring split `@classmethod`
from its `def` (found by the dogfood run on scale_llm.py's `LocalChatModel`).

No GGUF model required.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_python import iter_defs_with_info, patch_docstrings_textually  # noqa: E402


def main():
    src = (
        "class Model:\n"
        "    @classmethod\n"
        "    def make(cls):\n"
        "        return cls()\n"
        "\n"
        "\n"
        "def outer():\n"
        "    @staticmethod\n"
        "    def trick():\n"
        "        pass\n"
        "    return trick\n"
    )
    lines = src.split("\n")
    defs = iter_defs_with_info(ast.parse(src))
    doc_map = {id(d.node): f"doc for {d.qualname}" for d in defs}

    patched_lines = patch_docstrings_textually(lines, defs, doc_map)
    patched = "\n".join(patched_lines)

    ast.parse(patched)  # must still be valid Python

    assert "doc for Model" in patched, "class should get a docstring"
    assert "doc for Model.make" in patched, "decorated method should get a docstring"
    assert "doc for outer" in patched, "function with decorated first statement should get a docstring"

    # The class docstring must sit above the decorator: nothing may separate decorator and def.
    deco_idx = next(i for i, ln in enumerate(patched_lines) if ln.strip() == "@classmethod")
    assert patched_lines[deco_idx + 1].strip().startswith("def make"), \
        "decorator must be immediately followed by its def"
    doc_idx = next(i for i, ln in enumerate(patched_lines) if "doc for Model" in ln and "make" not in ln)
    assert doc_idx < deco_idx, "class docstring must be inserted above the decorated first member"

    print("PASS: class docstring lands above a decorated first member, output still valid Python")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
