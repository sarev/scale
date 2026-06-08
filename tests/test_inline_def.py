#!/usr/bin/env python3
"""
Inline / single-line defs must be left untouched (never corrupted), while normal
defs still receive docstrings. The patched output must remain valid Python.

Guards the bug where `def f(): return 1` had a docstring inserted above the
keyword, producing broken code.

No GGUF model required.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_python import iter_defs_with_info, patch_docstrings_textually  # noqa: E402


def main():
    src = (
        "def inline(): return 1\n"
        "\n"
        "\n"
        "def normal():\n"
        "    return 2\n"
        "\n"
        "\n"
        "class Inline: pass\n"
    )
    lines = src.split("\n")
    defs = iter_defs_with_info(ast.parse(src))
    doc_map = {id(d.node): f"doc for {d.qualname}" for d in defs}

    patched = "\n".join(patch_docstrings_textually(lines, defs, doc_map))

    ast.parse(patched)  # must still be valid Python

    assert "doc for normal" in patched, "normal def should get a docstring"
    assert "def inline(): return 1" in patched, "inline def line must be left verbatim"
    assert "class Inline: pass" in patched, "inline class line must be left verbatim"
    assert "doc for inline" not in patched, "inline def must be skipped"
    assert "doc for Inline" not in patched, "inline class must be skipped"

    print("PASS: inline defs skipped, normal def documented, output still valid Python")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
