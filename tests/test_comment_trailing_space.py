#!/usr/bin/env python3
"""
Generated doc-comments must never carry trailing whitespace - in particular a blank continuation line must be `" *"`,
not `" * "`. This used to leak from the C/JS JSDoc renderers (the `" * "` template adds a trailing space when the line
content is empty) and from the Python docstring patcher (a blank docstring line was left as bare indentation).

Guards (model-free) every doc-comment renderer across C, JS, and Python.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_c import _render_c_block_comment  # noqa: E402
from scale_javascript import _render_jsdoc_block  # noqa: E402
from scale_python import iter_defs_with_info, patch_docstrings_textually  # noqa: E402

DOC = "Summary line.\n\nDetail after a blank line."   # the blank line is the trap


def _assert_no_trailing(name, lines):
    for ln in lines:
        assert ln == ln.rstrip(), f"[{name}] trailing whitespace on line {ln!r}"


def test_c_and_js_block_renderers():
    for name, render, opener in (("c", _render_c_block_comment, "/*"), ("js", _render_jsdoc_block, "/**")):
        out = render(DOC, "")
        _assert_no_trailing(name, out)
        assert " *" in out, f"[{name}] the blank continuation line should be present"
        assert " * " not in out, f"[{name}] a blank line must be ' *', never ' * ' (with trailing space)"
        # Indented (nested) rendering must stay clean too.
        _assert_no_trailing(name + "-indented", render(DOC, "    "))


def test_python_docstring_patcher():
    src = "def f():\n    return 1\n"
    defs = iter_defs_with_info(ast.parse(src))
    out = patch_docstrings_textually(src.split("\n"), defs, {id(defs[0].node): DOC})
    _assert_no_trailing("python", out)
    assert any(ln.strip() == '"""' for ln in out), "a docstring was inserted"
    assert "Detail after a blank line." in "\n".join(out), "the docstring body is present"


def main():
    test_c_and_js_block_renderers()
    test_python_docstring_patcher()
    print("PASS: generated doc-comments carry no trailing whitespace (blank lines are ' *', not ' * ')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
