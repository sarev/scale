#!/usr/bin/env python3
"""
Within-function block boundaries for C: the tree-sitter provider must number only lines that legally begin
exactly one statement, at all nesting depths, while:

- excluding continuation lines, `a; b;` multi-statement lines, and the first statement of an inner suite,
- recursing into nested brace blocks (if/for/...) to find deeper statement starts,
- recording the correct indentation per boundary.

No GGUF model required: only the tree-sitter provider runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_c import iter_block_targets_c  # noqa: E402

SRC = (
    "void f(int a, int b) {\n"             # 1
    "    int x = a +\n"                    # 2  boundary (statement start)
    "            b;\n"                     # 3  continuation - excluded
    "    if (x > 0) {\n"                   # 4  boundary (opens a block)
    "        x = 0;\n"                     # 5  first line of the if-suite - excluded
    "        x = 1;\n"                     # 6  boundary (depth 1)
    "    }\n"                              # 7  close brace - not a statement start
    "    p = 1; q = 2;\n"                  # 8  two statements - excluded
    "    for (int i = 0; i < 3; i++) {\n"  # 9  boundary (opens a block)
    "        x += i;\n"                    # 10 first line of the for-suite - excluded
    "    }\n"                              # 11 close brace
    "    return;\n"                        # 12 boundary
    "}\n"                                  # 13
)


def main():
    lines = SRC.split("\n")
    targets = iter_block_targets_c(SRC, lines)
    assert len(targets) == 1, f"expected one function target, got {len(targets)}"
    t = targets[0]

    assert t.qualname == "f" and t.kind == "function"
    assert (t.body_start, t.body_end) == (2, 13), (t.body_start, t.body_end)

    assert set(t.boundary_lines) == {2, 4, 6, 9, 12}, f"unexpected boundaries: {t.boundary_lines}"
    assert 3 not in t.boundary_lines, "continuation line must be excluded"
    assert 5 not in t.boundary_lines and 10 not in t.boundary_lines, "first line of an inner suite must be excluded"
    assert 8 not in t.boundary_lines, "`a; b;` multi-statement line must be excluded"

    # Statement starts are found at every depth, with exact indentation.
    assert t.indent_of[2] == "    ", "top-level body indent should be 4 spaces"
    assert t.indent_of[6] == "        ", "nested-block indent should be 8 spaces"

    print("PASS: C block boundaries cover legal statement starts only, at all depths, recursing into nested blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
