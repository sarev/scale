#!/usr/bin/env python3
"""
Within-function block boundaries for JavaScript: the tree-sitter provider must number only lines that legally
begin exactly one statement, at all nesting depths, while:

- excluding continuation lines, `a; b;` multi-statement lines, and the first statement of an inner suite,
- treating a nested function as a single opaque boundary and not descending into it (it becomes its own target),
- recording the correct indentation per boundary.

No GGUF model required: only the tree-sitter provider runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_javascript import iter_block_targets_js  # noqa: E402

SRC = (
    "function f(a, b) {\n"   # 1
    "  let x = a +\n"        # 2  boundary (statement start)
    "          b;\n"         # 3  continuation - excluded
    "  if (x > 0) {\n"       # 4  boundary (opens a block)
    "    x = 0;\n"           # 5  first line of the if-suite - excluded
    "    x = 1;\n"           # 6  boundary (depth 1)
    "  }\n"                  # 7  close brace
    "  p = 1; q = 2;\n"      # 8  two statements - excluded
    "  function g() {\n"     # 9  nested function - opaque boundary, not descended
    "    return x;\n"        # 10 inside g - not an outer boundary
    "  }\n"                  # 11
    "  return x;\n"          # 12 boundary
    "}\n"                    # 13
)


def main():
    lines = SRC.split("\n")
    targets = {t.qualname: t for t in iter_block_targets_js(SRC, lines)}

    assert "f" in targets and "f.g" in targets, f"expected outer and nested targets, got {list(targets)}"

    f = targets["f"]
    assert set(f.boundary_lines) == {2, 4, 6, 9, 12}, f"unexpected boundaries: {f.boundary_lines}"
    assert 3 not in f.boundary_lines, "continuation line must be excluded"
    assert 5 not in f.boundary_lines, "first line of an inner suite must be excluded"
    assert 8 not in f.boundary_lines, "`a; b;` multi-statement line must be excluded"
    assert 10 not in f.boundary_lines and 11 not in f.boundary_lines, \
        "must not descend into a nested function"
    assert f.indent_of[6] == "    ", "nested-block indent should be 4 spaces"

    # The nested function is its own target with its own body boundaries.
    g = targets["f.g"]
    assert set(g.boundary_lines) == {10}, f"unexpected nested boundaries: {g.boundary_lines}"
    assert g.depth == 1, "nested function should be at depth 1"

    print("PASS: JS block boundaries cover legal statement starts only, treating nested functions as opaque")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
