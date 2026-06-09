#!/usr/bin/env python3
"""
Within-function block boundaries: the Python provider must number only lines that legally begin exactly one
statement, at all nesting depths, while:

- skipping a leading docstring (no comment is ever placed between a def and its docstring),
- excluding continuation lines, `a; b` multi-statement lines, and inline-compound inner statements,
- treating a nested definition as a single opaque boundary and not descending into it,
- recording the correct indentation per boundary,
- giving classes a block pass over their own body too.

No GGUF model required: only the AST-based provider runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_python import iter_block_targets  # noqa: E402

SRC_FUNC = (
    "def outer(a, b):\n"            # 1
    '    """Docstring."""\n'        # 2  (docstring - skipped)
    "    x = (a +\n"                # 3  boundary
    "         b)\n"                 # 4  continuation - excluded
    "    if x > 0:\n"               # 5  boundary
    "        y = 1\n"               # 6  first line of the if-suite - excluded
    "        z = 2\n"               # 7  boundary (depth 2)
    "    p = 1; q = 2\n"            # 8  two statements - excluded
    "\n"                           # 9
    "    def nested():\n"           # 10 nested def - opaque boundary, not descended
    "        inner = 5\n"           # 11 inside nested - not an outer boundary
    "        return inner\n"        # 12 inside nested
    "\n"                           # 13
    "    return x\n"                # 14 boundary
)

SRC_CLASS = (
    "class Foo:\n"                  # 1
    '    """Class doc."""\n'        # 2  (docstring - skipped)
    "    attr = 1\n"                # 3  boundary
    "    CONST = 2\n"               # 4  boundary
    "    def method(self):\n"       # 5  nested def - opaque boundary
    "        return self.attr\n"    # 6  inside method
)


def _by_name(targets):
    return {t.qualname: t for t in targets}


def main():
    func_lines = SRC_FUNC.split("\n")
    targets = _by_name(iter_block_targets(SRC_FUNC, func_lines))

    assert "outer" in targets and "outer.nested" in targets, "expected both routines as targets"

    outer = targets["outer"]
    assert set(outer.boundary_lines) == {3, 5, 7, 10, 14}, \
        f"unexpected outer boundaries: {outer.boundary_lines}"
    assert 2 not in outer.boundary_lines, "leading docstring must be skipped"
    assert 4 not in outer.boundary_lines, "continuation line must be excluded"
    assert 6 not in outer.boundary_lines, "first line of an inner suite must be excluded"
    assert 8 not in outer.boundary_lines, "`a; b` multi-statement line must be excluded"
    assert 11 not in outer.boundary_lines and 12 not in outer.boundary_lines, \
        "must not descend into a nested definition"

    # Indentation is captured exactly, at every depth.
    assert outer.indent_of[3] == "    ", "top-level body indent should be 4 spaces"
    assert outer.indent_of[7] == "        ", "nested-block indent should be 8 spaces"

    nested = targets["outer.nested"]
    assert set(nested.boundary_lines) == {11, 12}, f"unexpected nested boundaries: {nested.boundary_lines}"
    assert nested.depth == 1, "nested function should be at depth 1"

    # Classes get a block pass over their own body, treating methods as opaque boundaries.
    class_lines = SRC_CLASS.split("\n")
    ctargets = _by_name(iter_block_targets(SRC_CLASS, class_lines))
    foo = ctargets["Foo"]
    assert set(foo.boundary_lines) == {3, 4, 5}, f"unexpected class boundaries: {foo.boundary_lines}"
    assert foo.kind == "class"
    assert set(ctargets["Foo.method"].boundary_lines) == {6}, "method body should be its own target"

    print("PASS: block boundaries cover legal statement starts only, at all depths, without descending into defs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
