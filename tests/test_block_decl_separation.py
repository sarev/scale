#!/usr/bin/env python3
"""
The leading-declaration heuristic (C/JS): a scope that opens with a run of >= SEG_MIN_LEADING_DECLS (2) local
variable declarations gets a paragraph break before its first real (non-declaration) statement, so the declarations
read as their own block and the body does not run straight into them. The declarations themselves sit as the
uncommented lead-in (before the first break → never a chunk → no comment, i.e. "value 0").

Guards (model-free):
- a run of >= 2 leading declarations forces the break; a single leading declaration does not (no over-fragmenting);
- the declarations get no comment and the first real statement is separated by a blank;
- code is preserved.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_blocks import _apply_edits, code_preserved, SLASH_LINE_STYLE, SEG_MIN_LEADING_DECLS  # noqa: E402
import scale_c as c  # noqa: E402
import scale_javascript as js  # noqa: E402

C_MANY = (
    "void f(int a) {\n"        # 1
    "    int x = a;\n"         # 2  decl
    "    int y = a + 1;\n"     # 3  decl
    "    char *p = bar();\n"   # 4  decl
    "    use(x);\n"            # 5  first real statement -> break above it
    "    use(y);\n"            # 6
    "}\n"
)
C_ONE = (
    "void g(int a) {\n"        # 1
    "    int x = a;\n"         # 2  single decl
    "    use(x);\n"            # 3  no break (only one leading declaration)
    "    use2(x);\n"           # 4
    "}\n"
)
JS_MANY = (
    "function f(a) {\n"        # 1
    "  let x = a;\n"           # 2  decl
    "  let y = a + 1;\n"       # 3  decl
    "  const p = bar();\n"     # 4  decl
    "  use(x);\n"              # 5  first real statement -> break above it
    "  use(y);\n"              # 6
    "}\n"
)


def _check_separates(name, provider, src, first_real, decls):
    lines = src.split("\n")
    t = provider(src, lines)[0]
    starts = [s for s, _ in t.segments]
    fr = next(i for i, l in enumerate(lines, 1) if first_real in l)
    assert fr in starts, f"[{name}] expected a break above the first real statement: {starts}"

    out = _apply_edits(lines, [(s, "C", t.indent_of.get(s, "")) for s, _ in t.segments], SLASH_LINE_STYLE)
    assert code_preserved(lines, out, SLASH_LINE_STYLE), f"[{name}] code must be preserved"
    # The declarations are the uncommented lead-in: no comment line sits among them.
    di = [next(i for i, l in enumerate(out) if d in l) for d in decls]
    for i in di:
        assert not out[i - 1].lstrip().startswith("//"), f"[{name}] a leading declaration must not be commented"
    # A blank separates the last declaration from the first real statement.
    ri = next(i for i, l in enumerate(out) if first_real in l)
    assert out[ri - 1].lstrip().startswith("//") and out[ri - 2].strip() == "", \
        f"[{name}] the first real statement should be a fresh, blank-separated paragraph"


def main():
    assert SEG_MIN_LEADING_DECLS == 2
    _check_separates("c", c.iter_block_targets_c, C_MANY, "use(x);", ["int x = a;", "int y = a + 1;", "char *p"])
    _check_separates("js", js.iter_block_targets_js, JS_MANY, "use(x);", ["let x = a;", "let y = a + 1;", "const p"])

    # A single leading declaration must NOT force a break (avoid over-fragmenting).
    t = c.iter_block_targets_c(C_ONE, C_ONE.split("\n"))[0]
    assert [s for s, _ in t.segments] == [], f"a single leading declaration must not separate: {t.segments}"

    print("PASS: a leading run of >=2 declarations becomes its own uncommented block, body separated (C/JS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
