#!/usr/bin/env python3
"""
The leading-declaration heuristic (C/JS): a scope that opens with a run of >= SEG_MIN_LEADING_DECLS (2) local
variable declarations gets a paragraph break before its first real (non-declaration) statement, so the declarations
form their own chunk (the opening chunk) and the body does not run straight into them. Whether the declaration chunk
ends up with a comment is up to the value score (it scores low); here we test the *structure* only.

Guards (model-free):
- a run of >= 2 leading declarations splits into two chunks (decls, then body); a single leading declaration does
  not force a split (no over-fragmenting - the body is one opening chunk);
- the body chunk is separated from the declarations by a blank, while the declaration chunk sits flush under the
  opening brace (no leading blank);
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


def _check_separates(name, provider, src, first_real, first_decl):
    lines = src.split("\n")
    t = provider(src, lines)[0]
    starts = [s for s, _ in t.segments]
    dl = next(i for i, l in enumerate(lines, 1) if first_decl in l)   # first declaration (opening chunk)
    fr = next(i for i, l in enumerate(lines, 1) if first_real in l)   # first real statement (body chunk)

    # Two distinct chunks: the declaration block, then the body.
    assert dl in starts and fr in starts and dl < fr, \
        f"[{name}] expected separate decl and body chunks: {starts}"

    out = _apply_edits(lines, [(s, "C", t.indent_of.get(s, "")) for s, _ in t.segments], SLASH_LINE_STYLE)
    assert code_preserved(lines, out, SLASH_LINE_STYLE), f"[{name}] code must be preserved"

    # The declaration chunk sits flush under the opening brace (no leading blank); the body chunk is blank-separated.
    di = next(i for i, l in enumerate(out) if first_decl in l)
    assert out[di - 1].lstrip().startswith("//") and out[di - 2].rstrip().endswith("{"), \
        f"[{name}] the declaration chunk should sit directly under the opening brace, got {out[di - 2]!r}"
    ri = next(i for i, l in enumerate(out) if first_real in l)
    assert out[ri - 1].lstrip().startswith("//") and out[ri - 2].strip() == "", \
        f"[{name}] the body chunk should be a fresh, blank-separated paragraph"


def main():
    assert SEG_MIN_LEADING_DECLS == 2
    _check_separates("c", c.iter_block_targets_c, C_MANY, "use(x);", "int x = a;")
    _check_separates("js", js.iter_block_targets_js, JS_MANY, "use(x);", "let x = a;")

    # A single leading declaration must NOT force a split: the body stays one opening chunk.
    t = c.iter_block_targets_c(C_ONE, C_ONE.split("\n"))[0]
    assert len(t.segments) == 1, f"a single leading declaration must not split the body: {t.segments}"

    print("PASS: a leading run of >=2 declarations forms its own chunk, body separated (C/JS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
