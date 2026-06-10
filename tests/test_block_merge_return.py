#!/usr/bin/env python3
"""
A suite that is exactly `[simple_stmt, return]` must form a single paragraph anchored at the leading statement,
with no blank line above it (it sits at the start of an indent, right after the `{`/`:`). This guards that, for
Python, C, and JS:

- the leading statement is the chunk start and the `return` does NOT get its own break (the comment pass therefore
  sees both lines, not a bare `return`);
- the patcher inserts no blank between the block opener and the merged comment, and none between the statement and
  the return;
- a trailing `return` whose scope has *more* than two statements still gets its own paragraph (the merge is
  specific to the two-statement case);
- code is preserved.

No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_blocks import _apply_edits, code_preserved, PYTHON_STYLE, SLASH_LINE_STYLE  # noqa: E402
import scale_python as py  # noqa: E402
import scale_c as c  # noqa: E402
import scale_javascript as js  # noqa: E402

# (source, provider, style, opener, anchor, ret, trailing_return) per language.
PY_SRC = (
    "def f(a):\n"               # 1
    "    x = a\n"               # 2
    "    if a > 0:\n"           # 3  opener
    "        y = compute(a)\n"  # 4  anchor
    "        return y\n"        # 5  merged return
    "    z = other()\n"         # 6
    "    w = more()\n"          # 7
    "    return w\n"            # 8  trailing return (body has many statements)
)
C_SRC = (
    "void f(int a) {\n"        # 1
    "    int x = a;\n"         # 2
    "    if (a > 0) {\n"       # 3  opener
    "        int y = g(a);\n"  # 4  anchor
    "        return y;\n"      # 5  merged return
    "    }\n"                  # 6
    "    int z = o();\n"       # 7
    "    int w = m();\n"       # 8
    "    return w;\n"          # 9  trailing return
    "}\n"
)
JS_SRC = (
    "function f(a) {\n"        # 1
    "  let x = a;\n"           # 2
    "  if (a > 0) {\n"         # 3  opener
    "    let y = g(a);\n"      # 4  anchor
    "    return y;\n"          # 5  merged return
    "  }\n"                    # 6
    "  let z = o();\n"         # 7
    "  let w = m();\n"         # 8
    "  return w;\n"            # 9  trailing return
    "}\n"
)


def _target(provider, src):
    lines = src.split("\n")
    t = [t for t in provider(src, lines) if t.qualname == "f"][0]
    return lines, t


def _check(name, provider, style, src, opener, anchor, ret, trailing):
    lines, t = _target(provider, src)
    starts = [s for s, _ in t.segments]

    a_line = next(i for i, l in enumerate(lines, 1) if anchor in l)
    r_line = next(i for i, l in enumerate(lines, 1) if ret in l)
    assert a_line in starts, f"[{name}] the merge anchor ({anchor!r}) must start a paragraph: {starts}"
    assert r_line not in starts, f"[{name}] the merged return ({ret!r}) must not get its own break: {starts}"

    out = _apply_edits(lines, [(s, "C", t.indent_of.get(s, "")) for s, _ in t.segments], style)
    assert code_preserved(lines, out, style), f"[{name}] code must be preserved"

    # No blank between the opener and the merged comment, nor between the statement and the return.
    oi = next(i for i, l in enumerate(out) if opener in l)
    assert out[oi + 1].lstrip().startswith(style.line_prefix.strip()), \
        f"[{name}] expected the merged comment directly under the opener, got {out[oi + 1]!r}"
    ai = next(i for i, l in enumerate(out) if anchor in l)
    assert out[ai + 1].strip().startswith(ret.strip()) or ret in out[ai + 1], \
        f"[{name}] the return must directly follow the anchor statement, got {out[ai + 1]!r}"

    # The trailing return (in a many-statement body) DOES get a blank line above its comment.
    ti = next(i for i, l in enumerate(out) if trailing in l)
    # walk up past its comment to find the blank
    above = out[ti - 1]
    assert above.lstrip().startswith(style.line_prefix.strip()), f"[{name}] trailing return should carry a comment"
    assert out[ti - 2].strip() == "", f"[{name}] trailing return's paragraph should have a blank line above it"


def main():
    _check("python", py.iter_block_targets, PYTHON_STYLE, PY_SRC,
           "if a > 0:", "y = compute(a)", "return y", "return w")
    _check("c", c.iter_block_targets_c, SLASH_LINE_STYLE, C_SRC,
           "if (a > 0) {", "int y = g(a);", "return y;", "return w;")
    _check("js", js.iter_block_targets_js, SLASH_LINE_STYLE, JS_SRC,
           "if (a > 0) {", "let y = g(a);", "return y;", "return w;")
    print("PASS: [stmt; return] suites merge into one anchored paragraph (no leading blank) across Python/C/JS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
