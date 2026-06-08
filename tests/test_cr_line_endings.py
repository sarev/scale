#!/usr/bin/env python3
"""
Bare '\\r' (old-Mac) line endings must still map node rows to the right lines in
the C and JS workers (tree-sitter counts rows by '\\n' only).

No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_c as c           # noqa: E402
import scale_javascript as j  # noqa: E402


def main():
    c_lines = ["int first(void) {", "    return 1;", "}", "", "int second(void) {", "    return 2;", "}"]
    tree, sb = c._parse_c("\r".join(c_lines))
    cnames = {d.qualname: d.start for d in c.iter_defs_with_info_c(tree, sb)}
    assert cnames.get("first") == 1, cnames
    assert cnames.get("second") == 5, cnames  # would be 1 if rows collapsed under '\r'

    js_lines = ["function alpha() {", "  return 1;", "}", "", "function beta() {", "  return 2;", "}"]
    jtree, jsb = j._parse_js("\r".join(js_lines))
    jnames = {d.qualname: d.start for d in j.iter_defs_with_info_js(jtree, jsb)}
    assert jnames.get("alpha") == 1, jnames
    assert jnames.get("beta") == 5, jnames

    print("PASS: bare \\r line endings map to correct line numbers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
