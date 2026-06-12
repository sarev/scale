#!/usr/bin/env python3
"""
JS def-pass snippet assembly with MULTIPLE direct children: `assemble_snippet_for_js` walks the body with a forward
cursor, so the children must be spliced in ascending source order. A past bug sorted them descending (a leftover from
an in-place bottom-to-top splice idiom), which made the first gap swallow every earlier child's raw body, emitted the
stubs out of order, and re-appended the tail after the cursor jumped backwards - duplicating code in the model's view.

Guards: each child is replaced by its stub (doc + header, no body), stubs appear in source order, and every parent
body line appears exactly once. No GGUF model required: only the tree-sitter provider and the assembler run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_javascript import _parse_js, iter_defs_with_info_js, assemble_snippet_for_js  # noqa: E402

SRC = (
    "function outer(a) {\n"        # 1
    "  const x = a + 1;\n"         # 2
    "  function g(y) {\n"          # 3  first child
    "    return y * 2;\n"          # 4
    "  }\n"                        # 5
    "  const mid = g(x);\n"        # 6  gap BETWEEN the children
    "  function h(z) {\n"          # 7  second child
    "    return z - 1;\n"          # 8
    "  }\n"                        # 9
    "  return mid + h(x);\n"       # 10 tail after the last child
    "}\n"                          # 11
)


def main():
    lines = SRC.split("\n")
    tree, source_bytes = _parse_js(SRC)
    defs = iter_defs_with_info_js(tree, source_bytes)
    info_by_id = {id(d.node): d for d in defs}
    by_qualname = {d.qualname: d for d in defs}

    assert {"outer", "outer.g", "outer.h"} <= set(by_qualname), f"unexpected defs: {list(by_qualname)}"
    outer = by_qualname["outer"]
    assert len(outer.children_ids) == 2, "the fixture needs two direct children to exercise the splice order"

    docs_by_id = {
        id(by_qualname["outer.g"].node): "Doubles y.",
        id(by_qualname["outer.h"].node): "Decrements z.",
    }
    snippet = assemble_snippet_for_js(info_by_id, lines, id(outer.node), docs_by_id)

    # Children are stubs: doc + header only, bodies elided.
    assert "return y * 2" not in snippet and "return z - 1" not in snippet, \
        f"child bodies must be replaced by stubs:\n{snippet}"
    assert "Doubles y." in snippet and "Decrements z." in snippet, f"child docs must ride the stubs:\n{snippet}"

    # Stubs and gaps appear in source order.
    assert snippet.index("Doubles y.") < snippet.index("function g(") \
        < snippet.index("const mid") < snippet.index("Decrements z.") \
        < snippet.index("function h(") < snippet.index("return mid + h(x);"), \
        f"body must be spliced in source order:\n{snippet}"

    # Every parent body line appears exactly once (the descending-sort bug duplicated the tail).
    for needle in ("const x = a + 1;", "const mid = g(x);", "return mid + h(x);", "function g(y) {", "function h(z) {"):
        assert snippet.count(needle) == 1, f"{needle!r} must appear exactly once:\n{snippet}"

    print("PASS: JS snippet assembly splices multiple child stubs in source order, no duplicated or raw child code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
