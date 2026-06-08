#!/usr/bin/env python3
"""
Two same-named definitions must receive distinct docstrings.

Guards the bug where doc maps keyed by qualname let same-named defs overwrite
each other, so the patcher stamped one comment onto both. Doc maps are now keyed
by node identity.

No GGUF model required.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_python import iter_defs_with_info, patch_docstrings_textually  # noqa: E402


def main():
    src = "def f():\n    return 1\n\n\ndef f():\n    return 2\n"
    lines = src.split("\n")
    defs = iter_defs_with_info(ast.parse(src))
    assert len(defs) == 2, f"expected 2 defs, got {len(defs)}"
    assert defs[0].qualname == defs[1].qualname == "f", "both should share qualname 'f'"

    doc_map = {id(defs[0].node): "FIRST f docstring", id(defs[1].node): "SECOND f docstring"}
    patched = "\n".join(patch_docstrings_textually(lines, defs, doc_map))

    assert "FIRST f docstring" in patched, "first docstring missing"
    assert "SECOND f docstring" in patched, "second docstring missing (qualname collision!)"

    print("PASS: same-named defs received distinct docstrings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
