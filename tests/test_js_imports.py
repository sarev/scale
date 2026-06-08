#!/usr/bin/env python3
"""
JavaScript import/require discovery: CommonJS require() module names (including
destructured forms) must be parsed via the grammar's 'arguments' field.

Guards the fragile `init.child(i) for i in range(named_child_count)` traversal
that previously only worked by luck.

No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_javascript as j  # noqa: E402


def main():
    src = "const fs = require('fs');\nconst { join, resolve } = require('path');\n"
    tree, sb = j._parse_js(src)
    text = "\n".join(t for _, t in j._collect_imports_js(tree, sb))

    assert "fs" in text, text
    assert "path" in text, text
    assert "Requires" in text, text
    assert "join" in text and "resolve" in text, "destructured names should be listed"

    print("PASS: require() modules (incl. destructured) detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
