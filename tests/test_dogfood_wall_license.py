#!/usr/bin/env python3
"""
The dogfood wall must keep legal boilerplate in the module docstring while stripping the description below it.

Guards a real incident: the SCALE sources carry their Apache license *inside* the module docstring, and `wall()`
used to strip the whole docstring - so the dogfood run both lost the license from its output and never exercised
the file-doc pass's license-preservation path. The wall now strips only the lines below the docstring's last
legal-looking line (per `looks_legal`), keeping the boilerplate and the closing quotes; docstrings with no legal
text, and all function/class docstrings, are still stripped whole.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dogfood import wall  # noqa: E402

LICENSED = '''#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
with the License.

This module does interesting things, described at length here.
And on a second line.
"""

# a module-scope comment


def f():
    """A docstring to strip."""
    return 1
'''

PLAIN = '''"""
Just a description, no legal text at all.
"""


def f():
    return 1
'''


def main():
    walled = wall(LICENSED)
    ast.parse(walled)  # must still be valid Python
    assert "Copyright 2025 7th software Ltd." in walled, "copyright line lost"
    assert "Licensed under the Apache License" in walled, "license body lost"
    assert "interesting things" not in walled, "description not stripped"
    assert "And on a second line." not in walled, "description tail not stripped"
    assert "A docstring to strip." not in walled, "function docstring kept"
    assert "module-scope comment" not in walled, "full-line comment kept"
    # The truncated docstring stays well-formed: license end followed by the closing quotes.
    assert 'with the License.\n"""' in walled, "closing quotes not directly after the legal block"

    # No legal text -> the whole module docstring is stripped, as before.
    walled_plain = wall(PLAIN)
    ast.parse(walled_plain)
    assert "Just a description" not in walled_plain, "legal-free module docstring kept"

    print("PASS: the dogfood wall keeps module-docstring legal boilerplate and strips only the description")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
