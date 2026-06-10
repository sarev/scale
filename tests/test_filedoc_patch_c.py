#!/usr/bin/env python3
"""
End-to-end file-doc patching (model-free, via a fake LLM driving the real `scale_c.file_doc_target_c` +
`scale_filedoc.annotate_file_doc`):

- an existing description is replaced in place while the copyright/license block above it stays byte-for-byte;
- a header that is all license (no description) gets a description appended into its block - the license is untouched,
  the model's "this is the description" misclassification of the license lines is vetoed;
- a file that opens with code gets a fresh `/* */` header block at the top;
- the preservation guard turns any code-altering splice into a no-op.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_c as c  # noqa: E402
from scale_filedoc import annotate_file_doc, file_doc_preserved  # noqa: E402
from scale_blocks import SLASH_BLOCK_STYLE, code_preserved  # noqa: E402


@dataclass
class _Cfg:
    temperature: float = 0.2
    max_new_tokens: int = 320


class _FakeLLM:
    """Returns queued replies in order (classify turn first when eligible lines exist, then the generate turn)."""
    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, messages, cfg=None):
        return self.replies.pop(0)


def _run(src, replies):
    lines = src.split("\n")
    target = c.file_doc_target_c(src, lines)
    out = annotate_file_doc(_FakeLLM(replies), _Cfg(), [], lines, target, "Overview of the file.", "c")
    assert code_preserved(lines, out, SLASH_BLOCK_STYLE), "executable code must be preserved"
    return lines, out


REPLACE = (
    "/*\n"
    " * Copyright 2025 Acme.\n"
    " * Licensed under the MIT license.\n"
    " */\n"
    "\n"
    "// Old description line one.\n"
    "// Old description line two.\n"
    "\n"
    "#include <stdio.h>\n"
    "int main(void) { return 0; }\n"
)


def test_replace_preserves_license_block():
    _, out = _run(REPLACE, ["3-4", "Refined summary of what this file does."])
    text = "\n".join(out)
    assert " * Copyright 2025 Acme." in out, "copyright line preserved verbatim"
    assert " * Licensed under the MIT license." in out, "license line preserved verbatim"
    assert "Refined summary of what this file does." in text, "new description inserted"
    assert "Old description line one." not in text and "Old description line two." not in text, \
        "the old description lines were replaced"
    assert out.count("// Refined summary of what this file does.") == 1


LICENSE_ONLY = (
    "/*\n"
    " * Copyright 2025 Acme.\n"
    " * Licensed under the Apache License, Version 2.0.\n"
    " */\n"
    "#include <stdio.h>\n"
    "int f(void) { return 1; }\n"
)


def test_license_only_appends_description_and_vetoes_legal():
    # The model is told the two license lines ARE the description ("1-2"); the veto must reject that and append instead.
    _, out = _run(LICENSE_ONLY, ["1-2", "Holds the f helper used across the program."])
    text = "\n".join(out)
    assert " * Copyright 2025 Acme." in out and " * Licensed under the Apache License, Version 2.0." in out, \
        "license is preserved despite being misclassified as the description"
    assert "Holds the f helper used across the program." in text, "a description was appended"
    close = out.index(" */")
    desc = next(i for i, l in enumerate(out) if "Holds the f helper" in l)
    assert desc < close, "the appended description sits inside the block, before the closing */"


CODE_FIRST = (
    "#include <stdio.h>\n"
    "int main(void) { return 0; }\n"
)


def test_code_first_inserts_fresh_block():
    _, out = _run(CODE_FIRST, ["Minimal program entry point."])
    assert out[0] == "/*" and out[2] == " */", f"a fresh block must open the file: {out[:3]}"
    assert out[1] == " * Minimal program entry point."
    assert out[3] == "", "a blank line separates the header from the code"
    assert out[4] == "#include <stdio.h>", "code follows, unmoved"


def test_guard_rejects_code_altering_splice():
    old = ["// a comment", "int x = 1;", "int y = 2;"]
    style = SLASH_BLOCK_STYLE
    # A clean comment-only replacement at line 0 is allowed.
    good = ["// new comment", "int x = 1;", "int y = 2;"]
    assert file_doc_preserved(old, good, start=0, removed=1, added=1, style=style)
    # Removing a code line (start=1) must be rejected.
    bad = ["// a comment", "// sneaky", "int y = 2;"]
    assert not file_doc_preserved(old, bad, start=1, removed=1, added=1, style=style)


def main():
    test_replace_preserves_license_block()
    test_license_only_appends_description_and_vetoes_legal()
    test_code_first_inserts_fresh_block()
    test_guard_rejects_code_altering_splice()
    print("PASS: file-doc replaces/appends/inserts descriptions, preserves license + code, and guards splices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
