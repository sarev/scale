#!/usr/bin/env python3
"""
`scale_c.file_doc_target_c` gathers the whole leading-comment zone of a C file - which may span several contiguous
blocks (a `/* */` block, a blank, then a run of `//`), with no intervening code. It marks the pure-content comment
lines as description-eligible (excluding delimiters, blank continuations, and single-line `/* ... */`), stops the scan
at the first code/preprocessor line, and reports no zone for a file that opens with code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_c as c  # noqa: E402

MULTI = (
    "/*\n"                                    # 1
    " * Copyright 2025 Acme.\n"               # 2  eligible (legal, but still eligible -> veto is the engine's job)
    " * Licensed under the MIT license.\n"    # 3  eligible
    " */\n"                                   # 4  close delimiter (not eligible)
    "\n"                                      # 5  blank between blocks
    "// fileutil.c\n"                         # 6  eligible
    "// Reads and writes the workspace file.\n"  # 7  eligible
    "\n"                                      # 8  blank
    "#include <stdio.h>\n"                    # 9  first code line -> zone ends
    "int main(void) { return 0; }\n"          # 10
)

CODE_FIRST = (
    "#include <stdio.h>\n"
    "int main(void) { return 0; }\n"
)


def test_multi_block_zone():
    lines = MULTI.split("\n")
    t = c.file_doc_target_c(MULTI, lines)
    assert t is not None and t.has_zone
    inners = [inner for _, _, inner in t.eligible]
    assert inners == [
        "Copyright 2025 Acme.",
        "Licensed under the MIT license.",
        "fileutil.c",
        "Reads and writes the workspace file.",
    ], inners
    # The block-continuation prefix is reconstructed as " * ", the line-comment one as "// ".
    prefixes = [pre for _, pre, _ in t.eligible]
    assert prefixes[0] == " * " and prefixes[2] == "// ", prefixes
    # Eligible lines carry their 1-based source line numbers; the scan stopped before the #include (line 9).
    linenos = [ln for ln, _, _ in t.eligible]
    assert linenos == [2, 3, 6, 7], linenos
    assert not t.insert_fresh, "an existing zone is appended into, not freshly inserted"


def test_no_zone_when_code_first():
    lines = CODE_FIRST.split("\n")
    t = c.file_doc_target_c(CODE_FIRST, lines)
    assert t is not None and not t.has_zone
    assert t.eligible == [] and t.insert_fresh and t.insert_index == 0


def main():
    test_multi_block_zone()
    test_no_zone_when_code_first()
    print("PASS: file_doc_target_c gathers a multi-block leading zone and detects a code-first file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
