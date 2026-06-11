#!/usr/bin/env python3
"""
The header-reword manifest: a run-level, prose-only escalation of the file descriptions.

This guards (model-free):
- `apply_reword` locates the draft in the header zone by exact (whitespace-collapsed) match and replaces it with the
  reworded answer, re-decorated in the block's own prefix, with the licence/copyright lines untouched byte-for-byte,
- a draft that is NOT in the file (edited since emit) is a safe no-op, as is a "NONE" answer,
- the legal veto refuses a matched range that smells legal, and the preservation guard rejects a code-touching splice,
- Python's module-docstring zone rewords through its parse-based guard,
- `unfilled_rewords` counts empty answers ("NONE" is a deliberate keep and counts as filled),
- the manifest I/O round-trips and rejects a non-reword file.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_c import file_doc_target_c  # noqa: E402
from scale_python import file_doc_target_py  # noqa: E402
from scale_reword import (  # noqa: E402
    apply_reword, reword_manifest, read_reword_manifest, write_reword_manifest, unfilled_rewords,
)


C_SRC = [
    "/*",
    " * Copyright 2025 Example Corp. All rights reserved.",
    " *",
    " * Parses the configuration file and exposes its settings to the rest of the program.",
    " */",
    "",
    "#include <stdio.h>",
    "",
    "int load(void) { return 0; }",
]
C_DRAFT = "Parses the configuration file and exposes its settings to the rest of the program."


def _c_zone(lines):
    return file_doc_target_c("\n".join(lines), lines)


def test_c_reword_replaces_draft_keeps_licence():
    out, changed = apply_reword(C_SRC, _c_zone(C_SRC), C_DRAFT,
                                "Loads the interpreter's configuration, part of the wider BASIC project.")
    assert changed
    assert " * Copyright 2025 Example Corp. All rights reserved." in out, "the licence line survives byte-for-byte"
    assert any("wider BASIC project" in ln for ln in out)
    assert not any("exposes its settings" in ln for ln in out), "the draft is replaced"
    assert out[out.index("int load(void) { return 0; }")] == C_SRC[-1], "code untouched"


def test_miss_and_none_are_noops():
    out, changed = apply_reword(C_SRC, _c_zone(C_SRC), "A draft that is not in the file.", "New text.")
    assert not changed and out == C_SRC, "an unmatched draft must be a safe no-op"
    out, changed = apply_reword(C_SRC, _c_zone(C_SRC), C_DRAFT, "NONE")
    assert not changed and out == C_SRC, "a NONE answer keeps the draft"


def test_legal_veto():
    # If the recorded draft (wrongly) IS the licence line, the match is refused by the veto.
    out, changed = apply_reword(C_SRC, _c_zone(C_SRC),
                                "Copyright 2025 Example Corp. All rights reserved.", "New text.")
    assert not changed and out == C_SRC


PY_SRC = [
    '"""',
    "Reads the settings and applies them at startup.",
    '"""',
    "",
    "VALUE = 1",
]


def test_python_reword_through_parse_guard():
    zone = file_doc_target_py("\n".join(PY_SRC), PY_SRC)
    out, changed = apply_reword(PY_SRC, zone, "Reads the settings and applies them at startup.",
                                "Startup configuration loader for the demo app.")
    assert changed and any("Startup configuration loader" in ln for ln in out)
    assert "VALUE = 1" in out, "code untouched"
    # An answer that would break the docstring (embedded triple quote) is rejected by the parse-based guard.
    out2, changed2 = apply_reword(PY_SRC, zone, "Reads the settings and applies them at startup.",
                                  'Bad """ description.')
    assert not changed2 and out2 == PY_SRC


def test_manifest_io_and_completeness():
    entries = [
        {"path": "a.c", "language": "c", "role": "implementation", "draft": "d1", "context": None, "answer": None},
        {"path": "b.h", "language": "c", "role": "header", "draft": "d2", "context": None, "answer": "NONE"},
    ]
    m = reword_manifest("The project blurb.", entries)
    assert unfilled_rewords(m) == ["a.c"], "null is unfilled; NONE is a deliberate keep"

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "r.json"
        write_reword_manifest(p, m)
        again = read_reword_manifest(p)
        assert again["project_blurb"] == "The project blurb." and len(again["files"]) == 2

        # A non-reword manifest is rejected.
        bad = Path(tmp) / "bad.json"
        bad.write_text('{"version": 2, "tool": "scale", "requests": []}', encoding="utf-8")
        try:
            read_reword_manifest(bad)
            raise AssertionError("a function manifest must not read as a reword manifest")
        except ValueError:
            pass


def main():
    test_c_reword_replaces_draft_keeps_licence()
    test_miss_and_none_are_noops()
    test_legal_veto()
    test_python_reword_through_parse_guard()
    test_manifest_io_and_completeness()
    print("PASS: the header reword applies by exact draft match through the guards; misses and NONE are no-ops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
