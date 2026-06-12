#!/usr/bin/env python3
"""
The online file-description round (the scale-filedoc manifest), end to end and model-free:

- EMIT (`--emit-filedoc`): each target contributes its header zone's eligible lines (numbered for the range answer),
  its role, and its CURRENT skeleton - whole text for a file with no symbols (the binary guard's fallback).
- CHECK: the completeness counter needs BOTH answer halves (range + description); the explicit "NONE" description is
  a deliberate, filled decline.
- APPLY (`--apply-filedoc` / `apply_filedoc_entry`): a range answer replaces those lines re-wrapped in the host
  decoration; a "NONE" range inserts/appends a fresh description; the license veto and the entries-mismatch check
  make a bad answer a safe no-op; Python splices through the parse-based `_py_doc_preserved` guard.

No GGUF model required.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale import main  # noqa: E402
from scale_filedoc import (  # noqa: E402
    FILEDOC_TOOL, apply_filedoc_entry, read_filedoc_manifest, unfilled_descriptions, write_filedoc_manifest,
)
from scale_c import file_doc_target_c  # noqa: E402


C_SRC = (
    "/*\n"
    " * Copyright 2025 Example Corp. Licensed under the Apache License.\n"
    " *\n"
    " * Old terse description of the file.\n"
    " */\n"
    "\n"
    "#include <stdio.h>\n"
    "\n"
    "int helper(int x)\n"
    "{\n"
    "    return x + 1;\n"
    "}\n"
)

# No definitions at all: the skeleton falls back to the whole text, and the description is freshly inserted.
PY_SRC = (
    "import os\n"
    "\n"
    "LIMIT = 10\n"
    "PATHS = [os.sep]\n"
)


def _entry_for(manifest, name):
    return next(f for f in manifest["files"] if f["path"].endswith(name))


def test_emit_check_apply_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        c_path, py_path = root / "demo.c", root / "demo.py"
        c_path.write_text(C_SRC, encoding="utf-8", newline="\n")
        py_path.write_text(PY_SRC, encoding="utf-8", newline="\n")
        m_path = root / "filedoc.json"

        # ---- EMIT: zone entries + skeleton per file ----
        assert main(["--emit-filedoc", str(m_path), "--project-doc", "none", str(c_path), str(py_path)]) == 0
        manifest = read_filedoc_manifest(m_path)
        assert manifest["tool"] == FILEDOC_TOOL and manifest["description_spec"].strip()
        c_entry = _entry_for(manifest, "demo.c")
        assert c_entry["entries"] == ["Copyright 2025 Example Corp. Licensed under the Apache License.",
                                      "Old terse description of the file."], c_entry["entries"]
        assert c_entry["role"] == "implementation"
        assert "int helper(int x)" in c_entry["skeleton"], "the skeleton must carry the signatures"
        assert "return x + 1;" not in c_entry["skeleton"], "the skeleton must not carry bodies"
        py_entry = _entry_for(manifest, "demo.py")
        assert py_entry["skeleton"] == PY_SRC.rstrip("\n") or py_entry["skeleton"] == PY_SRC, \
            "a no-symbol file must ride whole (the skeleton fallback)"

        # ---- CHECK: both halves needed; a NONE description is a deliberate (filled) decline ----
        assert main(["--check-manifest", str(m_path)]) == 1
        assert sorted(unfilled_descriptions(manifest)) == sorted([str(c_path), str(py_path)])
        c_entry["answer"] = {"range": "2", "description": None}
        assert str(c_path) in unfilled_descriptions(manifest), "a missing description half stays unfilled"
        c_entry["answer"] = {"range": "2", "description": "Demonstration helpers for the smoke fixture."}
        py_entry["answer"] = {"range": "NONE", "description": "Constants shared by the demo scripts."}
        assert unfilled_descriptions(manifest) == []
        write_filedoc_manifest(m_path, manifest)
        assert main(["--check-manifest", str(m_path)]) == 0

        # ---- APPLY: range replace in host decoration (C); fresh module docstring (Python) ----
        assert main(["--apply-filedoc", str(m_path), str(c_path), str(py_path)]) == 0
        c_out = c_path.read_text(encoding="utf-8").split("\n")
        assert " * Demonstration helpers for the smoke fixture." in c_out, c_out
        assert " * Copyright 2025 Example Corp. Licensed under the Apache License." in c_out, \
            "the license line must survive byte-for-byte"
        assert " * Old terse description of the file." not in c_out, "the old description must be replaced"
        assert "int helper(int x)" in c_out and "    return x + 1;" in c_out, "code preserved"

        py_out = py_path.read_text(encoding="utf-8")
        assert py_out.startswith('"""\nConstants shared by the demo scripts.\n"""'), py_out
        assert py_out.endswith(PY_SRC), "the Python code must survive byte-for-byte below the new docstring"


def _zone(src):
    lines = src.split("\n")
    return lines, file_doc_target_c(src, lines)


def test_apply_entry_no_ops():
    lines, zone = _zone(C_SRC)
    entries = [inner for (_l, _p, inner) in zone.eligible]

    # Entries mismatch (the file changed since emit) -> safe no-op.
    stale = {"entries": ["Some other header"], "answer": {"range": "1", "description": "New prose."}}
    assert apply_filedoc_entry(lines, zone, stale) is None

    # A "NONE"/empty description -> deliberate decline, no-op.
    assert apply_filedoc_entry(lines, zone, {"entries": entries,
                                             "answer": {"range": "2", "description": "NONE"}}) is None
    assert apply_filedoc_entry(lines, zone, {"entries": entries,
                                             "answer": {"range": "2", "description": ""}}) is None

    # A range covering the license line -> the local veto refuses, regardless of the stronger model's classify.
    vetoed = apply_filedoc_entry(lines, zone, {"entries": entries,
                                               "answer": {"range": "1-2", "description": "New prose."}})
    assert vetoed is None, "the license veto must run locally at apply time"

    # The good path still works at this level: range 2 replaces only the description line.
    ok = apply_filedoc_entry(lines, zone, {"entries": entries,
                                           "answer": {"range": "2", "description": "New prose."}})
    assert ok is not None and " * New prose." in ok
    assert " * Copyright 2025 Example Corp. Licensed under the Apache License." in ok


def main_test():
    test_emit_check_apply_end_to_end()
    test_apply_entry_no_ops()
    print("PASS: filedoc manifest emit/check/apply - zone entries + skeleton ride out, both answer halves counted, "
          "range replace in host decoration, fresh Python docstring, veto/mismatch/NONE are safe no-ops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
