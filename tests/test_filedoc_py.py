#!/usr/bin/env python3
"""
File-doc for Python targets the MODULE DOCSTRING (a string literal, not a `#` comment). End-to-end via a fake LLM
(description prose from a stub summary_provider, classify turn only):

- an existing module-docstring description is replaced in place while licence lines inside the docstring stay put and
  the existing text is handed to the provider as a seed;
- a docstring that is all licence (no description) has a description appended (the licence-line misclassification is
  vetoed -> no seed);
- a file with no module docstring gets a fresh one inserted as the first statement;
- the module's code is unchanged in every case (verified via the AST-minus-module-docstring signature);
- the parse-based guard `_py_doc_preserved` rejects a code change or a description that would break the docstring.
"""
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_filedoc import annotate_file_doc  # noqa: E402
from scale_python import file_doc_target_py, _py_doc_preserved, _module_code_signature  # noqa: E402


@dataclass
class _Cfg:
    temperature: float = 0.2
    max_new_tokens: int = 320


class _FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, messages, cfg=None):
        return self.replies.pop(0)


def _run(src, classify_replies, description):
    lines = src.split("\n")
    target = file_doc_target_py(src, lines)
    seen = {}

    def provider(seed):
        seen["seed"] = seed
        return description

    out = annotate_file_doc(_FakeLLM(classify_replies), _Cfg(), [], lines, target, provider, "python")
    text = "\n".join(out)
    assert _module_code_signature(ast.parse(src)) == _module_code_signature(ast.parse(text)), \
        "module code (AST minus the module docstring) must be unchanged"
    return out, text, seen.get("seed")


REPLACE = (
    '"""\n'
    "Copyright 2025 Acme.\n"
    "Licensed under the MIT license.\n"
    "\n"
    "Reads and caches the process environment for the rest of the package.\n"
    '"""\n'
    "import os\n"
    "\n"
    "VALUE = os.getpid()\n"
)


def test_replace_updates_description_keeps_licence():
    out, text, seed = _run(REPLACE, ["3"], "Refreshed module description.")
    assert "Copyright 2025 Acme." in out and "Licensed under the MIT license." in out, "licence preserved verbatim"
    assert "Refreshed module description." in text and "Reads and caches the process environment" not in text
    assert seed == "Reads and caches the process environment for the rest of the package.", \
        f"existing description must seed the provider, got {seed!r}"
    assert ast.get_docstring(ast.parse(text)).strip().startswith("Copyright"), "still a valid module docstring"


LICENCE_ONLY = (
    '"""\n'
    "Copyright 2025 Acme.\n"
    "Licensed under the Apache License, Version 2.0.\n"
    '"""\n'
    "import os\n"
)


def test_licence_only_appends_and_vetoes():
    out, text, seed = _run(LICENCE_ONLY, ["1-2"], "Helpers for talking to the OS.")
    assert seed is None, "a vetoed (legal) classification must not seed the provider"
    assert "Copyright 2025 Acme." in out, "licence preserved"
    assert "Helpers for talking to the OS." in text, "a description was appended into the docstring"
    assert "Helpers for talking to the OS." in (ast.get_docstring(ast.parse(text)) or ""), "still in the docstring"


FRESH = (
    "import os\n"
    "\n"
    "def f():\n"
    "    return os.getpid()\n"
)


def test_fresh_inserts_module_docstring():
    out, text, seed = _run(FRESH, [], "Thin wrapper around the OS PID.")
    assert seed is None and out[0] == '"""' and out[2] == '"""', f"a fresh module docstring must open the file: {out[:3]}"
    assert ast.get_docstring(ast.parse(text)) == "Thin wrapper around the OS PID.", "it is the module docstring"
    assert out[3] == "" and out[4] == "import os", "a blank line then the unchanged code"


def test_guard_rejects_code_change_and_broken_docstring():
    old = ['"""', "Old.", '"""', "x = 1"]
    assert _py_doc_preserved(old, ['"""', "New.", '"""', "x = 1"], start=1, removed=1, added=1)
    # A changed code line shows up in the byte-identical suffix check.
    assert not _py_doc_preserved(old, ['"""', "New.", '"""', "x = 2"], start=1, removed=1, added=1)
    # A description carrying a triple-quote would break out of the docstring -> rejected.
    assert not _py_doc_preserved(old, ['"""', 'New """ oops.', '"""', "x = 1"], start=1, removed=1, added=1)


def main():
    test_replace_updates_description_keeps_licence()
    test_licence_only_appends_and_vetoes()
    test_fresh_inserts_module_docstring()
    test_guard_rejects_code_change_and_broken_docstring()
    print("PASS: Python file-doc updates/appends/inserts the module docstring, preserves licence + code, guards splices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
