#!/usr/bin/env python3
"""
File-doc for JavaScript reuses the shared brace-language scanner (validated thoroughly for C), so this is a light
wiring + patch check: a description inside a leading `/* ... */` header block is replaced while a licence line is
preserved, and a file that opens with code gets a fresh header block. Code is preserved in both.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_javascript as js  # noqa: E402
from scale_filedoc import annotate_file_doc  # noqa: E402
from scale_blocks import SLASH_BLOCK_STYLE, code_preserved  # noqa: E402


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
    target = js.file_doc_target_js(src, lines)
    out = annotate_file_doc(_FakeLLM(classify_replies), _Cfg(), [], lines, target, lambda seed: description, "js")
    assert code_preserved(lines, out, SLASH_BLOCK_STYLE), "executable code must be preserved"
    return out


REPLACE = (
    "/*\n"
    " * Copyright 2025 Acme.\n"
    " * Wraps the filesystem module for the rest of the service.\n"
    " */\n"
    "const x = require('fs');\n"
    "function f() { return x; }\n"
)


def test_replace_keeps_licence():
    out = _run(REPLACE, ["2"], "Reads files via the fs module.")
    text = "\n".join(out)
    assert " * Copyright 2025 Acme." in out, "copyright preserved verbatim"
    assert "Reads files via the fs module." in text and "Wraps the filesystem module" not in text


CODE_FIRST = (
    "const x = require('fs');\n"
    "function f() { return x; }\n"
)


def test_code_first_inserts_fresh_block():
    out = _run(CODE_FIRST, [], "Entry module wiring up the fs helpers.")
    assert out[0] == "/*" and out[2] == " */", f"a fresh block must open the file: {out[:3]}"
    assert out[1] == " * Entry module wiring up the fs helpers."
    assert out[3] == "" and out[4] == "const x = require('fs');", "code follows, unmoved"


def main():
    test_replace_keeps_licence()
    test_code_first_inserts_fresh_block()
    print("PASS: JS file-doc replaces/inserts the header description and preserves code (shared brace scanner)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
