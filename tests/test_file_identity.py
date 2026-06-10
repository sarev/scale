#!/usr/bin/env python3
"""
The file-identity context: the file's name (and, for C, whether it is a header or an implementation) is injected into
the priming context of the summary / definition / block passes, so a header's documentation reads as the public
contract while an implementation's reads as internal detail.

No GGUF model required: the LLM is stubbed and the summary cache is redirected to a temporary directory.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scale import _file_identity_note, prime_llm_for_comments, SummaryCache  # noqa: E402
from scale_llm import GenerationConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCALE_CFG = ROOT / "scale-cfg"

SRC = "int clamp(int v) { return v; }\n"


class FakeLLM:
    """Stub model: canned replies, large context window, never loads a GGUF."""
    n_ctx = 100_000
    ctx_margin = 0

    def estimate_tokens(self, text):
        return len(text) // 4

    def count_tokens(self, messages):
        return sum(self.estimate_tokens(m["content"]) for m in messages)

    def generate(self, messages, *, cfg=None, stop=None):
        if "source file" in messages[-1]["content"]:
            return "A summary."
        return "OK"


def test_note_classification():
    assert "header file" in _file_identity_note(Path("h/program-fns.h"), "c")
    assert "external contract" in _file_identity_note(Path("inc/foo.HPP"), "c")     # case-insensitive suffix
    assert "implementation file" in _file_identity_note(Path("c/program.c"), "c")
    assert "header file" not in _file_identity_note(Path("c/program.c"), "c")
    plain = _file_identity_note(Path("a/b/foo.py"), "python")
    assert "`foo.py`" in plain and "header" not in plain


def _identity_turn(messages):
    return next((m for m in messages
                 if m["role"] == "user" and "The file being documented is" in m["content"]), None)


def test_priming_injects_header_identity():
    with tempfile.TemporaryDirectory() as tmp:
        SummaryCache._CACHE_DIR = Path(tmp)
        SummaryCache._CACHE_INDEX = Path(tmp) / "index.pkl"
        messages = prime_llm_for_comments(
            FakeLLM(), GenerationConfig(), SCALE_CFG, Path("h/program-fns.h"),
            source_blob=SRC, language="c", no_cache=True)
    turn = _identity_turn(messages)
    assert turn is not None, "no file-identity turn was injected"
    assert "header file" in turn["content"]
    # The identity precedes the file-overview turn, so the summary and routine turns both see it.
    names = [m["content"] for m in messages if m["role"] == "user"]
    id_idx = next(i for i, c in enumerate(names) if "The file being documented is" in c)
    ov_idx = next(i for i, c in enumerate(names) if "here is an overview" in c)
    assert id_idx < ov_idx, "identity must come before the overview turn"


def test_priming_injects_impl_identity():
    with tempfile.TemporaryDirectory() as tmp:
        SummaryCache._CACHE_DIR = Path(tmp)
        SummaryCache._CACHE_INDEX = Path(tmp) / "index.pkl"
        messages = prime_llm_for_comments(
            FakeLLM(), GenerationConfig(), SCALE_CFG, Path("c/program.c"),
            source_blob=SRC, language="c", no_cache=True, template="blocks")
    turn = _identity_turn(messages)
    assert turn is not None and "implementation file" in turn["content"]


def main():
    test_note_classification()
    test_priming_injects_header_identity()
    test_priming_injects_impl_identity()
    print("PASS: file name + header/implementation role is injected into the summary/definition/block priming context")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
