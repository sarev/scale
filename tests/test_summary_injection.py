#!/usr/bin/env python3
"""
Regression: the whole-file summary must be injected into the LLM context.

Guards the critical bug where prime_llm_for_comments interpolated a stale 'OK'
(the system-prompt acknowledgement) instead of the generated summary, which
silently disabled the entire "summarise the file for context" feature.

No GGUF model required: the LLM is stubbed (with a large context window so the
one-pass summary path is taken) and the summary cache is redirected to a
temporary directory.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scale import prime_llm_for_comments, SummaryCache  # noqa: E402
from scale_llm import GenerationConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCALE_CFG = ROOT / "scale-cfg"
SENTINEL = "REAL_SUMMARY_SENTINEL: parses config, doubles numbers, counts."

SRC = (
    "import { readFileSync } from 'fs';\n"
    "function parseConfig(path) { return JSON.parse(readFileSync(path, 'utf-8')); }\n"
    "const double = (x) => x * 2;\n"
)


class FakeLLM:
    """Stub model: canned replies, large context window, never loads a GGUF."""
    n_ctx = 100_000
    ctx_margin = 0

    def __init__(self):
        self.calls = 0

    def estimate_tokens(self, text):
        return len(text) // 4

    def count_tokens(self, messages):
        return sum(self.estimate_tokens(m["content"]) for m in messages)

    def generate(self, messages, *, cfg=None, stop=None):
        self.calls += 1
        if self.calls == 1:
            return "OK"          # system-prompt acknowledgement
        if self.calls == 2:
            return SENTINEL      # the generated whole-file summary (one-pass path)
        return "OK"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        # Redirect the summary cache so the test never touches the real __cache__.
        SummaryCache._CACHE_DIR = Path(tmp)
        SummaryCache._CACHE_INDEX = Path(tmp) / "index.pkl"

        messages = prime_llm_for_comments(
            FakeLLM(), GenerationConfig(), SCALE_CFG, Path("virtual.js"),
            source_blob=SRC, language="js", no_cache=True,
        )

    overview = next((m for m in messages
                     if m["role"] == "user" and "here is an overview" in m["content"]), None)
    assert overview is not None, "no overview priming message was produced"
    assert SENTINEL in overview["content"], "the generated summary was NOT injected into the LLM context"

    print("PASS: whole-file summary is injected into the LLM context")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
