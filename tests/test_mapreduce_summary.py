#!/usr/bin/env python3
"""
File summarisation must use a single pass when the file fits, and fall back to chunked map-reduce
(map each chunk, then reduce the partials) when it does not. The large-file path ends in one extra
"shaping" turn that rewrites the reduced overall summary to the file-description spec.

Exercises scale._generate_file_summary directly with a stubbed LLM and a small context window, so no
GGUF model is required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale import _generate_file_summary  # noqa: E402
from scale_llm import GenerationConfig  # noqa: E402

BASE = [{"role": "system", "content": "sys"}, {"role": "assistant", "content": "OK"}]


class FakeLLM:
    """Stub model whose reply depends on which prompt it sees, so we can tell map/reduce apart."""
    def __init__(self, n_ctx):
        self.n_ctx = n_ctx
        self.ctx_margin = 0
        self.calls = 0

    def estimate_tokens(self, text):
        return len(text) // 4

    def count_tokens(self, messages):
        return sum(self.estimate_tokens(m["content"]) for m in messages)

    def generate(self, messages, *, cfg=None, stop=None):
        self.calls += 1
        content = messages[-1]["content"]
        # The summarise() prompt names its subject; map/reduce/shape/one-shot use distinct, unique words for it.
        if "draft overview" in content:
            return "SHAPED_SUMMARY"          # the final description-shaping turn (large-file path)
        if "consecutive parts" in content:
            return "REDUCED_SUMMARY"
        if "chunk" in content:
            return f"partial-{self.calls}"
        return "ONESHOT_SUMMARY"


def main():
    cfg = GenerationConfig()

    # 1. Small file + roomy context -> single pass.
    small = "def f(x):\n    return x + 1\n"
    llm = FakeLLM(n_ctx=100_000)
    summary = _generate_file_summary(llm, cfg, BASE, small, "python")
    assert summary == "ONESHOT_SUMMARY", summary
    assert llm.calls == 1, f"one-pass should make exactly one call, made {llm.calls}"

    # 2. Large file + small context -> map-reduce, then a final description-shaping turn.
    big = "\n".join(f"def func_{i}(x):\n    return x + {i}" for i in range(200))  # well over the budget
    llm = FakeLLM(n_ctx=2048)  # limit = 2048 - 0 - SUMMARY_MAX_TOKENS(1024) = 1024 tokens
    summary = _generate_file_summary(llm, cfg, BASE, big, "python")
    assert summary == "SHAPED_SUMMARY", summary
    assert llm.calls >= 4, f"map-reduce + shaping should make multiple calls (>=2 map + reduce + shape), made {llm.calls}"

    print("PASS: one-pass for small files, map-reduce + shaping turn for large files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
