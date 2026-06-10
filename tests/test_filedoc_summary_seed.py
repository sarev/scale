#!/usr/bin/env python3
"""
The whole-file summary is now written to a file-DESCRIPTION spec, and when an existing file description is supplied as
a seed it is woven into the generation prompt (so the unified summary incorporates the author's wording). This guards,
model-free (a stubbed LLM that just captures the prompt it is asked to summarise against):

- the description instruction (not the generic "note significant internal details" one) drives the summary turn;
- a provided seed appears in the prompt, introduced by the incorporate-the-existing-description clause;
- with no seed, that clause is absent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale import _generate_file_summary  # noqa: E402
from scale_llm import GenerationConfig  # noqa: E402

BASE = [{"role": "system", "content": "sys"}]
SRC = "int x;\n"
SEED_CLAUSE = "The file already carries this description"


class FakeLLM:
    """Captures each prompt; large context window so the single-pass description path is taken."""
    n_ctx = 100_000
    ctx_margin = 0

    def __init__(self):
        self.prompts = []

    def estimate_tokens(self, text):
        return len(text) // 4

    def count_tokens(self, messages):
        return sum(self.estimate_tokens(m["content"]) for m in messages)

    def generate(self, messages, *, cfg=None, stop=None):
        self.prompts.append(messages[-1]["content"])
        return "A file description."


def main():
    # With a seed: the existing description and the incorporate clause are in the prompt.
    llm = FakeLLM()
    _generate_file_summary(llm, GenerationConfig(), BASE, SRC, "c", seed="Parses the widget config.")
    prompt = llm.prompts[-1]
    assert "Lead with the file's role" in prompt, "the description spec (not the generic summary) must drive the turn"
    assert "Parses the widget config." in prompt, "the seed (existing description) must reach the prompt"
    assert SEED_CLAUSE in prompt, "the incorporate-the-existing-description clause must be present"

    # Without a seed: no incorporate clause.
    llm2 = FakeLLM()
    _generate_file_summary(llm2, GenerationConfig(), BASE, SRC, "c")
    assert SEED_CLAUSE not in llm2.prompts[-1], "no seed clause when no existing description is supplied"

    print("PASS: the summary uses the description spec and weaves in an existing-description seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
