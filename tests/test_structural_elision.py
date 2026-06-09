#!/usr/bin/env python3
"""
Structural elision of oversized routine snippets, and the shared summarise() primitive it is built on.

- summarise() caps the reply to the requested length and weaves the subject + text into the prompt.
- elide_structurally() leaves a snippet that already fits untouched; on a tight budget it collapses the DEEPEST body
  suite first (keeping its controlling header) into a `...  # <summary>` line, preserving the routine's shape and the
  shallower statements rather than cropping them away; the result stays valid Python and within budget.
- When the snippet cannot be parsed (e.g. non-Python), it falls back to the crude head/tail crop so the budget is still
  respected.

No GGUF model required.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_text import summarise, LENGTH_LINE, LENGTH_PARAGRAPH, LENGTH_PARAGRAPHS, MARKER_PYTHON  # noqa: E402
from scale_python import elide_structurally  # noqa: E402
from scale_llm import GenerationConfig  # noqa: E402


class Stub:
    """Stub model: token estimate = len/4, a settable snippet budget, and a canned one-line summary."""
    n_ctx = 4096
    ctx_margin = 0

    def __init__(self, budget=10_000):
        self._budget = budget
        self.last_max_tokens = None
        self.last_prompt = None

    def estimate_tokens(self, text):
        return len(text) // 4

    def count_tokens(self, messages):
        return sum(self.estimate_tokens(m["content"]) for m in messages)

    def snippet_budget(self, messages, cfg, **kwargs):
        return self._budget

    def generate(self, messages, cfg=None, stop=None):
        self.last_max_tokens = cfg.max_new_tokens if cfg else None
        self.last_prompt = messages[-1]["content"]
        return "  validate and accumulate the parts  "   # padded, to check stripping


DEEP = (
    "def f(data):\n"
    "    setup = init()\n"
    "    for row in data:\n"
    "        validate(row)\n"
    "        if row.ok:\n"
    "            for cell in row.cells:\n"
    "                a = cell.x + cell.y + cell.z\n"
    "                b = transform(a, factor, offset)\n"
    "                c = b * 2 - adjustment_value\n"
    "                d = clamp(c, lower_bound, upper_bound)\n"
    "                accumulate(d, into=running_total)\n"
    "                store(d, key=cell.identifier)\n"
    "    return setup\n"
)


def main():
    cfg = GenerationConfig(max_new_tokens=4096)
    msgs = [{"role": "system", "content": "system"}]

    # ---- 1. summarise(): per-length reply cap + subject/text woven into the prompt ----
    s = Stub()
    out = summarise(s, cfg, "x = 1", LENGTH_LINE, subject="a tiny snippet")
    assert out == "validate and accumulate the parts", "summary must be stripped"
    assert s.last_max_tokens == 64, f"one-line cap should be 64, got {s.last_max_tokens}"
    assert "a tiny snippet" in s.last_prompt and "x = 1" in s.last_prompt, "subject and text must be in the prompt"

    summarise(s, cfg, "y", LENGTH_PARAGRAPH)
    assert s.last_max_tokens == 200, f"paragraph cap should be 200, got {s.last_max_tokens}"
    summarise(s, cfg, "y", LENGTH_PARAGRAPHS)
    assert s.last_max_tokens == 512, f"paragraphs cap should be 512, got {s.last_max_tokens}"
    summarise(s, cfg, "y", LENGTH_LINE, max_tokens=999)
    assert s.last_max_tokens == 999, "an explicit max_tokens must override the per-length default"

    # ---- 2. Snippet that already fits is returned untouched ----
    s = Stub(budget=10_000)
    out, omitted = elide_structurally(s, cfg, msgs, DEEP, header_line_count=1)
    assert out == DEEP and omitted == 0, "a snippet within budget must be returned unchanged"

    # ---- 3. Tight budget: deepest block collapses first; skeleton and headers survive ----
    s = Stub(budget=60)
    out, omitted = elide_structurally(s, cfg, msgs, DEEP, header_line_count=1)
    ast.parse(out)                                              # still valid Python
    assert s.estimate_tokens(out) <= 60, "result must be within budget"
    assert omitted > 0, "a too-large snippet must have lines elided"
    assert "...  # validate and accumulate the parts" in out, "the deepest block becomes a summarised placeholder"
    assert "setup = init()" in out, "a shallow statement must survive (not a blanket crop)"
    assert "return setup" in out, "the tail of the routine must survive"
    assert "for cell in row.cells:" in out, "the controlling header of the collapsed block is kept"
    assert "store(d, key=cell.identifier)" not in out, "the deep body lines must be gone"
    assert MARKER_PYTHON.split("{")[0] not in out, "structural collapse alone should not need the crude crop marker here"

    # ---- 4. Unparseable snippet falls back to the crude crop (budget still respected) ----
    s = Stub(budget=20)
    junk = "\n".join(f"}}{{ not python line {i} ;;;" for i in range(40))
    out, omitted = elide_structurally(s, cfg, msgs, junk, header_line_count=1, marker=MARKER_PYTHON)
    assert omitted > 0, "the crop fallback must elide the over-budget junk"
    assert "lines omitted for brevity" in out, "the fallback must insert the crude crop marker"

    print("PASS: summarise caps replies; structural elision collapses deepest-first and falls back to the crop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
