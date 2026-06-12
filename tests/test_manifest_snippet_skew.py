#!/usr/bin/env python3
"""
Manifest snippet freshness across passes (the emit-side skew): when a routine with a NESTED def is escalated by BOTH
passes, the def pass records its span from the pre-patch source, but by block-pass time the nested def has gained a
locally generated docstring - so the block pass's chunk line ranges are computed against a LONGER span. A past bug
kept the def pass's stale snippet (first write won), leaving every chunk range below the nested docstring overrunning
the stored snippet by the inserted lines. `record_block` must make its own snippet authoritative.

Guards: the merged request's snippet is the block pass's snapshot (it contains the nested docstring), every chunk
range stays in bounds, and each range slices the snippet to exactly the text the block pass segmented. The model-free
apply phase is also exercised on top (it re-binds by bidx, so it must keep working regardless). No GGUF model required.
"""
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_python  # noqa: E402
import scale_blocks  # noqa: E402
from scale_blocks import PYTHON_STYLE  # noqa: E402
from scale_escalate import Escalation  # noqa: E402
from scale_llm import GenerationConfig  # noqa: E402


# 'heavy' exceeds the cutoff on its OWN body (nested defs are opaque to the scorer); 'inner' is trivial, so it is
# annotated locally in the def pass - that local docstring is what skews the parent's span between the passes.
SRC = (
    "def heavy(x):\n"
    "    def inner(v):\n"
    "        return v + 1\n"
    "    total = 0\n"
    "    if x and x > 0:\n"
    "        for i in range(x):\n"
    "            if i % 2:\n"
    "                total += inner(i)\n"
    "    elif x < 0:\n"
    "        total = -1\n"
    "    else:\n"
    "        try:\n"
    "            total = risky()\n"
    "        except ValueError:\n"
    "            total = 0\n"
    "        finally:\n"
    "            cleanup()\n"
    "    return total\n"
)


class StubLLM:
    """A minimal stand-in for LocalChatModel: canned replies, generous budget, no real generation."""
    n_ctx = 8192
    ctx_margin = 256

    def estimate_tokens(self, text):
        return max(1, len(text) // 4)

    def count_tokens(self, messages):
        return sum(self.estimate_tokens(m["content"]) for m in messages)

    def snippet_budget(self, messages, cfg, **kwargs):
        return 4000

    def generate(self, messages, cfg=None, stop=None):
        content = messages[-1]["content"]
        if "Write exactly the docstring" in content:
            return "Stub generated docstring."
        if "NEXT paragraph" in content:           # comment-pass prompt
            return "stub block note"
        nums = re.findall(r"(?m)^\s*(\d+)\|", content)
        return ", ".join(f"{n}-{n}" for n in nums) or "NONE"


def main():
    lines = SRC.split("\n")
    cfg = GenerationConfig(max_new_tokens=256)
    msgs = [{"role": "system", "content": "system"}]
    stub = StubLLM()

    heavy_node = next(n for n in ast.parse(SRC).body if getattr(n, "name", "") == "heavy")
    assert scale_python.cognitive_complexity(heavy_node) > 10, "test fixture 'heavy' must exceed the cutoff"

    esc = Escalation(threshold=10)

    # ---- DEF PASS: heavy deferred (span recorded from the PRE-patch source); inner docstringed locally ----
    defs = scale_python.iter_defs_with_info(ast.parse(SRC))
    doc_map = scale_python.generate_docstrings(stub, cfg, msgs, defs, SRC, lines, escalation=esc)
    assert {info.qualname for info in defs if id(info.node) in doc_map} == {"heavy.inner"}, \
        "'inner' must be annotated locally so the parent's span grows before the block pass"
    patched = scale_python.patch_docstrings_textually(lines, defs, doc_map)
    assert any("Stub generated docstring." in ln for ln in patched), "inner should get its local docstring"
    stale_snippet = esc.requests[0]["snippet"]
    assert "Stub generated docstring." not in stale_snippet, \
        "fixture: the def-pass span must predate the nested docstring"

    # ---- BLOCK PASS on the updated text: chunk ranges computed against the grown span ----
    patched_blob = "\n".join(patched)
    targets = scale_python.iter_block_targets(patched_blob, patched)
    emit = scale_blocks.annotate_blocks(stub, cfg, msgs, patched, targets, PYTHON_STYLE, escalation=esc)

    merged = [r for r in esc.requests if r["qualname"] == "heavy"]
    assert len(merged) == 1 and merged[0].get("def") is not None and merged[0].get("blocks") is not None, \
        "both passes must record into ONE per-routine request"

    # The skew fix: the block recording's snippet wins, so the ranges index into the snapshot they were computed from.
    snippet_lines = merged[0]["snippet"].split("\n")
    assert "Stub generated docstring." in merged[0]["snippet"], \
        "record_block must replace the def pass's stale snippet with the block pass's snapshot"
    # Ranges are checked against `patched` - the block pass's INPUT. (`emit` differs inside heavy's span: the nested
    # 'inner' is its own non-escalated target, so its local block edits land there; placement-by-bidx absorbs that.)
    header_idx = patched.index("def heavy(x):")
    for chunk in merged[0]["blocks"]["chunks"]:
        a, b = chunk["lines"]
        assert 1 <= a <= b <= len(snippet_lines), f"chunk range {chunk['lines']} overruns the stored snippet"
        assert snippet_lines[a - 1:b] == patched[header_idx + a - 1:header_idx + b], \
            f"chunk range {chunk['lines']} must slice the snippet to the text the block pass segmented"

    # ---- APPLY still lands the answers (placement is by bidx, independent of the snippet) ----
    manifest = esc.to_manifest("test.py", "python", "\n")
    for req in manifest["requests"]:
        if req.get("def") is not None:
            req["def"]["answer"] = "Heavy docstring from the stronger model."
        if req.get("blocks") is not None:
            for chunk in req["blocks"]["chunks"]:
                chunk["answer"] = "stronger note"
    final = scale_python.apply_manifest(emit, manifest)
    assert any("Heavy docstring" in ln for ln in final) and any("# stronger note" in ln for ln in final), \
        "the merged request must deliver both the docstring and the block comments"
    # The applied docstring is a statement, so `code_preserved` does not apply; instead every emit line must survive.
    for ln in (l for l in emit if l.strip()):
        assert ln in final, f"apply dropped or altered a line: {ln!r}"

    print("PASS: the block pass's snippet supersedes the def pass's stale span; chunk ranges index the right snapshot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
