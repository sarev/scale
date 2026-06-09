#!/usr/bin/env python3
"""
Selective-escalation manifest round-trip, end to end with a stubbed model:

- EMIT: a complex routine is deferred (recorded in the manifest) while a simple one is annotated locally, and the
  complex routine is left byte-for-byte untouched in the emit output.
- APPLY (model-free): a stronger model's answers from the manifest are patched into the complex routine - block comments
  re-bound by boundary index and the docstring re-bound by (qualname, sig_hash) - through the same insertion-only path,
  so the code signature is preserved. `NONE`/unfilled answers are honoured (no invented comment).

Guards the core promise of the feature: which model writes the words is irrelevant to the "no code is touched" guarantee.
No GGUF model required.
"""
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_python  # noqa: E402
import scale_blocks  # noqa: E402
from scale_blocks import PYTHON_STYLE, code_preserved  # noqa: E402
from scale_escalate import Escalation  # noqa: E402
from scale_llm import GenerationConfig  # noqa: E402


SRC = (
    "def simple():\n"
    "    a = 1\n"
    "    b = 2\n"
    "    c = 3\n"
    "    return a + b + c\n"
    "\n"
    "\n"
    "def heavy(x):\n"
    "    if x and x > 0:\n"
    "        for i in range(x):\n"
    "            if i % 2:\n"
    "                return i\n"
    "    elif x < 0:\n"
    "        return -1\n"
    "    else:\n"
    "        try:\n"
    "            return risky()\n"
    "        except ValueError:\n"
    "            return 0\n"
    "        finally:\n"
    "            cleanup()\n"
    "    return None\n"
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
        # Segment-pass prompt: turn each numbered (boundary) line in the view into its own chunk range.
        nums = re.findall(r"(?m)^\s*(\d+)\|", content)
        return ", ".join(f"{n}-{n}" for n in nums) or "NONE"


def main():
    lines = SRC.split("\n")
    cfg = GenerationConfig(max_new_tokens=256)
    msgs = [{"role": "system", "content": "system"}]
    stub = StubLLM()

    heavy_node = next(n for n in ast.parse(SRC).body if getattr(n, "name", "") == "heavy")
    assert scale_python.cognitive_complexity(heavy_node) > 10, "test fixture 'heavy' must exceed the cutoff"

    # ============ BLOCK PASS ============

    # ---- EMIT: heavy is deferred, simple is annotated locally, heavy stays untouched ----
    targets = scale_python.iter_block_targets(SRC, lines)
    esc = Escalation(threshold=10)
    emit = scale_blocks.annotate_blocks(stub, cfg, msgs, lines, targets, PYTHON_STYLE, escalation=esc)

    block_reqs = [r for r in esc.requests if r["pass"] == "block"]
    assert [r["qualname"] for r in block_reqs] == ["heavy"], \
        f"only 'heavy' should be escalated for blocks, got {[r['qualname'] for r in block_reqs]}"
    assert block_reqs[0]["chunks"], "the escalated routine must carry its (locally segmented) chunk recipe"
    assert any("# stub block note" in ln for ln in emit), "the simple routine should have been annotated locally"
    assert code_preserved(lines, emit, PYTHON_STYLE), "emit must not alter any code"

    # heavy's body lines must be byte-identical in the emit output (only simple was touched, above it).
    assert "# stub block note" not in "\n".join(emit[emit.index("def heavy(x):"):]), \
        "the deferred routine must be left untouched during emit"

    # ---- APPLY: fill the manifest with a stronger model's answers and patch them in (model-free) ----
    manifest = esc.to_manifest("test.py", "python", "\n")
    for req in manifest["requests"]:
        for ci, chunk in enumerate(req["chunks"]):
            chunk["answer"] = "NONE" if ci == 0 else f"heavy step {ci}"   # exercise NONE on the first chunk

    final = scale_python.apply_manifest("\n".join(emit), emit, manifest)

    assert any("# heavy step" in ln for ln in final), "the stronger model's block comments must be inserted"
    assert code_preserved(emit, final, PYTHON_STYLE), "apply must not alter any code"
    assert code_preserved(lines, final, PYTHON_STYLE), "the whole round-trip must preserve the original code"
    # The NONE chunk must paragraph (blank) without inventing a comment for that specific block.
    n_heavy_comments = sum(1 for ln in final if ln.strip().startswith("# heavy step"))
    assert n_heavy_comments == len(block_reqs[0]["chunks"]) - 1, "the NONE answer must not produce a comment"

    # ============ DEFINITION PASS ============

    # ---- EMIT: heavy's docstring deferred; simple's generated locally ----
    tree = ast.parse(SRC)
    defs = scale_python.iter_defs_with_info(tree)
    esc_d = Escalation(threshold=10)
    doc_map = scale_python.generate_docstrings(stub, cfg, msgs, defs, SRC, lines, escalation=esc_d)

    def_reqs = [r for r in esc_d.requests if r["pass"] == "def"]
    assert [r["qualname"] for r in def_reqs] == ["heavy"], "only 'heavy' should be escalated for the def pass"
    qn_in_map = {info.qualname for info in defs if id(info.node) in doc_map}
    assert qn_in_map == {"simple"}, f"only 'simple' should be commented locally, got {qn_in_map}"

    patched = scale_python.patch_docstrings_textually(lines, defs, doc_map)
    assert any("Stub generated docstring." in ln for ln in patched), "simple should get its local docstring"

    # ---- APPLY: patch heavy's docstring from the manifest ----
    manifest_d = esc_d.to_manifest("test.py", "python", "\n")
    for req in manifest_d["requests"]:
        req["answer"] = "Heavy routine docstring written by the stronger model."

    final_d = scale_python.apply_manifest("\n".join(patched), patched, manifest_d)
    assert any("Heavy routine docstring" in ln for ln in final_d), "heavy's deferred docstring must be applied"
    # The def pass inserts docstrings (which are statements), so it changes the code signature by design; the
    # guarantee is that no original code line is dropped or mutated - every one must still be present.
    for ln in (l for l in lines if l.strip()):
        assert ln in final_d, f"the def round-trip dropped or altered an original line: {ln!r}"

    print("PASS: emit defers complex routines untouched; apply patches the stronger model's answers, code preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
