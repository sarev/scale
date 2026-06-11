#!/usr/bin/env python3
"""
C selective escalation and the run-level (version-2) manifest:

- EMIT: with a manifest active, a doc-site redirected prototype is ALWAYS deferred (a public contract is the highest
  value per stronger-model token) and a `--codestats-json`-scored function's blocks defer too; deferred records are
  left byte-for-byte untouched.
- SLIM: `run_manifest` merges per-target collectors, stamps each request with its `file`, and dedupes byte-identical
  snippets across requests into a `snippet_ref` (the impl body that feeds a header prototype's prose never crosses
  the wire twice).
- APPLY (model-free): `apply_manifest_c` re-binds by (qualname, span hash), patches the prototype's doc above the
  declaration and the block comments by boundary index, all through the code-preservation guard.

No GGUF model required.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_blocks  # noqa: E402
import scale_c  # noqa: E402
from scale_blocks import SLASH_LINE_STYLE, SLASH_BLOCK_STYLE, code_preserved  # noqa: E402
from scale_escalate import Escalation, run_manifest, read_manifest, write_manifest, unfilled_answers  # noqa: E402
from scale_llm import GenerationConfig  # noqa: E402


SRC = (
    "int helper(int x);\n"
    "\n"
    "int helper(int x)\n"
    "{\n"
    "    int a = 1;\n"
    "    int b = 2;\n"
    "    if (x > 0) {\n"
    "        a = b + x;\n"
    "        b = a * 2;\n"
    "        a = a + b;\n"
    "    }\n"
    "    return a + b;\n"
    "}\n"
)


class StubLLM:
    """Canned replies; generous budget; fails the test if asked to write a deferred record's comment."""
    n_ctx = 8192
    ctx_margin = 256

    def estimate_tokens(self, text):
        return max(1, len(text) // 4)

    def count_tokens(self, messages):
        return sum(self.estimate_tokens(m["content"]) for m in messages)

    def snippet_budget(self, messages, cfg, **kwargs):
        return 4000

    def generate(self, messages, cfg=None, stop=None):
        return "stub"


def main():
    lines = SRC.split("\n")
    cfg = GenerationConfig(max_new_tokens=256)
    stub = StubLLM()
    key = "src/helper.c"

    # ---- EMIT, def pass: the doc-site prototype is always deferred when a manifest is active ----
    plan = scale_c.plan_doc_sites_c([(key, True, SRC, lines)], "auto")
    tree, sb = scale_c._parse_c(SRC)
    defs = scale_c.iter_defs_with_info_c(tree, sb)
    decls = [d for d in scale_c.iter_decls_with_info_c(tree, sb) if d.qualname in plan.for_file(key).header_names]
    esc = Escalation(threshold=10)
    doc_map = scale_c.generate_comments_c(stub, cfg, [], defs, SRC, lines,
                                          doc_plan=plan.for_file(key), decls=decls, escalation=esc)
    assert doc_map == {}, "the redirected definition is skipped and the prototype deferred - nothing local"
    proto_reqs = [r for r in esc.requests if r.get("def") is not None]
    assert [r["kind"] for r in proto_reqs] == ["declaration"], "the prototype must be the deferred def record"
    assert "int a = 1;" in proto_reqs[0]["snippet"], "the prototype's snippet is the implementation body"

    # ---- EMIT, block pass: a codestats-scored function defers its chunk recipe; the routine is untouched ----
    esc_b = Escalation(threshold=10, override={"helper": 25})
    targets = scale_c.iter_block_targets_c(SRC, lines)
    assert targets and targets[0].sig, "the C block target must carry its span-hash escalation identity"
    emit = scale_blocks.annotate_blocks(stub, cfg, [], lines, targets, SLASH_LINE_STYLE, escalation=esc_b)
    assert emit == lines, "the deferred routine must be left byte-for-byte untouched"
    block_reqs = [r for r in esc_b.requests if r.get("blocks") is not None]
    assert len(block_reqs) == 1 and block_reqs[0]["snippet"].startswith("int helper(int x)")

    # ---- SLIM: one run manifest; the duplicate snippet (impl body) collapses to a snippet_ref ----
    manifest = run_manifest([("src/helper.h", "c", "\n", esc), (key, "c", "\n", esc_b)], 10, "style")
    reqs = manifest["requests"]
    assert [r["file"] for r in reqs] == ["src/helper.h", key]
    assert reqs[0]["snippet"] and reqs[1].get("snippet") is None and reqs[1]["snippet_ref"] == reqs[0]["id"], \
        "a byte-identical snippet must cross the wire once, the second request referencing the first"
    assert len(unfilled_answers(manifest)) == 1 + len(block_reqs[0]["blocks"]["chunks"])

    # ---- APPLY: answers patched by span hash (def above the prototype; blocks by boundary index) ----
    for r in reqs:
        if r.get("def") is not None:
            r["def"]["answer"] = "Adds up the working values.\n\nReturns the combined total."
        if r.get("blocks") is not None:
            for ci, chunk in enumerate(r["blocks"]["chunks"]):
                chunk["answer"] = "NONE" if ci == 0 else f"c note {ci}"

    # The header request targets the prototype, which lives in this same file in the fixture; apply both sets here.
    sub = dict(manifest)
    final = scale_c.apply_manifest_c("\n".join(emit), emit, sub)

    proto_idx = final.index("int helper(int x);")
    assert "Adds up the working values." in "\n".join(final[:proto_idx]), \
        "the prototype's doc must be patched ABOVE the declaration"
    assert any(re.match(r"\s*// c note", ln) for ln in final), "the block answers must be inserted as // comments"
    # The def doc renders as a /* */ block, so the comparison style must recognise both comment forms.
    assert code_preserved(emit, final, SLASH_BLOCK_STYLE), "apply must not alter any code"

    # ---- I/O round-trip keeps the v2 shape ----
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.json"
        write_manifest(p, manifest)
        again = read_manifest(p)
        assert again["version"] == 2 and len(again["requests"]) == 2
        assert unfilled_answers(again) == []

    print("PASS: C escalation defers doc-site prototypes and scored routines, slims duplicate snippets, and the "
          "model-free C apply patches answers with code preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
