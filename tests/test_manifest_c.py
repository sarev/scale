#!/usr/bin/env python3
"""
C online emit and the run-level (version-2) manifest:

- EMIT (model-free): `collect_def_requests_c` records every documentable record - the doc-site plan's redirected
  definition is skipped and its header prototype requested instead, with the implementation body as the prose
  source - and `defer_block_targets` records each function's chunk recipe.
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
import scale_c  # noqa: E402
from scale_blocks import SLASH_BLOCK_STYLE, code_preserved, defer_block_targets  # noqa: E402
from scale_escalate import Escalation, run_manifest, read_manifest, write_manifest, unfilled_answers  # noqa: E402


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


def main():
    lines = SRC.split("\n")
    key = "src/helper.c"

    # ---- EMIT, def side: the doc-site plan redirects the definition's doc to its prototype ----
    plan = scale_c.plan_doc_sites_c([(key, True, SRC, lines)], "auto")
    esc = Escalation()
    n = scale_c.collect_def_requests_c(SRC, lines, esc, doc_plan=plan.for_file(key))
    assert n == 1, "the redirected definition is skipped and only the prototype requested"
    proto_reqs = [r for r in esc.requests if r.get("def") is not None]
    assert [r["kind"] for r in proto_reqs] == ["declaration"], "the prototype must be the deferred def record"
    assert "int a = 1;" in proto_reqs[0]["snippet"], "the prototype's snippet is the implementation body"

    # Without a plan, every definition is requested from its own span.
    esc_plain = Escalation()
    assert scale_c.collect_def_requests_c(SRC, lines, esc_plain) == 1
    assert esc_plain.requests[0]["kind"] == "function"

    # ---- EMIT, block side: every function's chunk recipe is deferred; the source untouched ----
    esc_b = Escalation()
    targets = scale_c.iter_block_targets_c(SRC, lines)
    assert targets and targets[0].sig, "the C block target must carry its span-hash re-binding identity"
    assert defer_block_targets(esc_b, lines, targets) == 1
    assert lines == SRC.split("\n"), "emit must leave the source untouched"
    block_reqs = [r for r in esc_b.requests if r.get("blocks") is not None]
    assert len(block_reqs) == 1 and block_reqs[0]["snippet"].startswith("int helper(int x)")

    # ---- SLIM: one run manifest; the duplicate snippet (impl body) collapses to a snippet_ref ----
    manifest = run_manifest([("src/helper.h", "c", "\n", esc), (key, "c", "\n", esc_b)], "style")
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
    final = scale_c.apply_manifest_c(lines, sub)

    proto_idx = final.index("int helper(int x);")
    assert "Adds up the working values." in "\n".join(final[:proto_idx]), \
        "the prototype's doc must be patched ABOVE the declaration"
    assert any(re.match(r"\s*// c note", ln) for ln in final), "the block answers must be inserted as // comments"
    # The def doc renders as a /* */ block, so the comparison style must recognise both comment forms.
    assert code_preserved(lines, final, SLASH_BLOCK_STYLE), "apply must not alter any code"

    # ---- I/O round-trip keeps the v2 shape ----
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "m.json"
        write_manifest(p, manifest)
        again = read_manifest(p)
        assert again["version"] == 2 and len(again["requests"]) == 2
        assert unfilled_answers(again) == []

    print("PASS: C online emit defers doc-site prototypes and block recipes, slims duplicate snippets, and the "
          "model-free C apply patches answers with code preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
