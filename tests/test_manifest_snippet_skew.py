#!/usr/bin/env python3
"""
Merged-request snippet consistency (the descendant of the old emit-side skew bug): when a routine with a NESTED def
is recorded by BOTH collectors, the manifest must carry ONE request with ONE snippet, and every block chunk's `lines`
range must index into exactly that snippet (the block recording's snapshot wins by contract - under the online emit
both collectors read the same pristine text, so the spans agree, but the invariant is what the apply phase relies
on). The nested def is its own request, with its own recipe, and the parent's chunk recipe treats it as one opaque
boundary. The model-free apply phase is exercised on top (it re-binds by bidx, so the answers land regardless).

No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_python  # noqa: E402
from scale_blocks import defer_block_targets  # noqa: E402
from scale_escalate import Escalation  # noqa: E402


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


def main():
    lines = SRC.split("\n")

    # ---- EMIT: both collectors over the same pristine text ----
    esc = Escalation()
    assert scale_python.collect_def_requests(SRC, lines, esc) == 2     # heavy + heavy.inner
    targets = scale_python.iter_block_targets(SRC, lines)
    defer_block_targets(esc, lines, targets)

    by_name = {r["qualname"]: r for r in esc.requests}
    assert set(by_name) == {"heavy", "heavy.inner"}, "each routine gets exactly one merged request"
    heavy = by_name["heavy"]
    assert heavy.get("def") is not None and heavy.get("blocks") is not None, \
        "both collectors must record into ONE per-routine request"

    # The invariant the apply phase relies on: every chunk range indexes the request's OWN snippet, slicing it to
    # exactly the text the structural segmenter grouped.
    snippet_lines = heavy["snippet"].split("\n")
    header_idx = lines.index("def heavy(x):")
    assert snippet_lines[0] == "def heavy(x):"
    for chunk in heavy["blocks"]["chunks"]:
        a, b = chunk["lines"]
        assert 1 <= a <= b <= len(snippet_lines), f"chunk range {chunk['lines']} overruns the stored snippet"
        assert snippet_lines[a - 1:b] == lines[header_idx + a - 1:header_idx + b], \
            f"chunk range {chunk['lines']} must slice the snippet to the segmented text"

    # The nested def is one opaque boundary in the parent's recipe, never split across chunks.
    inner_line_rel = lines.index("    def inner(v):") - header_idx + 1
    covering = [c for c in heavy["blocks"]["chunks"] if c["lines"][0] <= inner_line_rel <= c["lines"][1]]
    assert len(covering) == 1, "the nested def must fall inside exactly one parent chunk"

    # ---- APPLY still lands the answers (placement is by bidx, independent of the snippet) ----
    manifest = esc.to_manifest("test.py", "python", "\n")
    for req in manifest["requests"]:
        if req.get("def") is not None:
            req["def"]["answer"] = f"Docstring for {req['qualname']}."
        if req.get("blocks") is not None:
            for chunk in req["blocks"]["chunks"]:
                chunk["answer"] = "stronger note"
    final = scale_python.apply_manifest(lines, manifest)
    assert any("Docstring for heavy." in ln for ln in final) and any("# stronger note" in ln for ln in final), \
        "the merged request must deliver both the docstring and the block comments"
    # The applied docstring is a statement, so `code_preserved` does not apply; instead every line must survive.
    for ln in (l for l in lines if l.strip()):
        assert ln in final, f"apply dropped or altered a line: {ln!r}"

    print("PASS: merged requests carry one snippet; chunk ranges index it exactly; apply lands answers by bidx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
