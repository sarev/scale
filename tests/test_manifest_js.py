#!/usr/bin/env python3
"""
JavaScript online-manifest support, end to end and model-free:

- EMIT: `collect_def_requests_js` records every definition-like construct (functions, nested functions, classes,
  methods, arrow bindings) with its verbatim `header_start..end` span as snippet + hashed identity, and
  `defer_block_targets` records the chunk recipes - def + blocks merging into ONE request per routine because the
  block provider stamps `sig` over the SAME span convention (a mismatch would silently split a routine in two).
- APPLY (model-free): `apply_manifest_js` patches the def answers as JSDoc above the headers, re-parses, and lands
  the block answers by boundary index - shift-proof, because the def insertions move every line but never touch a
  routine's own span. `NONE` paragraph answers add the blank only; code is preserved throughout.

No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_javascript as sjs  # noqa: E402
from scale_blocks import SLASH_BLOCK_STYLE, code_preserved, defer_block_targets  # noqa: E402
from scale_escalate import Escalation, unfilled_answers  # noqa: E402


SRC = (
    "function outer(x) {\n"
    "    function inner(v) {\n"
    "        return v + 1;\n"
    "    }\n"
    "    let total = 0;\n"
    "    if (x > 0) {\n"
    "        total = inner(x);\n"
    "        total += 2;\n"
    "        total *= 3;\n"
    "    }\n"
    "    return total;\n"
    "}\n"
    "\n"
    "class Box {\n"
    "    get(k) {\n"
    "        const v = this.map[k];\n"
    "        if (v === undefined) {\n"
    "            return null;\n"
    "        }\n"
    "        return v;\n"
    "    }\n"
    "}\n"
    "\n"
    "const twice = (n) => {\n"
    "    const m = n * 2;\n"
    "    return m;\n"
    "};\n"
)


def main():
    lines = SRC.split("\n")

    # ---- EMIT: every def requested; routines with bodies add their chunk recipes into the SAME requests ----
    esc = Escalation(doc_style="style")
    n = sjs.collect_def_requests_js(SRC, lines, esc)
    names = [r["qualname"] for r in esc.requests]
    assert n == len(names) and {"outer", "outer.inner", "Box", "Box.get", "twice"} <= set(names), names

    targets = sjs.iter_block_targets_js(SRC, lines)
    assert all(t.sig for t in targets), "every JS block target must carry its span-hash identity"
    assert defer_block_targets(esc, lines, targets) >= 3   # outer, Box.get, twice (and inner if it has boundaries)
    assert lines == SRC.split("\n"), "emit must leave the source untouched"

    # Def + block merge: the block recording's span hash must equal the def recording's (same convention).
    outer = [r for r in esc.requests if r["qualname"] == "outer"]
    assert len(outer) == 1, "def and block recordings must merge into one request (span conventions must match)"
    assert outer[0].get("def") is not None and outer[0].get("blocks") is not None

    # Chunk ranges slice the request's own snippet to exactly the segmented text.
    snippet_lines = outer[0]["snippet"].split("\n")
    header_idx = lines.index("function outer(x) {")
    for chunk in outer[0]["blocks"]["chunks"]:
        a, b = chunk["lines"]
        assert 1 <= a <= b <= len(snippet_lines), f"chunk range {chunk['lines']} overruns the snippet"
        assert snippet_lines[a - 1:b] == lines[header_idx + a - 1:header_idx + b], \
            f"chunk range {chunk['lines']} must slice the snippet to the segmented text"

    # ---- FILL: a stronger model answers every slot ("NONE" on outer's first chunk) ----
    manifest = esc.to_manifest("test.js", "js", "\n")
    for req in manifest["requests"]:
        if req.get("def") is not None:
            req["def"]["answer"] = f"Documents {req['qualname']}."
        if req.get("blocks") is not None:
            for ci, chunk in enumerate(req["blocks"]["chunks"]):
                if req["qualname"] == "outer" and ci == 0:
                    chunk["answer"] = "NONE"
                else:
                    chunk["answer"] = f"{req['qualname']} note {ci}"
    assert unfilled_answers(manifest) == []

    # ---- APPLY: JSDoc above every header; block comments land despite the def-insertion line shifts ----
    final = sjs.apply_manifest_js(lines, manifest)

    for header, name in (("function outer(x) {", "outer"), ("class Box {", "Box"), ("const twice = (n) => {", "twice")):
        idx = final.index(header)
        above = "\n".join(final[max(0, idx - 6):idx])
        assert f"Documents {name}." in above, f"the JSDoc for {name} must sit above its header:\n{above}"
        assert "/**" in above, "the doc must render as a JSDoc block (no other comments in the file)"
    # The method too (shifted by the class's own JSDoc).
    get_idx = next(i for i, ln in enumerate(final) if ln.strip().startswith("get(k)"))
    assert "Documents Box.get." in "\n".join(final[max(0, get_idx - 6):get_idx])

    assert any("// outer note" in ln for ln in final), "outer's block comments must land after the line shifts"
    assert any("// Box.get note" in ln for ln in final), "method block comments must land"
    # The NONE chunk (outer's opener, right under the `{`) adds no comment line.
    body_first = next(i for i, ln in enumerate(final) if ln.strip() == "function inner(v) {")
    assert not final[body_first - 1].strip().startswith("//"), "a NONE answer must not produce a comment"

    # Code preserved: blanks/comments only (JSDoc blocks + // notes are both comment forms under the block style).
    assert code_preserved(lines, final, SLASH_BLOCK_STYLE), "apply must not alter any code"

    print("PASS: JS manifest round-trip - every def requested, merged def+block requests, shift-proof apply with "
          "JSDoc above headers, NONE honoured, code preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
