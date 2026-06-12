#!/usr/bin/env python3
"""
Online-mode manifest round-trip, end to end and model-free:

- EMIT: the collectors record every routine - def slot + block chunk recipe merged into one request per routine -
  and the source is left byte-for-byte untouched (no model is involved at all).
- APPLY (model-free): the stronger model's answers from the manifest are patched in - block comments re-bound by
  boundary index and the docstring re-bound by (qualname, sig_hash) - through the same insertion-only path, so the
  code signature is preserved. `NONE`/unfilled answers are honoured (no invented comment).

Guards the core promise of the feature: which model writes the words is irrelevant to the "no code is touched"
guarantee. No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_python  # noqa: E402
from scale_blocks import PYTHON_STYLE, code_preserved, defer_block_targets  # noqa: E402
from scale_escalate import Escalation, unfilled_answers  # noqa: E402


SRC = (
    "def simple():\n"
    "    a = 1\n"
    "    b = 2\n"
    "    c = 3\n"
    "    return a + b + c\n"
    "\n"
    "\n"
    "def heavy(x):\n"
    "    total = 0\n"
    "    seen = []\n"
    "    if x and x > 0:\n"
    "        for i in range(x):\n"
    "            if i % 2:\n"
    "                total += i\n"
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

    # ---- EMIT: every routine deferred (def slot + block recipe in ONE request); the source untouched ----
    esc = Escalation(doc_style="style")
    assert scale_python.collect_def_requests(SRC, lines, esc) == 2
    targets = scale_python.iter_block_targets(SRC, lines)
    assert defer_block_targets(esc, lines, targets) == 2

    assert [r["qualname"] for r in esc.requests] == ["simple", "heavy"]
    assert all(r.get("def") is not None and r.get("blocks") is not None for r in esc.requests), \
        "def + blocks must merge into one request per routine"
    assert lines == SRC.split("\n"), "emit must leave the source untouched"

    # SLIM: the routine's code rides the manifest once, as its verbatim span; each chunk references line ranges INTO
    # that snippet rather than duplicating the text.
    heavy = next(r for r in esc.requests if r["qualname"] == "heavy")
    snippet_lines = heavy["snippet"].split("\n")
    assert snippet_lines[0] == "def heavy(x):", "the snippet must be the routine's verbatim span"
    chunks = heavy["blocks"]["chunks"]
    assert chunks, "the deferred routine must carry its (structurally segmented) chunk recipe"
    header_idx = lines.index("def heavy(x):")
    for chunk in chunks:
        a, b = chunk["lines"]
        assert 1 <= a <= b <= len(snippet_lines), f"chunk range {chunk['lines']} overruns the snippet"
        assert snippet_lines[a - 1:b] == lines[header_idx + a - 1:header_idx + b], \
            f"chunk range {chunk['lines']} must slice the snippet to the segmented text"
        assert "text" not in chunk, "a chunk must not duplicate the snippet text"

    # ---- APPLY: fill the manifest with a stronger model's answers and patch them in (model-free) ----
    manifest = esc.to_manifest("test.py", "python", "\n")
    assert manifest["doc_style"] == "style"
    n_unfilled = len(unfilled_answers(manifest))
    assert n_unfilled == 2 + sum(len(r["blocks"]["chunks"]) for r in manifest["requests"]), \
        "every def + chunk slot must start unfilled"

    for req in manifest["requests"]:
        req["def"]["answer"] = f"Docstring for {req['qualname']} from the stronger model."
        for ci, chunk in enumerate(req["blocks"]["chunks"]):
            chunk["answer"] = "NONE" if ci == 0 else f"{req['qualname']} step {ci}"   # exercise NONE on chunk 0
    assert unfilled_answers(manifest) == [], "NONE counts as a deliberate (filled) decline"

    final = scale_python.apply_manifest(lines, manifest)

    assert any("Docstring for simple" in ln for ln in final) and any("Docstring for heavy" in ln for ln in final), \
        "the deferred docstrings must be applied"
    assert any("# heavy step" in ln for ln in final), "the stronger model's block comments must be inserted"
    # The applied docstrings are statements, so code_preserved (which ignores only blanks/comments) does not apply
    # across the def patch; instead every original code line must survive intact.
    for ln in (l for l in lines if l.strip()):
        assert ln in final, f"the round-trip dropped or altered an original line: {ln!r}"

    # A blocks-only manifest must additionally satisfy the strict guard: comments/blanks only.
    esc_b = Escalation()
    defer_block_targets(esc_b, lines, targets)
    manifest_b = esc_b.to_manifest("test.py", "python", "\n")
    for req in manifest_b["requests"]:
        for chunk in req["blocks"]["chunks"]:
            chunk["answer"] = "a block note"
    blocks_only = scale_python.apply_manifest(lines, manifest_b)
    assert code_preserved(lines, blocks_only, PYTHON_STYLE), "a block-only apply must not alter any code"

    # The NONE chunks must paragraph (blank) without inventing comments for those blocks.
    n_heavy_comments = sum(1 for ln in final if ln.strip().startswith("# heavy step"))
    assert n_heavy_comments == len(chunks) - 1, "the NONE answer must not produce a comment"

    print("PASS: online emit defers every routine untouched; apply patches the stronger model's answers, code preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
