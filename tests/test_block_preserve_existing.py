#!/usr/bin/env python3
"""
Online block emit preserves hand-written comments by default (known-issue #1).

A run over already-commented code used to degrade the very blocks that were most carefully written: the writer was
never told a chunk already had a comment, and any non-NONE answer replaced it. Now `defer_block_targets` surfaces an
attached comment as the chunk's `existing` text, and protects a *substantive* (multi-line) one by pre-answering NONE
(which keeps it verbatim at apply) and flagging `preserve`. A one-line comment is surfaced but left to the writer.
`--overwrite-comments` (preserve_existing=False) lifts the protection. No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_python  # noqa: E402
from scale_python import iter_block_targets  # noqa: E402
from scale_blocks import defer_block_targets, PYTHON_STYLE  # noqa: E402
from scale_escalate import Escalation, unfilled_answers  # noqa: E402


# A multi-line rationale sits above a compound boundary; a single note sits above the return boundary.
SRC = (
    "def f(x):\n"
    "    start(x)\n"
    "    y = compute(x)\n"
    "\n"
    "    # first line of the rationale\n"
    "    # second line, the subtle part\n"
    "    for item in y:\n"
    "        handle(item)\n"
    "        log(item)\n"
    "\n"
    "    # single note\n"
    "    return finish(y)\n"
)
RATIONALE = ("    # first line of the rationale", "    # second line, the subtle part")


def chunks_for(preserve_existing=True, style=PYTHON_STYLE):
    lines = SRC.split("\n")
    esc = Escalation(doc_style="s")
    defer_block_targets(esc, lines, iter_block_targets(SRC, lines),
                        style=style, preserve_existing=preserve_existing)
    return esc, {c["anchor"]: c for c in esc.requests[0]["blocks"]["chunks"]}


def main():
    # ---- 1. Default: multi-line comment protected, single-line surfaced but left to the writer ----
    esc, by_anchor = chunks_for()
    multi = by_anchor["for item in y:"]
    assert multi.get("existing", "").split("\n") == list(RATIONALE), "the rationale must be surfaced verbatim"
    assert multi.get("preserve") is True, "a multi-line comment must be flagged preserve"
    assert multi.get("answer") == "NONE", "a multi-line comment must be pre-answered NONE to protect it"

    single = by_anchor["return finish(y)"]
    assert single.get("existing") == "    # single note", "a single-line comment must still be surfaced"
    assert "preserve" not in single, "a single-line comment is fair game, not auto-preserved"
    assert single.get("answer") is None, "the single-line chunk stays open for the writer to decide"

    bare = by_anchor["start(x)"]
    assert "existing" not in bare and bare.get("answer") is None, "a chunk with no prior comment is untouched"

    # The pre-answered NONE counts as filled; only the two genuinely open slots remain (the preserved chunk is at
    # array index 1, so the open labels are block[0] and block[2]).
    manifest = esc.to_manifest("t.py", "python", "\n")
    rid = esc.requests[0]["id"]
    assert sorted(unfilled_answers(manifest)) == [f"{rid}:block[0]", f"{rid}:block[2]"], \
        "the preserved chunk must not count as unfilled work"

    # ---- 2. Round-trip: the protected rationale survives the apply byte-for-byte ----
    lines = SRC.split("\n")
    for req in manifest["requests"]:
        for chunk in req["blocks"]["chunks"]:
            if chunk.get("answer") is None:
                chunk["answer"] = "NONE"   # writer declines the open ones; the protected one is already NONE
    assert unfilled_answers(manifest) == []
    final = scale_python.apply_manifest(lines, manifest)
    for rline in RATIONALE:
        assert rline in final, f"the preserved rationale line was lost: {rline!r}"

    # ---- 3. --overwrite-comments lifts the protection (still surfaced, but open) ----
    _, by_anchor_ow = chunks_for(preserve_existing=False)
    multi_ow = by_anchor_ow["for item in y:"]
    assert multi_ow.get("existing", "").split("\n") == list(RATIONALE), "existing text is still surfaced"
    assert "preserve" not in multi_ow and multi_ow.get("answer") is None, \
        "--overwrite-comments must leave the slot open for the writer"

    # ---- 4. Without a style, existing comments are neither surfaced nor protected ----
    _, by_anchor_ns = chunks_for(style=None)
    assert all("existing" not in c and "preserve" not in c for c in by_anchor_ns.values()), \
        "no style means no comment detection"

    print("PASS: online block emit preserves substantive existing comments by default, --overwrite-comments opts out")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
