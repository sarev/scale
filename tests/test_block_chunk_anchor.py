#!/usr/bin/env python3
"""
Every deferred block chunk carries an `anchor` - the verbatim text of the line its comment attaches to - so the
online writer locates each chunk by matching that line in the snippet instead of counting through a body thick with
comments and blank lines (the off-by-N misplacement that put a comment above the wrong handler in a real run).

The anchor must equal the stripped boundary line, that line must actually appear in the request's own snippet, and
the existing `bidx`/`lines` mapping must be untouched. No GGUF model required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_python import iter_block_targets  # noqa: E402
from scale_blocks import defer_block_targets  # noqa: E402
from scale_escalate import Escalation  # noqa: E402


# A body with comments and blank lines between paragraphs - exactly where hand-counting drifts.
SRC = (
    "def handler(payload):\n"
    "    # existing note\n"
    "    parsed = parse(payload)\n"
    "    validate(parsed)\n"
    "\n"
    "    record = build_record(parsed)\n"
    "    persist(record)\n"
    "\n"
    "    try:\n"
    "        notify(record)\n"
    "    except SendError:\n"
    "        queue_retry(record)\n"
    "    return record\n"
)


def main():
    lines = SRC.split("\n")
    esc = Escalation(doc_style="style")
    targets = iter_block_targets(SRC, lines)
    n = defer_block_targets(esc, lines, targets)
    assert n == 1, f"the single routine must defer exactly one block recipe, got {n}"

    req = esc.requests[0]
    snip = req["snippet"].split("\n")
    chunks = req["blocks"]["chunks"]
    assert chunks, "the routine has multiple paragraphs and must produce chunks"

    for chunk in chunks:
        a, b = chunk["lines"]
        # The anchor is the verbatim (stripped) text of the chunk's first snippet line.
        assert chunk["anchor"] == snip[a - 1].strip(), \
            f"anchor {chunk['anchor']!r} must match snippet line {a} {snip[a - 1]!r}"
        # And it must be findable in the snippet the writer is handed.
        assert any(chunk["anchor"] == s.strip() for s in snip), "anchor must appear in the request's own snippet"
        # The bidx/lines contract the apply phase relies on is untouched.
        assert 1 <= a <= b <= len(snip), "chunk range must still index the snippet"
        assert isinstance(chunk["bidx"], int) and chunk["bidx"] >= 0, "bidx must survive unchanged"

    # The misplacement case in the wild: the `except` handler. Its chunk anchor must name the handler line,
    # not a neighbouring statement, so the writer can never attach its comment to the wrong block.
    except_anchor = [c["anchor"] for c in chunks if c["anchor"].startswith("except ")]
    if except_anchor:
        assert except_anchor[0] == "except SendError:", except_anchor[0]

    print("PASS: every deferred block chunk carries the verbatim anchor line, so writers map chunks without counting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
