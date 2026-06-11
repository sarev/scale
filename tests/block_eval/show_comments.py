#!/usr/bin/env python3
"""
Print the comment the comment pass produces for each chunk of each routine in a file - no patching, just inspection.

Mirrors `annotate_blocks` (segment -> per-chunk comment, with the short/long length note and the narrative thread of
earlier comments), calling the real engine functions so it reflects actual behaviour (including the nudge retry).

Usage:
    env/Scripts/python.exe tests/block_eval/show_comments.py <file.py>
    SCALE_MODEL=/path/model.gguf python tests/block_eval/show_comments.py <file.py>
"""
import sys
from pathlib import Path

import _harness as H
from scale_llm import GenerationConfig
from scale_python import iter_block_targets
from scale_blocks import (
    request_segments,
    request_block_comment,
    PYTHON_STYLE,
    SHORT_FUNCTION_CHUNKS,
)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1])
    src = target.read_text(encoding="utf-8")
    lines = src.split("\n")

    llm = H.load_model()
    cfg = GenerationConfig(max_new_tokens=8192, temperature=0.2)
    messages = H.prime(llm, cfg, target, src)
    seg_prompt = H.read_cfg("blocks.segment.txt")
    com_prompt = H.read_cfg("blocks.comment.txt")
    nudge = H.read_cfg("blocks.comment.nudge.txt")
    note_short = H.read_cfg("blocks.note.short.txt")
    note_long = H.read_cfg("blocks.note.long.txt")

    for t in iter_block_targets(src, lines):
        segs = request_segments(llm, cfg, messages, lines, t, seg_prompt)
        short = len(segs) <= SHORT_FUNCTION_CHUNKS
        note = note_short if short else note_long
        print(f"\n## {t.qualname}  ({len(segs)} chunks -> {'short' if short else 'long'} note)")
        priors: list[str] = []
        for a, b in segs:
            comment = request_block_comment(
                llm, cfg, messages, lines, t, a, b, PYTHON_STYLE,
                prior_comments=priors, length_note=note,
                prompt_template=com_prompt, nudge_template=nudge,
            )
            if comment:
                priors.append(comment)
            print(f"   [{a}-{b}] {lines[a - 1].strip()[:40]!r}")
            print(f"        -> {comment!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
