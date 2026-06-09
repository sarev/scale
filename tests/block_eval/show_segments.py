#!/usr/bin/env python3
"""
Print the chunk ranges the segment pass chooses for each routine in a file - no patching, just inspection.

Use this to judge whether range-segmentation is grouping the body sensibly (few coherent chunks) rather than
fragmenting it line-by-line.

Usage:
    ../.llm-venv/Scripts/python.exe tests/block_eval/show_segments.py <file.py>
    SCALE_MODEL=/path/model.gguf python tests/block_eval/show_segments.py <file.py>
"""
import sys
from pathlib import Path

import _harness as H
from scale_llm import GenerationConfig
from scale_python import iter_block_targets
from scale_blocks import request_segments


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

    for t in iter_block_targets(src, lines):
        segs = request_segments(llm, cfg, messages, lines, t, seg_prompt)
        print(f"\n## {t.qualname}  (body {t.body_start}-{t.body_end}, {len(t.boundary_lines)} candidate starts)")
        for a, b in segs:
            print(f"   {a:>3}-{b:<3} | {lines[a - 1].strip()[:64]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
