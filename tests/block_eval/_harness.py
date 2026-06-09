#!/usr/bin/env python3
"""
Shared setup for the model-dependent block-pass evaluation harnesses (`show_segments.py`, `show_comments.py`).

These are NOT model-free unit tests - they load a real GGUF and exercise the actual segment/comment passes so we can
eyeball block-pass quality as SCALE evolves. The model path defaults to the project default and can be overridden with
the SCALE_MODEL environment variable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# tests/block_eval/_harness.py -> project root is two levels up.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = os.environ.get("SCALE_MODEL") or str(
    ROOT.parent / "models" / "bartowski" / "Qwen2.5.1-Coder-7B-Instruct-GGUF"
    / "Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf"
)
SCALE_CFG = ROOT / "scale-cfg"


def load_model():
    """Load the evaluation model with the same context settings the CLI uses by default."""
    from scale_llm import LocalChatModel
    return LocalChatModel(DEFAULT_MODEL, n_ctx=12 * 1024, n_batch=256, n_gpu_layers=-1, verbose=False)


def prime(llm, cfg, src_path: Path, src: str):
    """Prime the block-pass conversation for a source file (system + summary + blocks template)."""
    from scale import prime_llm_for_comments
    return prime_llm_for_comments(
        llm, cfg, SCALE_CFG, src_path, src, "python", no_cache=True, template="blocks"
    )


def read_cfg(name: str) -> str:
    """Read a scale-cfg prompt file by name."""
    return (SCALE_CFG / name).read_text(encoding="utf-8")
