#!/usr/bin/env python3
"""
The two-turn block comment pass: every paragraph is ALWAYS summarised (turn 1, no opt-out), then that summary is
scored 1-5 for value (turn 2). The summary is always returned so it can feed later paragraphs' context; a summary
scoring below the threshold is tagged with the magic `VALUE_FLAG` so it is kept as context but skipped at output.

This guards (model-free, via a fake LLM):
- the parser helpers (`_parse_summary`, `_parse_score`, `_comment_to_insert`),
- that a high score returns a clean (insertable) summary and a low score returns a flagged one,
- that `value_threshold` (the `--comment-value` knob) moves the cut,
- that an unusable first reply is nudged once into a real summary,
- that the persistent message list is restored (append-then-pop) after every call.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_blocks import (  # noqa: E402
    BlockTarget, PYTHON_STYLE, VALUE_FLAG, COMMENT_VALUE_THRESHOLD,
    request_block_comment, _parse_summary, _parse_score, _comment_to_insert,
)


@dataclass
class _Cfg:
    temperature: float = 0.2
    max_new_tokens: int = 256


class _FakeLLM:
    """Returns queued replies in order; records that messages were balanced around each call."""
    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, messages, cfg=None):
        return self.replies.pop(0)


def _target():
    return BlockTarget(qualname="f", kind="def", header_start=1, header_end=1, body_start=2, body_end=3,
                       boundary_lines=(2,), indent_of={2: "    "}, doc="Does a thing.")


SRC = ["def f():", "    x = compute()", "    return x"]


def _run(replies, value_threshold=COMMENT_VALUE_THRESHOLD):
    messages = []
    out = request_block_comment(_FakeLLM(replies), _Cfg(), messages, SRC, _target(), 2, 3, PYTHON_STYLE,
                                prior_comments=[], value_threshold=value_threshold)
    assert messages == [], "the persistent message list must be restored after the turn"
    return out


def test_parsers():
    assert _parse_summary("  Walk newest first.  ", PYTHON_STYLE) == "Walk newest first."
    assert _parse_summary("# echoed delimiter", PYTHON_STYLE) == "echoed delimiter"
    assert _parse_summary(f"copied prior {VALUE_FLAG}", PYTHON_STYLE) == "copied prior"
    assert _parse_summary("NONE", PYTHON_STYLE) == "" and _parse_summary("   ", PYTHON_STYLE) == ""
    assert _parse_score("4") == 4 and _parse_score("score: 2/5") == 2
    assert _parse_score("no digit") == COMMENT_VALUE_THRESHOLD          # default on miss
    assert _comment_to_insert(f"x {VALUE_FLAG}") is None                # flagged -> not inserted
    assert _comment_to_insert("keep me") == "keep me" and _comment_to_insert(None) is None


def test_high_score_inserts_low_score_flags():
    hi = _run(["Short-circuit before the slow scan.", "5"])
    assert hi == "Short-circuit before the slow scan." and _comment_to_insert(hi) == hi

    lo = _run(["Return the computed value.", "1"])
    assert lo.endswith(VALUE_FLAG), f"a low score must flag the summary: {lo!r}"
    assert _comment_to_insert(lo) is None
    assert lo[: -len(VALUE_FLAG)].strip() == "Return the computed value.", "the summary survives for context"


def test_threshold_knob_moves_the_cut():
    # A score of 3 is kept at the default threshold (3) but flagged when --comment-value raises it to 4.
    assert not _run(["A useful heading.", "3"], value_threshold=3).endswith(VALUE_FLAG)
    assert _run(["A useful heading.", "3"], value_threshold=4).endswith(VALUE_FLAG)


def test_unusable_first_reply_is_nudged():
    # turn 1 -> "NONE" (unusable) -> nudge -> real summary -> turn 2 score.
    out = _run(["NONE", "Validate the inputs before use.", "4"])
    assert out == "Validate the inputs before use.", out


def main():
    test_parsers()
    test_high_score_inserts_low_score_flags()
    test_threshold_knob_moves_the_cut()
    test_unusable_first_reply_is_nudged()
    print("PASS: two-turn summarise-then-score flags low-value comments, honours the threshold, and nudges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
