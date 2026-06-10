#!/usr/bin/env python3
"""
The two-turn block comment pass: every paragraph is ALWAYS summarised (turn 1, no opt-out), then that summary is
scored 1-3 for value (turn 2; 1 = restates the code, 2 = signpost, 3 = intent/gotcha). The summary is always returned
so it can feed later paragraphs' context; a summary scoring below the threshold is tagged with the magic `VALUE_FLAG`
so it is kept as context but skipped at output.

This guards (model-free, via a fake LLM):
- the parser helpers (`_parse_summary`, `_parse_score`, `_comment_to_insert`),
- that a high score returns a clean (insertable) summary and a low score returns a flagged one,
- that `value_threshold` (the `--block-comments` density knob) moves the cut,
- that an unusable first reply is nudged once into a real summary,
- that the persistent message list is restored (append-then-pop) after every call.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_blocks import (  # noqa: E402
    BlockTarget, PYTHON_STYLE, VALUE_FLAG, COMMENT_VALUE_THRESHOLD,
    request_block_comment, annotate_blocks, _parse_summary, _parse_score, _comment_to_insert,
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
    assert _parse_score("3") == 3 and _parse_score("score: 2/3") == 2
    assert _parse_score("5") == 3 and _parse_score("0") == 1            # clamped to [1, 3] (tolerates an old-scale lapse)
    assert _parse_score("no digit") == COMMENT_VALUE_THRESHOLD          # default on miss
    assert _comment_to_insert(f"x {VALUE_FLAG}") is None                # flagged -> not inserted
    assert _comment_to_insert("keep me") == "keep me" and _comment_to_insert(None) is None


def test_high_score_inserts_low_score_flags():
    hi = _run(["Short-circuit before the slow scan.", "3"])
    assert hi == "Short-circuit before the slow scan." and _comment_to_insert(hi) == hi

    lo = _run(["Return the computed value.", "1"])
    assert lo.endswith(VALUE_FLAG), f"a low score must flag the summary: {lo!r}"
    assert _comment_to_insert(lo) is None
    assert lo[: -len(VALUE_FLAG)].strip() == "Return the computed value.", "the summary survives for context"


def test_threshold_knob_moves_the_cut():
    # A score of 2 (signpost) is kept at the default threshold (2) but flagged when --block-comments low raises it to 3.
    assert not _run(["A useful heading.", "2"], value_threshold=2).endswith(VALUE_FLAG)
    assert _run(["A useful heading.", "2"], value_threshold=3).endswith(VALUE_FLAG)


def test_unusable_first_reply_is_nudged():
    # turn 1 -> "NONE" (unusable) -> nudge -> real summary -> turn 2 score.
    out = _run(["NONE", "Validate the inputs before use.", "3"])
    assert out == "Validate the inputs before use.", out


class _BoomLLM:
    """Fails if asked to generate - proves no model work happens."""
    def generate(self, *a, **k):
        raise AssertionError("the model must not be called when the value threshold is above 3")


def test_threshold_above_max_skips_all_model_work_but_still_paragraphs():
    src = ["void f() {", "    a();", "    b();", "    c();", "}"]
    target = BlockTarget(qualname="f", kind="def", header_start=1, header_end=1, body_start=2, body_end=5,
                         boundary_lines=(2, 3, 4), indent_of={2: "    ", 3: "    ", 4: "    "}, segments=[(3, 4)])
    out = annotate_blocks(_BoomLLM(), _Cfg(), [], src, [target], PYTHON_STYLE, value_threshold=4)

    assert not any(ln.lstrip().startswith("#") for ln in out), "threshold > 3 must insert no comments"
    bi = out.index("    b();")
    assert out[bi - 1] == "", "the chunk must still be paragraphed with a blank line above it"
    assert [ln for ln in out if ln.strip()] == src, "only a blank line was added; code is untouched"


def main():
    test_parsers()
    test_high_score_inserts_low_score_flags()
    test_threshold_knob_moves_the_cut()
    test_unusable_first_reply_is_nudged()
    test_threshold_above_max_skips_all_model_work_but_still_paragraphs()
    print("PASS: two-turn summarise-then-score flags low-value comments, honours the threshold, and nudges")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
