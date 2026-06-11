#!/usr/bin/env python3
"""
The block-pass verification floor: the grounding gate on each note, the clean-context obviousness challenge on each
insertable note, and the per-routine story challenge on the note set - with the shared failure routing (regenerate
once with the verdict as feedback; a second failure promotes the routine to the manifest when one is active, leaving
it byte-for-byte untouched, else drops the comment(s) while keeping the paragraphing blanks).

This guards (model-free, via a fake LLM):
- an obviousness NO regenerates the note once (re-scored, re-challenged); a second NO tags it `CHALLENGE_FLAG`
  (kept as context, never written),
- a note whose backticked identifier exists nowhere in the run is nudged once, then flagged,
- the story challenge fires only on a routine longer than `SHORT_FUNCTION_CHUNKS` chunks with at least one insertable
  note (the length guard), regenerates the whole note set once on RESTATE, and on a second RESTATE drops the
  routine's comments (no manifest) or promotes the routine whole (manifest active: chunk recipe recorded, routine
  untouched),
- the persistent message list is balanced around every call.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_blocks import (  # noqa: E402
    BlockTarget, PYTHON_STYLE, CHALLENGE_FLAG, request_block_comment, annotate_blocks, _comment_to_insert,
)
from scale_escalate import Escalation  # noqa: E402
from scale_verify import Verifier  # noqa: E402


@dataclass
class _Cfg:
    temperature: float = 0.2
    max_new_tokens: int = 256


class _RouterLLM:
    """Routes each generate() call to a per-purpose reply queue by recognising the prompt's fixed wording."""

    def __init__(self, notes=None, scores=None, obvious=None, story=None):
        self.notes = list(notes or [])        # turn-1 summaries (and their nudge/feedback regenerations)
        self.scores = list(scores or [])      # 1-3 value scores
        self.obvious = list(obvious or [])    # YES/NO obviousness verdicts (clean context)
        self.story = list(story or [])        # STORY/RESTATE verdicts (clean context)
        self.story_calls = 0

    def generate(self, messages, cfg=None):
        prompt = messages[-1]["content"]
        if "single word YES or NO" in prompt:
            assert len(messages) == 1, "a challenge must run in a clean, single-turn context"
            return self.obvious.pop(0)
        if "single word STORY or RESTATE" in prompt:
            assert len(messages) == 1, "a challenge must run in a clean, single-turn context"
            self.story_calls += 1
            return self.story.pop(0)
        if "single digit" in prompt:
            return self.scores.pop(0)
        return self.notes.pop(0)


SRC = ["def f():", "    a = one()", "    b = two()", "    c = three()", "    return c"]
CORPUS = "\n".join(SRC)


def _target(segments):
    boundaries = (2, 3, 4, 5)
    return BlockTarget(qualname="f", kind="def", header_start=1, header_end=1, body_start=2, body_end=5,
                       boundary_lines=boundaries, indent_of={b: "    " for b in boundaries},
                       doc="Does a thing.", segments=segments)


def _verifier(llm):
    return Verifier(llm, _Cfg(), corpus=CORPUS)


def _request(llm, blob=(2, 2)):
    messages = []
    out = request_block_comment(llm, _Cfg(), messages, SRC, _target(None), blob[0], blob[1], PYTHON_STYLE,
                                prior_comments=[], verifier=_verifier(llm))
    assert messages == [], "the persistent message list must be balanced after the turn"
    return out


def test_obvious_pass_is_clean():
    out = _request(_RouterLLM(notes=["Fetch the seed value before the loop."], scores=["3"], obvious=["YES"]))
    assert out == "Fetch the seed value before the loop."
    assert _comment_to_insert(out) == out


def test_obvious_fail_then_regenerated_note_passes():
    llm = _RouterLLM(notes=["Assign one() to a.", "Seed the pipeline before the expensive calls."],
                     scores=["3", "3"], obvious=["NO", "YES"])
    out = _request(llm)
    assert out == "Seed the pipeline before the expensive calls."
    assert llm.notes == [] and llm.obvious == []


def test_obvious_fail_twice_flags_the_note():
    llm = _RouterLLM(notes=["Assign one() to a.", "Set a from one()."], scores=["3", "3"], obvious=["NO", "NO"])
    out = _request(llm)
    assert out.endswith(CHALLENGE_FLAG), out
    assert _comment_to_insert(out) is None, "a twice-failed note must not be written"
    assert out.replace(CHALLENGE_FLAG, "").strip() == "Set a from one().", "the note survives as context"


def test_block_gate_nudges_then_flags():
    # The note invents `frobnicate`; the nudge regeneration still invents it -> flagged, no scoring, no challenge.
    llm = _RouterLLM(notes=["Calls `frobnicate` on the input.", "Uses `frobnicate` again."], scores=[], obvious=[])
    out = _request(llm)
    assert out.endswith(CHALLENGE_FLAG)
    # A corrected regeneration proceeds to scoring and the challenge as normal.
    llm = _RouterLLM(notes=["Calls `frobnicate` on the input.", "Calls `one` to seed `a`."],
                     scores=["3"], obvious=["YES"])
    out = _request(llm)
    assert out == "Calls `one` to seed `a`."


SEGMENTS = [(2, 2), (3, 3), (4, 4), (5, 5)]  # 4 chunks > SHORT_FUNCTION_CHUNKS (3) -> story-challenge eligible


def _annotate(llm, segments=None, escalation=None):
    return annotate_blocks(llm, _Cfg(), [], SRC, [_target(segments if segments is not None else SEGMENTS)],
                           PYTHON_STYLE, value_threshold=2, escalation=escalation, verifier=_verifier(llm))


def test_story_pass_inserts_comments():
    llm = _RouterLLM(notes=[f"Step {i} of the pipeline, for a reason." for i in range(1, 5)],
                     scores=["3"] * 4, obvious=["YES"] * 4, story=["STORY"])
    out = _annotate(llm)
    assert llm.story_calls == 1
    assert sum(1 for ln in out if ln.lstrip().startswith("#")) == 4, out


def test_story_length_guard_skips_short_routines():
    llm = _RouterLLM(notes=["First.", "Second."], scores=["3", "3"], obvious=["YES", "YES"], story=[])
    _annotate(llm, segments=[(2, 3), (4, 5)])
    assert llm.story_calls == 0, "a routine at/below SHORT_FUNCTION_CHUNKS chunks must never face the story challenge"


def test_story_guard_skips_when_nothing_would_be_written():
    # All notes score 1 (below the threshold): nothing will be inserted, so there is no set to judge.
    llm = _RouterLLM(notes=[f"Note {i}." for i in range(1, 5)], scores=["1"] * 4, obvious=[], story=[])
    out = _annotate(llm)
    assert llm.story_calls == 0
    assert not any(ln.lstrip().startswith("#") for ln in out)


def test_story_fail_then_regenerated_set_passes():
    llm = _RouterLLM(
        notes=[f"Restating step {i}." for i in range(1, 5)] + [f"Intent of step {i}." for i in range(1, 5)],
        scores=["3"] * 8, obvious=["YES"] * 8, story=["RESTATE", "STORY"],
    )
    out = _annotate(llm)
    assert llm.story_calls == 2
    comments = [ln.strip() for ln in out if ln.lstrip().startswith("#")]
    assert comments == [f"# Intent of step {i}." for i in range(1, 5)], "the regenerated set must be the one written"


def test_story_fail_twice_drops_comments_keeps_paragraphing():
    llm = _RouterLLM(
        notes=[f"Restating step {i}." for i in range(1, 5)] + [f"Still restating {i}." for i in range(1, 5)],
        scores=["3"] * 8, obvious=["YES"] * 8, story=["RESTATE", "RESTATE"],
    )
    out = _annotate(llm)
    assert not any(ln.lstrip().startswith("#") for ln in out), "a twice-failed set must be dropped"
    assert [ln for ln in out if ln.strip()] == SRC, "code is untouched"
    assert out.index("    return c") - 1 >= 0 and out[out.index("    return c") - 1] == "", \
        "the paragraphing blanks must remain"


def test_story_fail_twice_with_manifest_promotes_routine_untouched():
    llm = _RouterLLM(
        notes=[f"Restating step {i}." for i in range(1, 5)] + [f"Still restating {i}." for i in range(1, 5)],
        scores=["3"] * 8, obvious=["YES"] * 8, story=["RESTATE", "RESTATE"],
    )
    esc = Escalation(threshold=999)
    out = _annotate(llm, escalation=esc)
    assert out == SRC, "a promoted routine must be left byte-for-byte untouched"
    assert len(esc.requests) == 1 and esc.requests[0].get("blocks") is not None
    assert [c["bidx"] for c in esc.requests[0]["blocks"]["chunks"]] == [0, 1, 2, 3], \
        "the chunk recipe rides the manifest"


def test_flagged_note_with_manifest_promotes_routine():
    # One note fails its obviousness challenge twice -> CHALLENGE_FLAG -> with a manifest the routine is promoted.
    llm = _RouterLLM(
        notes=["Restate a.", "Restate a again.", "Good two.", "Good three.", "Good four."],
        scores=["3"] * 5, obvious=["NO", "NO", "YES", "YES", "YES"], story=["STORY"],
    )
    esc = Escalation(threshold=999)
    out = _annotate(llm, escalation=esc)
    assert out == SRC, "a promoted routine must be left byte-for-byte untouched"
    assert len(esc.requests) == 1 and esc.requests[0].get("blocks") is not None


def main():
    test_obvious_pass_is_clean()
    test_obvious_fail_then_regenerated_note_passes()
    test_obvious_fail_twice_flags_the_note()
    test_block_gate_nudges_then_flags()
    test_story_pass_inserts_comments()
    test_story_length_guard_skips_short_routines()
    test_story_guard_skips_when_nothing_would_be_written()
    test_story_fail_then_regenerated_set_passes()
    test_story_fail_twice_drops_comments_keeps_paragraphing()
    test_story_fail_twice_with_manifest_promotes_routine_untouched()
    test_flagged_note_with_manifest_promotes_routine()
    print("PASS: block notes face the gate, obviousness and story challenges, with drop/promote failure routing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
