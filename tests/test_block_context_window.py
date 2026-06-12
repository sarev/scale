#!/usr/bin/env python3
"""
The block pass's context window: the comment turn shows each paragraph inside a raw ±`BLOCK_CONTEXT_LINES` window of
surrounding source (clamped to the routine, nudged back to the enclosing scope opener within `BLOCK_SCOPE_NUDGE_CAP`
lines), with exactly the paragraph's own lines gutter-marked `> `. Guards the bland/hallucinated comments that a bare,
contextless paragraph used to drive - and that the extra context stays the model's view only: callee notes land on
target lines alone, and the obviousness challenge still judges the pristine, unmarked paragraph.

Model-free: a fake LLM records the prompts and returns canned summary/score replies.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_blocks import (  # noqa: E402
    BLOCK_CONTEXT_LINES, BLOCK_SCOPE_NUDGE_CAP, BlockTarget, PYTHON_STYLE, _context_window, request_block_comment,
)


@dataclass
class _Cfg:
    temperature: float = 0.2
    max_new_tokens: int = 256


class _FakeLLM:
    """Returns queued replies in order, recording each turn's prompt."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def snippet_budget(self, messages, cfg):
        return 100000

    def estimate_tokens(self, text):
        return 1

    def generate(self, messages, cfg=None):
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0)


class _FakeVerifier:
    """Passes every check, recording the block text each obviousness challenge was shown."""

    def __init__(self):
        self.challenged = []

    def ungrounded(self, text):
        return []

    def gate_feedback(self, tokens):
        return "(gate feedback)"

    def challenge_obvious(self, block, comment):
        self.challenged.append(block)
        return True

    def obvious_feedback_for(self):
        return "(obvious feedback)"


def _make_target(lines, header_start=1, header_end=1):
    body_start = header_end + 1
    body_end = len(lines)
    boundaries = tuple(range(body_start, body_end + 1))
    indents = {n: lines[n - 1][:len(lines[n - 1]) - len(lines[n - 1].lstrip())] for n in boundaries}
    return BlockTarget(qualname="f", kind="def", header_start=header_start, header_end=header_end,
                       body_start=body_start, body_end=body_end, boundary_lines=boundaries,
                       indent_of=indents, doc="Does a thing.")


def test_window_clamps_to_the_routine():
    # A short routine: ±8 overshoots both edges, so the window is exactly header_start..body_end.
    src = ["def f():", "    a = 1", "    b = 2", "    return a + b"]
    lo, hi = _context_window(src, _make_target(src), 3, 3)
    assert (lo, hi) == (1, 4), (lo, hi)


def test_scope_nudge_reaches_past_the_plain_window():
    # The paragraph sits 13 lines below its `if` opener - beyond ±8, within the 24-line nudge cap - so the window's
    # lower edge is pulled back to the opener (and only the lower edge: hi is untouched).
    src = (["def g():", "    if cond:"]
           + [f"        fill_{i} = {i}" for i in range(12)]    # lines 3-14
           + ["        target = 1", "        target2 = 2", "    return 0"])
    target = _make_target(src)
    lo, hi = _context_window(src, target, 15, 16)
    assert lo == 2, lo                                          # the `if cond:` line, not 15-8=7
    assert hi == 17, hi
    # A paragraph already adjacent to its opener gains nothing: the ±N window reaches it anyway (never forward).
    lo2, _ = _context_window(src, target, 3, 4)
    assert lo2 == 1, lo2


def test_nudge_cap_bounds_the_walk():
    # The opener sits more than BLOCK_SCOPE_NUDGE_CAP lines above the paragraph, so the walk gives up and the plain
    # ±N window stands.
    filler = [f"        fill_{i} = {i}" for i in range(BLOCK_SCOPE_NUDGE_CAP + 4)]
    src = ["def g():", "    if cond:"] + filler + ["        target = 1"]
    blob = len(src)
    lo, _ = _context_window(src, _make_target(src), blob, blob)
    assert lo == blob - BLOCK_CONTEXT_LINES, lo


def test_exactly_the_target_lines_are_marked():
    src = ["def f():", "    a = 1", "    b = 2", "    c = 3", "    return a + b + c"]
    llm = _FakeLLM(["Sets up b and c.", "3"])
    messages = []
    out = request_block_comment(llm, _Cfg(), messages, src, _make_target(src), 3, 4, PYTHON_STYLE)
    assert out == "Sets up b and c."
    assert messages == []                                       # append-then-pop preserved
    turn1 = llm.prompts[0]
    assert ">     b = 2" in turn1 and ">     c = 3" in turn1    # the paragraph's lines carry the mark...
    assert "  def f():" in turn1 and "      a = 1" in turn1     # ...context lines a two-space gutter
    assert "      return a + b + c" in turn1
    marked = [ln for ln in turn1.split("\n") if ln.startswith("> ")]
    assert len(marked) == 2, marked                             # and nothing else is marked


def test_callee_notes_land_on_target_lines_only():
    # line_notes maps both a target line (3) and a context line (2); only the target line's note may be shown.
    src = ["def f():", "    y = helper(1)", "    z = other(y)", "    return z"]
    llm = _FakeLLM(["Delegate to other.", "3"])
    request_block_comment(llm, _Cfg(), [], src, _make_target(src), 3, 3, PYTHON_STYLE,
                          line_notes={2: "helper: Double x.", 3: "other: Process y."})
    turn1 = llm.prompts[0]
    assert ">     z = other(y)  # other: Process y." in turn1   # the target line carries its note
    assert "helper: Double x." not in turn1                     # the context line's note is withheld


def test_obviousness_challenge_sees_the_pristine_paragraph():
    src = ["def f():", "    a = 1", "    b = 2", "    c = 3", "    return a + b + c"]
    llm = _FakeLLM(["Sets up b and c.", "3"])
    verifier = _FakeVerifier()
    request_block_comment(llm, _Cfg(), [], src, _make_target(src), 3, 4, PYTHON_STYLE,
                          line_notes={3: "noise: Should never appear."}, verifier=verifier)
    assert verifier.challenged == ["    b = 2\n    c = 3"], verifier.challenged


def main():
    test_window_clamps_to_the_routine()
    test_scope_nudge_reaches_past_the_plain_window()
    test_nudge_cap_bounds_the_walk()
    test_exactly_the_target_lines_are_marked()
    test_callee_notes_land_on_target_lines_only()
    test_obviousness_challenge_sees_the_pristine_paragraph()
    print("PASS: the comment turn's context window clamps to the routine, nudges back to the scope opener under the "
          "cap, marks exactly the target lines, keeps notes off context lines, and challenges the pristine paragraph")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
