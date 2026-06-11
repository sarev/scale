#!/usr/bin/env python3
"""
The block pass's read-side callee annotations: `scale._block_callee_notes` re-parses the CURRENT (def-pass-annotated,
line-shifted) text and maps each routine's call lines to "callee: one-liner" notes via the graph's `call_map` and the
contract store; `request_block_comment` shows those notes appended to the call lines (and only the call lines) of the
paragraph it sends to the model; and the annotation is never written to the output - the patcher works from the
pristine source.

Model-free: a fake LLM records the prompts and returns canned summary/score replies.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_project as sp     # noqa: E402
import scale_python as spy     # noqa: E402
from scale import _block_callee_notes  # noqa: E402
from scale_blocks import BlockTarget, PYTHON_STYLE, request_block_comment, annotate_blocks  # noqa: E402


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


def test_block_callee_notes_track_shifted_lines():
    # The graph is built from the ORIGINAL text; the block pass then reads the def pass's output, where a freshly
    # inserted docstring has shifted every line. The annotator re-parses the current text, so the note lands on the
    # call's line in the text the block pass actually reads.
    original = "def run(xs):\n    return [helper(x) for x in xs]\n"
    util = "def helper(x):\n    return x * 2\n"
    graph = sp.build_project_graph({
        "core.py": spy.iter_symbols(original, original.split("\n")),
        "util.py": spy.iter_symbols(util, util.split("\n")),
    })
    store = sp.ContractStore(graph)
    store.update("util.py", "helper", "Double x.")     # as the def pass (or the lazy generator) would

    current = 'def run(xs):\n    """Run them all."""\n    return [helper(x) for x in xs]\n'
    notes = _block_callee_notes(spy.iter_symbols, current, current.split("\n"), "core.py", graph, store)
    assert notes == {"run": {3: "helper: Double x."}}, notes     # line 3 in the CURRENT text (was line 2)

    # A routine with no resolved-and-documented calls contributes no entry at all.
    plain = "def run(xs):\n    return [str(x) for x in xs]\n"
    assert _block_callee_notes(spy.iter_symbols, plain, plain.split("\n"), "core.py", graph, store) == {}


SRC = ["def f():", "    y = helper(1)", "    z = y + 1", "    return z"]


def _target(segments=None):
    return BlockTarget(qualname="f", kind="def", header_start=1, header_end=1, body_start=2, body_end=4,
                       boundary_lines=(2, 3, 4), indent_of={2: "    ", 3: "    ", 4: "    "},
                       doc="Does a thing.", segments=segments)


def test_annotation_shown_to_model_on_call_lines_only():
    llm = _FakeLLM(["Delegate the doubling to helper.", "3"])
    messages = []
    out = request_block_comment(llm, _Cfg(), messages, SRC, _target(), 2, 4, PYTHON_STYLE,
                                line_notes={2: "helper: Double x."})
    assert out == "Delegate the doubling to helper."
    assert messages == []                                        # append-then-pop preserved
    turn1 = llm.prompts[0]
    assert "y = helper(1)  # helper: Double x." in turn1         # the call line carries the note...
    assert "z = y + 1  #" not in turn1                           # ...and only the call line
    assert "return z  #" not in turn1


def test_annotation_is_never_written_to_the_output():
    llm = _FakeLLM(["Delegate the doubling to helper.", "3"])
    out = annotate_blocks(llm, _Cfg(), [], SRC, [_target(segments=[(2, 4)])], PYTHON_STYLE,
                          callee_annotations={"f": {2: "helper: Double x."}})
    assert "    # Delegate the doubling to helper." in out       # the scored comment is inserted
    assert not any("helper: Double x." in ln for ln in out), out  # the read-side note never reaches the file
    assert [ln for ln in out if ln.strip() and not ln.lstrip().startswith("#")] == SRC   # code untouched


def main():
    test_block_callee_notes_track_shifted_lines()
    test_annotation_shown_to_model_on_call_lines_only()
    test_annotation_is_never_written_to_the_output()
    print("PASS: block-pass callee annotations follow line shifts, mark only call lines in the model's view, and are "
          "never written to the output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
