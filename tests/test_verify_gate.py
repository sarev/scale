#!/usr/bin/env python3
"""
The def-pass verification floor: the deterministic backtick-grounding gate and the clean-context grounding challenge,
with the failure routing (one corrective regeneration each; a second failure writes the doc under a prominent warning
- a visible contract beats a silent gap).

This guards (model-free, via a fake LLM):
- `ungrounded_tokens` (the gate itself): backticked identifiers absent from the run's source are flagged; grounded
  ones, single characters, and non-identifier text are not,
- the gate's one nudge: a corrected regeneration passes; a still-ungrounded one fails the routine,
- the grounding challenge's parse (NONE passes, a verdict fails) and its regenerate-with-verdict retry,
- the routing: a twice-failed docstring is still written (visible, with a warning) - never a silent gap,
- the persistent message list is balanced (append-then-pop) around the whole pipeline.
"""
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_python import generate_docstrings, iter_defs_with_info  # noqa: E402
from scale_verify import Verifier, ungrounded_tokens, _first_word_verdict  # noqa: E402


@dataclass
class _Cfg:
    temperature: float = 0.2
    max_new_tokens: int = 256


class _RouterLLM:
    """Routes each generate() call to a per-purpose reply queue by recognising the prompt's fixed wording."""

    def __init__(self, main=None, grounding=None):
        self.main = list(main or [])              # docstring generations and regenerations (worker context)
        self.grounding = list(grounding or [])    # grounding-challenge verdicts (clean context)
        self.grounding_calls = 0

    def snippet_budget(self, messages, cfg):
        return 100000      # never elide

    def estimate_tokens(self, text):
        return 1

    def generate(self, messages, cfg=None):
        prompt = messages[-1]["content"]
        if "Reply with NONE or the list only" in prompt:
            self.grounding_calls += 1
            assert len(messages) == 1, "a challenge must run in a clean, single-turn context"
            return self.grounding.pop(0)
        return self.main.pop(0)


SRC = 'def f(x):\n    total = x + 1\n    return total\n'
CORPUS = SRC


def _doc(text: str) -> str:
    """Wrap a docstring body as the fenced reply the worker extracts."""
    return f'"""\n{text}\n"""'


def _run(llm):
    messages = []
    defs = iter_defs_with_info(ast.parse(SRC))
    verifier = Verifier(llm, _Cfg(), corpus=CORPUS)
    doc_map = generate_docstrings(llm, _Cfg(), messages, defs, SRC, SRC.split("\n"), verifier=verifier)
    assert messages == [], "the persistent message list must be balanced after the def pass"
    return defs, doc_map


def test_gate_tokeniser():
    corpus = "int err_read(FILE *fp) { return ERR_READ; }"
    # An invented identifier is flagged; a grounded one is not - even when a shorter real name is its prefix.
    assert ungrounded_tokens("Returns `ERR_READ_ERROR` on failure.", corpus) == ["ERR_READ_ERROR"]
    assert ungrounded_tokens("Returns `ERR_READ` on failure.", corpus) == []
    # Several words inside one span are each checked; duplicates are reported once; 1-char words are skipped.
    assert ungrounded_tokens("Calls `bogus_fn(x)` then `bogus_fn`.", corpus) == ["bogus_fn"]
    assert ungrounded_tokens("Loop variable `i`.", corpus) == []
    # No backticks, no flags - the gate only judges backticked identifiers.
    assert ungrounded_tokens("Handles MADE_UP_THING gracefully.", corpus) == []
    assert ungrounded_tokens("", corpus) == [] and ungrounded_tokens(None, corpus) == []


def test_verdict_parse():
    assert _first_word_verdict("NONE", "NONE") == "NONE"
    assert _first_word_verdict("  none.", "NONE") == "NONE"
    assert _first_word_verdict("- claims it retries", "NONE") is None
    assert _first_word_verdict("YES", "YES", "NO") == "YES"
    assert _first_word_verdict("No, it restates.", "YES", "NO") == "NO"
    assert _first_word_verdict("NOTE: unclear", "YES", "NO") is None      # word boundaries: NOTE is not NO
    assert _first_word_verdict("", "YES", "NO") is None


def test_gate_nudge_corrects():
    # First doc invents `ERR_BOGUS`; the nudge regeneration removes it; the grounding challenge then passes.
    llm = _RouterLLM(
        main=[_doc("Add one to `x`, reporting `ERR_BOGUS` on failure."), _doc("Add one to `x` and return it.")],
        grounding=["NONE"],
    )
    defs, doc_map = _run(llm)
    assert doc_map[id(defs[0].node)] == "Add one to `x` and return it."
    assert llm.grounding_calls == 1


def test_gate_failure_writes_with_warning():
    # Both attempts invent an identifier: the doc is still written (a visible contract beats a silent gap).
    llm = _RouterLLM(
        main=[_doc("Uses `ERR_BOGUS`."), _doc("Still uses `ERR_BOGUS`.")],
        grounding=[],
    )
    defs, doc_map = _run(llm)
    assert doc_map[id(defs[0].node)] == "Still uses `ERR_BOGUS`."
    assert llm.grounding_calls == 0, "a gate-failed doc must not reach the grounding challenge"


def test_challenge_regenerates_then_passes():
    # The first doc claims behaviour the code lacks; the verdict-fed regeneration passes the re-challenge.
    llm = _RouterLLM(
        main=[_doc("Add one to `x`, retrying on overflow."), _doc("Add one to `x` and return it.")],
        grounding=["- claims it retries on overflow; the code does not retry", "NONE"],
    )
    defs, doc_map = _run(llm)
    assert doc_map[id(defs[0].node)] == "Add one to `x` and return it."
    assert llm.grounding_calls == 2


def test_twice_failed_is_written_with_a_warning():
    # The challenge fails twice -> the docstring is still written (warn-and-write: the doubt is visible, never silent).
    llm = _RouterLLM(
        main=[_doc("Add one, retrying on overflow."), _doc("Add one, with exponential backoff.")],
        grounding=["- claims it retries", "- claims backoff"],
    )
    defs, doc_map = _run(llm)
    assert doc_map[id(defs[0].node)] == "Add one, with exponential backoff.", \
        "the last attempt must be written despite the failed challenge"
    assert llm.grounding_calls == 2, "exactly one regeneration is allowed"


def test_clean_pass_costs_one_challenge_turn():
    llm = _RouterLLM(main=[_doc("Add one to `x` and return the `total`.")], grounding=["NONE"])
    defs, doc_map = _run(llm)
    assert doc_map[id(defs[0].node)] == "Add one to `x` and return the `total`."
    assert llm.main == [] and llm.grounding == [], "no extra generations on a clean pass"


def main():
    test_gate_tokeniser()
    test_verdict_parse()
    test_gate_nudge_corrects()
    test_gate_failure_writes_with_warning()
    test_challenge_regenerates_then_passes()
    test_twice_failed_is_written_with_a_warning()
    test_clean_pass_costs_one_challenge_turn()
    print("PASS: the grounding gate and grounding challenge verify def docs, with warn-and-write failure routing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
