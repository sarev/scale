#!/usr/bin/env python3
"""
The lazy callee one-liner generator (`scale._make_callee_oneliner_context`): when the def pass documents a caller, any
resolved callee with no contract gets a one-line summary generated from its body in the retained run-file store - one
level only, never for an uncalled routine - and the result is cached on the `ContractStore` so later callers reuse it
(a callee that yields nothing is not retried). A callee that already has a contract (an existing-doc seed) costs no
model call at all.

Model-free: the LLM is a fake that returns a canned one-liner and counts its calls.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_project as sp     # noqa: E402
import scale_python as spy     # noqa: E402
from scale import _make_callee_oneliner_context  # noqa: E402


@dataclass
class _Cfg:
    temperature: float = 0.2
    max_new_tokens: int = 256


class _FakeLLM:
    """Returns a canned reply, recording every prompt (so the tests can see what was - and was not - summarised)."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def snippet_budget(self, messages, cfg):
        return 100000      # never elide

    def estimate_tokens(self, text):
        return 1

    def generate(self, messages, cfg=None):
        self.prompts.append(messages[-1]["content"])
        return self.reply


UTIL = "def helper(x):\n    return x * 2\n\ndef lonely():\n    return 0\n"
CORE = "def run(xs):\n    return [helper(x) for x in xs]\n"


def _fixture(util_src=UTIL):
    """Build the retained store, graph, and contract store over a two-file run (core.py calls util.py's helper)."""
    run_files, symbols_by_file = {}, {}
    for key, blob in (("util.py", util_src), ("core.py", CORE)):
        lines = blob.split("\n")
        syms = spy.iter_symbols(blob, lines)
        run_files[key] = sp.RunFile(path=Path(key), key=key, is_target=True, source_blob=blob,
                                    source_lines=lines, language="python", symbols=syms)
        symbols_by_file[key] = syms
    graph = sp.build_project_graph(symbols_by_file)
    return run_files, graph, sp.ContractStore(graph)


def test_generates_lazily_and_caches():
    run_files, graph, store = _fixture()
    llm = _FakeLLM("Double x and return it.")
    context = _make_callee_oneliner_context(llm, _Cfg(), run_files, graph, store)

    # Documenting the caller generates the undocumented callee's one-liner (exactly one model call) from its body.
    notes = context("core.py", "run")
    assert "helper: Double x and return it." in notes
    assert len(llm.prompts) == 1 and "return x * 2" in llm.prompts[0]
    assert store.contract(("util.py", "helper")) == "Double x and return it."

    # Cached: a second caller turn makes no further model call.
    assert "helper:" in context("core.py", "run")
    assert len(llm.prompts) == 1

    # Lazy: `lonely` is called by nothing, so its body is never summarised.
    assert not any("lonely" in p for p in llm.prompts)


def test_documented_callee_needs_no_generation():
    documented = 'def helper(x):\n    """Double x."""\n    return x * 2\n'
    run_files, graph, store = _fixture(util_src=documented)

    class _Boom:
        def generate(self, *a, **k):
            raise AssertionError("a callee with an existing-doc seed must not be summarised")

    context = _make_callee_oneliner_context(_Boom(), _Cfg(), run_files, graph, store)
    assert "helper: Double x." in context("core.py", "run")   # the seed, no model work


def test_failed_generation_is_not_retried():
    run_files, graph, store = _fixture()
    llm = _FakeLLM("")    # the model yields nothing usable
    context = _make_callee_oneliner_context(llm, _Cfg(), run_files, graph, store)

    assert "helper" not in context("core.py", "run")          # no contract -> omitted from the notes
    assert len(llm.prompts) == 1
    context("core.py", "run")                                 # a later turn does not retry the failed callee
    assert len(llm.prompts) == 1


def main():
    test_generates_lazily_and_caches()
    test_documented_callee_needs_no_generation()
    test_failed_generation_is_not_retried()
    print("PASS: callee one-liners generate lazily from the retained store, cache on the ContractStore, never run "
          "for uncalled or already-documented routines, and do not retry failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
