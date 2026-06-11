#!/usr/bin/env python3
"""
Two guards against the "OK" docstring regression and its recovery path:

- Priming no longer makes the model parrot "OK": no assistant turn is a bare acknowledgement (a fixed PRIMING_ACK is
  supplied instead), the priming text carries no "say OK" instruction, and the only generation during priming is the
  whole-file summary.
- The definition pass recovers from an unusable reply: a bare-"OK" docstring triggers one nudge, and if that still
  fails the routine is PROMOTED to the manifest for the stronger model (when escalation is active) rather than written
  as a placeholder. A good reply is used as-is with no nudge; with no manifest, a persistent failure falls back to the
  placeholder.

No GGUF model required.
"""
import ast
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale import prime_llm_for_comments, SummaryCache  # noqa: E402
from scale_text import PRIMING_ACK  # noqa: E402
from scale_python import generate_docstrings, iter_defs_with_info, _looks_like_ack  # noqa: E402
from scale_escalate import Escalation  # noqa: E402
from scale_llm import GenerationConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCALE_CFG = ROOT / "scale-cfg"
SRC = "def helper(x):\n    y = x + 1\n    return y\n"
GOOD = '"""\nReturn x plus one.\n\nParameters:\n- `x`: the value.\n\nReturns:\n- `x + 1`.\n"""'


class Stub:
    """Stub model returning a fixed reply (or a summary for summary prompts); counts generations; roomy budget."""
    n_ctx = 12288
    ctx_margin = 256

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def estimate_tokens(self, text):
        return len(text) // 4

    def count_tokens(self, messages):
        return sum(self.estimate_tokens(m["content"]) for m in messages)

    def snippet_budget(self, messages, cfg, **kwargs):
        return 9000

    def generate(self, messages, cfg=None, stop=None):
        self.calls += 1
        if "source file" in messages[-1]["content"]:
            return "A one-line module overview."     # the file-description summary (full or short squash)
        return self.reply


def _gen(stub, defs, escalation):
    return generate_docstrings(stub, GenerationConfig(max_new_tokens=512),
                               [{"role": "system", "content": "system"}], defs, SRC, SRC.split("\n"),
                               escalation=escalation)


def main():
    # ---- 1. Priming never parrots "OK" ----
    with tempfile.TemporaryDirectory() as tmp:
        SummaryCache._CACHE_DIR = Path(tmp)
        SummaryCache._CACHE_INDEX = Path(tmp) / "index.pkl"
        stub = Stub("unused")
        msgs = prime_llm_for_comments(stub, GenerationConfig(), SCALE_CFG, Path("x.py"),
                                      source_blob=SRC, language="python", no_cache=True)

    # The definition pass primes with the SHORT file description. With no full description cached and no file-doc/block
    # pass to need one, the short is generated DIRECTLY from the source - one summary call, not a full-then-condense
    # pair - and no ack turns are generated (they are supplied).
    assert stub.calls == 1, f"priming should generate the short summary directly (one call), not acks; made {stub.calls}"
    assert not any(m["role"] == "assistant" and m["content"].strip().upper() == "OK" for m in msgs), \
        "no priming turn may be a bare 'OK' (that conditions the model to reply OK to the first real task)"
    assert any(m["role"] == "assistant" and m["content"] == PRIMING_ACK for m in msgs), \
        "the fixed neutral acknowledgement should be used instead"
    say_ok = re.compile(r"(?i)\b(say|saying|state|stating|reply|respond)\b[^.\n]{0,20}\bok\b")
    assert not any(say_ok.search(m["content"]) for m in msgs), \
        "the priming text must not instruct the model to say OK"

    # ---- 2. A parroted acknowledgement (and a bare OK) count as unusable ----
    assert _looks_like_ack("OK") and _looks_like_ack(PRIMING_ACK), "OK and the priming-ack echo must be unusable"
    assert not _looks_like_ack("Return x plus one."), "a real docstring must not be mistaken for an acknowledgement"

    # ---- 3. Unusable local docstring is promoted to the manifest (escalation active) ----
    # Use the priming ack itself as the reply - the exact failure seen on `summarise` in the real run.
    esc = Escalation(threshold=999)            # complexity never escalates; force the local path then failure-promote
    stub_ack = Stub(PRIMING_ACK)
    dm = _gen(stub_ack, iter_defs_with_info(ast.parse(SRC)), esc)
    assert dm == {}, "a parroted-ack reply must not be written as a docstring"
    assert stub_ack.calls == 2, f"expected one attempt + one nudge, got {stub_ack.calls}"
    assert [r["qualname"] for r in esc.requests] == ["helper"], "an uncoaxable docstring must be promoted to the manifest"
    assert esc.requests[0].get("def") is not None

    # ---- 4. No manifest: a persistent failure falls back to the placeholder ----
    dm2 = _gen(Stub("OK"), iter_defs_with_info(ast.parse(SRC)), None)
    assert any("comment generation failed" in v for v in dm2.values()), "without a manifest, failure uses the placeholder"

    # ---- 5. A good reply is used as-is, with no nudge and no escalation ----
    esc3 = Escalation(threshold=999)
    stub_good = Stub(GOOD)
    dm3 = _gen(stub_good, iter_defs_with_info(ast.parse(SRC)), esc3)
    assert stub_good.calls == 1, "a usable docstring must not trigger a nudge"
    assert len(dm3) == 1 and "Return x plus one" in next(iter(dm3.values()))
    assert esc3.requests == [], "a good local docstring must not be escalated"

    print("PASS: priming never parrots OK; def pass nudges then promotes unusable docstrings, uses good ones as-is")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
