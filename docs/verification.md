# Verification: the grounding gate and challenge turns (`scale_verify.py`)

The local quality floor. Default-on for the def/block passes; `--no-verify` disables.

The local model's worst failures are confident inventions: identifiers that don't exist, behavioural claims the code
doesn't make, "comments" that restate the line below. A run-level `Verifier` (built in `main` over the retained
run-file store's concatenated source) checks each candidate *after* generation — structural fixes in clean contexts,
not longer generation prompts:

- **The backtick-grounding gate** (deterministic, model-free): house style backticks identifiers, so every
  identifier-shaped word inside a backticked span of a generated doc/comment must appear as a substring of the run's
  source text (convention-free, case-sensitive; would have caught the invented `ERR_*` names). On a miss: one nudge
  naming the offending tokens, then the shared failure routing.
- **Challenge turns** — small, single-aspect questions in a CLEAN context (a fresh single-turn message list: no
  priming, no summary, no priors; temp 0; constrained NONE / YES-NO / STORY-RESTATE replies so parsing stays
  robust): the **grounding challenge** (def docs: "list anything this comment claims that the code shown does not
  do"), the **obviousness challenge** (each insertable block note: "does this tell the reader anything not evident
  from the code?" — kills restatement and purpose-clause score-gaming), and the **story challenge** (per routine,
  after its chunks: do the notes as a set tell the routine's story or restate it? — length-guarded by
  `SHORT_FUNCTION_CHUNKS` and skipped when nothing would be written).
- **Failure routing**: a failure regenerates ONCE with the verdict as feedback (in the worker's own context); on a
  second failure, a block note is **dropped** (kept as `{priors}` context under `CHALLENGE_FLAG`, never written —
  wrongness is worse than absence; a twice-failed story drops the routine's whole note set while keeping the
  paragraphing blanks) and a def doc is **written under a prominent stderr warning** (a visible contract beats a
  silent gap). When the local floor isn't good enough, the answer is the [online mode](escalation.md), not
  per-routine promotion.

Wording lives in `scale-cfg/verify.*.txt` (gate nudge, the three challenge prompts, and their feedback texts),
overriding constants in `scale_verify.py` — see [prompt-tuning.md](prompt-tuning.md).
