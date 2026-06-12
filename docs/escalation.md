# Selective escalation to a stronger model (`scale_escalate.py`, `--emit-manifest`/`--apply-manifest`)

The local 7B is "good enough" for the bulk, but the comments worth a stronger model are exactly those for the most
involved routines — where the 7B's prose gets unreliable. This feature routes that split **without weakening the
code guarantee**, because the guarantee lives in the patcher, not the model: a stronger model's reply flows through
the *same* extractor + insertion-only patcher + `code_preserved` guard as the local one, so it can only ever change
comments/docstrings. **Python and C** targets escalate (JS runs locally only).

## Three routing signals

1. **Cognitive complexity**: every language has a **native scorer** — `scale_python.cognitive_complexity(node)` on
   SCALE's own `ast`, plus its tree-sitter mirrors `scale_c.cognitive_complexity_c(node)` and
   `scale_javascript.cognitive_complexity_js(node)` — all computing the same SonarSource-style score (`+1`/`+nesting`
   per `if`/`for`/`while`/`do`/`switch`/`try`/ternary; `+1` per `elif`/`else`/`except`/`catch`/`finally`
   continuation, with C/JS `else if` chains folded so they read as cheap continuations rather than nested ifs; `+1`
   per boolean-operator *sequence*, runs of the same operator collapsed; nested defs opaque; a bodyless C prototype
   scores 0), so one `--escalate-cognitive` cutoff is meaningful across languages. A routine escalates when its
   score **exceeds** `--escalate-cognitive` (default 10). `--codestats-json <report>` overrides the native score
   per-qualname. The scorers also stamp `BlockTarget.cognitive` in every block provider, so block-pass escalation
   routes natively too (JS carries the score but does not escalate).
2. **Verification failures** (see [verification.md](verification.md)): a doc/note that fails its gate or challenge
   twice is promoted, discarding the local attempt.
3. **C doc-site redirected prototypes**: with a manifest active they are *always* deferred — a public contract is
   the highest value per stronger-model token.

## Two phases around ONE run-level JSON manifest

(`scale_escalate` schema, version 2; v1 is read-upgraded.)

- **Emit** (`--emit-manifest`, any number of targets): the passes run with the local model as usual, but a deferred
  routine gets a **per-routine request** — both passes record into the *same* request keyed `(qualname, sig_hash)` —
  and is left **byte-for-byte untouched** in the emitted output. The request carries the routine's identity, ONE
  `snippet` (its verbatim source span), an optional `def` answer slot, and an optional `blocks` recipe (segmentation
  still runs locally; each chunk holds its boundary index `bidx` plus `lines`, its 1-based line range INTO the
  snippet — chunk text is never duplicated). When both passes record, the **block recording's snippet wins**: its
  chunk ranges were computed against the block pass's view of the source, which can be longer than the def pass's
  (nested routines gain docstrings in between). `run_manifest` merges the per-target collectors, stamps each request
  with its `file`, and **dedupes byte-identical snippets** into a `snippet_ref` (e.g. an impl body that feeds a
  header prototype's prose and is itself deferred crosses the wire once).
- **Apply** (`--apply-manifest`, model-free, any number of targets): requests are routed to their file, re-bound by
  `(qualname, sig_hash)` (Python: `node_sig`, AST-structural; C: `routine_text_hash`, the span-text hash — both
  shift-proof because the routine was untouched), then docstrings are patched and the text re-parsed before block
  answers are placed by boundary index (`scale_python.apply_manifest` / `scale_c.apply_manifest_c`).

## Completeness is a counter, not trust

`--check-manifest m.json` (model-free, no targets) prints every unfilled answer slot and exits nonzero while any
remains — the driver loops until it reports 0. `null`/whitespace is unfilled; the explicit string `"NONE"` is a
deliberate decline (a NONE block chunk paragraphs with a blank but invents no comment; an unanswered request leaves
its routine untouched).

## The `/scale` skill is the intended driver

(`.claude/skills/scale/`): emit run → Claude fills the function manifest in **batches of ~10 requests, each batch in
a fresh subagent context** (requests are self-contained: `doc_style` + the batch; the driver never reads code, it
only counts via the checker and loops) → model-free apply → a second local invocation (`--file-doc --emit-reword`:
pass-2 descriptions + header splice + reword manifest, see [file-doc.md](file-doc.md)) → Claude fills the reword
manifest in a single context (prose-only, small) → model-free `--apply-reword`. Two bounded Claude touchpoints, each
with a machine-checked completion condition; without escalation, ONE local invocation does everything and no Claude
tokens are spent.

The manifest carries a top-level **`doc_style`** (guidelines + the def-pass templates of the run's
escalation-capable languages) so the stronger model writes deferred docstrings to house style. The def-pass
"unusable reply" nudge (`DOCSTRING_NUDGE`) still runs first; promotion only happens after it fails too.
