# The online mode (`--online`, `scale_escalate.py`, `--emit-manifest`/`--apply-manifest`)

SCALE runs in one of **two modes**. **Offline** (the default): the local model writes everything; no manifest is
involved. **Online** (`--online --emit-manifest m.json`): *every* routine's comments and docstrings are deferred to
a stronger model (e.g. Claude Code) via one run-level JSON manifest. Dogfood review drove the split: escalated
output met the hand-curated quality bar while local-7B output carried a ~1-in-5 false-claim rate — and a tool whose
whole output must be proof-read isn't a serious tool, so the old *selective* escalation (cognitive-complexity
routing) was replaced by the whole-run deferral and the complexity machinery deleted.

Neither mode weakens the code guarantee, because the guarantee lives in the patcher, not the model: a stronger
model's reply flows through the *same* extractor + insertion-only patcher + `code_preserved` guard as a local one,
so it can only ever change comments/docstrings. **Python, C, and JS** targets are all supported online.

## Two phases around ONE run-level JSON manifest

(`scale_escalate` schema, version 2; v1 is read-upgraded.)

- **Emit** (`--online --emit-manifest`, any number of targets): **model-free and instant — the GGUF is never
  loaded.** Each target is parsed and every routine recorded as a **per-routine request** — the def collector and
  the block collector record into the *same* request keyed `(qualname, sig_hash)` — while the target itself is left
  **byte-for-byte untouched**. The request carries the routine's identity, ONE `snippet` (its verbatim source span),
  an optional `def` answer slot, and an optional `blocks` recipe (segmentation is structural and runs at emit; each
  chunk holds its boundary index `bidx` plus `lines`, its 1-based line range INTO the snippet — chunk text is never
  duplicated). The collectors are `scale_python.collect_def_requests`, `scale_c.collect_def_requests_c` (doc-site
  aware: a redirected definition is skipped and its header prototype requested instead, with the impl body as the
  prose source), `scale_javascript.collect_def_requests_js`, and the language-agnostic
  `scale_blocks.defer_block_targets`. `run_manifest` merges the per-target collectors, stamps each request with its
  `file`, and **dedupes byte-identical snippets** into a `snippet_ref` (e.g. an impl body that feeds a header
  prototype's prose and is itself deferred crosses the wire once). The local passes (`--file-doc`, bare
  `--block-spacing`, `--emit-reword`) have no online form and error out with a pointer.
- **Apply** (`--apply-manifest`, model-free, any number of targets): requests are routed to their file, re-bound by
  `(qualname, sig_hash)` (Python: `node_sig`, AST-structural; C: `routine_text_hash`, the verbatim span-text hash;
  JS: `_js_span_hash`, a comment/blank-stripped code-line hash, because applying a nested function's JSDoc inserts
  comment lines *inside* the parent's span — all shift-proof), then docstrings are patched and the text re-parsed
  before block answers are placed by boundary index (`scale_python.apply_manifest` / `scale_c.apply_manifest_c` /
  `scale_javascript.apply_manifest_js`).

The manifest carries a top-level **`doc_style`** (guidelines + the def-pass templates of the run's target languages)
so the stronger model writes deferred docs to house style.

## Completeness is a counter, not trust

`--check-manifest m.json` (model-free, no targets) prints every unfilled answer slot and exits nonzero while any
remains — the driver loops until it reports 0 (it also notes how many requests are checked out to outstanding
fragments). `null`/whitespace is unfilled; the explicit string `"NONE"` is a deliberate decline (a NONE block chunk
paragraphs with a blank but invents no comment; an unanswered request leaves its routine untouched). The same
checker dispatches the [file-description manifest](file-doc.md) (`scale-filedoc`) and the offline reword manifest by
their `tool` field.

## Fragments: SCALE owns the slot bookkeeping, so filling agents run in parallel

`--next-fragment m.json [--fragment-size N]` (model-free, no targets) checks out the next ≤N unfilled requests as a
**fragment** — a small, self-contained, valid mini-manifest written next to the master (`m.frag-001.json`, its path
printed; a monotonic `fragments_issued` counter never reuses a name). The selected requests are marked in the
master, so repeated calls hand out **disjoint** batches and any number of agents can fill their fragments
concurrently — no shared file, no agent ever reads the master or slices its own batch out of it. A `snippet_ref`
whose target lands in another fragment is inlined, so a fragment never sends its reader elsewhere; `doc_style`
travels in every fragment; the ordinary checker works on a fragment for agent self-checks. When nothing is left to
hand out the call exits nonzero (saying whether the manifest is complete or the remainder is checked out).

`--apply-manifest` is the merge gate: it first folds every sibling `*.frag-*.json` back into the master — matching
by `(id, file)` and chunk `bidx`, **first write wins** (a stale fragment can never clobber a filled slot) — writes
the merged master, deletes the spent fragment files, and then refuses to apply while any slot is unfilled: the
leftovers are listed, their checkout markers cleared (back into the pile for the next `--next-fragment`), and the
exit is nonzero. Only a complete master proceeds to the real apply, so the counter invariant is unchanged.

## The file-description round (a second manifest)

Online there is no local draft to reword, so after `--apply-manifest` the top-of-file descriptions get their own
round: model-free `--emit-filedoc fd.json` (each target's current skeleton + role + header-zone lines) → the
stronger model fills each file's `range` + `description` answer pair → model-free `--apply-filedoc fd.json`, which
runs the license veto and preservation guard locally. See [file-doc.md](file-doc.md).

## The `/scale` skill is the intended driver

(`.claude/skills/scale/`): model-free emit (instant; no model path needed) → Claude checks out fragments with
`--next-fragment` and fills them with **one fresh subagent per fragment, all in parallel** (fragments are
self-contained: `doc_style` + ~8 requests; the driver never reads code, it only hands out fragments and gates on the
merge) → model-free apply (merges fragments, errors incomplete rounds back to the fragment loop) → model-free
`--emit-filedoc` → Claude fills the file descriptions in a single context (prose-only, small) → model-free
`--apply-filedoc`. Every Claude touchpoint is bounded and machine-checked; offline, ONE local invocation does
everything and no Claude tokens are spent.
