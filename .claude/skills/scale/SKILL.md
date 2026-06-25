---
name: scale
description: Annotate source files (Python, C, or JavaScript; multi-file runs supported) with SCALE's online mode, deferring every routine's comments to Claude via the run manifest. Use when the user asks to "scale", annotate, or comment files with the high-quality/online path.
---

# SCALE online with Claude

SCALE annotates code comments by *patching parsed source* — it never lets a model re-emit code, so executable code,
indentation, and existing comments survive byte-for-byte. This skill runs the **online** flow: SCALE's model-free
machinery does the structure (parsing, segmentation, placement, guards, counters) while ALL the comment prose is
written by **you** (Claude) through two bounded, machine-checkable manifests:

1. the **function manifest** (`--online --emit-manifest`) — every routine's docstring/header-comment slot and
   within-function block-comment recipe;
2. the **file-description manifest** (`--emit-filedoc`) — every file's top-of-file description, written from its
   annotated skeleton (the splicing is never delegated to you).

Your text is patched back through SCALE's insertion-only patchers and code-preservation guards, so the guarantee
holds regardless of which model wrote the words. **Python, C, and JavaScript** targets are all supported, and no
GGUF model is ever loaded — every phase here is model-free and fast.

## Environment (this dev machine)

Always use the project virtualenv interpreter (see CLAUDE.md), from the project root `D:\Programming\SCALE`:

- Python: `env/Scripts/python.exe`
- Always pass `-l python`, `-l c`, or `-l js` explicitly.

## Inputs

Parse from the user's request:
- `TARGETS` — the file(s)/dir(s)/glob(s) to annotate (required). Multiple targets are annotated **in place**
  (confirm the tree is committed/backed up first); a single target may use `-o`.
- Passes — default `-c --block-comments medium` (definition docs + within-function blocks). Honour the user if they
  ask for only one. For C header+source runs add `--doc-site auto` and pass headers and sources together.

Pick working paths: `MANIFEST` (e.g. `scale-manifest.json`) and `FILEDOC` (e.g. `scale-filedoc.json`).

## Step 1 — Emit (model-free and instant; every routine collected into ONE run manifest)

```
env/Scripts/python.exe scale.py -c --block-comments medium -l <LANG> --online \
  --emit-manifest "<MANIFEST>" <TARGETS...> -v
```

(Single target: add `-o <EMITTED>` and work on that copy; multiple targets are referenced in place — emit writes
nothing into them.) No model path is needed; the emit completes in seconds. Do **not** pass `--file-doc` — online,
the file descriptions are Step 4's manifest round, after your docstrings are in.

If the manifest has **zero** requests, skip to Step 4.

## Step 2 — Fill the function manifest (round-based: a wave at a time, banked between rounds)

You are the driver. **Never read the target source files yourself** — every fragment is self-contained, and SCALE
does the slot bookkeeping (no agent ever touches the master or another agent's file).

**Work in rounds — do not check out the whole manifest up front.** For a large run (hundreds of routines) that would
mean dozens of simultaneous agents: wasteful and hard to oversee. Instead each round checks out a *wave*, fills it,
then applies to bank the finished work and learn what is left. The source is only ever patched once the master is
fully filled, so applying between rounds is safe and purely additive. Repeat the loop until the apply patches the
source:

1. **Check out one wave** — call `--next-fragment` up to **~15–20 times**, collecting each printed fragment path;
   stop at the wave cap even if more remain (the next round gets them). Stop early if it exits nonzero ("no fragment
   available" / "manifest complete").
   `env/Scripts/python.exe scale.py --next-fragment "<MANIFEST>" --fragment-size 8`
   Each call writes a small valid mini-manifest next to the master (e.g. `scale-manifest.frag-001.json`), marks those
   requests checked out in the master so fragments never overlap. **`--fragment-size` counts ROUTINES, not
   comments** — a routine averages ~2.4 block chunks plus maybe a def slot, so size 8 is roughly 20–25 answers; size
   a wave by answers with that ratio in mind.
2. **Spawn one fresh subagent per fragment, all in parallel** (Agent tool, `general-purpose`; one message, multiple
   Agent calls) — fragments share nothing, and each clean context keeps quality flat as the run grows. Tell each
   subagent:
   - its fragment path; it should Read the file directly (fragments are small) and fill EVERY `answer` field in it;
   - to follow the fragment's top-level `doc_style` for every def answer;
   - the request shape, fill order, and recommended fill method (below);
   - to self-check before finishing: `--check-manifest <FRAGMENT>` must exit 0.
3. **Apply to bank the round** (Step 3). It merges this wave's fragments into the master (first write wins), deletes
   the spent fragment files, and — while any slot is still unfilled — errors, lists what is missing, and returns the
   released requests to the pile. That is the signal to run the next round: back to 1. When the master is finally
   complete the same command patches the source and the loop ends. Never mark the step done on trust — only on the
   apply/checker exit code.

**Recommended fill method (for each subagent).** Hand-editing the `null`s is fragile — `"answer": null` is not
unique, so a blind find/replace is unsafe. Instead assign answers programmatically: load the fragment JSON, walk
`requests[]` in order and within each its block `chunks[]` in array order (then its `def` slot), set each `answer`,
assert no `answer` remains null, and dump the file back. The `--check-manifest` gate then confirms completeness. (A
chunk already carrying `"answer": "NONE"` with `"preserve": true` is pre-filled — skip it, do not overwrite.)

**Request shape (manifest version 2).** Each request is one routine: `qualname`, `kind`, `sig_hash`, `file`, and
`snippet` — the routine's verbatim source (if `snippet` is null, follow `snippet_ref` to the request in the same
fragment that carries the identical text — a ref whose target is in another fragment arrives already inlined; never
duplicate the text back). It has either or both of:
- `"def": {"answer": null}` — write the **doc body only** (no `"""`/`/* */` delimiters, no fences) describing
  purpose, parameters, return value, per `doc_style`. For `kind: "declaration"` (a C header prototype) write the
  **caller-facing contract** — the snippet is the implementation body, but the doc sits above the prototype.
- `"blocks": {"doc_summary", "length_note", "chunks": [{"bidx", "lines", "anchor", "answer"}]}` — each chunk is one
  paragraph inside the routine. `anchor` is the **verbatim text of the line the comment attaches to**: find that
  line in the snippet to see exactly which block the chunk covers — do not count lines (bodies thick with comments
  and blanks make counting drift). `lines` is the 1-based inclusive range into the snippet and disambiguates if an
  anchor happens to repeat. Set each chunk's `answer` to ONE short, useful line (the paragraph's point, reason, or
  gotcha — never a restatement), or the string `"NONE"` for a chunk that is genuinely self-evident. Leave `bidx`,
  `lines`, `anchor`, `sig_hash` etc. untouched.
  - **Do not clobber good prior work.** A chunk may carry `existing` — the comment already on that block. If it is
    adequate, answer `"NONE"` to keep it untouched; only write when the block has no comment or a poor one. A chunk
    that arrives already `"answer": "NONE"` with `"preserve": true` is a multi-line comment SCALE has protected for
    you — leave it exactly as is.

**Fill order.** For a request with both slots, fill the **block chunks first, in body order** (one line or `"NONE"`
each — by the last chunk you have read the whole routine), **then** write the `def` answer from that understanding.
Never echo code into any answer.

## Step 3 — Apply the function manifest (model-free; also the fragment merge gate)

```
env/Scripts/python.exe scale.py -l <LANG> --apply-manifest "<MANIFEST>" <TARGETS...> -v
```

No model loads. Sibling `*.frag-*.json` answers are merged into the master first (then the spent fragment files
are deleted); an incomplete master errors with the unfilled slots returned to the pile (loop back to Step 2.1).
Once complete, SCALE re-binds each answer by `(qualname, sig_hash)` and patches it through the same guards.

## Step 4 — Emit the file-description manifest (model-free; run on the APPLIED outputs)

```
env/Scripts/python.exe scale.py -l <LANG> --emit-filedoc "<FILEDOC>" <TARGETS...> -v
```

This records, per file: its `role` (header/implementation/other), its current structural `skeleton` (your freshly
applied docs included; a no-symbol file rides whole), and `entries` — the header zone's existing comment lines,
numbered from 1. The top-level `description_spec` is the spec each description must be written to, and
`project_doc` is the project's own overview for shared context.

## Step 5 — Fill the file-description manifest (single context — it is prose-only and small)

Read `FILEDOC` yourself and fill every file's `answer` pair:
- `"range"`: which numbered `entries` are the EXISTING description — `"START-END"`, a single number, or `"NONE"`
  when none of them is descriptive prose (shebang/copyright/licence/filename lines are NOT description). Never
  include a licence/legal line in the range (SCALE vetoes it locally anyway).
- `"description"`: the file's description, written from its `skeleton` per `description_spec` (flowing prose, no
  lists/headings), consistent across the run's files — or the string `"NONE"` to leave a file untouched.
Self-check: `--check-manifest "<FILEDOC>"` must exit 0 (both halves of every answer are required).

## Step 6 — Apply the file descriptions (model-free)

```
env/Scripts/python.exe scale.py -l <LANG> --apply-filedoc "<FILEDOC>" <TARGETS...> -v
```

Each answer is spliced through the same licence veto and preservation guard as the offline `--file-doc` pass; a
header zone that changed since emit is a safe no-op. Clean up the scratch manifests when done.

## Without Claude (offline)

If the user wants a purely local run, ONE invocation does everything (priming-grade description → function passes
with verification challenges → published description spliced last) — drop `--online`/the manifests, add
`--file-doc`, and pass the local model:
`env/Scripts/python.exe scale.py -c --block-comments medium --file-doc -l <LANG> -m "models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF/Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf" <TARGETS...> -v`

## Report

Tell the user how many routines and files were deferred and filled, both checkers' final counts, and where the
results were written. Note that all code was preserved byte-for-byte (SCALE's guards enforce this).

## Notes

- A dry run = stop after Step 1 and show the manifest.
- The emit/check/fragment/apply phases never load a model, so they are all fast; only the offline fallback needs
  the GGUF.
- **House style:** to pin a project's own comment templates without touching SCALE's global defaults, point
  `--config-dir <DIR>` at a folder of override prompts (or drop a `scale-cfg/`/`.scale-cfg/` beside the targets — it
  is discovered automatically). It overlays the built-ins per file, so it need only carry the templates it changes.
- **Line length:** pass `--line-length N` to wrap inserted block comments to N columns; online, set it on the emit
  (it is stored in the manifest) and/or the apply. Left off, comments are inserted unwrapped.

## Driver notes (operator pitfalls, not tool bugs)

- **Do not pre-plan the whole fan-out.** Work in rounds (Step 2): check out a wave, fill, apply, and let the apply
  tell you what is left rather than cutting every fragment up front.
- **Force UTF-8 when verifying.** Reading `git show HEAD:<file>` (or any source) without an explicit UTF-8 decode
  renders em dashes, ellipses and bullets as the platform default and fakes a "corruption" diff — SCALE preserves
  non-ASCII correctly. Always compare with an explicit UTF-8 decode.
