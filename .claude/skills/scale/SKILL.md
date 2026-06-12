---
name: scale
description: Annotate source files (Python or C; multi-file runs supported) with SCALE, using the local model for the bulk and escalating complex routines, verification failures, and C header prototypes to Claude. Use when the user asks to "scale", annotate, or comment files with the high-quality/escalation path.
---

# SCALE with Claude escalation

SCALE annotates code comments by *patching parsed source* — it never lets a model re-emit code, so executable code,
indentation, and existing comments survive byte-for-byte. This skill runs the **selective-escalation** flow: the local
GGUF model does the cheap bulk (file summaries, structural segmentation, comments for simple routines, and its own
verification challenges), while the high-value items are deferred to **you** (Claude) through two bounded,
machine-checkable manifests:

1. the **function manifest** (`--emit-manifest`) — routines routed in by cognitive complexity, by a twice-failed
   verification challenge, or because they are C doc-site prototypes (public contracts);
2. the **header-reword manifest** (`--emit-reword`) — every file's freshly generated description, reworded by you
   with cross-file consistency (prose only; the splicing is never delegated to you).

Your text is patched back through SCALE's insertion-only patchers and code-preservation guards, so the guarantee
holds regardless of which model wrote the words. Escalation supports **Python and C** targets (JS runs locally only).

## Environment (this dev machine)

Always use the project virtualenv interpreter and the in-project model (see CLAUDE.md), from the project root
`D:\Programming\SCALE`:

- Python: `env/Scripts/python.exe`
- Model (`-m`): `models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF/Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf`
- Always pass `-l python` or `-l c` explicitly.

## Inputs

Parse from the user's request:
- `TARGETS` — the file(s)/dir(s)/glob(s) to annotate (required). Multiple targets are annotated **in place**
  (confirm the tree is committed/backed up first); a single target may use `-o`.
- Passes — default `-c --block-comments medium` (definition docs + within-function blocks). Honour the user if they
  ask for only one, or for `--file-doc`/`--doc-site` behaviour. For C header+source runs add `--doc-site auto` and
  pass headers and sources together.
- `CUTOFF` — cognitive-complexity threshold; default **10** (`--escalate-cognitive`). Lower escalates more to you.

Pick working paths: `MANIFEST` (e.g. `scale-manifest.json`) and `REWORD` (e.g. `scale-reword.json`).

## Step 1 — Emit (local model + challenges; deferred routines collected into ONE run manifest)

```
env/Scripts/python.exe scale.py -c --block-comments medium -l <LANG> \
  -m "models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF/Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf" \
  --escalate-cognitive 10 --emit-manifest "<MANIFEST>" \
  <TARGETS...> -v
```

(Single target: add `-o <EMITTED>` and work on that copy; multiple targets are written in place.)
Do **not** pass `--file-doc` here — the published file descriptions belong to Step 4, after your docstrings are in.

If the manifest has **zero** requests, skip to Step 4.

## Step 2 — Fill the function manifest (fragments, PARALLEL fresh contexts, counter loop)

You are the driver. **Never read the target source files yourself** — every fragment is self-contained, and SCALE
does the slot bookkeeping (no agent ever touches the master or another agent's file). Loop:

1. Check out the work as fragments — call repeatedly until it exits nonzero ("no fragment available" /
   "manifest complete"), collecting each printed fragment path:
   `env/Scripts/python.exe scale.py --next-fragment "<MANIFEST>" --fragment-size 8`
   Each call writes a small valid mini-manifest next to the master (e.g. `scale-manifest.frag-001.json`) holding
   the next ~8 unfilled requests, and marks them checked out in the master, so every fragment is disjoint.
2. Spawn one **fresh subagent** (Agent tool, `general-purpose`) per fragment, **all in parallel** (one message,
   multiple Agent calls) — fragments share nothing, and each clean context keeps quality flat as the run grows.
   Tell each subagent:
   - its fragment path; it should Read the file directly (fragments are small) and fill EVERY `answer` field in it;
   - to follow the fragment's top-level `doc_style` for every def answer;
   - the request shape (below);
   - to self-check before finishing: `--check-manifest <FRAGMENT>` must exit 0.
3. When all agents have returned, run the apply (Step 3). It merges every sibling fragment into the master
   (first write wins), deletes the spent fragment files, and — if any slot is still unfilled — errors, lists the
   slots, and returns them to the pile: go back to 1 and hand them out again. Never mark the step done on trust —
   only on the apply/checker exit code.

**Request shape (manifest version 2).** Each request is one routine: `qualname`, `kind`, `sig_hash`, `cognitive`,
`file`, and `snippet` — the routine's verbatim source (if `snippet` is null, follow `snippet_ref` to the request in
the same fragment that carries the identical text — a ref whose target is in another fragment arrives already
inlined; never duplicate the text back). It has either or both of:
- `"def": {"answer": null}` — write the **doc body only** (no `"""`/`/* */` delimiters, no fences) describing
  purpose, parameters, return value, per `doc_style`. For `kind: "declaration"` (a C header prototype) write the
  **caller-facing contract** — the snippet is the implementation body, but the doc sits above the prototype.
- `"blocks": {"doc_summary", "length_note", "chunks": [{"bidx", "lines", "answer"}]}` — each chunk's `lines` is the
  1-based inclusive line range **into the snippet**. Set each chunk's `answer` to ONE short, useful line (the
  paragraph's point, reason, or gotcha — never a restatement), or the string `"NONE"` for a chunk that is genuinely
  self-evident. Leave `bidx`, `lines`, `sig_hash` etc. untouched.

## Step 3 — Apply the function manifest (model-free; also the fragment merge gate)

```
env/Scripts/python.exe scale.py -l <LANG> --apply-manifest "<MANIFEST>" <TARGETS...> -v
```

No model loads. Sibling `*.frag-*.json` answers are merged into the master first (then the spent fragment files
are deleted); an incomplete master errors with the unfilled slots returned to the pile (loop back to Step 2.1).
Once complete, SCALE re-binds each answer by `(qualname, sig_hash)` and patches it through the same guards.

## Step 4 — Published descriptions + reword manifest (second local invocation)

```
env/Scripts/python.exe scale.py --file-doc -l <LANG> \
  -m "models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF/Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf" \
  --emit-reword "<REWORD>" <TARGETS...> -v
```

This regenerates each file's description from its **annotated** skeleton (your docstrings included), splices it into
the header, and writes the reword manifest: the project blurb plus, per file, its name, role
(header/implementation/other), the spliced `draft`, an optional richer `context`, and an `answer` slot.

## Step 5 — Fill the reword manifest (single context — it is prose-only and small)

Read `REWORD` yourself and fill every `answer`: reword each `draft` so the run's descriptions read as one consistent
set — no duplicate role claims, parallel phrasing for a `foo.c`/`foo.h` pair, each grounded in its draft (and
`context` where present). Flowing prose only, no lists/headings; `"NONE"` keeps a draft that is already right. Check
completeness the same way: `--check-manifest "<REWORD>"` must exit 0.

## Step 6 — Apply the reword (model-free)

```
env/Scripts/python.exe scale.py -l <LANG> --apply-reword "<REWORD>" <TARGETS...> -v
```

Each draft is located by exact match and replaced through the preservation guard; a miss is a safe no-op. Clean up
the scratch manifests when done.

## Without escalation

If the user wants a purely local run, ONE invocation does everything (priming-grade description → function passes
with challenges → published description spliced last): add `--file-doc` to the Step 1 command and skip the manifests.

## Report

Tell the user which routines were escalated and why (`qualname` + `cognitive`; verification promotions and C
prototypes too), how many stayed local, both checkers' final counts, and where the results were written. Note that
all code was preserved byte-for-byte (SCALE's guards enforce this).

## Notes

- Cognitive complexity is scored natively for Python, C, and JS; `--codestats-json <report.json>` overrides the
  native scores per-qualname with the companion codestats tool's report (rarely needed now).
- A dry run = stop after Step 1 and show the manifest.
- `--no-verify` disables the local grounding gate / challenge turns (faster, lower quality floor) — only on request.
