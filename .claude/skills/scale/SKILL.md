---
name: scale
description: Annotate a Python source file with SCALE, using the local model for the bulk and escalating complex routines (cognitive complexity above a cutoff) to Claude for their comments/docstrings. Use when the user asks to "scale", annotate, or comment a Python file with the high-quality/escalation path.
---

# SCALE with Claude escalation

SCALE annotates code comments by *patching parsed source* — it never lets a model re-emit code, so executable code,
indentation, and existing comments survive byte-for-byte. This skill runs the **selective-escalation** flow: the local
GGUF model does the cheap bulk (file summary, structural segmentation, and comments for simple routines), while the
*complex* routines — those whose cognitive complexity exceeds the cutoff — are deferred to **you** (Claude) for their
comment/docstring prose. Your text is patched back through SCALE's same insertion-only path, so the guarantee holds
regardless of which model wrote the words.

This is **Python only** for now. The flow is three steps: emit → you answer → apply.

## Environment (this dev machine)

Always use the sibling virtualenv interpreter and the sibling model (see CLAUDE.md), from the project root
`D:\Programming\Aurora v2\scale`:

- Python: `../.llm-venv/Scripts/python.exe`
- Model (`-m`): `../models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF/Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf`
- Always pass `-l python`.

## Inputs

Parse from the user's request:
- `FILE` — the Python file to annotate (required).
- Passes — default `-c -b` (definition docstrings + within-function blocks). Honour the user if they ask for only one.
- `CUTOFF` — cognitive-complexity threshold for escalation; default **10** (`--escalate-cognitive`). Lower escalates
  more routines to you (better, costs more); higher keeps more on the local model.

Pick a working manifest path `MANIFEST` (e.g. `<FILE>.scale-manifest.json`) and an emit-output path
`EMITTED` (e.g. `<FILE>.scale-emit.py`). Annotate in place at the end unless the user gave `-o`.

## Step 1 — Emit (local model does the bulk, complex routines are deferred)

```
../.llm-venv/Scripts/python.exe scale.py -c -b -l python \
  -m "../models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF/Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf" \
  --escalate-cognitive 10 --emit-manifest "<MANIFEST>" \
  "<FILE>" -o "<EMITTED>" -v
```

This writes `EMITTED` (the file with local annotations; deferred routines left untouched) and `MANIFEST` (the requests
to answer). If the manifest has **zero** requests, nothing was complex enough to escalate — just move `EMITTED` into
place and report; you are done.

## Step 2 — Answer the manifest (you are the stronger model)

Read `MANIFEST`. It is JSON with a `requests` array. For each request, fill its `answer` slot(s). First read the house
style so your output matches the rest of the file:
- `scale-cfg/guidelines.md` and `scale-cfg/comment.python.txt` — docstring house style (definition pass).
- For block comments, keep it terse: **one** short, useful line per chunk — what the block accomplishes, or the
  reason / gotcha / subtlety behind it — the kind of line that helps a reader scan a long routine. Use `"NONE"`
  (the string) for a chunk that is genuinely self-evident. Do not restate a single obvious line.

Request shapes:
- `"pass": "def"` — write the **docstring body only** (no `"""`, no code fences) into the request's top-level
  `"answer"`. Use the provided `snippet` for context; follow the house style. For a `class`, summarise over its
  methods.
- `"pass": "block"` — the request has a `chunks` array; each chunk has `text` (the code lines) and an `answer` slot.
  Set each chunk's `"answer"` to one short comment line, or `"NONE"`. `doc_summary` (the routine's purpose) and
  `length_note` (short vs long bias) are there to guide you.

Write the filled JSON back to `MANIFEST` (Edit/Write the `answer` fields in place; leave `id`, `bidx`, `sig_hash`,
`text`, etc. untouched).

## Step 3 — Apply (model-free; patches your answers in)

```
../.llm-venv/Scripts/python.exe scale.py -l python \
  --apply-manifest "<MANIFEST>" "<EMITTED>" -o "<FINAL>" -v
```

No model loads. SCALE re-binds each answer to its routine and patches it through the same guard as the local passes.
Then move `FINAL` into place (overwrite `FILE` unless the user asked for a separate `-o`), and remove the scratch
`MANIFEST`/`EMITTED`/`FINAL` files.

## Report

Tell the user which routines were escalated (the manifest's `qualname` + `cognitive`), how many stayed local, and where
the result was written. Note that all code was preserved byte-for-byte (SCALE's guard enforces this).

## Notes

- To use the companion `codestats` tool as the source of complexity instead of SCALE's native scorer (e.g. to match a
  project's existing report), generate its JSON and pass `--codestats-json <report.json>` on the emit command.
- If the user wants a dry run, stop after Step 1 and show them the manifest (what would be escalated) without answering.
