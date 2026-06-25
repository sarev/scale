# Editing the LLM's behaviour without code changes (`scale-cfg/`)

Prompt behaviour lives in `scale-cfg/`. Tuning these examples/wording to match the target codebase is the primary
lever for output quality — prefer this over changing code. Each file overrides a built-in default constant, so a
missing file is never an error.

**Per-project overrides.** To tune behaviour for one codebase without editing the shared install, point
`--config-dir DIR` at a directory of override files (or drop a `scale-cfg/`/`.scale-cfg/` at or above the working
directory — it is discovered automatically). It overlays the built-in `scale-cfg/` *per file*: an override directory
need only carry the templates it changes, inheriting the rest. The names below are identical; only the directory
differs.

- **Definition pass**: `comment.txt` is the system prompt and `guidelines.md` the house-style rules (definition pass
  only); `comment.python.txt` / `comment.js.txt` / `comment.c.txt` are per-language definition-pass templates (with
  examples).
- **Block pass**: the per-language priming templates `blocks.python.txt` / `blocks.c.txt` / `blocks.js.txt` plus the
  per-turn wording files `blocks.segment.txt`, `blocks.comment.txt` (turn-1 summary), `blocks.score.txt` (turn-2
  value score), `blocks.comment.nudge.txt`, `blocks.note.short.txt`, and `blocks.note.long.txt` (each overriding a
  default constant in `scale_blocks.py`). Note: `{block}` is the paragraph *inside its raw context window*, the
  paragraph's own lines gutter-marked `> ` — a custom `blocks.comment.txt` / `blocks.score.txt` /
  `blocks.comment.nudge.txt` predating the window should be re-based on the shipped files, which tell the model to
  describe only the marked lines.
- **Whole-file summary** (which is also the `--file-doc` header description): `summary.txt` (the full
  file-description spec, `{language}`/`{seed}`) and `summary.short.txt` (the def-pass squash), overriding
  `SUMMARY_INSTRUCTION`/`SHORT_SUMMARY_INSTRUCTION` in `scale.py`.
- **File-doc pass**: only `filedoc.classify.txt` (which header lines are the existing description), overriding a
  default constant in `scale_filedoc.py`.
- **Project layer**: `project.txt` (the project-blurb instruction, overriding `PROJECT_BLURB_INSTRUCTION` in
  `scale_project.py`).
- **Verification layer**: `verify.gate.txt` (the grounding-gate nudge), `verify.grounding.txt` /
  `verify.grounding.feedback.txt`, `verify.obvious.txt` / `verify.obvious.feedback.txt`, and `verify.story.txt` /
  `verify.story.feedback.txt` (the three challenge prompts and their regeneration feedback, overriding constants in
  `scale_verify.py`).

Files are filled by literal `{placeholder}` substitution (`_fill`), so stray braces and code containing braces are
safe. Keep prompts terse — the local model's window is small, and long rule-laden prompts make small models drift
(see the design goals in [CLAUDE.md](../CLAUDE.md): prefer a structural fix or a second single-aspect turn over
growing a prompt).
