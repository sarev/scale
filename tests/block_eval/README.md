# Block-pass evaluation harnesses

Model-**dependent** tools for eyeballing the quality of the within-function block pass (`-b`) as SCALE evolves. Unlike
the fast, model-free regressions in `tests/` (run by `run_all.py`), these load a real GGUF and exercise the actual
segment/comment passes. They live in this subdirectory so `run_all.py`'s `test_*.py` glob never tries to run them.

Run them with the project venv. The model defaults to the project's Qwen2.5-Coder-7B; override with `SCALE_MODEL`:

```bash
# Make a worst-case "wall of statements" from a real file (strips body comments + blank lines, keeps docstrings):
python tests/block_eval/make_wall.py scale_text.py temp/wall.py

# Truer wall: also strip docstrings, so the file has NO comments at all bar a shebang (forces fresh docstrings too):
python tests/block_eval/make_wall.py --strip-docstrings scale_text.py temp/wall.py

# See how the segment pass chunks a file (no patching):
../.llm-venv/Scripts/python.exe tests/block_eval/show_segments.py temp/wall.py

# See the comment produced per chunk (no patching) - reflects the length note + nudge retry:
../.llm-venv/Scripts/python.exe tests/block_eval/show_comments.py temp/wall.py

# Full end-to-end: annotate and diff (note -l python; the wall can defeat language auto-guess):
../.llm-venv/Scripts/python.exe scale.py -b -nc -l python temp/wall.py -o temp/wall.scaled.py
git --no-pager diff --no-index temp/wall.py temp/wall.scaled.py
```

## Files

- `make_wall.py` — strip a file to a "wall" of statements (the hardest input for the block pass): blanks + `#` comments by default; add `--strip-docstrings` to also remove docstrings (no comments left bar a shebang), which additionally exercises fresh docstring generation/escalation.
- `show_segments.py` — print the chunk ranges the segment pass picks (is it grouping sensibly?).
- `show_comments.py` — print the comment per chunk (is it useful, restating, or wrong?).
- `_harness.py` — shared model/priming setup (resolves the project root + `SCALE_MODEL`).
- `samples/` — small fixtures with a known non-obvious "why" to probe comment quality:
  - `charge.py` — retry-with-exponential-backoff (thin docstring).
  - `seats.py` — sort families largest-first so big groups get contiguous blocks before the map fragments.

## What "good" looks like (current 7B baseline)

- **Paragraphing** is the reliable win — chunk boundaries line up with where a human would break the body.
- **Comments**: long functions get a useful per-block walkthrough; short ones stay conservative. Watch for the known
  failure modes — bland restatement, the occasional *incorrect*/misplaced comment, and US-spelling slips. Materially
  better comments need a stronger model for the comment pass.
