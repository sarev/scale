# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SCALE (Source Code Annotation with LLM Engine) generates/updates code comments and docstrings using a local GGUF LLM loaded via `llama.cpp`. The core guarantee: it only ever touches comments/docstrings — executable code, indentation, blank lines, and other comments are preserved byte-for-byte, because output is produced by *patching the parsed source*, never by having the LLM re-emit code.

## Running

### This dev environment (use these, not a bare `python`)

The dependencies (`llama-cpp-python`, `tree-sitter`, …) live in a **virtualenv that is a sibling of this project**, and the GGUF models live in a **sibling `models/` directory** — both one level *above* the project root:

```
<parent>/
  scale/          <- this project (cwd)
  .llm-venv/      <- the venv  (Windows interpreter: ../.llm-venv/Scripts/python.exe)
  models/         <- GGUF models, e.g. models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF/*.gguf
```

Always invoke through the venv interpreter — a bare `python` lacks `llama_cpp`/`tree_sitter` and will fail to import:

```bash
# Run the fast, model-free regression suite:
../.llm-venv/Scripts/python.exe tests/run_all.py

# Annotate with the block pass (the model used during block-pass development):
../.llm-venv/Scripts/python.exe scale.py -b -nc -l python \
  -m "../models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF/Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf" \
  path/to/file.py -o out.py
```

Two gotchas:
- **`-m` is effectively required here.** `DEFAULT_MODEL` in `scale.py` resolves under `scale/models/...`, but the models are in the sibling `../models/...`, so the default does not find a model — pass `-m <abs-or-relative-path-to-.gguf>`.
- **Pass `-l python` (etc.) on stripped/atypical files.** Language is auto-guessed from *content*, not the file extension, so a comment-stripped "wall" can be mis-detected; set it explicitly when in doubt.

### Flags

```bash
# Annotate a Python file in place (verbose). -c enables definition docstrings/header comments.
python scale.py -m /path/model.gguf -c path/to/file.py -o path/to/file.py -v

# Common runtime knobs used in practice (see tests.sh):
python scale.py -c file.py -o out.py -v --n-ctx 12288 --n-batch 256

# -b adds within-function block comments (Python only); -c and -b can be combined.
python scale.py -c -b file.py -o out.py -v

# Without -o, output is printed to stdout. --help lists all flags.
```

- `tests/run_all.py` is the fast, model-free regression suite (see Tests below). `tests/block_eval/` holds the model-dependent block-pass eval harnesses.
- `tests.sh` is an older end-to-end harness: it runs the tool against JS/C/Python sample files (paths are machine-specific to the author — adjust before running).
- `tune_n_batch.py` probes throughput to pick a good `--n-batch` for a given model/GPU.
- The default model path (`DEFAULT_MODEL` in `scale.py`) is a local Qwen2.5.1-Coder-7B-Instruct GGUF (Q5_K_M); override with `-m` (required in this environment — see above).

## Dependencies

`tree-sitter`, `tree-sitter-c`, `tree-sitter-javascript` (works on current versions — tested with 0.25/0.24/0.25; install/upgrade all three together since core + grammars share an ABI). Plus `llama-cpp-python` (GPU/CPU build matched to host CUDA) and `huggingface-hub`. See README for full install steps per OS.

Note: the C/JS workers construct the parser via a version-tolerant shim in `_load_*_language_and_parser()` that handles both the modern one-arg `Language(capsule)` API (>= 0.22) and the old two-arg `Language(capsule, name)` form (0.21). Keep that fallback if touching those loaders.

## Architecture

**Dispatcher + per-language workers.** `scale.py` is the CLI/orchestrator; each language has a worker module exposing the identical entry point:

```python
generate_language_comments(llm, cfg, messages, source_blob, source_lines) -> List[str]
```

`scale.py:generate_comments` dispatches on language to `scale_python` / `scale_javascript` / `scale_c`. **To add a language**, implement that same function and add a branch in the dispatcher — no other wiring.

| Module | Role |
| --- | --- |
| `scale.py` | CLI parsing, language guessing, summary cache, LLM priming, dispatch, file I/O |
| `scale_llm.py` | `LocalChatModel` (wraps `llama_cpp.Llama`) + `GenerationConfig` + per-family chat formatters + token-budget helpers |
| `scale_text.py` | Context-window helpers: `elide_to_budget`/`fit_snippet` for trimming oversized routine snippets |
| `scale_python.py` | Python worker — CPython `ast` based (definition pass + `iter_block_targets` block provider) |
| `scale_javascript.py` | JS worker — tree-sitter (ESM + CommonJS) |
| `scale_c.py` | C worker — tree-sitter |
| `scale_blocks.py` | Language-agnostic within-function "blob" pass: numbered-body boundary view, boundary/comment passes, insertion-only patcher + code-preservation guard |
| `scale_log.py` | `echo`/`error`/`set_verbosity` (global `VERBOSE`; `echo` is gated, `error` always to stderr) |

### The annotation pipeline (per file)

1. **Load** (`load_source`): read bytes, decode UTF-8 with `errors="surrogateescape"`, detect dominant line ending (`\n`/`\r`/`\r\n`). All subsequent splitting/joining uses that detected ending; output is written back with the same encoding+ending so undecodable bytes and newline style survive a round-trip. **Do not** use text-mode reads or `pathlib.read_text` for source content — that reintroduces newline translation (a previously-fixed bug).
2. **Prime LLM** (`prime_llm_for_comments`): builds a `messages` transcript = system prompt + a whole-file summary + a per-language style template. Both the system prompt and the template are pass-specific (`template` arg): the definition pass is `comment.txt` + `guidelines.md` (house-style doc-comment rules) with `comment.<lang>.txt`; the block pass is `comment.txt` alone (guidelines are **not** appended — they are def-pass-only noise that would waste the small window) with `blocks.<lang>.txt`. Only one template is ever live so the passes never share each other's guidance. The summary is cached (see below) since it's slow. For files too large to summarise in one pass, `_generate_file_summary` falls back to **chunked map-reduce** (`_split_source` → per-chunk summaries → `_reduce_summaries`), capped at `SUMMARY_MAX_TOKENS` so the priming context stays small regardless of file size.
3. **Worker** generates one comment per routine and patches them in.

Two annotation passes run in sequence (`generate_comments`), each re-priming and re-parsing the current text so spans stay valid: the **definition pass** (`-c`/`--comment`, all languages) writes one docstring/header comment per routine; the **block pass** (`-b`/`--blocks`, Python only for now) annotates logical groups of statements *inside* bodies. Either or both may run. The block pass is described under "Within-function block pass" below.

### Worker pattern (shared by all three language workers)

- Parse source into definition records (`DefInfo` / `DefInfoJS` / C equivalent) carrying qualname, node, and precise line spans (`header_start`/`header_end`, `start`/`end`).
- Process **deepest-nested definitions first**. When generating a parent's comment, nested children are collapsed to *just their signature + the docstring just generated for them* — the LLM sees immediate children's contracts but not their bodies. This keeps the context window small and lets a class comment summarize its methods.
- Each generation turn is appended to `messages`, then **popped after the reply** — turns are not accumulated, so context stays bounded (only system prompt + summary + template + current snippet are ever live).
- **Oversized routines are elided to fit the context window** via `scale_text.fit_snippet` before sending: the signature/header is kept and the middle of the body is replaced with a `... N lines omitted ...` marker (sized from `LocalChatModel.snippet_budget`). This only affects what the model *reads* — patching still uses the real source, so no code is lost.
- **Patch textually in reverse source order** (`patch_docstrings_textually` and JS/C analogues) so line indices stay valid as edits are applied. Existing docstrings/comment blocks are detected and replaced; otherwise a new block is inserted after the header. The LLM's reply is parsed to extract only the comment block (`extract_first_docstring` / `_extract_first_comment_block`) — never the surrounding code.

### Within-function block pass (`scale_blocks.py`, `-b`/`--blocks`)

Annotates logical groups of statements ("blobs") inside routine bodies, not whole definitions. It is line-based but never splits mid-statement, and is **insertion-only** so the byte-for-byte code guarantee holds.

- **Provider** (`scale_python.iter_block_targets`) returns one `BlockTarget` per function/method/class body. `boundary_lines` are the lines that begin **exactly one statement at the line's first non-blank column**, collected at all depths but **not descending into nested defs** (a nested def is one opaque boundary at its decorator-aware header line). Excluded: multi-statement (`a; b`), inline-compound (`if x: y`), continuation lines, a leading docstring, and the **first statement of an inner suite** (a blank as the first line of an `if`/`for`/… body reads badly; the routine's own first body statement stays eligible). Adding a language = implement this provider + a `blocks.<lang>.txt` template + wire `_block_provider_for`/`_block_style_for` in `scale.py`.
- **Segment pass** (`request_segments`): the body is rendered as a numbered view (`render_numbered_body`) with a line number **only on boundary lines** (over-long non-boundary runs collapse to a `« N lines elided »` band carrying no number). The model is asked for **chunk line ranges** (`start-end`), *not* split points — framing it as "group related lines" stops a weak model fragmenting the body line-by-line. `_parse_segments` snaps each start to a legal boundary (drops illegal starts) and clamps ends within the body / before the next chunk. Run **deterministically** (`SEGMENT_TEMPERATURE = 0.0`) — segmentation is structural.
- **Comment pass** (`request_block_comment`): one paragraph at a time, with only light context — the routine's docstring **summary** (`{doc}`, first line) and the comments **already written for earlier paragraphs** (`{priors}`); the file overview is already primed. Showing just the paragraph (not the whole function) keeps the model focused, and the running `priors` give a narrative thread that stops it repeating itself. The bias is softened toward a *useful* line (what the block accomplishes, or a reason/gotcha/subtlety) rather than a strict "why", since a pure why-gate makes the 7B abstain. If the first reply is neither a usable comment nor a clear `NONE` (an evasive non-answer), it is **nudged once** (`COMMENT_NUDGE`) to try again before giving up (`_is_explicit_none` distinguishes a deliberate decline from waffle). Low temperature (`COMMENT_TEMPERATURE = 0.1`). Append-then-pop per turn. **Length gate:** a routine with `≤ SHORT_FUNCTION_CHUNKS` (3) chunks gets a conservative note (`COMMENT_NOTE_SHORT` — don't echo the docstring), while a longer routine gets `COMMENT_NOTE_LONG` inviting a per-block walkthrough even if it lightly duplicates the docstring — light duplication walks a reader through a long body but is just noise in a short one.
- **Patcher** (`apply_blocks` / `_apply_edits`): in reverse line order, **every chunk gets a blank line above it** (insert-if-absent, never removing pre-existing blanks) — paragraphing a wall of statements into its blocks is value in itself, and is safe now that range-segmentation yields few, sensible chunks rather than line-by-line fragments. Where the model returned comment text it additionally replaces an existing same-indent comment or inserts a fresh one; `NONE` adds only the blank and **keeps any existing comment untouched** (never deletes a hand-written one). A **code-preservation guard** (`code_preserved`, comparing the non-blank/non-comment line signature) abandons any routine whose edit would alter code. `annotate_blocks` generates deepest-first, then applies all surviving edits in one reverse-order pass (chunks across routines are disjoint thanks to nested-def opacity).
- **Prompts are externalised/tunable.** Every piece of block-pass prompt *wording* lives in `scale-cfg` as a user-editable file, with a built-in default constant in `scale_blocks.py` as fallback: `blocks.segment.txt` (`SEGMENT_PROMPT`), `blocks.comment.txt` (`COMMENT_PROMPT`), `blocks.comment.nudge.txt` (`COMMENT_NUDGE`), `blocks.note.short.txt` / `blocks.note.long.txt` (`COMMENT_NOTE_SHORT`/`_LONG`), plus the priming template `blocks.python.txt`. Files are filled by literal `{placeholder}` substitution via `_fill`, so stray braces and code with braces are safe. Non-wording tuning knobs stay as code constants (`SHORT_FUNCTION_CHUNKS`, `*_TEMPERATURE`, `*_REPLY_TOKENS`). Keep prompts terse — the window is small.
- **Reality check (7B local model).** *Paragraphing* (blank lines splitting a wall into its blocks) is the dependable win — range-segmentation finds boundaries that line up with where a human would break the code. *Comments* are capped by model capability and tunable only within a narrow band: a strict "why"-gate makes the 7B abstain almost entirely; softening toward "a useful section-summary line" gets comments flowing but mixes genuinely useful headers (e.g. "Retry up to three times with exponential backoff") with bland restatements and the **occasional incorrect/misplaced comment** — the last being the real risk with experienced readers. The current `scale-cfg` prompts sit at the softened end (comment by default, `NONE` for the trivial). Tuning the `scale-cfg` examples toward the target codebase helps; materially better comments need a more capable model for the comment pass (the segment and comment passes are separate, so a stronger model could be wired in for comments alone).

### Summary cache (`SummaryCache` in `scale.py`)

File-backed cache under `__cache__/`: `index.pkl` maps source path → UID; `<uid>.txt` holds the summary. Writes are atomic (temp file + `replace`). Bypass with `--no-cache`/`-nc`. Same surrogateescape encoding as source.

### LLM layer (`scale_llm.py`)

- `LocalChatModel` builds prompts via one of several **chat formatters** (`qwen`/`chatml`, `llama3`, `llama2`, `mistral`, `phi3`) registered in `FORMATTERS`. Format is auto-detected from the model **filename** (`_auto_detect_format`, defaults to `qwen`) unless `--format` overrides it. Each formatter returns `(prompt, stop_tokens)` and opens an assistant turn as the generation anchor.
- `generate()` is the synchronous path used by workers; `progressive_generate()` streams. Both call `_check_context_budget`, which estimates prompt tokens (`_estimate_prompt_tokens`, refining a bytes-per-token ratio every 8 turns) and raises if the prompt cannot fit / warns if it leaves too little room.
- Token-budget helpers used by elision: `estimate_tokens(text)` (cheap, ratio-based) and `snippet_budget(messages, cfg)` (tokens free for a routine snippet = `n_ctx − ctx_margin − reply_reserve − count_tokens(messages)`; `reply_reserve` is `COMMENT_GENERATION_RESERVE`, not the full `max_new_tokens`).
- `LocalChatModel.download_model(...)` is a classmethod helper for pulling GGUFs from HuggingFace.

## Editing the LLM's behaviour without code changes

Prompt behaviour lives in `scale-cfg/`: `comment.txt` is the system prompt and `guidelines.md` the house-style rules (definition pass only); `comment.python.txt` / `comment.js.txt` / `comment.c.txt` are per-language definition-pass templates (with examples). The block pass adds `blocks.python.txt` (its priming template) plus the per-turn wording files `blocks.segment.txt`, `blocks.comment.txt`, `blocks.comment.nudge.txt`, `blocks.note.short.txt`, and `blocks.note.long.txt` (each overriding a default constant in `scale_blocks.py`). Tuning these examples/wording to match the target codebase is the primary lever for output quality — prefer this over changing code.

## Tests

`tests/` holds fast, **model-free** regression tests — the LLM is stubbed and only parsing/patching/caching run, so the suite finishes in seconds. Run all with `python tests/run_all.py`, or any single `tests/test_*.py` directly. Each guards a specific past bug; keep them green when touching the relevant area, and add one when fixing a new bug:

- `test_summary_injection.py` — the whole-file summary is actually injected into the LLM context. Guards the critical regression where a stale `"OK"` was interpolated instead of the summary, silently disabling the feature.
- `test_cache_invalidation.py` — `SummaryCache` reuses on identical content but regenerates when the source changes (content-hash keying), so edits don't get a stale summary.
- `test_qualname_collision.py` — same-named definitions get distinct docstrings. Guards against doc maps keyed by qualname (now keyed by node identity).
- `test_inline_def.py` — inline one-line `def`/`class` are skipped rather than corrupted, and output stays valid Python.
- `test_cr_line_endings.py` — bare `\r` (old-Mac) line endings still map tree-sitter rows to the right source lines, in both the C and JS workers.
- `test_js_imports.py` — CommonJS `require()` (including destructured forms) is parsed via the grammar's `arguments` field rather than fragile index traversal.
- `test_js_extractor.py` — the JS comment extractor accepts a `/** */` block, a ``` fence, or plain text, and preserves genuine content bullets instead of eating leading asterisks.
- `test_elision.py` — `elide_to_budget` preserves the header, fits the token budget, and reports the omitted-line count (degrades to header-only on a tiny budget).
- `test_chunk_split.py` — `_split_source` covers every line exactly once (chunks rejoin to the original), keeps each chunk within budget, and hard-splits over-long single lines.
- `test_mapreduce_summary.py` — `_generate_file_summary` uses one pass when the file fits and switches to map-reduce (multiple map calls + a reduce) when it doesn't.
- `test_block_boundaries.py` — the Python block provider numbers only legal statement starts at all depths, excludes continuation/`a; b`/inline-compound lines and leading docstrings, and does not descend into nested defs (which become single opaque boundaries); classes get a block pass too.
- `test_block_insertion.py` — the insertion patcher inserts blank+comment above chosen chunks, replaces an existing same-indent comment, treats `NONE` as a true no-op (code/blanks/existing comment untouched), preserves the code signature, and aborts the routine (original kept) when a forced edit would alter code.
- `test_block_numbered_view.py` — over-long non-boundary runs collapse to an elision band carrying no number (short runs shown verbatim); the segment parser keeps only ranges whose start is a legal boundary, clamps ends within the body / before the next chunk, and falls back to bare numbers as starts.

`tests/block_eval/` holds **model-dependent** evaluation harnesses (not run by `run_all.py` — it's a subdirectory, and the `test_*.py` glob is non-recursive) for eyeballing block-pass quality as the model/prompts change: `make_wall.py` (strip a file to a docstring-only wall), `show_segments.py` / `show_comments.py` (print the segmenter's chunks / the per-chunk comments without patching), and `samples/` fixtures. They load a real GGUF (`SCALE_MODEL` overrides the default). See `tests/block_eval/README.md`.

## Conventions

- The `temp/` directory holds generated/scratch outputs from `tests.sh`; it is not source.
- Comments and docstrings throughout are themselves SCALE-generated — keep that style if regenerating, but it means docstrings may occasionally lag the code.
