# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SCALE (Source Code Annotation with LLM Engine) generates/updates code comments and docstrings using a local GGUF LLM loaded via `llama.cpp`. The core guarantee: it only ever touches comments/docstrings — executable code, indentation, blank lines, and other comments are preserved byte-for-byte, because output is produced by *patching the parsed source*, never by having the LLM re-emit code.

## Running

```bash
# Annotate a Python file in place (verbose). -c enables comment generation.
python scale.py -m /path/model.gguf -c path/to/file.py -o path/to/file.py -v

# Common runtime knobs used in practice (see tests.sh):
python scale.py -c file.py -o out.py -v --n-ctx 12288 --n-batch 256

# Without -o, output is printed to stdout. --help lists all flags.
```

- `tests.sh` is the de-facto test harness: it runs the tool against JS/C/Python sample files (paths are machine-specific to the author — adjust before running). There is no unit-test suite.
- `tune_n_batch.py` probes throughput to pick a good `--n-batch` for a given model/GPU.
- The default model path (`DEFAULT_MODEL` in `scale.py`) is a local Qwen2.5.1-Coder-7B-Instruct GGUF (Q5_K_M); override with `-m`.

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
| `scale_llm.py` | `LocalChatModel` (wraps `llama_cpp.Llama`) + `GenerationConfig` + per-family chat formatters |
| `scale_python.py` | Python worker — CPython `ast` based |
| `scale_javascript.py` | JS worker — tree-sitter (ESM + CommonJS) |
| `scale_c.py` | C worker — tree-sitter |
| `scale_log.py` | `echo`/`error`/`set_verbosity` (global `VERBOSE`; `echo` is gated, `error` always to stderr) |

### The annotation pipeline (per file)

1. **Load** (`load_source`): read bytes, decode UTF-8 with `errors="surrogateescape"`, detect dominant line ending (`\n`/`\r`/`\r\n`). All subsequent splitting/joining uses that detected ending; output is written back with the same encoding+ending so undecodable bytes and newline style survive a round-trip. **Do not** use text-mode reads or `pathlib.read_text` for source content — that reintroduces newline translation (a previously-fixed bug).
2. **Prime LLM** (`prime_llm_for_comments`): builds a `messages` transcript = system prompt (`scale-cfg/comment.txt`) + a whole-file summary + the language comment template (`scale-cfg/comment.<lang>.txt`). The summary is cached (see below) since it's slow.
3. **Worker** generates one comment per routine and patches them in.

### Worker pattern (shared by all three language workers)

- Parse source into definition records (`DefInfo` / `DefInfoJS` / C equivalent) carrying qualname, node, and precise line spans (`header_start`/`header_end`, `start`/`end`).
- Process **deepest-nested definitions first**. When generating a parent's comment, nested children are collapsed to *just their signature + the docstring just generated for them* — the LLM sees immediate children's contracts but not their bodies. This keeps the context window small and lets a class comment summarize its methods.
- Each generation turn is appended to `messages`, then **popped after the reply** — turns are not accumulated, so context stays bounded (only system prompt + summary + template + current snippet are ever live).
- **Patch textually in reverse source order** (`patch_docstrings_textually` and JS/C analogues) so line indices stay valid as edits are applied. Existing docstrings/comment blocks are detected and replaced; otherwise a new block is inserted after the header. The LLM's reply is parsed to extract only the comment block (`extract_first_docstring` / `_extract_first_comment_block`) — never the surrounding code.

### Summary cache (`SummaryCache` in `scale.py`)

File-backed cache under `__cache__/`: `index.pkl` maps source path → UID; `<uid>.txt` holds the summary. Writes are atomic (temp file + `replace`). Bypass with `--no-cache`/`-nc`. Same surrogateescape encoding as source.

### LLM layer (`scale_llm.py`)

- `LocalChatModel` builds prompts via one of several **chat formatters** (`qwen`/`chatml`, `llama3`, `llama2`, `mistral`, `phi3`) registered in `FORMATTERS`. Format is auto-detected from the model **filename** (`_auto_detect_format`, defaults to `qwen`) unless `--format` overrides it. Each formatter returns `(prompt, stop_tokens)` and opens an assistant turn as the generation anchor.
- `generate()` is the synchronous path used by workers; `progressive_generate()` streams. Both estimate prompt token count (`_trim_needed`, refining a bytes-per-token estimate every 8 turns) and warn when nearing `n_ctx`.
- `LocalChatModel.download_model(...)` is a classmethod helper for pulling GGUFs from HuggingFace.

## Editing the LLM's behaviour without code changes

Prompt behaviour lives in `scale-cfg/`: `comment.txt` is the system prompt; `comment.python.txt` / `comment.js.txt` / `comment.c.txt` are per-language comment-style templates (with examples). Tuning these examples to match the target codebase is the primary lever for output quality — prefer this over changing code.

## Tests

`tests/` holds fast, **model-free** regression tests — the LLM is stubbed and only parsing/patching/caching run, so the suite finishes in seconds. Run all with `python tests/run_all.py`, or any single `tests/test_*.py` directly. Each guards a specific past bug; keep them green when touching the relevant area, and add one when fixing a new bug:

- `test_summary_injection.py` — the whole-file summary is actually injected into the LLM context. Guards the critical regression where a stale `"OK"` was interpolated instead of the summary, silently disabling the feature.
- `test_cache_invalidation.py` — `SummaryCache` reuses on identical content but regenerates when the source changes (content-hash keying), so edits don't get a stale summary.
- `test_qualname_collision.py` — same-named definitions get distinct docstrings. Guards against doc maps keyed by qualname (now keyed by node identity).
- `test_inline_def.py` — inline one-line `def`/`class` are skipped rather than corrupted, and output stays valid Python.
- `test_cr_line_endings.py` — bare `\r` (old-Mac) line endings still map tree-sitter rows to the right source lines, in both the C and JS workers.
- `test_js_imports.py` — CommonJS `require()` (including destructured forms) is parsed via the grammar's `arguments` field rather than fragile index traversal.
- `test_js_extractor.py` — the JS comment extractor accepts a `/** */` block, a ``` fence, or plain text, and preserves genuine content bullets instead of eating leading asterisks.

## Conventions

- The `temp/` directory holds generated/scratch outputs from `tests.sh`; it is not source.
- Comments and docstrings throughout are themselves SCALE-generated — keep that style if regenerating, but it means docstrings may occasionally lag the code.
