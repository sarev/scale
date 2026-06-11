# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

SCALE (Source Code Annotation with LLM Engine) generates/updates code comments and docstrings using a local GGUF LLM loaded via `llama.cpp`. The core guarantee: it only ever touches comments/docstrings — executable code, indentation, blank lines, and other comments are preserved byte-for-byte, because output is produced by *patching the parsed source*, never by having the LLM re-emit code.

## Design goals

Every change should serve (and never trade away) these three principles:

1. **Trustworthy and complete.** Pre-existing code is preserved byte-for-byte, and every file/routine the tool is pointed at gets processed — both guaranteed by deterministic machinery (patchers, guards, manifest counters), never by trusting a model's behaviour or diligence.
2. **Value-adding only.** A comment earns its place by giving insight or speeding a new reader's comprehension; restatements are filtered out. Output claims are checked against the code rather than assumed correct. Known limit: institutional knowledge (design rationale, cross-system contracts) cannot be regenerated — where it already exists, the ingest-and-update seeds preserve it.
3. **Local-first.** The local model does the volume; a stronger model (Claude, via the manifest) is consulted selectively — by complexity, detected failure, or contract criticality — through bounded, self-contained, machine-checkable work units. It is never handed an open-ended job, and the manifest scaffolding stays lean because in the worst case (a fully uncommented codebase) every routine's code crosses the wire once; nothing should make it cross twice.

A corollary that shapes solutions here: **generic over convention-specific.** Never assume a project's coding conventions (identifier casing, error-code returns, comment density) — mechanisms must work across languages and house styles, unless a language-specific path is generally valuable for that whole language. And prefer **structural fixes over prompt growth**: long rule-laden prompts make small models drift; a second small, single-aspect turn in a clean context (challenge/score patterns) beats adding rules to the first turn.

## This dev environment (use these, not a bare `python`)

The venv and the GGUF models are **siblings of the project root**:

```
<parent>/
  scale/          <- this project (cwd)
  .llm-venv/      <- the venv  (Windows interpreter: ../.llm-venv/Scripts/python.exe)
  models/         <- GGUF models, e.g. models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF/*.gguf
```

```bash
# Run the fast, model-free regression suite (seconds; always do this after changes):
../.llm-venv/Scripts/python.exe tests/run_all.py

# Canonical annotate run (definition docs + block comments):
../.llm-venv/Scripts/python.exe scale.py -c --block-comments medium -l python \
  -m "../models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF/Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf" \
  path/to/file.py -o out.py -v
```

Two gotchas:
- **`-m` is effectively required here.** `DEFAULT_MODEL` resolves under `scale/models/...`, but the models live in the sibling `../models/...`.
- **Pass `-l python` (etc.) explicitly on stripped/atypical files.** Language is auto-guessed from *content*, not the file extension.

## Architecture in one screen

`scale.py` is the CLI/orchestrator; each language worker (`scale_python` / `scale_c` / `scale_javascript`) exposes the identical entry point `generate_language_comments(llm, cfg, messages, source_blob, source_lines)`. Adding a language = implement that + a dispatcher branch.

Up to three passes run in sequence per file, each re-priming and re-parsing the current text: the **definition pass** (`-c`, one docstring/header comment per routine), the **block pass** (`--block-spacing`/`--block-comments`, comments on statement groups inside bodies), and the **file-doc pass** (`--file-doc`, the top-of-file description — runs **LAST**, so it draws on the docs just written). Around them sit: a **project-context layer** (run model, call graph, callee contracts, C header/impl doc-site), a **verification floor** (grounding gate + clean-context challenge turns, default-on), and **selective escalation** to a stronger model via run-level JSON manifests (Python and C).

| Module | One-liner |
| --- | --- |
| `scale.py` | CLI, language guessing, priming, summary cache, dispatch, file I/O |
| `scale_llm.py` | `LocalChatModel` over `llama_cpp`, chat formatters, token budgets |
| `scale_text.py` | `summarise` + crude snippet cropping (`fit_snippet`) |
| `scale_python.py` / `scale_c.py` / `scale_javascript.py` | Per-language workers: def-pass, block provider, native cognitive scorer (+ C: the `--doc-site` planner) |
| `scale_blocks.py` | Language-agnostic block pass: structural segmenter, two-turn comment+score, insertion-only patcher + guard |
| `scale_verify.py` | Grounding gate + challenge turns (the local quality floor) |
| `scale_escalate.py` | Escalation policy + run-level manifest (emit/check/apply) |
| `scale_reword.py` | Header-reword manifest (prose-only) |
| `scale_project.py` | Run-file store, project blurb, call graph, contract store |
| `scale_filedoc.py` | File-doc pass engine: classify → license veto → guarded splice |
| `scale_log.py` | `echo`/`error`/verbosity |

## Invariants — do not break these

- **The guarantee lives in the patchers, never the model.** Every model reply passes through an extractor + insertion-only patcher + a preservation guard (`code_preserved` / `file_doc_preserved` / `_py_doc_preserved`). Keep that split in any new feature.
- **Binary-safe I/O** (`load_source`): bytes + UTF-8 `surrogateescape` + detected line endings, written back the same way. Never use text-mode reads / `pathlib.read_text` for source content (reintroduces newline translation — a fixed bug).
- **Bounded context**: generation turns are appended to `messages` then popped after each reply; what the model *reads* may be elided/annotated, but patching always works from the pristine source lines.
- **Tree-sitter loader shim**: `_load_*_language_and_parser()` tolerates both the one-arg (≥ 0.22) and two-arg (0.21) `Language(...)` APIs; keep the fallback. Install/upgrade `tree-sitter` + grammars together (shared ABI).
- **Completeness is counted, not trusted**: manifest flows finish when `--check-manifest` reports 0, not when a model says done.

## Detailed documentation (read on demand)

- [docs/cli.md](docs/cli.md) — every flag with worked example commands (block pass, file-doc, references, doc-site, manifests, reword).
- [docs/architecture.md](docs/architecture.md) — the per-file pipeline (load/prime/summary), the full module table, the shared worker pattern, structural elision, doc-style matching, summary cache, LLM layer, dependencies.
- [docs/block-pass.md](docs/block-pass.md) — the within-function block pass: providers, the deterministic structural segmenter and its rules, the two-turn summarise-then-score comment pass, the patcher, prompt files, and the 7B reality check.
- [docs/verification.md](docs/verification.md) — the backtick-grounding gate, the three clean-context challenge turns, and the shared failure routing (regenerate once → promote or drop/warn).
- [docs/project-context.md](docs/project-context.md) — the multi-file run model and retained run-file store, project blurb, call graph (leaf-first order + callee contracts + lazy one-liners), ingest-and-update seeds, block-pass read-side call annotations, and the C header/impl doc-site.
- [docs/file-doc.md](docs/file-doc.md) — the top-of-file description pass: two-pass descriptions, classify + license veto + preservation guard, the Python module-docstring target, and the header-reword manifest.
- [docs/escalation.md](docs/escalation.md) — selective escalation: the three routing signals (native cognitive scorers per language), the run-level manifest schema and emit/apply phases, the completeness counter, and the `/scale` skill loop.
- [docs/prompt-tuning.md](docs/prompt-tuning.md) — every `scale-cfg/` prompt file and which constant it overrides; the primary lever for output quality.
- [docs/tests.md](docs/tests.md) — the full test catalogue (which past bug each test guards) and the model-dependent `tests/block_eval/` harnesses.

## Tests

`tests/run_all.py` runs the fast, **model-free** suite (LLM stubbed; finishes in seconds). Each test guards a specific past bug — keep them green when touching the relevant area, and **add one when fixing a new bug**. The per-test catalogue is in [docs/tests.md](docs/tests.md).

## Conventions

- The `temp/` directory holds generated/scratch outputs from `tests.sh`; it is not source.
- Comments and docstrings throughout are themselves SCALE-generated — keep that style if regenerating, but it means docstrings may occasionally lag the code.
