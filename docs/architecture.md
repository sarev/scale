# Architecture: dispatcher, workers, pipeline, LLM layer

How SCALE is put together: the per-language worker model, the per-file annotation pipeline, the shared worker
pattern, the summary cache, and the LLM layer. See [CLAUDE.md](../CLAUDE.md) for the one-screen overview and
the design goals every change must serve.

## Dispatcher + per-language workers

`scale.py` is the CLI/orchestrator; each language has a worker module exposing the identical entry point:

```python
generate_language_comments(llm, cfg, messages, source_blob, source_lines) -> List[str]
```

`scale.py:generate_comments` dispatches on language to `scale_python` / `scale_javascript` / `scale_c`. **To add a
language**, implement that same function and add a branch in the dispatcher — no other wiring.

| Module | Role |
| --- | --- |
| `scale.py` | CLI parsing, language guessing, summary cache, LLM priming, dispatch, file I/O |
| `scale_llm.py` | `LocalChatModel` (wraps `llama_cpp.Llama`) + `GenerationConfig` + per-family chat formatters + token-budget helpers |
| `scale_text.py` | Context-window helpers: `summarise` (the one reusable "compress text into one line/paragraph/paragraphs" call) + `elide_to_budget`/`fit_snippet` for the crude head/tail crop |
| `scale_python.py` | Python worker — CPython `ast` based (definition pass + `iter_block_targets` block provider + `structural_segments` adapter feeding the shared segmenter + `node_sig` (the manifest re-binding identity) + `collect_def_requests`/`apply_manifest` (the online emit/apply) + `elide_structurally` for snippet elision) |
| `scale_javascript.py` | JS worker — tree-sitter (ESM + CommonJS) + `iter_block_targets_js` block provider (nested-function-aware structural segmenter) + `_js_span_hash` (the comment-invariant manifest identity) + `collect_def_requests_js`/`apply_manifest_js` (the online emit/apply) |
| `scale_c.py` | C worker — tree-sitter + `iter_block_targets_c` block provider (structural segmenter) + `collect_def_requests_c`/`apply_manifest_c` (the online emit/apply, doc-site aware) + the header/implementation **doc-site** layer (`iter_decls_with_info_c` prototype collection, `plan_doc_sites_c`/`CDocPlan`/`CFileDocPlan` the model-free `--doc-site` planner; the def pass skips redirected definitions and documents prototypes from the impl body) |
| `scale_blocks.py` | Language-agnostic within-function "blob" pass: shared structural segmenter (`SegStatement` + `structural_breaks`), numbered-body boundary view, boundary/comment passes, insertion-only patcher + code-preservation guard, `CommentStyle`s (`PYTHON_STYLE`, `SLASH_LINE_STYLE`/`SLASH_BLOCK_STYLE`) |
| `scale_escalate.py` | The online mode's manifest plumbing: `Escalation` collector (per-routine merged requests), `routine_text_hash` (the C span-hash identity), `run_manifest` (the run-level version-2 manifest with cross-request `snippet_ref` slimming), `unfilled_answers` (the completeness counter), fragments, manifest read/write (v1 read-upgraded) |
| `scale_verify.py` | The local quality floor: `ungrounded_tokens` (the deterministic backtick-grounding gate over the run's source), the clean-context challenge turns (`challenge_grounding`/`challenge_obvious`/`challenge_story` — fresh single-turn context, temp 0, constrained replies), and `Verifier.verify_def` (the def-pass gate→challenge pipeline with one corrective regeneration each) |
| `scale_reword.py` | The header-reword manifest (prose-only, run-level): manifest build/read/write, `unfilled_rewords`, and the model-free `apply_reword` (exact-match draft location + guarded re-splice) |
| `scale_project.py` | Project-context layer above the per-file pipeline: the run model (`gather_files` expands target/`--reference` files/dirs/globs; `scan_run_files`/`RunFile` the retained run-file store — every run file loaded+parsed exactly once, shared by all pre-passes and the lazy one-liner generator) + `find_project_doc`/`resolve_project_doc`/`project_blurb` (`--project-doc`, a cached 2-3 sentence blurb injected into every file's priming). Also the **call graph** (`Symbol` with full spans + per-call-site lines, `build_project_graph`/`ProjectGraph` with the confident-only resolver + per-call `call_map` + SCC/topo leaf-first ordering, `ContractStore` with `contract`/`missing_callee_contracts`, `apply_doc_order`) that drives callee-before-caller documentation order, callee-contract context (lazily generated for undocumented callees), and the block pass's read-side call annotations |
| `scale_filedoc.py` | Language-agnostic file-level header doccomment pass (`--file-doc`): `FileDocTarget` (+ pluggable `preserved` guard), `looks_legal` license veto, `_parse_classify_range`, `annotate_file_doc` (classify→veto→splice-the-summary) delegating to the model-free `splice_description`, default `file_doc_preserved` guard, shared `scan_brace_leading_zone` (C/JS), `PYTHON_DOC_STYLE`; + the online `scale-filedoc` manifest (build/read/write, `unfilled_descriptions`, `apply_filedoc_entry`). Adapters: `scale_c.file_doc_target_c`, `scale_javascript.file_doc_target_js`, `scale_python.file_doc_target_py` (+ parse-based `_py_doc_preserved`) |
| `scale_log.py` | `echo`/`error`/`set_verbosity` (global `VERBOSE`; `echo` is gated, `error` always to stderr) |

## The annotation pipeline (per file)

1. **Load** (`load_source`): read bytes, decode UTF-8 with `errors="surrogateescape"`, detect dominant line ending
   (`\n`/`\r`/`\r\n`). All subsequent splitting/joining uses that detected ending; output is written back with the
   same encoding+ending so undecodable bytes and newline style survive a round-trip. **Do not** use text-mode reads
   or `pathlib.read_text` for source content — that reintroduces newline translation (a previously-fixed bug).
2. **Prime LLM** (`prime_llm_for_comments`): builds a `messages` transcript = system prompt + a whole-file summary +
   a per-language style template. The priming context turns are followed by a **fixed acknowledgement we supply
   ourselves** (`PRIMING_ACK`), *not* a model-generated "OK" — making a small model parrot "OK" through several
   handshakes conditions it to answer the first real request with "OK" too (a docstring of literally "OK"). So the
   priming makes no "say OK" asks and the model's first *generated* turn is a real comment (this also saves ~4
   generation calls per file). Both the system prompt and the template are pass-specific (`template` arg): the
   definition pass is `comment.txt` + `guidelines.md` (house-style doc-comment rules) with `comment.<lang>.txt`; the
   block pass is `comment.txt` alone (guidelines are **not** appended — they are def-pass-only noise that would
   waste the small window) with `blocks.<lang>.txt`. Only one template is ever live so the passes never share each
   other's guidance. **A one-line file-identity note** (`_file_identity_note`) naming the file and, for C,
   classifying it as a **header** (public interface — comments should read as the external contract) or an
   **implementation** (internal detail) is injected as a priming turn **before the summary**, so the file's role
   steers the summary, definition, and block passes (and the `--file-doc` header description, injected the same way
   in `_file_doc_pass`) — most sharply for a header, whose docs should describe the caller-facing contract rather
   than implementation detail.
3. **Worker** generates one comment per routine and patches them in.

Up to three annotation passes run in sequence (`generate_comments`), each re-priming and re-parsing the current text
so spans stay valid: the **definition pass** (`-c`/`--comment`, all languages) writes one docstring/header comment
per routine; the **block pass** (`--block-spacing`/`--block-comments`, Python/C/JS) annotates logical groups of
statements *inside* bodies (see [block-pass.md](block-pass.md)); the **file-doc pass** (`--file-doc`, Python/C/JS)
adds/updates the top-of-file header description. Any combination may run. The file-doc pass runs **LAST** — the
published description is generated from the *annotated* text's skeleton, so it draws on the docstrings the function
passes just wrote (the two-pass description model; see [file-doc.md](file-doc.md)).

### The whole-file summary (the description)

**The whole-file summary is written to a file-DESCRIPTION spec** (`summary.txt`/`SUMMARY_INSTRUCTION` — role-first,
grounded, flowing prose, no lists/quality-remarks) and is generated **from the file's structural skeleton**
(`scale_project.render_skeleton`: leading comments, signatures, class headers with method prototypes, C declarations
and top-level `#define`s, each symbol's existing doc — NO bodies; typically a small fraction of the file, so a
single call suffices and map-reduce becomes rare). The guard is binary: a file with **no symbols at all** is
summarised whole, by the old path. Descriptions are **two-pass**: pass 1 (priming-grade, cached by skeleton content
hash) is rendered from the *original* text and primes the def/block passes; pass 2 (published) is rendered by the
file-doc pass from the *current, annotated* text — see [file-doc.md](file-doc.md). Because the definition pass is
the most context-starved (the routine body matters more there than a detailed file overview), it primes with a
**short** one/two-sentence squash of that description (`_get_short_summary`, `summary.short.txt`); the block pass
(more headroom) primes with the **full** description (`_get_file_summary`). Both are cached (the short alongside the
full, same content-hash invalidation) since they're slow.

For files too large to summarise in one pass, `_generate_file_summary` falls back to **chunked map-reduce**
(`_split_source` → per-chunk summaries → `_reduce_summaries`, kept thorough) then a **final shaping turn** that
rewrites the overall summary to the description spec; everything is capped at `SUMMARY_MAX_TOKENS` so the priming
context stays small regardless of file size. **A de-listing guard** (`_reflow_if_listy`) closes both paths: the
small model sometimes ignores the prose-only spec and returns a numbered/bulleted/heading-structured summary (most
often on a large, map-reduced file), so a summary that `_looks_listy` is given **one reflow turn** to rewrite as
flowing prose, and if it still looks listy the markers are stripped deterministically (`_strip_list_markers`) — a
doc-comment never carries list/heading syntax.

## Worker pattern (shared by all three language workers)

- Parse source into definition records (`DefInfo` / `DefInfoJS` / C equivalent) carrying qualname, node, and precise
  line spans (`header_start`/`header_end`, `start`/`end`).
- Process **deepest-nested definitions first**. When generating a parent's comment, nested children are collapsed to
  *just their signature + the docstring just generated for them* — the LLM sees immediate children's contracts but
  not their bodies. This keeps the context window small and lets a class comment summarize its methods.
- **The assembled snippet preserves in-body comments.** All three workers send the author's standalone comments to
  the model as context: C/JS pull the body verbatim, and the Python `assemble_snippet_for` walks the body with a
  **cursor over source lines** (not statement-by-statement), carrying across the non-blank gaps between statements
  (standalone comments) while still stubbing nested defs. Without this the docstring model was blind to the very
  comments that explain intent.
- Each generation turn is appended to `messages`, then **popped after the reply** — turns are not accumulated, so
  context stays bounded (only system prompt + summary + template + current snippet are ever live).
- **Oversized routines are elided to fit the context window** before sending. Python uses **structural elision**
  (`scale_python.elide_structurally`): it repeatedly takes the *deepest* body suite, keeps its controlling header
  (`for …:`/`if …:`), and replaces just that suite with a single `...  # <one-line summary>` (via `summarise`),
  collapsing deepest-first one suite at a time and re-checking the budget after each, then climbing a level and
  re-summarising once a level is exhausted — so the routine's *shape and intent* survive instead of a blindly-cropped
  middle. If that can't get under budget (cap hit, won't parse, no nesting left) it falls back to the crude head/tail
  crop (`scale_text.fit_snippet`: keep signature, replace the middle with a `... N lines omitted ...` marker); C/JS
  use the crop directly. This only affects what the model *reads* — patching still uses the real source, so no code
  is lost. `summarise` is the shared primitive (also used by the whole-file summary and its map-reduce steps).
- **Patch textually in reverse source order** (`patch_docstrings_textually` and JS/C analogues) so line indices stay
  valid as edits are applied. Existing docstrings/comment blocks are detected and replaced; otherwise a new block is
  inserted after the header. The LLM's reply is parsed to extract only the comment block (`extract_first_docstring`
  / `_extract_first_comment_block`) — never the surrounding code. **C/JS match the file's prevailing doc-comment
  style:** the worker runs `_detect_doc_style_c`/`_detect_doc_style_js` (ignore the leading file-header banner — the
  first top-level comment — then `"line"` iff the remaining comments are all `//`, else `"block"`; a mix prefers the
  block form) and the patcher renders `//` line comments (`_render_c_line_comment`/`_render_js_line_comment`) or the
  `/* */`-`/** */` block, so generated docs don't introduce a block comment into a `//`-commented header. (Limit: a
  file whose only block comments are decorative `/* *** */` section dividers is still read as a mix → block;
  per-file detection also can't infer a style from a fully comment-stripped file — a project-level convention scan
  from the reference files would.)

## Summary cache (`SummaryCache` in `scale.py`)

File-backed cache under `__cache__/`: `index.pkl` maps source path → UID; `<uid>.txt` holds the summary. Writes are
atomic (temp file + `replace`). Bypass with `--no-cache`/`-nc`. Same surrogateescape encoding as source.

## LLM layer (`scale_llm.py`)

- `LocalChatModel` builds prompts via one of several **chat formatters** (`qwen`/`chatml`, `llama3`, `llama2`,
  `mistral`, `phi3`) registered in `FORMATTERS`. Format is auto-detected from the model **filename**
  (`_auto_detect_format`, defaults to `qwen`) unless `--format` overrides it. Each formatter returns
  `(prompt, stop_tokens)` and opens an assistant turn as the generation anchor.
- `generate()` is the synchronous path used by workers; `progressive_generate()` streams. Both call
  `_check_context_budget`, which estimates prompt tokens (`_estimate_prompt_tokens`, refining a bytes-per-token
  ratio every 8 turns) and raises if the prompt cannot fit / warns if it leaves too little room.
- Token-budget helpers used by elision: `estimate_tokens(text)` (cheap, ratio-based) and
  `snippet_budget(messages, cfg)` (tokens free for a routine snippet = `n_ctx − ctx_margin − reply_reserve −
  count_tokens(messages)`; `reply_reserve` is `COMMENT_GENERATION_RESERVE`, not the full `max_new_tokens`).
- `LocalChatModel.download_model(...)` is a classmethod helper for pulling GGUFs from HuggingFace.

## Dependencies

`tree-sitter`, `tree-sitter-c`, `tree-sitter-javascript` (works on current versions — tested with 0.25/0.24/0.25;
install/upgrade all three together since core + grammars share an ABI). Plus `llama-cpp-python` (GPU/CPU build
matched to host CUDA) and `huggingface-hub`. See README for full install steps per OS.

Note: the C/JS workers construct the parser via a version-tolerant shim in `_load_*_language_and_parser()` that
handles both the modern one-arg `Language(capsule)` API (>= 0.22) and the old two-arg `Language(capsule, name)` form
(0.21). Keep that fallback if touching those loaders.
