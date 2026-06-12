# Project-aware context (`scale_project.py`)

SCALE is otherwise single-file, which makes descriptions read generically (an `error.c` that never mentions it
belongs to a BASIC interpreter). `scale_project.py` adds a **project-context layer above** the per-file pipeline
that feeds in a broader view — but, because the local model's window is small, only as **distilled one-liners**,
never raw cross-file bodies. The patchers/guarantee are untouched; this only enriches the priming context.

## The run model

Positional targets accept **files, directories, and globs** (`gather_files` expands them to a deduped, ordered list
of source files — directories expand by `SOURCE_EXTS`, explicit files/globs are taken as-is). `main` loads the model
once and annotates each target: a single target writes to `-o`/stdout as before; **multiple targets are written in
place** (and `-o` is rejected). `--reference` (repeatable) captures read-only files/dirs/globs that SCALE consults
but never edits (a file that is also a target is dropped from the reference set). **References are never summarised
whole** (the old per-reference one-liner block was removed — it injected every reference into every target
regardless of use); they contribute through the call graph instead, per routine and only where used.
`project_context` is therefore the project blurb alone.

**The retained run-file store** (`scale_project.scan_run_files` → `RunFile`, built once in `main` when the def or
block pass runs) loads and parses every run file (targets ∪ references) **exactly once** — source text, language,
target flag, and `iter_symbols` output per file — and is the single input to `_build_call_graph`,
`_build_c_doc_plan`, and the lazy callee one-liner generator (no pre-pass re-loads a file).

## The project blurb

`resolve_project_doc` locates a project overview near the source (`--project-doc PATH`, else auto-detect by walking
up: `CLAUDE.md` preferred, then `README`/`ReadMe`/`readme` with any common extension, stopping at a `.git` root;
`--project-doc none` disables). `project_blurb` distils it once into a 2-3 sentence blurb (via `summarise` with
`scale-cfg/project.txt`'s spec, large docs head-cropped to budget), cached by the doc's content hash under
`__cache__/project-<hash>.txt`. `main` resolves+generates it and threads it as `project_blurb` into
`generate_comments`; `prime_llm_for_comments` and `_file_doc_pass` inject it as a turn **before the file-summary**
so both the file description and every routine turn understand the file's place in the project.

## The call graph (callee-before-caller ordering + callee contracts)

A routine's docstring otherwise can't draw on what the functions it *calls* do, and routines are documented by
nesting, not call dependency. A **model-free pre-pass** (`main` calls `_build_call_graph` over the retained run-file
store, before any model work) parses every run file into `Symbol`s — one per routine, carrying its
`parent_qualname`, header `signature`, full line span (`start`/`end` — what the lazy generator reads its body from),
its **whole** existing doc (`existing_doc`, delimiter-stripped; its first line via `_first_line` is the **seed
contract**), and its own call sites as `(name, kind, line)` triples classified `free`/`self`/`method` (each worker's
`iter_symbols` walks the routine's own body only, treating nested defs as opaque; JS uses the tree-stable `node.id`,
not Python `id()`, to skip nested-routine subtrees).

`build_project_graph` resolves calls **confidently — never guessing a receiver type**: a free function by name
(same-file first, else a run-wide *unique* match), `self.`/`this.` to the enclosing class's own method, and
`obj.m()` only when `m` is defined by **exactly one** class run-wide; anything ambiguous/typed/dynamic stays
unresolved and contributes no note (the call-site line is ignored by resolution, and older 2-tuple records still
resolve). Besides `edges`, the graph records `call_map` — caller → `{(name, kind): callee key}` — so call sites can
be re-found later by name+kind in re-parsed text.

This drives:

1. a **leaf-first documentation order** — a topological sort over two edge kinds, nesting (child→parent, so the
   existing child-stub mechanism still documents children first) and call (callee→caller), **condensed by SCC
   (Tarjan) + a deterministic Kahn order** so recursion / mutual recursion can't deadlock or loop — yielding a
   per-file symbol order (`graph.doc_order`) and a coarse target-file order (`graph.file_order`, callee's file
   first); and
2. a `ContractStore`, seeded from existing docs and **updated as the def pass writes each docstring** (its first
   line becomes the routine's contract), whose `callee_notes` injects a capped (`CALLEE_NOTES_CAP`, 6)
   "Functions/methods this routine calls:" block into a routine's generation turn.

The def pass takes three optional hooks (`doc_order`/`callee_context`/`on_doc`, threaded
`main`→`generate_comments`→`_def_pass`→each worker's `generate_language_comments`; absent ⇒ unchanged behaviour, so
existing callers/tests are unaffected): the worker processes in `doc_order` (via `apply_doc_order`, deepest-first
fallback keeps children ahead of parents), appends the callee notes, and calls `on_doc(qualname, docstring)` after
each routine so a callee in an earlier-ordered file informs a later caller.

## Lazy callee one-liners (def pass)

A *called but undocumented* routine used to contribute no note. Now the `callee_context` closure
(`scale._make_callee_oneliner_context`, bound once per run over `llm`/`cfg`/the run-file store/graph/store) fills
the gap **lazily**: before formatting a caller's notes it asks `store.missing_callee_contracts` which resolved
callees lack a contract, reads each one's signature+body from the retained store (`Symbol.start`/`end`), **elides it
to the snippet budget with the existing mechanism** (Python `elide_structurally` on the dedented body; C/JS the
`fit_snippet` crop), and makes a single `summarise(..., LENGTH_LINE)` call, then `store.update`s the result so every
later caller reuses it. Generation is **lazy** (only at the point of documenting a caller — a routine nothing calls
is never summarised; eager generation at graph build would have paid for unused routines), **shallow** (one level —
no recursion into the callee's own callees, so it cannot cascade or loop), and **failure-capped** (a callee that
yields nothing is marked attempted and not retried). The `ContractStore` stays a pure store; the generation logic
lives in the closure. Cost is bounded by the number of distinct used-but-undocumented callees in the run, and is
zero when everything called is documented.

## The routine's own doc as an ingest-and-update seed (def pass)

A routine being (re)documented that already has documentation now has that doc *in* its generation turn — the
routine-level analogue of the `--file-doc` description seed — so the model keeps what's accurate and updates what's
stale instead of re-deriving the contract blind. For **C/JS** the doc comment sits *above* the header, outside the
assembled snippet, so `generate_comments_c`/`generate_comments_js` read it from the current text
(`_doc_above_header`/`_doc_above_header_js`) and append an "already documented as follows — ingest and update" block
to the prompt; this needs no graph and fires on every run (single-file included). For **Python** the docstring is
inside the body, so the assembled snippet carries it natively and the prompt already says "updating any existing
comment".

## Block-pass callee annotations (read-side)

The routine's callee one-liners (including any lazily-generated ones — the def pass runs first) also reach the
**block pass**, as inline annotations on the call lines of each paragraph the comment turn reads:
`y = helper(x)  // helper: <one-liner>` (the language's own trailing-comment form via `CommentStyle.line_prefix`).
Because the block pass parses the *def pass's output*, the call-site lines recorded by the pre-pass are stale;
`scale._block_callee_notes` therefore re-runs `iter_symbols` over the **current** text (model-free) and matches each
fresh call site to its resolved callee by `(name, kind)` via `graph.call_map`, yielding
`{qualname -> {line -> "callee: one-liner"}}`, threaded
`generate_comments`→`_block_pass`→`annotate_blocks(callee_annotations=...)`→`request_block_comment(line_notes=...)`.
**Read-side only:** the annotation enriches what the model reads and is never written — the patcher works from the
pristine source lines, so a mismatch is harmless and the byte-for-byte guarantee is untouched.

## The C header/implementation doc-site (`--doc-site`, C only)

C documents an `extern` function where it is *declared* (the `.h` prototype), not where it is *defined* (the `.c`
body), so annotating a project's headers and sources together would otherwise put the contract in the wrong place. A
second **model-free pre-pass** (`scale_c.plan_doc_sites_c`, built by `main` from the same retained run-file store —
the C files are not re-loaded) parses every run C file for definitions (`iter_defs_with_info_c`) and **prototypes**
(`iter_decls_with_info_c` — `declaration` nodes with a real `function_declarator`, including multi-line and
`#ifdef`-nested ones; function-pointer/variable/typedef declarations and local declarations are ignored), grouped by
function name. Pairing is **confident-only** (mirrors the call graph): a name is redirected only when it has
**exactly one** definition in the run (`static` dupes ⇒ ambiguous ⇒ documented at the impl, logged).

Per `--doc-site`: **`auto`** (default) documents each **target** header prototype — prose generated **from the
unique definition's body** (the model still needs the body, but the doc is *placed* above the prototype), else from
the prototype alone — and **skips** the impl's def-docstring when that `.c` is a target; a decl-only function is
documented from its prototype; a def-only function is documented at the impl as before. **`impl`** never skips an
impl docstring (legacy), but still documents target prototypes from their own text so headers aren't blank. Only a
**target** header is ever written (a `--reference` file can still *supply* the body for prose).

The `CDocPlan` exposes, per file, the names to skip and the prototypes to document, plus an on-demand impl-snippet
provider (a pure read of the captured impl lines via `assemble_snippet_for_c`); `generate_comments_c` skips
redirected definitions and documents the prototype records, firing `on_doc` for both and recording each header's
full doc. **The impl's block pass is informed by the header doc:** after redirection the `.c` def has no doc above
it, so `iter_block_targets_c` takes a `doc_override(name)` (threaded `main`→`_block_pass`) that supplies the
header's generated doc as `BlockTarget.doc`. For this to be populated in time, `main` reorders targets so **a header
precedes the impl it is paired with** (`_order_header_before_impl`, composed with the call-graph `file_order`).

**Scope/limits:** C only; the patchers and byte-for-byte guarantee are untouched (a redirected impl is left exactly
as found); the only failure mode (header not a target) degrades to today's impl-site behaviour, never to corruption.
The decl symbols `iter_symbols` now also emits seed contracts but are **kept out of the free-function uniqueness
index** in `build_project_graph`, so a decl/def pair never makes a name ambiguous for call resolution.

## Scope/limits (the whole layer)

This layer belongs to the offline mode (the online emit is model-free and sends verbatim spans, so it needs no
priming context); resolution is confident-only and the file order is coarse (a forward/cyclic cross-file
reference falls back to the seed/empty contract — or a lazily-generated one-liner). The patchers and the
byte-for-byte guarantee are untouched — this layer only enriches the priming context and the visiting order.
