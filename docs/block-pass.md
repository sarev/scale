# The within-function block pass (`scale_blocks.py`, `--block-spacing`/`--block-comments`)

Annotates logical groups of statements ("blobs") inside routine bodies, not whole definitions. It is line-based but
never splits mid-statement, and is **insertion-only** so the byte-for-byte code guarantee holds.

## Provider

(`scale_python.iter_block_targets`, `scale_c.iter_block_targets_c`, `scale_javascript.iter_block_targets_js`)
returns one `BlockTarget` per function/method/class body. `boundary_lines` are the lines that begin **exactly one
statement at the line's first non-blank column**, collected at all depths but **not descending into nested defs** (a
nested def is one opaque boundary at its decorator-aware header line). Excluded: multi-statement (`a; b`),
inline-compound (`if x: y`), continuation lines, a leading docstring, and the **first statement of an inner suite**
(a blank as the first line of an `if`/`for`/… body reads badly; the routine's own first body statement stays
eligible). The C/JS providers mirror this over tree-sitter (walking `compound_statement`/`statement_block` suites
via small `_c_*`/`_js_*` helpers; C has no nesting, JS treats a nested function/class/function-valued binding as one
opaque boundary; JS skips class bodies — methods are targeted individually). Adding a language = implement this
provider + a `blocks.<lang>.txt` template + wire `_block_provider_for`/`_block_style_for` in `scale.py`.

## Segment pass

Produces the chunk `(start, end)` ranges. **All three languages use a deterministic structural segmenter**, *not*
the model: a provider precomputes the chunks and sets `BlockTarget.segments`, and `annotate_blocks` uses them when
present. The paragraph **rules live once** in the language-agnostic `scale_blocks.structural_breaks`, which operates
on normalised `SegStatement` records; each worker only emits records (`scale_python.structural_segments` over
CPython `ast`; the C/JS providers over tree-sitter). It places a paragraph break — only ever at a legal boundary
line — for: the first statement after a docstring (when the body has one); the statement after a nested def/class (a
def clearly ends a paragraph, ungated); a `return` whose preceding statement is at the same depth; a
compound/nested-def block of ≥ `SEG_MIN_BLOCK_LINES` (3) source lines; and the statement resuming after such a block
closes (a dedent).

- **The trailing-return rule has a refinement:** a suite that is exactly `[simple_stmt, return]` is *one* paragraph
  anchored at the leading statement (the return carries a `merge_anchor` set by the provider), not split before the
  return — so a tiny `[stmt; return]` block reads as a unit and, crucially, the comment pass sees the statement the
  return depends on rather than a bare `return` (a bare-return chunk gives the model nothing to describe and it
  hallucinates).
- **A leading-declaration heuristic (C/JS):** when a scope opens with ≥ `SEG_MIN_LEADING_DECLS` (2) local variable
  declarations, the provider sets `force_break` on the first real statement after them, so the declarations form
  their own paragraph and the body doesn't run straight into them (a single leading decl doesn't trigger it — that
  would over-fragment).
- **The opening paragraph is always emitted as a chunk** — `structural_breaks` prepends `[first body statement,
  first break − 1]` — so the routine's first block is summarised and scored like any other rather than silently
  dropped; a trivial opener (e.g. a declaration block) simply scores low and gets no comment, but a meaningful one is
  not lost (the patcher places no blank above it, since it sits right under the header). The anchor is the suite's
  first line (normally excluded), which the provider adds as a boundary; the patcher then suppresses the leading
  blank when a chunk sits right after a block opener (the line above ends `{`/`:`), so the comment lands directly
  under the opener.
- Two flags capture the only cross-language differences: `allow_first_in_scope` (off for brace languages — no
  in-body docstring) and `allow_after_def` (off for C — no nested functions). The `depth` field is any
  order-preserving nesting proxy (Python passes the source column; C/JS pass parent-chain depth).
- **Size, not cognitive complexity, gates triviality** (cc is nesting-dominated and wrongly demotes long-but-flat
  blocks). This is free, reproducible, and — being model-free — also makes the escalation manifest's chunk recipe
  deterministic across emit/apply. The Python segmenter was developed/validated against a corpus with
  `tests/block_eval/segment_eval.py` (model-free harness: wall a file, re-segment, diff vs the human paragraphing).
- **LLM fallback** (`request_segments`): now only a safety net for a target with no structural segmenter; it renders
  the body as a numbered view (`render_numbered_body`, a line number **only on boundary lines**; over-long
  non-boundary runs collapse to a `« N lines elided »` band) and asks for **chunk line ranges** (`start-end`), not
  split points; `_parse_segments` snaps each start to a legal boundary and clamps ends. Run **deterministically**
  (`SEGMENT_TEMPERATURE = 0.0`).

## Comment pass (`request_block_comment`)

**Two turns per paragraph**, addressing a systemic weakness: the model only ever sees its own paragraph's code plus
the file/function docs and the notes written for *earlier* paragraphs, so a silently-skipped paragraph used to leave
its successors blind to what it did (a prime source of hallucination — e.g. a bare `return true` chunk with no
preceding statement to describe).

- **Turn 1 (summary)** therefore *always* asks for a one-line description — there is no `NONE` opt-out — and
  **every** summary is kept in the running `{priors}` context, even ones that won't be written.
- **Turn 2 (score)** rates that summary on a deliberately narrow **1–3** scale — **1** restates the code, **2**
  signposts the block, **3** explains intent/a reason/a gotcha — for how much it helps a reader of the *code*. (The
  scale was 1–5; a diagnostic showed the 7B collapsed it — never using 2, parking ~80% at 4 — so the wide middle
  carried no signal. 1–3 with each rung spelled out, plus "restating the code is a 1 however well worded", gives
  cleaner signal; turn 1 is also coached to capture the paragraph's *point* rather than narrate the statements.)
- Only summaries scoring ≥ the threshold are inserted; a low-scoring one is suffixed with the magic `VALUE_FLAG`
  (`{@X@}`) so it still flows into later turns' context (which ignore it) but is recognised and skipped at output
  (`_comment_to_insert`) — carrying the value verdict on the comment string keeps the blast radius tiny (the return
  type stays `Optional[str]`). The threshold is `COMMENT_VALUE_THRESHOLD` (2 — drop bare restatements), surfaced on
  the CLI as **`--block-comments {high,medium,low}`** (high→1 keeps all, medium→2, low→3 intent-only;
  `--block-spacing` alone → 4, which no score clears, so the comment turns are skipped and the pass only
  paragraphs). `_parse_score` clamps the reply to [1,3] so an old-scale lapse (a stray 4/5) reads as top-of-scale.
  The `{length_note}` (short vs long routine, `SHORT_FUNCTION_CHUNKS`) biases the *score* turn's strictness.
- `BlockTarget.doc` is the routine's own documentation: Python reads it with `ast.get_docstring` on the
  (def-pass-annotated) parse; the C/JS providers read the comment block immediately above the header
  (`_doc_above_header`/`_doc_above_header_js`, reusing `_scan_existing_comment_block_above`). Because the block pass
  parses the **def pass's output**, a `-c --block-comments` run feeds each routine's freshly-written header doc into
  its own block-comment context — without it the model comments bodies blind to the routine's contract.
- When the run has a call graph, the paragraph's **call lines are additionally annotated read-side** with their
  callees' one-liners (`request_block_comment(line_notes=...)` / `annotate_blocks(callee_annotations=...)`; see
  [project-context.md](project-context.md)) — shown to the model in the language's own trailing-comment form, never
  written to the output.
- Turn 1 is `COMMENT_TEMPERATURE = 0.1`; the score turn is deterministic (`SCORE_TEMPERATURE = 0.0`, single-digit
  reply). Append-then-pop per paragraph; an unusable turn-1 reply is **nudged once** (`COMMENT_NUDGE`) to force a
  description.

## Patcher (`apply_blocks` / `_apply_edits`)

In reverse line order, **every chunk gets a blank line above it** (insert-if-absent, never removing pre-existing
blanks) — paragraphing a wall of statements into its blocks is value in itself, and is safe now that
range-segmentation yields few, sensible chunks rather than line-by-line fragments. **Exception:** a chunk that sits
at the start of a just-opened block (the line above ends with `{` or `:`) gets *no* leading blank — a blank as the
first line inside a brace/suite reads badly (this is what lets a merged `[stmt; return]` paragraph be commented
directly under the opener). Where the model returned comment text it additionally replaces an existing same-indent
comment or inserts a fresh one; `NONE` adds only the blank and **keeps any existing comment untouched** (never
deletes a hand-written one). A **code-preservation guard** (`code_preserved`, comparing the non-blank/non-comment
line signature) abandons any routine whose edit would alter code. `annotate_blocks` generates deepest-first, then
applies all surviving edits in one reverse-order pass (chunks across routines are disjoint thanks to nested-def
opacity).

## Prompts are externalised/tunable

Every piece of block-pass prompt *wording* lives in `scale-cfg` as a user-editable file, with a built-in default
constant in `scale_blocks.py` as fallback: `blocks.segment.txt` (`SEGMENT_PROMPT`), `blocks.comment.txt`
(`COMMENT_PROMPT`, the turn-1 summary), `blocks.score.txt` (`SCORE_PROMPT`, the turn-2 value score),
`blocks.comment.nudge.txt` (`COMMENT_NUDGE`), `blocks.note.short.txt` / `blocks.note.long.txt`
(`COMMENT_NOTE_SHORT`/`_LONG`, biasing the score), plus the per-language priming templates `blocks.python.txt` /
`blocks.c.txt` / `blocks.js.txt`. Files are filled by literal `{placeholder}` substitution via `_fill`, so stray
braces and code with braces are safe. Non-wording tuning knobs stay as code constants (`SHORT_FUNCTION_CHUNKS`,
`COMMENT_VALUE_THRESHOLD` — the `--block-comments` default, mapped from `low`/`medium`/`high` by
`BLOCK_COMMENT_LEVELS` in `scale.py`, `*_TEMPERATURE`, `*_REPLY_TOKENS`). Keep prompts terse — the window is small.

## Reality check (7B local model)

*Paragraphing* (blank lines splitting a wall into its blocks) is the dependable win — structural segmentation finds
boundaries that line up with where a human would break the code, and the `[stmt; return]` merge keeps tiny blocks
whole. *Comments* are capped by model capability, but two structural fixes raise the floor: (1) the block pass
parses the def-pass output so each routine's **header doc** is in the comment context, and (2) the
**always-summarise-then-score** two-turn pass means every paragraph contributes its truth to the running `{priors}`,
so a later paragraph is never reasoning about state an earlier (uncommented) one silently changed — the specific gap
that produced the worst hallucinations. The 1–3 **value score** + `--block-comments` density then filters bland
restatements out of the code (keeping them only as context). Residual limits remain — the 7B still mislabels
genuinely ambiguous near-duplicate branches — so materially better *prose* still wants a stronger model for the
comment turns (segment/comment/score turns are separate; a stronger model could drive comments alone, or via the
escalation manifest for the complex routines).
