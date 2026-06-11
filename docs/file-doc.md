# The file-level header doccomment pass (`scale_filedoc.py`, `--file-doc`)

Adds or updates the **top-of-file description** — the prose at the head of a file that says what it's for — without
ever disturbing the shebang, copyright/author boilerplate, or **license** that usually shares that header. Supported
for **all three languages**: C and JS share a brace-language scanner (`scale_filedoc.scan_brace_leading_zone`, used
by `scale_c.file_doc_target_c` / `scale_javascript.file_doc_target_js`) over the leading `/* */`+`//` comment zone;
Python (`scale_python.file_doc_target_py`) targets the **module docstring** instead — see the Python note below.
Opt-in via `--file-doc`, combinable with `-c` and the block flags; it runs **last**.

## The description prose and the safety model

**The description prose is the pass-2 (published) whole-file summary**, generated to the same description spec as
the pass-1 summary that primed the function passes — but from the **current, annotated text's skeleton**
(re-rendered model-free inside `_file_doc_pass`), so it draws on the docstrings the def pass just wrote rather than
re-deriving the file's role from bodies. A file with no symbols falls back to summarising its whole current text.

With `--emit-reword`, each successful splice also records the spliced draft (plus any richer map-reduce `context`)
into the run's **header-reword manifest** (`scale_reword.py`): the project blurb + per-file name/role/draft/answer
slots, for a stronger model to reword the run's descriptions with cross-file consistency; `--apply-reword`
re-splices the answers model-free by exact draft match through the same preservation guard (a miss — the file
changed — is a safe no-op; prose only, no licence/boilerplate ever enters the manifest).

The safety model is the same split that protects the other passes — **the guarantee lives in the patcher, not the
model** — but applied to legal text: the local model is used *only* to **classify** which existing header lines are
the editable description (and the summary generation itself writes the prose). It never re-emits any preserved text;
a deterministic patcher slices every kept line from the original source and does a single insertion/replacement of
comment lines. Two independent nets back the classification:

- **License veto** (`looks_legal`): a deterministic keyword match (SPDX, `Copyright`/`(c)`/`©`, "all rights
  reserved", "licensed under", license names, warranty/redistribution text). Any classified line that smells legal
  is dropped from the editable range — even a misclassification can't overwrite a license. Over-matching is the safe
  failure mode.
- **Preservation guard** (`file_doc_preserved`): after building the splice, it asserts the prefix-before and
  suffix-after are byte-for-byte identical and that only blank/comment lines were removed or added (no code
  touched). Any failure → return the file unchanged. A botched classify or render is a no-op, never a corruption.

## The flow (`annotate_file_doc`)

The C adapter gathers the **whole leading-comment zone** — which may span several contiguous blocks (mixed `/* … */`
and `//`, blank-separated, **no intervening code**) — and marks the pure-content comment lines as
description-eligible (delimiters, single-line `/* … */`, and blank ` *` continuations are never eligible, so always
preserved). **Classify** (temp 0.0): the eligible lines are shown numbered; the model returns the description's
range (`START-END`/`N`/`NONE`), mapped back to source lines and clamped to the zone (so it can never point at code).
**Fetch the prose**: the engine calls a `summary_provider` callback with the existing description text as a **seed**
— that runs `_get_file_summary`, so the unified summary is generated (or cache-loaded) *incorporating the author's
wording* (ingest-and-update), and `_sanitise_description` strips any stray delimiters. There is **no separate
generate turn**. **Patch**: replace the existing description in place (re-wrapped in the host block's own
` * `/`// ` decoration), or — when the zone has no usable description — append one into the last block, or — when
there's no header at all — insert a fresh `/* … */` block at the top.

Only the classify wording is externalised, to `scale-cfg/filedoc.classify.txt` (built-in default as fallback, filled
brace-safely via `_fill`); the description wording lives in `summary.txt`. Because the provider is the shared
`_get_file_summary`, a `--file-doc -c -b` run reuses one cached summary.

## Python targets the module docstring

Not a `#` comment — a Python file's top-of-file description is its module docstring (a string literal).
`file_doc_target_py` (over `ast`) makes the docstring's pure-content lines the eligible/description lines (the
`"""`/`'''` delimiters and blank lines are preserved, so a licence *inside* the docstring is protected by the same
classify+veto), or inserts a fresh `"""..."""` as the first statement (after any shebang/coding-cookie/leading
comments) when there is none, rendered via `PYTHON_DOC_STYLE`. Because a docstring's content lines are *not*
recognisable as comments line-by-line, the line-comment guard doesn't fit; the target carries a **pluggable
preservation guard** (`FileDocTarget.preserved`) and Python supplies `_py_doc_preserved` — a parse-based guard
asserting the lines outside the splice are byte-identical, the result still parses, and the module's code (its AST
with any leading module docstring removed, `_module_code_signature`) is unchanged. This also rejects a description
that would break the docstring (e.g. an embedded triple-quote). C/JS leave `preserved` unset and use the default
`file_doc_preserved`.
