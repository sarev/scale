# Code style

## Body layout: blobs, not wall of statements

Function body = sequence of named "blobs", each doing one thing. Make joins visible:

- Blank line above every blob. No exceptions.
- One-line comment above blob only if code doesn't speak for itself. One line default; two-three when *why* needs prose; more = smell.
- Blob typically 2-15 lines. Past that, extract helper or split into sub-blobs.
- Comments read stand-alone. Don't assume reader has internal vocabulary loaded: plain words first, symbol names only when clearest label.

Avoid:
- Wall of statements, no blanks, no comments: joins invisible.
- Multi-paragraph comments mid-function. Function-wide context goes in doc-comment; if middle needs paragraphs, function doing two things, split.
- Comments restating next line ("increment counter" above increment).
- Comments only protocol/internals expert can parse.

## Comment style

- Say *why* it exists + anything notable about behaviour, not what code does (unless opaque). Code shows what; prose shows why.
- Multi-line/own-line: block-comment form, closing delimiter on own line if >1 line. Trailing remarks: line-comment form (compact).
- Single noteworthy line: trailing comment, not stand-alone block above (keeps blob contiguous).
- Record/struct fields: trailing comment same line, column-aligned (most formatters preserve column).
- No history vocab in code/docs: no "Commit N", "Phase N", "alpha N", "stubs", "earlier commit". Describe code as-is; history lives in version control.

## Function doc-comments

Every function def gets doc-comment immediately above (exceptions below). Format:
1. One-line summary, ends with full stop.
2. Optional prose: non-obvious contract, ordering constraints, perf notes, edge cases. Omit if one-liner says all.
3. `Parameters:` indented list, one per line, name + description. Omit if none.
4. `Returns:` value + possible states. Omit for void.

Keep lean. Prose past ~6 lines = probably restating code; cut.

Interface/implementation split: doc-comment on declaration (header/interface) for external functions, on definition for internal ones. Never both.

### When required

Full block required for every def, one exception: **trivial helpers**. Trivial = body ~3 lines, no surprising args, no surprising return, name+signature say all (byte accessors, one-line wrappers). Summary line suffices. When in doubt, write full block: redundant `Parameters:` cheaper than 3am surprise contract.

Also skip full block:
- **Test/scaffolding shims**: harness hooks, debug dumps. One summary line.
- **Repetitive dispatchers sharing one signature** (e.g. command-handler family, same args+return). Section banner documents shared shape; each gets one-liner naming what it handles.

### Prose worth including

- **Call-site context**: where called from, when purpose only makes sense in context. Saves search.
- **Reentrancy/threading**: e.g. "no re-entrant requests", "interrupt-safe", "from transient callback".
- **Ownership transfers**: e.g. "on success, `block` ownership passes to callee", when types don't show it.
- **No-op contracts**: "safe with null", "idempotent on released slots". Saves caller guard.
- **Side effects** invisible from signature.

## Doc-comment hygiene

- **One doc-comment per function.** Rewriting prose: delete old block, don't leave it floating as free-standing note. Caveat closing off one function goes *inside that function's* doc-comment, not in gap before next (reads as next function's intro).
- **Nothing between doc-comment and def** except blank lines. No buffer decl, guard, forward decl, or constant def in gap: orphans comment, confuses readers+tooling. Move offending line above comment or below function. (Section banner above first function in group = only exception.)
- **Section banners** above first function in grouped set: good, header at scroll-glance. Shared invariant: write intro in banner; don't restate in every member.
- **No commented-out code under doc-comment.** Prose says "may need X here": implement X or delete speculative lines. Dead code under live prose rots fast.
