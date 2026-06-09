#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

Shared, language-agnostic engine for the within-function "blob" pass.

Where the definition pass comments whole routines, this pass annotates *logical groups of statements inside a routine
body*: it places a blank line above each group and, only where the code does not speak for itself, a short comment.
The shape is deliberately different from the definition pass:

- Segmentation is line-based, but a per-language provider (see `BlockTarget`) decides which lines may legally begin a
  block, so the model can never split mid-statement.
- The model only ever returns line numbers (block boundaries) and comment text - never code.
- Patching is insertion-only above legal statement starts: every original code line survives exactly once, in order,
  unmodified; only blank and comment lines are added or rewritten. Each chunk gets a blank line above it - paragraphing
  a wall of statements into its blocks is value in itself - and a comment too where the model had something to say
  (`NONE` adds the blank but no comment, and never deletes an existing one). A belt-and-braces guard re-checks this
  invariant per routine and abandons any routine whose edit would alter code.

The provider, the line numbers, and the patcher together preserve SCALE's core guarantee that executable code is never
touched. Only the per-language `BlockTarget` provider and `CommentStyle` differ between languages; everything here is
shared.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from scale_llm import LocalChatModel, GenerationConfig, Messages, Chunk
from scale_log import echo
from scale_text import fit_snippet, MARKER_PYTHON
from typing import Dict, List, Optional, Tuple
import re


# Numbered-view formatting. A contiguous run of more than this many non-boundary lines (continuation lines, or the
# opaque body of a nested definition) is collapsed to a single elision band so the boundary view stays compact. Bands
# carry no line number, so the model cannot pick a line inside one.
MAX_CONTEXT_RUN = 6
ELISION_BAND = "« {n} lines elided »"


# Reply-length caps (tokens) for the two block-pass turns. Both replies are tiny - a list of line ranges, or a one- to
# three-line comment - so capping them keeps the prompt+reply within the context window (avoiding spurious budget
# warnings) and discourages the model from rambling into a multi-line "what the code does" list.
SEGMENT_REPLY_TOKENS = 256
COMMENT_REPLY_TOKENS = 200


# Sampling for the two block-pass turns, overriding the run-wide config (the global default suits the definition pass).
# Segmenting the body is a structural decision, so it is fully deterministic; comment prose uses a low temperature -
# enough to read naturally, low enough to stay precise and avoid the all-NONE / restatement drift seen at higher
# temperatures during tuning.
SEGMENT_TEMPERATURE = 0.0
COMMENT_TEMPERATURE = 0.1


# Per-turn prompt defaults, kept terse on purpose (the local model has a small context window). These are the built-in
# fallbacks; the CLI normally overrides them with the user-editable `scale-cfg/blocks.segment.txt` and
# `blocks.comment.txt`. Placeholders are filled by simple substitution (`_fill`), so a prompt may use any of:
# `{kind}`, `{qualname}` (both); `{view}` (segment - the numbered body); `{doc}` (function summary), `{priors}` (the
# comments already written for earlier paragraphs of this routine), `{block}` (comment - the paragraph's own lines).
# Other braces in the text are left untouched.
#
# The segment prompt asks for line RANGES, not split points: framing the task as "group related lines into chunks"
# stops the weak model hyper-focusing on breaking up individual lines (which over-segments). Each chunk's start is
# snapped to a legal statement-start line; the end just bounds the chunk shown to the comment pass.
SEGMENT_PROMPT = (
    "Body of {kind} `{qualname}`. Each numbered line may begin a chunk; others cannot.\n"
    "Group the body into a FEW chunks, each a run of lines doing one thing (~2-15 lines). Give each chunk's line "
    "range as `start-end` (start must be a numbered line). Few chunks, not many - group related lines, do not split "
    "every line.\n\n"
    "{view}\n"
)
# The comment prompt sees ONE paragraph at a time with only light context - the function's purpose and the comments
# already written for earlier paragraphs (the file overview is already in the primed conversation). This keeps the
# model focused on the paragraph rather than re-judging the whole function, and the running comments give a narrative
# thread (and stop it repeating itself). The bias is to comment: NONE only for the genuinely self-evident.
COMMENT_PROMPT = (
    "Function `{qualname}` does: {doc}\n\n"
    "{length_note}\n\n"
    "Comments already written for earlier paragraphs of it:\n{priors}\n\n"
    "Here is the NEXT paragraph of its body:\n\n{block}\n\n"
    "Give ONE short, useful comment for this paragraph - what this group of lines accomplishes, or the reason / "
    "gotcha / subtlety / edge case behind it - the kind of line that helps a reader scan the function. Avoid "
    "trivially restating a single obvious line. Reply NONE only if the paragraph is truly trivial.\n"
    "Bare sentence, or NONE. No #, no quotes, no list."
)
# A routine with at most this many chunks is "short": its docstring plus the visible code already walk a reader
# through it, so block comments that merely echo the docstring are noise. Longer routines benefit from a per-block
# walkthrough even if it lightly duplicates the docstring. The matching note is injected into the comment prompt as
# {length_note}, nudging the model conservative on short routines and permissive on long ones. (The threshold is a
# tuning knob, not prompt wording, so it lives here; the note *text* is overridable from scale-cfg - see below.)
SHORT_FUNCTION_CHUNKS = 3

# Defaults for the remaining block-pass prompt wording. Like SEGMENT_PROMPT/COMMENT_PROMPT these are built-in
# fallbacks; the CLI overrides them with user-editable scale-cfg files:
#   COMMENT_NOTE_SHORT -> blocks.note.short.txt      COMMENT_NOTE_LONG -> blocks.note.long.txt
#   COMMENT_NUDGE      -> blocks.comment.nudge.txt
# COMMENT_NUDGE is a follow-up used ONLY when the first reply is neither a clear comment nor a clear NONE - a gentle
# push to try once more before giving up (no placeholders; the paragraph is still in the conversation).
COMMENT_NOTE_SHORT = (
    "This is a short routine, so its docstring largely covers it: reply NONE unless the paragraph adds something "
    "not already clear from the docstring and the code itself."
)
COMMENT_NOTE_LONG = (
    "This is a longer routine: a brief one-line heading that walks the reader through this paragraph is welcome, "
    "even if it lightly echoes the docstring."
)
COMMENT_NUDGE = (
    "Have another go at that paragraph. Even a short note on what it accomplishes, the reason behind it, a gotcha "
    "or an edge case helps a reader. Give ONE short line, or NONE only if it really is trivial.\n"
    "Bare sentence, or NONE. No #, no quotes, no list."
)


def _fill(template: str, **fields: str) -> str:
    """
    Substitute `{name}` placeholders in a prompt template by literal replacement.

    Unlike `str.format`, this leaves any other braces in the text untouched, so a user-edited prompt file is free to
    contain stray `{` / `}` (and the substituted code snippets, which routinely contain braces, are never re-scanned).

    Parameters:
    - `template`: The prompt template text.
    - `fields`: Placeholder name to replacement value.

    Returns:
    - The template with each `{name}` replaced by its value.
    """

    out = template
    for name, value in fields.items():
        out = out.replace("{" + name + "}", value)
    return out


@dataclass(frozen=True)
class CommentStyle:
    """
    Describe how comments are rendered for a language, so the shared engine can emit block comments without knowing the
    language.

    A single-line comment always uses `line_prefix`. A multi-line comment uses the block form when `block_open` is set
    (delimiters on their own lines, each interior line prefixed with `block_cont`); otherwise every line uses
    `line_prefix` (as in Python, which has no block-comment syntax).

    Attributes:
    - `line_prefix`: The line-comment introducer, including any trailing space (e.g. "# ", "// ").
    - `block_open`: The opening delimiter for a multi-line block comment, or None to always use line comments.
    - `block_cont`: The continuation prefix for interior lines of a block comment (e.g. " * ").
    - `block_close`: The closing delimiter for a multi-line block comment.
    """

    line_prefix: str
    block_open: Optional[str] = None
    block_cont: Optional[str] = None
    block_close: Optional[str] = None


# Python has no block-comment syntax: every comment line uses "# ".
PYTHON_STYLE = CommentStyle(line_prefix="# ")


@dataclass(frozen=True)
class BlockTarget:
    """
    One routine (function, method, or class) the block pass may annotate, described purely in terms of line numbers.

    A provider produces one `BlockTarget` per routine body. `boundary_lines` is the set of lines that begin exactly one
    statement at any nesting depth within the body, without descending into nested definitions (a nested definition is
    a single opaque boundary). The model chooses a subset of these as block starts; the engine never invents a boundary
    the provider did not offer.

    Attributes:
    - `qualname`: The fully qualified routine name (for logging/prompts), e.g. "Foo.bar".
    - `kind`: The routine kind, e.g. "def", "async def", or "class".
    - `header_start`: The first line of the signature/header, including decorators (1-based).
    - `header_end`: The last header line (the line before the first body statement).
    - `body_start`: The first body line (1-based).
    - `body_end`: The last body line, inclusive.
    - `boundary_lines`: Sorted tuple of lines that may legally begin a block.
    - `indent_of`: Maps each boundary line to its exact leading-whitespace string.
    - `depth`: Nesting depth (0 = module level), used to order generation deepest-first.
    - `doc`: The routine's own docstring/header comment (or ""), used as brief context for the comment pass.
    """

    qualname: str
    kind: str
    header_start: int
    header_end: int
    body_start: int
    body_end: int
    boundary_lines: Tuple[int, ...]
    indent_of: Dict[int, str] = field(default_factory=dict)
    depth: int = 0
    doc: str = ""


# ---------------------------- comment helpers ----------------------------


def _is_comment_line(line: str, style: CommentStyle) -> bool:
    """
    Report whether a line is a pure comment line in the given language (ignoring leading whitespace).

    A line carrying code with a trailing comment is not a pure comment line and returns False.

    Parameters:
    - `line`: The source line to test.
    - `style`: The comment-style descriptor for the language.

    Returns:
    - True if the stripped line begins with the line-comment prefix or a block-comment delimiter.
    """

    s = line.lstrip()
    if not s:
        return False
    if s.startswith(style.line_prefix.strip()):
        return True
    for delim in (style.block_open, style.block_cont, style.block_close):
        if delim and s.startswith(delim.strip()):
            return True
    return False


def render_comment_lines(text: str, indent: str, style: CommentStyle) -> List[str]:
    """
    Render comment `text` as one or more source lines at the given indentation, using the language's comment style.

    A single line of text becomes a single line comment. Multiple lines become a block comment when the style provides
    block delimiters, or a run of line comments otherwise (e.g. Python).

    Parameters:
    - `text`: The comment body (without any comment delimiters); may contain newlines.
    - `indent`: The exact leading-whitespace string to prefix each rendered line with.
    - `style`: The comment-style descriptor for the language.

    Returns:
    - The rendered comment as a list of source lines.
    """

    lines = text.split("\n")

    # Line-comment form: one prefix per line (the only option when no block delimiters are defined).
    if style.block_open is None or len(lines) == 1:
        return [(f"{indent}{style.line_prefix}{ln}").rstrip() for ln in lines]

    # Block-comment form: delimiters on their own lines, interior lines carry the continuation prefix.
    out = [f"{indent}{style.block_open}"]
    cont = style.block_cont if style.block_cont is not None else style.line_prefix
    out.extend((f"{indent}{cont}{ln}").rstrip() for ln in lines)
    out.append(f"{indent}{style.block_close}")
    return out


def _parse_comment_reply(reply: str, style: CommentStyle) -> Optional[str]:
    """
    Turn a model reply for a single block into clean comment text, or None when no comment is wanted.

    The density gate is built in: an empty reply, or the explicit sentinel `NONE`, yields None (the common case). Any
    surrounding code fence is removed, and any comment delimiters the model echoed back are stripped so the result is
    plain prose ready for `render_comment_lines`.

    Parameters:
    - `reply`: The raw model reply.
    - `style`: The comment-style descriptor for the language.

    Returns:
    - The cleaned comment text, or None if the model declined to comment this block.
    """

    text = reply.strip()
    if not text:
        return None

    # Strip a surrounding ``` code fence if present (keep the fenced content).
    fence = re.match(r"^```[^\n]*\n(.*)\n```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Explicit decline sentinel, in whatever light wrapping the model used.
    if text.strip("`\"'* ").upper() == "NONE":
        return None

    # Strip any comment delimiters the model echoed back, line by line.
    delims = [d for d in (style.line_prefix.strip(), style.block_open, style.block_close) if d]
    cleaned: List[str] = []
    for ln in text.split("\n"):
        s = ln.strip()
        for d in delims:
            if s.startswith(d):
                s = s[len(d):].strip()
                break
        # Drop a lone block continuation marker (e.g. "*") left after stripping.
        if style.block_cont and s == style.block_cont.strip():
            s = ""
        cleaned.append(s)

    # Trim leading/trailing blank lines introduced by delimiter stripping.
    while cleaned and not cleaned[0]:
        cleaned.pop(0)
    while cleaned and not cleaned[-1]:
        cleaned.pop()

    result = "\n".join(cleaned).strip()
    if not result or result.upper() == "NONE":
        return None
    return result


def _is_explicit_none(reply: str) -> bool:
    """
    Report whether a reply is a deliberate "no comment needed" answer (rather than an empty / evasive non-answer).

    Used to decide whether to nudge the model again: a clear NONE is accepted as-is, but an unusable non-answer earns
    one more, gentler attempt.

    Parameters:
    - `reply`: The raw model reply.

    Returns:
    - True if the reply's first line is essentially the word NONE.
    """

    text = reply.strip()
    fence = re.match(r"^```[^\n]*\n(.*)\n```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    head = text.split("\n", 1)[0].strip().strip("`\"'*.: ")
    return head.upper().startswith("NONE")


# ---------------------------- numbered view (boundary pass) ----------------------------


def render_numbered_body(source_lines: Chunk, target: BlockTarget) -> str:
    """
    Render a routine's body for the segment pass, numbering only the lines that may legally begin a chunk.

    Boundary lines are shown with their original source line number in a left gutter (cat -n style); all other lines
    are shown without a number so the model cannot pick them. A long run of non-boundary lines - continuation lines, or
    the opaque body of a nested definition - is collapsed to a single elision band, keeping the view compact and
    legible regardless of body size. The signature is included (unnumbered) for context.

    Parameters:
    - `source_lines`: The full source split into lines (1-based addressing via index+1).
    - `target`: The routine to render.

    Returns:
    - The numbered body as a single string.
    """

    boundary = set(target.boundary_lines)
    gutter = max(4, len(str(target.body_end)))

    def numbered(n: int, text: str) -> str:
        """Render a boundary line with its number in the gutter."""
        return f"{str(n).rjust(gutter)}| {text}"

    def unnumbered(text: str) -> str:
        """Render a non-boundary line with an empty gutter."""
        return f"{' ' * gutter}| {text}"

    out: List[str] = []

    # Context header: the signature lines, unnumbered.
    for ln in range(target.header_start, target.header_end + 1):
        if 1 <= ln <= len(source_lines):
            out.append(unnumbered(source_lines[ln - 1]))

    # Body: number boundary lines; collapse over-long non-boundary runs to an elision band.
    run: List[str] = []

    def flush_run() -> None:
        """Emit the accumulated non-boundary run, eliding it to a band when it is over-long."""
        if not run:
            return
        if len(run) > MAX_CONTEXT_RUN:
            out.append(unnumbered(ELISION_BAND.format(n=len(run))))
        else:
            out.extend(unnumbered(t) for t in run)
        run.clear()

    for ln in range(target.body_start, target.body_end + 1):
        if not (1 <= ln <= len(source_lines)):
            continue
        text = source_lines[ln - 1]
        if ln in boundary:
            flush_run()
            out.append(numbered(ln, text))
        else:
            run.append(text)
    flush_run()

    return "\n".join(out)


def _parse_segments(reply: str, boundary_lines: Tuple[int, ...], body_end: int) -> List[Tuple[int, int]]:
    """
    Extract chunk line ranges from a model reply, snapping starts to legal statement boundaries.

    The model is asked for `start-end` ranges. Each start must be a legal boundary (statement start) or the range is
    dropped, so the model can never begin a chunk at an illegal line. Starts are deduplicated and sorted; each end is
    clamped to lie within the body and not to overlap the next chunk. If the reply contains no ranges at all, every
    bare number that is a legal boundary is taken as a chunk start (with the end inferred from the next start) - so the
    pass still works if the model ignores the range format.

    Parameters:
    - `reply`: The raw model reply.
    - `boundary_lines`: The legal boundary line numbers offered to the model.
    - `body_end`: The last line of the routine body (clamps the final chunk's end).

    Returns:
    - A list of `(start, end)` chunk ranges, sorted by start, non-overlapping, with legal starts.
    """

    allowed = set(boundary_lines)

    pairs = re.findall(r"(\d+)\s*[-–—]\s*(\d+)", reply)
    if pairs:
        candidates = [(int(a), int(b)) for a, b in pairs if int(a) in allowed]
    else:
        # No ranges given: fall back to bare numbers as starts (ends inferred below).
        starts = sorted({int(tok) for tok in re.findall(r"\d+", reply)} & allowed)
        candidates = [(s, body_end) for s in starts]

    # Deduplicate by start (keep the first range seen for each start), sorted ascending.
    seen: set = set()
    ordered: List[List[int]] = []
    for a, b in sorted(candidates, key=lambda p: p[0]):
        if a in seen:
            continue
        seen.add(a)
        ordered.append([a, b])

    # Clamp each end: at least the start, at most the body end, and never past the next chunk's start.
    for i, pair in enumerate(ordered):
        next_start = ordered[i + 1][0] - 1 if i + 1 < len(ordered) else body_end
        pair[1] = min(max(pair[1], pair[0]), body_end, next_start)

    return [(a, b) for a, b in ordered]


def request_segments(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    source_lines: Chunk,
    target: BlockTarget,
    prompt_template: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """
    Ask the model to group the routine body into chunk line ranges, and return the sanitised ranges.

    The numbered body view is appended as a turn and popped after the reply, keeping the persistent context bounded.
    The view is elided to the snippet budget first so a very large body still fits the context window. Each chunk's
    start is snapped to `target.boundary_lines` (an illegal start is dropped), so the model can never begin a chunk at
    a line that is not a legal statement start. Asking for ranges rather than split points keeps the model grouping
    related lines instead of fragmenting the body line by line.

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The generation configuration.
    - `messages`: The persistent priming context (mutated then restored).
    - `source_lines`: The full source split into lines.
    - `target`: The routine being segmented.

    Returns:
    - A list of `(start, end)` chunk ranges, sorted by start, non-overlapping, with legal starts.
    """

    view = render_numbered_body(source_lines, target)

    # Keep the view within the context window. Eliding the middle only loses some candidate lines (they simply will
    # not get a chunk); it never risks the output, which is patched from the real source.
    view, _ = fit_snippet(llm, cfg, messages, view, header_line_count=1, marker=MARKER_PYTHON)

    prompt = _fill(prompt_template or SEGMENT_PROMPT, kind=target.kind, qualname=target.qualname, view=view)
    messages.append({"role": "user", "content": prompt})
    turn_cfg = replace(cfg, temperature=SEGMENT_TEMPERATURE, max_new_tokens=min(cfg.max_new_tokens, SEGMENT_REPLY_TOKENS))
    reply = llm.generate(messages, cfg=turn_cfg)
    messages.pop()

    segments = _parse_segments(reply, target.boundary_lines, target.body_end)
    echo(f"[blocks] '{target.qualname}': {len(segments)} chunk(s) identified")
    return segments


# ---------------------------- comment pass ----------------------------


def _doc_summary(doc: str) -> str:
    """
    Reduce a routine docstring to a one-line summary for use as brief context.

    Takes the first non-empty line, which by convention is the docstring's summary sentence; returns a short
    placeholder when there is no docstring.

    Parameters:
    - `doc`: The routine's docstring (may be empty or multi-line).

    Returns:
    - A single summary line.
    """

    for line in doc.splitlines():
        if line.strip():
            return line.strip()
    return "(no description)"


def request_block_comment(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    source_lines: Chunk,
    target: BlockTarget,
    blob_start: int,
    blob_end: int,
    style: CommentStyle,
    prior_comments: Optional[List[str]] = None,
    length_note: str = "",
    prompt_template: Optional[str] = None,
    nudge_template: Optional[str] = None,
) -> Optional[str]:
    """
    Ask the model for a comment for one paragraph of a routine, or None when it is genuinely self-evident.

    The paragraph is shown on its own with only light context - the routine's purpose (its docstring summary) and the
    comments already written for earlier paragraphs of the same routine; the file overview is already in the primed
    conversation. This keeps the model focused on the paragraph rather than re-judging the whole function, and the
    running comments give a narrative thread (and discourage repetition). The bias is to comment: NONE is for the
    genuinely obvious only.

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The generation configuration.
    - `messages`: The persistent priming context (mutated then restored).
    - `source_lines`: The full source split into lines.
    - `target`: The routine the paragraph belongs to.
    - `blob_start`: The first line of the paragraph (a chosen chunk start).
    - `blob_end`: The last line of the paragraph, inclusive.
    - `style`: The comment-style descriptor for the language.
    - `prior_comments`: Comments already written for earlier paragraphs of this routine (narrative context).
    - `length_note`: A short note (short- vs long-routine) injected as `{length_note}` to tune the comment bias.
    - `prompt_template`: Optional override for the comment prompt (defaults to `COMMENT_PROMPT`).
    - `nudge_template`: Optional override for the retry nudge (defaults to `COMMENT_NUDGE`).

    Returns:
    - The comment text, or None if no comment should be written for this paragraph.
    """

    block_text = "\n".join(source_lines[blob_start - 1:blob_end])
    priors = "\n".join(f"- {c}" for c in (prior_comments or [])) or "(none yet)"

    prompt = _fill(
        prompt_template or COMMENT_PROMPT,
        kind=target.kind, qualname=target.qualname, doc=_doc_summary(target.doc),
        length_note=length_note, priors=priors, block=block_text,
    )
    turn_cfg = replace(cfg, temperature=COMMENT_TEMPERATURE, max_new_tokens=min(cfg.max_new_tokens, COMMENT_REPLY_TOKENS))

    appended = 0
    messages.append({"role": "user", "content": prompt})
    appended += 1
    reply = llm.generate(messages, cfg=turn_cfg)
    parsed = _parse_comment_reply(reply, style)

    # If the model neither gave a usable comment nor a clear NONE, push it once to try harder before giving up.
    if parsed is None and not _is_explicit_none(reply):
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": nudge_template or COMMENT_NUDGE})
        appended += 2
        reply = llm.generate(messages, cfg=turn_cfg)
        parsed = _parse_comment_reply(reply, style)

    for _ in range(appended):
        messages.pop()

    return parsed


# ---------------------------- insertion patcher ----------------------------


def _code_signature(lines: Chunk, style: CommentStyle) -> List[str]:
    """
    Return the executable lines of `lines`, dropping blanks and pure comment lines.

    Used by the safety guard to verify that a block-pass edit changed only blank and comment lines.

    Parameters:
    - `lines`: The lines to filter.
    - `style`: The comment-style descriptor for the language.

    Returns:
    - The lines that carry code (neither blank nor a pure comment), in order.
    """

    return [ln for ln in lines if ln.strip() and not _is_comment_line(ln, style)]


def code_preserved(old_lines: Chunk, new_lines: Chunk, style: CommentStyle) -> bool:
    """
    Report whether two versions of the source carry identical code, ignoring blank and comment lines.

    This is the block pass's safety net: because every edit is insertion or replacement of blank/comment lines, the
    code signature must be unchanged. A mismatch means something went wrong and the edit must be abandoned.

    Parameters:
    - `old_lines`: The original lines.
    - `new_lines`: The candidate edited lines.
    - `style`: The comment-style descriptor for the language.

    Returns:
    - True if the executable lines are identical in both versions.
    """

    return _code_signature(old_lines, style) == _code_signature(new_lines, style)


def _existing_comment_start(out_lines: Chunk, stmt_idx: int, indent: str, style: CommentStyle) -> int:
    """
    Find the first index of the contiguous comment block sitting directly above a statement at the same indent.

    Only comment lines immediately above the statement, at exactly the statement's indentation, are treated as that
    block's own comment (so an unrelated, differently-indented comment is left untouched).

    Parameters:
    - `out_lines`: The working copy of the source lines.
    - `stmt_idx`: The 0-based index of the statement line.
    - `indent`: The statement's exact leading-whitespace string.
    - `style`: The comment-style descriptor for the language.

    Returns:
    - The 0-based start index of the existing comment block, or `stmt_idx` when there is none.
    """

    cs = stmt_idx
    i = stmt_idx - 1
    while i >= 0 and _is_comment_line(out_lines[i], style):
        leading = out_lines[i][: len(out_lines[i]) - len(out_lines[i].lstrip())]
        if leading != indent:
            break
        cs = i
        i -= 1
    return cs


def _apply_edits(
    source_lines: Chunk,
    edits: List[Tuple[int, Optional[str], str]],
    style: CommentStyle,
) -> Chunk:
    """
    Apply block edits to a copy of the source, inserting blank/comment lines above chosen statement starts.

    Edits are applied in reverse line order so earlier indices stay valid as lines are inserted. Every chunk gets a
    blank line above it (inserted only if absent - pre-existing blanks are never removed), which paragraphs a wall of
    statements into its blocks. Where the model returned comment text, the patcher additionally replaces an existing
    same-indent comment block directly above the statement, or inserts a fresh comment; a None comment adds only the
    blank and keeps any existing comment untouched (it never deletes one). Code lines are never moved.

    Parameters:
    - `source_lines`: The source split into lines (not mutated).
    - `edits`: Tuples of (boundary_line, comment_or_None, indent), one per chosen block start.
    - `style`: The comment-style descriptor for the language.

    Returns:
    - A new list of lines with the edits applied.
    """

    out = source_lines[:]

    for boundary, comment, indent in sorted(edits, key=lambda e: e[0], reverse=True):
        stmt_idx = boundary - 1
        if not (0 <= stmt_idx < len(out)):
            continue

        cs = _existing_comment_start(out, stmt_idx, indent, style)
        has_existing = cs < stmt_idx

        if comment:
            # New text: replace any existing same-indent comment, or insert above the statement.
            body = render_comment_lines(comment, indent, style)
            start = cs
        elif has_existing:
            # NONE never deletes: keep the existing comment exactly as it is, re-anchored below the blank.
            body = out[cs:stmt_idx]
            start = cs
        else:
            # NONE with no comment: paragraph the chunk anyway (a blank line breaks the wall into its blocks).
            body = []
            start = stmt_idx

        # Ensure a blank line above the chunk, unless one is already there or it is the file start.
        need_blank = start - 1 >= 0 and out[start - 1].strip() != ""
        out[start:stmt_idx] = ([""] if need_blank else []) + body

    return out


def apply_blocks(
    source_lines: Chunk,
    target: BlockTarget,
    blocks: List[Tuple[int, Optional[str]]],
    style: CommentStyle,
) -> Chunk:
    """
    Apply one routine's chosen blocks to the source, guarding that only blank/comment lines change.

    This is the single-routine entry point (used directly by tests and, per routine, by `annotate_blocks`). It renders
    and inserts the edits, then verifies the code signature is unchanged; if the guard fails, the original lines are
    returned untouched so a faulty edit can never corrupt the routine.

    Parameters:
    - `source_lines`: The source split into lines (not mutated).
    - `target`: The routine the blocks belong to (provides per-boundary indentation).
    - `blocks`: Tuples of (boundary_line, comment_or_None), one per chosen block start.
    - `style`: The comment-style descriptor for the language.

    Returns:
    - The edited lines, or the original lines unchanged if the safety guard failed.
    """

    edits = [(b, comment, target.indent_of.get(b, "")) for b, comment in blocks]
    candidate = _apply_edits(source_lines, edits, style)
    if not code_preserved(source_lines, candidate, style):
        echo(f"[blocks] Skipped '{target.qualname}': edit would alter code; keeping original")
        return source_lines[:]
    return candidate


# ---------------------------- orchestration ----------------------------


def annotate_blocks(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    source_lines: Chunk,
    targets: List[BlockTarget],
    style: CommentStyle,
    segment_prompt: Optional[str] = None,
    comment_prompt: Optional[str] = None,
    comment_nudge: Optional[str] = None,
    note_short: Optional[str] = None,
    note_long: Optional[str] = None,
) -> Chunk:
    """
    Run the full block pass over every routine and return the annotated source.

    Routines are processed deepest-first (consistent with the definition pass) so a model turn is small and self-
    contained; nested definitions are already opaque boundaries, so each routine's edits fall on distinct lines. For
    each routine the segment pass groups the body into chunk ranges and the comment pass writes (or declines) a comment
    per chunk; the routine's edits are kept only if they survive the per-routine code-preservation guard. All surviving
    edits are finally applied in one reverse-order pass over the original lines, so line shifts can never invalidate an
    edit.

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The generation configuration.
    - `messages`: The persistent priming context for this pass.
    - `source_lines`: The source split into lines (not mutated).
    - `targets`: The routines to annotate.
    - `style`: The comment-style descriptor for the language.
    - `segment_prompt`: Optional override for the segment-pass prompt template (defaults to `SEGMENT_PROMPT`).
    - `comment_prompt`: Optional override for the comment-pass prompt template (defaults to `COMMENT_PROMPT`).
    - `comment_nudge`: Optional override for the comment retry nudge (defaults to `COMMENT_NUDGE`).
    - `note_short`/`note_long`: Optional overrides for the short-/long-routine length notes (defaults
      `COMMENT_NOTE_SHORT`/`COMMENT_NOTE_LONG`).

    Returns:
    - The annotated source split into lines.
    """

    ordered = sorted(targets, key=lambda t: (t.depth, t.body_start, -t.body_end), reverse=True)

    all_edits: List[Tuple[int, Optional[str], str]] = []
    for target in ordered:
        if not target.boundary_lines:
            continue

        segments = request_segments(llm, cfg, messages, source_lines, target, segment_prompt)
        if not segments:
            continue

        # Short routines lean conservative (don't echo the docstring); longer ones invite a per-block walkthrough.
        if len(segments) <= SHORT_FUNCTION_CHUNKS:
            length_note = note_short or COMMENT_NOTE_SHORT
        else:
            length_note = note_long or COMMENT_NOTE_LONG

        # One comment turn per chunk, in body order, feeding earlier comments forward as narrative context.
        edits: List[Tuple[int, Optional[str], str]] = []
        prior_comments: List[str] = []
        for blob_start, blob_end in segments:
            comment = request_block_comment(
                llm, cfg, messages, source_lines, target, blob_start, blob_end, style,
                prior_comments=prior_comments, length_note=length_note,
                prompt_template=comment_prompt, nudge_template=comment_nudge,
            )
            if comment:
                prior_comments.append(comment)
            edits.append((blob_start, comment, target.indent_of.get(blob_start, "")))

        # Per-routine guard: simulate this routine's edits on the pristine source and keep them only if code is intact.
        trial = _apply_edits(source_lines, edits, style)
        if code_preserved(source_lines, trial, style):
            all_edits.extend(edits)
        else:
            echo(f"[blocks] Skipped '{target.qualname}': edit would alter code; keeping original")

    return _apply_edits(source_lines, all_edits, style)
