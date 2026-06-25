#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The language-agnostic engine for SCALE's block pass, which inserts paragraph spacing and short comments inside routine
bodies. Each language worker supplies `BlockTarget` records naming the lines where a comment may legally be placed;
everything from there - segmentation, generation, scoring and patching - is shared across languages.

Paragraphs are found either by the deterministic `structural_breaks` rules (docstrings, nested defs, returns,
substantial nested blocks and the dedents out of them) or by a model segmentation turn over a numbered view of the
body. Each paragraph then gets a two-turn comment-and-score exchange in `request_block_comment`, with an empty-reply
nudge, a grounding correction and an obviousness challenge deciding whether the comment is kept.

All edits are insertion-only and pass through `code_preserved`, the guard that proves the executable code is untouched
before a routine's changes are accepted. `apply_blocks` replays ready-made manifest answers through the same guard,
and `defer_block_targets` records targets in the online run manifest instead of annotating them locally.
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


# The comment turn shows each paragraph inside a raw window of surrounding source - this many lines either side,
# clamped to the routine - with the paragraph's own lines gutter-marked `> `. A bare paragraph used to be all the
# model saw, which drove bland or hallucinated notes; the window lets it read the neighbouring code without being
# asked to describe it.
BLOCK_CONTEXT_LINES = 8

# The window's lower edge is nudged further back to the paragraph's enclosing scope opener (the nearest line above
# with strictly smaller indentation) so a paragraph deep in an `if`/loop is read knowing what guards it. The walk up
# is bounded by this many lines; past it, the plain ±N window stands.
BLOCK_SCOPE_NUDGE_CAP = 24


# Reply-length caps (tokens) for the two block-pass turns. Both replies are tiny - a list of line ranges, or a one- to
# three-line comment - so capping them keeps the prompt+reply within the context window (avoiding spurious budget
# warnings) and discourages the model from rambling into a multi-line "what the code does" list.
SEGMENT_REPLY_TOKENS = 256
COMMENT_REPLY_TOKENS = 200
SCORE_REPLY_TOKENS = 8        # the value turn replies with a single digit


# Sampling for the block-pass turns, overriding the run-wide config (the global default suits the definition pass).
# Segmenting the body and scoring a comment's value are decisions, so they are fully deterministic; comment prose uses
# a low temperature - enough to read naturally, low enough to stay precise and avoid restatement drift.
SEGMENT_TEMPERATURE = 0.0
COMMENT_TEMPERATURE = 0.1
SCORE_TEMPERATURE = 0.0


# Per-turn prompt defaults, kept terse on purpose (the local model has a small context window). These are the built-in
# fallbacks; the CLI normally overrides them with the user-editable `scale-cfg/blocks.segment.txt`,
# `blocks.comment.txt` and `blocks.score.txt`. Placeholders are filled by simple substitution (`_fill`), so a prompt
# may use any of: `{kind}`, `{qualname}` (any turn); `{view}` (segment - the numbered body); `{doc}` (function
# summary), `{priors}` (one-line notes on earlier paragraphs), `{block}` (comment/score - the paragraph inside its
# raw context window, the paragraph's own lines gutter-marked `> `); `{length_note}` (score - the short/long
# strictness hint). Other braces in the text are left untouched.
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
# Two model turns annotate each paragraph. TURN 1 (the comment/summary) sees ONE paragraph with only light context -
# the function's purpose and one-line notes on the earlier paragraphs (the file overview is already primed). It must
# ALWAYS describe what the paragraph does in one line - never an opt-out. That line plays two roles: a candidate code
# comment, and the running record (`priors`) every later paragraph depends on, so the model is never left guessing
# what an earlier (uncommented) paragraph did - the gap that used to drive hallucinations.
COMMENT_PROMPT = (
    "Function `{qualname}` does: {doc}\n\n"
    "One-line notes on the earlier paragraphs of its body (your running context):\n{priors}\n\n"
    "Here is the NEXT paragraph of its body - the lines marked `>`. The unmarked lines are surrounding code, shown "
    "only so you can read the paragraph in context:\n\n{block}\n\n"
    "In ONE short line, capture this paragraph's POINT: why it is here, what it accomplishes for the function, or the "
    "reason / gotcha / subtlety / edge case behind it. Describe ONLY the `>`-marked lines, never the unmarked context. "
    "Assume the reader can already read the code - do NOT narrate the statements or restate what the lines plainly "
    "say. If the paragraph really is just a mechanical step, say that plainly and briefly. Always give a real "
    "description (later paragraphs rely on it); never reply 'NONE' or leave it blank.\n"
    "Bare sentence. No #, no quotes, no list."
)
# TURN 2 (the value score) judges whether that one-line note earns a place in the code, on a deliberately narrow 1-3
# scale (the small model collapsed the old 1-5 scale - it never used 2 and parked ~80% at 4, so the wide middle
# carried no signal). The three points are spelled out so the model knows what each means, and restatement is named
# as the bottom rung. The {length_note} biases strictness by routine size. EVERY note is kept as running context, but
# only notes scoring >= COMMENT_VALUE_THRESHOLD are inserted into the code; a low-scoring note is tagged with
# VALUE_FLAG - a magic, otherwise-meaningless marker - so it still flows into later turns' context (which ignore it)
# yet is recognised and skipped at the output stage. Keeping the carrier on the comment string (rather than a second
# return value) keeps the blast radius tiny.
SCORE_PROMPT = (
    "{length_note}\n\n"
    "Score the one-line note you just wrote: how much would it help a READER OF THE CODE, placed as a comment above "
    "that paragraph (the `>`-marked lines)?\n"
    "  1 - it just restates what the code already says (no insight beyond reading the lines themselves)\n"
    "  2 - it signposts the block - a heading that helps a reader navigate, even if it lightly echoes the code\n"
    "  3 - it explains intent, a reason, a gotcha, or behaviour that is not obvious from the code\n"
    "A note that only restates the code is a 1, however well it is worded. Reply with a single digit 1, 2 or 3 and "
    "nothing else."
)
VALUE_FLAG = "{@X@}"        # magic marker tagging a low-value note: kept for context, skipped at output
CHALLENGE_FLAG = "{@F@}"    # marker tagging a note that failed verification twice: kept for context, never written
COMMENT_VALUE_THRESHOLD = 2  # minimum 1-3 value score for a note to be written into the code (drop bare restatements)

# A routine with at most this many chunks is "short": its docstring plus the visible code already walk a reader
# through it, so its notes are scored strictly. Longer routines invite a per-block walkthrough. The matching note is
# injected into the SCORE prompt as {length_note}, biasing the model strict on short routines and generous on long
# ones. (The threshold itself is the hard knob above; the note *text* is overridable from scale-cfg - see below.)
SHORT_FUNCTION_CHUNKS = 3

# Defaults for the remaining block-pass prompt wording. Like SEGMENT_PROMPT/COMMENT_PROMPT/SCORE_PROMPT these are
# built-in fallbacks; the CLI overrides them with user-editable scale-cfg files:
#   COMMENT_NOTE_SHORT -> blocks.note.short.txt      COMMENT_NOTE_LONG -> blocks.note.long.txt
#   COMMENT_NUDGE      -> blocks.comment.nudge.txt   SCORE_PROMPT       -> blocks.score.txt
# COMMENT_NUDGE is a follow-up used ONLY when the first reply is not a usable description - a push to give one rather
# than punt (no placeholders; the paragraph is still in the conversation).
COMMENT_NOTE_SHORT = (
    "This is a short routine - its docstring and code already make it clear, so be strict: give 1 to anything that "
    "merely restates the code, and reserve 3 for a note that genuinely reveals intent or a gotcha."
)
COMMENT_NOTE_LONG = (
    "This is a longer routine, so a clear signpost earns its place: a good navigational heading is a 2 even if it "
    "lightly echoes the code - but still give 1 to a bare restatement, and 3 to genuine intent or a gotcha."
)
COMMENT_NUDGE = (
    "Describe what that paragraph (the `>`-marked lines) does in ONE short line - even if it is simple, say plainly "
    "what it accomplishes. Do not reply 'NONE' and do not leave it blank.\n"
    "Bare sentence. No #, no quotes, no list."
)


def _fill(template: str, **fields: str) -> str:
    """
    Substitute `{name}` placeholders in a template with the given field values.

    Uses plain sequential replacement rather than `str.format`, so braces that do not name a supplied field pass through untouched.

    Parameters:
    - `template`: Text containing `{name}` placeholders.
    - `fields`: Placeholder names mapped to their replacement strings.

    Returns:
    - The template with every supplied placeholder replaced.
    """

    # Sequential replace rather than str.format, so unrelated braces in the template cannot raise.
    out = template
    for name, value in fields.items():
        out = out.replace("{" + name + "}", value)
    return out


@dataclass(frozen=True)
class CommentStyle:

    """
    Comment delimiters for one source language.

    `line_prefix` is always present; the three `block_*` delimiters exist only for languages with a block-comment form, and rendering falls back to line comments when they are `None`.
    """

    # The block_* delimiters stay None for languages with no block-comment form, forcing line-comment rendering.
    line_prefix: str
    block_open: Optional[str] = None
    block_cont: Optional[str] = None
    block_close: Optional[str] = None


# Python has no block-comment syntax: every comment line uses "# ".
PYTHON_STYLE = CommentStyle(line_prefix="# ")

# C and JavaScript share `//` line comments and `/* ... */` block comments. The block pass defaults to the line
# form (a one-line `//` section header reads cleanly, mirroring Python's `#`); `--block-comment-style block`
# selects the block form, which only differs for the rare multi-line block comment (a single-line header always
# renders as `// ...` regardless). C99+ is required for `//` in C.
SLASH_LINE_STYLE = CommentStyle(line_prefix="// ")
SLASH_BLOCK_STYLE = CommentStyle(line_prefix="// ", block_open="/*", block_cont=" * ", block_close=" */")


@dataclass(frozen=True)
class BlockTarget:

    """
    One routine body targeted by the block pass.

    All line numbers are 1-based positions in the file. `boundary_lines` holds the only lines where a comment may be inserted, `indent_of` gives the indentation to use at each of them, and `segments`, when present, carries pre-computed paragraph ranges that bypass the structural segmenter.
    """

    # Line numbers are 1-based file positions; boundary_lines lists the only legal comment insertion points.
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
    sig: str = ""
    segments: Optional[List[Tuple[int, int]]] = None


# ---------------------------- structural segmentation (deterministic paragraph rules) ----------------------------


# A block (compound statement or nested definition) must span at least this many source lines to earn a paragraph
# break before it, or before the statement that resumes after it. Size is a better "is this trivial" gate than
# cognitive complexity, which is nesting-dominated and wrongly demotes long-but-flat blocks. 3 generalises across
# codebases (5 overfit one repo's style).
SEG_MIN_BLOCK_LINES = 3

# A scope that opens with at least this many local variable declarations has its first real (non-declaration)
# statement paragraphed off from them - so the declarations read as their own block and the body does not run
# straight into them. 2 ("a bunch") avoids over-fragmenting a single leading declaration. C/JS only (Python has no
# declaration statements); set by the providers via `SegStatement.force_break`.
SEG_MIN_LEADING_DECLS = 2


@dataclass(frozen=True)
class SegStatement:

    """
    Structural facts about one body statement, as consumed by the segmenter's break rules.

    `opens_block` and `closed_block` give the size of the block a statement opens or has just closed, `merge_anchor` ties a trailing return to the statement it belongs with, and `force_break` lets a provider demand a paragraph break regardless of the other rules.
    """

    # Per-statement structural facts the break rules consume; merge_anchor glues a trailing return to the statement it belongs with.
    start: int
    end: int
    depth: int
    is_return: bool
    is_def: bool
    opens_block: int
    first_in_scope: bool
    closed_block: int
    merge_anchor: Optional[int] = None
    force_break: bool = False


def structural_breaks(
    stmts: List[SegStatement],
    *,
    has_doc: bool,
    boundary_lines: Tuple[int, ...],
    body_end: int,
    min_block_lines: int = SEG_MIN_BLOCK_LINES,
    allow_after_def: bool = True,
    allow_first_in_scope: bool = True,
) -> List[Tuple[int, int]]:
    """
    Split a routine body into paragraph ranges using purely structural rules.

    Walks the statements in order and opens a paragraph at provider-forced breaks, after a docstring, after a nested def, around returns, at substantial nested blocks, and on the dedent out of one - the first matching rule wins. Only lines in `boundary_lines` may start a paragraph, and the body's first statement always does, so the returned ranges cover the whole body.

    Parameters:
    - `stmts`: Structural facts for each statement, in body order.
    - `has_doc`: Whether the routine has a docstring (enables the first-in-scope break).
    - `boundary_lines`: The only line numbers allowed to start a paragraph.
    - `body_end`: Last body line, used to close the final range.
    - `min_block_lines`: Minimum block size that forces breaks around a nested block.
    - `allow_after_def`: Permit the break after a nested def.
    - `allow_first_in_scope`: Permit the break separating a docstring from the first statement.

    Returns:
    - Inclusive `(start, end)` line ranges, one per paragraph, in body order.
    """

    # Only provider-approved boundary lines may start a paragraph.
    legal = set(boundary_lines)
    breaks: set = set()

    # Statements off a legal boundary can never open a paragraph; provider-forced breaks are honoured unconditionally.
    for i, s in enumerate(stmts):
        if s.start not in legal:
            continue
        prev = stmts[i - 1] if i > 0 else None
        if s.force_break:
            breaks.add(s.start)                      # provider-forced break (e.g. off a leading declaration block)

        # One rule ladder per statement - the first matching rule wins, so their order is deliberate.
        if s.first_in_scope:
            if has_doc and allow_first_in_scope:
                breaks.add(s.start)                  # blank separating a docstring from the first statement
        elif prev is not None and prev.is_def and allow_after_def:
            breaks.add(s.start)                      # blank after a nested def/method - it clearly ends a paragraph
        elif s.is_return and s.merge_anchor is not None:
            breaks.add(s.merge_anchor)               # a [stmt; return] suite is one paragraph, anchored at the stmt
        elif s.is_return and prev is not None and prev.depth == s.depth:
            breaks.add(s.start)                      # paragraph a trailing return off from the body above it
        elif s.opens_block >= min_block_lines:
            breaks.add(s.start)                      # a substantial block opens a new paragraph
        elif prev is not None and prev.depth > s.depth and (s.closed_block == 0 or s.closed_block >= min_block_lines):
            breaks.add(s.start)                      # resuming after a substantial nested block closed

    # The body's first statement always opens a paragraph, so the returned ranges cover the whole body.
    starts = sorted(breaks)
    if stmts and stmts[0].start in legal and (not starts or starts[0] != stmts[0].start):
        starts.insert(0, stmts[0].start)
    return [(st, (starts[j + 1] - 1) if j + 1 < len(starts) else body_end) for j, st in enumerate(starts)]


# ---------------------------- comment helpers ----------------------------


def _is_comment_line(line: str, style: CommentStyle) -> bool:
    """
    Report whether a line is a comment line under the given style.

    Both the line prefix and any block-comment delimiters are matched with surrounding whitespace stripped, so variants without the conventional trailing space still count; blank lines never do.

    Parameters:
    - `line`: The source line to test.
    - `style`: Comment delimiters for the file's language.

    Returns:
    - `True` if the line opens, continues, or closes a comment; `False` otherwise.
    """

    # Delimiters are compared stripped, so a marker without its usual trailing space still matches.
    s = line.lstrip()
    if not s:
        return False
    if s.startswith(style.line_prefix.strip()):
        return True

    # Continuation and closing lines of a block comment count as comment lines too.
    for delim in (style.block_open, style.block_cont, style.block_close):
        if delim and s.startswith(delim.strip()):
            return True

    return False


def render_comment_lines(text: str, indent: str, style: CommentStyle) -> List[str]:
    """
    Render comment text as indented source lines in the given style.

    Single-line text (or any style without a block form) is rendered as line comments; multi-line text is wrapped in the block delimiters, using the line prefix for continuation lines when the style defines no continuation marker. Every line is right-stripped so blank comment lines carry no trailing whitespace.

    Parameters:
    - `text`: The comment text, with embedded newlines for multiple lines.
    - `indent`: Indentation prepended to every emitted line.
    - `style`: Comment delimiters for the file's language.

    Returns:
    - The rendered comment as a list of source lines.
    """

    # Single-line text stays in compact line-comment form; only genuinely multi-line text earns the block delimiters.
    lines = text.split("\n")
    if style.block_open is None or len(lines) == 1:
        return [(f"{indent}{style.line_prefix}{ln}").rstrip() for ln in lines]
    out = [f"{indent}{style.block_open}"]
    cont = style.block_cont if style.block_cont is not None else style.line_prefix
    out.extend((f"{indent}{cont}{ln}").rstrip() for ln in lines)
    out.append(f"{indent}{style.block_close}")

    return out


def _parse_comment_reply(reply: str, style: CommentStyle) -> Optional[str]:
    """
    Extract the usable comment text from a raw model reply.

    The reply may arrive wrapped in a code fence and/or already dressed in the target language's comment delimiters; both are stripped so the caller holds bare prose. An explicit `NONE` - detected both before and after cleaning - means the model declined to comment.

    Parameters:
    - `reply`: The raw model reply text.
    - `style`: The `CommentStyle` whose delimiters are stripped from each line.

    Returns:
    - The cleaned comment text, or `None` if the reply was empty or an explicit `NONE`.
    """

    # Unwrap any code fence and honour an explicit NONE refusal before any delimiter stripping.
    text = reply.strip()
    if not text:
        return None
    fence = re.match(r"^```[^\n]*\n(.*)\n```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if text.strip("`\"'* ").upper() == "NONE":
        return None
    delims = [d for d in (style.line_prefix.strip(), style.block_open, style.block_close) if d]
    cleaned: List[str] = []

    # Clean each reply line of whatever comment dressing the model added.
    for ln in text.split("\n"):
        s = ln.strip()

        # Shed at most one leading delimiter per line, so a marker inside the prose itself survives.
        for d in delims:
            if s.startswith(d):
                s = s[len(d):].strip()
                break

        # A bare block-continuation marker is dressing, not content - keep the line but blank it.
        if style.block_cont and s == style.block_cont.strip():
            s = ""
        cleaned.append(s)

    # Trim blank edges and re-check: stripping may have exposed an empty or NONE-only reply.
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
    Report whether a model reply is an explicit `NONE` refusal.

    Only the first line is consulted, after unwrapping any surrounding code fence and stripping quote and markdown decoration, so a refusal followed by explanatory chatter still registers.

    Parameters:
    - `reply`: The raw model reply text.

    Returns:
    - `True` if the reply leads with `NONE`, otherwise `False`.
    """

    # Only the first line decides, after unwrapping any code fence and shedding quote/markdown dressing.
    text = reply.strip()
    fence = re.match(r"^```[^\n]*\n(.*)\n```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    head = text.split("\n", 1)[0].strip().strip("`\"'*.: ")

    # Prefix match, so a refusal followed by justification still counts.
    return head.upper().startswith("NONE")


# ---------------------------- numbered view (boundary pass) ----------------------------


def render_numbered_body(source_lines: Chunk, target: BlockTarget) -> str:
    """
    Render a routine's body as a numbered view for the segmentation prompt.

    Only the lines in `target.boundary_lines` carry numbers - they are the sole legal segment starts - while the header and the remaining body lines appear unnumbered as context, with long unnumbered runs collapsed into an elision band.

    Parameters:
    - `source_lines`: The pristine source lines of the whole file.
    - `target`: The `BlockTarget` whose header and body ranges are rendered.

    Returns:
    - The rendered view as a single newline-joined string.
    """

    # Only boundary lines will carry numbers; size the gutter once so the pipe column stays aligned.
    boundary = set(target.boundary_lines)
    gutter = max(4, len(str(target.body_end)))
    def numbered(n: int, text: str) -> str:
        """
        Render one source line with its right-aligned line number in the gutter.
        """

        return f"{str(n).rjust(gutter)}| {text}"

    def unnumbered(text: str) -> str:
        """
        Render one source line with a blank gutter, marking it as context only.
        """

        return f"{' ' * gutter}| {text}"

    out: List[str] = []

    # The header is context only - rendered unnumbered so it can never be claimed as a segment start.
    for ln in range(target.header_start, target.header_end + 1):
        if 1 <= ln <= len(source_lines):
            out.append(unnumbered(source_lines[ln - 1]))

    run: List[str] = []

    # Long unnumbered stretches collapse into an elision band to keep the view within prompt budget.
    def flush_run() -> None:
        """
        Flush the buffered run of context lines into the output.

        Runs longer than `MAX_CONTEXT_RUN` are replaced with a single elision band so the rendered view stays compact; the buffer is cleared either way.
        """

        if not run:
            return

        # Runs past `MAX_CONTEXT_RUN` become a single elision band rather than swelling the view.
        if len(run) > MAX_CONTEXT_RUN:
            out.append(unnumbered(ELISION_BAND.format(n=len(run))))
        else:
            out.extend(unnumbered(t) for t in run)

        run.clear()

    for ln in range(target.body_start, target.body_end + 1):
        if not (1 <= ln <= len(source_lines)):
            continue
        text = source_lines[ln - 1]

        # Boundary lines flush the pending run and appear numbered; everything between accumulates for possible elision.
        if ln in boundary:
            flush_run()
            out.append(numbered(ln, text))
        else:
            run.append(text)

    # Catch a run that reaches the end of the body.
    flush_run()

    return "\n".join(out)


def _parse_segments(reply: str, boundary_lines: Tuple[int, ...], body_end: int) -> List[Tuple[int, int]]:
    """
    Parse the model's segmentation reply into validated `(start, end)` line ranges.

    Explicit `start-end` pairs are preferred; failing that, every bare number naming a legal boundary becomes a segment start running to the body end. Starts outside `boundary_lines` are discarded, duplicate starts keep their first occurrence, and ends are clamped so segments never overlap the next segment or overrun the body.

    Parameters:
    - `reply`: The raw model reply listing the segments.
    - `boundary_lines`: The line numbers that are legal segment starts.
    - `body_end`: The last body line, used to cap every range.

    Returns:
    - The cleaned segments as a list of `(start, end)` tuples, ordered by start.
    """

    # Tolerate hyphen, en dash or em dash between range endpoints - models vary.
    allowed = set(boundary_lines)
    pairs = re.findall(r"(\d+)\s*[-–—]\s*(\d+)", reply)

    # Prefer explicit ranges; with none, each bare boundary number becomes a start running to the body end.
    if pairs:
        candidates = [(int(a), int(b)) for a, b in pairs if int(a) in allowed]
    else:
        starts = sorted({int(tok) for tok in re.findall(r"\d+", reply)} & allowed)
        candidates = [(s, body_end) for s in starts]

    seen: set = set()
    ordered: List[List[int]] = []

    # Sort by start and keep only the first claim on each start line.
    for a, b in sorted(candidates, key=lambda p: p[0]):
        if a in seen:
            continue
        seen.add(a)
        ordered.append([a, b])

    # Clamp each end so segments stay non-empty, never reach the next start, and never overrun the body.
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
    Ask the model to split a routine's body into commentable segments.

    The body is rendered as a numbered view, cropped to fit the context budget, then sent as a single low-temperature turn that is removed from `messages` afterwards; the reply is parsed into validated line ranges.

    Parameters:
    - `llm`: The local chat model used for the segmentation turn.
    - `cfg`: The base generation configuration; temperature and reply length are tightened for this turn.
    - `messages`: The shared conversation, restored to its prior state before returning.
    - `source_lines`: The pristine source lines of the whole file.
    - `target`: The `BlockTarget` describing the routine to segment.
    - `prompt_template`: Optional override for the default segmentation prompt.

    Returns:
    - The parsed segments as a list of `(start, end)` line tuples.
    """

    # One throwaway low-temperature turn: the prompt is popped straight after the reply so the shared context stays bounded.
    view = render_numbered_body(source_lines, target)
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
    Return the first non-blank line of a docstring, stripped, or a placeholder when there is none.

    Parameters:
    - `doc`: The docstring text to summarise; may be empty.

    Returns:
    - The first non-blank line, or `(no description)` when every line is blank.
    """

    for line in doc.splitlines():
        if line.strip():
            return line.strip()

    return "(no description)"


def _parse_summary(reply: str, style: CommentStyle) -> str:
    """
    Reduce a raw model reply to a single clean comment line.

    Tolerates common model misbehaviour: a wrapping code fence is unwrapped, an echoed comment delimiter is stripped, stray flags and quoting are scrubbed, and the usual refusal words collapse to the empty string.

    Parameters:
    - `reply`: The raw model reply; may be `None` or empty.
    - `style`: The comment style whose delimiters the model may have echoed.

    Returns:
    - The cleaned one-line summary, or an empty string when the model declined or produced nothing usable.
    """

    # Models often wrap the reply in a code fence despite instructions; unwrap it and keep only the first non-blank line.
    text = (reply or "").strip()
    fence = re.match(r"^```[^\n]*\n(.*)\n```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    first = next((ln.strip() for ln in text.split("\n") if ln.strip()), "")
    if not first:
        return ""

    # Strip an echoed comment delimiter so the style's own prefix is never doubled when the comment is inserted.
    for d in (style.line_prefix.strip(), style.block_open, style.block_close):
        if d and first.startswith(d):
            first = first[len(d):].strip()
            break

    # Treat the model's various ways of saying 'nothing worth noting' as a deliberate empty answer.
    first = first.replace(VALUE_FLAG, "").replace(CHALLENGE_FLAG, "").strip("`\"'* ").strip()
    if first.upper() in ("", "NONE", "SKIP", "N/A", "TRIVIAL"):
        return ""
    return first


def _parse_score(reply: str, default: int = COMMENT_VALUE_THRESHOLD) -> int:
    """
    Extract a 1-3 value score from a model reply.

    Parameters:
    - `reply`: The raw scoring reply; only its first digit is read.
    - `default`: The score assumed when the reply contains no digit at all.

    Returns:
    - The clamped score in the range 1 to 3, or `default` for a digit-free reply.
    """

    # The first digit anywhere counts; a digit-free reply defaults to the keep threshold rather than silently dropping the comment.
    m = re.search(r"[0-9]", reply or "")
    return min(3, max(1, int(m.group()))) if m else default


def _comment_to_insert(comment: Optional[str]) -> Optional[str]:
    """
    Decide whether a block-pass result should actually be inserted into the source.

    Parameters:
    - `comment`: The comment text, possibly carrying a trailing rejection flag, or `None`.

    Returns:
    - The comment unchanged, or `None` when it is absent or flagged as rejected.
    """

    # A trailing flag marks a comment that was generated but rejected (low value or failed challenge): it is kept for logging only, never inserted.
    if comment and comment.rstrip().endswith((VALUE_FLAG, CHALLENGE_FLAG)):
        return None
    return comment


def _strip_note_flags(comment: Optional[str]) -> str:
    """
    Remove the internal rejection flags from a comment so it can be shown to a reader.

    Parameters:
    - `comment`: The comment text, possibly flagged, or `None`.

    Returns:
    - The flag-free text, stripped of surrounding whitespace; empty when `comment` is `None`.
    """

    out = comment or ""
    for flag in (VALUE_FLAG, CHALLENGE_FLAG):
        out = out.replace(flag, "")
    return out.strip()


def _context_window(source_lines: Chunk, target: BlockTarget, blob_start: int, blob_end: int) -> Tuple[int, int]:
    """
    Choose the range of source lines shown around a paragraph in the block-pass prompt.

    The window is a fixed number of lines either side of the paragraph, clamped to the routine's bounds, then widened backwards (within a cap) to reach the statement that opens the paragraph's enclosing scope, so the model can see which branch or loop the paragraph sits in.

    Parameters:
    - `source_lines`: The pristine source, one entry per line.
    - `target`: The routine being commented; supplies the clamping bounds.
    - `blob_start`: First line of the paragraph (1-based).
    - `blob_end`: Last line of the paragraph (1-based).

    Returns:
    - A `(lo, hi)` pair of 1-based inclusive line numbers to display.
    """

    # Start from a fixed-size window clamped to the routine; the paragraph's own indent anchors the scope search below.
    lo = max(target.header_start, blob_start - BLOCK_CONTEXT_LINES)
    hi = min(target.body_end, blob_end + BLOCK_CONTEXT_LINES)
    first = source_lines[blob_start - 1] if 1 <= blob_start <= len(source_lines) else ""
    para_indent = first[:len(first) - len(first.lstrip())]
    floor = max(target.header_start, blob_start - BLOCK_SCOPE_NUDGE_CAP)

    # Scan upwards, within a capped distance, for the nearest non-blank line indented less than the paragraph: the statement that opened its enclosing scope.
    for ln in range(blob_start - 1, floor - 1, -1):
        if not (1 <= ln <= len(source_lines)):
            break
        text = source_lines[ln - 1]
        if not text.strip():
            continue
        indent = text[:len(text) - len(text.lstrip())]

        # Widen the window back to take in the scope opener, so the prompt shows which branch or loop the paragraph belongs to.
        if len(indent) < len(para_indent):
            lo = min(lo, ln)        # never forward: the ±N window already reaches at least this far back
            break

    return lo, hi


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
    score_template: Optional[str] = None,
    value_threshold: int = COMMENT_VALUE_THRESHOLD,
    line_notes: Optional[Dict[int, str]] = None,
    verifier=None,
    feedback: Optional[str] = None,
) -> Optional[str]:
    """
    Generate, verify and score one block comment for a paragraph of a routine's body.

    Runs the two-turn comment-then-score exchange in the shared conversation, with up to one nudge on an empty reply, one grounding correction, and one regenerate-and-rescore retry when the verifier's obviousness challenge fails. Every message appended here is popped before returning, so the caller's conversation is left exactly as it arrived.

    Parameters:
    - `llm`: The local chat model used for every turn.
    - `cfg`: Base generation settings; temperature and reply budget are tightened per turn.
    - `messages`: The shared conversation; restored to its incoming state on every exit path.
    - `source_lines`: The pristine source, one entry per line.
    - `target`: The routine that owns the paragraph.
    - `blob_start`: First line of the paragraph (1-based).
    - `blob_end`: Last line of the paragraph (1-based).
    - `style`: The comment style, used to clean delimiters from replies.
    - `prior_comments`: Comments already written for this routine, shown to discourage repetition.
    - `length_note`: Scoring guidance keyed to the routine's length.
    - `prompt_template`: Optional override for the built-in comment prompt.
    - `nudge_template`: Optional override for the empty-reply nudge prompt.
    - `score_template`: Optional override for the value-scoring prompt.
    - `value_threshold`: Minimum score (1-3) a comment must reach to be kept.
    - `line_notes`: Optional per-line callee notes appended to the paragraph's lines in the prompt view.
    - `verifier`: Optional grounding/obviousness verifier; `None` disables both checks.
    - `feedback`: Optional extra instruction appended to the first prompt.

    Returns:
    - The comment text when kept; the text with a trailing internal flag when generated but dropped (low value or failed challenge); or `None` when no usable reply was produced.
    """

    # Build the source view the model will see: the scope-aware context window around the paragraph.
    lo, hi = _context_window(source_lines, target, blob_start, blob_end)
    view_lines: List[str] = []

    for ln in range(lo, hi + 1):
        if not (1 <= ln <= len(source_lines)):
            continue
        text = source_lines[ln - 1]

        # Paragraph lines are marked with '>' to set them apart from mere context; callee one-liners ride along as trailing notes.
        if blob_start <= ln <= blob_end:
            note = line_notes.get(ln) if line_notes else None
            if note and text.strip():
                text = f"{text.rstrip()}  {style.line_prefix.rstrip()} {note}"
            view_lines.append(f"> {text}")
        else:
            view_lines.append(f"  {text}")

    # First comment turn at a low temperature; `appended` counts every message pushed so the shared context can be unwound on every exit path.
    block_text = "\n".join(view_lines)
    priors = "\n".join(f"- {c}" for c in (prior_comments or [])) or "(none yet)"
    prompt = _fill(
        prompt_template or COMMENT_PROMPT,
        kind=target.kind, qualname=target.qualname, doc=_doc_summary(target.doc),
        priors=priors, block=block_text,
    )
    if feedback:
        prompt += "\n\n" + feedback
    turn_cfg = replace(cfg, temperature=COMMENT_TEMPERATURE, max_new_tokens=min(cfg.max_new_tokens, COMMENT_REPLY_TOKENS))
    appended = 0
    messages.append({"role": "user", "content": prompt})
    appended += 1
    summary = _parse_summary(llm.generate(messages, cfg=turn_cfg), style)

    # An empty or refused first reply earns exactly one nudge before giving up.
    if not summary:
        messages.append({"role": "assistant", "content": "(no answer)"})
        messages.append({"role": "user", "content": nudge_template or COMMENT_NUDGE})
        appended += 2
        summary = _parse_summary(llm.generate(messages, cfg=turn_cfg), style)

    # Still nothing after the nudge: restore the context and report no comment.
    if not summary:
        for _ in range(appended):
            messages.pop()
        return None

    # Grounding gate: every backticked identifier in the summary must actually appear in the code.
    if verifier is not None:
        tokens = verifier.ungrounded(summary)

        # Ungrounded identifiers get one corrective turn naming the offending tokens.
        if tokens:
            echo(f"[verify] '{target.qualname}' L{blob_start}-{blob_end}: ungrounded identifiers {tokens}; nudging")
            messages.append({"role": "assistant", "content": summary})
            messages.append({"role": "user", "content": verifier.gate_feedback(tokens)})
            appended += 2
            new = _parse_summary(llm.generate(messages, cfg=turn_cfg), style)
            if new:
                summary = new

            # A second grounding failure is final: unwind the conversation and drop the comment.
            if verifier.ungrounded(summary):
                for _ in range(appended):
                    messages.pop()
                echo(f"[verify] '{target.qualname}' L{blob_start}-{blob_end}: still ungrounded; comment dropped")

                # The flag suffix keeps the text visible in logs while marking it as never-to-insert.
                return f"{summary} {CHALLENGE_FLAG}"

    # Score turn: with the summary now on record, the model rates its value in the same context at a colder temperature.
    messages.append({"role": "assistant", "content": summary})
    score_prompt = _fill(score_template or SCORE_PROMPT, kind=target.kind, qualname=target.qualname,
                         length_note=length_note, block=block_text)
    messages.append({"role": "user", "content": score_prompt})
    appended += 2
    score_cfg = replace(cfg, temperature=SCORE_TEMPERATURE, max_new_tokens=min(cfg.max_new_tokens, SCORE_REPLY_TOKENS))
    score = _parse_score(llm.generate(messages, cfg=score_cfg))
    kept = score >= value_threshold
    challenge_failed = False

    # Only comments that passed the score face the obviousness challenge, judged against the pristine, unannotated paragraph.
    if kept and verifier is not None:
        pristine = "\n".join(source_lines[blob_start - 1:blob_end])

        # One retry on a failed challenge: retract the score turn, regenerate with the obviousness feedback, then rescore from scratch.
        if not verifier.challenge_obvious(pristine, summary):
            echo(f"[verify] '{target.qualname}' L{blob_start}-{blob_end}: obviousness challenge failed; regenerating")
            messages.pop()                                   # retract the score turn; the note turn is still open
            appended -= 1
            messages.append({"role": "user", "content": verifier.obvious_feedback_for()})
            appended += 1
            new = _parse_summary(llm.generate(messages, cfg=turn_cfg), style)
            if new:
                summary = new
            messages.append({"role": "assistant", "content": summary})
            messages.append({"role": "user", "content": score_prompt})
            appended += 2
            score = _parse_score(llm.generate(messages, cfg=score_cfg))
            kept = score >= value_threshold

            # Failing the challenge twice is final; the comment is dropped rather than retried again.
            if kept and not verifier.challenge_obvious(pristine, summary):
                echo(f"[verify] '{target.qualname}' L{blob_start}-{blob_end}: failed twice; comment dropped")
                challenge_failed = True

    # Unwind every turn this call added: the shared context must leave exactly as it arrived.
    for _ in range(appended):
        messages.pop()
    if challenge_failed:
        return f"{summary} {CHALLENGE_FLAG}"
    echo(f"[block] {target.qualname} L{blob_start}-{blob_end} score={score}/{value_threshold} "
         f"{'KEEP' if kept else 'drop'}: {summary}")

    # Low-value comments are returned flagged rather than as `None`, so callers can still log what was rejected.
    return summary if kept else f"{summary} {VALUE_FLAG}"


# ---------------------------- insertion patcher ----------------------------


def _code_signature(lines: Chunk, style: CommentStyle) -> List[str]:
    """
    Reduce a chunk to its executable lines for preservation comparison.

    Parameters:
    - `lines`: The source lines to filter.
    - `style`: The comment style used to recognise comment lines.

    Returns:
    - The non-blank, non-comment lines, in their original order.
    """

    # Ignoring blanks and comments means the preservation check sees only executable text, so comment insertion never trips it.
    return [ln for ln in lines if ln.strip() and not _is_comment_line(ln, style)]


def code_preserved(old_lines: Chunk, new_lines: Chunk, style: CommentStyle) -> bool:
    """
    Check that an edited line list differs from the original only in comments and blank lines.

    This is the block-pass preservation guard: both sides are reduced to a comment-stripped code signature, so any change to executable text fails the comparison.

    Parameters:
    - `old_lines`: The original source lines.
    - `new_lines`: The candidate (edited) source lines.
    - `style`: The comment style used to recognise and strip comment lines.

    Returns:
    - `True` if the executable code is preserved byte-for-byte, otherwise `False`.
    """

    return _code_signature(old_lines, style) == _code_signature(new_lines, style)


def _existing_comment_start(out_lines: Chunk, stmt_idx: int, indent: str, style: CommentStyle) -> int:
    """
    Find where the comment run attached to a statement begins.

    Scans upward from the statement over contiguous comment lines, claiming only those at the statement's exact indentation; a run at a different indent is treated as belonging elsewhere and left alone.

    Parameters:
    - `out_lines`: The working list of source lines.
    - `stmt_idx`: The 0-based index of the statement line.
    - `indent`: The statement's leading whitespace; claimed comment lines must match it exactly.
    - `style`: The comment style used to recognise comment lines.

    Returns:
    - The 0-based index of the first line of the attached comment run, or `stmt_idx` itself when there is none.
    """

    # `cs` staying at `stmt_idx` is the no-comment-found result.
    cs = stmt_idx
    i = stmt_idx - 1

    # Walk upward over contiguous comment lines, claiming only those at the statement's own indent - anything else belongs to an enclosing scope.
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
    Splice block comments and paragraph spacing into a copy of the source lines.

    Edits are applied bottom-up so each boundary's line number stays valid throughout. A non-empty comment replaces any comment run already attached to the boundary statement; a `None` comment keeps an existing run as-is. A blank separator line is inserted above the blob unless the previous line is blank or opens a suite.

    Parameters:
    - `source_lines`: The pristine source lines; never mutated.
    - `edits`: `(boundary, comment, indent)` triples, where `boundary` is the 1-based line number of the statement starting a blob.
    - `style`: The comment style used to render and recognise comment lines.

    Returns:
    - A new line list with all edits applied.
    """

    out = source_lines[:]

    # Apply edits bottom-up so earlier boundaries' line numbers stay valid as lines are inserted or removed.
    for boundary, comment, indent in sorted(edits, key=lambda e: e[0], reverse=True):
        stmt_idx = boundary - 1
        if not (0 <= stmt_idx < len(out)):
            continue
        cs = _existing_comment_start(out, stmt_idx, indent, style)
        has_existing = cs < stmt_idx

        # A fresh comment replaces any run already attached to the statement; with none supplied, an existing run is kept verbatim.
        if comment:
            body = render_comment_lines(comment, indent, style)
            start = cs
        elif has_existing:
            body = out[cs:stmt_idx]
            start = cs
        else:
            body = []
            start = stmt_idx

        # Insert a separating blank above the blob, but never directly under a line that opens a suite.
        prev_line = out[start - 1] if start - 1 >= 0 else ""
        opens_suite = prev_line.rstrip().endswith(("{", ":"))
        need_blank = start - 1 >= 0 and prev_line.strip() != "" and not opens_suite
        out[start:stmt_idx] = ([""] if need_blank else []) + body

    return out


def apply_blocks(
    source_lines: Chunk,
    target: BlockTarget,
    blocks: List[Tuple[int, Optional[str]]],
    style: CommentStyle,
) -> Chunk:
    """
    Apply a routine's ready-made block comments to the source, guarded against code changes.

    This is the manifest-apply counterpart to `annotate_blocks`: the comments arrive pre-decided, so the work is just building the edits, trial-applying them, and keeping the result only if the preservation guard passes.

    Parameters:
    - `source_lines`: The pristine source lines.
    - `target`: The block target the comments belong to; supplies per-boundary indentation and the name used in diagnostics.
    - `blocks`: `(boundary, comment)` pairs; a `None` comment inserts paragraph spacing only.
    - `style`: The comment style used to render comment lines.

    Returns:
    - The patched line list, or an untouched copy of the original if the edit would alter code.
    """

    # Pair each boundary with its recorded indentation and trial-apply the ready-made comments.
    edits = [(b, comment, target.indent_of.get(b, "")) for b, comment in blocks]
    candidate = _apply_edits(source_lines, edits, style)

    # Preservation guard: any patch that changes the comment-stripped code is rejected outright.
    if not code_preserved(source_lines, candidate, style):
        echo(f"[blocks] Skipped '{target.qualname}': edit would alter code; keeping original")
        return source_lines[:]

    return candidate


# ---------------------------- orchestration ----------------------------


def defer_block_targets(
    escalation,
    source_lines: Chunk,
    targets: List[BlockTarget],
    note_short: Optional[str] = None,
    note_long: Optional[str] = None,
    style: Optional[CommentStyle] = None,
    preserve_existing: bool = True,
) -> int:
    """
    Record every usable block target in the run manifest instead of annotating it locally.

    Each target's full span (header through body) is captured once as a self-contained snippet, with chunk line ranges rebased to 1-based positions within it. Each chunk also carries an `anchor` - the verbatim text of its boundary line - so the writer can locate the chunk by matching that line rather than counting through a body thick with comments and blanks (the off-by-N failure mode). Targets missing boundaries, segments, or a signature hash are skipped, as they cannot be re-bound at apply time.

    When a `style` is supplied, any comment already attached to a chunk's boundary is surfaced as the chunk's `existing` text so the writer can choose not to clobber it. A *substantive* (multi-line) existing comment is protected by default: its answer is pre-filled `NONE` (which keeps the comment verbatim at apply time) and flagged `preserve`, so a run over already-commented code never silently degrades hand-written rationale. `preserve_existing=False` (the `--overwrite-comments` path) lifts that protection, leaving every slot unfilled for the writer to decide.

    Parameters:
    - `escalation`: The run-manifest collector that receives each block record.
    - `source_lines`: The pristine source lines of the file.
    - `targets`: The block targets produced by the structural segmenter.
    - `note_short`: Optional override for the short-routine length note.
    - `note_long`: Optional override for the long-routine length note.
    - `style`: The comment style used to recognise existing comments; when `None`, existing comments are neither surfaced nor protected.
    - `preserve_existing`: When true (the default), pre-fill `NONE` for chunks whose boundary already carries a multi-line comment, protecting prior work.

    Returns:
    - The number of targets actually deferred.
    """

    count = 0

    # Targets missing boundaries, segments, or a signature hash cannot be re-bound at apply time, so they never enter the manifest.
    for target in sorted(targets, key=lambda t: t.body_start):
        if not target.boundary_lines or not target.segments or not target.sig:
            continue

        # The scoring note follows routine length: short routines get the stricter wording.
        if len(target.segments) <= SHORT_FUNCTION_CHUNKS:
            length_note = note_short or COMMENT_NOTE_SHORT
        else:
            length_note = note_long or COMMENT_NOTE_LONG

        # Capture the whole span once and rebase chunk ranges to 1-based snippet lines, so each record is self-contained.
        span = "\n".join(source_lines[target.header_start - 1:target.body_end])
        chunks = []
        for s, e in target.segments:
            chunk = {"bidx": target.boundary_lines.index(s),
                     "lines": [s - target.header_start + 1, e - target.header_start + 1],
                     "anchor": source_lines[s - 1].strip()}

            # Surface a comment already attached to this boundary, and protect a multi-line one by pre-answering NONE.
            if style is not None:
                indent = target.indent_of.get(s, "")
                cs = _existing_comment_start(source_lines, s - 1, indent, style)
                if cs < s - 1:
                    existing = source_lines[cs:s - 1]
                    chunk["existing"] = "\n".join(existing)
                    if preserve_existing and len(existing) >= 2:
                        chunk["answer"] = "NONE"
                        chunk["preserve"] = True
            chunks.append(chunk)

        escalation.record_block(
            qualname=target.qualname, kind=target.kind, sig_hash=target.sig,
            doc_summary=_doc_summary(target.doc), length_note=length_note, chunks=chunks, snippet=span,
        )
        count += 1

    return count


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
    score_prompt: Optional[str] = None,
    value_threshold: Optional[int] = None,
    callee_annotations: Optional[Dict[str, Dict[int, str]]] = None,
    verifier=None,
) -> Chunk:
    """
    Run the block pass: generate, verify, and splice paragraph comments into every target.

    Targets are processed deepest-first. Each one's segments are taken as precomputed or requested from the model, every blob gets a comment-and-score exchange, and for longer routines the assembled notes face a story challenge with one feedback-driven regeneration before the routine's comments are dropped. Each routine's edits must pass the code-preservation guard in isolation; survivors are applied to the pristine source in a single final pass. A `value_threshold` above 3 disables comment generation entirely, leaving paragraph spacing only.

    Parameters:
    - `llm`: The local chat model used for segmenting, commenting, and scoring.
    - `cfg`: Generation settings for the model calls.
    - `messages`: The primed chat history; generation turns are appended then popped per reply.
    - `source_lines`: The pristine source lines of the file.
    - `targets`: The block targets to annotate.
    - `style`: The comment style used to render and recognise comments.
    - `segment_prompt`: Optional override for the segmentation prompt.
    - `comment_prompt`: Optional override for the comment prompt.
    - `comment_nudge`: Optional override for the comment nudge template.
    - `note_short`: Optional override for the short-routine length note.
    - `note_long`: Optional override for the long-routine length note.
    - `score_prompt`: Optional override for the scoring prompt.
    - `value_threshold`: Minimum 1-3 score a comment must reach to be kept; above 3 switches commenting off.
    - `callee_annotations`: Optional per-routine map of line numbers to call-site notes fed into the comment turns.
    - `verifier`: Optional verifier supplying the story-challenge turns.

    Returns:
    - A new line list with all surviving comments and paragraph spacing applied.
    """

    # Deepest-nested targets go first; a threshold above 3 is unreachable by any score, so the comment turns are skipped and only spacing remains.
    ordered = sorted(targets, key=lambda t: (t.depth, t.body_start, -t.body_end), reverse=True)
    threshold = value_threshold if value_threshold is not None else COMMENT_VALUE_THRESHOLD
    comments_off = threshold > 3  # no 1-3 score can clear it, so skip the comment turns entirely (paragraph only)
    all_edits: List[Tuple[int, Optional[str], str]] = []

    for target in ordered:
        if not target.boundary_lines:
            continue

        # Prefer the segmenter's precomputed paragraphs; only fall back to asking the model when none were supplied.
        if target.segments is not None:
            segments = target.segments
        else:
            segments = request_segments(llm, cfg, messages, source_lines, target, segment_prompt)

        if not segments:
            continue

        # The length note steers scoring strictness: short routines tolerate fewer navigational comments.
        if len(segments) <= SHORT_FUNCTION_CHUNKS:
            length_note = note_short or COMMENT_NOTE_SHORT
        else:
            length_note = note_long or COMMENT_NOTE_LONG

        # Call-graph read-side annotations for this routine, if any were gathered.
        line_notes = (callee_annotations or {}).get(target.qualname)

        # Closure so a failed story challenge can regenerate every chunk with the verifier's feedback attached.
        def run_chunks(feedback: Optional[str] = None) -> Tuple[List[Tuple[int, Optional[str], str]], List[str]]:
            """
            Generate one comment edit per segment of the current target.

            Accepted comments are fed back as context for later blobs, and an edit is recorded for every blob even when no comment is written, so paragraph spacing is always applied.

            Parameters:
            - `feedback`: Optional verifier feedback from a failed story challenge, passed to each comment turn on regeneration.

            Returns:
            - An `(edits, prior_comments)` tuple: the per-blob `(boundary, comment, indent)` edits and the raw notes produced.
            """

            edits: List[Tuple[int, Optional[str], str]] = []
            prior_comments: List[str] = []

            # Each kept comment is fed back as context so later blobs do not repeat it; with comments off the model is never consulted.
            for blob_start, blob_end in segments:
                if comments_off:
                    comment = None               # threshold > 3: paragraph only, no model work
                else:
                    comment = request_block_comment(
                        llm, cfg, messages, source_lines, target, blob_start, blob_end, style,
                        prior_comments=prior_comments, length_note=length_note,
                        prompt_template=comment_prompt, nudge_template=comment_nudge, score_template=score_prompt,
                        value_threshold=threshold, line_notes=line_notes, verifier=verifier, feedback=feedback,
                    )
                    if comment:
                        prior_comments.append(comment)  # keep every summary (incl. flagged ones) as context

                # An edit is recorded for every blob, comment or not, so paragraph spacing is always applied.
                edits.append((blob_start, _comment_to_insert(comment), target.indent_of.get(blob_start, "")))

            return edits, prior_comments

        edits, notes = run_chunks()
        story_failed = False

        # The story challenge runs only where it can pay off: a verifier present, comments actually written, and a routine long enough to have a narrative.
        if (verifier is not None and not comments_off and len(segments) > SHORT_FUNCTION_CHUNKS
                and any(_comment_to_insert(n) for n in notes)):
            signature = "\n".join(source_lines[target.header_start - 1:target.header_end])

            # On a failed challenge, regenerate once with the verifier's feedback rather than dropping straight away.
            if not verifier.challenge_story(signature, _doc_summary(target.doc), [_strip_note_flags(n) for n in notes]):
                echo(f"[verify] '{target.qualname}': story challenge failed; regenerating the routine's notes once")
                edits, notes = run_chunks(verifier.story_feedback_for())

                # Still failing after the retry marks the routine for dropping - there is never a third attempt.
                if (any(_comment_to_insert(n) for n in notes)
                        and not verifier.challenge_story(signature, _doc_summary(target.doc),
                                                         [_strip_note_flags(n) for n in notes])):
                    story_failed = True

        # Dropping keeps the paragraph spacing: `None` comments still insert blank separators, just no text.
        if story_failed:
            echo(f"[verify] '{target.qualname}': story challenge failed twice; dropping its comments")
            edits = [(b, None, indent) for b, _comment, indent in edits]

        trial = _apply_edits(source_lines, edits, style)

        # Per-routine guard: a bad patch costs only this routine's comments, never the whole file's.
        if code_preserved(source_lines, trial, style):
            all_edits.extend(edits)
        else:
            echo(f"[blocks] Skipped '{target.qualname}': edit would alter code; keeping original")

    # All surviving edits are spliced into the pristine source in one final pass.
    return _apply_edits(source_lines, all_edits, style)
