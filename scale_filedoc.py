#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The language-agnostic engine for SCALE's file-level header doccomment pass (`--file-doc`).

A source file's existing header often mixes a shebang, copyright/author boilerplate, license text, and a prose
description of what the file does. Only the description should be (re)written; everything else - especially the license -
must survive byte-for-byte. Telling those apart is genuine judgement, so the local model is used to *classify* which line
range is the description, but it is never trusted to re-emit the preserved text: a deterministic patcher slices every
kept line from the original source and only ever inserts or replaces comment lines. Two independent safety nets back the
classification up - a license-keyword veto (`looks_legal`) that refuses to overwrite anything legal-looking, and a
preservation guard (`file_doc_preserved`) that turns any edit which would touch code into a no-op.

The per-language adapter (e.g. `scale_c.file_doc_target_c`) supplies a `FileDocTarget` describing the leading-comment
zone - which may span several contiguous blocks - and the slots where a description can be replaced, appended, or freshly
inserted. The engine here orchestrates the two model turns (classify, then generate) and the single guarded splice.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple
import re
import textwrap

from scale_blocks import CommentStyle, _is_comment_line, _fill
from scale_log import echo

Chunk = List[str]


# ---- Tunables (wording lives in scale-cfg/filedoc.*.txt; these are the built-in fallbacks). ----

CLASSIFY_TEMPERATURE = 0.0
CLASSIFY_REPLY_TOKENS = 16
GENERATE_TEMPERATURE = 0.2
GENERATE_REPLY_TOKENS = 160

# Width the generated description is wrapped to (the per-line decoration prefix is subtracted from this).
WRAP_WIDTH = 118

CLASSIFY_PROMPT = (
    "Below are the comment lines that make up the existing header of a {language} source file, one per numbered "
    "entry. Identify the contiguous range of entries that form the human-readable DESCRIPTION of what the file does "
    "- as opposed to a shebang, copyright or author boilerplate, a filename, dates, or license/legal text.\n\n"
    "{entries}\n\n"
    "Reply with just the range as `START-END` (inclusive, using the numbers above), or a single number for one line, "
    "or the word NONE if there is no descriptive prose. Give only the range or NONE - no other words."
)

GENERATE_PROMPT = (
    "Write the file-level description for the top of this {language} source file: the short note that tells a "
    "developer opening it what the file is for. Use AT MOST THREE SENTENCES, one short paragraph.{existing}\n\n"
    "Lead with the file's role in the wider program, then name the main things it defines or the key operations it "
    "provides. Stay grounded in the overview above: do not invent APIs or behaviour. Do NOT pad with generic remarks "
    "about logging, debugging, tracing, error handling, or the headers it includes - leave those out unless they are "
    "the whole point of the file. Mention a caveat only if it is genuinely important to a reader.\n\n"
    "Write only the description prose: no comment markers (no /*, *, //, or \"\"\"), no heading, no file name, no "
    "copyright or license text, and no preamble. Give only the description."
)

# Deterministic legal-text veto. Substrings are matched case-insensitively against a candidate description line; any
# hit means the line is treated as legal boilerplate and is never offered up for rewriting, regardless of how the model
# classified it. Over-matching is the safe failure mode here (we simply decline to touch that line).
LICENSE_MARKERS: Tuple[str, ...] = (
    "spdx-license-identifier",
    "copyright",
    "(c)",
    "©",                       # ©
    "all rights reserved",
    "permission is hereby granted",
    "redistribution and use",
    "without warranty",
    "warranties or conditions",
    "licensed under",
    "general public license",
    "apache license",
    "mit license",
    "bsd license",
    "mozilla public license",
    "gnu ",
)


def looks_legal(text: str) -> bool:
    """
    Report whether a line of header text looks like legal/license boilerplate that must not be rewritten.

    A case-insensitive substring match against `LICENSE_MARKERS`. This is the deterministic safety net behind the
    model's classification: even if the model labels a legal line as "description", this veto keeps it untouched.

    Parameters:
    - `text`: The candidate header line (delimiters already stripped, or not - matching is substring-based).

    Returns:
    - True if the text contains any known license/legal marker.
    """

    low = text.lower()
    return any(marker in low for marker in LICENSE_MARKERS)


@dataclass
class FileDocTarget:
    """
    A language-neutral description of a file's leading-comment zone and where its description can be edited.

    The per-language adapter builds this; the engine consumes it. The leading zone may span several contiguous comment
    blocks (blank-separated, with no intervening code) - they are gathered together so the description can be found and
    updated wherever it sits.

    Fields:
    - `eligible`: The description-candidate comment lines, in order, as `(lineno_1b, prefix, inner)` tuples. `prefix` is
      the exact leading decoration of the raw line (indent + comment delimiter + spacing, e.g. `" * "` or `"// "`);
      `inner` is the remaining text. Only pure-content comment lines are eligible - block open/close delimiters,
      single-line `/* ... */` comments, and blank continuation lines are excluded (and thus always preserved).
    - `insert_index`: 0-based line index at which a fresh or appended description should be inserted.
    - `insert_prefix`: The per-line decoration prefix to use when appending into an existing block (ignored when
      `insert_fresh` is set).
    - `insert_fresh`: When True, render a brand-new comment block (delimiters and all) via `style` at `insert_index`;
      when False, append plain `insert_prefix`-decorated lines into an existing block.
    - `style`: The language `CommentStyle` used to render a fresh block.
    - `indent`: Leading whitespace for a freshly inserted block (usually empty at file scope).
    - `has_zone`: Whether the file already has a leading-comment zone at all.
    """

    eligible: List[Tuple[int, str, str]] = field(default_factory=list)
    insert_index: int = 0
    insert_prefix: str = ""
    insert_fresh: bool = True
    style: Optional[CommentStyle] = None
    indent: str = ""
    has_zone: bool = False


def _parse_classify_range(reply: str, n: int) -> Optional[Tuple[int, int]]:
    """
    Parse the classify turn's reply into a 1-based inclusive range over the `n` eligible entries.

    Accepts `START-END`, a single number, or NONE/empty. The range is clamped to `[1, n]` and rejected if inverted.

    Parameters:
    - `reply`: The model's raw reply.
    - `n`: The number of eligible entries shown to the model.

    Returns:
    - `(start, end)` 1-based inclusive, or None when there is no usable range.
    """

    if n <= 0:
        return None
    text = reply.strip().lower()
    if not text or text.startswith("none"):
        return None

    m = re.search(r"(\d+)\s*(?:-|to|–)\s*(\d+)", text)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"\d+", text)
        if not m:
            return None
        start = end = int(m.group(0))

    start = max(1, min(start, n))
    end = max(1, min(end, n))
    if end < start:
        return None
    return start, end


def _wrap(text: str, prefix: str) -> List[str]:
    """
    Wrap description prose to the configured width, leaving room for the per-line decoration `prefix`.

    Blank lines in the input separate paragraphs and are preserved (rendered as empty strings, which the caller
    decorates). Each non-blank paragraph is independently filled.

    Parameters:
    - `text`: The description prose (may contain blank-line-separated paragraphs).
    - `prefix`: The decoration that will be prepended to each output line (its length reduces the wrap width).

    Returns:
    - The wrapped lines (without the prefix applied).
    """

    width = max(40, WRAP_WIDTH - len(prefix))
    out: List[str] = []
    paragraphs = re.split(r"\n\s*\n", text.strip())
    for i, para in enumerate(paragraphs):
        if i:
            out.append("")                       # blank line between paragraphs
        collapsed = " ".join(para.split())
        if collapsed:
            out.extend(textwrap.wrap(collapsed, width=width))
    return out or [""]


def file_doc_preserved(old_lines: Chunk, new_lines: Chunk, start: int, removed: int, added: int,
                       style: CommentStyle) -> bool:
    """
    Verify a single-region file-doc splice changed only comment/blank lines and left everything else byte-for-byte.

    The splice replaced `old_lines[start:start + removed]` with `added` new lines. This guard confirms three things:
    the prefix before the region and the suffix after it are identical in both versions; every removed line was blank
    or a pure comment (so no code was deleted); and every inserted line is blank or a pure comment (so no code was
    introduced). Any failure means the edit must be abandoned.

    Parameters:
    - `old_lines`: The original lines.
    - `new_lines`: The candidate edited lines.
    - `start`: 0-based start index of the spliced region.
    - `removed`: Number of original lines removed at `start`.
    - `added`: Number of new lines inserted at `start`.
    - `style`: The language comment-style descriptor.

    Returns:
    - True if the splice is safe (only comment/blank lines touched, code preserved).
    """

    if old_lines[:start] != new_lines[:start]:
        return False
    if old_lines[start + removed:] != new_lines[start + added:]:
        return False
    region_old = old_lines[start:start + removed]
    region_new = new_lines[start:start + added]
    for ln in region_old + region_new:
        if ln.strip() and not _is_comment_line(ln, style):
            return False
    return True


def annotate_file_doc(
    llm,
    cfg,
    base_messages,
    source_lines: Chunk,
    target: FileDocTarget,
    summary: str,
    language: str,
    *,
    classify_prompt: Optional[str] = None,
    generate_prompt: Optional[str] = None,
) -> Chunk:
    """
    Run the file-doc pass: classify the existing description (if any), generate fresh prose, and splice it in safely.

    The model performs at most two turns - a deterministic classify turn (only when there are eligible header lines)
    and a generate turn - and never re-emits any preserved text. The result is applied as one guarded splice; if the
    guard rejects it, the original lines are returned unchanged.

    Parameters:
    - `llm`: A model exposing `generate`.
    - `cfg`: The base generation configuration (cloned per turn with a capped reply length).
    - `base_messages`: The primed context (system prompt + file overview + any style note). Not mutated.
    - `source_lines`: The source split into lines.
    - `target`: The `FileDocTarget` from the language adapter.
    - `summary`: The whole-file summary, used to ground the generated description.
    - `language`: The language identifier, woven into the prompts.
    - `classify_prompt`/`generate_prompt`: Optional prompt overrides (default to the built-in constants).

    Returns:
    - The annotated source split into lines (unchanged if there was nothing to do or the guard failed).
    """

    style = target.style
    if style is None:
        return source_lines

    classify_tmpl = classify_prompt or CLASSIFY_PROMPT
    generate_tmpl = generate_prompt or GENERATE_PROMPT

    # ---- Turn 1: classify which eligible lines are the description (skipped when none are eligible). ----
    desc_range: Optional[Tuple[int, int]] = None
    if target.eligible:
        entries = "\n".join(f"{i}. {inner}" for i, (_, _, inner) in enumerate(target.eligible, start=1))
        prompt = _fill(classify_tmpl, language=language, entries=entries)
        turn_cfg = replace(cfg, max_new_tokens=min(cfg.max_new_tokens, CLASSIFY_REPLY_TOKENS),
                           temperature=CLASSIFY_TEMPERATURE)
        messages = base_messages + [{"role": "user", "content": prompt}]
        reply = llm.generate(messages, cfg=turn_cfg)
        desc_range = _parse_classify_range(reply, len(target.eligible))

        # Safety veto: refuse to treat any legal-looking line as editable description.
        if desc_range is not None:
            lo, hi = desc_range
            if any(looks_legal(target.eligible[i - 1][2]) for i in range(lo, hi + 1)):
                echo("file-doc: classified range looks like license/legal text; leaving it untouched.")
                desc_range = None

    # The existing description text we are updating (if any), fed to the generate turn.
    existing_text = ""
    if desc_range is not None:
        lo, hi = desc_range
        existing_text = " ".join(target.eligible[i - 1][2] for i in range(lo, hi + 1)).strip()

    # ---- Turn 2: generate the description prose. ----
    existing_clause = (
        f" The current description reads: \"{existing_text}\" - refine and correct it where the code has moved on, "
        f"keeping any wording that is still accurate." if existing_text else ""
    )
    prompt = _fill(generate_tmpl, language=language, existing=existing_clause)
    turn_cfg = replace(cfg, max_new_tokens=min(cfg.max_new_tokens, GENERATE_REPLY_TOKENS),
                       temperature=GENERATE_TEMPERATURE)
    messages = base_messages + [
        {"role": "user", "content": f"Here is an overview of the whole file:\n\n{summary}"},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": prompt},
    ]
    description = llm.generate(messages, cfg=turn_cfg).strip()
    description = _sanitise_description(description)
    if not description:
        echo("file-doc: the model produced no usable description; leaving the file unchanged.")
        return source_lines

    # ---- Build the single splice. ----
    if desc_range is not None:
        # Replace the existing description lines in place, reusing the first line's decoration prefix.
        lo, hi = desc_range
        start_lineno = target.eligible[lo - 1][0]
        end_lineno = target.eligible[hi - 1][0]
        prefix = target.eligible[lo - 1][1]
        start = start_lineno - 1                 # 0-based
        removed = end_lineno - start_lineno + 1
        new_block = [f"{prefix}{ln}".rstrip() for ln in _wrap(description, prefix)]
    elif target.insert_fresh:
        # No header (or no usable description and no block to extend): insert a fresh comment block at the top. We
        # render the block form directly rather than via render_comment_lines, which would collapse a one-line
        # description to a single line comment - a file header wants the block delimiters even when the prose is short.
        start = target.insert_index
        removed = 0
        if style.block_open is not None:
            cont = style.block_cont or style.line_prefix
            body = _wrap(description, target.indent + cont)
            new_block = (
                [f"{target.indent}{style.block_open}"]
                + [f"{target.indent}{cont}{ln}".rstrip() for ln in body]
                + [f"{target.indent}{style.block_close}", ""]
            )
        else:
            body = _wrap(description, target.indent + style.line_prefix)
            new_block = [f"{target.indent}{style.line_prefix}{ln}".rstrip() for ln in body] + [""]
    else:
        # A zone exists but has no description: append one into the existing block, after a blank separator line.
        start = target.insert_index
        removed = 0
        prefix = target.insert_prefix
        new_block = [prefix.rstrip()] + [f"{prefix}{ln}".rstrip() for ln in _wrap(description, prefix)]

    out_lines = source_lines[:start] + new_block + source_lines[start + removed:]

    if not file_doc_preserved(source_lines, out_lines, start, removed, len(new_block), style):
        echo("file-doc: the edit would have altered code or preserved text; abandoning it.")
        return source_lines

    echo("file-doc: " + ("updated the existing file description."
                         if desc_range is not None else "inserted a file description."))
    return out_lines


def _sanitise_description(text: str) -> str:
    """
    Strip any stray comment delimiters or code fences the model may have wrapped the description in.

    The generate turn is asked for bare prose, but small models sometimes add `/* */`, leading `*`/`//`, or a ``` fence
    anyway. We remove those decorations so the patcher controls the rendering, and drop a leading bare file name if the
    model echoed one.

    Parameters:
    - `text`: The raw generate-turn reply.

    Returns:
    - The cleaned description prose (possibly empty).
    """

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    lines = text.split("\n")
    cleaned: List[str] = []
    for ln in lines:
        s = ln.strip()
        if s in ("/*", "*/", "/**"):
            continue
        s = re.sub(r"^/\*+\s?", "", s)
        s = re.sub(r"\s?\*+/$", "", s)
        s = re.sub(r"^(\*|//|#)\s?", "", s)
        cleaned.append(s)
    return "\n".join(cleaned).strip()
