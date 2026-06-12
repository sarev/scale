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
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import json
import re
import textwrap

from scale_blocks import CommentStyle, _is_comment_line, _fill
from scale_log import echo

Chunk = List[str]


# ---- Tunables (wording lives in scale-cfg/filedoc.*.txt; these are the built-in fallbacks). ----

CLASSIFY_TEMPERATURE = 0.0
CLASSIFY_REPLY_TOKENS = 16

# Width the description is wrapped to (the per-line decoration prefix is subtracted from this).
WRAP_WIDTH = 118

CLASSIFY_PROMPT = (
    "Below are the comment lines that make up the existing header of a {language} source file, one per numbered "
    "entry. Identify the contiguous range of entries that form the human-readable DESCRIPTION of what the file does "
    "- as opposed to a shebang, copyright or author boilerplate, a filename, dates, or license/legal text.\n\n"
    "{entries}\n\n"
    "Reply with just the range as `START-END` (inclusive, using the numbers above), or a single number for one line, "
    "or the word NONE if there is no descriptive prose. Give only the range or NONE - no other words."
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
    - `style`: The language `CommentStyle` used to render a fresh block. For Python this is `PYTHON_DOC_STYLE`, which
      renders a triple-quoted module docstring rather than a comment.
    - `indent`: Leading whitespace for a freshly inserted block (usually empty at file scope).
    - `has_zone`: Whether the file already has a leading-comment zone at all.
    - `preserved`: Optional language-specific preservation guard `(old, new, start, removed, added) -> bool`. When set,
      the engine uses it instead of the default line-comment `file_doc_preserved` (Python supplies a parse-based guard,
      since a module docstring's content lines are not recognisable as comments line-by-line).
    """

    eligible: List[Tuple[int, str, str]] = field(default_factory=list)
    insert_index: int = 0
    insert_prefix: str = ""
    insert_fresh: bool = True
    style: Optional[CommentStyle] = None
    indent: str = ""
    has_zone: bool = False
    preserved: Optional[Callable[[Chunk, Chunk, int, int, int], bool]] = None


# Python's "file header" is the module docstring (a string literal), not a comment block. Rendering it as a block-style
# comment with `"""` delimiters and no continuation prefix produces a standard triple-quoted module docstring. Its empty
# `line_prefix` is never fed to `_is_comment_line` because the Python adapter supplies its own (parse-based) guard.
PYTHON_DOC_STYLE = CommentStyle(line_prefix="", block_open='"""', block_cont="", block_close='"""')


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
            # Never split hyphenated words: the reword matcher relocates this text by whitespace-collapsed
            # comparison, and a "role-" / "based" break re-joins as "role- based", which no longer matches.
            out.extend(textwrap.wrap(collapsed, width=width, break_on_hyphens=False, break_long_words=False))
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


def splice_description(
    source_lines: Chunk,
    target: FileDocTarget,
    desc_range: Optional[Tuple[int, int]],
    description: str,
) -> Optional[Chunk]:
    """
    Splice a file description into the leading zone - the model-free half of the file-doc pass.

    Three splice shapes: replace the `desc_range` lines in place (re-wrapped in the first line's decoration prefix);
    insert a fresh comment block at the top (`desc_range` None and `insert_fresh`); or append into the existing block
    (`desc_range` None, a zone with no description). The deterministic safety nets both run here: the license veto
    refuses to overwrite any legal-looking line in the range, and the preservation guard (the target's own parse-based
    guard for Python, the comment-line `file_doc_preserved` otherwise) turns any edit that would touch code into a
    refusal. Both the offline pass and the online `--apply-filedoc` round go through this one function, so the guards
    hold regardless of which model wrote the prose.

    Parameters:
    - `source_lines`: The source split into lines (not mutated).
    - `target`: The `FileDocTarget` from the language adapter (scanned from these same lines).
    - `desc_range`: The 1-based inclusive range over `target.eligible` to replace, or None to insert/append.
    - `description`: The description prose (sanitised here; comment delimiters/fences are stripped).

    Returns:
    - The new lines, or None on refusal (no style, empty prose, legal veto, or guard failure).
    """

    style = target.style
    if style is None:
        return None
    description = _sanitise_description(description)
    if not description:
        return None

    # Safety veto: refuse to overwrite any legal-looking line, regardless of how the range was classified.
    if desc_range is not None:
        lo, hi = desc_range
        if any(looks_legal(target.eligible[i - 1][2]) for i in range(lo, hi + 1)):
            echo("file-doc: the description range looks like license/legal text; leaving it untouched.")
            return None

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

    # The guard is pluggable: brace languages use the line-comment `file_doc_preserved`; Python supplies a parse-based
    # guard (its docstring content lines are not line-recognisable as comments).
    if target.preserved is not None:
        ok = target.preserved(source_lines, out_lines, start, removed, len(new_block))
    else:
        ok = file_doc_preserved(source_lines, out_lines, start, removed, len(new_block), style)
    if not ok:
        echo("file-doc: the edit would have altered code or preserved text; abandoning it.")
        return None
    return out_lines


def annotate_file_doc(
    llm,
    cfg,
    base_messages,
    source_lines: Chunk,
    target: FileDocTarget,
    summary_provider: Callable[[Optional[str]], str],
    language: str,
    *,
    classify_prompt: Optional[str] = None,
    on_description: Optional[Callable[[str], None]] = None,
) -> Chunk:
    """
    Run the file-doc pass: classify the existing description (if any), then splice the file summary in as the header.

    The whole-file summary *is* the file description (see `scale.py`), so there is no separate generate turn - the
    model only does a deterministic classify turn (when there are eligible header lines) to find which lines are the
    existing description, then `summary_provider` returns the description prose (seeded with that existing text so the
    author's wording is incorporated). The model never re-emits any preserved text; the result is applied as one
    guarded splice, and if the guard rejects it the original lines are returned unchanged.

    Parameters:
    - `llm`: A model exposing `generate`.
    - `cfg`: The base generation configuration (cloned for the classify turn with a capped reply length).
    - `base_messages`: The primed context (system prompt). Not mutated.
    - `source_lines`: The source split into lines.
    - `target`: The `FileDocTarget` from the language adapter.
    - `summary_provider`: Callable taking the existing description text (or None) and returning the file-description
      prose to insert. This is where the whole-file summary is produced/fetched.
    - `language`: The language identifier, woven into the classify prompt.
    - `classify_prompt`: Optional classify-prompt override (defaults to the built-in constant).
    - `on_description`: Optional callback invoked with the (sanitised) description text after a SUCCESSFUL splice -
      i.e. the exact prose now sitting in the header. The header-reword manifest records it as each file's draft,
      which its model-free apply later re-finds by exact match.

    Returns:
    - The annotated source split into lines (unchanged if there was nothing to do or the guard failed).
    """

    style = target.style
    if style is None:
        return source_lines

    classify_tmpl = classify_prompt or CLASSIFY_PROMPT

    # ---- Classify which eligible lines are the existing description (skipped when none are eligible). ----
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

    # The existing description text we are updating (if any), fed to the summary provider as a seed.
    existing_text = ""
    if desc_range is not None:
        lo, hi = desc_range
        existing_text = " ".join(target.eligible[i - 1][2] for i in range(lo, hi + 1)).strip()

    # ---- Fetch the file-description prose (the whole-file summary, seeded with any existing description). ----
    description = _sanitise_description(summary_provider(existing_text or None))
    if not description:
        echo("file-doc: no usable description was produced; leaving the file unchanged.")
        return source_lines

    # ---- The model-free splice (three shapes + veto + guard live in splice_description). ----
    out_lines = splice_description(source_lines, target, desc_range, description)
    if out_lines is None:
        return source_lines

    echo("file-doc: " + ("updated the existing file description."
                         if desc_range is not None else "inserted a file description."))
    if on_description is not None:
        on_description(description)
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


# ---- The online file-description round (the scale-filedoc manifest) ----
#
# Online, the local model never runs, so there is no local draft to reword: the file description itself is deferred
# to the stronger model as a second, run-level manifest round after --apply-manifest. The answer is a coupled pair -
# WHICH header lines are the existing description (the classify decision) plus the replacement prose - so this is a
# manifest type of its own rather than a reword (whose apply locates a known local draft by exact match). Emit gives
# the stronger model each file's CURRENT skeleton (rich with the freshly applied docs) and the header zone's eligible
# lines; apply re-derives the zone, checks it is unchanged, and splices through the same veto + preservation guard as
# the offline pass.

FILEDOC_VERSION = 1
FILEDOC_TOOL = "scale-filedoc"

# Cap (characters) on the raw project-overview text carried in the manifest - context, not a payload.
FILEDOC_PROJECT_DOC_CAP = 12000


def filedoc_manifest(description_spec: str, project_doc: str, entries: List[dict]) -> dict:
    """
    Build the run's file-description manifest from the spec, the project overview, and the per-file entries.

    Parameters:
    - `description_spec`: The file-description instruction (scale-cfg/summary.txt or the built-in default) the
      stronger model writes each description to.
    - `project_doc`: The raw resolved project-overview text (capped by the caller), so descriptions share the big
      picture.
    - `entries`: One dict per target file:
      `{"path", "language", "role", "skeleton", "entries": [...], "answer": {"range": None, "description": None}}`,
      where `skeleton` is the file's current structural skeleton (or its whole text when it has no symbols) and
      `entries` are the header zone's eligible inner texts, numbered from 1 for the `range` answer.

    Returns:
    - The manifest dictionary ready to be written as JSON.
    """

    return {
        "version": FILEDOC_VERSION,
        "tool": FILEDOC_TOOL,
        "description_spec": description_spec,
        "project_doc": project_doc,
        "files": entries,
    }


def write_filedoc_manifest(path: Path, manifest: dict) -> None:
    """Write a file-description manifest to disk as indented JSON (UTF-8)."""

    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_filedoc_manifest(path: Path) -> dict:
    """
    Read and lightly validate a file-description manifest.

    Parameters:
    - `path`: The manifest file path.

    Returns:
    - The parsed manifest dictionary.

    Raises:
    - `ValueError`: If the file is not a filedoc manifest of a supported version.
    """

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("tool") != FILEDOC_TOOL
            or manifest.get("version") != FILEDOC_VERSION):
        raise ValueError(f"{path}: not a SCALE filedoc manifest of version {FILEDOC_VERSION}")
    manifest.setdefault("files", [])
    return manifest


def unfilled_descriptions(manifest: dict) -> List[str]:
    """
    Return the path of every file entry whose answer pair has not been filled in (the completeness counter).

    An answer needs BOTH halves: `range` ("START-END", a single number, or "NONE" for no existing description) and
    `description` (the prose, or the explicit "NONE" to decline the file - a deliberate decline counts as filled).
    A missing/blank half leaves the entry unfilled.

    Parameters:
    - `manifest`: The parsed filedoc manifest.

    Returns:
    - The unfilled entries' paths.
    """

    out: List[str] = []
    for f in manifest.get("files", []):
        answer = f.get("answer") or {}
        if not str(answer.get("range") or "").strip() or not str(answer.get("description") or "").strip():
            out.append(f.get("path", "?"))
    return out


def apply_filedoc_entry(source_lines: Chunk, target: FileDocTarget, entry: dict) -> Optional[Chunk]:
    """
    Apply one filedoc manifest entry to a file (the model-free apply phase of the online file-description round).

    The header zone is re-derived by the caller from the CURRENT text and compared against the entry's recorded
    `entries`: any mismatch means the file's header changed since emit, so the recorded `range` can no longer be
    trusted and the apply is a safe no-op. The answer's `range` is parsed with the same `_parse_classify_range` the
    offline classify turn uses, then `splice_description` runs the splice - the license veto and the preservation
    guard fire locally exactly as offline, so the stronger model's classification is never blindly trusted.

    Parameters:
    - `source_lines`: The file's current lines.
    - `target`: The file's `FileDocTarget`, freshly scanned from these same lines.
    - `entry`: The manifest file entry carrying `entries` and the filled `answer` pair.

    Returns:
    - The new lines, or None when nothing was (safely) applicable: a zone mismatch, a "NONE"/empty description, or a
      veto/guard refusal.
    """

    current = [inner for (_lineno, _prefix, inner) in target.eligible]
    recorded = [str(e) for e in (entry.get("entries") or [])]
    if current != recorded:
        echo("filedoc: the header zone changed since emit; leaving the file unchanged.")
        return None

    answer = entry.get("answer") or {}
    description = str(answer.get("description") or "").strip()
    if not description or description.upper() == "NONE":
        return None
    desc_range = _parse_classify_range(str(answer.get("range") or ""), len(target.eligible))
    return splice_description(source_lines, target, desc_range, description)


def scan_brace_leading_zone(source_lines: Chunk, style: CommentStyle) -> Optional[FileDocTarget]:
    """
    Build a `FileDocTarget` for a brace-language file (C, JS) by scanning its leading-comment zone.

    Gathers the entire run of leading comments at the top of the file - which may span several contiguous blocks
    (mixed `/* ... */` and `//`, separated by blank lines but with no intervening code) - into one target. The scan
    starts after an optional shebang, collects every comment line until the first real code/preprocessor line, and
    marks the pure-content comment lines (block continuations and `//` lines) as description-eligible while leaving
    delimiters, single-line `/* ... */` comments, and blank continuations to be preserved.

    Parameters:
    - `source_lines`: The source split into individual lines.
    - `style`: The comment style used to render a fresh block when the file has no leading comments.

    Returns:
    - A `FileDocTarget`, or None if the file is empty.
    """

    if not source_lines or not any(ln.strip() for ln in source_lines):
        return None

    n = len(source_lines)
    i = 0
    if source_lines[0].lstrip().startswith("#!"):     # preserve a shebang (e.g. `#!/usr/bin/env node`) if present
        i = 1
    fresh_index = i

    eligible: List[Tuple[int, str, str]] = []
    first_comment = None                 # 0-based index of the first comment line
    last_comment = None                  # 0-based index of the last comment line
    append_index = None                  # 0-based insertion point for an appended description
    append_prefix = ""
    in_block = False

    idx = i
    while idx < n:
        line = source_lines[idx]
        stripped = line.strip()
        leading_ws = line[: len(line) - len(line.lstrip())]

        if in_block:
            first_comment = idx if first_comment is None else first_comment
            last_comment = idx
            closes = "*/" in stripped
            if not closes:
                body = stripped
                if body.startswith("*"):
                    inner = body.lstrip("*").strip()
                    prefix = leading_ws + "* "
                else:
                    inner = body
                    prefix = leading_ws
                if inner:
                    eligible.append((idx + 1, prefix, inner))
            else:
                in_block = False
                append_index = idx          # append before this closing delimiter
                append_prefix = leading_ws + "* "
            idx += 1
            continue

        if stripped.startswith("/*"):
            first_comment = idx if first_comment is None else first_comment
            last_comment = idx
            if "*/" in stripped[2:]:          # single-line /* ... */ - preserve whole, not eligible
                append_index = idx + 1
                append_prefix = "// "
            else:
                in_block = True
            idx += 1
            continue

        if stripped.startswith("//"):
            first_comment = idx if first_comment is None else first_comment
            last_comment = idx
            inner = stripped.lstrip("/").strip()
            prefix = leading_ws + "// "
            if inner:
                eligible.append((idx + 1, prefix, inner))
            append_index = idx + 1            # append after this line-comment
            append_prefix = prefix
            idx += 1
            continue

        if stripped == "":
            idx += 1                          # blank: a leading blank, or a gap between blocks; keep scanning
            continue

        break                                 # first real code / preprocessor line ends the zone

    has_zone = first_comment is not None
    if not has_zone:
        return FileDocTarget(eligible=[], insert_index=fresh_index, insert_fresh=True,
                             style=style, indent="", has_zone=False)

    return FileDocTarget(
        eligible=eligible,
        insert_index=append_index if append_index is not None else last_comment + 1,
        insert_prefix=append_prefix or "// ",
        insert_fresh=False,
        style=style,
        indent="",
        has_zone=True,
    )
