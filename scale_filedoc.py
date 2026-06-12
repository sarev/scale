#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The engine behind SCALE's top-of-file description pass. `annotate_file_doc` runs the offline flow over one file: a
classification turn decides which existing header lines form the current description, the summary provider writes the
new prose (updating the old text rather than starting from nothing), and `splice_description` performs the edit -
replacing the classified range in place, extending the surviving comment block, or opening a fresh one.

The model only classifies and writes; safety lives in the deterministic machinery around it. `looks_legal` vetoes any
range that resembles licence or legal boilerplate, `_sanitise_description` strips fences and comment markup from the
model's prose, and `file_doc_preserved` proves the splice changed nothing but comment and blank lines before the edit
is accepted.

The module also carries the online `scale-filedoc` round: building, writing, reading and completeness-checking the
filedoc manifest, plus `apply_filedoc_entry`, which re-binds each answer to the live header zone by content and
applies it through the same guarded splice. `scan_brace_leading_zone` supplies the shared header-zone scanner for the
brace languages.
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
    Heuristically decide whether text looks like licence or legal boilerplate.

    Parameters:
    - `text`: The candidate text to test.

    Returns:
    - `True` if any known licence marker appears in the text (case-insensitively), otherwise `False`.
    """

    low = text.lower()
    return any(marker in low for marker in LICENSE_MARKERS)


@dataclass
class FileDocTarget:

    """
    Placement plan for a file's top-of-file description.

    Parameters:
    - `eligible`: Existing description candidates as `(line number, comment prefix, text)` triples.
    - `insert_index`: 0-based line index at which a fresh description is spliced in.
    - `insert_prefix`: Comment prefix to reuse when extending an existing comment block.
    - `insert_fresh`: `True` to open a brand-new comment block rather than extend one.
    - `style`: The language's comment style; `None` means no safe target was found.
    - `indent`: Indentation applied to a freshly inserted block.
    - `has_zone`: Whether the file already has a recognised header comment zone.
    - `preserved`: Optional override for the preservation guard used to vet the splice.
    """

    # eligible rows are (line number, comment prefix, text) triples describing the existing header lines.
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
    Parse a classifier reply into a 1-based inclusive line range.

    Accepts ranges such as `3-7` or `3 to 7`, or a single line number, tolerating surrounding prose; both ends are clamped to `1..n`.

    Parameters:
    - `reply`: The raw model reply naming the description lines.
    - `n`: The number of eligible lines on offer.

    Returns:
    - A `(start, end)` tuple of 1-based inclusive indices, or `None` if the reply names no usable range.
    """

    # An empty or none-style reply is the model declining: there is no description block to report.
    if n <= 0:
        return None
    text = reply.strip().lower()
    if not text or text.startswith("none"):
        return None
    m = re.search(r"(\d+)\s*(?:-|to|–)\s*(\d+)", text)

    # Fall back to a lone number when the reply names a single line rather than a range.
    if m:
        start, end = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"\d+", text)
        if not m:
            return None
        start = end = int(m.group(0))

    # Clamp both ends into 1..n; a range still inverted after clamping is treated as no answer.
    start = max(1, min(start, n))
    end = max(1, min(end, n))
    if end < start:
        return None
    return start, end


def _wrap(text: str, prefix: str) -> List[str]:
    """
    Wrap description text into comment-width lines.

    Paragraphs (separated by blank lines) are re-flowed individually with a blank entry between them; original line breaks within a paragraph are discarded, and words are never split.

    Parameters:
    - `text`: The raw description text.
    - `prefix`: The comment prefix the lines will sit behind; its length reduces the wrap width.

    Returns:
    - The wrapped lines; never empty (a single empty string when the text is blank).
    """

    # Guarantee at least 40 columns of text even when the comment prefix is long.
    width = max(40, WRAP_WIDTH - len(prefix))
    out: List[str] = []
    paragraphs = re.split(r"\n\s*\n", text.strip())

    # Re-flow each paragraph from scratch: the text's original line breaks are deliberately discarded.
    for i, para in enumerate(paragraphs):
        if i:
            out.append("")                       # blank line between paragraphs
        collapsed = " ".join(para.split())
        if collapsed:
            out.extend(textwrap.wrap(collapsed, width=width, break_on_hyphens=False, break_long_words=False))

    # Never return an empty list, so callers can always build a comment block.
    return out or [""]


def file_doc_preserved(old_lines: Chunk, new_lines: Chunk, start: int, removed: int, added: int,
                       style: CommentStyle) -> bool:
    """
    Check that a file-doc splice changed nothing but comment and blank lines.

    The guard demands byte-for-byte equality outside the edited window, and that every non-blank line inside the window - in both the old and new text - is a comment in the given style.

    Parameters:
    - `old_lines`: The source lines before the edit.
    - `new_lines`: The source lines after the edit.
    - `start`: 0-based index of the first edited line.
    - `removed`: Number of lines removed from the old text.
    - `added`: Number of lines added in the new text.
    - `style`: The comment style used to recognise comment lines.

    Returns:
    - `True` if the edit is confined to comments and blank lines, otherwise `False`.
    """

    # Everything outside the edited window must survive byte-for-byte.
    if old_lines[:start] != new_lines[:start]:
        return False
    if old_lines[start + removed:] != new_lines[start + added:]:
        return False
    region_old = old_lines[start:start + removed]
    region_new = new_lines[start:start + added]

    # Inside the window, any non-blank line that is not a comment vetoes the edit.
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
    Splice a file description into the source, guarded against touching anything but comments.

    Three placements are supported: replacing an existing description range in place, opening a fresh comment block at the insertion point, or extending the surviving header comment under its own prefix. A range containing licence-like text is vetoed outright, and the result must pass the preservation guard or the edit is abandoned.

    Parameters:
    - `source_lines`: The pristine source lines to splice into.
    - `target`: The placement plan (comment style, insertion point, eligible lines, optional guard override).
    - `desc_range`: 1-based inclusive range into `target.eligible` naming the existing description, or `None` to insert anew.
    - `description`: The new description text.

    Returns:
    - The new list of source lines, or `None` if the splice was vetoed or failed the guard.
    """

    # Bail out early when there is no comment style to write in or nothing survives sanitisation.
    style = target.style
    if style is None:
        return None
    description = _sanitise_description(description)
    if not description:
        return None

    # Licence veto: inspect the chosen range before rewriting anything.
    if desc_range is not None:
        lo, hi = desc_range

        # A single legal-looking line anywhere in the range is enough to abandon the rewrite.
        if any(looks_legal(target.eligible[i - 1][2]) for i in range(lo, hi + 1)):
            echo("file-doc: the description range looks like license/legal text; leaving it untouched.")
            return None

    # Replace the existing description in place, reusing its comment prefix so the block keeps its established look.
    if desc_range is not None:
        lo, hi = desc_range
        start_lineno = target.eligible[lo - 1][0]
        end_lineno = target.eligible[hi - 1][0]
        prefix = target.eligible[lo - 1][1]
        start = start_lineno - 1                 # 0-based
        removed = end_lineno - start_lineno + 1
        new_block = [f"{prefix}{ln}".rstrip() for ln in _wrap(description, prefix)]
    elif target.insert_fresh:
        start = target.insert_index
        removed = 0

        # No existing description: a fresh insert opens a brand-new comment block, otherwise the text rides under the surviving header's prefix.
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
        start = target.insert_index
        removed = 0
        prefix = target.insert_prefix
        new_block = [prefix.rstrip()] + [f"{prefix}{ln}".rstrip() for ln in _wrap(description, prefix)]

    # Build the candidate output; nothing is committed until the guard passes.
    out_lines = source_lines[:start] + new_block + source_lines[start + removed:]

    # The target may supply its own preservation guard; otherwise the comment-only default applies.
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
    Run the file-doc pass over one brace-language file: classify its existing header, write a description, and splice it in.

    The model is used only to classify which header lines form the current description and (via `summary_provider`) to write the new prose; the actual edit is done by `splice_description`, so every untouched line is preserved byte-for-byte. A classified range that looks like license or legal text is vetoed and left alone, and the existing description (when found) is fed to the writer so it is updated rather than regenerated from nothing.

    Parameters:
    - `llm`: The local chat model used for the classification turn.
    - `cfg`: Generation settings; the classify turn runs with a reduced token cap and temperature.
    - `base_messages`: The primed conversation that the classify turn is appended to.
    - `source_lines`: The file's pristine source lines.
    - `target`: The scanned header zone with its eligible lines and insertion point.
    - `summary_provider`: Callback that produces the description text, given the existing description or `None`.
    - `language`: Language name interpolated into the classify prompt.
    - `classify_prompt`: Optional override for the classification prompt template.
    - `on_description`: Optional callback invoked with the final description once it has been applied.

    Returns:
    - The updated source lines, or `source_lines` unchanged if the style is missing, no usable description was produced, or the splice was refused.
    """

    # No comment style means this language cannot host a file description, so bail out before any model work.
    style = target.style
    if style is None:
        return source_lines
    classify_tmpl = classify_prompt or CLASSIFY_PROMPT
    desc_range: Optional[Tuple[int, int]] = None

    # Show the model the numbered header lines and ask which range, if any, is the existing description - a tightly capped, low-temperature turn.
    if target.eligible:
        entries = "\n".join(f"{i}. {inner}" for i, (_, _, inner) in enumerate(target.eligible, start=1))
        prompt = _fill(classify_tmpl, language=language, entries=entries)
        turn_cfg = replace(cfg, max_new_tokens=min(cfg.max_new_tokens, CLASSIFY_REPLY_TOKENS),
                           temperature=CLASSIFY_TEMPERATURE)
        messages = base_messages + [{"role": "user", "content": prompt}]
        reply = llm.generate(messages, cfg=turn_cfg)
        desc_range = _parse_classify_range(reply, len(target.eligible))

        if desc_range is not None:
            lo, hi = desc_range

            # License veto: anything in the range that reads as legal text must never be rewritten, so drop the classification instead.
            if any(looks_legal(target.eligible[i - 1][2]) for i in range(lo, hi + 1)):
                echo("file-doc: classified range looks like license/legal text; leaving it untouched.")
                desc_range = None

    existing_text = ""

    # Hand the current description to the writer so it updates the prose rather than starting from scratch.
    if desc_range is not None:
        lo, hi = desc_range
        existing_text = " ".join(target.eligible[i - 1][2] for i in range(lo, hi + 1)).strip()

    # The prose itself comes from the caller's summary provider; strip any comment markup it leaked.
    description = _sanitise_description(summary_provider(existing_text or None))

    if not description:
        echo("file-doc: no usable description was produced; leaving the file unchanged.")
        return source_lines

    # The splice is the only step that edits the file; a None result means the preservation guard refused, so keep the original lines.
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
    Strip code fences and comment markup from a model-produced file description.

    The splicer adds the comment delimiters itself, so any fences, block-comment delimiters or leading `*`/`//`/`#` markers the model emitted are removed to avoid doubled-up syntax.

    Parameters:
    - `text`: The raw description text returned by the model.

    Returns:
    - The cleaned plain-prose description, possibly empty.
    """

    text = text.strip()

    # Models sometimes wrap the reply in a code fence; peel it off before the line-level cleaning.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()

    lines = text.split("\n")
    cleaned: List[str] = []

    # Strip any comment delimiters the model added - the splicer supplies its own markup, so leftovers would double up.
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
    Build a run-level filedoc manifest dictionary from its parts.

    Parameters:
    - `description_spec`: The instructions describing what a good file description contains.
    - `project_doc`: The project blurb shared by every entry.
    - `entries`: The per-file entry dictionaries.

    Returns:
    - The manifest dictionary, stamped with the filedoc tool name and version.
    """

    return {
        "version": FILEDOC_VERSION,
        "tool": FILEDOC_TOOL,
        "description_spec": description_spec,
        "project_doc": project_doc,
        "files": entries,
    }


def write_filedoc_manifest(path: Path, manifest: dict) -> None:
    """
    Write a filedoc manifest to disk as pretty-printed UTF-8 JSON.

    Parameters:
    - `path`: Destination file path.
    - `manifest`: The manifest dictionary to serialise.
    """

    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_filedoc_manifest(path: Path) -> dict:
    """
    Load a filedoc manifest from disk and validate its identity.

    Parameters:
    - `path`: Path of the manifest JSON file.

    Returns:
    - The manifest dictionary, with the `files` list guaranteed to be present.

    Notes:
    Raises `ValueError` if the file is not a SCALE filedoc manifest of the expected version.
    """

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))

    # Reject foreign or wrong-version manifests up front rather than failing obscurely mid-apply.
    if (not isinstance(manifest, dict) or manifest.get("tool") != FILEDOC_TOOL
            or manifest.get("version") != FILEDOC_VERSION):
        raise ValueError(f"{path}: not a SCALE filedoc manifest of version {FILEDOC_VERSION}")

    # Guarantee a files list so callers can iterate without guarding.
    manifest.setdefault("files", [])

    return manifest


def unfilled_descriptions(manifest: dict) -> List[str]:
    """
    List the manifest files whose description answers are still unfilled.

    An entry only counts as filled when its answer carries both a non-blank `range` and a non-blank `description`; completeness is decided by this count, never by trusting the filler.

    Parameters:
    - `manifest`: The filedoc manifest dictionary.

    Returns:
    - The paths of every file still awaiting an answer; empty when the manifest is complete.
    """

    out: List[str] = []

    # An entry counts as filled only when both the range and the description carry non-blank text.
    for f in manifest.get("files", []):
        answer = f.get("answer") or {}
        if not str(answer.get("range") or "").strip() or not str(answer.get("description") or "").strip():
            out.append(f.get("path", "?"))

    return out


def apply_filedoc_entry(source_lines: Chunk, target: FileDocTarget, entry: dict) -> Optional[Chunk]:
    """
    Apply one filedoc manifest answer to a file's source lines.

    The entry is re-bound by content: if the live header zone no longer matches the lines recorded at emit time, the answer is discarded and the file left untouched. Application goes through the same guarded `splice_description` path as the offline pass.

    Parameters:
    - `source_lines`: The file's pristine source lines.
    - `target`: The freshly scanned header zone for the file.
    - `entry`: The manifest entry holding the recorded header lines and the filled answer.

    Returns:
    - The updated source lines, or `None` if the zone has drifted, the answer is empty or `NONE`, or the splice was refused.
    """

    # Re-bind by content: gather the live header lines and what the emit phase recorded, for comparison.
    current = [inner for (_lineno, _prefix, inner) in target.eligible]
    recorded = [str(e) for e in (entry.get("entries") or [])]

    if current != recorded:
        # Any drift since emit makes the answer untrustworthy, so refuse to apply rather than risk a bad splice.
        echo("filedoc: the header zone changed since emit; leaving the file unchanged.")
        return None

    # An empty or explicit NONE answer means no description is wanted; otherwise recover the classified range from the reply text.
    answer = entry.get("answer") or {}
    description = str(answer.get("description") or "").strip()
    if not description or description.upper() == "NONE":
        return None
    desc_range = _parse_classify_range(str(answer.get("range") or ""), len(target.eligible))

    # Application goes through the same guarded splice as the offline pass.
    return splice_description(source_lines, target, desc_range, description)


def scan_brace_leading_zone(source_lines: Chunk, style: CommentStyle) -> Optional[FileDocTarget]:
    """
    Scan the leading comment zone of a brace-language file and describe where a file description can live.

    Walks the top of the file (after any shebang) through `/* ... */` blocks, `//` runs and blank lines until the first real code line. Every non-blank comment line is recorded as eligible for rewriting, together with the prefix needed to re-emit it in place; a one-line `/* ... */` is preserved whole and is never eligible. The scan also tracks the best append point for new text: inside an open block just before its `*/`, or after the last `//` line.

    Parameters:
    - `source_lines`: The file's source lines.
    - `style`: The comment style to use when a fresh description must be inserted.

    Returns:
    - A `FileDocTarget` describing the eligible lines and insertion point; one with `has_zone=False` and a fresh insert position when there are no leading comments, or `None` for an effectively empty file.
    """

    # Empty files have no zone; otherwise skip any shebang and set up the scan state for the leading-comment walk.
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

    # Walk the leading lines one by one until the first real code line ends the zone.
    while idx < n:
        line = source_lines[idx]
        stripped = line.strip()
        leading_ws = line[: len(line) - len(line.lstrip())]

        # Inside an open block comment every line extends the zone; watch for the closing delimiter.
        if in_block:
            first_comment = idx if first_comment is None else first_comment
            last_comment = idx
            closes = "*/" in stripped

            if not closes:
                body = stripped

                # Remember each line's decoration prefix so a rewrite can be re-emitted with identical formatting.
                if body.startswith("*"):
                    inner = body.lstrip("*").strip()
                    prefix = leading_ws + "* "
                else:
                    inner = body
                    prefix = leading_ws

                # Only non-blank lines are rewrite candidates; the closing delimiter fixes where appended text would slot in.
                if inner:
                    eligible.append((idx + 1, prefix, inner))
            else:
                in_block = False
                append_index = idx          # append before this closing delimiter
                append_prefix = leading_ws + "* "

            idx += 1
            continue

        # A new block comment joins the zone whether or not it closes on the same line.
        if stripped.startswith("/*"):
            first_comment = idx if first_comment is None else first_comment
            last_comment = idx

            # A one-line block comment is kept verbatim and never eligible, so any new text must follow it as a line comment.
            if "*/" in stripped[2:]:          # single-line /* ... */ - preserve whole, not eligible
                append_index = idx + 1
                append_prefix = "// "
            else:
                in_block = True

            idx += 1
            continue

        # Each line comment is individually eligible; the append point trails the last one so additions extend the run.
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

    # No leading comments at all: report a fresh insertion point just after any shebang.
    if not has_zone:
        return FileDocTarget(eligible=[], insert_index=fresh_index, insert_fresh=True,
                             style=style, indent="", has_zone=False)

    # With a zone, prefer the recorded append point, falling back to just past the last comment line.
    return FileDocTarget(
        eligible=eligible,
        insert_index=append_index if append_index is not None else last_comment + 1,
        insert_prefix=append_prefix or "// ",
        insert_fresh=False,
        style=style,
        indent="",
        has_zone=True,
    )
