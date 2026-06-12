#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The header-reword manifest: a run-level, prose-only escalation of the file descriptions to a stronger model.

The local model writes each file's header description in isolation, so a run's descriptions read as disconnected
single-file prose - duplicate role claims, no parallel phrasing between a `foo.c` and its `foo-fns.h`. This manifest
fixes that with one bounded, machine-checkable unit: it carries (a) the project blurb and (b), per target file, its
name, its role classification (header / implementation / other), and the LOCAL DRAFT description that `--file-doc`
just spliced - so the stronger model sees the big picture first and then rewords every description with cross-file
consistency.

**Prose only - the splicing rules are never delegated to the stronger model.** No author, licence, or boilerplate text
enters the manifest, and the model's answers never touch the file directly: the model-free apply locates each draft in
its file's header zone by EXACT MATCH (emit wrote it, so it is known verbatim - the textual analogue of the function
manifest's sig_hash re-binding), replaces it through the existing preservation guard, and treats a miss (the file was
edited in between) as a safe no-op. Where a no-function file's description came from a map-reduce, the richer
pre-shaping summary rides along as `context` (better input for the reword); the `draft` is always the spliced text,
because it is the locator.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import json

from scale_filedoc import FileDocTarget, file_doc_preserved, looks_legal, _sanitise_description, _wrap
from scale_log import echo

Chunk = List[str]


REWORD_VERSION = 1
REWORD_TOOL = "scale-reword"


def reword_manifest(project_blurb: str, entries: List[dict]) -> dict:
    """
    Build the run's header-reword manifest from the project blurb and the per-file draft entries.

    Parameters:
    - `project_blurb`: The run's distilled project background (shown first, so the rewords share the big picture).
    - `entries`: One dict per target file: `{"path", "language", "role", "draft", "context", "answer": None}`.

    Returns:
    - The manifest dictionary ready to be written as JSON.
    """

    return {
        "version": REWORD_VERSION,
        "tool": REWORD_TOOL,
        "project_blurb": project_blurb,
        "files": entries,
    }


def write_reword_manifest(path: Path, manifest: dict) -> None:
    """Write a reword manifest to disk as indented JSON (UTF-8)."""

    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_reword_manifest(path: Path) -> dict:
    """
    Read and lightly validate a header-reword manifest.

    Parameters:
    - `path`: The manifest file path.

    Returns:
    - The parsed manifest dictionary.

    Raises:
    - `ValueError`: If the file is not a reword manifest of a supported version.
    """

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("tool") != REWORD_TOOL
            or manifest.get("version") != REWORD_VERSION):
        raise ValueError(f"{path}: not a SCALE reword manifest of version {REWORD_VERSION}")
    manifest.setdefault("files", [])
    return manifest


def unfilled_rewords(manifest: dict) -> List[str]:
    """
    Return the path of every file entry whose `answer` has not been filled in (the completeness counter).

    An answer of `null`/whitespace is unfilled; the explicit string `"NONE"` (keep the draft) counts as filled.

    Parameters:
    - `manifest`: The parsed reword manifest.

    Returns:
    - The unfilled entries' paths.
    """

    return [f.get("path", "?") for f in manifest.get("files", []) if not str(f.get("answer") or "").strip()]


def apply_reword(source_lines: Chunk, target: FileDocTarget, draft: str, answer: str) -> Tuple[Chunk, bool]:
    """
    Replace a file's header description with a reworded one, locating the draft by exact (whitespace-collapsed) match.

    The splice machinery mirrors `--file-doc`: the matched lines are replaced with the answer re-wrapped in the same
    per-line decoration, and the whole edit goes through the preservation guard (the target's own parse-based guard
    for Python, the comment-line guard otherwise). Any failure - the draft not found (the file changed since emit), a
    legal-looking line inside the match, an answer of NONE, or a guard rejection - leaves the file unchanged.

    Parameters:
    - `source_lines`: The file's current lines.
    - `target`: The file's `FileDocTarget` (the header zone, freshly scanned from the same lines).
    - `draft`: The local draft description recorded at emit (the locator and the text being replaced).
    - `answer`: The stronger model's reworded description ("NONE" keeps the draft).

    Returns:
    - `(lines, changed)`: the (possibly unchanged) lines, and whether the reword was applied.
    """

    text = str(answer or "").strip()
    if not text or text.upper() == "NONE" or not (draft or "").strip():
        return source_lines, False

    entries = target.eligible

    # Collapse whitespace, then heal hyphen line-breaks: the wrapper may have split "role-based" as "role-" +
    # "based", which the plain line join turns into "role- based". Applied to BOTH sides the transform is
    # self-consistent, so equal underlying texts always match.
    def _collapse(s: str) -> str:
        return " ".join(s.split()).replace("- ", "-")

    want = _collapse(draft)
    found: Optional[Tuple[int, int]] = None
    for i in range(len(entries)):
        acc: List[str] = []
        for j in range(i, len(entries)):
            acc.append(entries[j][2])
            got = _collapse(" ".join(acc))
            if got == want:
                found = (i, j)
                break
            if len(got) >= len(want):
                break
        if found:
            break
    if found is None:
        echo("reword: the draft description was not found in the header zone (file changed?); leaving it unchanged.")
        return source_lines, False

    i, j = found
    # Belt and braces: even a matched range is refused if any line smells legal (the draft never should).
    if any(looks_legal(entries[k][2]) for k in range(i, j + 1)):
        echo("reword: the matched range looks like license/legal text; leaving it unchanged.")
        return source_lines, False

    start = entries[i][0] - 1                  # 0-based
    end = entries[j][0] - 1
    removed = end - start + 1
    prefix = entries[i][1]
    description = _sanitise_description(text)
    if not description:
        return source_lines, False
    new_block = [f"{prefix}{ln}".rstrip() for ln in _wrap(description, prefix)]
    out = source_lines[:start] + new_block + source_lines[end + 1:]

    if target.preserved is not None:
        ok = target.preserved(source_lines, out, start, removed, len(new_block))
    elif target.style is not None:
        ok = file_doc_preserved(source_lines, out, start, removed, len(new_block), target.style)
    else:
        ok = False
    if not ok:
        echo("reword: the edit would have altered code or preserved text; abandoning it.")
        return source_lines, False
    return out, True
