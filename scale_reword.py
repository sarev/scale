#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The header-reword manifest used by SCALE's offline file-doc flow: a prose-only round in which a stronger model rewords
each file's freshly drafted top-of-file description without ever seeing or touching code. The module builds, writes,
reads and validates the manifest, and `unfilled_rewords` lists the entries still awaiting an answer, so completeness
is counted rather than trusted.

`apply_reword` performs the guarded splice: the original draft is located in the header zone by whitespace-insensitive
matching, the answer is sanitised and rewrapped under the original comment prefix, and the edit is abandoned if the
answer is empty, the draft no longer matches, the matched range looks like legal text, or the preservation guard
reports any change outside the description.
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
    Build a header-reword manifest dictionary.

    Parameters:
    - `project_blurb`: A short project description included to give the rewording model context.
    - `entries`: The per-file reword entries to embed.

    Returns:
    - The manifest dictionary, stamped with the reword tool name and version.
    """

    return {
        "version": REWORD_VERSION,
        "tool": REWORD_TOOL,
        "project_blurb": project_blurb,
        "files": entries,
    }


def write_reword_manifest(path: Path, manifest: dict) -> None:
    """
    Write a reword manifest to disk as pretty-printed UTF-8 JSON.

    Parameters:
    - `path`: Destination path for the manifest file.
    - `manifest`: The manifest dictionary to serialise.
    """

    # Non-ASCII prose is written verbatim rather than escaped, keeping the file human-readable.
    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_reword_manifest(path: Path) -> dict:
    """
    Load and validate a header-reword manifest from disk.

    Parameters:
    - `path`: The path of the manifest JSON file.

    Returns:
    - The parsed manifest dictionary, with a `files` list guaranteed to be present.

    Notes:
    Raises `ValueError` if the file is not a SCALE reword manifest of the expected version.
    """

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))

    # Refuse anything that is not exactly this tool and version - silently misreading a foreign manifest would be worse than failing.
    if (not isinstance(manifest, dict) or manifest.get("tool") != REWORD_TOOL
            or manifest.get("version") != REWORD_VERSION):
        raise ValueError(f"{path}: not a SCALE reword manifest of version {REWORD_VERSION}")

    # Guarantee `files` exists so callers can iterate without guarding.
    manifest.setdefault("files", [])

    return manifest


def unfilled_rewords(manifest: dict) -> List[str]:
    """
    List the paths of reword manifest entries whose answers are still blank.

    Parameters:
    - `manifest`: The reword manifest dictionary to inspect.

    Returns:
    - The path of each entry with no usable answer; empty when the manifest is fully filled.
    """

    return [f.get("path", "?") for f in manifest.get("files", []) if not str(f.get("answer") or "").strip()]


def apply_reword(source_lines: Chunk, target: FileDocTarget, draft: str, answer: str) -> Tuple[Chunk, bool]:
    """
    Splice a reworded header description into the source lines.

    The draft is located in the header zone by whitespace-insensitive matching of a contiguous run of eligible lines, and that run is replaced by the sanitised answer rewrapped with the original comment prefix. The edit is abandoned, returning the lines unchanged, if the answer is empty or NONE, the draft no longer matches, the matched range looks like legal text, or the preservation guard reports that anything outside the description changed.

    Parameters:
    - `source_lines`: The current source lines to patch.
    - `target`: The file-doc target describing the header zone and its preservation check.
    - `draft`: The original description text the answer was written against.
    - `answer`: The reworded description, or empty/NONE to decline.

    Returns:
    - A tuple of the (possibly patched) lines and `True` if the reword was applied, `False` otherwise.
    """

    # An empty or NONE answer is a deliberate decline - return unchanged before touching anything.
    text = str(answer or "").strip()
    if not text or text.upper() == "NONE" or not (draft or "").strip():
        return source_lines, False
    entries = target.eligible
    def _collapse(s: str) -> str:
        """
        Collapse whitespace and rejoin hyphen-split words so differently wrapped text compares equal.

        Parameters:
        - `s`: The text to normalise.

        Returns:
        - The collapsed single-line form.
        """

        # Rejoining words split at a hyphen makes wrapped and unwrapped text compare equal.
        return " ".join(s.split()).replace("- ", "-")

    # Compare in collapsed form so the rewrapping applied when the draft was emitted cannot break the match.
    want = _collapse(draft)
    found: Optional[Tuple[int, int]] = None

    # Search every start line for a contiguous run of header lines whose collapsed text equals the draft.
    for i in range(len(entries)):
        acc: List[str] = []

        for j in range(i, len(entries)):
            acc.append(entries[j][2])
            got = _collapse(" ".join(acc))

            if got == want:
                found = (i, j)
                break

            # Once the accumulated text overshoots the draft, no longer run from this start can match - prune early.
            if len(got) >= len(want):
                break

        if found:
            break

    # No match means the header changed since the draft was taken; refuse to guess at a splice point.
    if found is None:
        echo("reword: the draft description was not found in the header zone (file changed?); leaving it unchanged.")
        return source_lines, False

    i, j = found

    # Licence veto: never reword a range that looks like legal text.
    if any(looks_legal(entries[k][2]) for k in range(i, j + 1)):
        echo("reword: the matched range looks like license/legal text; leaving it unchanged.")
        return source_lines, False

    # Splice the sanitised description back in with the original comment prefix and wrapping; nothing outside the matched range is touched.
    start = entries[i][0] - 1                  # 0-based
    end = entries[j][0] - 1
    removed = end - start + 1
    prefix = entries[i][1]
    description = _sanitise_description(text)
    if not description:
        return source_lines, False
    new_block = [f"{prefix}{ln}".rstrip() for ln in _wrap(description, prefix)]
    out = source_lines[:start] + new_block + source_lines[end + 1:]

    # Prefer the target's own preservation check; with no check available at all, fail closed.
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
