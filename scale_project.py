#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The project-context layer that sits above SCALE's per-file pipeline.

SCALE is otherwise single-file: it primes on one file's summary and annotates it in isolation, which makes file
descriptions read generically (an `error.c` that never mentions it belongs to a BASIC interpreter). This module gives
the per-file pipeline a *broader view* - but only as small, distilled facts, because the local model's context window
is tight. The byte-for-byte code guarantee is unaffected: nothing here patches source, it only produces context strings
that are fed into the existing priming.

It locates a project overview document (`CLAUDE.md` / `README.*`) near the files being annotated and distils it once
into a short, cached "project blurb" that is injected into every file's priming context.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import glob
import hashlib

from scale_text import summarise, LENGTH_PARAGRAPH
from scale_log import echo


# Source extensions used when a target/reference path is a directory (so a directory expands to just its source files,
# not every file). Explicitly named files and glob matches are taken as-is regardless of extension.
SOURCE_EXTS = (".py", ".c", ".h", ".js", ".mjs", ".cjs")

# Cap on how many read-only reference files are summarised into the shared context, so the injected one-liners stay
# small regardless of how large the --reference set is (the overflow is logged, not silently dropped).
MAX_REFERENCE_FILES = 16


# Reply-length cap for the project blurb. It is background context shown above many files, so it is kept to a couple of
# sentences regardless of how large the source overview document is.
PROJECT_BLURB_MAX_TOKENS = 256

# Built-in default for the blurb instruction (overridable via scale-cfg/project.txt).
PROJECT_BLURB_INSTRUCTION = (
    "Summarise what this software project is, for a developer who is about to read its source files. In two or three "
    "sentences say what the project does, its domain, and any key concepts or terminology a reader should know. This "
    "is background shown above individual files, so keep it short and general: do not describe individual files, "
    "functions, or APIs, and do not pad with build/usage/installation detail. Plain prose - no headings, no lists."
)

# The cache lives alongside SCALE's summary cache (same directory convention, no import dependency on scale.py).
_CACHE_DIR = Path(__file__).resolve().parent / "__cache__"

# Project-overview document discovery. `CLAUDE.md` is preferred; otherwise a README with any common extension, any case.
_PREFERRED_NAME = "claude.md"
_README_STEM = "readme"
_DOC_SUFFIXES = ("", ".md", ".markdown", ".rst", ".txt")


def _read_text_bytes(path: Path) -> str:
    """
    Read a file as text using the same surrogateescape decoding SCALE uses for source, returning "" if unreadable.

    Parameters:
    - `path`: The file to read.

    Returns:
    - The decoded contents, or "" on any read error.
    """

    try:
        return path.read_bytes().decode("utf-8", errors="surrogateescape")
    except OSError:
        return ""


def _read_optional(path: Path) -> Optional[str]:
    """Return a config file's text if it exists, otherwise None (a local copy so this module needs no scale.py import)."""

    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def find_project_doc(start: Path, max_levels: int = 8) -> Optional[Path]:
    """
    Locate the nearest project-overview document by walking up from `start`.

    From the directory of `start` (or `start` itself if it is a directory) and upwards, each directory is searched for
    `CLAUDE.md` (preferred) or a `README` with any common extension (`.md`/`.markdown`/`.rst`/`.txt`/none), matched
    case-insensitively. The first match wins (so a doc nearest the file beats one further up). The walk stops at a
    repository root (a directory containing `.git`, inclusive), the filesystem root, or after `max_levels` directories.

    Parameters:
    - `start`: A target file or directory to search from.
    - `max_levels`: The maximum number of directories to ascend.

    Returns:
    - The path to the chosen document, or None if none is found.
    """

    d = (start if start.is_dir() else start.parent).resolve()
    for _ in range(max_levels):
        try:
            entries = [p for p in d.iterdir() if p.is_file()]
        except OSError:
            entries = []

        preferred = [p for p in entries if p.name.lower() == _PREFERRED_NAME]
        if preferred:
            return preferred[0]

        readmes = [p for p in entries if p.stem.lower() == _README_STEM and p.suffix.lower() in _DOC_SUFFIXES]
        if readmes:
            # Prefer a Markdown README, then settle ties by name for determinism.
            readmes.sort(key=lambda p: (p.suffix.lower() != ".md", p.name))
            return readmes[0]

        if (d / ".git").exists() or d.parent == d:   # repo root (inclusive) or filesystem root
            break
        d = d.parent
    return None


def _blurb_cache_path(doc_text: str) -> Path:
    """Return the cache file for a blurb, keyed on a content hash of the source document (so edits invalidate it)."""

    digest = hashlib.sha256(doc_text.encode("utf-8", errors="surrogateescape")).hexdigest()
    return _CACHE_DIR / f"project-{digest}.txt"


def _read_cache(path: Path) -> Optional[str]:
    """Read a cached blurb, or None if absent."""

    try:
        return path.read_bytes().decode("utf-8", errors="surrogateescape")
    except FileNotFoundError:
        return None


def _write_cache(path: Path, text: str) -> None:
    """Atomically write a blurb to the cache (temp file + replace)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(text.encode("utf-8", errors="surrogateescape"))
    tmp.replace(path)


def _crop_to_budget(text: str, llm, budget_tokens: int) -> str:
    """
    Crop a document to a token budget, keeping the top (where a README/overview states what the project is).

    Parameters:
    - `text`: The document text.
    - `llm`: A model exposing `estimate_tokens`.
    - `budget_tokens`: The maximum estimated tokens to keep.

    Returns:
    - The original text if it fits, else its leading lines up to the budget plus a truncation marker.
    """

    if llm.estimate_tokens(text) <= budget_tokens:
        return text
    kept: list[str] = []
    for line in text.splitlines():
        kept.append(line)
        if llm.estimate_tokens("\n".join(kept)) > budget_tokens:
            kept.pop()
            break
    return "\n".join(kept) + "\n\n[... overview document truncated ...]"


def project_blurb(llm, cfg, scale_path: Path, doc_path: Path, no_cache: bool = False) -> str:
    """
    Distil a project-overview document into a short, cached "project blurb" for priming context.

    The blurb is a couple of sentences describing the project, its domain, and key terminology - background to be shown
    above individual files so the per-file passes stop reading as if each file stands alone. It is cached by the
    document's content hash (so editing the doc regenerates it). Large documents are cropped to a token budget first.

    Parameters:
    - `llm`: A model exposing `generate`, `estimate_tokens`, `n_ctx`, and `ctx_margin`.
    - `cfg`: The base generation configuration.
    - `scale_path`: The SCALE configuration directory (for the optional `project.txt` instruction override).
    - `doc_path`: The overview document to distil.
    - `no_cache`: When True, regenerate rather than loading a cached blurb.

    Returns:
    - The blurb text, or "" if the document is empty/unreadable.
    """

    doc_text = _read_text_bytes(doc_path)
    if not doc_text.strip():
        return ""

    cache_path = _blurb_cache_path(doc_text)
    if not no_cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            echo(f"Loaded project blurb from cache ({doc_path.name})...")
            return cached

    echo(f"Distilling project overview from {doc_path.name}...")
    instruction = _read_optional(scale_path / "project.txt") or PROJECT_BLURB_INSTRUCTION
    budget = max(256, llm.n_ctx - llm.ctx_margin - PROJECT_BLURB_MAX_TOKENS - 64)
    cropped = _crop_to_budget(doc_text, llm, budget)
    blurb = summarise(llm, cfg, cropped, LENGTH_PARAGRAPH, subject="a software project's overview document",
                      max_tokens=PROJECT_BLURB_MAX_TOKENS, instruction=instruction)
    _write_cache(cache_path, blurb)
    return blurb


def gather_files(patterns: List[str], exts: Tuple[str, ...] = SOURCE_EXTS) -> List[Path]:
    """
    Expand a list of path patterns into a deduplicated, ordered list of existing files.

    Each pattern is resolved as: a glob (when it contains `*`/`?`/`[`, with `**` recursion); a directory (expanded
    recursively to files whose extension is in `exts`); or a single named file (taken as-is, any extension). Order is
    deterministic (sorted by path) and duplicates (by resolved path) are removed, so overlapping patterns are safe.

    Parameters:
    - `patterns`: The raw path/glob/dir strings (e.g. from the CLI).
    - `exts`: The source extensions a directory expands to.

    Returns:
    - The matching existing files, sorted, without duplicates.
    """

    out: List[Path] = []
    seen = set()
    for pattern in patterns:
        p = Path(pattern)
        if any(ch in pattern for ch in "*?["):
            candidates = [Path(m) for m in glob.glob(pattern, recursive=True)]
        elif p.is_dir():
            candidates = [c for c in sorted(p.rglob("*")) if c.suffix.lower() in exts]
        else:
            candidates = [p]
        for c in candidates:
            if not c.is_file():
                continue
            key = c.resolve()
            if key not in seen:
                seen.add(key)
                out.append(c)
    return sorted(out, key=lambda x: str(x))


def compose_project_context(blurb: str, related: List[Tuple[str, str]]) -> str:
    """
    Format the project blurb and a list of related-file one-liners into a single context string for priming.

    Parameters:
    - `blurb`: The project overview blurb (may be "").
    - `related`: `(filename, one_line_summary)` pairs for read-only reference files the run should be aware of.

    Returns:
    - A combined context string (possibly ""), suitable for injecting as a priming turn.
    """

    parts: List[str] = []
    if blurb.strip():
        parts.append(blurb.strip())
    related = [(name, summary.strip()) for name, summary in related if summary.strip()]
    if related:
        lines = "\n".join(f"- {name}: {summary}" for name, summary in related)
        parts.append("Related files in this project (read-only, for reference):\n" + lines)
    return "\n\n".join(parts)


def resolve_project_doc(project_doc_arg: str, start: Path) -> Optional[Path]:
    """
    Resolve the project-overview document from the `--project-doc` argument and the target location.

    Parameters:
    - `project_doc_arg`: The CLI value: "" to auto-detect, "none" to disable, or an explicit path.
    - `start`: A target file/directory to auto-detect from.

    Returns:
    - The resolved document path, or None when disabled or not found.
    """

    if project_doc_arg.strip().lower() == "none":
        return None
    if project_doc_arg:
        p = Path(project_doc_arg)
        return p if p.is_file() else None
    return find_project_doc(start)
