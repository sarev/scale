#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The command-line entry point and orchestrator for SCALE: it parses the arguments, loads each source file, primes the
local model, and dispatches the per-language workers before writing the annotated result back out. `main` drives every
mode in order - the model-free manifest utilities, the online emit that collects deferred requests without loading a
model, and the offline annotate path that runs the local LLM over each target.

Around that sit the shared services the passes rely on: `load_source` reads files binary-safely and guesses the
language from content rather than extension; `SummaryCache` keeps per-file summaries on disk, invalidated by content
hash; and a chunk-and-reduce summariser produces the file descriptions used to prime the model, merging hierarchically
when a file exceeds the context window.

`generate_comments` sequences the up-to-three passes over a file - definitions, blocks, then the top-of-file
description - re-priming before each, while the run-level helpers assemble the wider context: scanning the run's
files, building the project call graph and lazy callee one-liners, planning the C header/implementation doc sites, and
ordering targets so headers and callee files are processed first.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from scale_llm import LocalChatModel, GenerationConfig, llm_formatters, Messages, Chunk
from scale_log import echo, error, set_verbosity
from scale_text import (summarise, fit_snippet, LENGTH_LINE, LENGTH_PARAGRAPH, LENGTH_PARAGRAPHS, PRIMING_ACK,
                        MARKER_PYTHON, MARKER_C, MARKER_JS)
import scale_escalate
import scale_filedoc
import scale_project
import scale_reword
import scale_verify
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import argparse
import hashlib
import pickle
import re
import sys
import textwrap
import uuid


# Based upon Qwen2.5.1-Coder-7B-Instruct (a code-specialised model that produces
# the best comments of those tested here).
#
# 7B parameters, 5-bit quantised (Q5_K_M), ~5.2GB, context length 32768.
#
DEFAULT_MODEL = "./models/bartowski/Qwen2.5.1-Coder-7B-Instruct-GGUF/Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf"


# Languages with a worker module wired into generate_comments().
SUPPORTED_LANGUAGES = ("python", "js", "c")


# Reply-length cap (tokens) for summary generation. Summaries are kept short on purpose so the injected overview - and
# therefore the persistent priming context used for every routine - stays small regardless of source file size.
SUMMARY_MAX_TOKENS = 1024


# Rough token allowance for the fixed wrapper text `summarise` wraps around the content, used only when deciding
# whether a summary fits in one pass; a small over-estimate just biases marginally towards the (safe) map-reduce path.
SUMMARY_WRAPPER_TOKENS = 64


# Reply-length cap for the SHORT file summary - the squashed file description used to prime the (context-starved)
# definition pass, where the routine body matters more than a detailed file overview.
SHORT_SUMMARY_MAX_TOKENS = 110


# The whole-file summary is written to a file-DESCRIPTION spec, so the one piece of prose serves both the `--file-doc`
# header and the per-routine priming context (overridable via scale-cfg/summary.txt). `{language}` and `{seed}` are
# filled by literal substitution. The block pass uses this full description; the definition pass uses a one/two-sentence
# squash of it (SHORT_SUMMARY_INSTRUCTION, scale-cfg/summary.short.txt) to spend its scarce context on the body.
SUMMARY_INSTRUCTION = (
    "Describe what this {language} source file is for, as the file-level overview a developer would read at the top "
    "of the file. Lead with the file's role in the wider program, then the main things it defines or the key "
    "operations it provides, grouping related functionality rather than listing every function. Stay grounded in the "
    "code: do not invent APIs or behaviour, and do not pad with generic remarks about logging, debugging, tracing, "
    "error handling, or the headers it includes unless that is genuinely the point of the file. Do not comment on how "
    "well-organised, clean, or readable the code is - describe what it does, not its quality.\n\n"
    "Write two or three short paragraphs of FLOWING PROSE only. No bulleted, dashed, or numbered lists; no section "
    "headings (e.g. \"Key operations:\"); no comment markers. Keep it tight.{seed}"
)

SHORT_SUMMARY_INSTRUCTION = (
    "In one or two sentences, say what this {language} source file is for and the kind of code it contains - quick "
    "context for a reader, not a full description. Plain prose, no list, no preamble."
)


# File suffixes treated as headers (a public-interface declaration file rather than an implementation). Knowing a file
# is a header changes how its documentation should read - the external contract callers rely on, not internal detail -
# so the file's identity is injected into the priming context of the summary, definition, and block passes.
_HEADER_SUFFIXES = {".h", ".hpp", ".hh", ".hxx", ".h++", ".hp"}

# `--block-comments` density -> the 1-3 value threshold a paragraph comment must clear to be written. 'high' keeps all
# (threshold 1), 'medium' drops bare restatements (2), 'low' keeps only intent/gotcha notes (3). Spacing-only (no
# --block-comments) uses 4, which no 1-3 score can clear, so the comment turns are skipped entirely.
BLOCK_COMMENT_LEVELS = {"high": 1, "medium": 2, "low": 3}


def _file_identity_note(src_path: Path, language: str) -> str:
    """
    Build the prompt sentence identifying the file being documented.

    Header files get an extended steer: their documentation should describe the caller-facing contract rather than implementation detail.

    Parameters:
    - `src_path`: Path of the source file being documented.
    - `language`: Language identifier for the file (e.g. `"c"`).

    Returns:
    - A sentence naming the file, ready for inclusion in a prompt.
    """

    name = src_path.name

    # Headers earn a longer note steering documentation towards the caller-facing contract rather than internals.
    if src_path.suffix.lower() in _HEADER_SUFFIXES:
        return (f"The file being documented is `{name}`, a header file: it declares a module's public interface, so "
                f"its documentation should describe the external contract a caller relies on - each function's "
                f"purpose, its parameters and its return value - rather than internal implementation detail.")

    if language == "c":
        return f"The file being documented is `{name}`, a C implementation file."
    return f"The file being documented is `{name}`."


def _file_role(src_path: Path, language: str) -> str:
    """
    Classify a source file as a header, a C implementation file, or other.

    Parameters:
    - `src_path`: Path of the source file.
    - `language`: Language identifier for the file.

    Returns:
    - `"header"`, `"implementation"` or `"other"`.
    """

    # Only C files are split into header versus implementation; every other language counts as `other`.
    if src_path.suffix.lower() in _HEADER_SUFFIXES:
        return "header"
    if language == "c":
        return "implementation"
    return "other"


# ---------------------------- CLI harness ----------------------------


def load_source(src_path: Path, language: Optional[str] = None) -> Tuple[str, Chunk, str, str]:
    """
    Load a source file binary-safely and determine its line ending and language.

    The file is read as bytes and decoded with UTF-8 `surrogateescape`, so undecodable content survives a later write-back unchanged. The dominant line ending is chosen by counting, and when no language is supplied it is guessed from the content, never the file extension. Exits the process if the file does not exist.

    Parameters:
    - `src_path`: Path of the source file to load.
    - `language`: Optional language override; `None` or empty triggers content-based guessing.

    Returns:
    - A tuple of the full source blob, the list of source lines, the detected line-ending string and the resolved language.
    """

    # Content-based heuristic: the language is judged from the lines themselves, never the file extension.
    def guess_language(source_lines: List[str]) -> str:
        """
        Guess the programming language from source content alone.

        A recognised shebang on the first line is decisive; otherwise every non-blank line votes for the languages whose idioms it matches, with weights reflecting how distinctive each cue is, and the highest total wins. Falls back to `"text"` when nothing scores.

        Parameters:
        - `source_lines`: The source split into individual lines.

        Returns:
        - A language identifier such as `"python"`, `"c"` or `"js"`, or `"text"` when undetermined.
        """

        if not source_lines:
            return "text"
        first = source_lines[0].strip()

        # A recognised shebang is decisive and short-circuits the scoring below.
        if first.startswith("#!"):
            sh = first.lower()
            if "python" in sh:
                return "python"
            if "bash" in sh or sh.endswith("/sh"):
                return "sh"
            if "node" in sh:
                return "js"

        # Seeding `text` at zero guarantees the max() below always has a winner, even when nothing matches.
        stripped = [s.strip() for s in source_lines if s.strip()]
        scores = defaultdict(int)
        scores["text"] = 0

        # Each line votes for the languages whose idioms it matches; weights reflect how distinctive a cue is.
        for line in stripped:
            last_char = line[-1]
            uline = line.upper()
            if line.startswith(("#include", "#define ")):
                scores["c"] += 2

            # Excluding `public`/`final` stops Java declarations masquerading as C.
            if line.startswith(("extern", "static")) and last_char == ";" and not any(tok in line for tok in ["public", "final"]):
                scores["c"] += 2
                scores["cpp"] += 2  # also common in C++

            # Distinctive C++, Python and JavaScript markers.
            if "using namespace" in line or line.startswith("template<"):
                scores["cpp"] += 3
            if line.startswith(("public:", "private:", "protected:")):
                scores["cpp"] += 2
            if last_char == ":" and line.startswith(("def ", "class ")):
                scores["python"] += 3
            if last_char != ";" and line.startswith(("import ", "from ")):
                scores["python"] += 2
            if line.startswith(("function ", "export ", "const ", "let ", "var ")):
                scores["js"] += 2
            if line.startswith("import ") and " from " in line and last_char == ";":
                scores["js"] += 2
            if any(tok in line for tok in ("document.", "window.", "console.", "JSON.", "=>")):
                scores["js"] += 3

            # A brace-style class line is ambiguous, so both candidates get the same small boost.
            if last_char == "{" and "class " in line:
                scores["js"] += 1
                scores["java"] += 1  # also common in Java

            if line.startswith("package "):
                scores["java"] += 3
                scores["go"] += 2  # also boost Go (both use package)

            # Java, Go, shell and Visual Basic markers.
            if line.startswith(("import java.", "import javax.")):
                scores["java"] += 3
            if line.startswith("public class "):
                scores["java"] += 3
            if "System.out." in line or "public static void main" in line:
                scores["java"] += 3
            if line.startswith(("import (", "func ")):
                scores["go"] += 3
            if "fmt." in line or line.startswith("go "):
                scores["go"] += 2
            if line.startswith("echo ") or line in ("fi", "done", "esac"):
                scores["sh"] += 2
            if uline.startswith(("SUB ", "FUNCTION ", "DIM ", "PRINT ")):
                scores["vb"] += 3
            if uline.startswith(("MODULE ", "IMPORTS ", "PUBLIC CLASS ")):
                scores["vb"] += 2

        # On a tie the language scored earliest wins, since max() keeps the first maximum.
        best = max(scores.items(), key=lambda kv: kv[1])

        return best[0]

    echo(f"Loading source file '{str(src_path)}'...")

    if not src_path.is_file():
        echo(f"Error: file not found: {src_path}")
        sys.exit(1)

    # Binary-safe read: surrogateescape preserves undecodable bytes, and the \r\n count is subtracted so bare \r and \n are tallied separately.
    raw = src_path.read_bytes()
    source_blob = raw.decode("utf-8", errors="surrogateescape")
    count_rn = source_blob.count("\r\n")
    count_r = source_blob.count("\r") - count_rn  # bare \r not part of \r\n
    count_n = source_blob.count("\n") - count_rn  # bare \n not part of \r\n

    # The majority line ending wins, so the file is written back in its dominant convention.
    if count_rn > max(count_r, count_n):
        line_ending = "\r\n"
    else:
        line_ending = "\r" if count_r > count_n else "\n"

    # An explicitly supplied language always takes precedence; guessing only fills the gap.
    source_lines = source_blob.split(line_ending)
    if language is None or language == "":
        language = guess_language(source_lines)
    echo(f"Language set to '{language}'...")

    return source_blob, source_lines, line_ending, language


class SummaryCache:
    """
    Disk-backed cache of per-file summaries, invalidated by content hash.

    Each source path maps, via a pickled index, to a stable UID under `__cache__/`, where the full summary, the squashed short summary and a SHA-256 of the source are stored as separate files. Cached text is only honoured while the stored hash matches the current source, and every write is atomic (temp file then rename) so a crash never leaves a torn entry.
    """

    # The cache lives beside the tool itself, with a pickled index mapping source paths to stable UIDs.
    _CACHE_DIR = (Path(__file__).resolve().parent) / "__cache__"
    _CACHE_INDEX = _CACHE_DIR / "index.pkl"

    def __init__(self, source_path: Path, source_blob: str) -> None:
        """
        Bind this cache entry to a source file and load any still-valid summaries.

        Looks the path up in the on-disk index, allocating and persisting a fresh UID on first sight, then loads the cached full and short summaries only if the stored content hash matches the current source; otherwise both start empty.

        Parameters:
        - `source_path`: Path of the source file this entry belongs to.
        - `source_blob`: The file's decoded content, hashed for invalidation.
        """

        # Hash the surrogateescape-encoded blob so the digest reflects the file's original bytes.
        self._summary: Optional[str] = None
        self._short: Optional[str] = None
        self._hash = hashlib.sha256(source_blob.encode("utf-8", errors="surrogateescape")).hexdigest()
        index = self._load_index()
        key = str(source_path)
        uid = index.get(key)

        # First sight of this path: mint a UID and persist the index mapping straight away.
        if uid is None:
            uid = uuid.uuid4().hex
            index[key] = uid
            self._save_index(index)

        self._uid = uid
        self._data_path = self._CACHE_DIR / f"{self._uid}.txt"        # the full (description) summary
        self._short_path = self._CACHE_DIR / f"{self._uid}.short.txt"  # the squashed summary for the definition pass
        self._hash_path = self._CACHE_DIR / f"{self._uid}.sha256"     # content hash for invalidation

        try:
            cached_hash = self._hash_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            cached_hash = None

        # Summaries are only trusted while the source is unchanged; any edit silently invalidates both.
        if cached_hash == self._hash:
            self._summary = self._read_optional_text(self._data_path)
            self._short = self._read_optional_text(self._short_path)
        else:
            self._summary = None
            self._short = None

    @property
    def summary(self) -> Optional[str]:
        """
        Return the cached full summary, or `None` when absent or stale.

        Returns:
        - The full description summary text, or `None` if nothing valid is cached.
        """

        return self._summary

    @summary.setter
    def summary(self, text: str) -> None:
        """
        Set the full summary and persist it to the cache.

        The content hash is written alongside the text, stamping the entry as valid for the current source.

        Parameters:
        - `text`: The full summary text to cache.
        """

        # Writing the hash file alongside the text stamps the entry as valid for the current source.
        self._summary = text
        self._atomic_write_bytes(
            self._data_path,
            text.encode("utf-8", errors="surrogateescape"),
        )
        self._atomic_write_bytes(self._hash_path, self._hash.encode("utf-8"))

    @property
    def short(self) -> Optional[str]:
        """
        Return the cached short summary, or `None` if none has been loaded or generated yet.

        Returns:
        - The short summary string, or `None` when absent.
        """

        return self._short

    @short.setter
    def short(self, text: str) -> None:
        """
        Set the short summary and persist it to disk.

        The source hash is written alongside the summary so the on-disk entry stays bound to the exact source it was generated from.

        Parameters:
        - `text`: The new short summary text.
        """

        # Persist the source hash alongside the summary so the on-disk entry stays bound to its source.
        self._short = text
        self._atomic_write_bytes(self._short_path, text.encode("utf-8", errors="surrogateescape"))
        self._atomic_write_bytes(self._hash_path, self._hash.encode("utf-8"))

    @staticmethod
    def _read_optional_text(path: Path) -> Optional[str]:
        """
        Read a file as UTF-8 text, returning `None` if it does not exist.

        Decoding uses `surrogateescape` so undecodable bytes survive a round trip with the binary-safe writer.

        Parameters:
        - `path`: The file to read.

        Returns:
        - The decoded file contents, or `None` when the file is missing.
        """

        # A missing file is an expected state, not an error; surrogateescape lets undecodable bytes round-trip.
        try:
            return path.read_bytes().decode("utf-8", errors="surrogateescape")
        except FileNotFoundError:
            return None

    @classmethod
    def _load_index(cls) -> dict[str, str]:
        """
        Load the on-disk summary-cache index.

        Any failure (missing file, unpicklable data, or a non-dict payload) degrades to an empty index, so a corrupt cache rebuilds itself rather than aborting the run.

        Returns:
        - The mapping read from the index file, or an empty dict on any failure.
        """

        try:
            with cls._CACHE_INDEX.open("rb") as f:
                # Any unreadable or non-dict index degrades to empty, so a corrupt cache rebuilds rather than aborting.
                obj = pickle.load(f)
                return obj if isinstance(obj, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception:
            return {}

    @classmethod
    def _save_index(cls, index: dict[str, str]) -> None:
        """
        Persist the summary-cache index to disk.

        The index is written to a temporary sibling and renamed into place, so a crash mid-write never leaves a truncated index behind.

        Parameters:
        - `index`: The mapping of cache keys to entries to persist.
        """

        # Write to a temporary sibling then rename, so a crash never leaves a truncated index behind.
        cls._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cls._CACHE_INDEX.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cls._CACHE_INDEX)

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        """
        Write bytes to a file atomically.

        Data goes to a temporary sibling first and is renamed into place, so readers never observe a partially written file. Parent directories are created as needed.

        Parameters:
        - `path`: The destination file path.
        - `data`: The raw bytes to write.
        """

        # Write-then-rename keeps the update atomic; readers never see partial content.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            f.write(data)
        tmp.replace(path)


def _hard_split_line(line: str, chunk_budget: int, estimate_fn: Callable[[str], int]) -> List[str]:
    """
    Split a single over-budget line into pieces that each fit the chunk token budget.

    The characters-per-token ratio is estimated from the line itself, and each piece is sized with a 10% safety margin so an optimistic estimate cannot overflow the budget.

    Parameters:
    - `line`: The line to split.
    - `chunk_budget`: The maximum token cost allowed per piece.
    - `estimate_fn`: Callable that estimates the token count of a string.

    Returns:
    - The list of pieces in order; an empty line yields a single-element list.
    """

    # Estimate the chars-per-token ratio from the line itself, sizing pieces with a 10% safety margin.
    if not line:
        return [line]
    per_token = max(1, len(line) // max(1, estimate_fn(line)))
    piece_chars = max(1, int(chunk_budget * per_token * 0.9))

    return [line[i:i + piece_chars] for i in range(0, len(line), piece_chars)]


def _split_source(source_blob: str, chunk_budget: int, estimate_fn: Callable[[str], int]) -> List[str]:
    """
    Split source text into chunks that each fit within a token budget.

    Lines are packed greedily into the current chunk. A line that alone exceeds the budget is hard-split into fitting pieces, and a blank line ends a chunk early once it is three-quarters full so breaks tend to fall on paragraph boundaries.

    Parameters:
    - `source_blob`: The full source text to split.
    - `chunk_budget`: The maximum estimated token cost per chunk.
    - `estimate_fn`: Callable that estimates the token count of a string.

    Returns:
    - The list of chunk strings, in source order.
    """

    # State for the greedy packer: the chunk under construction and its running token estimate.
    lines = source_blob.split("\n")
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    # Close out the in-progress chunk and reset the accumulator.
    def flush() -> None:
        """
        Append the buffered lines to `chunks` as a single entry and reset the running accumulator.

        Safe to call with an empty buffer: it then does nothing, so flushing at the end of the loop never emits a blank chunk.
        """

        # The reset below must rebind the enclosing accumulators, not create new locals.
        nonlocal current, current_tokens

        # A no-op on an empty buffer, so a trailing flush never emits a blank chunk.
        if current:
            chunks.append("\n".join(current))
            current = []
            current_tokens = 0

    for line in lines:
        cost = estimate_fn(line) + 1  # +1 approximates the joining newline

        # A line that alone exceeds the budget is flushed past and hard-split into fitting pieces.
        if cost > chunk_budget:
            flush()
            chunks.extend(_hard_split_line(line, chunk_budget, estimate_fn))
            continue

        # Flush before overflowing; a blank line once the chunk is 75% full also ends it, so breaks favour paragraph boundaries.
        if current and current_tokens + cost > chunk_budget:
            flush()
        current.append(line)
        current_tokens += cost
        if not line.strip() and current_tokens >= chunk_budget * 0.75:
            flush()

    # Emit any trailing partial chunk.
    flush()

    return chunks


def _group_by_budget(partials: List[str], budget: int, estimate_fn: Callable[[str], int]) -> List[List[str]]:
    """
    Pack consecutive part-summaries into greedy, token-budgeted groups for hierarchical reduction.

    Every group keeps at least two parts even when that overruns the budget, and a lone full-size group is force-split in half, so each reduction round strictly shrinks the list and the caller's recursion terminates.

    Parameters:
    - `partials`: The ordered part-summaries to pack.
    - `budget`: Approximate token allowance for one group's combined text.
    - `estimate_fn`: Callable returning a token estimate for a string.

    Returns:
    - A list of groups, each a list of consecutive summaries, covering `partials` in order.
    """

    groups: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0

    for s in partials:
        cost = estimate_fn(s) + 8  # small allowance for the "Part N:" framing

        # Never close a group below two parts: each merge round must strictly shrink the list, even if one part alone busts the budget.
        if len(current) >= 2 and current_tokens + cost > budget:
            groups.append(current)
            current = []
            current_tokens = 0

        current.append(s)
        current_tokens += cost

    if current:
        groups.append(current)

    # If everything fitted into one group the caller would recurse on identical input, so force a split to guarantee progress.
    if len(groups) == 1 and len(partials) > 1:
        mid = len(partials) // 2
        groups = [partials[:mid], partials[mid:]]

    return groups


def _reduce_summaries(
    llm: LocalChatModel,
    summary_cfg: GenerationConfig,
    base_messages: Messages,
    partials: List[str],
    language: str,
    limit: int,
    base_overhead: int,
) -> str:
    """
    Merge per-chunk summaries into one overall summary that fits the model's context window.

    If the labelled parts fit under `limit` they are merged in a single summarise turn; otherwise they are packed into token-budgeted groups, each group is collapsed to one summary, and the function recurses on the shorter list. `_group_by_budget` guarantees each round shrinks the list, so the recursion terminates.

    Parameters:
    - `llm`: The loaded local chat model.
    - `summary_cfg`: Generation settings for the summarise turns.
    - `base_messages`: Priming messages prepended to each turn.
    - `partials`: Per-chunk summaries, in file order.
    - `language`: Source language name, used in the summarise subject.
    - `limit`: Token ceiling for one summarise input.
    - `base_overhead`: Token cost of `base_messages`, counted once by the caller.

    Returns:
    - The single merged summary string.
    """

    # Recursion base case; the `Part N` labels keep the original file order visible to the model.
    if len(partials) == 1:
        return partials[0]
    subject = f"summaries of consecutive parts of one {language} source file"
    combined = "\n\n".join(f"Part {i}: {s}" for i, s in enumerate(partials, start=1))

    # Fast path: everything fits, so merge in a single summarise turn.
    if base_overhead + llm.estimate_tokens(combined) + SUMMARY_WRAPPER_TOKENS <= limit:
        return summarise(llm, summary_cfg, combined, LENGTH_PARAGRAPHS,
                         base_messages=base_messages, subject=subject, max_tokens=SUMMARY_MAX_TOKENS)

    # Over budget: pack the parts into groups, keeping 64 tokens of slack for the summarise framing.
    groups = _group_by_budget(partials, max(1, limit - base_overhead - 64), llm.estimate_tokens)
    reduced: List[str] = []

    # Collapse each group to one summary; the grouping rules guarantee the list shrinks.
    for group in groups:
        sub = "\n\n".join(f"Part {i}: {s}" for i, s in enumerate(group, start=1))
        reduced.append(summarise(llm, summary_cfg, sub, LENGTH_PARAGRAPHS,
                                 base_messages=base_messages, subject=subject, max_tokens=SUMMARY_MAX_TOKENS))

    # Recurse until the merged summaries fit a single pass.
    return _reduce_summaries(llm, summary_cfg, base_messages, reduced, language, limit, base_overhead)


def _fill_summary_instruction(desc_spec: Optional[str], language: str, seed: Optional[str]) -> str:
    """
    Build the file-description instruction by filling the `{language}` and `{seed}` placeholders.

    Parameters:
    - `desc_spec`: Optional caller-supplied template; falls back to `SUMMARY_INSTRUCTION` when `None`.
    - `language`: Source language name substituted for `{language}`.
    - `seed`: Existing file description, if any; when present the instruction asks the model to keep its still-accurate wording.

    Returns:
    - The completed instruction string.
    """

    # An empty clause makes the `{seed}` placeholder vanish cleanly when there is no existing description.
    template = desc_spec if desc_spec is not None else SUMMARY_INSTRUCTION
    seed_clause = ""

    # Ingest-and-update: ask the model to keep the still-accurate wording of an existing description rather than rewrite from scratch.
    if seed and seed.strip():
        seed_clause = (
            f' The file already carries this description: "{seed.strip()}". Keep any wording that is still accurate '
            f"and correct or extend the rest."
        )

    # Plain `replace` rather than `str.format`, so stray braces elsewhere in the template are harmless.
    return template.replace("{language}", language).replace("{seed}", seed_clause)


# A line that opens with a list/heading marker: numbered ("1." / "2)"), bulleted ("- "/"* "/"+ "/"• "), a markdown
# heading ("## "), or a bold label ("**Key operations**:"). The file-DESCRIPTION spec asks for flowing prose, but a
# small model summarising a large file via map-reduce sometimes returns a structured list anyway - this catches it.
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:\d+[.)]\s|[-*+•]\s|#{1,6}\s|\*\*[^*\n]+\*\*\s*:)")


def _looks_listy(text: str) -> bool:
    """
    Heuristically detect whether text is formatted as a list rather than flowing prose.

    Parameters:
    - `text`: The candidate description; `None` is treated as empty.

    Returns:
    - `True` if two or more list or heading markers are present, otherwise `False`.
    """

    # Two markers minimum: a single match may be incidental prose rather than list formatting.
    return len(_LIST_MARKER_RE.findall(text or "")) >= 2


def _strip_list_markers(text: str) -> str:
    """
    Strip leading list/heading markers and bold emphasis from each line of the text.

    Deterministic fallback for when the reflow turn still returns list-formatted text: only the markers go, the wording and line order are preserved.

    Parameters:
    - `text`: The text to clean; `None` is treated as empty.

    Returns:
    - The cleaned text with its lines rejoined by newlines.
    """

    out: List[str] = []

    # Only the markers go; each line's wording is kept, so no information is lost.
    for ln in (text or "").split("\n"):
        s = re.sub(r"^\s*(?:\d+[.)]|[-*+•]|#{1,6})\s+", "", ln)   # leading number/bullet/heading marker
        s = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", s)                      # **bold** -> bold
        out.append(s)

    return "\n".join(out)


def _reflow_if_listy(llm: LocalChatModel, cfg: GenerationConfig, base_messages: Messages, text: str,
                     max_tokens: int) -> str:
    """
    Rewrite a list-formatted file description as flowing prose, with a deterministic fallback.

    Text that does not look like a list passes through unchanged. Otherwise one corrective LLM turn asks for prose; if the rewrite is empty or still looks like a list, the markers are stripped instead, so the result is never worse than the input.

    Parameters:
    - `llm`: The loaded local chat model.
    - `cfg`: Base generation settings; `max_new_tokens` is overridden with `max_tokens` for the turn.
    - `base_messages`: Priming messages prepended to the rewrite turn.
    - `text`: The candidate description.
    - `max_tokens`: Token cap for the rewritten description.

    Returns:
    - Flowing prose: the original text, the rewrite, or a marker-stripped fallback.
    """

    # One corrective turn, then a deterministic fallback: if the rewrite still looks like a list, just strip the markers.
    if not _looks_listy(text):
        return text
    echo("File description came back as a list; asking for flowing prose...")
    prompt = (
        "The following file description uses lists, numbering, headings, or bold markers. Rewrite it as two or three "
        "short paragraphs of flowing prose that preserve the information but contain NO numbered or bulleted lists, NO "
        "headings, and no bold or asterisk markers. Give only the rewritten description.\n\n" + text
    )
    turn_cfg = replace(cfg, max_new_tokens=max_tokens)
    reflowed = llm.generate((base_messages or []) + [{"role": "user", "content": prompt}], cfg=turn_cfg).strip()
    if reflowed and not _looks_listy(reflowed):
        return reflowed
    return _strip_list_markers(reflowed or text)


def _generate_file_summary(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    base_messages: Messages,
    source_blob: str,
    language: str,
    desc_spec: Optional[str] = None,
    seed: Optional[str] = None,
    skeleton: bool = False,
    capture: Optional[dict] = None,
) -> str:
    """
    Produce a prose description of a source file, chunking and reducing when it exceeds the context window.

    Small inputs are summarised in one pass. Larger ones are split into budget-sized chunks, summarised chunk by chunk, hierarchically merged, then condensed under the real description instruction; both paths finish with a reflow guard so list-formatted replies never escape.

    Parameters:
    - `llm`: The loaded local chat model.
    - `cfg`: Base generation settings; `max_new_tokens` is overridden for summary turns.
    - `base_messages`: Priming messages prepended to every turn.
    - `source_blob`: The source text to describe.
    - `language`: Source language name, used in prompts.
    - `desc_spec`: Optional instruction template overriding the stock one.
    - `seed`: Existing file description to preserve and update, if any.
    - `skeleton`: `True` when `source_blob` is a structural skeleton with the function bodies omitted.
    - `capture`: Optional dict; on the chunked path the merged thorough draft is stored under `"thorough"`.

    Returns:
    - The file description as flowing prose.
    """

    # Budget setup; the subject wording also tells the model when it is reading a skeleton rather than full source.
    summary_cfg = replace(cfg, max_new_tokens=SUMMARY_MAX_TOKENS)
    limit = llm.n_ctx - llm.ctx_margin - SUMMARY_MAX_TOKENS
    base_overhead = llm.count_tokens(base_messages) if base_messages else 0
    description_instruction = _fill_summary_instruction(desc_spec, language, seed)
    what = (f"a structural skeleton of a {language} source file - its header comments, signatures, and existing "
            f"documentation, with the function bodies omitted") if skeleton else f"a complete {language} source file"

    # Single-pass path when the whole file fits the context window.
    if base_overhead + llm.estimate_tokens(source_blob) + SUMMARY_WRAPPER_TOKENS <= limit:
        # The reflow guard stops list-formatted replies escaping as the file description.
        result = summarise(llm, summary_cfg, source_blob, LENGTH_PARAGRAPHS, base_messages=base_messages,
                           subject=what, max_tokens=SUMMARY_MAX_TOKENS,
                           instruction=description_instruction)
        return _reflow_if_listy(llm, summary_cfg, base_messages, result, SUMMARY_MAX_TOKENS)

    # Too large for one pass: split the source into budget-sized chunks for map-reduce summarising.
    chunk_budget = max(1, limit - base_overhead - 64)
    chunks = _split_source(source_blob, chunk_budget, llm.estimate_tokens)
    echo(f"Source too large for a single-pass summary; summarising in {len(chunks)} chunk(s)...")
    partials: List[str] = []

    # Map step: one paragraph per chunk, kept in file order.
    for idx, chunk in enumerate(chunks, start=1):
        partials.append(summarise(llm, summary_cfg, chunk, LENGTH_PARAGRAPH, base_messages=base_messages,
                                  subject=f"chunk {idx} of {len(chunks)} of {what}",
                                  max_tokens=SUMMARY_MAX_TOKENS))
        echo(f"Summarised part {idx}/{len(chunks)}")

    # Reduce, expose the thorough draft to the caller, then condense under the real description instruction.
    overall = _reduce_summaries(llm, summary_cfg, base_messages, partials, language, limit, base_overhead)
    if capture is not None:
        capture["thorough"] = overall
    result = summarise(llm, summary_cfg, overall, LENGTH_PARAGRAPHS, base_messages=base_messages,
                       subject=f"a draft overview of a {language} source file", max_tokens=SUMMARY_MAX_TOKENS,
                       instruction=description_instruction)

    return _reflow_if_listy(llm, summary_cfg, base_messages, result, SUMMARY_MAX_TOKENS)


def _head_crop(text: str, llm: LocalChatModel, budget_tokens: int) -> str:
    """
    Crop text to a token budget by keeping only whole leading lines.

    The text is rebuilt a line at a time until the token estimate would exceed the budget; the line that tips it over is dropped, so the result never splits a line and always fits.

    Parameters:
    - `text`: The text to crop.
    - `llm`: The model whose tokeniser provides the token estimates.
    - `budget_tokens`: The maximum number of tokens the result may occupy.

    Returns:
    - The text unchanged if it already fits, otherwise its longest whole-line prefix within the budget.
    """

    # Nothing to crop when the whole text already fits the budget.
    if llm.estimate_tokens(text) <= budget_tokens:
        return text
    kept: List[str] = []

    # Grow the crop one whole line at a time so the result never splits a line mid-way.
    for line in text.splitlines():
        kept.append(line)

        # Drop the line that tipped the estimate over the budget, then stop - the kept prefix always fits.
        if llm.estimate_tokens("\n".join(kept)) > budget_tokens:
            kept.pop()
            break

    return "\n".join(kept)


def _get_file_summary(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    scale_path: Path,
    src_path: Path,
    source_blob: str,
    language: str,
    base_messages: Messages,
    no_cache: Optional[bool] = False,
    seed: Optional[str] = None,
    skeleton: bool = False,
    capture: Optional[dict] = None,
) -> str:
    """
    Return the full description of a source file, generating and caching it on a miss.

    The summary cache is keyed on the source content, so a stale entry can never be reused after an edit. On a miss (or when caching is disabled) a fresh summary is generated, with an optional `summary.txt` in the config directory overriding the built-in instruction.

    Parameters:
    - `llm`: The local chat model used to generate the summary.
    - `cfg`: Generation settings for the model.
    - `scale_path`: Directory holding the prompt override files.
    - `src_path`: Path of the source file being summarised.
    - `source_blob`: The full source text (or its skeleton) to summarise.
    - `language`: Name of the source language.
    - `base_messages`: Priming messages the summary request builds on.
    - `no_cache`: When `True`, ignore any cached summary and regenerate.
    - `seed`: Optional existing description used to seed the new summary.
    - `skeleton`: `True` when `source_blob` is a structural skeleton rather than full source.
    - `capture`: Optional dictionary that collects intermediate generation details.

    Returns:
    - The full summary text, freshly generated or straight from the cache.
    """

    # The cache is keyed on the source content, so any edit invalidates it automatically.
    summary_cache = SummaryCache(src_path, source_blob)

    # Generate only on a miss (or when caching is off); an optional summary.txt overrides the built-in instruction.
    if no_cache is False and summary_cache.summary:
        echo("Loaded full source summary from cache...")
    else:
        echo("Generating full source summary..." + (" (from the file skeleton)" if skeleton else ""))
        desc_spec = _read_optional(scale_path / "summary.txt") or SUMMARY_INSTRUCTION
        summary_cache.summary = _generate_file_summary(
            llm, cfg, base_messages, source_blob, language, desc_spec=desc_spec, seed=seed, skeleton=skeleton,
            capture=capture)

    return summary_cache.summary


def _get_short_summary(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    scale_path: Path,
    src_path: Path,
    source_blob: str,
    language: str,
    base_messages: Messages,
    no_cache: Optional[bool] = False,
    skeleton: bool = False,
) -> str:
    """
    Return a one-line description of a source file, generating and caching it on a miss.

    When the full summary is already cached it is condensed rather than re-reading the source; otherwise the source itself - head-cropped to the context budget left after the priming turns - is summarised directly. An optional `summary.short.txt` overrides the built-in instruction.

    Parameters:
    - `llm`: The local chat model used to generate the summary.
    - `cfg`: Generation settings for the model.
    - `scale_path`: Directory holding the prompt override files.
    - `src_path`: Path of the source file being summarised.
    - `source_blob`: The full source text (or its skeleton) to summarise.
    - `language`: Name of the source language.
    - `base_messages`: Priming messages the summary request builds on.
    - `no_cache`: When `True`, ignore the cache and regenerate.
    - `skeleton`: `True` when `source_blob` is a structural skeleton rather than full source.

    Returns:
    - The one-line summary text, freshly generated or straight from the cache.
    """

    # Short summaries share the content-keyed cache with the full ones.
    cache = SummaryCache(src_path, source_blob)

    # A cache hit skips the model entirely.
    if no_cache is False and cache.short:
        echo("Loaded short source summary from cache...")
        return cache.short

    # An optional summary.short.txt overrides the built-in instruction; the reduced token cap holds the reply to a single line.
    instruction = (_read_optional(scale_path / "summary.short.txt") or SHORT_SUMMARY_INSTRUCTION).replace(
        "{language}", language)
    short_cfg = replace(cfg, max_new_tokens=SHORT_SUMMARY_MAX_TOKENS)
    full = cache.summary if (no_cache is False and cache.summary) else None

    # Condense the cached full summary when one exists; otherwise summarise the source directly, head-cropped to whatever context room is left after the priming turns and the reply.
    if full:
        echo("Condensing the file description for the definition pass...")
        short = summarise(llm, short_cfg, full, LENGTH_LINE, base_messages=base_messages,
                          subject=f"a fuller description of a {language} source file",
                          max_tokens=SHORT_SUMMARY_MAX_TOKENS, instruction=instruction)
    else:
        echo("Summarising the source directly into a one-line description...")
        what = (f"a structural skeleton (signatures and docs, no bodies) of a {language} source file"
                if skeleton else f"a {language} source file")
        budget = max(256, llm.n_ctx - llm.ctx_margin - SHORT_SUMMARY_MAX_TOKENS
                     - llm.count_tokens(base_messages) - SUMMARY_WRAPPER_TOKENS)
        short = summarise(llm, short_cfg, _head_crop(source_blob, llm, budget), LENGTH_LINE,
                          base_messages=base_messages, subject=what,
                          max_tokens=SHORT_SUMMARY_MAX_TOKENS, instruction=instruction)

    cache.short = short

    return short


def prime_llm_for_comments(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    scale_path: Path,
    src_path: Path,
    source_blob: str,
    language: str,
    no_cache: Optional[bool] = False,
    template: str = "comment",
    project_context: str = "",
    skeleton: Optional[str] = None,
) -> Messages:
    """
    Build the priming message history that precedes every generation turn for a file.

    The history is a sequence of acknowledged turns: the system commenting brief (plus optional house-style guidelines for the comment template), any wider-project background, the file's identity, a summary of the file, and finally the per-language task template. The comment template uses the cheap one-line summary; every other template gets the full description.

    Parameters:
    - `llm`: The local chat model being primed.
    - `cfg`: Generation settings for the model.
    - `scale_path`: Directory holding the prompt and template files.
    - `src_path`: Path of the source file being annotated.
    - `source_blob`: The full source text of the file.
    - `language`: Name of the source language, used to pick the template.
    - `no_cache`: When `True`, regenerate summaries instead of using the cache.
    - `template`: Which task template to load (`comment` by default).
    - `project_context`: Optional background text on the wider project.
    - `skeleton`: Optional structural skeleton used in place of the source when summarising.

    Returns:
    - The list of priming messages, ready for generation turns to be appended.
    """

    # Assemble the system prompt: the core commenting brief plus, for the comment template, the optional house-style guidelines.
    echo("Priming LLM...")
    comment_path = scale_path / "comment.txt"
    comment_prompt = comment_path.read_text(encoding="utf-8")
    guidelines_path = scale_path / "guidelines.md"
    if template == "comment" and guidelines_path.is_file():
        comment_prompt = f"{comment_prompt}\n\n{guidelines_path.read_text(encoding='utf-8')}"
    template_path = scale_path / f"{template}.{language}.txt"
    template_prompt = template_path.read_text(encoding="utf-8")
    messages = []
    messages.append({"role": "system", "content": comment_prompt})

    # Wider-project background goes in first so every later turn can lean on it.
    if project_context:
        messages.append({"role": "user", "content":
            "Here is some background on the wider project this file belongs to:\n\n" + project_context})
        messages.append({"role": "assistant", "content": PRIMING_ACK})

    # Name the file being worked on; a supplied skeleton stands in for the full source when summarising.
    messages.append({"role": "user", "content": _file_identity_note(src_path, language)})
    messages.append({"role": "assistant", "content": PRIMING_ACK})
    summary_source = skeleton if skeleton else source_blob

    # The definition pass only needs a one-line description; every other template gets the full summary.
    if template == "comment":
        summary = _get_short_summary(llm, cfg, scale_path, src_path, summary_source, language, messages,
                                     no_cache=no_cache, skeleton=bool(skeleton))
    else:
        summary = _get_file_summary(llm, cfg, scale_path, src_path, summary_source, language, messages,
                                    no_cache=no_cache, skeleton=bool(skeleton))

    # Each priming turn is paired with a canned acknowledgement so the history reads as settled context before generation starts.
    reply_length = 1 + summary.count("\n")
    echo(f"Source file summarised? {reply_length} lines of summary created")
    echo(f"\n{summary}\n")
    messages.append({"role": "user", "content":
        "To give you context, here is an overview of what the program as a whole does:\n\n"
        f"{summary}"})
    messages.append({"role": "assistant", "content": PRIMING_ACK})
    messages.append({"role": "user", "content": template_prompt})
    messages.append({"role": "assistant", "content": PRIMING_ACK})

    return messages


def _def_pass(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    source_blob: str,
    source_lines: List[str],
    language: str,
    doc_order: Optional[List[str]] = None,
    callee_context: Optional[Callable[[str], str]] = None,
    on_doc: Optional[Callable[[str, str], None]] = None,
    doc_plan=None,
    verifier=None,
) -> List[str]:
    """
    Dispatch the definition pass to the worker for the given language.

    Every worker exposes the same `generate_language_comments` entry point; the C worker additionally accepts a doc-site plan, so it is invoked on a separate path. Workers are imported lazily so only the requested language's dependencies load.

    Parameters:
    - `llm`: The local chat model used for generation.
    - `cfg`: Generation settings for the model.
    - `messages`: The primed message history to generate against.
    - `source_blob`: The full source text of the file.
    - `source_lines`: The source split into lines, used for patching.
    - `language`: Name of the source language to dispatch on.
    - `doc_order`: Optional ordering of routine names to document.
    - `callee_context`: Optional callable mapping a routine name to context about its callees.
    - `on_doc`: Optional callback invoked with each routine's name and finished doc.
    - `doc_plan`: Doc-site placement plan, honoured by the C worker only.
    - `verifier`: Optional verification hook applied to generated docs.

    Returns:
    - The updated source lines with definition documentation applied.

    Notes:
    Raises `ValueError` for unsupported languages.
    """

    # Workers are imported lazily so only the requested language's dependencies load.
    if language == "python":
        from scale_python import generate_language_comments
    elif language == "js":
        from scale_javascript import generate_language_comments
    elif language == "c":
        # The C worker alone takes the doc-site plan, hence its early return; anything else is unsupported.
        from scale_c import generate_language_comments
        return generate_language_comments(llm, cfg, messages, source_blob, source_lines,
                                          doc_order=doc_order, callee_context=callee_context, on_doc=on_doc,
                                          doc_plan=doc_plan, verifier=verifier)
    else:
        raise ValueError(f"Unsupported language '{language}'")

    return generate_language_comments(llm, cfg, messages, source_blob, source_lines,
                                      doc_order=doc_order, callee_context=callee_context, on_doc=on_doc,
                                      verifier=verifier)


def _block_provider_for(language: str):
    """
    Return the block-target iterator for the given language.

    Workers are imported lazily so only the requested language's module loads. Unsupported languages raise rather than degrade: the block pass cannot run at all without a per-language segmenter.

    Parameters:
    - `language`: Name of the source language.

    Returns:
    - The worker's block-target iterator.

    Notes:
    Raises `NotImplementedError` for languages without block-pass support.
    """

    if language == "python":
        from scale_python import iter_block_targets
        return iter_block_targets

    if language == "c":
        from scale_c import iter_block_targets_c
        return iter_block_targets_c

    if language == "js":
        from scale_javascript import iter_block_targets_js
        return iter_block_targets_js

    # No graceful fallback here - the block pass cannot run without a per-language segmenter.
    raise NotImplementedError(
        f"The block pass does not yet support '{language}'."
    )


def _symbol_provider_for(language: str):
    """
    Return the symbol iterator for the given language, or `None` when unsupported.

    Unlike the block provider, this degrades gracefully: a `None` result lets callers skip symbol scanning rather than fail the run.

    Parameters:
    - `language`: Name of the source language.

    Returns:
    - The worker's `iter_symbols` function, or `None` for unsupported languages.
    """

    if language == "python":
        from scale_python import iter_symbols
        return iter_symbols

    if language == "c":
        from scale_c import iter_symbols
        return iter_symbols

    if language == "js":
        from scale_javascript import iter_symbols
        return iter_symbols

    # Returning None, rather than raising, lets callers simply skip symbol scanning for unsupported languages.
    return None


def _scan_run_files(targets: List[Path], references: List[Path], language_arg: Optional[str]):
    """
    Scan the run's target and reference files for the symbols they define.

    A thin adapter over `scale_project.scan_run_files` that supplies the language-aware loader and per-language symbol providers, keeping the project-level scanner itself language-agnostic.

    Parameters:
    - `targets`: Files the run will annotate.
    - `references`: Read-only files supplying context for the run.
    - `language_arg`: Optional language override applied when loading each file.

    Returns:
    - The scanned run-file records produced by `scale_project.scan_run_files`.
    """

    # Supplies the language-aware loader and symbol providers, keeping the project-level scanner language-agnostic.
    return scale_project.scan_run_files(
        targets, references,
        load=lambda p: load_source(p, language_arg),
        provider_for=_symbol_provider_for,
    )


def _build_call_graph(run_files: Dict[str, "scale_project.RunFile"]):
    """
    Build the project call graph and its contract store from the run files.

    Parameters:
    - `run_files`: Mapping of file key to `RunFile` records whose symbol tables seed the graph.

    Returns:
    - A `(graph, contract_store)` pair, or `(None, None)` when there are no run files.
    """

    # The graph sees only the extracted symbol tables, never the source text itself.
    if not run_files:
        return None, None
    graph = scale_project.build_project_graph({k: rf.symbols for k, rf in run_files.items()})

    # The contract store is created alongside the graph so callee one-liners have somewhere to accumulate during the run.
    return graph, scale_project.ContractStore(graph)


def _build_c_doc_plan(run_files: Dict[str, "scale_project.RunFile"], policy: str):
    """
    Plan the C header/implementation doc sites for the run.

    Parameters:
    - `run_files`: Mapping of file key to `RunFile` records; only C-language entries are considered.
    - `policy`: The doc-site placement policy handed through to `plan_doc_sites_c`.

    Returns:
    - The doc-site plan, or `None` when the run contains no C files.
    """

    # Only the run's C files take part; the import is deferred so non-C runs never load the C worker.
    files = [(rf.key, rf.is_target, rf.source_blob, rf.source_lines)
             for rf in run_files.values() if rf.language == "c"]
    if not files:
        return None
    from scale_c import plan_doc_sites_c

    return plan_doc_sites_c(files, policy)


def _make_callee_oneliner_context(llm: LocalChatModel, cfg: GenerationConfig, run_files, graph, store):
    """
    Create a lazy provider of callee one-liner context over the project call graph.

    The returned closure fills in missing callee contracts on demand: each undocumented callee is summarised by the local model at most once (fruitless attempts are remembered and never retried), and successful one-liners are written back to the contract store for reuse across the run.

    Parameters:
    - `llm`: The loaded local chat model used to summarise undocumented callees.
    - `cfg`: Generation settings for the summarisation turns.
    - `run_files`: Mapping of file key to `RunFile` records supplying callee source.
    - `graph`: The project call graph holding the symbol table.
    - `store`: The contract store that caches callee one-liners.

    Returns:
    - A `context(file_key, qualname)` callable yielding the callee notes for one routine.
    """

    # One shared memo across the whole run: a callee that yielded nothing is never summarised twice.
    attempted: set = set()

    # Distil a callee's (budget-trimmed) source into a single caller-facing line.
    def generate_oneliner(sym) -> str:
        """
        Summarise one call-graph symbol into a single caller-facing line.

        The symbol's source is sliced from its run file, trimmed to the token budget, and summarised by the local model from its caller's point of view.

        Parameters:
        - `sym`: The call-graph symbol to summarise.

        Returns:
        - A one-line description of the routine, or an empty string when no usable source is available.
        """

        # Slice the callee's body straight from its run file, giving up silently when there is nothing usable to summarise.
        rf = run_files.get(sym.file)
        if rf is None or sym.end < sym.start or sym.end <= 0:
            return ""
        snippet = "\n".join(rf.source_lines[sym.start - 1:sym.end])
        if not snippet.strip():
            return ""
        header_lines = max(1, len(sym.signature.split("\n")))

        # Trim the body to the token budget first: structural elision for Python, crude cropping for the rest.
        if rf.language == "python":
            from scale_python import elide_structurally
            snippet, _ = elide_structurally(llm, cfg, [], textwrap.dedent(snippet), header_lines, MARKER_PYTHON)
        else:
            marker = MARKER_C if rf.language == "c" else MARKER_JS
            snippet, _ = fit_snippet(llm, cfg, [], snippet, header_lines, marker)

        # The bare name reads better in the prompt than the full dotted qualname.
        simple = sym.qualname.rsplit(".", 1)[-1]

        # Frame the summary around what the routine does for its caller - the only view the documenting turn needs.
        return summarise(
            llm, cfg, snippet, LENGTH_LINE,
            subject=f"a {rf.language} routine named `{simple}`, called by code that is being documented",
            instruction="In one short line, state what this routine does for its caller - just the line, no preamble.",
        )

    # The provider proper: top up any missing contracts for this routine, then hand back its callee notes.
    def context(file_key: str, qualname: str) -> str:
        """
        Return the callee notes for one routine, generating any missing one-liners first.

        Parameters:
        - `file_key`: The run-file key of the routine being documented.
        - `qualname`: The qualified name of that routine.

        Returns:
        - The formatted callee notes from the contract store, which may be empty.
        """

        # Try to generate a one-liner for each callee still missing a contract, skipping any that already failed once.
        for key in store.missing_callee_contracts(file_key, qualname):
            if key in attempted:
                continue                                   # tried before and yielded nothing - do not retry
            attempted.add(key)
            sym = graph.symbols.get(key)
            if sym is None:
                continue
            one = generate_oneliner(sym)

            # Successes go back into the store, so later routines in the run get them for free.
            if one:
                echo(f"[callgraph] Generated one-liner for undocumented callee '{key[1]}'")
                store.update(key[0], key[1], one)

        return store.callee_notes(file_key, qualname)

    return context


def _block_callee_notes(provider, source_blob: str, source_lines: List[str], file_key: str, graph, store):
    """
    Collect read-side callee-contract annotations for every routine in a file.

    For each routine the provider yields, every call whose target has a known contract becomes a `name: contract` note keyed by its source line, ready for the block pass to show alongside the code being commented.

    Parameters:
    - `provider`: The language worker's routine provider over the parsed source.
    - `source_blob`: The file's full source text.
    - `source_lines`: The same source split into lines.
    - `file_key`: The run-file key identifying this file in the graph.
    - `graph`: The project call graph with its resolved call map.
    - `store`: The contract store holding callee one-liners.

    Returns:
    - A mapping of routine qualname to `{line: notes}` annotations; routines with no usable calls are omitted.
    """

    notes: Dict[str, Dict[int, str]] = {}

    # Routines with no resolved calls in the graph cannot carry notes, so they are skipped outright.
    for sym in provider(source_blob, source_lines):
        cmap = graph.call_map.get((file_key, sym.qualname))
        if not cmap:
            continue
        per_line: Dict[int, List[str]] = {}

        # Attach each callee's known contract to its call line, deduplicating so one line never repeats a note.
        for call in sym.calls:
            if len(call) < 3 or not call[2]:
                continue
            name, kind, line = call[0], call[1], call[2]
            target = cmap.get((name, kind))
            contract = store.contract(target) if target is not None else None
            if not contract:
                continue
            bucket = per_line.setdefault(line, [])
            note = f"{name}: {contract}"
            if note not in bucket:
                bucket.append(note)

        # Several calls on the same line collapse into one '; '-joined annotation.
        if per_line:
            notes[sym.qualname] = {ln: "; ".join(ns) for ln, ns in per_line.items()}

    return notes


def _order_header_before_impl(targets: List[Path], pairs: List[Tuple[str, str]]) -> List[Path]:
    """
    Reorder run targets so every C header is processed before its implementation.

    A topological sort over the header/implementation pairs guarantees a header's freshly written docs already exist when its implementation file is reached; targets not named in any pair keep their command-line order.

    Parameters:
    - `targets`: The run's target paths in command-line order.
    - `pairs`: `(header_key, impl_key)` resolved-path pairs that must keep that relative order.

    Returns:
    - The same targets, re-ordered to honour every applicable pair.
    """

    # Work in resolved-path keys so the constraint pairs and the CLI targets compare reliably.
    if not pairs:
        return targets
    keys = [str(t.resolve()) for t in targets]
    by_key = {str(t.resolve()): t for t in targets}
    idx = {k: i for i, k in enumerate(keys)}
    kset = set(keys)
    succ: dict = {k: set() for k in keys}

    # Only pairs with both ends actually in this run constrain the order; anything else is ignored.
    for hk, ik in pairs:
        if hk in kset and ik in kset and hk != ik:
            succ[hk].add(ik)                       # the header precedes the implementation

    # Reuse the call graph's leaf-first sort; the index tiebreak preserves the original command-line order wherever no pair forces a swap.
    ordered = scale_project._leaf_first_order(keys, succ, tiebreak=lambda k: idx[k])

    return [by_key[k] for k in ordered]


def _block_style_for(language: str, comment_style: str = "line"):
    """
    Select the block-pass comment style for a language.

    Parameters:
    - `language`: The language name (`python`, `c`, `js` or `javascript`).
    - `comment_style`: For C and JavaScript, `block` selects `/* */` comments; anything else selects `//` line comments.

    Returns:
    - The matching style object from `scale_blocks`.

    Notes:
    Raises `NotImplementedError` for languages the block pass does not yet support.
    """

    # The deferred import keeps scale_blocks out of runs that never use the block pass; unsupported languages fail loudly rather than guessing a style.
    from scale_blocks import PYTHON_STYLE, SLASH_LINE_STYLE, SLASH_BLOCK_STYLE
    if language == "python":
        return PYTHON_STYLE
    if language in ("c", "js", "javascript"):
        return SLASH_BLOCK_STYLE if comment_style == "block" else SLASH_LINE_STYLE
    raise NotImplementedError(
        f"The block pass does not yet support '{language}'."
    )


def _read_optional(path: Path) -> Optional[str]:
    """
    Read a UTF-8 text file, returning `None` if it does not exist.

    Parameters:
    - `path`: The file to read.

    Returns:
    - The file's text, or `None` when `path` is not a file.
    """

    return path.read_text(encoding="utf-8") if path.is_file() else None


def _block_pass(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    scale_path: Path,
    messages: Messages,
    source_blob: str,
    source_lines: List[str],
    language: str,
    comment_style: str = "line",
    comment_value: Optional[int] = None,
    doc_override: Optional[Callable[[str], Optional[str]]] = None,
    callee_annotations: Optional[Dict[str, Dict[int, str]]] = None,
    verifier=None,
) -> List[str]:
    """
    Run the within-function block pass over one file.

    Resolves the language-specific paragraph provider and comment style, then delegates to the language-agnostic `annotate_blocks` engine, forwarding any prompt-override files found under `scale_path`.

    Parameters:
    - `llm`: The loaded chat model.
    - `cfg`: Generation settings for the model.
    - `scale_path`: Directory searched for optional prompt-override files.
    - `messages`: The primed conversation the comments are generated within.
    - `source_blob`: The file's full text.
    - `source_lines`: The file's lines, used for patching.
    - `language`: Source language (`python`, `c` or `js`).
    - `comment_style`: Comment delimiter for C/JS: `line` or `block`.
    - `comment_value`: Optional score threshold; lower-valued comments are dropped.
    - `doc_override`: C only - callable returning replacement doc text for a routine, or `None`.
    - `callee_annotations`: Optional per-routine call-site notes shown to the model.
    - `verifier`: Optional verification hook applied to generated comments.

    Returns:
    - The updated source lines with block spacing and comments applied.
    """

    from scale_blocks import annotate_blocks
    provider = _block_provider_for(language)
    style = _block_style_for(language, comment_style)

    # Only the C provider understands a doc-site override; other languages take the plain call.
    if doc_override is not None and language == "c":
        targets = provider(source_blob, source_lines, doc_override=doc_override)
    else:
        targets = provider(source_blob, source_lines)

    # Each prompt file is optional - a missing override leaves the engine's built-in default in place.
    return annotate_blocks(
        llm, cfg, messages, source_lines, targets, style,
        segment_prompt=_read_optional(scale_path / "blocks.segment.txt"),
        comment_prompt=_read_optional(scale_path / "blocks.comment.txt"),
        comment_nudge=_read_optional(scale_path / "blocks.comment.nudge.txt"),
        note_short=_read_optional(scale_path / "blocks.note.short.txt"),
        note_long=_read_optional(scale_path / "blocks.note.long.txt"),
        score_prompt=_read_optional(scale_path / "blocks.score.txt"),
        value_threshold=comment_value,
        callee_annotations=callee_annotations,
        verifier=verifier,
    )


def _file_doc_target_for(language: str):
    """
    Return the file-doc target provider for the given language.

    Parameters:
    - `language`: Source language (`python`, `c` or `js`).

    Returns:
    - The language's `file_doc_target_*` callable.

    Notes:
    Raises `NotImplementedError` for unsupported languages; callers catch this to skip the pass cleanly.
    """

    if language == "c":
        # Imports are deferred so only the selected language's worker module is loaded.
        from scale_c import file_doc_target_c
        return file_doc_target_c

    if language == "js":
        from scale_javascript import file_doc_target_js
        return file_doc_target_js

    if language == "python":
        from scale_python import file_doc_target_py
        return file_doc_target_py

    # Callers catch this to skip the pass cleanly for unsupported languages.
    raise NotImplementedError(f"The --file-doc pass does not yet support '{language}'.")


def _file_doc_pass(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    scale_path: Path,
    src_path: Path,
    source_blob: str,
    source_lines: List[str],
    language: str,
    no_cache: Optional[bool] = False,
    project_context: str = "",
    on_file_doc: Optional[Callable[[str, Optional[str]], None]] = None,
) -> List[str]:
    """
    Run the top-of-file description pass over one file.

    Resolves the language's header-zone target, summarises the file (from a structural skeleton where one can be rendered, otherwise the full text), and splices the description in via `annotate_file_doc`. Unsupported languages and files with no annotatable zone are skipped, returning the lines unchanged.

    Parameters:
    - `llm`: The loaded chat model.
    - `cfg`: Generation settings for the model.
    - `scale_path`: Directory holding the prompt files.
    - `src_path`: Path of the file being annotated, used for the identity note and the summary cache.
    - `source_blob`: The file's current full text.
    - `source_lines`: The file's current lines.
    - `language`: Source language (`python`, `c` or `js`).
    - `no_cache`: When `True`, bypass the cached file summary.
    - `project_context`: Optional project background primed before the description turns.
    - `on_file_doc`: Optional callback receiving the spliced description and the captured thorough summary.

    Returns:
    - The updated source lines, or the originals when nothing could be annotated.
    """

    from scale_filedoc import annotate_file_doc

    # A language without file-doc support is reported and skipped rather than failing the run.
    try:
        provider = _file_doc_target_for(language)
    except NotImplementedError as exc:
        error(str(exc))
        return source_lines

    target = provider(source_blob, source_lines)

    if target is None:
        echo("file-doc: nothing to annotate.")
        return source_lines

    # Describe the whole text by default; a structural skeleton replaces it below when one can be built.
    description_source = source_blob
    is_skeleton = False
    sym_provider = _symbol_provider_for(language)

    # A skeleton (signatures and docs only) keeps large files inside the model's context budget.
    if sym_provider is not None:
        try:
            skel = scale_project.render_skeleton(source_lines, language, sym_provider(source_blob, source_lines))
        except Exception:
            skel = None      # an unparseable file simply falls back to the whole-text path

        if skel:
            description_source = skel
            is_skeleton = True

    # The description turns get a fresh conversation; the per-routine priming is not reused.
    comment_prompt = (scale_path / "comment.txt").read_text(encoding="utf-8")
    base: Messages = [{"role": "system", "content": comment_prompt}]

    # Project background goes in first so the description can place the file within the wider codebase.
    if project_context:
        base = base + [
            {"role": "user", "content":
                "Here is some background on the wider project this file belongs to:\n\n" + project_context},
            {"role": "assistant", "content": PRIMING_ACK},
        ]

    # The identity note pins down which file is being described; `capture` will receive the thorough summary.
    base = base + [
        {"role": "user", "content": _file_identity_note(src_path, language)},
        {"role": "assistant", "content": PRIMING_ACK},
    ]
    capture: dict = {}

    # Wrapped as a closure so the engine can request the summary lazily, seeded with any existing description.
    def summary_provider(seed: Optional[str]) -> str:
        """
        Fetch the (possibly cached) file summary on demand, seeded with any existing description.

        Parameters:
        - `seed`: An existing description to refresh, or `None` to write one from scratch.

        Returns:
        - The summary text used as the file's new description.
        """

        return _get_file_summary(
            llm, cfg, scale_path, src_path, description_source, language, base, no_cache=no_cache, seed=seed,
            skeleton=is_skeleton, capture=capture)

    # The callback adapter forwards the captured thorough summary alongside the description, then the splice engine takes over.
    on_description = None
    if on_file_doc is not None:
        on_description = lambda desc: on_file_doc(desc, capture.get("thorough"))
    return annotate_file_doc(
        llm, cfg, base, source_lines, target, summary_provider, language,
        classify_prompt=_read_optional(scale_path / "filedoc.classify.txt"),
        on_description=on_description,
    )


def generate_comments(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    scale_path: Path,
    src_path: Path,
    dst_path: Path,
    source_blob: str,
    source_lines: List[str],
    line_ending: str,
    language: str,
    no_cache: Optional[bool] = False,
    do_comment: bool = True,
    do_blocks: bool = False,
    do_file_doc: bool = False,
    block_comment_style: str = "line",
    comment_value: Optional[int] = None,
    project_context: str = "",
    doc_order: Optional[List[str]] = None,
    callee_context: Optional[Callable[[str], str]] = None,
    on_doc: Optional[Callable[[str, str], None]] = None,
    doc_plan=None,
    doc_override: Optional[Callable[[str], Optional[str]]] = None,
    block_callee_notes: Optional[Callable[[str, List[str]], Dict[str, Dict[int, str]]]] = None,
    verifier=None,
    skeleton: Optional[str] = None,
    on_file_doc: Optional[Callable[[str, Optional[str]], None]] = None,
) -> int:
    """
    Run the requested annotation passes over one file and write the result.

    Up to three passes run in sequence - definitions, blocks, then the file doc - each priming afresh and working on the text the previous pass produced. The file-doc pass runs last so the description can draw on the newly written docstrings.

    Parameters:
    - `llm`: The loaded chat model.
    - `cfg`: Generation settings for the model.
    - `scale_path`: Directory holding the prompt files.
    - `src_path`: Path of the source file being annotated.
    - `dst_path`: Output path; falsy prints the result to stdout instead.
    - `source_blob`: The file's original full text.
    - `source_lines`: The file's original lines.
    - `line_ending`: Detected line ending, used to rejoin lines between passes and on output.
    - `language`: Source language (`python`, `c` or `js`).
    - `no_cache`: When `True`, bypass the cached file summary.
    - `do_comment`: Run the definition (docstring/header-comment) pass.
    - `do_blocks`: Run the within-function block pass.
    - `do_file_doc`: Run the top-of-file description pass.
    - `block_comment_style`: Comment delimiter for C/JS block comments: `line` or `block`.
    - `comment_value`: Optional score threshold; lower-valued block comments are dropped.
    - `project_context`: Optional project background primed before each pass.
    - `doc_order`: Optional routine ordering for the definition pass (e.g. leaf-first).
    - `callee_context`: Optional callable supplying callee contract notes for a routine.
    - `on_doc`: Optional callback invoked with each routine's new documentation.
    - `doc_plan`: Optional C doc-site plan deciding where each routine is documented.
    - `doc_override`: Optional callable returning replacement doc text for a routine (C).
    - `block_callee_notes`: Optional callable producing per-routine call-site annotations for the block pass.
    - `verifier`: Optional verification hook applied to generated text.
    - `skeleton`: Optional pre-rendered structural skeleton used when priming.
    - `on_file_doc`: Optional callback receiving the file description and thorough summary.

    Returns:
    - `0` on success, or `1` when the block pass was requested for an unsupported language.
    """

    new_lines = source_lines

    # Definition pass first, so later passes can read the routine docs it writes.
    if do_comment:
        messages = prime_llm_for_comments(
            llm, cfg, scale_path, src_path, source_blob, language, no_cache=no_cache, template="comment",
            project_context=project_context, skeleton=skeleton,
        )
        current_blob = line_ending.join(new_lines)
        new_lines = _def_pass(llm, cfg, messages, current_blob, new_lines, language,
                              doc_order=doc_order, callee_context=callee_context, on_doc=on_doc, doc_plan=doc_plan,
                              verifier=verifier)

    # Fail fast on an unsupported language before any model time is spent.
    if do_blocks:
        try:
            _block_provider_for(language)  # fail fast (and cleanly) on an unsupported language
        except NotImplementedError as exc:
            error(str(exc))
            return 1

        # The block pass re-primes on the current text, so its prompts see the freshly written docstrings.
        current_blob = line_ending.join(new_lines)
        messages = prime_llm_for_comments(
            llm, cfg, scale_path, src_path, source_blob, language, no_cache=no_cache, template="blocks",
            project_context=project_context, skeleton=skeleton,
        )
        annotations = block_callee_notes(current_blob, new_lines) if block_callee_notes is not None else None
        new_lines = _block_pass(llm, cfg, scale_path, messages, current_blob, new_lines, language,
                                comment_style=block_comment_style, comment_value=comment_value,
                                doc_override=doc_override, callee_annotations=annotations, verifier=verifier)

    # The file-doc pass runs last by design: the description draws on everything the earlier passes wrote.
    if do_file_doc:
        current_blob = line_ending.join(new_lines)
        new_lines = _file_doc_pass(
            llm, cfg, scale_path, src_path, current_blob, new_lines, language, no_cache=no_cache,
            project_context=project_context, on_file_doc=on_file_doc,
        )

    # Encoding with `surrogateescape` round-trips undecodable bytes; without a destination the result goes to stdout.
    if dst_path:
        out_bytes = line_ending.join(new_lines).encode("utf-8", errors="surrogateescape")
        dst_path.write_bytes(out_bytes)
        echo(f"Updated source written to {dst_path}")
    else:
        print("\n".join(new_lines))

    return 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse the command-line arguments.

    The `--offline`/`--online` pair forms a mutually exclusive group, and the help strings double as the flag reference, so the behavioural detail lives there.

    Parameters:
    - `argv`: Argument sequence to parse, or `None` to use `sys.argv`.

    Returns:
    - The parsed argument namespace.
    """

    # The help strings are the user-facing flag reference, so they carry the full behavioural detail.
    formatters = sorted(llm_formatters()) + ['auto']
    p = argparse.ArgumentParser(description="SCALE: Source Code Annotation with LLM Engine")
    p.add_argument("source", nargs="*",
                   help="Source file(s) to annotate - paths, directories (expanded to source files), or globs "
                        "(e.g. \"src/**/*.c\"). Multiple targets are written in place; -o is only valid for one. "
                        "Optional only with --check-manifest.")
    p.add_argument("--model", "-m", default="", help="Optional path to GGUF model file")
    p.add_argument("--output", "-o", default="", help="Optional output filename")
    p.add_argument("--comment", "-c", action="store_true", help="Add and update definition docstrings/header comments")
    p.add_argument("--block-spacing", "-b", action="store_true",
                   help="Within-function block pass (Python, C, JS): split each routine body into paragraphs with "
                        "blank lines. On its own this adds spacing only; add --block-comments to also write comments.")
    p.add_argument("--file-doc", "-fd", action="store_true",
                   help="Add or update a file-level header doccomment (Python module docstring, or C/JS header "
                        "comment), preserving shebang/copyright/license/boilerplate byte-for-byte.")
    p.add_argument("--block-comment-style", "-bs", default="line", choices=("line", "block"),
                   help="Delimiter for block-pass comments in C/JS: 'line' (//) or 'block' (/* */). Ignored for Python.")
    p.add_argument("--block-comments", "-bc", choices=("low", "medium", "high"), default=None,
                   help="Write within-function block comments at the chosen density (implies the block pass): 'high' "
                        "keeps all of them, 'medium' drops bare restatements, 'low' keeps only intent/gotcha notes.")
    p.add_argument("--language", "-l", default=None, help="Source file language. SCALE currently supports: 'python', 'js', 'c'")
    p.add_argument("--project-doc", "-p", default="", metavar="PATH",
                   help="Project overview to distil into background context for every file (e.g. CLAUDE.md/README). "
                        "Default: auto-detect near the source. 'none' disables it; or pass an explicit path.")
    p.add_argument("--reference", "-r", action="append", default=[], metavar="PATH",
                   help="Read-only file(s)/dir(s)/glob(s) for SCALE to consult for context but never edit (e.g. the "
                        "project's headers). Repeatable. References are parsed into the call graph: they resolve "
                        "calls and supply per-routine contracts to the targets that use them.")
    p.add_argument("--doc-site", default="auto", choices=("auto", "impl"),
                   help="C only: where an extern function is documented. 'auto' (default) documents a target header's "
                        "prototype (prose from the definition's body) and skips the implementation's docstring; 'impl' "
                        "keeps the implementation docstring (legacy) while still documenting target prototypes.")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the post-generation verification (the backtick-grounding gate and the clean-context "
                        "challenge turns that catch invented identifiers, unsupported claims, and restatement). "
                        "Faster, but the local model's output is written unchecked.")
    p.add_argument("--verbose", "-v", action="store_true", help="Output progress information to stdout")
    p.add_argument("--very-verbose", "-vv", action="store_true", help="Output LLM debug information to stdout")
    p.add_argument("--no-cache", "-nc", action="store_true", help="Don't load the summary text for this file from cache")
    p.add_argument("--n-ctx", type=int, default=16 * 1024, help="Number of tokens to use as context")
    p.add_argument("--max-new-tokens", "-M", type=int, default=2 * 1024, help="Maximum number of new tokens to generate")
    p.add_argument("--format", "-f", default="auto", help=f"Chat format override. One of {formatters}")
    p.add_argument("--temperature", "-T", type=float, default=0.2, help="Temperature value for the LLM")
    p.add_argument("--top-p", "-P", type=float, default=0.9, help="Top-p value for the LLM")
    p.add_argument("--top-k", "-K", type=int, default=60, help="Top-k value for the LLM")
    p.add_argument("--repeat-penalty", "-R", type=float, default=1.05, help="Repeat penalty value for the LLM")
    p.add_argument("--n-batch", type=int, default=256, help="Number of batches to process")
    p.add_argument("--n-gpu-layers", "-g", type=int, default=-1, help="Number of GPU layers to use")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true",
                      help="Annotate everything with the local model (the default; no manifest is involved).")
    mode.add_argument("--online", action="store_true",
                      help="Defer EVERY routine's comments/docstrings to a stronger model via the run manifest "
                           "(requires --emit-manifest). Model-free and instant: the GGUF is never loaded.")
    p.add_argument("--emit-manifest", default="", metavar="PATH",
                   help="Online emit phase: write every routine's comment request for ALL targets to this one "
                        "run-level manifest (model-free; the targets are left untouched). Requires --online.")
    p.add_argument("--apply-manifest", default="", metavar="PATH",
                   help="Apply phase: patch a stronger model's answers from this manifest into the targets. No model "
                        "is loaded.")
    p.add_argument("--check-manifest", default="", metavar="PATH",
                   help="Completeness check (model-free): print the manifest's unfilled-answer count and exit "
                        "nonzero if any answer is missing. Needs no source targets.")
    p.add_argument("--next-fragment", default="", metavar="MANIFEST",
                   help="Check out the next batch of unfilled requests from this master manifest as a small "
                        "self-contained fragment file (a valid manifest; written next to the master, its path "
                        "printed). Repeated calls hand out disjoint batches, so fragments can be filled by "
                        "parallel agents. Exits nonzero when there is nothing to hand out. Needs no source targets.")
    p.add_argument("--fragment-size", type=int, default=8, metavar="N",
                   help="Maximum requests per fragment for --next-fragment (default 8).")
    p.add_argument("--emit-filedoc", default="", metavar="PATH",
                   help="Model-free: write the run-level file-description manifest - each target's current skeleton, "
                        "role, and header-zone lines - for a stronger model to fill (the online --file-doc round; "
                        "run it on the --apply-manifest outputs).")
    p.add_argument("--apply-filedoc", default="", metavar="PATH",
                   help="Model-free: splice each target's header description from this filedoc manifest (the answers' "
                        "classify range + prose), through the same license veto and preservation guard as --file-doc.")
    p.add_argument("--emit-reword", default="", metavar="PATH",
                   help="With --file-doc: also write a run-level header-reword manifest carrying the project blurb "
                        "and each file's role + freshly spliced draft description, for a stronger model to reword "
                        "with cross-file consistency. Prose only - no licence/boilerplate text is included.")
    p.add_argument("--apply-reword", default="", metavar="PATH",
                   help="Model-free: replace each target's draft header description with the reworded answer from "
                        "this manifest. The draft is located by exact match; a miss is a safe no-op.")

    return p.parse_args(argv)


def _apply_manifest_file(
    src_path: Path,
    dst_path: Optional[Path],
    language: str,
    source_lines: List[str],
    line_ending: str,
    manifest: dict,
) -> int:
    """
    Patch a manifest's answers into one target file - no model is loaded.

    Parameters:
    - `src_path`: Path of the target source file.
    - `dst_path`: Output path; `None` prints the result to stdout instead.
    - `language`: Source language (`python`, `c` or `js`).
    - `source_lines`: The target's current lines.
    - `line_ending`: Detected line ending used to rejoin the output.
    - `manifest`: The parsed run manifest carrying the answers.

    Returns:
    - `0` on success, or `1` for an unsupported language.
    """

    # The three language appliers share one signature, so each import is aliased to a common name.
    if language == "python":
        from scale_python import apply_manifest
    elif language == "c":
        from scale_c import apply_manifest_c as apply_manifest
    elif language == "js":
        from scale_javascript import apply_manifest_js as apply_manifest
    else:
        error(f"--apply-manifest supports Python, C, and JS only (got '{language}').")
        return 1

    new_lines = apply_manifest(source_lines, manifest)

    # Encoding with `surrogateescape` round-trips undecodable bytes; without a destination the result goes to stdout.
    if dst_path:
        dst_path.write_bytes(line_ending.join(new_lines).encode("utf-8", errors="surrogateescape"))
        echo(f"Applied manifest answers; written to {dst_path}")
    else:
        print("\n".join(new_lines))

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the SCALE command line and return the process exit code.

    This is the orchestrator behind every mode, dispatched in order: the model-free manifest utilities (`--check-manifest`, `--next-fragment`, `--apply-manifest`, `--apply-reword`, `--emit-filedoc`, `--apply-filedoc`); the online emit (`--online --emit-manifest`), which collects every routine's deferred comment request into a run-level manifest without loading a model; and finally the offline annotate path, which loads the local LLM, assembles the run-wide context (project blurb, call graph, verifier, C doc-site plan) and annotates each target in dependency order.

    Parameters:
    - `argv`: Argument list to parse instead of `sys.argv` when given.

    Returns:
    - `0` on success, or a non-zero code when any mode fails, work remains unfilled, or a file could not be processed.
    """

    # Paths anchor on the script's own directory so the default model and scale-cfg prompts resolve regardless of cwd.
    args = _parse_args(argv)
    set_verbosity(args.verbose)
    root = Path(__file__).resolve().parent
    scale_path = root / "scale-cfg"
    model = Path(args.model) if args.model else root / DEFAULT_MODEL
    dst_path = Path(args.output) if args.output else None

    # --check-manifest: sniff the raw JSON first, because the tool tag decides which manifest reader applies.
    if args.check_manifest:
        import json as _json
        raw = _json.loads(Path(args.check_manifest).read_text(encoding="utf-8"))

        # Reword and filedoc manifests count per-file answers; anything else is a run manifest counted per request.
        if isinstance(raw, dict) and raw.get("tool") == scale_reword.REWORD_TOOL:
            manifest = scale_reword.read_reword_manifest(Path(args.check_manifest))
            missing = scale_reword.unfilled_rewords(manifest)
            total = len(manifest.get("files", []))
        elif isinstance(raw, dict) and raw.get("tool") == scale_filedoc.FILEDOC_TOOL:
            manifest = scale_filedoc.read_filedoc_manifest(Path(args.check_manifest))
            missing = scale_filedoc.unfilled_descriptions(manifest)
            total = len(manifest.get("files", []))
        else:
            manifest = scale_escalate.read_manifest(Path(args.check_manifest))
            missing = scale_escalate.unfilled_answers(manifest)
            total = len(manifest.get("requests", []))

        # Report every unfilled slot plus any requests still checked out to fragments; the exit code drives the fill loop.
        for slot in missing:
            print(f"unfilled: {slot}")
        print(f"{len(missing)} unfilled answer(s) in {args.check_manifest} ({total} request(s))")
        out = sum(1 for r in manifest.get("requests", []) if r.get(scale_escalate.FRAGMENT_KEY))
        if out:
            print(f"{out} request(s) checked out to outstanding fragments")
        return 1 if missing else 0

    # --next-fragment: carve the next bounded work unit off the master manifest.
    if args.next_fragment:
        master_path = Path(args.next_fragment)
        manifest = scale_escalate.read_manifest(master_path)

        # A complete manifest has nothing to hand out - refuse rather than issue an empty fragment.
        if not scale_escalate.unfilled_answers(manifest):
            print(f"manifest complete: no unfilled answers in {master_path}")
            return 1

        # The master's issue counter names the fragment; building it checks out a batch of unfilled requests.
        name = scale_escalate.next_fragment_name(manifest, master_path.name)
        fragment = scale_escalate.build_fragment(manifest, args.fragment_size, name)

        # Every unfilled request is already checked out elsewhere; only --apply-manifest can release them.
        if fragment is None:
            outstanding = sorted({str(r.get(scale_escalate.FRAGMENT_KEY))
                                  for r in manifest.get("requests", []) if r.get(scale_escalate.FRAGMENT_KEY)})
            print("no fragment available: every unfilled request is checked out "
                  f"({', '.join(outstanding)}); --apply-manifest merges and releases them")

            return 1

        # Re-save the master too so the checkout markers and issue counter persist; the bare path on stdout is the machine-readable result.
        frag_path = master_path.with_name(name)
        scale_escalate.write_manifest(frag_path, fragment)
        scale_escalate.write_manifest(master_path, manifest)  # persist the checkout markers + issue counter
        print(str(frag_path))

        return 0

    # Every remaining mode operates on concrete source files expanded from the CLI patterns.
    targets = scale_project.gather_files(args.source)

    if not targets:
        error(f"No source files matched: {' '.join(args.source)}")
        return 1

    # --apply-manifest: multiple targets are patched in place, so a single -o destination is ambiguous.
    if args.apply_manifest:
        if dst_path is not None and len(targets) > 1:
            error("-o/--output cannot be used with multiple targets; multiple targets are patched in place.")
            return 1

        # Sibling fragment files are found via the master's naming scheme; lingering checkout markers alone also count as fragmented.
        manifest = scale_escalate.read_manifest(Path(args.apply_manifest))
        master_path = Path(args.apply_manifest)
        stem, dot, suffix = master_path.name.rpartition(".")
        if not dot:
            stem, suffix = master_path.name, "json"
        frag_paths = sorted(master_path.parent.glob(f"{stem}.frag-*.{suffix}"))
        fragmented = bool(frag_paths) or any(r.get(scale_escalate.FRAGMENT_KEY)
                                             for r in manifest.get("requests", []))

        # Fold every fragment's answers back into the master before any patching.
        if fragmented:
            for fp in frag_paths:
                merged = scale_escalate.merge_fragment(manifest, scale_escalate.read_manifest(fp))
                echo(f"Merged {merged} answer(s) from {fp.name}")

            # Snapshot what is still missing, release stale checkouts, persist the master, then delete the spent fragments.
            missing = scale_escalate.unfilled_answers(manifest)
            released = scale_escalate.release_unfilled(manifest)
            scale_escalate.write_manifest(master_path, manifest)
            for fp in frag_paths:
                fp.unlink()  # spent: their answers now live in the master, written above

            # An incomplete merge aborts the apply; the released requests can be handed out again.
            if missing:
                for slot in missing:
                    print(f"unfilled: {slot}")
                error(f"{len(missing)} answer(s) still unfilled after merging {len(frag_paths)} fragment(s); "
                      f"{released} request(s) returned to the pile - hand them out again with --next-fragment.")

                return 1

        # Index the manifest by resolved path so each target receives only its own requests.
        file_entries = {str(Path(f.get("path", "")).resolve()): f for f in manifest.get("files", [])}
        by_file: Dict[str, list] = {}
        for r in manifest.get("requests", []):
            by_file.setdefault(str(Path(r.get("file", "")).resolve()), []).append(r)
        rc = 0

        for target in targets:
            key = str(target.resolve())
            requests = by_file.get(key, [])

            if not requests:
                echo(f"No manifest requests for {target}; leaving it unchanged.")
                continue

            # An explicit -l wins; otherwise trust the language the manifest recorded at emit time.
            entry = file_entries.get(key, {})
            language = args.language.lower() if args.language else entry.get("language")
            _blob, source_lines, line_ending, language = load_source(target, language)

            if language not in SUPPORTED_LANGUAGES:
                error(f"Unsupported language '{language}'. SCALE supports: {', '.join(SUPPORTED_LANGUAGES)}")
                rc = 1
                continue

            # Each file is patched against a per-file view of the manifest; -o only applies when there is a single target.
            sub_manifest = dict(manifest)
            sub_manifest["requests"] = requests
            out = dst_path if len(targets) == 1 else target
            frc = _apply_manifest_file(target, out, language, source_lines, line_ending, sub_manifest)
            if frc != 0:
                rc = frc

        return rc

    # --apply-reword: splice approved header rewordings back into their files.
    if args.apply_reword:
        if dst_path is not None and len(targets) > 1:
            error("-o/--output cannot be used with multiple targets; multiple targets are patched in place.")
            return 1

        manifest = scale_reword.read_reword_manifest(Path(args.apply_reword))
        entries = {str(Path(f.get("path", "")).resolve()): f for f in manifest.get("files", [])}
        rc = 0

        for target in targets:
            entry = entries.get(str(target.resolve()))

            # Only entries with a non-blank answer touch their file.
            if entry is None or not str(entry.get("answer") or "").strip():
                echo(f"No reword answer for {target}; leaving it unchanged.")
                continue

            language = args.language.lower() if args.language else entry.get("language")
            source_blob, source_lines, line_ending, language = load_source(target, language)

            # Languages without a header-zone provider cannot take a reword.
            try:
                provider = _file_doc_target_for(language)
            except NotImplementedError as exc:
                error(str(exc))
                rc = 1
                continue

            zone = provider(source_blob, source_lines)

            if zone is None:
                echo(f"reword: {target} has no header zone; leaving it unchanged.")
                continue

            # The recorded draft guards the splice - the reword only lands where the original prose is still recognised.
            new_lines, changed = scale_reword.apply_reword(
                source_lines, zone, entry.get("draft", ""), entry.get("answer", ""))
            out = dst_path if len(targets) == 1 else target

            # Written back binary-safe with the file's own line endings; without -o the result goes to stdout.
            if out:
                out.write_bytes(line_ending.join(new_lines).encode("utf-8", errors="surrogateescape"))
                echo(f"{'Reworded' if changed else 'Unchanged'}: written to {out}")
            else:
                print("\n".join(new_lines))

        return rc

    # --emit-filedoc: build one self-contained description request per file, seeded with the project doc.
    if args.emit_filedoc:
        spec = _read_optional(scale_path / "summary.txt") or SUMMARY_INSTRUCTION
        project_doc_text = ""
        project_doc = scale_project.resolve_project_doc(args.project_doc, targets[0])

        # The project doc is capped, and an unreadable one is treated as absent rather than fatal.
        if project_doc is not None:
            try:
                project_doc_text = project_doc.read_text(encoding="utf-8", errors="replace")
                project_doc_text = project_doc_text[:scale_filedoc.FILEDOC_PROJECT_DOC_CAP]
            except OSError:
                project_doc_text = ""

        entries: List[dict] = []
        rc = 0

        for target in targets:
            language = args.language.lower() if args.language else None
            source_blob, source_lines, line_ending, language = load_source(target, language)

            if language not in SUPPORTED_LANGUAGES:
                error(f"Skipping {target}: unsupported language '{language}' (SCALE supports: "
                      f"{', '.join(SUPPORTED_LANGUAGES)}).")
                rc = 1
                continue

            try:
                provider = _file_doc_target_for(language)
            except NotImplementedError as exc:
                error(str(exc))
                rc = 1
                continue

            zone = provider(source_blob, source_lines)

            if zone is None:
                echo(f"filedoc: {target} has nothing to describe; skipping.")
                continue

            # A parsed skeleton keeps the request small; only unparseable files ship their full text.
            skeleton = None
            sym_provider = _symbol_provider_for(language)

            if sym_provider is not None:
                try:
                    skeleton = scale_project.render_skeleton(
                        source_lines, language, sym_provider(source_blob, source_lines))
                except Exception:
                    skeleton = None      # an unparseable file simply falls back to the whole-text path

            # The request carries everything needed remotely: skeleton or full text, the file's role, and the existing header entries to update.
            entries.append({
                "path": str(target),
                "language": language,
                "role": _file_role(target, language),
                "skeleton": skeleton if skeleton else source_blob,
                "entries": [inner for (_lineno, _prefix, inner) in zone.eligible],
                "answer": {"range": None, "description": None},
            })

        manifest = scale_filedoc.filedoc_manifest(spec, project_doc_text, entries)
        scale_filedoc.write_filedoc_manifest(Path(args.emit_filedoc), manifest)
        echo(f"Wrote {len(entries)} file-description request(s) to {args.emit_filedoc}")

        return rc

    # --apply-filedoc: splice the returned descriptions into each file's header zone.
    if args.apply_filedoc:
        if dst_path is not None and len(targets) > 1:
            error("-o/--output cannot be used with multiple targets; multiple targets are patched in place.")
            return 1

        manifest = scale_filedoc.read_filedoc_manifest(Path(args.apply_filedoc))
        file_entries = {str(Path(f.get("path", "")).resolve()): f for f in manifest.get("files", [])}
        rc = 0

        for target in targets:
            entry = file_entries.get(str(target.resolve()))

            if entry is None:
                echo(f"No file-description answer for {target}; leaving it unchanged.")
                continue

            language = args.language.lower() if args.language else entry.get("language")
            source_blob, source_lines, line_ending, language = load_source(target, language)

            try:
                provider = _file_doc_target_for(language)
            except NotImplementedError as exc:
                error(str(exc))
                rc = 1
                continue

            zone = provider(source_blob, source_lines)

            if zone is None:
                echo(f"filedoc: {target} has no header zone; leaving it unchanged.")
                continue

            # A None result means the splice was declined (unchanged or vetoed); the original lines are kept.
            new_lines = scale_filedoc.apply_filedoc_entry(source_lines, zone, entry)
            changed = new_lines is not None
            final = new_lines if changed else source_lines
            out = dst_path if len(targets) == 1 else target

            if out:
                out.write_bytes(line_ending.join(final).encode("utf-8", errors="surrogateescape"))
                echo(f"{'Described' if changed else 'Unchanged'}: written to {out}")
            else:
                print("\n".join(final))

        return rc

    # Annotation proper starts here; the --block-comments level maps to the minimum usefulness score a block comment must earn.
    do_blocks = args.block_spacing or args.block_comments is not None
    block_threshold = BLOCK_COMMENT_LEVELS.get(args.block_comments, 4)

    # --online defers everything through the manifest, so it is meaningless without somewhere to write one.
    if args.online:
        if not args.emit_manifest:
            error("--online requires --emit-manifest PATH (the deferred comment requests must be written somewhere).")
            return 1

        # The next checks reject offline-only flags that have no meaning in an online emit.
        if args.file_doc:
            error("--file-doc is a local pass; online, run the file-description round with --emit-filedoc / "
                  "--apply-filedoc after the manifest has been applied.")
            return 1

        if args.emit_reword:
            error("--emit-reword belongs to the offline --file-doc pass; online, use --emit-filedoc instead.")
            return 1

        if args.block_spacing and args.block_comments is None:
            # Spacing alone has nothing to defer; conversely a manifest path without --online would silently change nothing.
            error("--block-spacing alone is deterministic local work with nothing to defer; run it offline "
                  "(or add --block-comments to defer the comments).")
            return 1
    elif args.emit_manifest:
        error("--emit-manifest requires --online (offline runs are manifest-free; the manifest defers ALL routines).")
        return 1

    # No pass requested means a successful no-op.
    if not (args.comment or do_blocks or args.file_doc):
        return 0

    if dst_path is not None and len(targets) > 1:
        error("-o/--output cannot be used with multiple targets; multiple targets are annotated in place.")
        return 1

    if args.emit_reword and not args.file_doc:
        error("--emit-reword requires --file-doc (the manifest carries the freshly spliced descriptions).")
        return 1

    # References feed context only; anything that is also a target is dropped so it is not processed twice.
    references = scale_project.gather_files(args.reference) if args.reference else []
    target_keys = {p.resolve() for p in targets}
    references = [r for r in references if r.resolve() not in target_keys]

    # Online emit: scan the run once and assemble the doc style that travels inside the manifest (guidelines plus each target language's comment spec).
    if args.online:
        language_arg = args.language.lower() if args.language else None
        run_files = _scan_run_files(targets, references, language_arg)
        c_plan = _build_c_doc_plan(run_files, args.doc_site) if args.comment else None
        langs = sorted({rf.language for rf in run_files.values()
                        if rf.is_target and rf.language in SUPPORTED_LANGUAGES})
        pieces = [t for t in (_read_optional(scale_path / "guidelines.md"),) if t]
        pieces += [t for lang in langs for t in (_read_optional(scale_path / f"comment.{lang}.txt"),) if t]
        emit_doc_style = "\n\n".join(pieces)
        emit_parts: List[Tuple[str, str, str, "scale_escalate.Escalation"]] = []
        rc = 0

        for target in targets:
            language = language_arg
            source_blob, source_lines, line_ending, language = load_source(target, language)

            if language not in SUPPORTED_LANGUAGES:
                error(f"Skipping {target}: unsupported language '{language}' (SCALE supports: "
                      f"{', '.join(SUPPORTED_LANGUAGES)}).")
                rc = 1
                continue

            # One collector per file accumulates that file's deferred requests.
            escalation = scale_escalate.Escalation(doc_style=emit_doc_style)

            # Per-language collectors record one definition request per routine; C additionally threads the header/impl doc-site plan.
            if args.comment:
                if language == "python":
                    from scale_python import collect_def_requests
                    n = collect_def_requests(source_blob, source_lines, escalation)
                elif language == "js":
                    from scale_javascript import collect_def_requests_js
                    n = collect_def_requests_js(source_blob, source_lines, escalation)
                else:
                    from scale_c import collect_def_requests_c
                    doc_plan = c_plan.for_file(str(target.resolve())) if c_plan is not None else None
                    n = collect_def_requests_c(source_blob, source_lines, escalation, doc_plan=doc_plan)

                echo(f"[emit] {target}: {n} definition request(s)")

            # Block recipes are deferred the same way, with the short/long length notes baked into each request.
            if do_blocks:
                from scale_blocks import defer_block_targets
                provider = _block_provider_for(language)
                n = defer_block_targets(
                    escalation, source_lines, provider(source_blob, source_lines),
                    note_short=_read_optional(scale_path / "blocks.note.short.txt"),
                    note_long=_read_optional(scale_path / "blocks.note.long.txt"),
                )
                echo(f"[emit] {target}: {n} block recipe(s)")

            emit_parts.append((str(target), language, line_ending, escalation))

            # During emit -o just copies the source; the comments arrive later via --apply-manifest.
            if dst_path is not None:
                dst_path.write_bytes(source_blob.encode("utf-8", errors="surrogateescape"))
                echo(f"Emit copy written to {dst_path}")

        # Every file's requests merge into a single run-level manifest.
        manifest = scale_escalate.run_manifest(emit_parts, emit_doc_style)
        scale_escalate.write_manifest(Path(args.emit_manifest), manifest)
        echo(f"Wrote {len(manifest['requests'])} request(s) to {args.emit_manifest}")

        return rc

    # Offline path begins: only now is the local model loaded (every mode above is model-free), followed by generation settings and the optional project blurb.
    echo("Loading the LLM...")
    llm = LocalChatModel(
        str(model),
        chat_format=args.format,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_gpu_layers=args.n_gpu_layers,
        verbose=args.very_verbose,
    )
    echo("Initialising LLM configuration...")
    cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repeat_penalty=args.repeat_penalty,
    )
    project_blurb = ""
    project_doc = scale_project.resolve_project_doc(args.project_doc, targets[0])
    if project_doc is not None:
        project_blurb = scale_project.project_blurb(llm, cfg, scale_path, project_doc, no_cache=args.no_cache)
    project_context = project_blurb
    run_files: dict = {}
    graph = store = callee_oneliners = None
    language_arg = args.language.lower() if args.language else None

    # The run-file scan and call graph are only built when a pass will actually consume them.
    if args.comment or do_blocks:
        run_files = _scan_run_files(targets, references, language_arg)
        graph, store = _build_call_graph(run_files)

    verifier = None

    # Verification is on by default; the whole-run corpus lets the grounding gate check claims against any file in the run.
    if run_files and not args.no_verify:
        verifier = scale_verify.Verifier(
            llm, cfg,
            corpus="\n".join(rf.source_blob for rf in run_files.values()),
            gate_nudge=_read_optional(scale_path / "verify.gate.txt"),
            grounding_prompt=_read_optional(scale_path / "verify.grounding.txt"),
            grounding_feedback=_read_optional(scale_path / "verify.grounding.feedback.txt"),
            obvious_prompt=_read_optional(scale_path / "verify.obvious.txt"),
            obvious_feedback=_read_optional(scale_path / "verify.obvious.feedback.txt"),
            story_prompt=_read_optional(scale_path / "verify.story.txt"),
            story_feedback=_read_optional(scale_path / "verify.story.feedback.txt"),
        )

    # Callee one-liners are generated lazily through this closure rather than up front.
    if graph is not None and store is not None:
        callee_oneliners = _make_callee_oneliner_context(llm, cfg, run_files, graph, store)

        # Targets are reordered leaf-first so callees are documented before their callers.
        if args.comment:
            key_to_path = {str(t.resolve()): t for t in targets}
            targets = [key_to_path[k] for k in graph.file_order(list(key_to_path.keys()))]

    c_plan = None

    # C header/impl pairs are documented header-first so the implementation can reuse the header's doc.
    if args.comment:
        c_plan = _build_c_doc_plan(run_files, args.doc_site)
        if c_plan is not None and c_plan.pairs:
            targets = _order_header_before_impl(targets, c_plan.pairs)

    reword_entries: List[dict] = []
    rc = 0

    # Main per-file annotation loop.
    for target in targets:
        language = args.language.lower() if args.language else None
        source_blob, source_lines, line_ending, language = load_source(target, language)

        if language not in SUPPORTED_LANGUAGES:
            error(f"Skipping {target}: unsupported language '{language}' (SCALE supports: "
                  f"{', '.join(SUPPORTED_LANGUAGES)}).")
            continue

        doc_order = callee_context = on_doc = block_notes = None

        # Default arguments pin this iteration's file key into each closure - late binding would leak the final target.
        if graph is not None and store is not None:
            file_key = str(target.resolve())
            doc_order = graph.doc_order(file_key)
            callee_context = lambda q, fk=file_key: callee_oneliners(fk, q)
            on_doc = lambda q, doc, fk=file_key: store.update(fk, q, doc)

            if do_blocks:
                sym_provider = _symbol_provider_for(language)

                # Block-pass call annotations need a symbol provider; languages without one simply go without.
                if sym_provider is not None:
                    block_notes = (lambda blob, lines, fk=file_key, p=sym_provider:
                                   _block_callee_notes(p, blob, lines, fk, graph, store))

        doc_plan = doc_override = None

        # For C, the doc-site plan can supply a routine's doc from its header instead of generating it afresh.
        if c_plan is not None and language == "c":
            doc_plan = c_plan.for_file(str(target.resolve()))
            doc_override = lambda name, p=c_plan: p.header_doc(name)

        # A pre-rendered skeleton gives the file-doc pass a structural overview without re-reading the whole file.
        skeleton = None
        rf = run_files.get(str(target.resolve()))
        if rf is not None:
            skeleton = scale_project.render_skeleton(rf.source_lines, rf.language, rf.symbols)
        on_file_doc = None

        # --emit-reword captures each freshly spliced description as a manifest draft for a later prose-improvement round.
        if args.emit_reword:
            on_file_doc = (lambda desc, thorough, t=target, lang=language:
                           reword_entries.append({
                               "path": str(t), "language": lang, "role": _file_role(t, lang),
                               "draft": desc, "context": thorough, "answer": None}))

        # All requested passes for this file run inside generate_comments; a failure is recorded but does not stop the run.
        out = dst_path if len(targets) == 1 else target
        echo(f"Annotating {target}...")
        frc = generate_comments(
            llm, cfg, scale_path, target, out,
            source_blob, source_lines, line_ending, language,
            no_cache=args.no_cache,
            do_comment=args.comment,
            do_blocks=do_blocks,
            do_file_doc=args.file_doc,
            block_comment_style=args.block_comment_style,
            comment_value=block_threshold,
            project_context=project_context,
            doc_order=doc_order,
            callee_context=callee_context,
            on_doc=on_doc,
            doc_plan=doc_plan,
            doc_override=doc_override,
            block_callee_notes=block_notes,
            verifier=verifier,
            skeleton=skeleton,
            on_file_doc=on_file_doc,
        )
        if frc != 0:
            rc = frc

        # Freshly minted header docs are propagated to their implementation files' store entries so later files see the contract.
        if c_plan is not None and store is not None:
            for nm, doc in list(c_plan.header_docs.items()):
                ik = c_plan.impl_file.get(nm)
                if ik:
                    store.update(ik, nm, doc)

    # The collected drafts ship as a single reword manifest once every file is done.
    if args.emit_reword:
        reword = scale_reword.reword_manifest(project_blurb, reword_entries)
        scale_reword.write_reword_manifest(Path(args.emit_reword), reword)
        echo(f"Wrote {len(reword_entries)} description draft(s) to {args.emit_reword}")

    return rc


if __name__ == "__main__":  # pragma: no cover
    """
    This block ensures that the script can be run directly, invoking the `main` function and exiting with its return value.
    The `# pragma: no cover` comment is used to exclude this line from coverage reports, likely because it is not part of
    the normal execution flow and would always exit immediately.
    """
    raise SystemExit(main())
