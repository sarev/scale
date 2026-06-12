#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

This program is a source code annotation tool called SCALE, which uses a Large Language Model (LLM) to generate comments and
summaries for the provided source code. It supports various programming languages, including Python, JavaScript, and C.

The tool can be run from the command line with optional arguments to customise its behaviour. It loads the source file,
determines the language, and then primes the LLM with a system prompt and the source file as context. The LLM is then used
to generate comments and summaries for the code.
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
    Return a one-line note naming the file being documented and (for C) whether it is a header or an implementation.

    A header's documentation describes the external interface; an implementation's describes internal behaviour. Naming
    the file and its role in the priming context steers the summary, definition, and block passes accordingly - most
    sharply for a header, whose comments should read as the public contract rather than implementation detail.

    Parameters:
    - `src_path`: The source file being documented.
    - `language`: The resolved language identifier.

    Returns:
    - A short context sentence.
    """

    name = src_path.name
    if src_path.suffix.lower() in _HEADER_SUFFIXES:
        return (f"The file being documented is `{name}`, a header file: it declares a module's public interface, so "
                f"its documentation should describe the external contract a caller relies on - each function's "
                f"purpose, its parameters and its return value - rather than internal implementation detail.")
    if language == "c":
        return f"The file being documented is `{name}`, a C implementation file."
    return f"The file being documented is `{name}`."


def _file_role(src_path: Path, language: str) -> str:
    """
    Classify a file's role for the header-reword manifest: "header", "implementation", or "other".

    The same suffix rule that drives `_file_identity_note`, reduced to a label the reword manifest can carry so the
    stronger model words a header's description as a public contract and an implementation's as internal detail.

    Parameters:
    - `src_path`: The source file.
    - `language`: The resolved language identifier.

    Returns:
    - The role label.
    """

    if src_path.suffix.lower() in _HEADER_SUFFIXES:
        return "header"
    if language == "c":
        return "implementation"
    return "other"


# ---------------------------- CLI harness ----------------------------


def load_source(src_path: Path, language: Optional[str] = None) -> Tuple[str, Chunk, str, str]:
    """
    Load a source file and return its contents, along with other information.

    This function loads the specified source file, determines its line ending, and guesses the programming language.
    It returns a tuple containing the complete source file text, individual lines, line ending string, and file suffix.

    Parameters:
    - `src_path`: The path to the source file to be loaded.
    - `language`: The optional language identifier (default: None).

    Returns:
    - A tuple containing:
      - `source_blob`: The complete text of the source file as a single string (with original line endings).
      - `source_lines`: The source file split into individual lines.
      - `line_ending`: The source file line ending string ('\n', '\r', or '\r\n').
      - `language`: The guessed language identifier, e.g. "c", "cpp", "js", etc.

    Notes:
    - If the file is not found, an error message is printed and the program exits.
    - If the language is not specified, it is guessed based on the source code heuristics.
    """

    def guess_language(source_lines: List[str]) -> str:
        """
        Guess the programming language from the supplied lines of source code.

        This function applies heuristics to determine the likely programming language
        based on the presence of specific keywords, syntax, and patterns in the source code.

        Args:
        - source_lines: The list of source file lines of interest.

        Returns:
        - The guessed language identifier, e.g.
          - "c" - C (including C header files)
          - "cpp" - C++
          - "js" - JavaScript / TypeScript
          - "python" - Python
          - "sh" - Bash / shell
          - "vb" - Visual Basic
          - "java" - Java
          - "go" - Go
          - "text" - Unknown
        """

        if not source_lines:
            return "text"

        first = source_lines[0].strip()

        # Shebang detection (strong signal, early exit)
        if first.startswith("#!"):
            sh = first.lower()
            if "python" in sh:
                return "python"
            if "bash" in sh or sh.endswith("/sh"):
                return "sh"
            if "node" in sh:
                return "js"

        stripped = [s.strip() for s in source_lines if s.strip()]
        scores = defaultdict(int)
        scores["text"] = 0

        # Iterate over all non-empty lines, with all leading- and trailing spaces removed
        for line in stripped:
            last_char = line[-1]
            uline = line.upper()

            # ---- C ----
            if line.startswith(("#include", "#define ")):
                scores["c"] += 2
            if line.startswith(("extern", "static")) and last_char == ";" and not any(tok in line for tok in ["public", "final"]):
                scores["c"] += 2
                scores["cpp"] += 2  # also common in C++

            # ---- C++ ----
            if "using namespace" in line or line.startswith("template<"):
                scores["cpp"] += 3
            if line.startswith(("public:", "private:", "protected:")):
                scores["cpp"] += 2

            # ---- Python ----
            if last_char == ":" and line.startswith(("def ", "class ")):
                scores["python"] += 3
            if last_char != ";" and line.startswith(("import ", "from ")):
                scores["python"] += 2

            # ---- JavaScript / TypeScript ----
            if line.startswith(("function ", "export ", "const ", "let ", "var ")):
                scores["js"] += 2
            if line.startswith("import ") and " from " in line and last_char == ";":
                scores["js"] += 2
            if any(tok in line for tok in ("document.", "window.", "console.", "JSON.", "=>")):
                scores["js"] += 3
            if last_char == "{" and "class " in line:
                scores["js"] += 1
                scores["java"] += 1  # also common in Java

            # ---- Java ----
            if line.startswith("package "):
                scores["java"] += 3
                scores["go"] += 2  # also boost Go (both use package)
            if line.startswith(("import java.", "import javax.")):
                scores["java"] += 3
            if line.startswith("public class "):
                scores["java"] += 3
            if "System.out." in line or "public static void main" in line:
                scores["java"] += 3

            # ---- Go ----
            if line.startswith(("import (", "func ")):
                scores["go"] += 3
            if "fmt." in line or line.startswith("go "):
                scores["go"] += 2

            # ---- Bash / shell ----
            if line.startswith("echo ") or line in ("fi", "done", "esac"):
                scores["sh"] += 2

            # ---- Visual Basic ----
            if uline.startswith(("SUB ", "FUNCTION ", "DIM ", "PRINT ")):
                scores["vb"] += 3
            if uline.startswith(("MODULE ", "IMPORTS ", "PUBLIC CLASS ")):
                scores["vb"] += 2

        best = max(scores.items(), key=lambda kv: kv[1])
        return best[0]

    echo(f"Loading source file '{str(src_path)}'...")
    if not src_path.is_file():
        echo(f"Error: file not found: {src_path}")
        sys.exit(1)

    # Load the whole file into a string
    raw = src_path.read_bytes()
    source_blob = raw.decode("utf-8", errors="surrogateescape")

    # Count newline styles
    count_rn = source_blob.count("\r\n")
    count_r = source_blob.count("\r") - count_rn  # bare \r not part of \r\n
    count_n = source_blob.count("\n") - count_rn  # bare \n not part of \r\n

    # Find the most common one
    if count_rn > max(count_r, count_n):
        line_ending = "\r\n"
    else:
        line_ending = "\r" if count_r > count_n else "\n"

    # print(f"count_rn {count_rn} count_r {count_r} count_n {count_n}")
    # print(f"line_ending = {ord(line_ending)}")
    # exit(0)

    # Create a version of the source code as a 'chunk' (list of strings, one per line)
    source_lines = source_blob.split(line_ending)

    # Determine what the language is that we're dealing with
    if language is None or language == "":
        language = guess_language(source_lines)
    echo(f"Language set to '{language}'...")

    return source_blob, source_lines, line_ending, language


class SummaryCache:
    """
    File-backed summary cache.

    This class provides a file-backed cache for storing summaries of source code files. It maps a source file path to a
    stable unique identifier (UID) stored in an index file, and stores the summary in a separate data file associated
    with the UID.

    The cache is designed to be atomic, ensuring that writes to both the index and summary files are atomic operations.

    Parameters:
    - `source_path`: The path to the source code file.

    Notes:
    - The cache directory is located at `<cache_dir>/<uid>.summary`, where `<cache_dir>` is a fixed directory
      and `<uid>` is the unique identifier for the source file.
    - The index file contains a dictionary mapping source file paths to their associated UIDs.
    - The summary file contains a human-readable summary of the source code as a UTF-8 encoded string with surrogate escape.
    """
    _CACHE_DIR = (Path(__file__).resolve().parent) / "__cache__"
    _CACHE_INDEX = _CACHE_DIR / "index.pkl"

    def __init__(self, source_path: Path, source_blob: str) -> None:
        """
        Initialise the instance with a source path and its current contents.

        This call loads or generates a unique identifier (UID) for the given source path and stores it in the index file.
        The UID is used to identify the associated data file, which contains a summary of the source code. A SHA-256 hash
        of `source_blob` is recorded alongside the summary so that a cached summary is only reused while the source file
        is unchanged; editing the file invalidates the cache automatically.

        Parameters:
        - `source_path`: The path to the source code file.
        - `source_blob`: The current contents of the source file (used to compute the invalidation hash).

        Returns:
        - None
        """

        self._summary: Optional[str] = None
        self._short: Optional[str] = None
        self._hash = hashlib.sha256(source_blob.encode("utf-8", errors="surrogateescape")).hexdigest()

        # Load or create index
        index = self._load_index()

        key = str(source_path)
        uid = index.get(key)
        if uid is None:
            uid = uuid.uuid4().hex
            index[key] = uid
            self._save_index(index)

        self._uid = uid
        self._data_path = self._CACHE_DIR / f"{self._uid}.txt"        # the full (description) summary
        self._short_path = self._CACHE_DIR / f"{self._uid}.short.txt"  # the squashed summary for the definition pass
        self._hash_path = self._CACHE_DIR / f"{self._uid}.sha256"     # content hash for invalidation

        # Load the existing summaries only if they were generated from the same source content.
        try:
            cached_hash = self._hash_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            cached_hash = None

        if cached_hash == self._hash:
            self._summary = self._read_optional_text(self._data_path)
            self._short = self._read_optional_text(self._short_path)
        else:
            # Stale (or missing) hash: ignore any cached summaries so they are regenerated.
            self._summary = None
            self._short = None

    @property
    def summary(self) -> Optional[str]:
        """
        Return a human-readable summary of this instance.

        Returns:
        - `str`: The summary string, or `None` if no summary is available.
        """

        return self._summary

    @summary.setter
    def summary(self, text: str) -> None:
        """
        Set the summary text for this instance and persist it in the cache.

        Parameters:
        - `text`: The new summary text as a string.
        """

        self._summary = text
        self._atomic_write_bytes(
            self._data_path,
            text.encode("utf-8", errors="surrogateescape"),
        )
        # Record the content hash so this summary is reused only while the source is unchanged.
        self._atomic_write_bytes(self._hash_path, self._hash.encode("utf-8"))

    @property
    def short(self) -> Optional[str]:
        """
        Return the squashed (one/two-sentence) summary used to prime the definition pass, or None if not cached.
        """

        return self._short

    @short.setter
    def short(self, text: str) -> None:
        """
        Persist the squashed summary in the cache, tagged with the same content hash as the full summary.

        Parameters:
        - `text`: The short summary text.
        """

        self._short = text
        self._atomic_write_bytes(self._short_path, text.encode("utf-8", errors="surrogateescape"))
        self._atomic_write_bytes(self._hash_path, self._hash.encode("utf-8"))

    @staticmethod
    def _read_optional_text(path: Path) -> Optional[str]:
        """
        Read a cache text file with surrogateescape decoding, returning None if it is absent.

        Parameters:
        - `path`: The cache file to read.

        Returns:
        - The decoded contents, or None when the file does not exist.
        """

        try:
            return path.read_bytes().decode("utf-8", errors="surrogateescape")
        except FileNotFoundError:
            return None

    @classmethod
    def _load_index(cls) -> dict[str, str]:
        """
        Load the index from cache.

        If the index file exists, load it from disc and return its contents as a dictionary.
        If the file is missing or corrupt, start with an empty index.

        Returns:
            dict[str, str]: The loaded index, or an empty dictionary if loading failed.
        """

        try:
            with cls._CACHE_INDEX.open("rb") as f:
                obj = pickle.load(f)
                return obj if isinstance(obj, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception:
            # Corrupt index: start fresh rather than crashing
            return {}

    @classmethod
    def _save_index(cls, index: dict[str, str]) -> None:
        """
        Save the index to a temporary file and then replace the existing cache index.

        This method creates a new temporary file with a `.pkl.tmp` suffix, writes the index to it using pickle,
        and then replaces the original cache index file with the temporary one.
        """

        cls._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cls._CACHE_INDEX.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cls._CACHE_INDEX)

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        """
        Atomically write bytes to a file.

        Create the directory for the file if it does not exist, then write the data to a temporary file.
        Finally, replace the original file with the temporary one, ensuring that the operation is atomic.

        Parameters:
        - `path`: The path to the file to be written.
        - `data`: The bytes to be written to the file.

        Returns:
        - None
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            f.write(data)
        tmp.replace(path)


def _hard_split_line(line: str, chunk_budget: int, estimate_fn: Callable[[str], int]) -> List[str]:
    """
    Hard-split a single over-long line into pieces that each fit a token budget.

    This is a last resort for pathological input such as minified code with no usable line breaks. The split is by
    character count, sized from the budget using the supplied token estimate.

    Parameters:
    - `line`: The over-long line.
    - `chunk_budget`: The maximum estimated tokens per piece.
    - `estimate_fn`: A cheap function estimating the token count of a string.

    Returns:
    - A list of substrings of `line`, each within budget.
    """

    if not line:
        return [line]
    # Estimate characters-per-token from this very line, then size pieces a little under budget.
    per_token = max(1, len(line) // max(1, estimate_fn(line)))
    piece_chars = max(1, int(chunk_budget * per_token * 0.9))
    return [line[i:i + piece_chars] for i in range(0, len(line), piece_chars)]


def _split_source(source_blob: str, chunk_budget: int, estimate_fn: Callable[[str], int]) -> List[str]:
    """
    Split source text into chunks that each fit within a token budget.

    Splitting happens on line boundaries so chunks stay readable, and a break is preferred at a blank line once a
    chunk is reasonably full. A single line longer than the whole budget (e.g. minified code) is hard-split by
    characters as a last resort.

    Parameters:
    - `source_blob`: The complete source text.
    - `chunk_budget`: The maximum estimated tokens per chunk.
    - `estimate_fn`: A cheap function estimating the token count of a string.

    Returns:
    - A list of chunk strings. For input without over-long lines, these rejoin with '\\n' to the original text.
    """

    lines = source_blob.split("\n")
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush() -> None:
        """Append the accumulated lines (if any) as a chunk and reset the accumulator."""
        nonlocal current, current_tokens
        if current:
            chunks.append("\n".join(current))
            current = []
            current_tokens = 0

    for line in lines:
        cost = estimate_fn(line) + 1  # +1 approximates the joining newline

        # A single over-long line cannot fit any chunk: flush, then hard-split it by characters.
        if cost > chunk_budget:
            flush()
            chunks.extend(_hard_split_line(line, chunk_budget, estimate_fn))
            continue

        # Start a new chunk if appending this line would overflow the current one.
        if current and current_tokens + cost > chunk_budget:
            flush()

        current.append(line)
        current_tokens += cost

        # Prefer to break at a blank line once the chunk is reasonably full.
        if not line.strip() and current_tokens >= chunk_budget * 0.75:
            flush()

    flush()
    return chunks


def _group_by_budget(partials: List[str], budget: int, estimate_fn: Callable[[str], int]) -> List[List[str]]:
    """
    Group consecutive partial summaries so each group's combined text fits a token budget.

    A minimum group size of two guarantees the reduction makes progress (the number of groups is always fewer than
    the number of inputs), preventing unbounded recursion when summaries are individually large.

    Parameters:
    - `partials`: The partial summaries to group.
    - `budget`: The maximum estimated tokens per group.
    - `estimate_fn`: A cheap function estimating the token count of a string.

    Returns:
    - A list of groups, each a list of consecutive partial summaries.
    """

    groups: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0
    for s in partials:
        cost = estimate_fn(s) + 8  # small allowance for the "Part N:" framing
        if len(current) >= 2 and current_tokens + cost > budget:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(s)
        current_tokens += cost
    if current:
        groups.append(current)

    # Guarantee progress: if everything landed in one group, force a split so recursion shrinks the input.
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
    Combine partial summaries into a single overall summary, recursing if they do not all fit at once.

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `summary_cfg`: The (capped) generation configuration to use for summaries.
    - `base_messages`: The primed context to summarise against (not mutated).
    - `partials`: The partial summaries to combine.
    - `language`: The source language identifier (used to phrase the prompt).
    - `limit`: The usable prompt-token limit (n_ctx minus margins and the summary reply reserve).
    - `base_overhead`: The token cost of `base_messages`, precomputed.

    Returns:
    - The combined overall summary text.
    """

    if len(partials) == 1:
        return partials[0]

    subject = f"summaries of consecutive parts of one {language} source file"
    combined = "\n\n".join(f"Part {i}: {s}" for i, s in enumerate(partials, start=1))

    # If the combined summaries fit, reduce them in a single pass.
    if base_overhead + llm.estimate_tokens(combined) + SUMMARY_WRAPPER_TOKENS <= limit:
        return summarise(llm, summary_cfg, combined, LENGTH_PARAGRAPHS,
                         base_messages=base_messages, subject=subject, max_tokens=SUMMARY_MAX_TOKENS)

    # Otherwise reduce in groups first, then recurse on the (smaller) set of group summaries.
    groups = _group_by_budget(partials, max(1, limit - base_overhead - 64), llm.estimate_tokens)
    reduced: List[str] = []
    for group in groups:
        sub = "\n\n".join(f"Part {i}: {s}" for i, s in enumerate(group, start=1))
        reduced.append(summarise(llm, summary_cfg, sub, LENGTH_PARAGRAPHS,
                                 base_messages=base_messages, subject=subject, max_tokens=SUMMARY_MAX_TOKENS))
    return _reduce_summaries(llm, summary_cfg, base_messages, reduced, language, limit, base_overhead)


def _fill_summary_instruction(desc_spec: Optional[str], language: str, seed: Optional[str]) -> str:
    """
    Build the file-description instruction for the summary turn, filling `{language}` and `{seed}`.

    Substitution is literal (so an existing description carrying braces is safe). When `seed` is given, a clause is
    woven in asking the model to keep accurate wording and correct or extend the rest, so the unified summary
    incorporates the author's existing file description.

    Parameters:
    - `desc_spec`: The instruction template (defaults to `SUMMARY_INSTRUCTION`).
    - `language`: The source language identifier.
    - `seed`: The existing file description to fold in, or None.

    Returns:
    - The filled instruction text.
    """

    template = desc_spec if desc_spec is not None else SUMMARY_INSTRUCTION
    seed_clause = ""
    if seed and seed.strip():
        seed_clause = (
            f' The file already carries this description: "{seed.strip()}". Keep any wording that is still accurate '
            f"and correct or extend the rest."
        )
    return template.replace("{language}", language).replace("{seed}", seed_clause)


# A line that opens with a list/heading marker: numbered ("1." / "2)"), bulleted ("- "/"* "/"+ "/"• "), a markdown
# heading ("## "), or a bold label ("**Key operations**:"). The file-DESCRIPTION spec asks for flowing prose, but a
# small model summarising a large file via map-reduce sometimes returns a structured list anyway - this catches it.
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:\d+[.)]\s|[-*+•]\s|#{1,6}\s|\*\*[^*\n]+\*\*\s*:)")


def _looks_listy(text: str) -> bool:
    """Return True if `text` reads as a list/headings rather than flowing prose (>= 2 list/heading markers)."""
    return len(_LIST_MARKER_RE.findall(text or "")) >= 2


def _strip_list_markers(text: str) -> str:
    """Deterministically remove leading list/heading markers and bold emphasis (the last-resort de-list fallback)."""
    out: List[str] = []
    for ln in (text or "").split("\n"):
        s = re.sub(r"^\s*(?:\d+[.)]|[-*+•]|#{1,6})\s+", "", ln)   # leading number/bullet/heading marker
        s = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", s)                      # **bold** -> bold
        out.append(s)
    return "\n".join(out)


def _reflow_if_listy(llm: LocalChatModel, cfg: GenerationConfig, base_messages: Messages, text: str,
                     max_tokens: int) -> str:
    """
    If a generated file description came back as a list/headings, ask the model once to rewrite it as flowing prose.

    The summary spec asks for prose, but the small model occasionally ignores that on a large (map-reduced) file. One
    reflow turn usually fixes it; if it still looks listy (or comes back empty) the markers are stripped
    deterministically so the final description never carries list/heading syntax into a doc-comment.

    Parameters:
    - `llm`/`cfg`: The model and base generation config.
    - `base_messages`: The priming context to rewrite against.
    - `text`: The candidate description.
    - `max_tokens`: The reply-token cap for the reflow turn.

    Returns:
    - A description in flowing prose (reflowed, or marker-stripped as a fallback), or the original if it was fine.
    """

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
    Summarise a whole source file as a file-level DESCRIPTION, falling back to chunked map-reduce when it is too large.

    The result is the one piece of prose used both as the `--file-doc` header and as the per-routine priming context,
    so it is written to a file-description spec (`desc_spec`, default `SUMMARY_INSTRUCTION`) rather than a generic
    summary. Files that fit the window are described in a single request. Larger files are split into context-sized
    chunks, each summarised independently (the "map" step, kept thorough/generic to preserve detail), the partials are
    combined into one overall summary (the "reduce" step, recursive if needed), and then a single final turn reshapes
    that overall summary to the description spec - so the map-reduce keeps the detail while only the last turn applies
    the description shape (and folds in any `seed`).

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The base generation configuration (cloned with a smaller `max_new_tokens` for summaries).
    - `base_messages`: The primed context to summarise against (typically the system prompt); not mutated.
    - `source_blob`: The complete source text.
    - `language`: The source language identifier (used to phrase the prompts).
    - `desc_spec`: The file-description instruction template (default `SUMMARY_INSTRUCTION`).
    - `seed`: An existing file description to incorporate, or None.
    - `skeleton`: Whether `source_blob` is a structural skeleton (signatures, docs, header comments - no bodies)
      rather than the complete file; only the prompt's subject wording changes.
    - `capture`: Optional dict; when the map-reduce path runs, its pre-shaping thorough summary is recorded under
      `"thorough"` (richer context for the header-reword manifest than the shaped description).

    Returns:
    - The file-description summary text.
    """

    summary_cfg = replace(cfg, max_new_tokens=SUMMARY_MAX_TOKENS)
    limit = llm.n_ctx - llm.ctx_margin - SUMMARY_MAX_TOKENS
    base_overhead = llm.count_tokens(base_messages) if base_messages else 0
    description_instruction = _fill_summary_instruction(desc_spec, language, seed)
    what = (f"a structural skeleton of a {language} source file - its header comments, signatures, and existing "
            f"documentation, with the function bodies omitted") if skeleton else f"a complete {language} source file"

    # Fast path: the whole text fits in a single turn, written straight to the description spec.
    if base_overhead + llm.estimate_tokens(source_blob) + SUMMARY_WRAPPER_TOKENS <= limit:
        result = summarise(llm, summary_cfg, source_blob, LENGTH_PARAGRAPHS, base_messages=base_messages,
                           subject=what, max_tokens=SUMMARY_MAX_TOKENS,
                           instruction=description_instruction)
        return _reflow_if_listy(llm, summary_cfg, base_messages, result, SUMMARY_MAX_TOKENS)

    # Map: summarise the text in context-sized chunks (64 tokens of headroom for the wrapper text). The map/reduce
    # steps stay thorough/generic so no detail is lost before the final description-shaping turn.
    chunk_budget = max(1, limit - base_overhead - 64)
    chunks = _split_source(source_blob, chunk_budget, llm.estimate_tokens)
    echo(f"Source too large for a single-pass summary; summarising in {len(chunks)} chunk(s)...")

    partials: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        partials.append(summarise(llm, summary_cfg, chunk, LENGTH_PARAGRAPH, base_messages=base_messages,
                                  subject=f"chunk {idx} of {len(chunks)} of {what}",
                                  max_tokens=SUMMARY_MAX_TOKENS))
        echo(f"Summarised part {idx}/{len(chunks)}")

    # Reduce the partials into one overall summary, then reshape it to the file-description spec in a final turn.
    overall = _reduce_summaries(llm, summary_cfg, base_messages, partials, language, limit, base_overhead)
    if capture is not None:
        capture["thorough"] = overall
    result = summarise(llm, summary_cfg, overall, LENGTH_PARAGRAPHS, base_messages=base_messages,
                       subject=f"a draft overview of a {language} source file", max_tokens=SUMMARY_MAX_TOKENS,
                       instruction=description_instruction)
    return _reflow_if_listy(llm, summary_cfg, base_messages, result, SUMMARY_MAX_TOKENS)


def _head_crop(text: str, llm: LocalChatModel, budget_tokens: int) -> str:
    """
    Keep the leading lines of `text` that fit a token budget (a cheap head crop for a one-line direct summary).

    A file's top - includes/imports and the first definitions - carries enough to say what the file is, so the head
    is kept and the tail dropped when the whole file would not fit. Returned unchanged if it already fits.

    Parameters:
    - `text`: The source text.
    - `llm`: A model exposing `estimate_tokens`.
    - `budget_tokens`: The maximum estimated tokens to keep.

    Returns:
    - The text, or its leading lines up to the budget.
    """

    if llm.estimate_tokens(text) <= budget_tokens:
        return text
    kept: List[str] = []
    for line in text.splitlines():
        kept.append(line)
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
    Return the full file-description summary, using the file-backed cache when possible and generating it otherwise.

    This is the one piece of prose shared by the `--file-doc` header and the block-pass priming context (the summary
    is slow to produce). `source_blob` is whatever text the description is to be generated from - normally the file's
    structural skeleton (see `scale_project.render_skeleton`; `skeleton=True`), or the whole file when it has no
    symbols to skeletonise. The cache is keyed on that content, so a different skeleton (e.g. the re-rendered one
    after the function passes) regenerates rather than reusing a stale description; neither `base_messages` nor
    `seed` affects cache identity, and on a cache hit the `seed` is moot (the description already exists).

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The base generation configuration.
    - `scale_path`: The SCALE configuration directory (for the optional `summary.txt` instruction override).
    - `src_path`: The source file path (the cache key).
    - `source_blob`: The text to describe (the skeleton, or the complete source).
    - `language`: The source language identifier.
    - `base_messages`: The priming context a freshly generated summary is produced against (typically the system prompt).
    - `no_cache`: When True, regenerate the summary rather than loading a cached one.
    - `seed`: An existing file description to fold into a freshly generated summary, or None.
    - `skeleton`: Whether `source_blob` is a skeleton (adjusts the prompt's subject wording only).

    Returns:
    - The full file-description summary text.
    """

    summary_cache = SummaryCache(src_path, source_blob)
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
    Return the squashed file description used to prime the (context-starved) definition pass.

    The short summary is a one/two-sentence note giving quick context while spending almost no window - the definition
    pass cares far more about the routine body than a detailed file overview. When a full description already exists
    (this file is a target whose file-doc/block pass produced one) it is condensed from that, so the two stay
    consistent; otherwise (a reference file, or a `-c`-only run) it is generated straight from the source, so we never
    pay for a full map-reduced description only to squash it. Cached alongside the full summary (same content-hash key).

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The base generation configuration.
    - `scale_path`: The SCALE configuration directory (for the optional `summary.short.txt` instruction override).
    - `src_path`: The source file path (the cache key).
    - `source_blob`: The text to describe (the file's skeleton, or the complete source - see `_get_file_summary`).
    - `language`: The source language identifier.
    - `base_messages`: The priming context the condensation is produced against.
    - `no_cache`: When True, regenerate rather than loading from cache.
    - `skeleton`: Whether `source_blob` is a skeleton (adjusts the prompt's subject wording only).

    Returns:
    - The short file-description summary text.
    """

    cache = SummaryCache(src_path, source_blob)
    if no_cache is False and cache.short:
        echo("Loaded short source summary from cache...")
        return cache.short

    instruction = (_read_optional(scale_path / "summary.short.txt") or SHORT_SUMMARY_INSTRUCTION).replace(
        "{language}", language)
    short_cfg = replace(cfg, max_new_tokens=SHORT_SUMMARY_MAX_TOKENS)

    # Lazy: if a full file description already exists (a target whose file-doc/block pass produced it), condense that
    # so the short stays consistent with the header. Otherwise - a read-only reference, or a definition-only (`-c`)
    # run that never needs the full - summarise the source DIRECTLY into one line, rather than generating a full
    # (possibly map-reduced) description only to squash it and throw it away.
    full = cache.summary if (no_cache is False and cache.summary) else None
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
    Prepare the Large Language Model (LLM) for generating comments by priming it with a system prompt and the source file as context.

    Parameters:
    - `llm`: The LocalChatModel instance to be primed.
    - `cfg`: The GenerationConfig instance controlling the LLM's behavior.
    - `scale_path`: The path to the SCALE tool's configuration directory.
    - `src_path`: The path to the source file to be annotated.
    - `source_blob`: The source code as a string.
    - `language`: The programming language of the source code.
    - `no_cache`: Generate new summary - don't use the cached version.
    - `project_context`: Optional project background (the project blurb plus any related-file one-liners; see
      `scale_project`); injected before the file summary so both the summary and the per-routine turns understand the
      file's place in the wider project.
    - `template`: Which per-language style template to load as the final priming turn: `"comment"` loads
      `comment.<lang>.txt` (the definition/docstring pass) and `"blocks"` loads `blocks.<lang>.txt` (the
      within-function "blob" pass). Only one is ever loaded so the two passes never share each other's guidance.
    - `skeleton`: Optional structural skeleton of the file (`scale_project.render_skeleton`). When given, the
      whole-file description is generated from it instead of the full source - a fraction of the tokens, so a single
      call suffices where map-reduce used to kick in. None (a file with no symbols) keeps the whole-file path.

    Returns:
    - A list of messages exchanged between the system and the LLM, including the priming prompts and the generated responses.
    """

    echo("Priming LLM...")

    # Load the system prompt for doing comment generation. For the definition pass, append the house-style guidelines
    # (the doc-comment rules and density guidance it depends on). The block pass deliberately omits them: its own
    # `blocks.<lang>.txt` template carries the blob guidance, and the small context window is better spent on the
    # routine snippet than on def-pass doc-comment rules that do not apply.
    comment_path = scale_path / "comment.txt"
    comment_prompt = comment_path.read_text(encoding="utf-8")
    guidelines_path = scale_path / "guidelines.md"
    if template == "comment" and guidelines_path.is_file():
        comment_prompt = f"{comment_prompt}\n\n{guidelines_path.read_text(encoding='utf-8')}"

    # Load the user prompt specifying the style template for the language in question. The pass selects the template:
    # the definition pass uses "comment", the within-function blob pass uses "blocks".
    template_path = scale_path / f"{template}.{language}.txt"
    template_prompt = template_path.read_text(encoding="utf-8")

    # Prime the LLM with our system prompt
    messages = []
    messages.append({"role": "system", "content": comment_prompt})

    # Inject the project context (if any) before the file summary, so the broader-project background informs both the
    # generated summary and every routine turn that follows (see scale_project).
    if project_context:
        messages.append({"role": "user", "content":
            "Here is some background on the wider project this file belongs to:\n\n" + project_context})
        messages.append({"role": "assistant", "content": PRIMING_ACK})

    # Name the file (and, for C, whether it is a header or an implementation) so the summary and every routine/block
    # turn write to the file's role - a header's external contract vs an implementation's internal detail.
    messages.append({"role": "user", "content": _file_identity_note(src_path, language)})
    messages.append({"role": "assistant", "content": PRIMING_ACK})

    # Now summarise the file for context - from its structural skeleton when one is available (signatures + docs, a
    # fraction of the tokens), else the whole file (chunked map-reduce kicks in automatically for large files; see
    # _generate_file_summary). The summary is generated against the system prompt alone (the only turn so far). The
    # definition pass is context-starved - the routine body matters more there - so it gets the SHORT (squashed) file
    # description; the block pass has more room, so it gets the full one.
    summary_source = skeleton if skeleton else source_blob
    if template == "comment":
        summary = _get_short_summary(llm, cfg, scale_path, src_path, summary_source, language, messages,
                                     no_cache=no_cache, skeleton=bool(skeleton))
    else:
        summary = _get_file_summary(llm, cfg, scale_path, src_path, summary_source, language, messages,
                                    no_cache=no_cache, skeleton=bool(skeleton))

    reply_length = 1 + summary.count("\n")
    echo(f"Source file summarised? {reply_length} lines of summary created")
    echo(f"\n{summary}\n")

    # Establish the working context with plain turns, each followed by a fixed acknowledgement we supply ourselves
    # (see PRIMING_ACK) - the model is never asked to generate an "OK", so its first *generated* turn is a real comment.
    messages.append({"role": "user", "content":
        "To give you context, here is an overview of what the program as a whole does:\n\n"
        f"{summary}"})
    messages.append({"role": "assistant", "content": PRIMING_ACK})

    # The preferred comment format / style template.
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
    Run the definition (docstring/header-comment) pass for the resolved language.

    Parameters:
    - `llm`: The LocalChatModel instance used for generating comments.
    - `cfg`: The GenerationConfig instance containing configuration settings.
    - `messages`: The primed conversation context for this pass.
    - `source_blob`: The complete source text to annotate (with original line endings).
    - `source_lines`: The same source text split into individual lines.
    - `language`: The programming language identifier (already validated).
    - `doc_order`/`callee_context`/`on_doc`: Optional call-graph hooks threaded to the worker (all three languages);
      absent, the worker's behaviour is unchanged.
    - `doc_plan`: Optional per-file `--doc-site` plan (C only); redirects docs to header prototypes and skips the
      redirected definitions' docstrings.
    - `verifier`: Optional `scale_verify.Verifier` (the grounding gate + challenge turns), threaded to every worker.

    Returns:
    - The annotated source split into individual lines.
    """

    if language == "python":
        from scale_python import generate_language_comments
    elif language == "js":
        from scale_javascript import generate_language_comments
    elif language == "c":
        from scale_c import generate_language_comments
        # Only the C worker has a header/implementation doc-site plan.
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
    Return the per-language block-target provider used by the within-function "blob" pass.

    The provider has the signature `provider(source_blob, source_lines) -> List[BlockTarget]` and identifies, for each
    function/method/class body, the lines that may legally begin a block (see `scale_blocks`).

    Parameters:
    - `language`: The programming language identifier (already validated).

    Returns:
    - The provider callable for `language`.

    Raises:
    - `NotImplementedError`: If the language has no block provider yet (currently only Python is wired).
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
    raise NotImplementedError(
        f"The block pass does not yet support '{language}'."
    )


def _symbol_provider_for(language: str):
    """
    Return the per-language `iter_symbols` provider for the call-graph pre-pass, or None if the language has none.

    The provider has the signature `provider(source_blob, source_lines) -> List[scale_project.Symbol]` and is
    model-free. None is returned (rather than raising) for an unsupported language so the pre-pass can simply skip a
    file it cannot parse for symbols, without aborting the run.

    Parameters:
    - `language`: The programming language identifier (already validated).

    Returns:
    - The provider callable, or None.
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
    return None


def _scan_run_files(targets: List[Path], references: List[Path], language_arg: Optional[str]):
    """
    Load and parse every run file exactly once, returning the retained run-file store (the merged model-free pre-pass).

    Thin binding of `scale_project.scan_run_files` to this module's loader and symbol providers. The store feeds the
    call graph, the C doc-site plan, and the lazy callee one-liner generator, so no pre-pass re-loads or re-parses a
    file.

    Parameters:
    - `targets`: The files to be annotated.
    - `references`: The read-only reference files (consulted, never written).
    - `language_arg`: The forced `--language` value (lowercased), or None to auto-detect per file.

    Returns:
    - The `{file key -> scale_project.RunFile}` store (possibly empty).
    """

    return scale_project.scan_run_files(
        targets, references,
        load=lambda p: load_source(p, language_arg),
        provider_for=_symbol_provider_for,
    )


def _build_call_graph(run_files: Dict[str, "scale_project.RunFile"]):
    """
    Build the run's call graph and contract store from the retained run-file store (model-free).

    The resulting `ProjectGraph` drives leaf-first documentation order, callee-context injection, and the block pass's
    read-side call annotations; the seeded `ContractStore` carries each routine's one-line contract (from existing docs
    at first, then refined as the def pass writes docstrings or the lazy generator fills a gap).

    Parameters:
    - `run_files`: The store from `_scan_run_files`.

    Returns:
    - A `(graph, store)` pair, or `(None, None)` if the store is empty.
    """

    if not run_files:
        return None, None
    graph = scale_project.build_project_graph({k: rf.symbols for k, rf in run_files.items()})
    return graph, scale_project.ContractStore(graph)


def _build_c_doc_plan(run_files: Dict[str, "scale_project.RunFile"], policy: str):
    """
    Build the C header/implementation documentation-site plan from the retained run-file store (model-free).

    The plan decides where each C function's doc lives (see `scale_c.plan_doc_sites_c`). References supply
    bodies/prototypes but are never written (only target prototypes are documentation sites).

    Parameters:
    - `run_files`: The store from `_scan_run_files`.
    - `policy`: The `--doc-site` policy (`"auto"` or `"impl"`).

    Returns:
    - A `scale_c.CDocPlan`, or None when the run has no C file.
    """

    files = [(rf.key, rf.is_target, rf.source_blob, rf.source_lines)
             for rf in run_files.values() if rf.language == "c"]
    if not files:
        return None
    from scale_c import plan_doc_sites_c
    return plan_doc_sites_c(files, policy)


def _make_callee_oneliner_context(llm: LocalChatModel, cfg: GenerationConfig, run_files, graph, store):
    """
    Bind the lazy callee one-liner generator over the run, returning the def pass's `callee_context` lookup.

    The returned `context(file_key, qualname)` is what each file's `callee_context` hook closes over: before formatting
    the routine's callee notes, it generates a one-line contract for any **resolved callee that has none** - reading
    the callee's signature+body from the retained run-file store, eliding it to the context budget with the language's
    existing mechanism (`elide_structurally` for Python, the `fit_snippet` crop for C/JS), and making a single
    `summarise(..., LENGTH_LINE)` call. Generation is **lazy** (it happens only when a caller is being documented, so a
    routine nothing calls is never summarised), **shallow** (one level - the callee's own callees are not recursed
    into), and **cached** (the result is stored via `store.update`, and a callee that yields nothing is not retried),
    so the cost is bounded by the number of distinct used-but-undocumented callees in the run.

    Parameters:
    - `llm`/`cfg`: The model and base generation configuration.
    - `run_files`: The retained run-file store (`_scan_run_files`).
    - `graph`/`store`: The run's `ProjectGraph` and `ContractStore`.

    Returns:
    - `context(file_key, qualname) -> notes` (the callee-contract block for that routine, possibly "").
    """

    attempted: set = set()

    def generate_oneliner(sym) -> str:
        """Summarise one callee's body from the run-file store into a one-line contract ("" when it cannot be)."""
        rf = run_files.get(sym.file)
        if rf is None or sym.end < sym.start or sym.end <= 0:
            return ""
        snippet = "\n".join(rf.source_lines[sym.start - 1:sym.end])
        if not snippet.strip():
            return ""
        header_lines = max(1, len(sym.signature.split("\n")))
        # Fit the body to the (empty-context) snippet budget so a large callee cannot blow the window. Python keeps
        # the routine's shape by summarising its deepest suites; C/JS crop the middle. Dedent first so a nested
        # routine still parses for the structural path (its patcher is not involved - this is a read-only view).
        if rf.language == "python":
            from scale_python import elide_structurally
            snippet, _ = elide_structurally(llm, cfg, [], textwrap.dedent(snippet), header_lines, MARKER_PYTHON)
        else:
            marker = MARKER_C if rf.language == "c" else MARKER_JS
            snippet, _ = fit_snippet(llm, cfg, [], snippet, header_lines, marker)
        simple = sym.qualname.rsplit(".", 1)[-1]
        return summarise(
            llm, cfg, snippet, LENGTH_LINE,
            subject=f"a {rf.language} routine named `{simple}`, called by code that is being documented",
            instruction="In one short line, state what this routine does for its caller - just the line, no preamble.",
        )

    def context(file_key: str, qualname: str) -> str:
        """Fill any missing callee contracts for this routine (lazily, once each), then return its callee notes."""
        for key in store.missing_callee_contracts(file_key, qualname):
            if key in attempted:
                continue                                   # tried before and yielded nothing - do not retry
            attempted.add(key)
            sym = graph.symbols.get(key)
            if sym is None:
                continue
            one = generate_oneliner(sym)
            if one:
                echo(f"[callgraph] Generated one-liner for undocumented callee '{key[1]}'")
                store.update(key[0], key[1], one)
        return store.callee_notes(file_key, qualname)

    return context


def _block_callee_notes(provider, source_blob: str, source_lines: List[str], file_key: str, graph, store):
    """
    Build the block pass's read-side call annotations for one file: `{qualname -> {line -> "callee: one-liner"}}`.

    The def pass may have shifted lines since the pre-pass parsed the original file, so the call-site lines recorded in
    the graph are stale; instead the **current** text is re-parsed with the language's `iter_symbols` (model-free) and
    each fresh call site is matched to its resolved callee by `(name, kind)` via `graph.call_map` - giving the
    annotation the call's line in the text the block pass actually reads. Only calls whose callee has a contract in
    the store contribute; several noted calls on one line are joined with "; ".

    Parameters:
    - `provider`: The language's `iter_symbols`.
    - `source_blob`/`source_lines`: The CURRENT (def-pass-annotated) text the block pass will parse.
    - `file_key`: The file's run key.
    - `graph`/`store`: The run's `ProjectGraph` and `ContractStore`.

    Returns:
    - The per-routine line-annotation maps (empty entries omitted).
    """

    notes: Dict[str, Dict[int, str]] = {}
    for sym in provider(source_blob, source_lines):
        cmap = graph.call_map.get((file_key, sym.qualname))
        if not cmap:
            continue
        per_line: Dict[int, List[str]] = {}
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
        if per_line:
            notes[sym.qualname] = {ln: "; ".join(ns) for ln, ns in per_line.items()}
    return notes


def _order_header_before_impl(targets: List[Path], pairs: List[Tuple[str, str]]) -> List[Path]:
    """
    Reorder target files so each header precedes the implementation it is paired with, preserving the prior order.

    A stable topological sort over the `(header_key, impl_key)` constraints with the current position as the tiebreak,
    so the call-graph file order is kept except where a header must move ahead of its impl. Any cycle (which a clean
    header/impl split cannot produce) collapses harmlessly via the shared SCC ordering.

    Parameters:
    - `targets`: The current target ordering.
    - `pairs`: `(header_file_key, impl_file_key)` constraints from the doc-site plan.

    Returns:
    - The targets reordered.
    """

    if not pairs:
        return targets
    keys = [str(t.resolve()) for t in targets]
    by_key = {str(t.resolve()): t for t in targets}
    idx = {k: i for i, k in enumerate(keys)}
    kset = set(keys)
    succ: dict = {k: set() for k in keys}
    for hk, ik in pairs:
        if hk in kset and ik in kset and hk != ik:
            succ[hk].add(ik)                       # the header precedes the implementation
    ordered = scale_project._leaf_first_order(keys, succ, tiebreak=lambda k: idx[k])
    return [by_key[k] for k in ordered]


def _block_style_for(language: str, comment_style: str = "line"):
    """
    Return the comment-style descriptor that drives block-comment rendering for `language`.

    Parameters:
    - `language`: The programming language identifier (already validated).
    - `comment_style`: `"line"` or `"block"` - the `--block-comment-style` choice. Ignored for Python (which has
      no block-comment syntax); for C/JS it selects `//` line comments or `/* ... */` block comments.

    Returns:
    - A `scale_blocks.CommentStyle` describing the language's comment delimiters.

    Raises:
    - `NotImplementedError`: If the language has no block provider yet.
    """

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
    Return the text of a config file if it exists, otherwise None.

    Used for the user-editable block-pass prompt overrides, which fall back to the built-in defaults when absent.

    Parameters:
    - `path`: The file to read.

    Returns:
    - The file contents, or None if the file does not exist.
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
    Run the within-function "blob" pass: annotate logical groups of statements inside each routine body.

    The block-pass prompt wording is loaded from `scale-cfg` so users can tune it without touching code (a missing
    file falls back to the engine's built-in default): `blocks.segment.txt`, `blocks.comment.txt`,
    `blocks.comment.nudge.txt` (the retry nudge), `blocks.note.short.txt` / `blocks.note.long.txt` (the short-/
    long-routine notes), and `blocks.score.txt` (the value-score turn).

    Parameters:
    - `llm`: The LocalChatModel instance used for generating comments.
    - `cfg`: The GenerationConfig instance containing configuration settings.
    - `scale_path`: The path to the SCALE configuration directory.
    - `messages`: The primed conversation context for this pass.
    - `source_blob`: The complete source text to annotate (with original line endings).
    - `source_lines`: The same source text split into individual lines.
    - `language`: The programming language identifier (already validated).
    - `callee_annotations`: Optional per-routine call annotations (`_block_callee_notes`) shown read-side on the
      paragraphs' call lines.

    Returns:
    - The annotated source split into individual lines.
    """

    from scale_blocks import annotate_blocks
    provider = _block_provider_for(language)
    style = _block_style_for(language, comment_style)
    # The C provider accepts a `--doc-site` header-doc override (so a redirected implementation's block pass still has
    # the routine's contract); other providers do not take it.
    if doc_override is not None and language == "c":
        targets = provider(source_blob, source_lines, doc_override=doc_override)
    else:
        targets = provider(source_blob, source_lines)
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
    Return the per-language file-doc target provider, or raise if the language is not yet supported.

    The provider has the signature `provider(source_blob, source_lines) -> Optional[FileDocTarget]` and describes the
    file's leading-comment zone (see `scale_filedoc`).

    Parameters:
    - `language`: The programming language identifier (already validated).

    Returns:
    - The provider callable for `language`.

    Raises:
    - `NotImplementedError`: If the language has no file-doc provider yet.
    """

    if language == "c":
        from scale_c import file_doc_target_c
        return file_doc_target_c
    if language == "js":
        from scale_javascript import file_doc_target_js
        return file_doc_target_js
    if language == "python":
        from scale_python import file_doc_target_py
        return file_doc_target_py
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
    Run the file-level header doccomment pass: add or update the top-of-file description, preserving everything else.

    This pass runs LAST in the per-file pipeline (after the definition and block passes), because the published
    description is generated from the file's CURRENT skeleton - now rich with the freshly written docstrings - rather
    than from the bodies (the two-pass description model: pass 1, the priming-grade description from the original
    skeleton, fed the function passes; this is pass 2). The local model only classifies which existing header lines
    are the description; a file with no symbols falls back to summarising its whole text. The deterministic patcher
    in `scale_filedoc` preserves shebang/copyright/license/boilerplate byte-for-byte (see that module).

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The GenerationConfig instance.
    - `scale_path`: The path to the SCALE configuration directory.
    - `src_path`: The source file path (the summary cache key).
    - `source_blob`: The CURRENT source text (the earlier passes' output).
    - `source_lines`: The same text split into lines.
    - `language`: The programming language identifier (already validated).
    - `no_cache`: When True, regenerate the summary rather than loading a cached one.
    - `on_file_doc`: Optional callback `(spliced_description, thorough_or_None)` fired after a successful splice -
      the header-reword manifest (`--emit-reword`) records these as the file's draft (and its richer map-reduce
      context, when one was produced).

    Returns:
    - The annotated source split into lines (unchanged if the language is unsupported or there is nothing to do).
    """

    from scale_filedoc import annotate_file_doc
    try:
        provider = _file_doc_target_for(language)
    except NotImplementedError as exc:
        error(str(exc))
        return source_lines

    target = provider(source_blob, source_lines)
    if target is None:
        echo("file-doc: nothing to annotate.")
        return source_lines

    # The published description's input: the CURRENT text's skeleton (model-free re-render - the def pass's fresh
    # docstrings are in it), or the whole current text when the file has no symbols (the binary guard).
    description_source = source_blob
    is_skeleton = False
    sym_provider = _symbol_provider_for(language)
    if sym_provider is not None:
        try:
            skel = scale_project.render_skeleton(source_lines, language, sym_provider(source_blob, source_lines))
        except Exception:
            skel = None      # an unparseable file simply falls back to the whole-text path
        if skel:
            description_source = skel
            is_skeleton = True

    # Minimal priming for the classify turn: the system prompt plus any project blurb (so the file description, which
    # is the whole-file summary, understands the file's place in the wider project). The description prose is fetched
    # (and cached) via the provider below, seeded with whatever existing description the classify turn finds.
    comment_prompt = (scale_path / "comment.txt").read_text(encoding="utf-8")
    base: Messages = [{"role": "system", "content": comment_prompt}]
    if project_context:
        base = base + [
            {"role": "user", "content":
                "Here is some background on the wider project this file belongs to:\n\n" + project_context},
            {"role": "assistant", "content": PRIMING_ACK},
        ]
    # Name the file and its role (header vs implementation) so the generated header description reads accordingly.
    base = base + [
        {"role": "user", "content": _file_identity_note(src_path, language)},
        {"role": "assistant", "content": PRIMING_ACK},
    ]

    capture: dict = {}

    def summary_provider(seed: Optional[str]) -> str:
        return _get_file_summary(
            llm, cfg, scale_path, src_path, description_source, language, base, no_cache=no_cache, seed=seed,
            skeleton=is_skeleton, capture=capture)

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
    Generate comments for the provided (already-loaded) source code using a Large Language Model (LLM).

    Up to three passes run in sequence on the current text. The definition pass (`do_comment`) writes/updates one
    docstring or header comment per routine; the block pass (`do_blocks`) annotates logical groups of statements
    inside routine bodies; the file-doc pass (`do_file_doc`) runs LAST, splicing the published file description -
    generated from the then-current skeleton, so it draws on the freshly written docstrings. Each pass primes its own
    fresh context with the appropriate style template so they never share each other's guidance, and each re-parses
    the previous pass's output so line spans stay valid. The result is written to the destination path (or stdout).

    Parameters:
    - `llm`: The LocalChatModel instance used for generating comments.
    - `cfg`: The GenerationConfig instance containing configuration settings.
    - `scale_path`: The path to the SCALE tool installation directory.
    - `src_path`: The path to the source file to be annotated.
    - `dst_path`: The path where the updated source file will be written (optional).
    - `source_blob`: The complete source text (with original line endings).
    - `source_lines`: The source text split into individual lines.
    - `line_ending`: The detected line ending used to re-join the output.
    - `language`: The programming language of the source code (already resolved and validated).
    - `no_cache`: Generate new summary - don't use the cached version.
    - `do_comment`: Run the definition (docstring/header-comment) pass.
    - `do_blocks`: Run the within-function block pass.
    - `doc_order`/`callee_context`/`on_doc`: Optional call-graph hooks for the definition pass (see `_def_pass`),
      bound by the caller over the shared `ContractStore` and this file. Absent, the def pass behaves as before.
    - `doc_plan`: Optional per-file `--doc-site` plan (C only) threaded to the def pass; `doc_override` is the matching
      block-pass header-doc lookup so a redirected implementation's block pass keeps the routine's contract.
    - `block_callee_notes`: Optional call-graph hook for the block pass: `(current_blob, current_lines) -> {qualname ->
      {line -> "callee: one-liner"}}`. Called on the def pass's output (lines have shifted, so the annotations must be
      derived from the current text) and shown read-side on the paragraphs' call lines. Absent, behaviour is unchanged.
    - `verifier`: Optional `scale_verify.Verifier` (the grounding gate + challenge turns), threaded to the definition
      and block passes. Absent, behaviour is unchanged.
    - `skeleton`: Optional pass-1 (priming-grade) skeleton of the ORIGINAL source (`scale_project.render_skeleton`),
      used to generate the description that primes the definition and block passes. None falls back to summarising
      the whole file (a no-symbol file, or a caller that did not pre-render one).
    - `on_file_doc`: Optional callback `(spliced_description, thorough_or_None)` fired by the file-doc pass after a
      successful splice (the `--emit-reword` collector; see `_file_doc_pass`).

    Returns:
    - 0 if the operation was successful, or an error number.

    Notes:
    - Supported languages are Python, JavaScript, and C (the block pass supports all three).
    - If the destination path is not provided, the generated comments will be printed to the console.
    """

    # The priming summary is generated from the original source's skeleton (or the whole original source) for both
    # function passes, so the content-hash cache stays warm; only the worker input advances from one pass to the next.
    new_lines = source_lines

    if do_comment:
        messages = prime_llm_for_comments(
            llm, cfg, scale_path, src_path, source_blob, language, no_cache=no_cache, template="comment",
            project_context=project_context, skeleton=skeleton,
        )
        current_blob = line_ending.join(new_lines)
        new_lines = _def_pass(llm, cfg, messages, current_blob, new_lines, language,
                              doc_order=doc_order, callee_context=callee_context, on_doc=on_doc, doc_plan=doc_plan,
                              verifier=verifier)

    if do_blocks:
        try:
            _block_provider_for(language)  # fail fast (and cleanly) on an unsupported language
        except NotImplementedError as exc:
            error(str(exc))
            return 1
        current_blob = line_ending.join(new_lines)
        messages = prime_llm_for_comments(
            llm, cfg, scale_path, src_path, source_blob, language, no_cache=no_cache, template="blocks",
            project_context=project_context, skeleton=skeleton,
        )
        # Derive the read-side call annotations from the CURRENT text (the def pass shifted lines), so each one lands
        # on the calling line the block pass actually reads.
        annotations = block_callee_notes(current_blob, new_lines) if block_callee_notes is not None else None
        new_lines = _block_pass(llm, cfg, scale_path, messages, current_blob, new_lines, language,
                                comment_style=block_comment_style, comment_value=comment_value,
                                doc_override=doc_override, callee_annotations=annotations, verifier=verifier)

    # The file-doc pass runs LAST: the published description is generated from the CURRENT text's skeleton, so it
    # draws on the docstrings the function passes just wrote (the two-pass description model). A top-of-file edit
    # cannot disturb the earlier passes - they have already run - and the pass primes its own minimal context.
    if do_file_doc:
        current_blob = line_ending.join(new_lines)
        new_lines = _file_doc_pass(
            llm, cfg, scale_path, src_path, current_blob, new_lines, language, no_cache=no_cache,
            project_context=project_context, on_file_doc=on_file_doc,
        )

    # Write the output
    if dst_path:
        out_bytes = line_ending.join(new_lines).encode("utf-8", errors="surrogateescape")
        dst_path.write_bytes(out_bytes)
        echo(f"Updated source written to {dst_path}")
    else:
        print("\n".join(new_lines))

    return 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments and return an argparse.Namespace object.

    This function takes optional command-line arguments, parses them using the ArgumentParser,
    and returns an argparse.Namespace object containing the parsed arguments.

    Parameters:
    - argv: Optional list of strings to parse as command-line arguments. If not provided, sys.argv[1:] is used.

    Returns:
    - An argparse.Namespace object containing the parsed arguments.
    """

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

    # The two modes: offline (default; everything local) vs online (everything deferred to a stronger model).
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

    # The online file-description round (a second, model-free manifest after --apply-manifest).
    p.add_argument("--emit-filedoc", default="", metavar="PATH",
                   help="Model-free: write the run-level file-description manifest - each target's current skeleton, "
                        "role, and header-zone lines - for a stronger model to fill (the online --file-doc round; "
                        "run it on the --apply-manifest outputs).")
    p.add_argument("--apply-filedoc", default="", metavar="PATH",
                   help="Model-free: splice each target's header description from this filedoc manifest (the answers' "
                        "classify range + prose), through the same license veto and preservation guard as --file-doc.")

    # The header-reword manifest (prose-only stronger-model reword of the offline --file-doc descriptions).
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
    Patch a stronger model's manifest answers into the source and write the result (the apply phase entry point).

    No model is loaded: the answers are inserted through the same insertion-only patchers as the local passes.

    Parameters:
    - `src_path`: The source file (the emit-phase output) being patched.
    - `dst_path`: Where to write the result, or None to print to stdout.
    - `language`: The resolved language identifier.
    - `source_lines`: The source split into individual lines.
    - `line_ending`: The detected line ending used to re-join the output.
    - `manifest`: The parsed manifest dictionary with answers filled in (its `requests` already filtered to this file).

    Returns:
    - 0 on success, or an error number.
    """

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

    if dst_path:
        dst_path.write_bytes(line_ending.join(new_lines).encode("utf-8", errors="surrogateescape"))
        echo(f"Applied manifest answers; written to {dst_path}")
    else:
        print("\n".join(new_lines))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Main entry point of the script. This function processes command-line arguments, prepares the model and configuration,
    and performs the segmentation and optional commenting and logging passes on the source code.

    Parameters:
    - argv: Optional sequence of command-line arguments. If not provided, defaults to None.
    - args: Parsed command-line arguments containing various settings such as model path, source file, output destination,
            and other options like format, context size, etc.

    Returns:
    - rc: Return code indicating the success or failure of the operations performed.
    """

    args = _parse_args(argv)

    set_verbosity(args.verbose)

    root = Path(__file__).resolve().parent
    scale_path = root / "scale-cfg"
    model = Path(args.model) if args.model else root / DEFAULT_MODEL
    dst_path = Path(args.output) if args.output else None

    # ---- Check phase: model-free completeness counter over a manifest (needs no targets, loads no model). ----
    if args.check_manifest:
        import json as _json
        raw = _json.loads(Path(args.check_manifest).read_text(encoding="utf-8"))
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
        for slot in missing:
            print(f"unfilled: {slot}")
        print(f"{len(missing)} unfilled answer(s) in {args.check_manifest} ({total} request(s))")
        out = sum(1 for r in manifest.get("requests", []) if r.get(scale_escalate.FRAGMENT_KEY))
        if out:
            print(f"{out} request(s) checked out to outstanding fragments")
        return 1 if missing else 0

    # ---- Fragment phase: check out the next batch of unfilled requests for a (parallel) filling agent. ----
    if args.next_fragment:
        master_path = Path(args.next_fragment)
        manifest = scale_escalate.read_manifest(master_path)
        if not scale_escalate.unfilled_answers(manifest):
            print(f"manifest complete: no unfilled answers in {master_path}")
            return 1
        name = scale_escalate.next_fragment_name(manifest, master_path.name)
        fragment = scale_escalate.build_fragment(manifest, args.fragment_size, name)
        if fragment is None:
            outstanding = sorted({str(r.get(scale_escalate.FRAGMENT_KEY))
                                  for r in manifest.get("requests", []) if r.get(scale_escalate.FRAGMENT_KEY)})
            print("no fragment available: every unfilled request is checked out "
                  f"({', '.join(outstanding)}); --apply-manifest merges and releases them")
            return 1
        frag_path = master_path.with_name(name)
        scale_escalate.write_manifest(frag_path, fragment)
        scale_escalate.write_manifest(master_path, manifest)  # persist the checkout markers + issue counter
        print(str(frag_path))
        return 0

    # Expand the target patterns (files / directories / globs) into a concrete, deduplicated, ordered file list.
    targets = scale_project.gather_files(args.source)
    if not targets:
        error(f"No source files matched: {' '.join(args.source)}")
        return 1

    # ---- Apply phase: model-free. Patch a stronger model's manifest answers into each target (the emit output). ----
    if args.apply_manifest:
        if dst_path is not None and len(targets) > 1:
            error("-o/--output cannot be used with multiple targets; multiple targets are patched in place.")
            return 1
        manifest = scale_escalate.read_manifest(Path(args.apply_manifest))

        # The parallel-fill protocol: fold sibling fragment answers back into the master first. Fragments imply
        # strict completeness - an unfilled slot is an error and is returned to the pile for --next-fragment.
        master_path = Path(args.apply_manifest)
        stem, dot, suffix = master_path.name.rpartition(".")
        if not dot:
            stem, suffix = master_path.name, "json"
        frag_paths = sorted(master_path.parent.glob(f"{stem}.frag-*.{suffix}"))
        fragmented = bool(frag_paths) or any(r.get(scale_escalate.FRAGMENT_KEY)
                                             for r in manifest.get("requests", []))
        if fragmented:
            for fp in frag_paths:
                merged = scale_escalate.merge_fragment(manifest, scale_escalate.read_manifest(fp))
                echo(f"Merged {merged} answer(s) from {fp.name}")
            missing = scale_escalate.unfilled_answers(manifest)
            released = scale_escalate.release_unfilled(manifest)
            scale_escalate.write_manifest(master_path, manifest)
            for fp in frag_paths:
                fp.unlink()  # spent: their answers now live in the master, written above
            if missing:
                for slot in missing:
                    print(f"unfilled: {slot}")
                error(f"{len(missing)} answer(s) still unfilled after merging {len(frag_paths)} fragment(s); "
                      f"{released} request(s) returned to the pile - hand them out again with --next-fragment.")
                return 1

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
            entry = file_entries.get(key, {})
            language = args.language.lower() if args.language else entry.get("language")
            _blob, source_lines, line_ending, language = load_source(target, language)
            if language not in SUPPORTED_LANGUAGES:
                error(f"Unsupported language '{language}'. SCALE supports: {', '.join(SUPPORTED_LANGUAGES)}")
                rc = 1
                continue
            sub_manifest = dict(manifest)
            sub_manifest["requests"] = requests
            out = dst_path if len(targets) == 1 else target
            frc = _apply_manifest_file(target, out, language, source_lines, line_ending, sub_manifest)
            if frc != 0:
                rc = frc
        return rc

    # ---- Reword-apply phase: model-free. Re-splice each target's header description from the reword manifest. ----
    if args.apply_reword:
        if dst_path is not None and len(targets) > 1:
            error("-o/--output cannot be used with multiple targets; multiple targets are patched in place.")
            return 1
        manifest = scale_reword.read_reword_manifest(Path(args.apply_reword))
        entries = {str(Path(f.get("path", "")).resolve()): f for f in manifest.get("files", [])}
        rc = 0
        for target in targets:
            entry = entries.get(str(target.resolve()))
            if entry is None or not str(entry.get("answer") or "").strip():
                echo(f"No reword answer for {target}; leaving it unchanged.")
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
                echo(f"reword: {target} has no header zone; leaving it unchanged.")
                continue
            new_lines, changed = scale_reword.apply_reword(
                source_lines, zone, entry.get("draft", ""), entry.get("answer", ""))
            out = dst_path if len(targets) == 1 else target
            if out:
                out.write_bytes(line_ending.join(new_lines).encode("utf-8", errors="surrogateescape"))
                echo(f"{'Reworded' if changed else 'Unchanged'}: written to {out}")
            else:
                print("\n".join(new_lines))
        return rc

    # ---- Filedoc emit phase: model-free. Write the run-level file-description manifest (the online --file-doc
    # round): each target's CURRENT skeleton (run this after --apply-manifest, so the fresh docs are in it), its
    # role, and its header zone's eligible lines, for a stronger model to classify-and-describe. ----
    if args.emit_filedoc:
        spec = _read_optional(scale_path / "summary.txt") or SUMMARY_INSTRUCTION
        project_doc_text = ""
        project_doc = scale_project.resolve_project_doc(args.project_doc, targets[0])
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

            # The skeleton of the CURRENT text (the applied docs are in it); a no-symbol file rides whole.
            skeleton = None
            sym_provider = _symbol_provider_for(language)
            if sym_provider is not None:
                try:
                    skeleton = scale_project.render_skeleton(
                        source_lines, language, sym_provider(source_blob, source_lines))
                except Exception:
                    skeleton = None      # an unparseable file simply falls back to the whole-text path
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

    # ---- Filedoc apply phase: model-free. Splice each target's header description from the filled manifest. ----
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

    # The block pass runs when spacing is requested or comments are requested (comments imply the pass). The comment
    # density maps to the 1-3 value threshold; with spacing only, 4 disables the comment turns (paragraph only).
    do_blocks = args.block_spacing or args.block_comments is not None
    block_threshold = BLOCK_COMMENT_LEVELS.get(args.block_comments, 4)

    # The two-mode contract: online and the manifest go together (the deferred requests must land somewhere, and a
    # manifest only exists to carry the online mode's requests), and the local-only passes have no online form.
    if args.online:
        if not args.emit_manifest:
            error("--online requires --emit-manifest PATH (the deferred comment requests must be written somewhere).")
            return 1
        if args.file_doc:
            error("--file-doc is a local pass; online, run the file-description round with --emit-filedoc / "
                  "--apply-filedoc after the manifest has been applied.")
            return 1
        if args.emit_reword:
            error("--emit-reword belongs to the offline --file-doc pass; online, use --emit-filedoc instead.")
            return 1
        if args.block_spacing and args.block_comments is None:
            error("--block-spacing alone is deterministic local work with nothing to defer; run it offline "
                  "(or add --block-comments to defer the comments).")
            return 1
    elif args.emit_manifest:
        error("--emit-manifest requires --online (offline runs are manifest-free; the manifest defers ALL routines).")
        return 1

    if not (args.comment or do_blocks or args.file_doc):
        # Nothing to do without --comment, --block-spacing/--block-comments, or --file-doc.
        return 0

    if dst_path is not None and len(targets) > 1:
        error("-o/--output cannot be used with multiple targets; multiple targets are annotated in place.")
        return 1
    if args.emit_reword and not args.file_doc:
        error("--emit-reword requires --file-doc (the manifest carries the freshly spliced descriptions).")
        return 1

    # Read-only reference files (consulted for context, never edited); a file that is also a target is not a reference.
    references = scale_project.gather_files(args.reference) if args.reference else []
    target_keys = {p.resolve() for p in targets}
    references = [r for r in references if r.resolve() not in target_keys]

    # ---- Online emit phase: model-free and instant (the GGUF is never loaded). Every routine in every target is
    # recorded as a manifest request - the def pass's docstring slot and/or the block pass's chunk recipe - and the
    # targets themselves are left byte-for-byte untouched (the apply phase patches the answers in later). ----
    if args.online:
        language_arg = args.language.lower() if args.language else None
        run_files = _scan_run_files(targets, references, language_arg)
        c_plan = _build_c_doc_plan(run_files, args.doc_site) if args.comment else None

        # One doc_style copy for the run: the house guidelines plus each target language's style template, so the
        # stronger model writes deferred docs to the same spec the local model is primed with.
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
            escalation = scale_escalate.Escalation(doc_style=emit_doc_style)
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

            # The emit output is byte-identical to the input: in place there is nothing to write; with -o (single
            # target) the original bytes are copied so the apply phase can be pointed at the copy.
            if dst_path is not None:
                dst_path.write_bytes(source_blob.encode("utf-8", errors="surrogateescape"))
                echo(f"Emit copy written to {dst_path}")

        manifest = scale_escalate.run_manifest(emit_parts, emit_doc_style)
        scale_escalate.write_manifest(Path(args.emit_manifest), manifest)
        echo(f"Wrote {len(manifest['requests'])} request(s) to {args.emit_manifest}")
        return rc

    # Prepare model and config (loaded once for the whole run).
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

    # Build the shared project context once for the whole run: the project blurb (auto-detected near the targets
    # unless --project-doc says otherwise; 'none' disables). References contribute below through the call graph - they
    # are parsed for resolution and seed contracts, never summarised whole.
    project_blurb = ""
    project_doc = scale_project.resolve_project_doc(args.project_doc, targets[0])
    if project_doc is not None:
        project_blurb = scale_project.project_blurb(llm, cfg, scale_path, project_doc, no_cache=args.no_cache)
    project_context = project_blurb

    # Load and parse every run file once (the retained store), then build the call graph over targets ∪ references
    # (model-free). The graph drives the definition pass's documentation order (callees/children first), the callee-
    # contract context injected into each routine's turn - generating a missing callee's one-liner lazily from the
    # retained store - and the block pass's read-side call annotations; the seeded store accumulates contracts across
    # files. Targets are reordered so a callee's file is documented before a caller's (coarse, by file).
    run_files: dict = {}
    graph = store = callee_oneliners = None
    language_arg = args.language.lower() if args.language else None
    if args.comment or do_blocks:
        run_files = _scan_run_files(targets, references, language_arg)
        graph, store = _build_call_graph(run_files)

    # The verification harness (the grounding gate + challenge turns): built over the run's source text so a
    # backticked identifier in a generated comment can be checked against everything the run can see. Default-on for
    # any pass that generates comments; --no-verify opts out.
    verifier = None
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
    if graph is not None and store is not None:
        callee_oneliners = _make_callee_oneliner_context(llm, cfg, run_files, graph, store)
        if args.comment:
            key_to_path = {str(t.resolve()): t for t in targets}
            targets = [key_to_path[k] for k in graph.file_order(list(key_to_path.keys()))]

    # Build the C header/implementation documentation-site plan (model-free, from the same retained store) and reorder
    # targets so a header is documented before the implementation it is paired with (so the impl's block pass can
    # reuse the header doc). Only the definition pass acts on it; a non-C run yields None and changes nothing.
    c_plan = None
    if args.comment:
        c_plan = _build_c_doc_plan(run_files, args.doc_site)
        if c_plan is not None and c_plan.pairs:
            targets = _order_header_before_impl(targets, c_plan.pairs)

    # Annotate each target in turn (single target -> -o or stdout; multiple targets -> in place).
    reword_entries: List[dict] = []

    rc = 0
    for target in targets:
        language = args.language.lower() if args.language else None
        source_blob, source_lines, line_ending, language = load_source(target, language)
        if language not in SUPPORTED_LANGUAGES:
            error(f"Skipping {target}: unsupported language '{language}' (SCALE supports: "
                  f"{', '.join(SUPPORTED_LANGUAGES)}).")
            continue

        # Bind the call-graph hooks for this file over the shared store (closing the file key into each closure). The
        # store accumulates across files, so a callee documented in an earlier-ordered target informs a later caller;
        # `callee_context` additionally generates a one-liner, lazily, for a resolved callee that has no contract yet.
        # The block-pass hook re-derives call-site lines from the current text, so annotations survive line shifts.
        doc_order = callee_context = on_doc = block_notes = None
        if graph is not None and store is not None:
            file_key = str(target.resolve())
            doc_order = graph.doc_order(file_key)
            callee_context = lambda q, fk=file_key: callee_oneliners(fk, q)
            on_doc = lambda q, doc, fk=file_key: store.update(fk, q, doc)
            if do_blocks:
                sym_provider = _symbol_provider_for(language)
                if sym_provider is not None:
                    block_notes = (lambda blob, lines, fk=file_key, p=sym_provider:
                                   _block_callee_notes(p, blob, lines, fk, graph, store))

        # Bind the C doc-site plan for this file (redirected defs / documentable prototypes) and the block-pass
        # header-doc override (so a redirected implementation's block pass still sees the routine's contract).
        doc_plan = doc_override = None
        if c_plan is not None and language == "c":
            doc_plan = c_plan.for_file(str(target.resolve()))
            doc_override = lambda name, p=c_plan: p.header_doc(name)

        # The pass-1 (priming-grade) skeleton, rendered model-free from the retained store's parse of the original
        # text. None (no symbols, or a file the store skipped) keeps the whole-file summary path.
        skeleton = None
        rf = run_files.get(str(target.resolve()))
        if rf is not None:
            skeleton = scale_project.render_skeleton(rf.source_lines, rf.language, rf.symbols)

        # The header-reword collector: each successful --file-doc splice records this file's draft description (and
        # any richer map-reduce context) so the run-level reword manifest can be written after the loop.
        on_file_doc = None
        if args.emit_reword:
            on_file_doc = (lambda desc, thorough, t=target, lang=language:
                           reword_entries.append({
                               "path": str(t), "language": lang, "role": _file_role(t, lang),
                               "draft": desc, "context": thorough, "answer": None}))

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

        # Propagate any header docs generated this file to the implementation's contract key, so a later caller that
        # resolved to the definition still sees the freshly-generated contract (a redirected def writes no docstring).
        if c_plan is not None and store is not None:
            for nm, doc in list(c_plan.header_docs.items()):
                ik = c_plan.impl_file.get(nm)
                if ik:
                    store.update(ik, nm, doc)

    # Serialise the header-reword manifest (the freshly spliced descriptions, for a cross-file consistency reword).
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
