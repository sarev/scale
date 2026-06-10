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
from scale_text import summarise, LENGTH_LINE, LENGTH_PARAGRAPH, LENGTH_PARAGRAPHS, PRIMING_ACK
import scale_escalate
import scale_project
from typing import Callable, List, Optional, Sequence, Tuple
import argparse
import hashlib
import pickle
import sys
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


def _generate_file_summary(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    base_messages: Messages,
    source_blob: str,
    language: str,
    desc_spec: Optional[str] = None,
    seed: Optional[str] = None,
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

    Returns:
    - The file-description summary text.
    """

    summary_cfg = replace(cfg, max_new_tokens=SUMMARY_MAX_TOKENS)
    limit = llm.n_ctx - llm.ctx_margin - SUMMARY_MAX_TOKENS
    base_overhead = llm.count_tokens(base_messages) if base_messages else 0
    description_instruction = _fill_summary_instruction(desc_spec, language, seed)

    # Fast path: the whole file fits in a single turn, written straight to the description spec.
    if base_overhead + llm.estimate_tokens(source_blob) + SUMMARY_WRAPPER_TOKENS <= limit:
        return summarise(llm, summary_cfg, source_blob, LENGTH_PARAGRAPHS, base_messages=base_messages,
                         subject=f"a complete {language} source file", max_tokens=SUMMARY_MAX_TOKENS,
                         instruction=description_instruction)

    # Map: summarise the file in context-sized chunks (64 tokens of headroom for the wrapper text). The map/reduce
    # steps stay thorough/generic so no detail is lost before the final description-shaping turn.
    chunk_budget = max(1, limit - base_overhead - 64)
    chunks = _split_source(source_blob, chunk_budget, llm.estimate_tokens)
    echo(f"Source too large for a single-pass summary; summarising in {len(chunks)} chunk(s)...")

    partials: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        partials.append(summarise(llm, summary_cfg, chunk, LENGTH_PARAGRAPH, base_messages=base_messages,
                                  subject=f"chunk {idx} of {len(chunks)} of a {language} source file",
                                  max_tokens=SUMMARY_MAX_TOKENS))
        echo(f"Summarised part {idx}/{len(chunks)}")

    # Reduce the partials into one overall summary, then reshape it to the file-description spec in a final turn.
    overall = _reduce_summaries(llm, summary_cfg, base_messages, partials, language, limit, base_overhead)
    return summarise(llm, summary_cfg, overall, LENGTH_PARAGRAPHS, base_messages=base_messages,
                     subject=f"a draft overview of a {language} source file", max_tokens=SUMMARY_MAX_TOKENS,
                     instruction=description_instruction)


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
) -> str:
    """
    Return the full file-description summary, using the file-backed cache when possible and generating it otherwise.

    This is the one piece of prose shared by the `--file-doc` header and the block-pass priming context (the summary
    is slow to produce). The cache is keyed on the source content, so neither `base_messages` nor `seed` affects cache
    identity; on a cache hit the `seed` is moot (the description already exists).

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The base generation configuration.
    - `scale_path`: The SCALE configuration directory (for the optional `summary.txt` instruction override).
    - `src_path`: The source file path (the cache key).
    - `source_blob`: The complete source text.
    - `language`: The source language identifier.
    - `base_messages`: The priming context a freshly generated summary is produced against (typically the system prompt).
    - `no_cache`: When True, regenerate the summary rather than loading a cached one.
    - `seed`: An existing file description to fold into a freshly generated summary, or None.

    Returns:
    - The full file-description summary text.
    """

    summary_cache = SummaryCache(src_path, source_blob)
    if no_cache is False and summary_cache.summary:
        echo("Loaded full source summary from cache...")
    else:
        echo("Generating full source summary...")
        desc_spec = _read_optional(scale_path / "summary.txt") or SUMMARY_INSTRUCTION
        summary_cache.summary = _generate_file_summary(
            llm, cfg, base_messages, source_blob, language, desc_spec=desc_spec, seed=seed)
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
) -> str:
    """
    Return the squashed file description used to prime the (context-starved) definition pass.

    The short summary is a one/two-sentence condensation of the full file description, so it stays consistent with it
    while spending almost no context - the definition pass cares far more about the routine body than a detailed file
    overview. It is cached alongside the full summary (same content-hash invalidation).

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The base generation configuration.
    - `scale_path`: The SCALE configuration directory (for the optional `summary.short.txt` instruction override).
    - `src_path`: The source file path (the cache key).
    - `source_blob`: The complete source text.
    - `language`: The source language identifier.
    - `base_messages`: The priming context the condensation is produced against.
    - `no_cache`: When True, regenerate rather than loading from cache.

    Returns:
    - The short file-description summary text.
    """

    cache = SummaryCache(src_path, source_blob)
    if no_cache is False and cache.short:
        echo("Loaded short source summary from cache...")
        return cache.short

    # Condense the full description (generated/cached on demand) into a quick one/two-sentence note.
    full = _get_file_summary(llm, cfg, scale_path, src_path, source_blob, language, base_messages, no_cache=no_cache)
    echo("Condensing the file description for the definition pass...")
    instruction = (_read_optional(scale_path / "summary.short.txt") or SHORT_SUMMARY_INSTRUCTION).replace(
        "{language}", language)
    short_cfg = replace(cfg, max_new_tokens=SHORT_SUMMARY_MAX_TOKENS)
    short = summarise(llm, short_cfg, full, LENGTH_LINE, base_messages=base_messages,
                      subject=f"a fuller description of a {language} source file",
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

    # Now summarise the whole file for context (chunked map-reduce kicks in automatically for large files; see
    # _generate_file_summary). The summary is generated against the system prompt alone (the only turn so far). The
    # definition pass is context-starved - the routine body matters more there - so it gets the SHORT (squashed) file
    # description; the block pass has more room, so it gets the full one (the same prose --file-doc puts in the header).
    if template == "comment":
        summary = _get_short_summary(llm, cfg, scale_path, src_path, source_blob, language, messages, no_cache=no_cache)
    else:
        summary = _get_file_summary(llm, cfg, scale_path, src_path, source_blob, language, messages, no_cache=no_cache)

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
    escalation=None,
    doc_order: Optional[List[str]] = None,
    callee_context: Optional[Callable[[str], str]] = None,
    on_doc: Optional[Callable[[str, str], None]] = None,
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
    - `escalation`: Optional `scale_escalate.Escalation` (Python only).
    - `doc_order`/`callee_context`/`on_doc`: Optional call-graph hooks threaded to the worker (all three languages);
      absent, the worker's behaviour is unchanged.

    Returns:
    - The annotated source split into individual lines.
    """

    if language == "python":
        from scale_python import generate_language_comments
        # Only the Python worker understands selective escalation for now.
        return generate_language_comments(llm, cfg, messages, source_blob, source_lines, escalation=escalation,
                                          doc_order=doc_order, callee_context=callee_context, on_doc=on_doc)
    elif language == "js":
        from scale_javascript import generate_language_comments
    elif language == "c":
        from scale_c import generate_language_comments
    else:
        raise ValueError(f"Unsupported language '{language}'")
    return generate_language_comments(llm, cfg, messages, source_blob, source_lines,
                                      doc_order=doc_order, callee_context=callee_context, on_doc=on_doc)


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
        f"The --blocks pass does not yet support '{language}'."
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


def _build_call_graph(targets: List[Path], references: List[Path], language_arg: Optional[str]):
    """
    Build the run's call graph and contract store from every target and reference file (the model-free pre-pass).

    Each file is parsed for its symbols (definitions + resolved-or-not call sites); a file that cannot be loaded or has
    no symbol provider is skipped. The resulting `ProjectGraph` drives leaf-first documentation order and callee-context
    injection; the seeded `ContractStore` carries each routine's one-line contract (from existing docs at first, then
    refined as the def pass writes docstrings).

    Parameters:
    - `targets`: The files to be annotated.
    - `references`: The read-only reference files (seed contracts only; never documented).
    - `language_arg`: The forced `--language` value (lowercased), or None to auto-detect per file.

    Returns:
    - A `(graph, store)` pair, or `(None, None)` if no file yielded symbols.
    """

    symbols_by_file: dict = {}
    for f in list(targets) + list(references):
        try:
            blob, lines, _le, lang = load_source(f, language_arg)
        except OSError:
            continue
        provider = _symbol_provider_for(lang) if lang in SUPPORTED_LANGUAGES else None
        if provider is None:
            continue
        symbols_by_file[str(f.resolve())] = provider(blob, lines)

    if not symbols_by_file:
        return None, None
    graph = scale_project.build_project_graph(symbols_by_file)
    return graph, scale_project.ContractStore(graph)


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
        f"The --blocks pass does not yet support '{language}'."
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
    escalation=None,
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

    Returns:
    - The annotated source split into individual lines.
    """

    from scale_blocks import annotate_blocks
    provider = _block_provider_for(language)
    style = _block_style_for(language, comment_style)
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
        escalation=escalation,
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
) -> List[str]:
    """
    Run the file-level header doccomment pass: add or update the top-of-file description, preserving everything else.

    The local model only classifies which existing header lines are the description; the description prose itself is
    the whole-file summary (the same prose that primes the block pass), seeded with the existing description so the
    author's wording is incorporated. The deterministic patcher in `scale_filedoc` preserves shebang/copyright/license/
    boilerplate byte-for-byte (see that module).

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The GenerationConfig instance.
    - `scale_path`: The path to the SCALE configuration directory.
    - `src_path`: The source file path (the summary cache key).
    - `source_blob`: The complete source text.
    - `source_lines`: The same source split into lines.
    - `language`: The programming language identifier (already validated).
    - `no_cache`: When True, regenerate the summary rather than loading a cached one.

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

    def summary_provider(seed: Optional[str]) -> str:
        return _get_file_summary(
            llm, cfg, scale_path, src_path, source_blob, language, base, no_cache=no_cache, seed=seed)

    return annotate_file_doc(
        llm, cfg, base, source_lines, target, summary_provider, language,
        classify_prompt=_read_optional(scale_path / "filedoc.classify.txt"),
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
    escalation=None,
    project_context: str = "",
    doc_order: Optional[List[str]] = None,
    callee_context: Optional[Callable[[str], str]] = None,
    on_doc: Optional[Callable[[str, str], None]] = None,
) -> int:
    """
    Generate comments for the provided (already-loaded) source code using a Large Language Model (LLM).

    Up to two passes run in sequence on the current text. The definition pass (`do_comment`) writes/updates one
    docstring or header comment per routine; the block pass (`do_blocks`) annotates logical groups of statements
    inside routine bodies. Each pass primes its own fresh context with the appropriate style template so the two never
    share each other's guidance, and the block pass re-parses the (possibly already-annotated) output of the first
    pass so line spans stay valid. The result is written to the destination path (or stdout).

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
    - `escalation`: Optional `scale_escalate.Escalation`; when supplied, complex routines are deferred to its manifest
      instead of being commented by the local model (the caller serialises the manifest afterwards).
    - `doc_order`/`callee_context`/`on_doc`: Optional call-graph hooks for the definition pass (see `_def_pass`),
      bound by the caller over the shared `ContractStore` and this file. Absent, the def pass behaves as before.

    Returns:
    - 0 if the operation was successful, or an error number.

    Notes:
    - Supported languages are Python, JavaScript, and C (the block pass supports all three).
    - If the destination path is not provided, the generated comments will be printed to the console.
    """

    # The summary is primed from the original source for every pass, so the content-hash cache stays warm; only the
    # worker input advances from one pass to the next.
    new_lines = source_lines

    # The file-doc pass runs first: a top-of-file comment edit does not disturb routine spans, and the later passes
    # then see the updated header. It primes its own minimal context (it does not need the def-pass guidelines).
    if do_file_doc:
        new_lines = _file_doc_pass(
            llm, cfg, scale_path, src_path, source_blob, new_lines, language, no_cache=no_cache,
            project_context=project_context,
        )

    if do_comment:
        messages = prime_llm_for_comments(
            llm, cfg, scale_path, src_path, source_blob, language, no_cache=no_cache, template="comment",
            project_context=project_context,
        )
        current_blob = line_ending.join(new_lines)
        new_lines = _def_pass(llm, cfg, messages, current_blob, new_lines, language, escalation=escalation,
                              doc_order=doc_order, callee_context=callee_context, on_doc=on_doc)

    if do_blocks:
        try:
            _block_provider_for(language)  # fail fast (and cleanly) on an unsupported language
        except NotImplementedError as exc:
            error(str(exc))
            return 1
        current_blob = line_ending.join(new_lines)
        messages = prime_llm_for_comments(
            llm, cfg, scale_path, src_path, source_blob, language, no_cache=no_cache, template="blocks",
            project_context=project_context,
        )
        new_lines = _block_pass(llm, cfg, scale_path, messages, current_blob, new_lines, language,
                                comment_style=block_comment_style, comment_value=comment_value, escalation=escalation)

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
    - model: Path to GGUF model file (optional).
    - output: Output filename (optional).
    - comment: Add and update comments in the source code (optional).
    - language: Source file language (optional). SCALE currently supports 'python', 'js', 'c'.
    - verbose: Output progress information to stdout (optional).
    - very-verbose: Output LLM debug information to stdout (optional).
    - no-cache: Don't load the summary text for this file from cache.
    - n-ctx: Number of tokens to use as context (default: 32 * 1024).
    - max-new-tokens: Maximum number of new tokens to generate (default: 8 * 1024).
    - format: Chat format override (default: 'auto').
    - temperature: Temperature value for the LLM (default: 0.2).
    - top-p: Top-p value for the LLM (default: 0.9).
    - top-k: Top-k value for the LLM (default: 60).
    - repeat-penalty: Repeat penalty value for the LLM (default: 1.05).
    - n-batch: Number of batches to process (default: 512).
    - n-gpu-layers: Number of GPU layers to use (default: -1).

    Returns:
    - An argparse.Namespace object containing the parsed arguments.
    """

    formatters = sorted(llm_formatters()) + ['auto']

    p = argparse.ArgumentParser(description="SCALE: Source Code Annotation with LLM Engine")
    p.add_argument("source", nargs="+",
                   help="Source file(s) to annotate - paths, directories (expanded to source files), or globs "
                        "(e.g. \"src/**/*.c\"). Multiple targets are written in place; -o is only valid for one.")
    p.add_argument("--model", "-m", default="", help="Optional path to GGUF model file")
    p.add_argument("--output", "-o", default="", help="Optional output filename")
    p.add_argument("--comment", "-c", action="store_true", help="Add and update definition docstrings/header comments")
    p.add_argument("--blocks", "-b", action="store_true", help="Add and update within-function block comments (Python, C, JS)")
    p.add_argument("--file-doc", action="store_true",
                   help="Add or update a file-level header doccomment (Python module docstring, or C/JS header "
                        "comment), preserving shebang/copyright/license/boilerplate byte-for-byte.")
    p.add_argument("--block-comment-style", default="line", choices=("line", "block"),
                   help="Delimiter for block-pass comments in C/JS: 'line' (//) or 'block' (/* */). Ignored for Python.")
    p.add_argument("--comment-value", type=int, default=None, metavar="N",
                   help="Block pass: only write a paragraph comment whose model-rated value is >= N (1-5; default 3). "
                        "Higher is stricter; 1 keeps all. N above 5 (e.g. 6) skips the comment turns entirely and only "
                        "paragraphs the body (no model work). Lower-value notes still inform later paragraphs' context.")
    p.add_argument("--language", "-l", default=None, help="Source file language. SCALE currently supports: 'python', 'js', 'c'")
    p.add_argument("--project-doc", default="", metavar="PATH",
                   help="Project overview to distil into background context for every file (e.g. CLAUDE.md/README). "
                        "Default: auto-detect near the source. 'none' disables it; or pass an explicit path.")
    p.add_argument("--reference", action="append", default=[], metavar="PATH",
                   help="Read-only file(s)/dir(s)/glob(s) for SCALE to consult for context but never edit (e.g. the "
                        "project's headers). Repeatable. Their one-line summaries are shared with every target.")
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
    p.add_argument("--n-gpu-layers", type=int, default=-1, help="Number of GPU layers to use")

    # Selective escalation to a stronger model (Python only for now).
    p.add_argument("--emit-manifest", default="", metavar="PATH",
                   help="Emit phase: write deferred comment requests for complex routines to this manifest "
                        "(local model still annotates the rest of the file).")
    p.add_argument("--apply-manifest", default="", metavar="PATH",
                   help="Apply phase: patch a stronger model's answers from this manifest into the source. No model is "
                        "loaded; the source should be the emit-phase output.")
    p.add_argument("--escalate-cognitive", type=int, default=10, metavar="N",
                   help="Escalate any routine whose cognitive complexity exceeds N to the manifest (default 10).")
    p.add_argument("--codestats-json", default="", metavar="PATH",
                   help="Optional precomputed codestats JSON report; its cognitive scores override the native ones "
                        "when deciding what to escalate.")

    return p.parse_args(argv)


def _reference_oneliners(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    scale_path: Path,
    references: List[Path],
    project_blurb: str,
    no_cache: bool,
) -> List[Tuple[str, str]]:
    """
    Produce a one-line summary for each read-only reference file, as shared context for the run.

    Each reference file gets the same squashed one-liner the definition pass uses (`_get_short_summary`, cached), so a
    target being annotated can be told, briefly, what the project's other files (e.g. its headers) are for. The list is
    capped (`scale_project.MAX_REFERENCE_FILES`) so the injected context stays small; the overflow is logged, not
    silently dropped.

    Parameters:
    - `llm`/`cfg`: The model and config.
    - `scale_path`: The SCALE configuration directory.
    - `references`: The read-only reference files (already expanded).
    - `project_blurb`: The project blurb, used as context when summarising each reference.
    - `no_cache`: Forwarded to the summary cache.

    Returns:
    - `(filename, one_line_summary)` pairs, capped.
    """

    if not references:
        return []

    base: Messages = [{"role": "system", "content": (scale_path / "comment.txt").read_text(encoding="utf-8")}]
    if project_blurb:
        base = base + [
            {"role": "user", "content":
                "Here is some background on the wider project this file belongs to:\n\n" + project_blurb},
            {"role": "assistant", "content": PRIMING_ACK},
        ]

    cap = scale_project.MAX_REFERENCE_FILES
    if len(references) > cap:
        echo(f"Summarising {cap} of {len(references)} reference files for context (the rest are omitted)...")

    out: List[Tuple[str, str]] = []
    for ref in references[:cap]:
        blob, _lines, _le, lang = load_source(ref, None)
        if not blob.strip():
            continue
        oneliner = _get_short_summary(llm, cfg, scale_path, ref, blob, lang, base, no_cache=no_cache)
        out.append((ref.name, oneliner))
    return out


def _apply_manifest_file(
    src_path: Path,
    dst_path: Optional[Path],
    language: str,
    source_blob: str,
    source_lines: List[str],
    line_ending: str,
    manifest: dict,
) -> int:
    """
    Patch a stronger model's manifest answers into the source and write the result (the apply phase entry point).

    No model is loaded: the answers are inserted through the same insertion-only patchers as the local passes. Only
    Python is supported for now (the only language with a block provider and manifest applier).

    Parameters:
    - `src_path`: The source file (the emit-phase output) being patched.
    - `dst_path`: Where to write the result, or None to print to stdout.
    - `language`: The resolved language identifier.
    - `source_blob`: The source as a single string.
    - `source_lines`: The source split into individual lines.
    - `line_ending`: The detected line ending used to re-join the output.
    - `manifest`: The parsed manifest dictionary with answers filled in.

    Returns:
    - 0 on success, or an error number.
    """

    if language != "python":
        error(f"--apply-manifest currently supports Python only (got '{language}').")
        return 1

    from scale_python import apply_manifest
    new_lines = apply_manifest(source_blob, source_lines, manifest)

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

    # Expand the target patterns (files / directories / globs) into a concrete, deduplicated, ordered file list.
    targets = scale_project.gather_files(args.source)
    if not targets:
        error(f"No source files matched: {' '.join(args.source)}")
        return 1

    # ---- Apply phase: model-free, single target. Patch a stronger model's manifest answers into the emit output. ----
    if args.apply_manifest:
        if len(targets) != 1:
            error("--apply-manifest operates on a single source file.")
            return 1
        src_path = targets[0]
        manifest = scale_escalate.read_manifest(Path(args.apply_manifest))
        language = args.language.lower() if args.language else manifest.get("language")
        source_blob, source_lines, line_ending, language = load_source(src_path, language)
        if language not in SUPPORTED_LANGUAGES:
            error(f"Unsupported language '{language}'. SCALE supports: {', '.join(SUPPORTED_LANGUAGES)}")
            return 1
        return _apply_manifest_file(src_path, dst_path, language, source_blob, source_lines, line_ending, manifest)

    if not (args.comment or args.blocks or args.file_doc):
        # Nothing to do without --comment, --blocks, or --file-doc.
        return 0

    if dst_path is not None and len(targets) > 1:
        error("-o/--output cannot be used with multiple targets; multiple targets are annotated in place.")
        return 1
    if args.emit_manifest and len(targets) != 1:
        error("--emit-manifest operates on a single source file.")
        return 1

    # Read-only reference files (consulted for context, never edited); a file that is also a target is not a reference.
    references = scale_project.gather_files(args.reference) if args.reference else []
    target_keys = {p.resolve() for p in targets}
    references = [r for r in references if r.resolve() not in target_keys]

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

    # Build the shared project context once for the whole run: the project blurb (auto-detected near the targets unless
    # --project-doc says otherwise; 'none' disables) plus one-line summaries of the read-only reference files.
    project_blurb = ""
    project_doc = scale_project.resolve_project_doc(args.project_doc, targets[0])
    if project_doc is not None:
        project_blurb = scale_project.project_blurb(llm, cfg, scale_path, project_doc, no_cache=args.no_cache)
    related = _reference_oneliners(llm, cfg, scale_path, references, project_blurb, args.no_cache)
    project_context = scale_project.compose_project_context(project_blurb, related)

    # Build the call graph once over targets ∪ references (model-free). It drives the definition pass's documentation
    # order (callees/children first) and the callee-contract context injected into each routine's turn; the seeded
    # store accumulates contracts across files. Only relevant when the def pass runs. Targets are reordered so a
    # callee's file is documented before a caller's (coarse, by file).
    graph = store = None
    language_arg = args.language.lower() if args.language else None
    if args.comment:
        graph, store = _build_call_graph(targets, references, language_arg)
        if graph is not None:
            key_to_path = {str(t.resolve()): t for t in targets}
            targets = [key_to_path[k] for k in graph.file_order(list(key_to_path.keys()))]

    # Annotate each target in turn (single target -> -o or stdout; multiple targets -> in place).
    rc = 0
    for target in targets:
        language = args.language.lower() if args.language else None
        source_blob, source_lines, line_ending, language = load_source(target, language)
        if language not in SUPPORTED_LANGUAGES:
            error(f"Skipping {target}: unsupported language '{language}' (SCALE supports: "
                  f"{', '.join(SUPPORTED_LANGUAGES)}).")
            continue

        # ---- Emit phase: a per-target selective-escalation policy (single target only, guarded above). ----
        escalation = None
        if args.emit_manifest:
            if language != "python":
                error("--emit-manifest currently supports Python only.")
                return 1
            override = scale_escalate.load_codestats_json(Path(args.codestats_json)) if args.codestats_json else None
            doc_style = "\n\n".join(
                t for t in (_read_optional(scale_path / "guidelines.md"),
                            _read_optional(scale_path / f"comment.{language}.txt")) if t
            )
            escalation = scale_escalate.Escalation(
                threshold=args.escalate_cognitive, override=override, doc_style=doc_style)

        # Bind the call-graph hooks for this file over the shared store (closing the file key into each closure). The
        # store accumulates across files, so a callee documented in an earlier-ordered target informs a later caller.
        doc_order = callee_context = on_doc = None
        if graph is not None and store is not None:
            file_key = str(target.resolve())
            doc_order = graph.doc_order(file_key)
            callee_context = lambda q, fk=file_key: store.callee_notes(fk, q)
            on_doc = lambda q, doc, fk=file_key: store.update(fk, q, doc)

        out = dst_path if len(targets) == 1 else target
        echo(f"Annotating {target}...")
        frc = generate_comments(
            llm, cfg, scale_path, target, out,
            source_blob, source_lines, line_ending, language,
            no_cache=args.no_cache,
            do_comment=args.comment,
            do_blocks=args.blocks,
            do_file_doc=args.file_doc,
            block_comment_style=args.block_comment_style,
            comment_value=args.comment_value,
            escalation=escalation,
            project_context=project_context,
            doc_order=doc_order,
            callee_context=callee_context,
            on_doc=on_doc,
        )
        if frc != 0:
            rc = frc

        # Serialise the deferred requests so a stronger model can answer them (then re-run with --apply-manifest).
        if frc == 0 and escalation is not None:
            manifest = escalation.to_manifest(str(target), language, line_ending)
            scale_escalate.write_manifest(Path(args.emit_manifest), manifest)
            echo(f"Wrote {len(escalation.requests)} escalation request(s) to {args.emit_manifest}")

    return rc


if __name__ == "__main__":  # pragma: no cover
    """
    This block ensures that the script can be run directly, invoking the `main` function and exiting with its return value.
    The `# pragma: no cover` comment is used to exclude this line from coverage reports, likely because it is not part of
    the normal execution flow and would always exit immediately.
    """
    raise SystemExit(main())
