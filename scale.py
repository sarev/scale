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
import scale_escalate
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
        self._data_path = self._CACHE_DIR / f"{self._uid}.txt"      # one summary file per uid
        self._hash_path = self._CACHE_DIR / f"{self._uid}.sha256"   # content hash for invalidation

        # Load the existing summary only if it was generated from the same source content.
        try:
            cached_hash = self._hash_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            cached_hash = None

        if cached_hash == self._hash:
            try:
                raw = self._data_path.read_bytes()
                self._summary = raw.decode("utf-8", errors="surrogateescape")
            except FileNotFoundError:
                self._summary = None
        else:
            # Stale (or missing) hash: ignore any cached summary so it is regenerated.
            self._summary = None

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

    combined = "\n\n".join(f"Part {i}: {s}" for i, s in enumerate(partials, start=1))
    reduce_prompt = (
        f"Below are summaries of consecutive parts of a single {language} source file. "
        "Combine them into one concise overall summary of what the whole file does. "
        "Do not add any conversational discussion:\n\n"
        f"{combined}"
    )

    # If the combined summaries fit, reduce them in a single pass.
    if base_overhead + llm.estimate_tokens(reduce_prompt) <= limit:
        return llm.generate(base_messages + [{"role": "user", "content": reduce_prompt}], cfg=summary_cfg)

    # Otherwise reduce in groups first, then recurse on the (smaller) set of group summaries.
    groups = _group_by_budget(partials, max(1, limit - base_overhead - 64), llm.estimate_tokens)
    reduced: List[str] = []
    for group in groups:
        sub = "\n\n".join(f"Part {i}: {s}" for i, s in enumerate(group, start=1))
        prompt = (
            f"Below are summaries of consecutive parts of a single {language} source file. "
            "Combine them into one concise summary. Do not add any conversational discussion:\n\n"
            f"{sub}"
        )
        reduced.append(llm.generate(base_messages + [{"role": "user", "content": prompt}], cfg=summary_cfg))
    return _reduce_summaries(llm, summary_cfg, base_messages, reduced, language, limit, base_overhead)


def _generate_file_summary(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    base_messages: Messages,
    source_blob: str,
    language: str,
) -> str:
    """
    Summarise a whole source file, falling back to chunked map-reduce when it is too large for one pass.

    The summary primes the per-routine commenting turns with an understanding of the file as a whole. Files that fit
    the context window are summarised in a single request. Larger files are split into context-sized chunks, each
    summarised independently (the "map" step), and those partial summaries are then combined into one overall summary
    (the "reduce" step, applied recursively if the partials are themselves too large). Every request uses a capped
    reply length so the resulting summary - and therefore the persistent priming context - stays small regardless of
    file size.

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The base generation configuration (cloned with a smaller `max_new_tokens` for summaries).
    - `base_messages`: The primed context to summarise against (typically the system prompt and its acknowledgement);
      this list is not mutated.
    - `source_blob`: The complete source text.
    - `language`: The source language identifier (used to phrase the prompts).

    Returns:
    - The overall summary text for the file.
    """

    summary_cfg = replace(cfg, max_new_tokens=SUMMARY_MAX_TOKENS)
    limit = llm.n_ctx - llm.ctx_margin - SUMMARY_MAX_TOKENS
    base_overhead = llm.count_tokens(base_messages) if base_messages else 0

    one_shot = (
        "To help you understand the fuller context of each chunk that I supply, here is the original source file as a whole:\n\n"
        f"{source_blob}\n\n"
        "Please output a detailed summary of what this program does, along with some highlights of its internal workings.\n"
        "Do not ask follow-up questions or add any conversational discussion. Just give me the detailed summary."
    )

    # Fast path: the whole file fits in a single summarisation turn.
    if base_overhead + llm.estimate_tokens(one_shot) <= limit:
        return llm.generate(base_messages + [{"role": "user", "content": one_shot}], cfg=summary_cfg)

    # Map: summarise the file in context-sized chunks (64 tokens of headroom for the wrapper text).
    chunk_budget = max(1, limit - base_overhead - 64)
    chunks = _split_source(source_blob, chunk_budget, llm.estimate_tokens)
    echo(f"Source too large for a single-pass summary; summarising in {len(chunks)} chunk(s)...")

    partials: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        prompt = (
            f"This is part {idx} of {len(chunks)} of a larger {language} source file. "
            "Briefly summarise what this part of the code does. Do not add any conversational discussion:\n\n"
            f"{chunk}"
        )
        partials.append(llm.generate(base_messages + [{"role": "user", "content": prompt}], cfg=summary_cfg))
        echo(f"Summarised part {idx}/{len(chunks)}")

    # Reduce: combine the partial summaries into a single overall summary.
    return _reduce_summaries(llm, summary_cfg, base_messages, partials, language, limit, base_overhead)


def prime_llm_for_comments(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    scale_path: Path,
    src_path: Path,
    source_blob: str,
    language: str,
    no_cache: Optional[bool] = False,
    template: str = "comment",
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
    - `template`: Which per-language style template to load as the final priming turn: `"comment"` loads
      `comment.<lang>.txt` (the definition/docstring pass) and `"blocks"` loads `blocks.<lang>.txt` (the
      within-function "blob" pass). Only one is ever loaded so the two passes never share each other's guidance.

    Returns:
    - A list of messages exchanged between the system and the LLM, including the priming prompts and the generated responses.
    """

    # Initialise the summary cache (keyed on path + a hash of the current contents)
    summary_cache = SummaryCache(src_path, source_blob)

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

    # The LLM should simply reply with "OK" here
    reply = llm.generate(messages, cfg=cfg)
    echo(f"System prompt ingested? {reply}")
    messages.append({"role": "assistant", "content": reply})

    # Now summarise the whole file for context. This automatically switches to a chunked map-reduce
    # approach when the file is too large to summarise in one pass (see _generate_file_summary).
    if no_cache is False and summary_cache.summary:
        echo("Loaded full source summary from cache...")
    else:
        echo("Generating full source summary...")
        summary_cache.summary = _generate_file_summary(llm, cfg, messages, source_blob, language)

    reply_length = 1 + summary_cache.summary.count("\n")
    echo(f"Source file summarised? {reply_length} lines of summary created")
    echo(f"\n{summary_cache.summary}\n")

    # Provide the summary (not the whole file) as the working context.
    prompt = (
        "To give you context, here is an overview of what the program as a whole does:\n\n"
        f"{summary_cache.summary}\n\n"
        "Please confirm you are ready to continue by saying 'OK'.\n"
    )
    messages.append({"role": "user", "content": prompt})
    reply = llm.generate(messages, cfg=cfg)
    echo(f"Summary ingested? {reply}")
    messages.append({"role": "assistant", "content": reply})

    # Now tell the LLM what the preferred comment format is
    messages.append({"role": "user", "content": template_prompt})
    reply = llm.generate(messages, cfg=cfg)
    echo(f"Template ingested? {reply}")
    messages.append({"role": "assistant", "content": reply})

    return messages


def _def_pass(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    source_blob: str,
    source_lines: List[str],
    language: str,
    escalation=None,
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

    Returns:
    - The annotated source split into individual lines.
    """

    if language == "python":
        from scale_python import generate_language_comments
        # Only the Python worker understands selective escalation for now.
        return generate_language_comments(llm, cfg, messages, source_blob, source_lines, escalation=escalation)
    elif language == "js":
        from scale_javascript import generate_language_comments
    elif language == "c":
        from scale_c import generate_language_comments
    else:
        raise ValueError(f"Unsupported language '{language}'")
    return generate_language_comments(llm, cfg, messages, source_blob, source_lines)


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
    raise NotImplementedError(
        f"The --blocks pass does not yet support '{language}' (Python only for now)."
    )


def _block_style_for(language: str):
    """
    Return the comment-style descriptor that drives block-comment rendering for `language`.

    Parameters:
    - `language`: The programming language identifier (already validated).

    Returns:
    - A `scale_blocks.CommentStyle` describing the language's comment delimiters.

    Raises:
    - `NotImplementedError`: If the language has no block provider yet (currently only Python is wired).
    """

    from scale_blocks import PYTHON_STYLE
    if language == "python":
        return PYTHON_STYLE
    raise NotImplementedError(
        f"The --blocks pass does not yet support '{language}' (Python only for now)."
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
    escalation=None,
) -> List[str]:
    """
    Run the within-function "blob" pass: annotate logical groups of statements inside each routine body.

    The block-pass prompt wording is loaded from `scale-cfg` so users can tune it without touching code (a missing
    file falls back to the engine's built-in default): `blocks.segment.txt`, `blocks.comment.txt`,
    `blocks.comment.nudge.txt` (the retry nudge), and `blocks.note.short.txt` / `blocks.note.long.txt` (the short-/
    long-routine length notes).

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
    style = _block_style_for(language)
    targets = provider(source_blob, source_lines)
    return annotate_blocks(
        llm, cfg, messages, source_lines, targets, style,
        segment_prompt=_read_optional(scale_path / "blocks.segment.txt"),
        comment_prompt=_read_optional(scale_path / "blocks.comment.txt"),
        comment_nudge=_read_optional(scale_path / "blocks.comment.nudge.txt"),
        note_short=_read_optional(scale_path / "blocks.note.short.txt"),
        note_long=_read_optional(scale_path / "blocks.note.long.txt"),
        escalation=escalation,
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
    escalation=None,
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

    Returns:
    - 0 if the operation was successful, or an error number.

    Notes:
    - Supported languages are Python, JavaScript, and C (the block pass is currently Python only).
    - If the destination path is not provided, the generated comments will be printed to the console.
    """

    # The summary is primed from the original source for both passes, so the content-hash cache stays warm; only the
    # worker input advances from one pass to the next.
    new_lines = source_lines

    if do_comment:
        messages = prime_llm_for_comments(
            llm, cfg, scale_path, src_path, source_blob, language, no_cache=no_cache, template="comment"
        )
        new_lines = _def_pass(llm, cfg, messages, source_blob, new_lines, language, escalation=escalation)

    if do_blocks:
        try:
            _block_provider_for(language)  # fail fast (and cleanly) on an unsupported language
        except NotImplementedError as exc:
            error(str(exc))
            return 1
        current_blob = line_ending.join(new_lines)
        messages = prime_llm_for_comments(
            llm, cfg, scale_path, src_path, source_blob, language, no_cache=no_cache, template="blocks"
        )
        new_lines = _block_pass(llm, cfg, scale_path, messages, current_blob, new_lines, language, escalation=escalation)

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
    p.add_argument("source", help="Path to source file")
    p.add_argument("--model", "-m", default="", help="Optional path to GGUF model file")
    p.add_argument("--output", "-o", default="", help="Optional output filename")
    p.add_argument("--comment", "-c", action="store_true", help="Add and update definition docstrings/header comments")
    p.add_argument("--blocks", "-b", action="store_true", help="Add and update within-function block comments (Python only for now)")
    p.add_argument("--language", "-l", default=None, help="Source file language. SCALE currently supports: 'python', 'js', 'c'")
    p.add_argument("--verbose", "-v", action="store_true", help="Output progress information to stdout")
    p.add_argument("--very-verbose", "-vv", action="store_true", help="Output LLM debug information to stdout")
    p.add_argument("--no-cache", "-nc", action="store_true", help="Don't load the summary text for this file from cache")
    p.add_argument("--n-ctx", type=int, default=12 * 1024, help="Number of tokens to use as context")
    p.add_argument("--max-new-tokens", "-M", type=int, default=8 * 1024, help="Maximum number of new tokens to generate")
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
    src_path = Path(args.source)
    dst_path = Path(args.output) if args.output else None

    # ---- Apply phase: model-free. Patch a stronger model's manifest answers into the (emit-phase) source. ----
    if args.apply_manifest:
        manifest = scale_escalate.read_manifest(Path(args.apply_manifest))
        language = args.language.lower() if args.language else manifest.get("language")
        source_blob, source_lines, line_ending, language = load_source(src_path, language)
        if language not in SUPPORTED_LANGUAGES:
            error(f"Unsupported language '{language}'. SCALE supports: {', '.join(SUPPORTED_LANGUAGES)}")
            return 1
        return _apply_manifest_file(src_path, dst_path, language, source_blob, source_lines, line_ending, manifest)

    if not (args.comment or args.blocks):
        # Nothing to do without --comment or --blocks.
        return 0

    # Load the source and resolve the language up front, so an unsupported file fails fast
    # before the (slow) model load.
    language = args.language.lower() if args.language else None
    source_blob, source_lines, line_ending, language = load_source(src_path, language)
    if language not in SUPPORTED_LANGUAGES:
        error(f"Unsupported language '{language}'. SCALE supports: {', '.join(SUPPORTED_LANGUAGES)}")
        return 1

    # ---- Emit phase: build the selective-escalation policy that defers complex routines to a manifest. ----
    escalation = None
    if args.emit_manifest:
        if language != "python":
            error("--emit-manifest currently supports Python only.")
            return 1
        override = scale_escalate.load_codestats_json(Path(args.codestats_json)) if args.codestats_json else None
        escalation = scale_escalate.Escalation(threshold=args.escalate_cognitive, override=override)

    # Prepare model and config
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

    echo("Generating comments...")
    rc = generate_comments(
        llm, cfg, scale_path, src_path, dst_path,
        source_blob, source_lines, line_ending, language,
        no_cache=args.no_cache,
        do_comment=args.comment,
        do_blocks=args.blocks,
        escalation=escalation,
    )

    # Serialise the deferred requests so a stronger model can answer them (then re-run with --apply-manifest).
    if rc == 0 and escalation is not None:
        manifest = escalation.to_manifest(str(src_path), language, line_ending)
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
