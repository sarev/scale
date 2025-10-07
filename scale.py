#!/usr/bin/env python3
"""
This module TODO.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from scale_llm import LocalChatModel, GenerationConfig, llm_formatters, Messages, Chunk
from scale_log import echo, error, set_verbosity
from typing import List, Optional, Sequence, Tuple
import argparse
import sys


# DEFAULT_MODEL = "../models/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"


# Based upon meta-llama/Llama-3.1-8B-Instruct
#
# 8B parameters, 6-bit quantised, 6.6GB, context length 131072.
#
DEFAULT_MODEL = "../models/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"


# ---------------------------- CLI harness ----------------------------


def load_source(src_path: Path, language: Optional[str] = None) -> Tuple[str, Chunk, str, str]:
    """
    Load a source file and return its contents, along with other information.

    Parameters:
    - source: The source file path.

    Returns:
    - A tuple containing:
      - The complete text of the source file as a single string (with original line endings).
      - The source file split into individual lines.
      - The source file line ending string ('\n', '\r', or '\r\n').
      - The file suffix for the source language (e.g. "python" or "c").
    """

    def guess_language(source_lines: List[str]) -> str:
        """
        Apply heuristics to try to guess the programming language from the supplied lines of source code.

        Args:
        - source_lines: the list of source file lines of interest.

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
    source_blob = src_path.read_text(encoding="utf-8")

    # Count newline styles
    count_rn = source_blob.count("\r\n")
    count_r = source_blob.count("\r") - count_rn  # bare \r not part of \r\n
    count_n = source_blob.count("\n") - count_rn  # bare \n not part of \r\n

    # Find the most common one
    if count_rn > max(count_r, count_n):
        line_ending = "\r\n"
    else:
        line_ending = "\r" if count_r > count_n else "\n"

    # Create a version of the source code as a 'chunk' (list of strings, one per line)
    source_lines = source_blob.split(line_ending)

    # Determine what the language is that we're dealing with
    if language is None or language == "":
        language = guess_language(source_lines)
    echo(f"Language set to '{language}'...")

    return source_blob, source_lines, line_ending, language


def prime_llm_for_comments(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    scale_path: Path,
    source_blob: str,
    language: str
) -> Messages:
    """."""

    echo("Priming LLM...")

    # Load the system prompt for doing comment generation
    comment_path = scale_path / "comment.txt"
    comment_prompt = comment_path.read_text(encoding="utf-8")

    # Load the user prompt specifying the comment format template for the language in question
    template_path = scale_path / f"comment.{language}.txt"
    template_prompt = template_path.read_text(encoding="utf-8")

    # Prime the LLM with our system prompt
    messages = []
    messages.append({"role": "system", "content": comment_prompt})

    # The LLM should simply reply with "OK" here
    reply = llm.generate(messages, cfg=cfg)
    echo(f"System prompt ingested? {reply}")
    messages.append({"role": "assistant", "content": reply})

    # Now provide the LLM with the entire source file as context
    echo("Generating full source summary...")
    source_context = "To help you understand the fuller context of each chunk that I supply, here is the original source file as a whole:\n\n"
    source_context += f"{source_blob}\n\nPlease output a detailed summary of what this program does, along with some highlights of its internal workings.\n"
    messages.append({"role": "user", "content": source_context})
    reply = llm.generate(messages, cfg=cfg)
    reply_length = reply.count("\n")
    echo(f"Source file summarised? {reply_length} lines of summary created")
    echo(f"\n{reply}\n")

    # Replace the source file with its summary in the LLM's context
    messages.pop()
    reply = f"To give you context, here is an overview of what the program as a whole does:\n\n{reply}\n\nPlease confirm you are ready to continue by saying 'OK'.\n"
    messages.append({"role": "user", "content": reply})
    reply = llm.generate(messages, cfg=cfg)
    echo(f"Summary ingested? {reply}")
    messages.append({"role": "assistant", "content": reply})

    # Now tell the LLM what the preferred comment format is
    messages.append({"role": "user", "content": template_prompt})
    reply = llm.generate(messages, cfg=cfg)
    echo(f"Template ingested? {reply}")
    messages.append({"role": "assistant", "content": reply})

    return messages


def generate_comments(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    scale_path: Path,
    src_path: Path,
    dst_path: Path,
    language: Optional[str] = None
) -> int:
    """
    .
    """

    source_blob, source_lines, line_ending, language = load_source(src_path, language)
    messages = prime_llm_for_comments(llm, cfg, scale_path, source_blob, language)
    if language == "python":
        from scale_python import generate_language_comments
        new_lines = generate_language_comments(llm, cfg, messages, source_blob, source_lines)
    elif language == "js":
        from scale_javascript import generate_language_comments
        new_lines = generate_language_comments(llm, cfg, messages, source_blob, source_lines)
    elif language == "c":
        from scale_c import generate_language_comments
        new_lines = generate_language_comments(llm, cfg, messages, source_blob, source_lines)
    else:
        error(f"Unsupported language '{language}'")
        return 1

    # Write the output
    if dst_path:
        dst_path.write_text(line_ending.join(new_lines), encoding="utf-8")
        echo(f"Updated source written to {dst_path}")
    else:
        print("\n".join(new_lines))

    return 0


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments and return an argparse.Namespace object.

    Parameters:
    - argv: Optional list of strings to parse as command-line arguments. If not provided, sys.argv[1:] is used.

    Returns:
    - An argparse.Namespace object containing the parsed arguments.
    """

    formatters = sorted(llm_formatters()) + ['auto']

    p = argparse.ArgumentParser(description="SCALE: Source Code Annotation with LLM Engine")
    p.add_argument("source", help="Path to source file")
    p.add_argument("--model", "-m", default="", help="Optional path to GGUF model file")
    p.add_argument("--output", "-o", default="", help="Optional output filename")
    p.add_argument("--comment", "-c", action="store_true", help="Add and update comments")
    p.add_argument("--language", "-l", default=None, help="Source file language. SCALE currently supports: 'python', 'js'")
    p.add_argument("--verbose", "-v", action="store_true", help="Output progress information to stdout")
    p.add_argument("--very-verbose", "-vv", action="store_true", help="Output LLM debug information to stdout")
    p.add_argument("--n-ctx", type=int, default=32 * 1024)
    p.add_argument("--max-new-tokens", "-M", type=int, default=8 * 1024)
    p.add_argument("--format", "-f", default="auto", help=f"Chat format override. One of {formatters}")
    p.add_argument("--temperature", "-T", type=float, default=0.2)
    p.add_argument("--top-p", "-P", type=float, default=0.9)
    p.add_argument("--top-k", "-K", type=int, default=60)
    p.add_argument("--repeat-penalty", "-R", type=float, default=1.05)
    p.add_argument("--n-batch", type=int, default=512)
    p.add_argument("--n-gpu-layers", type=int, default=-1)

    return p.parse_args(argv)


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

    if args.comment:
        echo("Generating comments...")
        language = args.language.lower() if args.language else None
        return generate_comments(llm, cfg, scale_path, src_path, dst_path, language)

    return 0


if __name__ == "__main__":  # pragma: no cover
    """
    This block ensures that the script can be run directly, invoking the `main` function and exiting with its return value.
    The `# pragma: no cover` comment is used to exclude this line from coverage reports, likely because it is not part of
    the normal execution flow and would always exit immediately.
    """
    raise SystemExit(main())
