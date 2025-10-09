#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

This Python program, `scale_c.py`, is a tool for generating and inserting documentation comments into C source code files.

It uses the Tree-sitter C parser to analyse the code, identify function definitions, and then asks a language model (LLM)
to provide documentation comments for each function. The comments are then inserted or replaced above the corresponding
function declarations in the source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from scale_llm import LocalChatModel, GenerationConfig, Messages, Chunk
from scale_log import echo
from tree_sitter import Parser, Language  # type: ignore
from tree_sitter_c import language as c_language
from typing import Dict, List, Optional, Tuple, Any
import textwrap


# ---------------- Tree-sitter bootstrap (handles capsule vs Language object, and both Parser APIs)

def _load_c_language_and_parser() -> Tuple[Language, Parser]:
    """
    Load the C language and parser.

    This function loads the C language and creates a parser instance. If the `c_language()` call returns a `Language` object, it
    is used directly. Otherwise, it is assumed to be a label and a new `Language` instance is created with the label "C".

    Parameters:
    - None

    Returns:
    - A tuple containing the loaded `Language` object and the created `Parser` instance.
    """

    ptr_or_lang: Any = c_language()

    # Wrap capsule -> Language, or accept Language directly.
    if isinstance(ptr_or_lang, Language):
        lang = ptr_or_lang
    else:
        # Second arg is a label only.
        lang = Language(ptr_or_lang, "C")

    # Some builds expose Parser().set_language(...), others Parser(Language)
    try:
        p = Parser()
        p.set_language(lang)
    except AttributeError:
        p = Parser(lang)

    return lang, p


C_LANGUAGE, C_PARSER = _load_c_language_and_parser()


# ---------------- Utilities

def _to_1based(row0: int) -> int:
    """
    Convert a 0-based row index to a 1-based index.

    Parameters:
    - `row0`: The 0-based row index to convert.

    Returns:
    - The corresponding 1-based row index.
    """

    return row0 + 1


def _row_of(point) -> int:
    """
    Return the row number of a tree-sitter point.

    This function supports both tuple and object forms of points. If the input is an object with a `.row` attribute, it is used directly.
    If the input is a tuple, its first element (the row number) is returned.

    Parameters:
    - `point`: The tree-sitter point to extract the row from.

    Returns:
    - The row number of the point as an integer.
    """
    try:
        return point.row  # object with .row
    except AttributeError:
        return point[0]   # tuple (row, col)


def _line_span_from_node(n) -> Tuple[int, int]:
    """
    Return the inclusive 1-based start and end line span for a given node.

    Parameters:
    - `n`: The node for which to compute the line span.

    Returns:
    - A tuple of two integers representing the 1-based start and end line numbers.
    """
    return _to_1based(_row_of(n.start_point)), _to_1based(_row_of(n.end_point))


def _get_text_for_lines(source_lines: Chunk, a: int, b: int) -> str:
    """
    Extract a range of lines from the source code.

    Parameters:
    - `source_lines`: The list of source code lines.
    - `a`: The starting line number (inclusive).
    - `b`: The ending line number (exclusive).

    Returns:
    - A string containing the extracted lines, joined by newline characters. If `a` is greater than `b`, an empty string is returned.
    """

    a = max(1, a)
    b = min(len(source_lines), b)
    if a > b:
        return ""
    return "\n".join(source_lines[a - 1:b])


def _leading_spaces_count(line: str) -> int:
    """
    Count the number of leading spaces in a line.

    Parameters:
    - `line`: The input string to count leading spaces from.

    Returns:
    - The number of leading spaces in the input string.
    """

    return len(line) - len(line.lstrip(" "))


def _scan_existing_comment_block_above(source_lines: Chunk, header_start_line_1b: int) -> Optional[Tuple[int, int]]:
    """
    Detect an existing comment block immediately above the header start line.

    This function scans the source lines for a contiguous block of comments that ends just above the header.
    It supports both C-style block comments '/* ... */' and C-style line comments '//'.

    Parameters:
    - `source_lines`: The source code lines to scan.
    - `header_start_line_1b`: The line number of the header start (1-based).

    Returns:
    - A tuple `(start_line_1b, end_line_1b)` representing the start and end line numbers of the comment block,
      or `None` if no comment block is found.
    """
    i = header_start_line_1b - 2  # zero-based line just above header
    if i < 0:
        return None

    # Case A: block comment ending above
    if source_lines[i].rstrip().endswith("*/"):
        j = i
        while j >= 0:
            if source_lines[j].lstrip().startswith("/*"):
                return (j + 1, i + 1)
            j -= 1
        # malformed; fall through

    # Case B: contiguous '//' lines
    j = i
    saw_slash = False
    while j >= 0:
        stripped = source_lines[j].lstrip()
        if stripped.startswith("//"):
            saw_slash = True
            j -= 1
            continue
        if stripped == "":
            break  # blank breaks adjacency
        break
    if saw_slash:
        start_1b = j + 2
        return (start_1b, i + 1)

    return None


# ---------------- C DefInfo

@dataclass(frozen=True)
class DefInfoC:
    """
    Represents information about a C function definition.

    Attributes:
        qualname (str): The name of the function.
        node (object): The tree-sitter Node object representing the function definition.
        kind (str): The type of node, always "function".
        start (int): The line number where the function definition starts (1-based).
        end (int): The line number where the function definition ends (inclusive).
        header_start (int): The line number where the function header starts.
        header_end (int): The line number where the function header ends (line before body starts '{').
        depth (int): The nesting depth of the function, always 0 for C (no nested functions).
        parent_id (Optional[int]): The ID of the parent node, always None.
        children_ids (Tuple[int, ...]): A tuple of IDs of child nodes, always empty.
    """

    qualname: str            # function name
    node: object             # tree_sitter.Node (function_definition)
    kind: str                # "function"
    start: int               # definition start line (1-based)
    end: int                 # definition end line (inclusive)
    header_start: int        # header start line
    header_end: int          # header end line (line before body starts '{')
    depth: int               # always 0 for C (no nested functions), but keep for shape parity
    parent_id: Optional[int] # always None
    children_ids: Tuple[int, ...] = field(default_factory=tuple)  # always empty


# ---------------- Collect function definitions (ignores forward declarations)

def iter_defs_with_info_c(source_blob: str) -> List[DefInfoC]:
    """
    Collect real function definitions from a C file.

    This function parses the input C source code, identifies function definitions, and collects metadata.
    Forward declarations are excluded from the results.

    Parameters:
    - `source_blob`: The input C source code as a string.

    Returns:
    - A sorted list of `DefInfoC` objects, each containing information about a real function definition.
    """
    source_bytes = source_blob.encode("utf-8", errors="replace")
    tree = C_PARSER.parse(source_bytes)
    root = tree.root_node

    results: List[DefInfoC] = []

    def header_span_for_function(fn_node) -> Tuple[int, int]:
        """
        Return the line span of the header for a function.

        The header ends on the line before the compound statement (body) begins, unless there is no body,
        in which case it spans the entire function definition.
        """
        # In C grammar, function_definition has field 'declarator' and 'body'
        body = fn_node.child_by_field_name("body")
        if body and body.type == "compound_statement":
            start, _ = _line_span_from_node(fn_node)
            header_end = _to_1based(_row_of(body.start_point)) - 1
            return start, header_end
        # Fallback: no body? Treat header as single line (should not happen for function_definition)
        s, _e = _line_span_from_node(fn_node)
        return s, s

    def function_name(fn_node) -> str:
        """
        Extract function identifier text from the declarator subtree.

        Parameters:
        - `fn_node`: The declarator subtree node to extract the function identifier from.

        Returns:
        - The extracted function identifier as a string, or "<anonymous>" if not found.
        """
        # function_definition -> declarator (function_declarator) -> declarator (pointer? direct_declarator)
        decl = fn_node.child_by_field_name("declarator")
        if decl is None:
            return "<anonymous>"
        # Find the innermost identifier under the declarator
        stack = [decl]
        while stack:
            n = stack.pop()
            if n.type == "identifier":
                # slice from bytes to handle non-ASCII robustly
                return source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
            # traverse children
            for i in range(n.named_child_count - 1, -1, -1):
                stack.append(n.named_child(i))
        return "<anonymous>"

    # DFS over named nodes
    def walk(n) -> None:
        """
        Walk the abstract syntax tree (AST) and collect function definitions.

        This function recursively traverses the AST, identifying function definitions and collecting metadata.
        For each function definition, it extracts the qualified name, line span, header span, and other relevant information.

        Parameters:
        - `n`: The current node in the AST to process.

        Returns:
        - None
        """

        if n.type == "function_definition":
            qual = function_name(n) or "<anonymous>"
            s, e = _line_span_from_node(n)
            h_s, h_e = header_span_for_function(n)
            results.append(DefInfoC(
                qualname=qual,
                node=n,
                kind="function",
                start=s,
                end=e,
                header_start=h_s,
                header_end=h_e,
                depth=0,
                parent_id=None,
                children_ids=tuple(),
            ))
            # no nested functions in C; still walk body for completeness if needed
        for i in range(n.named_child_count):
            walk(n.named_child(i))

    walk(root)
    # Multiple definitions due to #ifdef blocks are kept; caller may disambiguate by span.
    return sorted(results, key=lambda d: (d.start, d.end))


# ---------------- Snippet assembly

def assemble_snippet_for_c(source_lines: Chunk, info: DefInfoC) -> str:
    """
    Assemble a snippet of C code for the function, including its header and body.

    This method does not attempt to suppress inner functions, as C does not support nested functions.

    Parameters:
    - `source_lines`: The input source code lines.
    - `info`: A DefInfoC object containing metadata about the function definition.

    Returns:
    - A string representing the assembled snippet of C code.
    """
    header_text = _get_text_for_lines(source_lines, info.header_start, info.header_end)
    body_text = _get_text_for_lines(source_lines, info.header_end + 1, info.end)
    parts: List[str] = [header_text]
    if body_text:
        if header_text and not header_text.endswith("\n"):
            parts.append("\n")
        parts.append(body_text)
    return "".join(parts)


# ---------------- LLM exchange

def _render_c_block_comment(text: str, base_indent: str) -> List[str]:
    """
    Render a C-style block comment from the provided text.

    Parameters:
    - `text`: The text to be rendered as a block comment.
    - `base_indent`: The base indentation for the comment lines.

    Returns:
    - A list of strings representing the rendered C-style block comment.
    """
    lines = text.splitlines()
    out = [f"{base_indent}/*"]
    if lines:
        for ln in lines:
            out.append(f"{base_indent} * {ln.rstrip()}")
    else:
        out.append(f"{base_indent} * (no documentation)")
    out.append(f"{base_indent} */")
    return out


def _extract_first_c_comment_block(reply: str) -> str:
    """
    Extract the first C block comment body from the LLM reply.

    Supports either explicit '/* ... */' fences or plain text (then we take all, dedented).
    Returns the inner text only (without '/*' and '*/').

    Parameters:
    - `reply`: The LLM reply string to extract the comment from.

    Returns:
    - The extracted C block comment body as a string.
    """
    txt = textwrap.dedent(reply)
    # Prefer fenced block
    start = txt.find("/*")
    end = txt.find("*/", start + 2) if start != -1 else -1
    if start != -1 and end != -1:
        inner = txt[start + 2:end]
        # strip leading '*' common to C doc blocks
        lines = inner.splitlines()
        cleaned = []
        for ln in lines:
            stripped = ln.lstrip()
            if stripped.startswith("*"):
                stripped = stripped[1:]
                if stripped.startswith(" "):
                    stripped = stripped[1:]
            cleaned.append(stripped.rstrip())
        return "\n".join(cleaned).strip()
    # Fallback: use all text, dedented
    return txt.strip()


def generate_comments_c(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    defs: List[DefInfoC],
    source_blob: str,
    source_lines: Chunk
) -> Dict[Tuple[int, int], str]:
    """
    Generate doc comments for each function definition.

    This function generates documentation comments for a list of C function definitions.
    It assembles a snippet of code for each function, prompts the language model to generate a comment,
    extracts the first C block comment from the response, and stores the result in a dictionary keyed by the
    function's header span.

    Parameters:
    - `llm`: The language model instance used for generating comments.
    - `cfg`: The generation configuration.
    - `messages`: A list of messages exchanged with the language model.
    - `defs`: A list of function definition information.
    - `source_blob`: The original source code as a string.
    - `source_lines`: The source code broken down into chunks.

    Returns:
    - A dictionary mapping each function's header span to its corresponding documentation comment.
    """
    doc_map: Dict[Tuple[int, int], str] = {}

    # No nesting to worry about; process in source order
    for info in defs:
        snippet = assemble_snippet_for_c(source_lines, info)

        echo("\n[C] Snippet...\n")
        echo(snippet)

        prompt = (
            "Write exactly the documentation comment for this C function. "
            "Return a C block comment suitable to be placed immediately above the function "
            "declaration (no code), describing purpose, parameters and return value.\n\n"
            f"{snippet}\n"
        )
        messages.append({"role": "user", "content": prompt})
        reply = llm.generate(messages, cfg=cfg)
        echo(f"\n[C] LLM output:\n\n{reply}")
        messages.pop()

        body = _extract_first_c_comment_block(reply)
        if not body:
            body = f"function `{info.qualname}` - documentation generation failed."
        doc_map[(info.header_start, info.header_end)] = body

    return doc_map


# ---------------- Textual patcher

def patch_comments_textually_c(source_lines: Chunk, defs: List[DefInfoC], doc_map: Dict[Tuple[int, int], str]) -> Chunk:
    """
    Insert or replace documentation blocks for each C function.

    This function updates the source code by inserting or replacing documentation blocks above each C function definition.
    It checks if a '/* ... */' or contiguous '//' block already exists immediately above the header, and replaces it if so.
    Otherwise, it inserts a new block immediately above the header.

    The `doc_map` dictionary is used to store the documentation comments for each function definition, with keys being
    tuples of (header_start, header_end) to distinguish multiple definitions of the same name.

    Parameters:
    - `source_lines`: The original source code as a list of lines.
    - `defs`: A list of `DefInfoC` objects containing information about each C function definition.
    - `doc_map`: A dictionary mapping function definition keys to their corresponding documentation comments.

    Returns:
    - The updated source code with inserted or replaced documentation blocks.
    """
    out_lines = source_lines[:]

    # Edit bottom-up to keep line numbers stable
    for info in sorted(defs, key=lambda d: d.start, reverse=True):
        key = (info.header_start, info.header_end)
        if key not in doc_map:
            continue

        doc = doc_map[key].rstrip()

        header_line_text = source_lines[info.header_start - 1]
        indent = header_line_text[: len(header_line_text) - len(header_line_text.lstrip())]
        new_block_lines = _render_c_block_comment(doc, indent)

        existing = _scan_existing_comment_block_above(out_lines, info.header_start)
        if existing:
            s_1b, e_1b = existing
            out_lines[s_1b - 1: e_1b] = new_block_lines
        else:
            insert_at = info.header_start - 1
            out_lines[insert_at:insert_at] = new_block_lines  # no forced blank line

    return out_lines


# ---------------- Orchestrator

def generate_language_comments(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    source_blob: str,
    source_lines: Chunk
) -> Chunk:
    """
    Generate language comments for a given C source code.

    This function performs the following steps:

    - Parses the C source code using Tree-sitter C.
    - Collects function definitions, ignoring forward declarations.
    - Asks the LLM to generate block comment text for each definition.
    - Patches the original source code by inserting or replacing blocks above function headers.
    - Returns the updated source lines.

    Parameters:
    - `llm`: The LocalChatModel instance used for generating comments.
    - `cfg`: The GenerationConfig instance containing configuration settings.
    - `messages`: The Messages object containing any relevant messages.
    - `source_blob`: The raw C source code as a string.
    - `source_lines`: The original source code split into lines.

    Returns:
    - The updated source lines with generated comments.
    """
    echo("Parsing C source with Tree-sitter...")
    defs = iter_defs_with_info_c(source_blob)
    echo(f"Found {len(defs)} C function definition(s)")

    echo("Generating C comments...\n")
    doc_map = generate_comments_c(llm, cfg, messages, defs, source_blob, source_lines)

    echo("Applying C patches...\n")
    return patch_comments_textually_c(source_lines, defs, doc_map)
