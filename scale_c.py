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
from scale_blocks import BlockTarget, SegStatement, structural_breaks, SEG_MIN_LEADING_DECLS, SLASH_BLOCK_STYLE
from scale_filedoc import FileDocTarget, scan_brace_leading_zone
from scale_llm import LocalChatModel, GenerationConfig, Messages, Chunk
from scale_log import echo
from scale_text import fit_snippet, MARKER_C
from tree_sitter import Parser, Language  # type: ignore
from tree_sitter_c import language as c_language
from typing import Dict, List, Optional, Tuple, Any
import re
import textwrap


# ---------------- Tree-sitter bootstrap (handles capsule vs Language object, and both Parser APIs)

def _load_c_language_and_parser() -> Tuple[Language, Parser]:
    """
    Load the C language and parser.

    This function loads the C language and creates a parser instance. It handles cases where `c_language()` returns a
    `Language` object or a label, and returns a tuple containing the loaded `Language` object and the created `Parser`
    instance.

    Returns:
    - A tuple of two objects: the loaded `Language` object and the created `Parser` instance.
    """

    ptr_or_lang: Any = c_language()

    # Wrap capsule -> Language, or accept Language directly.
    if isinstance(ptr_or_lang, Language):
        lang = ptr_or_lang
    else:
        try:
            lang = Language(ptr_or_lang)        # new API (tree-sitter >= 0.22)
        except TypeError:
            lang = Language(ptr_or_lang, "C")   # old 0.21 API (second arg is a label)

    # Some builds expose Parser().set_language(...), others Parser(Language)
    try:
        p = Parser()
        p.set_language(lang)
    except AttributeError:
        p = Parser(lang)

    return lang, p


C_LANGUAGE, C_PARSER = _load_c_language_and_parser()


def _parse_c(source_blob: str) -> Tuple[Any, bytes]:
    """
    Normalise line endings and parse C source once.

    Tree-sitter counts rows by '\\n' only, whereas the source may use '\\r' or '\\r\\n'. Line endings are normalised to
    '\\n' so that node rows line up with the caller's line-split source. The same normalised bytes are returned and used
    for any node-text extraction, keeping byte offsets self-consistent with the parse tree.

    Parameters:
    - `source_blob`: The C source code as a string.

    Returns:
    - A tuple of the parsed tree and the normalised source bytes it was parsed from.
    """

    norm = source_blob.replace("\r\n", "\n").replace("\r", "\n")
    source_bytes = norm.encode("utf-8", errors="replace")
    return C_PARSER.parse(source_bytes), source_bytes


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

    This function supports both tuple and object forms of points. If the input is an object with a `.row` attribute,
    it is used directly. Otherwise, it assumes the input is a tuple (row, col) and returns the first element (the
    row number).

    Parameters:
    - `point`: The tree-sitter point to extract the row from.

    Returns:
    - The row number of the point as an integer.
    """
    
    try:
        return point.row  # object with .row
    except AttributeError:
        return point[0]   # tuple (row, col)


def _col_of(point) -> int:
    """
    Return the 0-based column of a tree-sitter point, tolerating both the object (`.column`) and tuple (`[1]`) forms.

    Parameters:
    - `point`: The tree-sitter point to extract the column from.

    Returns:
    - The 0-based column of the point as an integer.
    """

    try:
        return point.column  # object with .column
    except AttributeError:
        return point[1]      # tuple (row, col)


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

    This method returns a string containing the specified range of lines, joined by newline characters.
    If the start line number is greater than the end line number, an empty string is returned.

    Parameters:
    - `source_lines`: The list of source code lines.
    - `a`: The starting line number (inclusive).
    - `b`: The ending line number (exclusive).

    Returns:
    - A string containing the extracted lines.
    """

    a = max(1, a)
    b = min(len(source_lines), b)
    if a > b:
        return ""
    return "\n".join(source_lines[a - 1:b])


def _node_text(source_bytes: bytes, n) -> str:
    """
    Decode the exact source slice for a node.

    Parameters:
    - `source_bytes`: The bytes containing the source code.
    - `n`: The node for which to decode the source slice.

    Returns:
    - A string representing the decoded source slice, or an empty string if `n` is `None`.
    """

    if n is None:
        return ""

    # Use 'replace' to be robust to any odd bytes in the file.
    return source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")


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


def file_doc_target_c(source_blob: str, source_lines: Chunk) -> Optional[FileDocTarget]:
    """
    Build the file-level header doccomment target for a C source file.

    Gathers the entire leading-comment zone at the top of the file - which may span several contiguous comment blocks
    (mixed `/* ... */` and `//`, separated by blank lines but with no intervening code) - into one target. The scan
    starts after an optional shebang, collects every comment line until the first real code or preprocessor line, and
    marks the pure-content comment lines (block continuations and `//` lines) as description-eligible while leaving
    delimiters, single-line `/* ... */` comments, and blank continuations to be preserved. The slots for replacing,
    appending, or freshly inserting a description are computed from the zone's shape.

    Parameters:
    - `source_blob`: The complete source text (unused; accepted for provider-signature symmetry).
    - `source_lines`: The source split into individual lines.

    Returns:
    - A `FileDocTarget`, or None if the file is empty.
    """

    return scan_brace_leading_zone(source_lines, SLASH_BLOCK_STYLE)


# ---------------- C DefInfo


@dataclass(frozen=True)
class DefInfoC:
    """
    Represents information about a C function definition.

    This class abstracts over the details of a C function definition, providing a structured representation
    that can be used to generate documentation comments for the function.

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

    Note:
        This class is designed to be used in conjunction with the `LLM exchange` module to generate
        documentation comments for C functions.
    """

    qualname: str
    node: object
    kind: str
    start: int
    end: int
    header_start: int
    header_end: int
    depth: int
    parent_id: Optional[int]
    children_ids: Tuple[int, ...] = field(default_factory=tuple)


# ---------------- Include discovery --------------------------------------------------------------


def _display_include_target(tok: str) -> str:
    """
    Render the include target as it appeared, preserving any quotes or angle brackets.

    If the token is already quoted (e.g. `"...` or `'...'`) or enclosed in angle brackets (`<...>`), return it unchanged.
    Otherwise, return the bare token (e.g. a macro name).
    """

    tok = tok.strip()
    # Already quoted or angled? keep as-is
    if (len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ('"', "'")) or (tok.startswith("<") and tok.endswith(">")):
        return tok
    return tok


def _collect_includes_c(tree, source_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Collect a list of #include directives in source order.

    This function parses the given C source code and extracts a plain-English list of #include directives.
    It supports various forms of includes, including:

    *   `#include <...>`
    *   `#include "..."`
    *   `#include MACRO_NAME`

    The function returns a list of tuples containing the line number and a description of each include directive.

    Parameters:
    - `tree`: The parsed Tree-sitter tree for the source.
    - `source_bytes`: The (normalised) source bytes the tree was parsed from.

    Returns:
    - A list of tuples, where each tuple contains the line number (1-based) and a description of the include directive.
    """

    out: List[Tuple[int, str]] = []
    root = tree.root_node

    def add(n, payload: str) -> None:
        """
        Add a payload to the output list at the specified line number.

        Parameters:
        - `n`: A Tree-sitter node representing the position where the payload should be added.
        - `payload`: The string to be appended to the output list.

        Notes:
        The line number is calculated from the 0-based row of the node's start point, converted to 1-based.
        """

        ln = _to_1based(_row_of(n.start_point))
        out.append((ln, payload))

    stack = [root]
    while stack:
        n = stack.pop()
        t = n.type

        if t == "preproc_include":
            # Prefer the grammar field if available
            path_node = n.child_by_field_name("path")
            if path_node is not None:
                raw = _node_text(source_bytes, path_node).strip()
                # Normalise display but preserve delimiters if present
                target = _display_include_target(raw)
                add(n, f"- {target}")
            else:
                # Fallback: parse the directive text
                full = _node_text(source_bytes, n)
                m = re.search(r"<([^>]+)>", full)
                if m:
                    add(n, f"- <{m.group(1)}>")
                else:
                    m = re.search(r'"([^"\n\r]+)"', full)
                    if m:
                        add(n, f'- "{m.group(1)}"')
                    else:
                        # Try to catch identifier-like includes (e.g. macros)
                        m = re.search(r"#\s*include\s+([A-Za-z_]\w*)", full)
                        target = m.group(1) if m else "<unknown>"
                        add(n, f"- Includes {target}")
            # Do not descend into children of this directive
            continue

        # Generic DFS over named children
        for i in range(n.named_child_count - 1, -1, -1):
            stack.append(n.named_child(i))

    out.sort(key=lambda t: t[0])
    return out


def describe_includes_c(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    tree,
    source_bytes: bytes
) -> None:
    """
    Build a list of #includes from the source code and feed it to the LLM as extra context.

    This method mirrors the Python/JS flow by echoing the includes, then pushing a short 'OK' acknowledgement prompt.
    It collects the includes using `_collect_includes_c`, formats them into a payload, and appends a prompt to the message list.
    The LLM is then asked to generate a response based on the updated message list, which is echoed back to the user.

    Parameters:
    - `llm`: The LocalChatModel instance used for LLM interactions.
    - `cfg`: The GenerationConfig instance used for LLM generation.
    - `messages`: The Messages instance used for storing and sending messages.
    - `tree`: The parsed Tree-sitter tree for the source.
    - `source_bytes`: The (normalised) source bytes the tree was parsed from.

    Notes:
    This method does not return any value, as it is designed to update the message list and prompt the LLM for a response.
    """

    items = _collect_includes_c(tree, source_bytes)
    if not items:
        return
    lines = [text for _, text in items]
    payload = "\n".join(lines)

    echo(f"\n[C] Includes...\n{payload}")
    prompt = (
        "For additional context, here is a list of includes within this program:\n\n"
        f"{payload}\n\n"
        "Please respond by saying 'OK'. No other commentary is required at this time."
    )
    messages.append({"role": "user", "content": prompt})

    reply = llm.generate(messages, cfg=cfg)
    echo(f"\n[C] LLM output: {reply}")
    messages.append({"role": "assistant", "content": reply})


# ---------------- Collect function definitions (ignores forward declarations)


def iter_defs_with_info_c(tree, source_bytes: bytes) -> List[DefInfoC]:
    """
    Collect real function definitions from a C file.

    This function walks a parsed C tree, identifies function definitions, and collects metadata.
    Forward declarations are excluded from the results.

    Parameters:
    - `tree`: The parsed Tree-sitter tree for the source.
    - `source_bytes`: The (normalised) source bytes the tree was parsed from.

    Returns:
    - A sorted list of `DefInfoC` objects, each containing information about a real function definition.
    """

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

        This function traverses the declarator subtree to find the function identifier node.
        If found, it returns the identifier as a string. If not found, it returns "<anonymous>".

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
        Collect function definitions from the abstract syntax tree (AST).

        This function recursively traverses the AST, identifying function definitions and collecting metadata.
        For each function definition, it extracts the qualified name, line span, header span, and other relevant information.

        Parameters:
        - `n`: The current node in the AST to process.

        Notes:
        - Function definitions are identified by their type (`"function_definition"`).
        - Anonymous functions are assigned a default qualified name (`"<anonymous>"`).
        - The function collects metadata for each definition, including the qualified name, line span, header span, and more.
        - The collected metadata is stored in a `DefInfoC` object.
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


# ---------------- Within-function block targets (the `-b` block pass)


# Statements that open a brace block whose source-line span gates a paragraph break before/after it. C has no
# nested functions, so there is no "after a def" rule and no opaque-definition handling; a bare `{ ... }` scope
# block is treated as a block too.
_C_BLOCK_OPENERS = {"if_statement", "for_statement", "while_statement", "do_statement", "switch_statement",
                    "compound_statement"}


def _named_children(n) -> List:
    """Return the named children of a tree-sitter node as a list."""
    return [n.named_child(i) for i in range(n.named_child_count)]


def _is_c_statement(n) -> bool:
    """Report whether a node is a C statement (a `declaration` or any `*_statement`), filtering out comments etc."""
    return n.type == "declaration" or n.type.endswith("_statement")


def _c_suite(node) -> Optional[List]:
    """
    Return the statement list held by a body/branch node, or None.

    Unwraps an `else_clause` to the branch inside it; expands a `compound_statement` to its statement children; and
    treats a single unbraced statement (e.g. the body of `for (...) x++;`) as a one-element suite - the analogue of
    Python's `_sub_statement_lists` entries.

    Parameters:
    - `node`: A body/branch node (or None).

    Returns:
    - The statement list, or None if the node holds no statements.
    """

    if node is None:
        return None
    if node.type == "else_clause":
        kids = _named_children(node)
        return _c_suite(kids[0]) if kids else None
    if node.type == "compound_statement":
        stmts = [c for c in _named_children(node) if _is_c_statement(c)]
        return stmts or None
    if _is_c_statement(node):
        return [node]
    return None


def _c_statement_lists(stmt) -> List[List]:
    """
    Return the statement lists nested directly inside a C compound statement (the analogue of Python suites).

    Covers the consequence/alternative of an `if`, the body of a `for`/`while`/`do`, each `case` body of a `switch`,
    and the contents of a bare `{ ... }` block. Used to recurse into a routine body one nesting level at a time.

    Parameters:
    - `stmt`: The statement to inspect.

    Returns:
    - A list of statement lists (empty for a simple statement).
    """

    t = stmt.type
    lists: List[List] = []
    if t == "if_statement":
        for fld in ("consequence", "alternative"):
            sl = _c_suite(stmt.child_by_field_name(fld))
            if sl:
                lists.append(sl)
    elif t in ("for_statement", "while_statement", "do_statement"):
        sl = _c_suite(stmt.child_by_field_name("body"))
        if sl:
            lists.append(sl)
    elif t == "switch_statement":
        body = stmt.child_by_field_name("body")
        if body is not None:
            for case in _named_children(body):
                if case.type == "case_statement":
                    cl = [c for c in _named_children(case) if _is_c_statement(c)]
                    if cl:
                        lists.append(cl)
    elif t == "compound_statement":
        cl = [c for c in _named_children(stmt) if _is_c_statement(c)]
        if cl:
            lists.append(cl)
    return lists


def _c_span(node) -> int:
    """Source-line span of a node (its triviality measure for the block-size gate)."""
    return _row_of(node.end_point) - _row_of(node.start_point) + 1


def _c_closed_block(prev_node, resume_start: int, body_node) -> Optional[object]:
    """
    Return the outermost brace block that closed between `prev_node` and a dedent resuming at `resume_start`.

    Climbs the parent chain from the previous statement (stopping at the routine body), keeping the outermost block
    opener that ended before the resuming line - the tree-sitter analogue of `scale_python._seg_closed_block`.

    Parameters:
    - `prev_node`: The statement immediately before the dedent.
    - `resume_start`: The 1-based line the dedent resumes on.
    - `body_node`: The routine's body block (the climb stops here).

    Returns:
    - The outermost closed block node, or None if none closed.
    """

    best = None
    n = prev_node.parent
    while n is not None and n is not body_node:
        if n.type in _C_BLOCK_OPENERS and _to_1based(_row_of(n.end_point)) < resume_start:
            best = n
        n = n.parent
    return best


def _collect_body_c(body_node, source_lines: Chunk) -> Tuple[List[int], Dict[int, str], List[SegStatement]]:
    """
    Walk a C function body, returning its legal block boundaries, their indentation, and the segmenter records.

    Mirrors `scale_python._body_boundaries` + `_seg_records`: every statement is recorded at every nesting depth
    (recursing into nested brace blocks), the first statement of an inner suite is excluded from the boundaries, and
    a line is a boundary only if it begins exactly one (non-suite-leading) statement at its first non-blank column -
    which naturally drops `a; b;` lines, continuation lines, and inline inner statements. The full statement list
    (including suite-leading statements) is returned for the segmenter, which needs the body's complete shape to
    reason about returns and dedents.

    Parameters:
    - `body_node`: The function's `compound_statement` body node.
    - `source_lines`: The full source split into lines.

    Returns:
    - A tuple `(boundary_lines, indent_of, seg_statements)`.
    """

    line_count: Dict[int, int] = {}
    line_col: Dict[int, int] = {}
    recorded: List[Tuple[int, int, object, bool]] = []  # (start, depth, node, first_in_scope)
    merge_map: Dict[int, int] = {}                       # return line -> anchor (preceding statement) line
    force_break_lines: set = set()                       # first real statement after a leading declaration run

    def walk(stmts: List, is_top: bool, depth: int) -> None:
        """Record each statement and recurse into its nested suites, tracking depth and suite position."""
        # A suite that is exactly `[simple_stmt, return]` is one paragraph: anchor it at the leading statement so the
        # comment pass sees both lines (a return alone gives it nothing to describe).
        if (len(stmts) == 2 and stmts[1].type == "return_statement"
                and stmts[0].type not in _C_BLOCK_OPENERS):
            a = _to_1based(_row_of(stmts[0].start_point))
            r = _to_1based(_row_of(stmts[1].start_point))
            if a != r:
                merge_map[r] = a
        # Leading-declaration heuristic: a scope opening with a run of declarations gets its first real statement
        # paragraphed off, so the declarations read as their own block (a `return` first statement is left to the
        # merge / trailing-return rules).
        ndecl = 0
        while ndecl < len(stmts) and stmts[ndecl].type == "declaration":
            ndecl += 1
        if ndecl >= SEG_MIN_LEADING_DECLS and ndecl < len(stmts) and stmts[ndecl].type != "return_statement":
            force_break_lines.add(_to_1based(_row_of(stmts[ndecl].start_point)))
        for idx, stmt in enumerate(stmts):
            skip = (not is_top) and idx == 0  # the first line of an inner suite is never a block start
            start = _to_1based(_row_of(stmt.start_point))
            if not skip:
                line_count[start] = line_count.get(start, 0) + 1
                line_col[start] = _col_of(stmt.start_point)
            recorded.append((start, depth, stmt, is_top and idx == 0))
            for sub in _c_statement_lists(stmt):
                walk(sub, False, depth + 1)

    top_stmts = [c for c in _named_children(body_node) if _is_c_statement(c)]
    walk(top_stmts, True, 0)

    # Boundary lines: exactly one statement start, at the line's first non-blank column.
    boundary_lines: List[int] = []
    indent_of: Dict[int, str] = {}
    for line, count in line_count.items():
        if count != 1 or not (1 <= line <= len(source_lines)):
            continue
        text = source_lines[line - 1]
        if text[: line_col[line]].strip() != "":
            continue
        indent_of[line] = text[: len(text) - len(text.lstrip())]
        boundary_lines.append(line)

    # A `[stmt; return]` anchor is the first statement of its suite (normally excluded); make it an addressable
    # boundary so the merged paragraph can be commented there.
    for anchor in set(merge_map.values()):
        if 1 <= anchor <= len(source_lines) and anchor not in indent_of:
            text = source_lines[anchor - 1]
            indent_of[anchor] = text[: len(text) - len(text.lstrip())]
            boundary_lines.append(anchor)
    boundary_lines.sort()

    # Segmenter records, in source order, with depth/return/block-span/dedent annotations.
    recorded.sort(key=lambda r: r[0])
    seg_statements: List[SegStatement] = []
    for i, (start, depth, node, first_in_scope) in enumerate(recorded):
        end = _to_1based(_row_of(node.end_point))
        opens = _c_span(node) if node.type in _C_BLOCK_OPENERS else 0
        closed = 0
        if i > 0 and recorded[i - 1][1] > depth:
            blk = _c_closed_block(recorded[i - 1][2], start, body_node)
            closed = _c_span(blk) if blk is not None else 0
        seg_statements.append(SegStatement(
            start=start, end=end, depth=depth,
            is_return=node.type == "return_statement", is_def=False,
            opens_block=opens, first_in_scope=first_in_scope, closed_block=closed,
            merge_anchor=merge_map.get(start) if node.type == "return_statement" else None,
            force_break=start in force_break_lines,
        ))

    return boundary_lines, indent_of, seg_statements


def _doc_above_header(source_lines: Chunk, header_start: int) -> str:
    """
    Return the documentation text of the comment block immediately above a function header (or "" if there is none).

    This is the C analogue of Python's `ast.get_docstring`: the block provider parses the (possibly def-pass-
    annotated) source, so the `/* ... */` or `//` doc block the definition pass wrote above the function is read
    back here and fed to the block-comment pass as the routine's purpose - giving C the same per-routine context
    Python gets from a docstring.

    Parameters:
    - `source_lines`: The source split into lines (the same text the provider parses).
    - `header_start`: The 1-based line of the function header.

    Returns:
    - The doc text with comment delimiters/gutters stripped, or "" when no comment block sits above the header.
    """

    rng = _scan_existing_comment_block_above(source_lines, header_start)
    if rng is None:
        return ""
    s, e = rng
    block = "\n".join(source_lines[s - 1:e])
    if "/*" in block:
        return _extract_first_c_comment_block(block)
    # A `//` run: strip the leading slashes from each line.
    return "\n".join(ln.strip()[2:].strip() if ln.strip().startswith("//") else ln.strip()
                     for ln in block.split("\n")).strip()


def iter_block_targets_c(source_blob: str, source_lines: Chunk) -> List[BlockTarget]:
    """
    Build the within-function block targets for a C source file.

    Each function body becomes one `BlockTarget` carrying its header/body line spans, the lines that may legally
    begin a block, and the deterministic structural segmentation (so the block pass needs no model to segment). This
    is the C implementation of the language-agnostic provider interface consumed by `scale_blocks.annotate_blocks`.
    C has no nested functions, so the after-def / first-in-scope (post-docstring) rules are disabled.

    Parameters:
    - `source_blob`: The complete source text (parsed with Tree-sitter C).
    - `source_lines`: The same source split into individual lines.

    Returns:
    - A list of `BlockTarget`, one per function, in source order.
    """

    tree, source_bytes = _parse_c(source_blob)
    targets: List[BlockTarget] = []

    for info in iter_defs_with_info_c(tree, source_bytes):
        body_node = info.node.child_by_field_name("body")
        if body_node is None or body_node.type != "compound_statement":
            continue

        boundary_lines, indent_of, seg_statements = _collect_body_c(body_node, source_lines)
        top_stmts = [c for c in _named_children(body_node) if _is_c_statement(c)]
        if not top_stmts:
            continue
        body_start = _to_1based(_row_of(top_stmts[0].start_point))

        segments = structural_breaks(
            seg_statements, has_doc=False, boundary_lines=tuple(boundary_lines), body_end=info.end,
            allow_after_def=False, allow_first_in_scope=False,
        )
        targets.append(
            BlockTarget(
                qualname=info.qualname,
                kind="function",
                header_start=info.header_start,
                header_end=info.header_end,
                body_start=body_start,
                body_end=info.end,
                boundary_lines=tuple(boundary_lines),
                indent_of=indent_of,
                depth=0,
                doc=_doc_above_header(source_lines, info.header_start),
                segments=segments,
            )
        )

    return targets


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

    This function takes in a string of text and an indentation level, then formats it as a multi-line
    C-style block comment. If the input text is empty, it will render a placeholder comment instead.

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
            # rstrip the whole line, not just the content: a blank doc line would otherwise leave " * " with a
            # trailing space (the user-visible nuisance this guards against).
            out.append(f"{base_indent} * {ln.rstrip()}".rstrip())
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

        # Elide the body if this function is too large for the context window (the patch is unaffected).
        header_lines = max(1, info.header_end - info.header_start + 1)
        snippet, omitted = fit_snippet(llm, cfg, messages, snippet, header_lines, MARKER_C)
        if omitted:
            echo(f"[C] Elided {omitted} body line(s) from '{info.qualname}' to fit the context window")

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
    tree, source_bytes = _parse_c(source_blob)
    defs = iter_defs_with_info_c(tree, source_bytes)
    echo(f"Found {len(defs)} C function definition(s)")

    # Provide a list of #includes to the LLM (if there are any)
    echo("Identifying #includes...")
    describe_includes_c(llm, cfg, messages, tree, source_bytes)

    echo("Generating C comments...\n")
    doc_map = generate_comments_c(llm, cfg, messages, defs, source_blob, source_lines)

    echo("Applying C patches...\n")
    return patch_comments_textually_c(source_lines, defs, doc_map)
