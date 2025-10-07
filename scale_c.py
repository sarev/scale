#!/usr/bin/env python3
# scale_c.py

# pip install tree-sitter-c

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
    return row0 + 1


def _row_of(point) -> int:
    """Row from a tree-sitter point; supports tuple or object forms."""
    try:
        return point.row  # object with .row
    except AttributeError:
        return point[0]   # tuple (row, col)


def _line_span_from_node(n) -> Tuple[int, int]:
    """Inclusive 1-based start/end line span for node."""
    return _to_1based(_row_of(n.start_point)), _to_1based(_row_of(n.end_point))


def _get_text_for_lines(source_lines: Chunk, a: int, b: int) -> str:
    a = max(1, a)
    b = min(len(source_lines), b)
    if a > b:
        return ""
    return "\n".join(source_lines[a - 1:b])


def _leading_spaces_count(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _scan_existing_comment_block_above(source_lines: Chunk, header_start_line_1b: int) -> Optional[Tuple[int, int]]:
    """
    Detect an existing comment block immediately above header_start_line:
      - a block '/* ... */' whose last line ends just above the header, or
      - a contiguous run of '//' lines immediately above the header (no blank line).
    Returns (start_line_1b, end_line_1b), or None.
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
    Parse C file and collect real function definitions:
      - Node type: function_definition
      - Forward declarations are not included (they are 'declaration' with function declarator but no body).
    """
    source_bytes = source_blob.encode("utf-8", errors="replace")
    tree = C_PARSER.parse(source_bytes)
    root = tree.root_node

    results: List[DefInfoC] = []

    def header_span_for_function(fn_node) -> Tuple[int, int]:
        """
        Header ends on line before the compound_statement (body) begins.
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
    Snippet for LLM: header (possibly multi-line) plus body.
    We do not attempt to suppress inner functions (C does not support nested functions).
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
    Render as a C-style block comment. We keep it neutral (no JSDoc tags unless the text has them).
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
    Key doc_map by the (header_start, header_end) span so duplicates under different #ifs can co-exist.
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
    Insert or replace documentation blocks for each C function:
      - If there is a '/* ... */' or contiguous '//' block ending immediately above the header, replace it.
      - Otherwise insert a new block immediately above the header.
    Keys in doc_map are (header_start, header_end) to distinguish multiple definitions of the same name.
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
    End-to-end for C:
      - parse with Tree-sitter C,
      - collect function definitions (ignoring forward declarations),
      - ask LLM for block comment text per definition,
      - patch textually by inserting/replacing blocks above headers,
      - return updated source lines.
    """
    echo("Parsing C source with Tree-sitter...")
    defs = iter_defs_with_info_c(source_blob)
    echo(f"Found {len(defs)} C function definition(s)")

    echo("Generating C comments...\n")
    doc_map = generate_comments_c(llm, cfg, messages, defs, source_blob, source_lines)

    echo("Applying C patches...\n")
    return patch_comments_textually_c(source_lines, defs, doc_map)
