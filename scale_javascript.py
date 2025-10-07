#!/usr/bin/env python3
# scale_js_tsitter.py
# pip install tree-sitter-languages

from __future__ import annotations

from dataclasses import dataclass, field, replace
from scale_llm import LocalChatModel, GenerationConfig, Messages, Chunk
from scale_log import echo
from tree_sitter import Parser, Language
from tree_sitter_javascript import language as js_language
from typing import Dict, List, Optional, Tuple, Iterable, Any
import textwrap


# ---- Tree-sitter setup ------------------------------------------------------


LanguageT = object
ParserT = object


# python -m pip uninstall -y tree-sitter-languages tree-sitter tree-sitter-javascript
# python -m pip install -U "tree-sitter==0.21.0" "tree-sitter-javascript==0.21.0"


def _load_js_language_and_parser() -> Tuple[Language, Parser]:
    # Get the export (capsule or Language)
    ptr_or_lang: Any = js_language()

    # Wrap capsule -> Language, or accept Language directly.
    if isinstance(ptr_or_lang, Language):
        lang = ptr_or_lang
    else:
        # The second argument is just a label for the language; any string is fine.
        lang = Language(ptr_or_lang, "JavaScript")

    # Create a parser using whichever API this wheel exposes
    try:
        p = Parser()
        p.set_language(lang)              # classic API
    except AttributeError:
        # Fallback: some builds use Parser(Language) ctor
        p = Parser(lang)

    return lang, p


JS_LANGUAGE, JS_PARSER = _load_js_language_and_parser()


# ---- DefInfo for JavaScript --------------------------------------------------


@dataclass(frozen=True)
class DefInfoJS:
    qualname: str                  # e.g. "foo", "ClassName.method", "obj.method"
    node: object                   # tree_sitter.Node
    kind: str                      # "function" | "class" | "method" | "var_func" | "var_arrow" | "obj_method"
    start: int                     # header start line (1-based; includes the 'function'/'class' etc.)
    end: int                       # definition end line (inclusive)
    header_start: int              # header start line (same as start)
    header_end: int                # header end line (line before body block begins)
    depth: int                     # 0 = module level
    parent_id: Optional[int]       # id(parent node) or None
    children_ids: Tuple[int, ...] = field(default_factory=tuple)


# ---- General utilities ------------------------------------------------------


def _to_1based(row0: int) -> int:
    return row0 + 1


def _line_span_from_node(n) -> Tuple[int, int]:
    # convert 0-based rows to 1-based line numbers
    return _to_1based(_row_of(n.start_point)), _to_1based(_row_of(n.end_point))


def _node_field(n, field: str):
    return n.child_by_field_name(field)


def _node_text(source_bytes: bytes, n) -> str:
    return source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")


def _get_text_for_lines(source_lines: Chunk, a: int, b: int) -> str:
    a = max(1, a)
    b = min(len(source_lines), b)
    if a > b:
        return ""
    return "\n".join(source_lines[a - 1:b])


def _leading_spaces_count(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _row_of(point) -> int:
    """Return 0-based row from a tree-sitter point (supports tuple or object)."""
    try:
        return point.row            # object with .row
    except AttributeError:
        return point[0]             # tuple (row, column)


# ---- Qualname extraction ----------------------------------------------------


def _ident_text(source_bytes: bytes, n) -> Optional[str]:
    if n is None:
        return None
    if n.type in ("identifier", "property_identifier"):
        return _node_text(source_bytes, n)
    if n.type == "private_property_identifier":  # #method
        return _node_text(source_bytes, n)
    if n.type == "string":
        # object literal key: "foo"
        s = _node_text(source_bytes, n).strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"', "`"):
            return s[1:-1]
        return s
    return None


def _qual_for_function_decl(source_bytes: bytes, n, scope: List[str]) -> str:
    name = _ident_text(source_bytes, _node_field(n, "name")) or "<anonymous>"
    return ".".join(scope + [name]) if scope else name


def _qual_for_class_decl(source_bytes: bytes, n, scope: List[str]) -> str:
    name = _ident_text(source_bytes, _node_field(n, "name")) or "<anonymous>"
    return ".".join(scope + [name]) if scope else name


def _qual_for_method(source_bytes: bytes, n, scope: List[str]) -> str:
    # method_definition: field 'name'
    name = _ident_text(source_bytes, _node_field(n, "name")) or "<anonymous>"
    return ".".join(scope + [name]) if scope else name


def _qual_for_var_declarator(source_bytes: bytes, declarator, scope: List[str]) -> str:
    # variable_declarator: field 'name' (identifier / pattern)
    name = _ident_text(source_bytes, _node_field(declarator, "name")) or "<anonymous>"
    return ".".join(scope + [name]) if scope else name


def _qual_for_obj_method(source_bytes: bytes, pair_or_method, scope: List[str], enclosing_obj: Optional[str]) -> str:
    # object literal method or pair with function value
    key_node = _node_field(pair_or_method, "name") or _node_field(pair_or_method, "key")
    method = _ident_text(source_bytes, key_node) or "<anonymous>"
    if enclosing_obj:
        parts = scope + [enclosing_obj, method]
    else:
        parts = scope + [method]
    return ".".join(parts) if parts else method


# ---- Header end heuristics --------------------------------------------------


def _header_end_for_function_like(n) -> int:
    """
    If body is a statement_block, header ends on the line before the block's start.
    For arrow functions with expression body, header is on the function line.
    """
    body = _node_field(n, "body") if n.type != "method_definition" else _node_field(n, "body")
    if body and body.type in ("statement_block", "class_body"):
        return _to_1based(_row_of(body.start_point)) - 1
    if n.type == "method_definition":
        val = _node_field(n, "value")
        if val:
            blk = _node_field(val, "body")
            if blk and blk.type == "statement_block":
                return _to_1based(_row_of(blk.start_point)) - 1
    return _to_1based(_row_of(n.start_point))

# ---- AST walk and DefInfo collection ---------------------------------------


def _iter_children(n) -> Iterable:
    for i in range(n.named_child_count):
        yield n.named_child(i)


def iter_defs_with_info_js(source_blob: str) -> List[DefInfoJS]:
    """
    Parse the JS module and collect definition-like constructs:
      - function_declaration
      - class_declaration
      - method_definition in class_body
      - variable_declarator with init function_expression or arrow_function
      - object method in object (method_definition) and pair with function/arrow value
    """
    source_bytes = source_blob.encode("utf-8", errors="replace")
    tree = JS_PARSER.parse(source_bytes)
    root = tree.root_node

    results: List[DefInfoJS] = []
    children_map: Dict[int, List[int]] = {}
    scope_names: List[str] = []
    scope_nodes: List[object] = []

    def add_child(parent_node: Optional[object], child_node: object) -> None:
        if parent_node is None:
            return
        children_map.setdefault(id(parent_node), []).append(id(child_node))

    def walk(node) -> None:
        t = node.type

        def mk_info(qualname: str, kind: str) -> DefInfoJS:
            start_1b, end_1b = _line_span_from_node(node)
            header_end = _header_end_for_function_like(node)
            return DefInfoJS(
                qualname=qualname,
                node=node,
                kind=kind,
                start=start_1b,
                end=end_1b,
                header_start=start_1b,
                header_end=header_end,
                depth=len(scope_nodes),
                parent_id=id(scope_nodes[-1]) if scope_nodes else None,
            )

        # ---- function_declaration ----
        if t == "function_declaration":
            qual = _qual_for_function_decl(source_bytes, node, scope_names)
            info = mk_info(qual, "function")
            results.append(info)
            add_child(scope_nodes[-1] if scope_nodes else None, node)
            scope_nodes.append(node)
            scope_names.append(qual.split(".")[-1])
            for c in _iter_children(node):
                walk(c)
            scope_names.pop()
            scope_nodes.pop()
            return

        # ---- class_declaration ----
        if t == "class_declaration":
            qual = _qual_for_class_decl(source_bytes, node, scope_names)
            info = mk_info(qual, "class")
            results.append(info)
            add_child(scope_nodes[-1] if scope_nodes else None, node)
            scope_nodes.append(node)
            scope_names.append(qual.split(".")[-1])
            for c in _iter_children(node):
                walk(c)
            scope_names.pop()
            scope_nodes.pop()
            return

        # ---- method_definition (class body) ----
        if t == "method_definition":
            qual = _qual_for_method(source_bytes, node, scope_names)
            info = mk_info(qual, "method")
            results.append(info)
            add_child(scope_nodes[-1] if scope_nodes else None, node)
            for c in _iter_children(node):
                walk(c)
            return

        # ---- variable_declarator with function/arrow ----
        if t == "variable_declarator":
            init = _node_field(node, "value")
            if init and init.type in ("function", "function_expression", "arrow_function"):
                # Node types vary slightly across grammar versions. Be permissive.
                qual = _qual_for_var_declarator(source_bytes, node, scope_names)
                kind = "var_arrow" if init.type == "arrow_function" else "var_func"
                info = mk_info(qual, kind)
                results.append(info)
                add_child(scope_nodes[-1] if scope_nodes else None, node)
                for c in _iter_children(node):
                    walk(c)
                return

        # ---- object property as method ----
        # Two shapes appear: 'method_definition' inside object, or 'pair' with value function/arrow
        if t == "pair" or (t == "method_definition" and node.parent and node.parent.type == "object"):
            enclosing = None
            # If parent chain is a variable_declarator, use its id as object name
            p = node.parent
            while p is not None and p.type not in ("program",):
                if p.type == "variable_declarator":
                    enclosing = _ident_text(source_bytes, _node_field(p, "name"))
                    break
                p = p.parent
            # Identify if this is a method-like construct
            value = _node_field(node, "value")
            is_method = (node.type == "method_definition") or (value and value.type in ("function", "function_expression", "arrow_function"))
            if is_method:
                qual = _qual_for_obj_method(source_bytes, node, scope_names, enclosing)
                info = mk_info(qual, "obj_method")
                results.append(info)
                add_child(scope_nodes[-1] if scope_nodes else None, node)
                for c in _iter_children(node):
                    walk(c)
                return

        # Generic descent
        for c in _iter_children(node):
            walk(c)

    walk(root)

    # Finalise children_ids immutably
    completed: List[DefInfoJS] = []
    for info in results:
        kids = tuple(children_map.get(id(info.node), []))
        completed.append(replace(info, children_ids=kids))

    return sorted(completed, key=lambda d: d.start)


# ---- Depth order -------------------------------------------------------------


def deepest_first_js(defs: List[DefInfoJS]) -> List[DefInfoJS]:
    # depth desc; stable: start asc, end desc
    return sorted(defs, key=lambda d: (d.depth, d.start, -d.end), reverse=True)


# ---- Snippet assembly (stubbing direct children) ----------------------------


def _render_jsdoc_block(text: str, base_indent: str) -> List[str]:
    lines = text.splitlines()
    out = [f"{base_indent}/**"]
    if lines:
        for ln in lines:
            out.append(f"{base_indent} * {ln.rstrip()}")
    else:
        out.append(f"{base_indent} * (no documentation)")
    out.append(f"{base_indent} */")
    return out


def _extract_first_comment_block(reply: str) -> str:
    lines = textwrap.dedent(reply).split("\n")
    stripped = [ln.strip() for ln in lines]
    start_idx = None
    try:
        start_idx = stripped.index("/**") + 1
    except ValueError:
        return ""

    for end_idx in range(start_idx, len(lines) + 1):
        if "*/" in lines[end_idx]:
            break
    else:
        end_idx = len(lines) + 1

    lines = lines[start_idx:end_idx]
    lines = [line.lstrip(" *") for line in lines]
    lines = "\n".join(lines)

    return textwrap.dedent(lines).strip()


def assemble_snippet_for_js(
    info_by_id: Dict[int, DefInfoJS],
    source_lines: Chunk,
    node_id: int,
    docs_by_id: Dict[int, str],
) -> str:
    """
    Header: lines header_start..header_end
    Body:   direct children replaced with [JSDoc + header], body omitted; other lines preserved.
    We splice by line ranges to preserve indentation and comments.
    """
    info = info_by_id[node_id]

    header_text = _get_text_for_lines(source_lines, info.header_start, info.header_end)
    body_chunks: List[str] = []

    direct_children = [info_by_id[cid] for cid in info.children_ids]
    direct_children.sort(key=lambda d: d.start, reverse=True)  # replace from bottom to top

    cursor = info.header_end + 1
    for child in direct_children:
        if cursor <= child.start - 1:
            body_chunks.append(_get_text_for_lines(source_lines, cursor, child.start - 1))
        # child stub
        header_last_line = source_lines[child.header_end - 1] if 1 <= child.header_end <= len(source_lines) else ""
        indent = " " * _leading_spaces_count(header_last_line)
        child_doc = docs_by_id.get(id(child.node), "")
        jsdoc_block = "\n".join(_render_jsdoc_block(child_doc, indent))
        child_header = _get_text_for_lines(source_lines, child.header_start, child.header_end)
        body_chunks.append(jsdoc_block + ("\n" if jsdoc_block and child_header else "") + child_header)
        cursor = child.end + 1

    if cursor <= info.end:
        body_chunks.append(_get_text_for_lines(source_lines, cursor, info.end))

    parts: List[str] = [header_text]
    if body_chunks:
        if header_text and not header_text.endswith("\n"):
            parts.append("\n")
        parts.append("\n".join(ch for ch in body_chunks if ch))
    return "".join(parts)


# ---- LLM driver -------------------------------------------------------------


def generate_comments_js(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    defs: List[DefInfoJS],
    source_blob: str,
    source_lines: Chunk
) -> Dict[str, str]:
    """
    Generate JSDoc content for JS definitions. Deepest-first so parents see child stubs.
    """
    docs_by_qualname: Dict[str, str] = {}
    docs_by_node_id: Dict[int, str] = {}
    info_by_id: Dict[int, DefInfoJS] = {id(d.node): d for d in defs}

    for info in deepest_first_js(defs):
        node_id = id(info.node)
        snippet = assemble_snippet_for_js(info_by_id, source_lines, node_id, docs_by_node_id)

        echo("\n[JS] Snippet...\n")
        echo(snippet)

        prompt = (
            "Write exactly the documentation comment for this JavaScript program chunk. "
            "Return only the comment content (no code), suitable for a JSDoc block placed "
            "immediately above the definition:\n\n"
            f"{snippet}\n"
        )
        messages.append({"role": "user", "content": prompt})
        reply = llm.generate(messages, cfg=cfg)
        echo(f"\n[JS] LLM output:\n\n{reply}")
        messages.pop()

        doc = _extract_first_comment_block(reply)
        if not doc:
            doc = f"{info.kind} `{info.qualname}` - documentation generation failed."
        docs_by_qualname[info.qualname] = doc
        docs_by_node_id[node_id] = doc

    return docs_by_qualname


# ---- Textual patcher (insert/replace above headers) -------------------------


def _scan_existing_comment_block_above(source_lines: Chunk, header_start_line_1b: int) -> Optional[Tuple[int, int]]:
    """
    Detect an existing comment block immediately above header_start_line:
      - a block comment '/* ... */' whose last line ends just above the header, or
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


def patch_comments_textually_js(source_lines: Chunk, defs: List[DefInfoJS], doc_map: Dict[str, str]) -> Chunk:
    """
    Insert or replace documentation blocks for each JS definition:
    - If there is a block comment or a contiguous '//' block ending immediately above the header, replace it.
    - Otherwise insert a new JSDoc block immediately above the header.
    """
    out_lines = source_lines[:]

    for info in sorted(defs, key=lambda d: d.start, reverse=True):
        qual = info.qualname
        if qual not in doc_map:
            continue
        doc = doc_map[qual]

        header_line_text = source_lines[info.header_start - 1]
        indent = header_line_text[: len(header_line_text) - len(header_line_text.lstrip())]

        new_block_lines = _render_jsdoc_block(doc, indent)

        existing = _scan_existing_comment_block_above(out_lines, info.header_start)
        if existing:
            s_1b, e_1b = existing
            out_lines[s_1b - 1: e_1b] = new_block_lines
        else:
            insert_at = info.header_start - 1
            out_lines[insert_at:insert_at] = [""] + new_block_lines

    return out_lines


# ---- Orchestrator -----------------------------------------------------------


def generate_language_comments(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    source_blob: str,
    source_lines: Chunk
) -> Chunk:
    """
    End-to-end for JavaScript using Tree-sitter:
      - parse,
      - collect DefInfoJS,
      - generate JSDoc text with the LLM (deepest-first),
      - patch textually by inserting/replacing comment blocks above headers,
      - return the updated source lines.
    """
    echo("Parsing JavaScript source with Tree-sitter...")
    defs = iter_defs_with_info_js(source_blob)
    echo(f"Found {len(defs)} JS definitions")

    echo("Generating JSDoc comments...\n")
    doc_map = generate_comments_js(llm, cfg, messages, defs, source_blob, source_lines)

    echo("Applying JS patches...\n")
    return patch_comments_textually_js(source_lines, defs, doc_map)
