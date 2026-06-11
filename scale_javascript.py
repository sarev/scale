#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

This Python program is a tool for generating JSDoc comments for JavaScript code using a large language model (LLM)
as a content generator.

It leverages the Tree-sitter library to parse and analyse the JavaScript source code, identifying definition-like constructs
such as functions, classes, methods, and variables. The LLM is then used to generate documentation comments for these
definitions in a deepest-first order, ensuring that parents see child stubs. Finally, the program applies textual patches to
insert or replace existing comment blocks above headers, resulting in an updated source code with generated JSDoc comments.

# Highlights of Internal Workings

1. **Tree-sitter Setup**: The program initialises Tree-sitter by loading the JavaScript language and parser using the
   `tree_sitter_javascript` library.
2. **DefInfo Collection**: It defines a data class `DefInfoJS` to represent information about each definition-like construct,
   including its name, node, kind, start and end lines, header start and end lines, depth, parent ID, and children IDs. The
   program then uses Tree-sitter to parse the JavaScript source code and collect DefInfo instances for each definition.
3. **LLM-driven JSDoc Generation**: The program generates JSDoc comments for each definition using the LLM. It prompts the
   LLM with a snippet of the code, asking it to produce exactly the documentation comment for that specific JavaScript
   program chunk. The LLM output is then processed to extract the first comment block.
4. **Textual Patching**: The program applies textual patches to insert or replace existing comment blocks above headers.
   It scans for existing comment blocks immediately above each header and replaces them with the generated JSDoc comments
   if found. Otherwise, it inserts a new JSDoc block above the header.
5. **Orchestrator**: The `generate_language_comments` function orchestrates the entire process, from parsing and DefInfo
   collection to LLM-driven JSDoc generation and textual patching.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from scale_blocks import BlockTarget, SegStatement, structural_breaks, SEG_MIN_LEADING_DECLS, SLASH_BLOCK_STYLE
from scale_filedoc import FileDocTarget, scan_brace_leading_zone
from scale_llm import LocalChatModel, GenerationConfig, Messages, Chunk
from scale_log import echo, error
from scale_project import Symbol, apply_doc_order
from scale_text import fit_snippet, MARKER_JS, PRIMING_ACK
from tree_sitter import Parser, Language
from tree_sitter_javascript import language as js_language
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import textwrap


# ---- Tree-sitter setup ------------------------------------------------------


LanguageT = object
ParserT = object


# python -m pip uninstall -y tree-sitter-languages tree-sitter tree-sitter-javascript
# python -m pip install -U "tree-sitter==0.21.0" "tree-sitter-javascript==0.21.0"


def _load_js_language_and_parser() -> Tuple[Language, Parser]:
    """
    Load the JavaScript language and parser.

    This function initialises the JavaScript language and parser, handling different API versions.

    Returns:
        A tuple containing the loaded `Language` instance and the initialised `Parser` instance.
    """

    ptr_or_lang: Any = js_language()

    # Wrap capsule -> Language, or accept Language directly.
    if isinstance(ptr_or_lang, Language):
        lang = ptr_or_lang
    else:
        try:
            lang = Language(ptr_or_lang)                 # new API (tree-sitter >= 0.22)
        except TypeError:
            lang = Language(ptr_or_lang, "JavaScript")   # old 0.21 API (second arg is a label)

    # Create a parser using whichever API this wheel exposes
    try:
        p = Parser()
        p.set_language(lang)              # classic API
    except AttributeError:
        # Fallback: some builds use Parser(Language) ctor
        p = Parser(lang)

    return lang, p


JS_LANGUAGE, JS_PARSER = _load_js_language_and_parser()


def _parse_js(source_blob: str) -> Tuple[Any, bytes]:
    """
    Normalise line endings and parse JavaScript source once.

    Tree-sitter counts rows by '\\n' only, whereas the source may use '\\r' or '\\r\\n'. Line endings are normalised to
    '\\n' so that node rows line up with the caller's line-split source. The same normalised bytes are returned and used
    for any node-text extraction, keeping byte offsets self-consistent with the parse tree.

    Parameters:
    - `source_blob`: The JavaScript source code as a string.

    Returns:
    - A tuple of the parsed tree and the normalised source bytes it was parsed from.
    """

    norm = source_blob.replace("\r\n", "\n").replace("\r", "\n")
    source_bytes = norm.encode("utf-8", errors="replace")
    return JS_PARSER.parse(source_bytes), source_bytes


# ---- DefInfo for JavaScript --------------------------------------------------


@dataclass(frozen=True)
class DefInfoJS:
    """
    Represents information about a JavaScript definition-like construct.

    Attributes:
        qualname (str): The qualified name of the definition, e.g. "foo", "ClassName.method", "obj.method".
        node (object): The Tree-sitter node representing the definition.
        kind (str): The type of definition, e.g. "function", "class", "method", "var_func", "var_arrow", "obj_method".
        start (int): The 1-based line number where the definition starts.
        end (int): The 1-based line number where the definition ends (inclusive).
        header_start (int): The 1-based line number where the definition's header starts (same as `start`).
        header_end (int): The 1-based line number where the definition's header ends (line before body block begins).
        depth (int): The nesting depth of the definition, with 0 indicating a module-level definition.
        parent_id (Optional[int]): The ID of the parent node, or `None` if this is a top-level definition.
        children_ids (Tuple[int, ...]): A tuple of IDs of child nodes.

    Note:
        This class abstracts over various types of JavaScript definitions, including functions, classes, methods, and variables.
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


# ---- General utilities ------------------------------------------------------


def _to_1based(row0: int) -> int:
    """
    Convert a 0-based row index to a 1-based index.

    Parameters:
    - `row0`: The 0-based row index to convert.

    Returns:
    - The corresponding 1-based row index.
    """

    return row0 + 1


def _line_span_from_node(n) -> Tuple[int, int]:
    """
    Convert a Tree-sitter node to its corresponding line span.

    Returns:
    - A tuple of two integers representing the 1-based line numbers for the start and end points of the node.
    """

    return _to_1based(_row_of(n.start_point)), _to_1based(_row_of(n.end_point))


def _node_field(n, field: str):
    """
    Get the child node of a Tree-sitter node by its field name.

    Parameters:
    - `n`: The parent Tree-sitter node.
    - `field`: The name of the field to retrieve.

    Returns:
    - The child node with the specified field name, or `None` if not found.
    """

    return n.child_by_field_name(field)


def _node_text(source_bytes: bytes, n) -> str:
    """
    Extract the text content of a Tree-sitter node from the source code bytes.

    This function returns a string representation of the text within the specified node, replacing any invalid UTF-8
    sequences with a replacement character.

    Parameters:
    - `source_bytes`: The raw bytes of the source code.
    - `n`: The Tree-sitter node to extract the text from.

    Returns:
    - A string containing the text content of the node, or an error message if decoding fails.
    """

    return source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")


def _string_value(s: str) -> str:
    """
    Best-effort unquote of a JS string literal's contents.

    Leaves the original text if it is not a simple quoted literal.

    Parameters:
    - `s`: The input string to be unquoted.

    Returns:
    - The unquoted string, or the original string if it is not a simple quoted literal.
    """

    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"', "`"):
        return s[1:-1]
    return s


def _get_text_for_lines(source_lines: Chunk, a: int, b: int) -> str:
    """
    Extract a contiguous range of lines from the source code.

    This function extracts a range of lines from the input `source_lines` chunk, starting from line `a`
    (inclusive) and ending at line `b` (inclusive). The range is clipped to ensure that `a` is not less
    than 1 and `b` does not exceed the total number of lines in the chunk.

    Parameters:
    - `source_lines`: The input chunk containing the source code.
    - `a`: The starting line index (inclusive).
    - `b`: The ending line index (inclusive).

    Returns:
    - A string representation of the specified lines, or an empty string if the range is invalid.
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
    - An integer representing the number of leading spaces.
    """

    return len(line) - len(line.lstrip(" "))


def _row_of(point) -> int:
    """
    Return the 0-based row from a tree-sitter point.

    This function supports both object and tuple representations of points. For objects with a `.row` attribute,
    it directly returns the row value. For tuples, it extracts the first element (the row) and returns it.

    Parameters:
    - `point`: The tree-sitter point from which to extract the row.

    Returns:
    - The 0-based row of the point as an integer.
    """
    try:
        return point.row            # object with .row
    except AttributeError:
        return point[0]             # tuple (row, column)


def _col_of(point) -> int:
    """
    Return the 0-based column from a tree-sitter point, tolerating both the object (`.column`) and tuple (`[1]`) forms.

    Parameters:
    - `point`: The tree-sitter point from which to extract the column.

    Returns:
    - The 0-based column of the point as an integer.
    """
    try:
        return point.column         # object with .column
    except AttributeError:
        return point[1]             # tuple (row, column)


# ---- Import discovery (ESM + CommonJS) -------------------------------------


def _collect_imports_js(tree, source_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Collect a plain-English list of import/require statements in source order.

    This function supports ES module imports and CommonJS require(...) calls.
    It returns a list of tuples, where each tuple contains the line number and a description of the import/require statement.

    Parameters:
    - `tree`: The parsed Tree-sitter tree for the source.
    - `source_bytes`: The (normalised) source bytes the tree was parsed from.

    Returns:
    - A list of tuples, where each tuple contains the line number and a description of the import/require statement.
    """
    out: List[Tuple[int, str]] = []
    src_b = source_bytes
    root = tree.root_node

    def add(n, text: str) -> None:
        """
        Add a line of text to the output.

        Parameters:
        - `n`: The node to determine the line number from.
        - `text`: The text to append to the output.

        Returns:
        - None
        """

        ln = _to_1based(_row_of(n.start_point))
        out.append((ln, text))

    # Walk everything; imports can appear anywhere at top-level in ESM, and
    # require(...) can appear nested.
    stack = [root]
    while stack:
        n = stack.pop()
        t = n.type

        # ---- ES module: import ----
        # Tree-sitter JavaScript/TypeScript use 'import_statement'.
        # Be tolerant and also accept 'import_declaration' in case of grammar variants.
        if t in ("import_statement", "import_declaration"):
            src_node = _node_field(n, "source")
            if src_node is None:
                # Fallback: some versions expose the string as the last named child
                cand = n.named_children[-1] if n.named_child_count else None
                src_node = cand if cand and cand.type == "string" else None
            module_text = _string_value(_node_text(src_b, src_node)) if src_node else "<unknown>"

            clause = _node_field(n, "import_clause")
            if clause is None:
                # import 'mod';
                add(n, f"- Imports {module_text} for side effects")
            else:
                # default import: identifier directly under clause
                for c in (clause.named_children or []):
                    if c.type in ("identifier",):
                        local = _node_text(src_b, c)
                        add(n, f"- Imports default from {module_text} as {local}")
                    elif c.type == "namespace_import":
                        # import * as ns from 'mod'
                        name = _node_field(c, "name")
                        local = _node_text(src_b, name) if name else "<namespace>"
                        add(n, f"- Imports everything from {module_text} as {local}")
                    elif c.type == "named_imports":
                        # import { a, b as c } from 'mod'
                        for spec in (c.named_children or []):
                            if spec.type != "import_specifier":
                                continue
                            orig = _node_field(spec, "name") or _node_field(spec, "imported_name")
                            alias = _node_field(spec, "alias")
                            otext = _node_text(src_b, orig) if orig else "<unknown>"
                            if alias:
                                atext = _node_text(src_b, alias)
                                add(n, f"- Imports {otext} from {module_text} as {atext}")
                            else:
                                add(n, f"- Imports {otext} from {module_text}")
                # fall through: do not descend into children of import declarations further
            continue

        # ---- CommonJS: const X = require('mod') / const {a: b} = require('mod') ----
        if t == "variable_declarator":
            init = _node_field(n, "value")
            callee = _node_field(init, "function") if init and init.type == "call_expression" else None
            if callee and _node_text(src_b, callee) == "require":
                # Extract module name from the first argument if it is a string literal
                mod = "<unknown>"
                args = _node_field(init, "arguments")
                if args and args.named_child_count >= 1:
                    arg0 = args.named_child(0)
                    if arg0 and arg0.type == "string":
                        mod = _string_value(_node_text(src_b, arg0))

                # Binding side: identifier or object_pattern
                name_node = _node_field(n, "name")
                if name_node is not None and name_node.type == "identifier":
                    local = _node_text(src_b, name_node)
                    add(n, f"- Requires {mod} as {local}")
                elif name_node is not None and name_node.type == "object_pattern":
                    # const {a, b: c} = require('mod')
                    props: List[str] = []
                    for k in (name_node.named_children or []):
                        if k.type == "pair":
                            k_name = _node_field(k, "key")
                            k_alias = _node_field(k, "value")
                            oname = _node_text(src_b, k_name) if k_name else "<key>"
                            aname = _node_text(src_b, k_alias) if k_alias else oname
                            props.append(f"{oname} as {aname}" if aname != oname else oname)
                        elif k.type in ("identifier", "shorthand_property_identifier_pattern"):
                            props.append(_node_text(src_b, k))
                    if props:
                        add(n, f"- Requires {mod} (destructured: {', '.join(props)})")
                    else:
                        add(n, f"- Requires {mod}")
                else:
                    add(n, f"- Requires {mod}")
                # do not descend further into this declarator
                continue

        # ---- CommonJS: bare require('mod') ----
        if t == "call_expression":
            callee = _node_field(n, "function")
            if callee and _node_text(src_b, callee) == "require":
                args = _node_field(n, "arguments")
                mod = "<unknown>"
                if args and args.named_child_count >= 1:
                    arg0 = args.named_child(0)
                    if arg0 and arg0.type == "string":
                        mod = _string_value(_node_text(src_b, arg0))
                add(n, f"- Requires {mod} for side effects")
                # still descend, but not necessary; continue to keep traversal simple

        # generic DFS
        for i in range(n.named_child_count - 1, -1, -1):
            stack.append(n.named_child(i))

    out.sort(key=lambda t: t[0])
    return out


def describe_imports_js(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    tree,
    source_bytes: bytes
) -> None:
    """
    Build a list of imports/requires from the JavaScript source code and feed it to the LLM as extra context.

    This function mirrors the Python flow by echoing the imports, pushing a short 'OK' acknowledgement prompt,
    and then generating LLM output with the provided context.

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The GenerationConfig instance.
    - `messages`: The Messages instance.
    - `tree`: The parsed Tree-sitter tree for the source.
    - `source_bytes`: The (normalised) source bytes the tree was parsed from.

    Notes:
    - If no imports are found, this function returns immediately.
    - The LLM output is echoed and appended to the messages list.
    """

    items = _collect_imports_js(tree, source_bytes)
    if not items:
        return

    lines = [text for _, text in items]
    payload = "\n".join(lines)

    echo(f"\n[JS] Imports...\n{payload}")
    prompt = (
        "For additional context, here is a list of imports within this program:\n\n"
        f"{payload}"
    )
    messages.append({"role": "user", "content": prompt})
    # A fixed acknowledgement we supply ourselves (not a model-generated "OK") - saves a generation call and avoids
    # conditioning a small model to answer the first real request with "OK" too (see PRIMING_ACK).
    messages.append({"role": "assistant", "content": PRIMING_ACK})


# ---- Qualname extraction ----------------------------------------------------


def _ident_text(source_bytes: bytes, n) -> Optional[str]:
    """
    Extract the text of an identifier or string node from the source code.

    This function takes a bytes representation of the source code and a node to extract the text from.
    It returns the extracted text as a string, or `None` if the node is not an identifier or string.

    Parameters:
    - `source_bytes`: The bytes representation of the source code.
    - `n`: The node to extract the text from.

    Returns:
    - The extracted text as a string, or `None` if the node is not an identifier or string.
    """

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


def _qual_name(source_bytes: bytes, n, scope: List[str]) -> str:
    """
    Build a fully qualified name for any declaration-like node that exposes a 'name' field (functions,
    classes, methods, variable declarators).

    The identifier is read from the node's 'name' field; if absent, '<anonymous>' is used. The resulting
    name is prefixed by the dot-joined scope if provided.

    Parameters:
    - source_bytes: Source text as bytes (used by _ident_text).
    - n: Tree-sitter node for the declaration.
    - scope: Enclosing scope components, outermost to innermost.

    Returns:
    - Qualified name string, e.g. 'Outer.Inner.func' or '<anonymous>'.
    """

    name = _ident_text(source_bytes, _node_field(n, "name")) or "<anonymous>"
    return ".".join(scope + [name]) if scope else name


def _qual_name_for_obj_method(source_bytes: bytes, pair_or_method, scope: List[str], enclosing_obj: Optional[str]) -> str:
    """
    Construct a qualified name for an object method.

    This function generates a string representing the fully qualified name of an object method.
    It takes into account whether the method is enclosed within another object and constructs
    the name accordingly.

    Parameters:
    - `source_bytes`: The source code bytes containing the method definition.
    - `pair_or_method`: The object literal method or pair with a function value.
    - `scope`: A list of strings representing the current scope.
    - `enclosing_obj`: The name of the enclosing object, if any.

    Returns:
    - A string representing the fully qualified name of the object method.
    """

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
    Determine the line number of the end of a function-like header.

    For statement blocks, the header ends on the line before the block's start.
    For arrow functions with expression body, the header is on the function line.
    For method definitions, if the value has a statement block, the header ends on the line before the block's start.
    Otherwise, the header ends on the line of the method definition.

    Parameters:
    - `n`: The Tree-sitter node representing the function-like construct.

    Returns:
    - The 1-based line number of the end of the function-like header.
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
    """
    Yield the named children of a node.

    This generator iterates over the named child nodes of the given node, yielding each child in turn.

    Parameters:
    - `n`: The node to iterate over its named children.

    Yields:
    - Each named child node of the given node.
    """

    for i in range(n.named_child_count):
        yield n.named_child(i)


def iter_defs_with_info_js(tree, source_bytes: bytes) -> List[DefInfoJS]:
    """
    Collect definition-like constructs from the given JavaScript source code.

    This function walks a parsed JS module and identifies functions, classes, methods, variables, and object methods.
    It creates DefInfoJS instances for each definition, which contain information such as the qualified name, type,
    start and end lines, header start and end lines, depth, parent ID, and children IDs.

    Parameters:
    - `tree`: The parsed Tree-sitter tree for the source.
    - `source_bytes`: The (normalised) source bytes the tree was parsed from.

    Returns:
    - A sorted list of DefInfoJS instances representing the definition-like constructs in the source code.
    """

    root = tree.root_node

    results: List[DefInfoJS] = []
    children_map: Dict[int, List[int]] = {}
    scope_names: List[str] = []
    scope_nodes: List[object] = []

    def add_child(parent_node: Optional[object], child_node: object) -> None:
        """
        Add a child node to the parent's children map.

        Parameters:
        - `parent_node`: The parent node to add the child to. If `None`, do nothing.
        - `child_node`: The child node to be added.

        Returns:
        - None

        Notes:
        This method updates the internal children map, which is used to keep track of node relationships.
        The `children_map` dictionary is keyed by parent node IDs and contains lists of child node IDs.
        """

        if parent_node is None:
            return
        children_map.setdefault(id(parent_node), []).append(id(child_node))

    def walk(node) -> None:
        """
        Collects information about definition-like constructs in the JavaScript source code.

        This function recursively traverses the abstract syntax tree (AST), identifying and processing functions, classes,
        methods, variables, and object methods. It creates DefInfoJS instances for each definition, which contain
        information such as the qualified name, type, start and end lines, header start and end lines, depth, parent ID,
        and children IDs.

        Parameters:
        - `node`: The current node in the AST being processed.

        Notes:
        - This function uses a recursive approach to traverse the AST.
        - It relies on other functions (e.g. `_qual_name`, `_header_end_for_function_like`) to extract relevant information
          from the node and its context.
        """

        t = node.type

        def mk_info(qualname: str, kind: str) -> DefInfoJS:
            """
            Create a DefInfoJS instance for the given definition.

            Parameters:
            - `qualname`: The qualified name of the definition.
            - `kind`: The type of definition (e.g. function, class, variable).

            Returns:
            - A DefInfoJS instance containing information about the definition, including its node, kind, start and end lines,
              header start and end lines, depth, parent ID, and children IDs.

            Notes:
            - The `start_1b` and `end_1b` values are obtained from the `_line_span_from_node` function.
            - The `header_end` value is obtained from the `_header_end_for_function_like` function.
            - The `depth` value represents the nesting level of the definition within the current scope.
            - The `parent_id` value is the ID of the parent node in the scope, or `None` if the definition is at the top level.
            """

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
            qual = _qual_name(source_bytes, node, scope_names)
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
            qual = _qual_name(source_bytes, node, scope_names)
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
            qual = _qual_name(source_bytes, node, scope_names)
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
                qual = _qual_name(source_bytes, node, scope_names)
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
                qual = _qual_name_for_obj_method(source_bytes, node, scope_names, enclosing)
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


def _collect_calls_js(node, source_bytes: bytes, def_node_ids: set) -> List[Tuple[str, str]]:
    """
    Collect a JS routine's own call sites for the call-graph resolver, without descending into nested routines.

    The walk starts at the routine node and descends through everything *except* the subtrees of other recognised
    definitions (any node whose id is in `def_node_ids`, except this routine's own node) - so a nested function/class/
    function-valued binding is opaque and resolved as its own routine. This id-set approach is precise where a
    type-based skip would not be: a `const f = () => {...}` is recorded as a `variable_declarator`, so its arrow value
    is *not* a separate def node and is correctly walked. Each `call_expression` is classified by its callee:
      - `f(...)` -> `("f", "free", line)`;
      - `this.m(...)` -> `("m", "self", line)`;
      - any other `obj.m(...)` -> `("m", "method", line)`.
    The `line` is the call's 1-based source line - resolution ignores it; the block pass's read-side call annotations
    use it.

    Parameters:
    - `node`: The routine's tree-sitter node.
    - `source_bytes`: The source bytes the tree was parsed from.
    - `def_node_ids`: The ids of every routine node in the file (used to treat nested routines as opaque).

    Returns:
    - The call sites as `(name, kind, line)` triples.
    """

    root_id = node.id
    calls: List[Tuple[str, str, int]] = []

    def visit(n) -> None:
        """Record `n` if it is a call site, then descend - skipping the subtree of any *other* recognised definition."""
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            line = n.start_point[0] + 1
            if fn is not None:
                if fn.type == "identifier":
                    calls.append((_node_text(source_bytes, fn), "free", line))
                elif fn.type == "member_expression":
                    obj = fn.child_by_field_name("object")
                    prop = fn.child_by_field_name("property")
                    if prop is not None:
                        name = _node_text(source_bytes, prop)
                        if obj is not None and obj.type == "this":
                            calls.append((name, "self", line))
                        else:
                            calls.append((name, "method", line))
        for i in range(n.named_child_count):
            child = n.named_child(i)
            if child.id in def_node_ids and child.id != root_id:
                continue   # a nested routine: opaque, resolved on its own
            visit(child)

    visit(node)
    return calls


def iter_symbols(source_blob: str, source_lines: Chunk) -> List[Symbol]:
    """
    Extract the call-graph `Symbol` for every routine in a JS file (the model-free pre-pass per-file step).

    Reuses `iter_defs_with_info_js` for the definition records, then attaches each routine's parent qualname, header
    signature, the existing doc comment above it (the seed contract, via `_doc_above_header_js`) and its classified
    call sites (`_collect_calls_js`, which treats nested routines as opaque). Returns `[]` if the file cannot be parsed.

    Parameters:
    - `source_blob`: The complete source text.
    - `source_lines`: The same source split into lines.

    Returns:
    - One `Symbol` per definition (the `file` field is left blank for `build_project_graph` to stamp).
    """

    try:
        tree, source_bytes = _parse_js(source_blob)
    except Exception:
        return []

    defs = iter_defs_with_info_js(tree, source_bytes)
    # `iter_defs_with_info_js` keys parent/child relationships by Python `id()` of the node objects it captured; reuse
    # that for parent lookup. For the call walk (a fresh descent that yields new node wrappers) use the tree-stable
    # `node.id` instead, since Python `id()` is not stable across separate node accesses.
    qual_by_pyid: Dict[int, str] = {id(d.node): d.qualname for d in defs}
    def_node_ids = {d.node.id for d in defs}

    symbols: List[Symbol] = []
    for d in defs:
        signature = "\n".join(source_lines[d.header_start - 1:d.header_end]) if d.header_end >= d.header_start else ""
        parent_qualname = qual_by_pyid.get(d.parent_id) if d.parent_id is not None else None
        symbols.append(Symbol(
            qualname=d.qualname, kind=d.kind, signature=signature, start=d.start, end=d.end, depth=d.depth,
            parent_qualname=parent_qualname, existing_doc=_doc_above_header_js(source_lines, d.header_start),
            calls=_collect_calls_js(d.node, source_bytes, def_node_ids),
        ))
    return symbols


# ---- Within-function block targets (the `-b` block pass) --------------------


# Statements that open a brace block whose source-line span gates a paragraph break before/after it. Nested
# definitions also open a block (a substantial nested def earns a break before it, and one after it - see the
# after-def rule), but they are opaque: the walk records them as a single boundary and does not descend.
_JS_BLOCK_OPENERS = {"if_statement", "for_statement", "for_in_statement", "while_statement", "do_statement",
                     "try_statement", "switch_statement", "statement_block"}
_JS_FUNCTION_VALUES = {"arrow_function", "function", "function_expression", "generator_function"}


def _named_children(n) -> List:
    """Return the named children of a tree-sitter node as a list."""
    return [n.named_child(i) for i in range(n.named_child_count)]


def _is_js_statement(n) -> bool:
    """Report whether a node is a JS statement (`*_statement`, `*_declaration`, or a bare `{ }` block)."""
    t = n.type
    return t.endswith("_statement") or t.endswith("_declaration") or t == "statement_block"


def _js_is_def_stmt(node) -> bool:
    """
    Report whether a body statement introduces a nested definition (so it is opaque to the parent's block pass).

    Covers `function`/`class` declarations and a `const`/`let`/`var` whose initialiser is a function or arrow - the
    same constructs `iter_defs_with_info_js` treats as their own routines. Such a statement is one boundary and is
    not descended into; its body is annotated when it is processed as its own target.

    Parameters:
    - `node`: The statement node to test.

    Returns:
    - True if the statement is or introduces a nested definition.
    """

    if node.type in ("function_declaration", "generator_function_declaration", "class_declaration"):
        return True
    if node.type in ("lexical_declaration", "variable_declaration"):
        for d in _named_children(node):
            if d.type == "variable_declarator":
                val = _node_field(d, "value")
                if val is not None and val.type in _JS_FUNCTION_VALUES:
                    return True
    return False


def _js_suite(node) -> Optional[List]:
    """
    Return the statement list held by a body/branch node, or None.

    Unwraps an `else_clause` to the branch inside it; expands a `statement_block` to its statement children; and
    treats a single unbraced statement (e.g. `if (x) return;`) as a one-element suite.

    Parameters:
    - `node`: A body/branch node (or None).

    Returns:
    - The statement list, or None if the node holds no statements.
    """

    if node is None:
        return None
    if node.type == "else_clause":
        kids = _named_children(node)
        return _js_suite(kids[0]) if kids else None
    if node.type == "statement_block":
        stmts = [c for c in _named_children(node) if _is_js_statement(c)]
        return stmts or None
    if _is_js_statement(node):
        return [node]
    return None


def _js_statement_lists(stmt) -> List[List]:
    """
    Return the statement lists nested directly inside a JS compound statement (the analogue of Python suites).

    Covers an `if`'s consequence/alternative, a loop body, a `try`'s block / catch / finally, each `case`/`default`
    of a `switch`, and the contents of a bare `{ }` block. Nested definitions are handled by the caller (opaque), so
    they are not expanded here.

    Parameters:
    - `stmt`: The statement to inspect.

    Returns:
    - A list of statement lists (empty for a simple statement).
    """

    t = stmt.type
    lists: List[List] = []
    if t == "if_statement":
        for fld in ("consequence", "alternative"):
            sl = _js_suite(_node_field(stmt, fld))
            if sl:
                lists.append(sl)
    elif t in ("for_statement", "for_in_statement", "while_statement", "do_statement"):
        sl = _js_suite(_node_field(stmt, "body"))
        if sl:
            lists.append(sl)
    elif t == "try_statement":
        for fld in ("body", "handler", "finalizer"):
            node = _node_field(stmt, fld)
            if node is None:
                continue
            if node.type in ("catch_clause", "finally_clause"):
                node = _node_field(node, "body")
            sl = _js_suite(node)
            if sl:
                lists.append(sl)
    elif t == "switch_statement":
        body = _node_field(stmt, "body")
        if body is not None:
            for case in _named_children(body):
                if case.type in ("switch_case", "switch_default"):
                    cl = [c for c in _named_children(case) if _is_js_statement(c)]
                    if cl:
                        lists.append(cl)
    elif t == "statement_block":
        cl = [c for c in _named_children(stmt) if _is_js_statement(c)]
        if cl:
            lists.append(cl)
    return lists


def _is_js_decl(node) -> bool:
    """Report whether a node is a plain variable declaration (a `let`/`const`/`var` that is not a function binding)."""
    return node.type in ("lexical_declaration", "variable_declaration") and not _js_is_def_stmt(node)


def _js_span(node) -> int:
    """Source-line span of a node (its triviality measure for the block-size gate)."""
    return _row_of(node.end_point) - _row_of(node.start_point) + 1


def _js_opens_block(node) -> int:
    """Return the block span this statement opens (control block or nested def), or 0 if it opens none."""
    if node.type in _JS_BLOCK_OPENERS or _js_is_def_stmt(node):
        return _js_span(node)
    return 0


def _js_closed_block(prev_node, resume_start: int, body_node) -> Optional[object]:
    """Return the outermost brace block that closed between `prev_node` and a dedent resuming at `resume_start`."""
    best = None
    n = prev_node.parent
    while n is not None and n is not body_node:
        if (n.type in _JS_BLOCK_OPENERS or _js_is_def_stmt(n)) and _to_1based(_row_of(n.end_point)) < resume_start:
            best = n
        n = n.parent
    return best


def _js_body_block(node):
    """Return the `statement_block` body of a function-like def node (or None for an expression-bodied arrow)."""
    b = _node_field(node, "body")
    if b is not None and b.type == "statement_block":
        return b
    val = _node_field(node, "value")
    if val is not None:
        b = _node_field(val, "body")
        if b is not None and b.type == "statement_block":
            return b
    return None


def _collect_body_js(body_node, source_lines: Chunk) -> Tuple[List[int], Dict[int, str], List[SegStatement]]:
    """
    Walk a JS function body, returning its legal block boundaries, their indentation, and the segmenter records.

    Mirrors `scale_python._body_boundaries` + `_seg_records`: every statement is recorded at every nesting depth
    (recursing into nested brace blocks but treating a nested definition as one opaque boundary), the first
    statement of an inner suite is excluded from the boundaries, and a line is a boundary only if it begins exactly
    one (non-suite-leading) statement at its first non-blank column - which drops `a; b;`, continuation lines, and
    inline inner statements.

    Parameters:
    - `body_node`: The function's `statement_block` body node.
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
        """Record each statement and recurse into its nested suites; nested definitions are opaque (not descended)."""
        # A suite that is exactly `[simple_stmt, return]` is one paragraph: anchor it at the leading statement so the
        # comment pass sees both lines (a return alone gives it nothing to describe).
        if (len(stmts) == 2 and stmts[1].type == "return_statement"
                and stmts[0].type not in _JS_BLOCK_OPENERS and not _js_is_def_stmt(stmts[0])):
            a = _to_1based(_row_of(stmts[0].start_point))
            r = _to_1based(_row_of(stmts[1].start_point))
            if a != r:
                merge_map[r] = a
        # Leading-declaration heuristic: a scope opening with a run of `let`/`const`/`var` declarations gets its first
        # real statement paragraphed off, so the declarations read as their own block.
        ndecl = 0
        while ndecl < len(stmts) and _is_js_decl(stmts[ndecl]):
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
            if _js_is_def_stmt(stmt):
                continue
            for sub in _js_statement_lists(stmt):
                walk(sub, False, depth + 1)

    top_stmts = [c for c in _named_children(body_node) if _is_js_statement(c)]
    walk(top_stmts, True, 0)

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

    recorded.sort(key=lambda r: r[0])
    seg_statements: List[SegStatement] = []
    for i, (start, depth, node, first_in_scope) in enumerate(recorded):
        end = _to_1based(_row_of(node.end_point))
        closed = 0
        if i > 0 and recorded[i - 1][1] > depth:
            blk = _js_closed_block(recorded[i - 1][2], start, body_node)
            closed = _js_span(blk) if blk is not None else 0
        seg_statements.append(SegStatement(
            start=start, end=end, depth=depth,
            is_return=node.type == "return_statement", is_def=_js_is_def_stmt(node),
            opens_block=_js_opens_block(node), first_in_scope=first_in_scope, closed_block=closed,
            merge_anchor=merge_map.get(start) if node.type == "return_statement" else None,
            force_break=start in force_break_lines,
        ))

    return boundary_lines, indent_of, seg_statements


def _doc_above_header_js(source_lines: Chunk, header_start: int) -> str:
    """
    Return the documentation text of the comment block immediately above a definition header (or "" if none).

    The JS analogue of Python's `ast.get_docstring`: the block provider parses the (possibly def-pass-annotated)
    source, so the JSDoc / `//` block the definition pass wrote above the routine is read back here and fed to the
    block-comment pass as the routine's purpose - giving JS the same per-routine context Python gets from a docstring.

    Parameters:
    - `source_lines`: The source split into lines (the same text the provider parses).
    - `header_start`: The 1-based line of the definition header.

    Returns:
    - The doc text with comment delimiters/gutters stripped, or "" when no comment block sits above the header.
    """

    rng = _scan_existing_comment_block_above(source_lines, header_start)
    if rng is None:
        return ""
    s, e = rng
    block = "\n".join(source_lines[s - 1:e])
    if "/*" in block:
        return _extract_first_comment_block(block)
    # A `//` run: strip the leading slashes from each line.
    return "\n".join(ln.strip()[2:].strip() if ln.strip().startswith("//") else ln.strip()
                     for ln in block.split("\n")).strip()


def file_doc_target_js(source_blob: str, source_lines: Chunk) -> Optional[FileDocTarget]:
    """
    Build the file-level header doccomment target for a JavaScript source file.

    JS file headers are the same shape as C's - a leading run of `/* ... */` and/or `//` comment blocks (after an
    optional `#!/usr/bin/env node` shebang) before the first code - so this delegates to the shared brace-language
    scanner. The description is rendered/updated as a `/* ... */` block.

    Parameters:
    - `source_blob`: The complete source text (unused; accepted for provider-signature symmetry).
    - `source_lines`: The source split into individual lines.

    Returns:
    - A `FileDocTarget`, or None if the file is empty.
    """

    return scan_brace_leading_zone(source_lines, SLASH_BLOCK_STYLE)


# ---- Cognitive complexity (the escalation routing signal) -------------------

# Constructs that score +1 plus the nesting penalty and deepen nesting for their contents (SonarSource B1+B2 rules).
# `if_statement` and `try_statement` are handled separately (chain folding / flat catch-finally continuations).
_JS_NESTING_TYPES = {"for_statement", "for_in_statement", "while_statement", "do_statement", "switch_statement"}

# Nested definitions are opaque: each is scored separately when processed as its own routine.
_JS_OPAQUE_TYPES = _JS_FUNCTION_VALUES | {
    "function_declaration", "generator_function_declaration", "method_definition", "class_declaration", "class",
}

# Logical operators: one increment per operator *sequence* (`a && b && c` scores once, mirroring Python's `BoolOp`).
_JS_LOGICAL_OPS = {"&&", "||"}


def cognitive_complexity_js(node) -> int:
    """
    Compute the SonarSource-style Cognitive Complexity of a single JS routine's own body.

    The tree-sitter mirror of `scale_python.cognitive_complexity` (same rules, so one `--escalate-cognitive` cutoff
    is meaningful across languages):

      - +1 (and +1 per enclosing nesting level) for each `if` / `for` / `for-in`/`for-of` / `while` / `do` /
        `switch` / `try` and ternary (`ternary_expression`);
      - +1 (with no nesting penalty) for each `else if` / `else` / `catch` / `finally` continuation - an `else if`
        ladder is folded iteratively so it reads as cheap continuations rather than ever-deeper nested `if`s;
      - +1 for each `&&` / `||` operator sequence (a run of the same operator scores once, like Python's `BoolOp`).

    Nested functions, arrows, methods and classes are opaque - the walk does not descend into them - so each routine
    is scored on its own control flow. The node may be any shape `iter_defs_with_info_js` records (a declaration, a
    `method_definition`, or a `variable_declarator`/`pair` whose value is a function/arrow): the function value is
    unwrapped first, and an expression-bodied arrow (`x => cond ? a : b`) scores its expression.

    Parameters:
    - `node`: The routine's def node (any `DefInfoJS.node` shape) whose body is scored.

    Returns:
    - The cognitive complexity as a non-negative integer.
    """

    if node is None:
        return 0

    # Unwrap a variable_declarator / object pair to the function value it binds; a class node scores 0 itself
    # (its methods are their own targets).
    fn = node
    if fn.type not in _JS_OPAQUE_TYPES:
        val = _node_field(fn, "value")
        if val is not None and val.type in _JS_FUNCTION_VALUES:
            fn = val
    body = _node_field(fn, "body")
    if body is None:
        return 0

    score = 0

    def visit_if(n, nesting: int) -> None:
        """Score an `if_statement`, folding its `else if` chain into +1 continuations."""
        nonlocal score
        score += 1 + nesting
        visit(_node_field(n, "condition"), nesting, None)
        visit(_node_field(n, "consequence"), nesting + 1, None)
        alt = _node_field(n, "alternative")
        while alt is not None:
            # The grammar wraps the branch in an `else_clause`; unwrap to the statement inside.
            inner = alt
            if alt.type == "else_clause":
                kids = _named_children(alt)
                inner = kids[0] if kids else None
            if inner is None:
                return
            if inner.type == "if_statement":  # `else if`: a continuation, not a fresh nested if
                score += 1
                visit(_node_field(inner, "condition"), nesting, None)
                visit(_node_field(inner, "consequence"), nesting + 1, None)
                alt = _node_field(inner, "alternative")
            else:  # plain `else`
                score += 1
                visit(inner, nesting + 1, None)
                alt = None

    def visit(n, nesting: int, logical_op) -> None:
        """Walk `n` adding to `score`; `logical_op` is the operator of an enclosing logical run (for collapsing)."""
        nonlocal score
        if n is None or n.type in _JS_OPAQUE_TYPES:
            return

        if n.type == "if_statement":
            visit_if(n, nesting)
            return

        if n.type in _JS_NESTING_TYPES:
            score += 1 + nesting
            body_node = _node_field(n, "body")
            for c in _named_children(n):
                # The header parts (initialiser/condition/update) stay at this level; the body deepens.
                visit(c, nesting + 1 if (body_node is None or c.id == body_node.id) else nesting, None)
            return

        if n.type == "try_statement":  # try +1+nesting; each catch/finally is a flat +1 continuation
            score += 1 + nesting
            visit(_node_field(n, "body"), nesting + 1, None)
            handler = _node_field(n, "handler")
            if handler is not None:
                score += 1
                visit(_node_field(handler, "body"), nesting + 1, None)
            finalizer = _node_field(n, "finalizer")
            if finalizer is not None:
                score += 1
                visit(_node_field(finalizer, "body"), nesting + 1, None)
            return

        if n.type == "ternary_expression":  # `cond ? a : b`
            score += 1 + nesting
            visit(_node_field(n, "condition"), nesting, None)
            visit(_node_field(n, "consequence"), nesting + 1, None)
            visit(_node_field(n, "alternative"), nesting + 1, None)
            return

        if n.type == "binary_expression":
            op_node = _node_field(n, "operator")
            op = op_node.text.decode("utf-8", "replace") if op_node is not None else ""
            if op in _JS_LOGICAL_OPS:
                if op != logical_op:  # a run of the same operator scores once
                    score += 1
                for c in _named_children(n):
                    visit(c, nesting, op)
                return

        for c in _named_children(n):
            visit(c, nesting, None)

    # An expression-bodied arrow has no statement_block; score the expression itself.
    visit(body, 0, None)
    return score


def iter_block_targets_js(source_blob: str, source_lines: Chunk) -> List[BlockTarget]:
    """
    Build the within-function block targets for a JavaScript source file.

    Each function, method, and function/arrow-valued binding with a brace body becomes one `BlockTarget` carrying
    its header/body line spans, the lines that may legally begin a block, and the deterministic structural
    segmentation. Class bodies (which hold member definitions, not statements) are skipped - their methods are
    targeted individually. Nested definitions are opaque within a parent and annotated as their own targets, so JS
    keeps the after-def paragraph rule (unlike C). This is the JS implementation of the provider interface consumed
    by `scale_blocks.annotate_blocks`.

    Parameters:
    - `source_blob`: The complete source text (parsed with Tree-sitter JavaScript).
    - `source_lines`: The same source split into individual lines.

    Returns:
    - A list of `BlockTarget`, one per routine body, in source order.
    """

    tree, source_bytes = _parse_js(source_blob)
    targets: List[BlockTarget] = []

    for info in iter_defs_with_info_js(tree, source_bytes):
        body_node = _js_body_block(info.node)
        if body_node is None:
            continue

        boundary_lines, indent_of, seg_statements = _collect_body_js(body_node, source_lines)
        top_stmts = [c for c in _named_children(body_node) if _is_js_statement(c)]
        if not top_stmts:
            continue
        body_start = _to_1based(_row_of(top_stmts[0].start_point))
        body_end = _to_1based(_row_of(body_node.end_point))

        segments = structural_breaks(
            seg_statements, has_doc=False, boundary_lines=tuple(boundary_lines), body_end=body_end,
            allow_after_def=True, allow_first_in_scope=False,
        )
        targets.append(
            BlockTarget(
                qualname=info.qualname,
                kind=info.kind,
                header_start=info.header_start,
                header_end=info.header_end,
                body_start=body_start,
                body_end=body_end,
                boundary_lines=tuple(boundary_lines),
                indent_of=indent_of,
                depth=info.depth,
                doc=_doc_above_header_js(source_lines, info.header_start),
                cognitive=cognitive_complexity_js(info.node),
                segments=segments,
            )
        )

    return targets


# ---- Depth order -------------------------------------------------------------


def deepest_first_js(defs: List[DefInfoJS]) -> List[DefInfoJS]:
    """
    Sort the definitions in a deepest-first order.

    This function takes a list of `DefInfoJS` instances and returns them sorted by depth, start line, and end line.
    The sorting is stable, meaning that equal elements maintain their original order. The sort key is defined as:

    - Depth (deepest first)
    - Start line (earlier lines come first)
    - End line (later lines come first)

    Parameters:
    - `defs`: A list of `DefInfoJS` instances to be sorted.

    Returns:
    - A sorted list of `DefInfoJS` instances.
    """

    return sorted(defs, key=lambda d: (d.depth, d.start, -d.end), reverse=True)


# ---- Snippet assembly (stubbing direct children) ----------------------------


def _render_jsdoc_block(text: str, base_indent: str) -> List[str]:
    """
    Render a JSDoc comment block with the given text and base indentation.

    This function formats a string of text as a JSDoc comment block, including a leading `/**` marker
    and a trailing `*/` terminator. If the input text is empty, it displays a message indicating that
    there is no documentation.

    Parameters:
    - `text`: The input text to be formatted as a JSDoc comment block.
    - `base_indent`: The base indentation string for the comment block.

    Returns:
    - A list of strings representing the formatted JSDoc comment block.
    """

    lines = text.splitlines()
    out = [f"{base_indent}/**"]
    if lines:
        for ln in lines:
            # rstrip the whole line: a blank doc line would otherwise leave " * " with a trailing space.
            out.append(f"{base_indent} * {ln.rstrip()}".rstrip())
    else:
        out.append(f"{base_indent} * (no documentation)")
    out.append(f"{base_indent} */")
    return out


def _render_js_line_comment(text: str, base_indent: str) -> List[str]:
    """
    Render a documentation comment as `//` line comments (the line-comment counterpart to `_render_jsdoc_block`).

    Used when a file's existing doc comments are `//`-style, so generated docs match the file's convention instead of
    always introducing a `/** ... */` JSDoc block.

    Parameters:
    - `text`: The comment text (possibly multi-line).
    - `base_indent`: The base indentation for the comment lines.

    Returns:
    - A list of `//`-prefixed comment lines.
    """

    lines = text.splitlines()
    if not lines:
        return [f"{base_indent}// (no documentation)"]
    # rstrip the whole line so a blank doc line is "//" rather than "// " with a trailing space.
    return [f"{base_indent}// {ln.rstrip()}".rstrip() for ln in lines]


def _detect_doc_style_js(tree, source_bytes: bytes) -> str:
    """
    Detect whether a JS file documents its code with `//` line comments or `/* ... */` (JSDoc) block comments.

    Only the leading file-header banner (the first top-level comment) is ignored, so every other comment is weighed
    (including the first definition's own doc). Following "if both styles are present, prefer the block form", the
    result is `"line"` only when the remaining comments use `//` and no `/* ... */`, otherwise `"block"` (the default,
    including a file with no other comments).

    Parameters:
    - `tree`: The parsed Tree-sitter tree.
    - `source_bytes`: The source bytes the tree was parsed from.

    Returns:
    - `"line"` or `"block"`.
    """

    root = tree.root_node
    banner_start = (root.children[0].start_byte
                    if root.children and root.children[0].type == "comment" else None)

    has_line = has_block = False
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "comment" and n.start_byte != banner_start:
            if source_bytes[n.start_byte:n.start_byte + 2] == b"//":
                has_line = True
            else:
                has_block = True
        for i in range(n.named_child_count):
            stack.append(n.named_child(i))

    return "line" if (has_line and not has_block) else "block"


def _strip_jsdoc_gutters(block: List[str]) -> str:
    """
    Strip the leading JSDoc gutter from each line of a comment block and join the result.

    Only a single leading '*' (and one following space) is removed per line, so genuine content such as a
    '* bullet' is preserved rather than eaten.

    Parameters:
    - `block`: The comment-body lines (without the surrounding '/**' and '*/' fences).

    Returns:
    - The cleaned comment text as a single, stripped string.
    """

    cleaned: List[str] = []
    for ln in block:
        s = ln.lstrip()
        if s.startswith("*"):
            s = s[1:]
            if s.startswith(" "):
                s = s[1:]
        cleaned.append(s.rstrip())
    return "\n".join(cleaned).strip()


def _extract_first_comment_block(reply: str) -> str:
    """
    Extract the documentation body from an LLM reply, tolerant of how the model wrapped it.

    The model does not always honour the request for a bare JSDoc block, so several shapes are accepted, in order
    of preference:

    1. A JSDoc `/** ... */` block (the requested form).
    2. A fenced code block ``` ... ``` (optionally tagged, e.g. ```js), using its contents.
    3. As a last resort, the whole reply treated as the comment body.

    Parameters:
    - `reply`: The raw text returned by the LLM.

    Returns:
    - The extracted comment text, or an empty string only if the reply is effectively empty.
    """

    lines = textwrap.dedent(reply).split("\n")
    stripped = [ln.strip() for ln in lines]

    # 1. Preferred: a JSDoc '/** ... */' block.
    try:
        start_idx = stripped.index("/**") + 1
        for end_idx in range(start_idx, len(lines)):
            if "*/" in lines[end_idx]:
                break
        else:
            end_idx = len(lines)
        return _strip_jsdoc_gutters(lines[start_idx:end_idx])
    except ValueError:
        pass

    # 2. Fallback: a fenced code block ``` ... ``` (the tag line, if any, is dropped).
    fence_idxs = [i for i, s in enumerate(stripped) if s.startswith("```")]
    if len(fence_idxs) >= 2:
        return _strip_jsdoc_gutters(lines[fence_idxs[0] + 1:fence_idxs[1]])

    # 3. Last resort: treat the whole reply as the comment body.
    return _strip_jsdoc_gutters(lines)


def assemble_snippet_for_js(
    info_by_id: Dict[int, DefInfoJS],
    source_lines: Chunk,
    node_id: int,
    docs_by_id: Dict[int, str]
) -> str:
    """
    Assemble a snippet of JavaScript code for a given node ID.

    This function generates a snippet by splicing together the header and body of the source code.
    The header is obtained from the `source_lines` chunk, spanning from `header_start` to `header_end`.
    The body is constructed by replacing direct children with their corresponding JSDoc comments and headers,
    and preserving other lines. The replacement is done in a deepest-first order to ensure that parents see
    child stubs.

    Parameters:
    - `info_by_id`: A dictionary mapping node IDs to `DefInfoJS` objects.
    - `source_lines`: A chunk of source code lines.
    - `node_id`: The ID of the node for which to assemble the snippet.
    - `docs_by_id`: A dictionary mapping node IDs to their corresponding JSDoc comments.

    Returns:
    - A string representing the assembled snippet of JavaScript code.
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
    source_lines: Chunk,
    doc_order: Optional[List[str]] = None,
    callee_context: Optional[Callable[[str], str]] = None,
    on_doc: Optional[Callable[[str, str], None]] = None,
    verifier=None,
) -> Dict[int, str]:
    """
    Generate JSDoc content for JS definitions in a deepest-first order, ensuring parents see child stubs.

    This function orchestrates the generation of JSDoc comments for JavaScript definitions by:

    1.  Assembling snippets of code for each definition.
    2.  Prompting the LLM to generate documentation comments for these snippets.
    3.  Processing the LLM output to extract the first comment block.
    4.  Storing the generated comments in dictionaries keyed by qualification name and node ID.

    Parameters:
        - `llm`: The LocalChatModel instance used to interact with the LLM.
        - `cfg`: The GenerationConfig instance containing configuration settings.
        - `messages`: A list of messages exchanged with the LLM.
        - `defs`: A list of DefInfoJS instances representing JavaScript definitions.
        - `source_blob`: The source code as a string.
        - `source_lines`: A Chunk object containing the source code lines.

    Returns:
        - A dictionary mapping qualification names to generated JSDoc comments.
    """

    docs_by_node_id: Dict[int, str] = {}
    info_by_id: Dict[int, DefInfoJS] = {id(d.node): d for d in defs}

    # Default order is deepest-first (children before parents, so a parent sees its child stubs). A call-graph
    # `doc_order` overrides it (callees/children first), still keeping children ahead of parents via the fallback key.
    if doc_order:
        ordered = apply_doc_order(defs, lambda d: d.qualname, doc_order, lambda d: (-d.depth, d.start, -d.end))
    else:
        ordered = deepest_first_js(defs)

    for info in ordered:
        node_id = id(info.node)
        snippet = assemble_snippet_for_js(info_by_id, source_lines, node_id, docs_by_node_id)

        # Elide the body if this routine is too large for the context window (the patch is unaffected).
        header_lines = max(1, info.header_end - info.header_start + 1)
        snippet, omitted = fit_snippet(llm, cfg, messages, snippet, header_lines, MARKER_JS)
        if omitted:
            echo(f"[JS] Elided {omitted} body line(s) from '{info.qualname}' to fit the context window")

        echo("\n[JS] Snippet...\n")
        echo(snippet)

        if info.kind == "class":
            prompt = (
                "Write exactly the documentation comment for this JavaScript class. "
                "Use the nested method documentation to inform your description, but remember the class abstracts "
                "over all of them: summarise the class as a whole rather than repeating any single method. "
                "Return only the comment content (no code), suitable for a JSDoc block placed immediately above the class:\n\n"
                f"{snippet}\n"
            )
        else:
            prompt = (
                "Write exactly the documentation comment for this JavaScript program chunk. "
                "Return only the comment content (no code), suitable for a JSDoc block placed "
                "immediately above the definition:\n\n"
                f"{snippet}\n"
            )

        # Ingest-and-update: a JS doc comment sits ABOVE the header, outside the snippet, so surface the routine's own
        # existing doc as a seed - the model keeps what is still accurate and updates what is stale, rather than
        # re-deriving the contract blind (the routine-level analogue of the --file-doc description seed).
        existing = _doc_above_header_js(source_lines, info.header_start)
        if existing:
            prompt += (
                "\nThe routine is already documented as follows. Ingest and update this existing comment - keep "
                "whatever is still accurate, correct anything stale, and reformat it to the requested style - rather "
                f"than writing from scratch:\n\n{existing}\n"
            )

        # Inject the one-line contracts of the routines this one calls (call-graph context).
        if callee_context is not None:
            notes = callee_context(info.qualname)
            if notes:
                prompt += "\n" + notes + "\n"

        messages.append({"role": "user", "content": prompt})
        reply = llm.generate(messages, cfg=cfg)
        echo(f"\n[JS] LLM output:\n\n{reply}")
        messages.pop()

        doc = _extract_first_comment_block(reply)
        if doc and verifier is not None:
            # Verification (the quality floor): the grounding gate + grounding challenge, each allowing one
            # corrective regeneration in this routine's own context (snippet -> previous answer -> feedback).
            last_reply = [reply]

            def regenerate(feedback: str, _prompt: str = prompt) -> str:
                """Regenerate the comment with reviewer feedback; '' when the retry is unusable."""
                messages.append({"role": "user", "content": _prompt})
                messages.append({"role": "assistant", "content": last_reply[0]})
                messages.append({"role": "user", "content": feedback})
                retry = llm.generate(messages, cfg=cfg)
                for _ in range(3):
                    messages.pop()
                d = _extract_first_comment_block(retry)
                if not d:
                    return ""
                last_reply[0] = retry
                return d

            doc, ok = verifier.verify_def(snippet, doc, regenerate, label=info.qualname)
            if not ok:
                error(f"[verify] '{info.qualname}': comment failed verification twice; writing it anyway - "
                      f"review this comment")
        if not doc:
            doc = f"{info.kind} `{info.qualname}` - documentation generation failed."
        docs_by_node_id[node_id] = doc
        # Publish this routine's freshly-generated contract for later callers.
        if on_doc is not None:
            on_doc(info.qualname, doc)

    return docs_by_node_id


# ---- Textual patcher (insert/replace above headers) -------------------------


def _scan_existing_comment_block_above(source_lines: Chunk, header_start_line_1b: int) -> Optional[Tuple[int, int]]:
    """
    Detect an existing comment block immediately above the header.

    This function scans the source code lines above the specified header start line to identify either:
    - a block comment '/* ... */' whose last line ends just above the header, or
    - a contiguous run of '//' lines immediately above the header (no blank line).

    Returns a tuple containing the start and end line numbers of the existing comment block, or `None` if no such block is found.
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


def patch_comments_textually_js(source_lines: Chunk, defs: List[DefInfoJS], doc_map: Dict[int, str],
                                style: str = "block") -> Chunk:
    """
    Insert or replace documentation blocks for each JS definition.

    This function iterates over the definitions in a deepest-first order and inserts or replaces
    documentation blocks above their corresponding headers. If an existing block comment or contiguous
    '//' block is found immediately above the header, it is replaced with a new JSDoc block. Otherwise,
    a new JSDoc block is inserted immediately above the header.

    Parameters:
    - `source_lines`: The input source code as a list of lines.
    - `defs`: A list of `DefInfoJS` objects representing the definitions to be processed.
    - `doc_map`: A dictionary mapping definition names to their corresponding documentation strings.
    - `style`: `"block"` to render each doc as a `/** ... */` JSDoc block (default) or `"line"` to render it as `//`
      line comments (chosen to match the file's prevailing doc-comment convention; see `_detect_doc_style_js`).

    Returns:
    - The modified source code with inserted or replaced documentation blocks.
    """

    render = _render_js_line_comment if style == "line" else _render_jsdoc_block
    out_lines = source_lines[:]

    for info in sorted(defs, key=lambda d: d.start, reverse=True):
        node_id = id(info.node)
        if node_id not in doc_map:
            continue
        doc = doc_map[node_id]

        header_line_text = source_lines[info.header_start - 1]
        indent = header_line_text[: len(header_line_text) - len(header_line_text.lstrip())]

        new_block_lines = render(doc, indent)

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
    source_lines: Chunk,
    doc_order: Optional[List[str]] = None,
    callee_context: Optional[Callable[[str], str]] = None,
    on_doc: Optional[Callable[[str, str], None]] = None,
    verifier=None,
) -> Chunk:
    """
    Generate JSDoc comments for a JavaScript source code chunk.

    This function orchestrates the end-to-end process of generating JSDoc comments for a given JavaScript source code:

    1. Parse the JavaScript source using Tree-sitter.
    2. Collect information about definition-like constructs (e.g., functions, classes, methods, variables) in the source code.
    3. Generate JSDoc comments for each definition using a large language model (LLM), following a deepest-first order to ensure
       that parent definitions see child stubs.
    4. Textually patch the source code by inserting or replacing existing comment blocks above headers with the generated JSDoc comments.

    Parameters:
    - `llm`: The local chat model used for generating JSDoc comments.
    - `cfg`: The generation configuration.
    - `messages`: The messages object.
    - `source_blob`: The JavaScript source code as a string.
    - `source_lines`: The source code chunk.
    - `doc_order`/`callee_context`/`on_doc`: Optional call-graph hooks (see `generate_comments_js`); absent, behaviour
      is unchanged.

    Returns:
    - The updated source code chunk with generated JSDoc comments.
    """

    echo("Parsing JavaScript source with Tree-sitter...")
    tree, source_bytes = _parse_js(source_blob)
    defs = iter_defs_with_info_js(tree, source_bytes)
    echo(f"Found {len(defs)} JS definitions")

    # Match the file's prevailing doc-comment convention (a file whose docs are `//` should not be given `/** */`
    # JSDoc blocks); the leading banner is ignored, and a mix prefers the block form.
    style = _detect_doc_style_js(tree, source_bytes)
    echo(f"Doc-comment style for this file: {style}")

    # Provide a list of imports/requires to the LLM (if there are any)
    echo("Identifying imports/requires...")
    describe_imports_js(llm, cfg, messages, tree, source_bytes)

    echo("Generating JSDoc comments...\n")
    doc_map = generate_comments_js(llm, cfg, messages, defs, source_blob, source_lines,
                                   doc_order=doc_order, callee_context=callee_context, on_doc=on_doc,
                                   verifier=verifier)

    echo("Applying JS patches...\n")
    return patch_comments_textually_js(source_lines, defs, doc_map, style=style)
