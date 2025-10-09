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


# ---- Import discovery (ESM + CommonJS) -------------------------------------


def _collect_imports_js(source_blob: str) -> List[Tuple[int, str]]:
    """
    Collect a plain-English list of import/require statements in source order.

    This function supports ES module imports and CommonJS require(...) calls.
    It returns a list of tuples, where each tuple contains the line number and a description of the import/require statement.

    Parameters:
    - `source_blob`: The JavaScript source code as a string.

    Returns:
    - A list of tuples, where each tuple contains the line number and a description of the import/require statement.
    """
    out: List[Tuple[int, str]] = []
    src_b = source_blob.encode("utf-8", errors="replace")
    tree = JS_PARSER.parse(src_b)
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
                # Extract module name from first argument if string literal
                args = [init.child(i) for i in range(init.named_child_count)] if init else []
                mod = None
                for a in args:
                    if a.type == "arguments" and a.named_child_count >= 1:
                        arg0 = a.named_child(0)
                        if arg0 and arg0.type == "string":
                            mod = _string_value(_node_text(src_b, arg0))
                mod = mod or "<unknown>"

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
    source_blob: str
) -> None:
    """
    Build a list of imports/requires from the JavaScript source code and feed it to the LLM as extra context.

    This function mirrors the Python flow by echoing the imports, pushing a short 'OK' acknowledgement prompt,
    and then generating LLM output with the provided context.

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The GenerationConfig instance.
    - `messages`: The Messages instance.
    - `source_blob`: The JavaScript source code as a string.

    Notes:
    - If no imports are found, this function returns immediately.
    - The LLM output is echoed and appended to the messages list.
    """

    items = _collect_imports_js(source_blob)
    if not items:
        return

    lines = [text for _, text in items]
    payload = "\n".join(lines)

    echo(f"\n[JS] Imports...\n{payload}")
    prompt = (
        "For additional context, here is a list of imports within this program:\n\n"
        f"{payload}\n\n"
        "Please respond by saying 'OK'. No other commentary is required at this time."
    )
    messages.append({"role": "user", "content": prompt})

    reply = llm.generate(messages, cfg=cfg)
    echo(f"\n[JS] LLM output:\n\n{reply}")
    messages.append({"role": "assistant", "content": reply})


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


def iter_defs_with_info_js(source_blob: str) -> List[DefInfoJS]:
    """
    Collect definition-like constructs from the given JavaScript source code.

    This function parses the JS module and identifies functions, classes, methods, variables, and object methods.
    It creates DefInfoJS instances for each definition, which contain information such as the qualified name, type,
    start and end lines, header start and end lines, depth, parent ID, and children IDs.

    Parameters:
    - `source_blob`: The JavaScript source code to parse.

    Returns:
    - A sorted list of DefInfoJS instances representing the definition-like constructs in the source code.
    """

    source_bytes = source_blob.encode("utf-8", errors="replace")
    tree = JS_PARSER.parse(source_bytes)
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
            out.append(f"{base_indent} * {ln.rstrip()}")
    else:
        out.append(f"{base_indent} * (no documentation)")
    out.append(f"{base_indent} */")
    return out


def _extract_first_comment_block(reply: str) -> str:
    """
    Extract the first JSDoc-style comment block from a reply string.

    This function removes any leading whitespace and dedents the input string, then searches for the first occurrence of
    a JSDoc-style comment block (starting with `/**`). It extracts the comment block up to the matching `*/`, removes any
    leading asterisks from each line, and returns the resulting comment block as a single string.

    Parameters:
    - `reply`: The input string containing the reply.

    Returns:
    - The extracted comment block, or an empty string if no JSDoc-style comment block is found.
    """

    lines = textwrap.dedent(reply).split("\n")
    stripped = [ln.strip() for ln in lines]
    start_idx = None
    try:
        start_idx = stripped.index("/**") + 1
    except ValueError:
        return ""

    for end_idx in range(start_idx, len(lines)):
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
    source_lines: Chunk
) -> Dict[str, str]:
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


def patch_comments_textually_js(source_lines: Chunk, defs: List[DefInfoJS], doc_map: Dict[str, str]) -> Chunk:
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

    Returns:
    - The modified source code with inserted or replaced documentation blocks.
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

    Returns:
    - The updated source code chunk with generated JSDoc comments.
    """

    echo("Parsing JavaScript source with Tree-sitter...")
    defs = iter_defs_with_info_js(source_blob)
    echo(f"Found {len(defs)} JS definitions")

    # Provide a list of imports/requires to the LLM (if there are any)
    echo("Identifying imports/requires...")
    describe_imports_js(llm, cfg, messages, source_blob)

    echo("Generating JSDoc comments...\n")
    doc_map = generate_comments_js(llm, cfg, messages, defs, source_blob, source_lines)

    echo("Applying JS patches...\n")
    return patch_comments_textually_js(source_lines, defs, doc_map)
