#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The JavaScript language worker: it parses source with tree-sitter (shimmed over both the 0.21 and 0.22 binding APIs),
discovers every function, class and method - including arrow and function expressions bound to variables and methods
inside object literals - and supplies the per-language pieces the shared pipeline needs: the definition pass, the
block-pass segmenter, the symbol scanner, and the file-doc target.

The definition pass works deepest-first: each routine is shown to the model with its direct children collapsed to
their already-generated JSDoc blocks, so a parent's prompt is informed by child documentation without re-sending child
code. Imports (ES and CommonJS alike) are described into the chat context first, existing comments are ingested and
updated rather than discarded, and `patch_comments_textually_js` splices the results in bottom-up, matching the file's
`//` or JSDoc style.

For the online mode, `collect_def_requests_js` records each routine in the run manifest keyed by a comment-stripped
span hash (`_js_span_hash`), so `apply_manifest_js` can re-bind answers to code unchanged since emit; every edit must
pass the preservation guard before it is kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from scale_blocks import BlockTarget, SegStatement, structural_breaks, SEG_MIN_LEADING_DECLS, SLASH_BLOCK_STYLE
from scale_escalate import routine_text_hash
from scale_filedoc import FileDocTarget, scan_brace_leading_zone
from scale_llm import LocalChatModel, GenerationConfig, Messages, Chunk
from scale_log import echo, error
from scale_project import Symbol, apply_doc_order
from scale_text import fit_snippet, MARKER_JS, PRIMING_ACK
from tree_sitter import Parser, Language
from tree_sitter_javascript import language as js_language
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import re
import textwrap


# ---- Tree-sitter setup ------------------------------------------------------


LanguageT = object
ParserT = object


# python -m pip uninstall -y tree-sitter-languages tree-sitter tree-sitter-javascript
# python -m pip install -U "tree-sitter==0.21.0" "tree-sitter-javascript==0.21.0"


def _load_js_language_and_parser() -> Tuple[Language, Parser]:
    """
    Load the tree-sitter JavaScript grammar and a parser bound to it, tolerating both old and new binding APIs.

    The `tree-sitter` package changed both the `Language` and `Parser` construction APIs between 0.21 and 0.22; this shim tries the new form first and falls back to the old one, so either installed version works.

    Returns:
    - A `(Language, Parser)` tuple ready to parse JavaScript source.
    """

    # The grammar binding hands back either a ready Language or a raw pointer, depending on the installed tree-sitter version.
    ptr_or_lang: Any = js_language()

    # Wrap a raw pointer with whichever Language constructor this version accepts: the one-arg form (>= 0.22) first, then the old 0.21 two-arg form.
    if isinstance(ptr_or_lang, Language):
        lang = ptr_or_lang
    else:
        try:
            lang = Language(ptr_or_lang)                 # new API (tree-sitter >= 0.22)
        except TypeError:
            lang = Language(ptr_or_lang, "JavaScript")   # old 0.21 API (second arg is a label)

    # Parser binding changed across versions too: prefer the classic set_language call, falling back to passing the language at construction.
    try:
        p = Parser()
        p.set_language(lang)              # classic API
    except AttributeError:
        p = Parser(lang)

    return lang, p


JS_LANGUAGE, JS_PARSER = _load_js_language_and_parser()


def _parse_js(source_blob: str) -> Tuple[Any, bytes]:
    """
    Parse JavaScript source with tree-sitter and return the tree alongside the exact bytes that were parsed.

    Line endings are normalised to LF before encoding, so node rows and byte offsets always index into the returned bytes rather than the caller's original text.

    Parameters:
    - `source_blob`: The JavaScript source text.

    Returns:
    - A `(tree, source_bytes)` tuple; node offsets refer to `source_bytes`.
    """

    # Fold CRLF/CR to plain LF before encoding so node coordinates map cleanly onto line numbers.
    norm = source_blob.replace("\r\n", "\n").replace("\r", "\n")
    source_bytes = norm.encode("utf-8", errors="replace")

    return JS_PARSER.parse(source_bytes), source_bytes


# ---- DefInfo for JavaScript --------------------------------------------------


@dataclass(frozen=True)
class DefInfoJS:

    """
    One JavaScript definition (function, method or class) discovered in the parse tree.

    Line numbers are 1-based. `parent_id` and `children_ids` link nested definitions by index into the flat discovery list, so nesting can be walked without revisiting tree-sitter nodes.
    """

    # `start`/`end` cover the whole definition while the header pair spans just the signature line(s).
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
    Convert a 0-based tree-sitter row index to a 1-based source line number.
    """

    return row0 + 1


def _line_span_from_node(n) -> Tuple[int, int]:
    """
    Return a tree-sitter node's extent as a 1-based inclusive (start, end) line span.
    """

    return _to_1based(_row_of(n.start_point)), _to_1based(_row_of(n.end_point))


def _node_field(n, field: str):
    """
    Fetch a tree-sitter node's child by field name; `None` when the field is absent.
    """

    return n.child_by_field_name(field)


def _node_text(source_bytes: bytes, n) -> str:
    """
    Decode the slice of source bytes spanned by a tree-sitter node, replacing any undecodable bytes.
    """

    return source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")


def _string_value(s: str) -> str:
    """
    Strip one pair of matching quotes from a string literal's text.

    Parameters:
    - `s`: The literal text, possibly still wrapped in single, double or backtick quotes.

    Returns:
    - The unquoted contents, or the trimmed input unchanged if it is not quote-wrapped.
    """

    # Only a matching outer quote pair (single, double or backtick) is removed; escapes inside are left untouched.
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"', "`"):
        return s[1:-1]
    return s


def _get_text_for_lines(source_lines: Chunk, a: int, b: int) -> str:
    """
    Return the source text spanned by an inclusive, 1-based line range.
    """

    a = max(1, a)
    b = min(len(source_lines), b)
    if a > b:
        return ""
    return "\n".join(source_lines[a - 1:b])


def _leading_spaces_count(line: str) -> int:
    """
    Count the leading space characters at the start of a line.
    """

    return len(line) - len(line.lstrip(" "))


def _row_of(point) -> int:
    """
    Return the row of a tree-sitter point, whichever shape the binding uses.

    Parameters:
    - `point`: Either a Point-like object exposing `.row` or a plain `(row, column)` tuple.

    Returns:
    - The zero-based row as an `int`.
    """

    # Bridges tree-sitter binding versions: older ones return plain (row, column) tuples instead of Point objects.
    try:
        return point.row            # object with .row
    except AttributeError:
        return point[0]             # tuple (row, column)


def _col_of(point) -> int:
    """
    Return the column of a tree-sitter point, whichever shape the binding uses.

    Parameters:
    - `point`: Either a Point-like object exposing `.column` or a plain `(row, column)` tuple.

    Returns:
    - The zero-based column as an `int`.
    """

    # Same binding-version bridge as the row accessor: the point may be an object or a plain tuple.
    try:
        return point.column         # object with .column
    except AttributeError:
        return point[1]             # tuple (row, column)


# ---- Import discovery (ESM + CommonJS) -------------------------------------


def _collect_imports_js(tree, source_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Collect a one-line description of every import in a JavaScript syntax tree.

    An iterative walk recognises three shapes: ES `import` statements (side-effect, default, namespace and named clauses, including aliases), CommonJS `require(...)` declarators (identifier or destructuring bindings), and bare `require(...)` calls made purely for side effects. Module paths that are not string literals are reported as `<unknown>`.

    Parameters:
    - `tree`: The parsed tree-sitter syntax tree for the file.
    - `source_bytes`: The raw source bytes the tree was parsed from.

    Returns:
    - A list of `(line, description)` tuples sorted by 1-based source line.
    """

    out: List[Tuple[int, str]] = []
    src_b = source_bytes
    root = tree.root_node

    # Findings carry their 1-based source line so the final report can be sorted into file order.
    def add(n, text: str) -> None:
        """
        Record one import description against its 1-based source line.

        Parameters:
        - `n`: The tree-sitter node the description was derived from.
        - `text`: The ready-formatted description line.
        """

        # Tree-sitter rows are zero-based; the report wants 1-based line numbers.
        ln = _to_1based(_row_of(n.start_point))
        out.append((ln, text))

    # Iterative stack walk of the whole tree, so deep sources cannot hit the recursion limit.
    stack = [root]

    while stack:
        n = stack.pop()
        t = n.type

        # ES-module `import` statements: identify the module string, then describe each binding in the clause.
        if t in ("import_statement", "import_declaration"):
            src_node = _node_field(n, "source")

            # Some grammar versions expose no `source` field; fall back to a trailing string child.
            if src_node is None:
                cand = n.named_children[-1] if n.named_child_count else None
                src_node = cand if cand and cand.type == "string" else None

            module_text = _string_value(_node_text(src_b, src_node)) if src_node else "<unknown>"
            clause = _node_field(n, "import_clause")

            # One report line per binding: side-effect-only, default, namespace and each named specifier.
            if clause is None:
                add(n, f"- Imports {module_text} for side effects")
            else:
                for c in (clause.named_children or []):
                    if c.type in ("identifier",):
                        local = _node_text(src_b, c)
                        add(n, f"- Imports default from {module_text} as {local}")
                    elif c.type == "namespace_import":
                        name = _node_field(c, "name")
                        local = _node_text(src_b, name) if name else "<namespace>"
                        add(n, f"- Imports everything from {module_text} as {local}")
                    elif c.type == "named_imports":
                        for spec in (c.named_children or []):
                            if spec.type != "import_specifier":
                                continue
                            orig = _node_field(spec, "name") or _node_field(spec, "imported_name")
                            alias = _node_field(spec, "alias")
                            otext = _node_text(src_b, orig) if orig else "<unknown>"

                            # Aliased specifiers record both the exported and local names.
                            if alias:
                                atext = _node_text(src_b, alias)
                                add(n, f"- Imports {otext} from {module_text} as {atext}")
                            else:
                                add(n, f"- Imports {otext} from {module_text}")

            # Nothing of further interest below an import node, so its children are skipped.
            continue

        # CommonJS bindings: declarators whose initialiser is a `require(...)` call.
        if t == "variable_declarator":
            init = _node_field(n, "value")
            callee = _node_field(init, "function") if init and init.type == "call_expression" else None

            # The callee is matched by literal text, so only direct `require` calls are recognised.
            if callee and _node_text(src_b, callee) == "require":
                mod = "<unknown>"
                args = _node_field(init, "arguments")

                # Only a literal string argument yields a module name; dynamic paths stay `<unknown>`.
                if args and args.named_child_count >= 1:
                    arg0 = args.named_child(0)
                    if arg0 and arg0.type == "string":
                        mod = _string_value(_node_text(src_b, arg0))

                name_node = _node_field(n, "name")

                # A plain identifier binds the whole module; an object pattern is unpacked below.
                if name_node is not None and name_node.type == "identifier":
                    local = _node_text(src_b, name_node)
                    add(n, f"- Requires {mod} as {local}")
                elif name_node is not None and name_node.type == "object_pattern":
                    props: List[str] = []

                    # Both `{a: b}` renaming pairs and shorthand `{a}` properties feed the destructured-name list.
                    for k in (name_node.named_children or []):
                        if k.type == "pair":
                            k_name = _node_field(k, "key")
                            k_alias = _node_field(k, "value")
                            oname = _node_text(src_b, k_name) if k_name else "<key>"
                            aname = _node_text(src_b, k_alias) if k_alias else oname
                            props.append(f"{oname} as {aname}" if aname != oname else oname)
                        elif k.type in ("identifier", "shorthand_property_identifier_pattern"):
                            props.append(_node_text(src_b, k))

                    # A pattern with no readable names still earns a bare module line.
                    if props:
                        add(n, f"- Requires {mod} (destructured: {', '.join(props)})")
                    else:
                        add(n, f"- Requires {mod}")
                else:
                    add(n, f"- Requires {mod}")

                # Handled, so do not descend and re-report the inner call as a side-effect require.
                continue

        # Bare `require(...)` calls reached outside a declarator count as side-effect imports.
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

        # Children are pushed in reverse so the stack pops them in source order.
        for i in range(n.named_child_count - 1, -1, -1):
            stack.append(n.named_child(i))

    # Sorting by line keeps the report in file order whatever the traversal did.
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
    Prime the conversation with a summary of the file's imports.

    Collects every ES and CommonJS import and, when any exist, appends the list as a user turn followed by a canned assistant acknowledgement; no model call is made. Files with no imports leave the conversation untouched.

    Parameters:
    - `llm`: The local chat model (unused; present for the shared provider signature).
    - `cfg`: The generation configuration (unused; present for the shared provider signature).
    - `messages`: The running conversation the priming turns are appended to.
    - `tree`: The parsed tree-sitter syntax tree for the file.
    - `source_bytes`: The raw source bytes the tree was parsed from.
    """

    # Priming only: the list is appended with a canned acknowledgement, so no model call is made here.
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
    messages.append({"role": "assistant", "content": PRIMING_ACK})


# ---- Qualname extraction ----------------------------------------------------


def _ident_text(source_bytes: bytes, n) -> Optional[str]:
    """
    Return the textual name carried by an identifier-like node, if any.

    Accepts plain, property and private (`#name`) identifiers, plus string nodes, whose surrounding quotes or backticks are stripped. Safe with `None`.

    Parameters:
    - `source_bytes`: The raw source bytes the node was parsed from.
    - `n`: The tree-sitter node to read, or `None`.

    Returns:
    - The name as a string, or `None` when the node carries no usable name.
    """

    if n is None:
        return None
    if n.type in ("identifier", "property_identifier"):
        return _node_text(source_bytes, n)
    if n.type == "private_property_identifier":  # #method
        return _node_text(source_bytes, n)

    # String keys count as names too: strip a matching quote or backtick pair when present.
    if n.type == "string":
        s = _node_text(source_bytes, n).strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"', "`"):
            return s[1:-1]
        return s

    return None


def _qual_name(source_bytes: bytes, n, scope: List[str]) -> str:
    """
    Build the dot-separated qualified name for a definition node.

    Parameters:
    - `source_bytes`: The raw source bytes the tree was parsed from.
    - `n`: The tree-sitter node whose `name` field supplies the leaf identifier.
    - `scope`: Enclosing scope names, outermost first.

    Returns:
    - The dotted qualified name; the leaf is `<anonymous>` when the node carries no name.
    """

    name = _ident_text(source_bytes, _node_field(n, "name")) or "<anonymous>"
    return ".".join(scope + [name]) if scope else name


def _qual_name_for_obj_method(source_bytes: bytes, pair_or_method, scope: List[str], enclosing_obj: Optional[str]) -> str:
    """
    Build the qualified name for a method defined inside an object literal.

    Parameters:
    - `source_bytes`: The raw source bytes the tree was parsed from.
    - `pair_or_method`: The object `pair` or `method_definition` node naming the method.
    - `scope`: Enclosing scope names, outermost first.
    - `enclosing_obj`: Name of the variable holding the object literal, or `None` when unknown.

    Returns:
    - The dotted qualified name, including the owning object's name when available.
    """

    # Object pairs name the method under `key`; method definitions carry it under `name`.
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
    Return the 1-based line on which a function-like node's header ends.

    The header runs up to, but not including, the line where the statement or class body opens; nodes without a recognisable body report their own first line.

    Parameters:
    - `n`: The tree-sitter function, class, or method node.

    Returns:
    - The 1-based line number of the last header line.
    """

    # The header ends on the line just before the statement or class body opens.
    body = _node_field(n, "body") if n.type != "method_definition" else _node_field(n, "body")
    if body and body.type in ("statement_block", "class_body"):
        return _to_1based(_row_of(body.start_point)) - 1

    # Method bodies hang off the nested `value` function node rather than the method node itself.
    if n.type == "method_definition":
        val = _node_field(n, "value")

        if val:
            blk = _node_field(val, "body")
            if blk and blk.type == "statement_block":
                return _to_1based(_row_of(blk.start_point)) - 1

    # No recognisable body, so the header collapses to the node's own first line.
    return _to_1based(_row_of(n.start_point))


# ---- AST walk and DefInfo collection ---------------------------------------


def _iter_children(n) -> Iterable:
    """
    Yield the named children of a tree-sitter node, skipping anonymous tokens.

    Parameters:
    - `n`: The tree-sitter node to iterate.

    Returns:
    - An iterable over the node's named child nodes.
    """

    for i in range(n.named_child_count):
        yield n.named_child(i)


def iter_defs_with_info_js(tree, source_bytes: bytes) -> List[DefInfoJS]:
    """
    Collect every function, class, and method definition in a JavaScript parse tree.

    Recognised forms include declarations, class methods, function and arrow expressions bound to variables, and methods inside object literals. Each record carries the qualified name, line span, header extent, nesting depth, and parent/child links.

    Parameters:
    - `tree`: The tree-sitter parse tree for the file.
    - `source_bytes`: The raw source bytes the tree was parsed from.

    Returns:
    - A list of `DefInfoJS` records sorted by start line.
    """

    # Scope stacks plus an id()-keyed child map let the walk build qualified names and parent/child links as it goes.
    root = tree.root_node
    results: List[DefInfoJS] = []
    children_map: Dict[int, List[int]] = {}
    scope_names: List[str] = []
    scope_nodes: List[object] = []

    def add_child(parent_node: Optional[object], child_node: object) -> None:
        """
        Record a parent-to-child link in the shared children map.

        Parameters:
        - `parent_node`: The enclosing definition node, or `None` for top-level definitions (in which case nothing is recorded).
        - `child_node`: The definition node to register under the parent.
        """

        if parent_node is None:
            return
        children_map.setdefault(id(parent_node), []).append(id(child_node))

    def walk(node) -> None:
        """
        Recursively visit a node and record every definition form it matches.

        Function and class declarations push a naming scope before descending; methods, variable-bound functions, and object-literal methods record themselves without opening one. Unmatched nodes are simply descended into.

        Parameters:
        - `node`: The tree-sitter node to visit.
        """

        t = node.type

        def mk_info(qualname: str, kind: str) -> DefInfoJS:
            """
            Build a `DefInfoJS` record for the node currently being visited.

            Captures the line span, header extent, and the depth and identity of the enclosing scope at the time of the call; `children_ids` is filled in after the walk completes.

            Parameters:
            - `qualname`: The dotted qualified name for the definition.
            - `kind`: The definition kind label, e.g. `function`, `class`, or `method`.

            Returns:
            - A `DefInfoJS` describing the node.
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

        # Function declarations open a naming scope so nested definitions are qualified beneath them.
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

        # Class declarations scope the same way, putting members under the class name.
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

        # Methods are recorded without opening a scope, so helpers nested inside one qualify under the class rather than the method.
        if t == "method_definition":
            qual = _qual_name(source_bytes, node, scope_names)
            info = mk_info(qual, "method")
            results.append(info)
            add_child(scope_nodes[-1] if scope_nodes else None, node)
            for c in _iter_children(node):
                walk(c)
            return

        # Variable bindings only count as definitions when initialised with a function or arrow expression.
        if t == "variable_declarator":
            init = _node_field(node, "value")

            if init and init.type in ("function", "function_expression", "arrow_function"):
                qual = _qual_name(source_bytes, node, scope_names)
                kind = "var_arrow" if init.type == "arrow_function" else "var_func"
                info = mk_info(qual, kind)
                results.append(info)
                add_child(scope_nodes[-1] if scope_nodes else None, node)
                for c in _iter_children(node):
                    walk(c)
                return

        # Object-literal methods: find the variable the literal is bound to so the qualified name carries the owning object.
        if t == "pair" or (t == "method_definition" and node.parent and node.parent.type == "object"):
            enclosing = None
            p = node.parent

            while p is not None and p.type not in ("program",):
                if p.type == "variable_declarator":
                    enclosing = _ident_text(source_bytes, _node_field(p, "name"))
                    break

                p = p.parent

            # A plain key/value pair only counts when its value is actually a function.
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

        # Default case: keep descending so definitions nested inside arbitrary constructs are still found.
        for c in _iter_children(node):
            walk(c)

    walk(root)
    completed: List[DefInfoJS] = []

    # children_ids can only be known once the walk has finished, so back-fill each frozen record via replace().
    for info in results:
        kids = tuple(children_map.get(id(info.node), []))
        completed.append(replace(info, children_ids=kids))

    # Sorting guarantees source order regardless of the order the walk visited branches.
    return sorted(completed, key=lambda d: d.start)


def _collect_calls_js(node, source_bytes: bytes, def_node_ids: set) -> List[Tuple[str, str, int]]:
    """
    Collect the calls made directly by a JavaScript routine.

    Walks the routine's subtree recording each call expression as a `(name, kind, line)` tuple, where `kind` is `free` for a bare identifier, `self` for a call through `this`, and `method` for any other member access. Subtrees that are themselves routine definitions are skipped, so calls inside nested functions are attributed to those functions rather than to this one.

    Parameters:
    - `node`: The tree-sitter node of the routine to scan.
    - `source_bytes`: The encoded source the tree was parsed from.
    - `def_node_ids`: Node ids of every routine definition in the file, used to recognise nested routines.

    Returns:
    - A list of `(callee_name, kind, line)` tuples in source order, with 1-based line numbers.
    """

    # The root's own id is noted so the walker below can exempt it from the nested-routine skip.
    root_id = node.id
    calls: List[Tuple[str, str, int]] = []

    # Walker: record each call as (name, kind, line), classed as a free name, a `this` method, or a foreign-object method.
    def visit(n) -> None:
        """
        Recursively scan one subtree, appending any call expressions found to the enclosing `calls` list.

        Nested routine definitions are not descended into - they are opaque here and resolved on their own - with the root node itself as the one exception.

        Parameters:
        - `n`: The tree-sitter node to scan.
        """

        # tree-sitter rows are 0-based; calls are reported with 1-based source lines.
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            line = n.start_point[0] + 1

            # A bare identifier is a free call; member expressions need the receiver inspected before classifying.
            if fn is not None:
                if fn.type == "identifier":
                    calls.append((_node_text(source_bytes, fn), "free", line))
                elif fn.type == "member_expression":
                    obj = fn.child_by_field_name("object")
                    prop = fn.child_by_field_name("property")

                    # The callee name is the property alone - any object path is deliberately dropped.
                    if prop is not None:
                        name = _node_text(source_bytes, prop)

                        # Calls through `this` are the object's own methods; any other receiver makes it a plain method call.
                        if obj is not None and obj.type == "this":
                            calls.append((name, "self", line))
                        else:
                            calls.append((name, "method", line))

        # Descend into children, leaving nested routine bodies to be analysed as their own symbols.
        for i in range(n.named_child_count):
            child = n.named_child(i)
            if child.id in def_node_ids and child.id != root_id:
                continue   # a nested routine: opaque, resolved on its own
            visit(child)

    visit(node)

    return calls


def iter_symbols(source_blob: str, source_lines: Chunk) -> List[Symbol]:
    """
    Extract the documentable symbols from JavaScript source.

    Parses the source with tree-sitter and builds one `Symbol` per routine or class, carrying its header signature text, its parent's qualified name, any existing doc comment above the header, and the calls it makes. Unparseable source yields an empty list rather than raising.

    Parameters:
    - `source_blob`: The full source text.
    - `source_lines`: The source split into lines, used to recover header text.

    Returns:
    - A list of `Symbol` records in definition order, or an empty list if the source cannot be parsed.
    """

    # Unparseable source means no symbols, never an exception.
    try:
        tree, source_bytes = _parse_js(source_blob)
    except Exception:
        return []

    # Two lookups: parent qualnames keyed by node identity, and the def-node id set the call collector uses to skip nested routines.
    defs = iter_defs_with_info_js(tree, source_bytes)
    qual_by_pyid: Dict[int, str] = {id(d.node): d.qualname for d in defs}
    def_node_ids = {d.node.id for d in defs}
    symbols: List[Symbol] = []

    # Assemble one Symbol per definition: header text, parent link, any existing doc above the header, and its outgoing calls.
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
    """
    Return a tree-sitter node's named children as a list.
    """

    return [n.named_child(i) for i in range(n.named_child_count)]


def _is_js_statement(n) -> bool:
    """
    Report whether a tree-sitter node is a JavaScript statement or declaration.

    Parameters:
    - `n`: The tree-sitter node to test.

    Returns:
    - `True` for any statement, declaration or statement-block node, otherwise `False`.
    """

    # Suffix matching covers every *_statement and *_declaration node type without enumerating the grammar.
    t = n.type
    return t.endswith("_statement") or t.endswith("_declaration") or t == "statement_block"


def _js_is_def_stmt(node) -> bool:
    """
    Report whether a statement introduces a routine or class definition.

    Covers plain function, generator and class declarations, plus `const`/`let`/`var` declarations whose initialiser is a function-valued expression.

    Parameters:
    - `node`: The statement node to test.

    Returns:
    - `True` if the statement defines a routine or class, otherwise `False`.
    """

    if node.type in ("function_declaration", "generator_function_declaration", "class_declaration"):
        return True

    # A `const f = () => ...` style binding also counts: any declarator with a function-valued initialiser.
    if node.type in ("lexical_declaration", "variable_declaration"):
        for d in _named_children(node):
            if d.type == "variable_declarator":
                val = _node_field(d, "value")
                if val is not None and val.type in _JS_FUNCTION_VALUES:
                    return True

    return False


def _js_suite(node) -> Optional[List]:
    """
    Normalise a node into the list of statements it contains, if any.

    Accepts a bare statement (yielding a one-element list), a statement block, or an else clause (unwrapped recursively to whatever it holds). Empty blocks and non-statement nodes yield `None`.

    Parameters:
    - `node`: The candidate suite node, or `None`.

    Returns:
    - A non-empty list of statement nodes, or `None` if there is no suite.
    """

    if node is None:
        return None

    # An else clause wraps either a block or an `else if`; unwrap to whichever it holds.
    if node.type == "else_clause":
        kids = _named_children(node)
        return _js_suite(kids[0]) if kids else None

    if node.type == "statement_block":
        # Empty blocks count as having no suite at all.
        stmts = [c for c in _named_children(node) if _is_js_statement(c)]
        return stmts or None

    # A braceless single-statement body becomes a one-element suite.
    if _is_js_statement(node):
        return [node]
    return None


def _js_statement_lists(stmt) -> List[List]:
    """
    List the statement suites nested directly inside one compound statement.

    Each branch or body (if/else arms, loop bodies, try/catch/finally bodies, switch cases, bare blocks) becomes one list of statement nodes. Only the immediate level is returned; callers recurse into the returned statements themselves.

    Parameters:
    - `stmt`: The statement node to inspect.

    Returns:
    - A list of statement-node lists, one per non-empty nested suite; empty if the statement nests none.
    """

    t = stmt.type
    lists: List[List] = []

    # Field-addressed suites: if/else arms, loop bodies and try parts - catch and finally clauses must first be unwrapped to their inner block.
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

        # Switch cases and bare blocks carry their statements as direct children rather than named fields.
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
    """
    Report whether a node is a plain `let`/`const`/`var` declaration.

    Parameters:
    - `node`: The tree-sitter statement node to test.

    Returns:
    - `True` for variable and lexical declarations that do not bind a function; `False` otherwise.
    """

    return node.type in ("lexical_declaration", "variable_declaration") and not _js_is_def_stmt(node)


def _js_span(node) -> int:
    """
    Count the source lines a node spans, inclusive of both ends.

    Parameters:
    - `node`: The tree-sitter node to measure.

    Returns:
    - The number of lines from the node's start row to its end row.
    """

    return _row_of(node.end_point) - _row_of(node.start_point) + 1


def _js_opens_block(node) -> int:
    """
    Return the line span of a statement that opens a nested block, or 0.

    Parameters:
    - `node`: The tree-sitter statement node to test.

    Returns:
    - The node's inclusive line span when it is a compound statement or function definition, otherwise `0`.
    """

    # The span doubles as the truth value, so callers get the block's weight for free.
    if node.type in _JS_BLOCK_OPENERS or _js_is_def_stmt(node):
        return _js_span(node)
    return 0


def _js_closed_block(prev_node, resume_start: int, body_node) -> Optional[object]:
    """
    Find the outermost block around a statement that closes before segmentation resumes.

    Walks the ancestor chain from `prev_node` up to (but excluding) `body_node`, keeping the highest block-opening ancestor whose end row falls before `resume_start`.

    Parameters:
    - `prev_node`: The last statement recorded before the dedent.
    - `resume_start`: 1-based line on which the next statement begins.
    - `body_node`: The function body node that bounds the upward search.

    Returns:
    - The outermost qualifying block node, or `None` when no enclosing block closed before `resume_start`.
    """

    best = None
    n = prev_node.parent

    # Keep climbing so `best` ends up as the outermost block that closed before the resume line.
    while n is not None and n is not body_node:
        if (n.type in _JS_BLOCK_OPENERS or _js_is_def_stmt(n)) and _to_1based(_row_of(n.end_point)) < resume_start:
            best = n
        n = n.parent

    return best


def _js_body_block(node):
    """
    Locate the statement block that forms a function-like node's body.

    Parameters:
    - `node`: The tree-sitter node of a function, method or declaration.

    Returns:
    - The `statement_block` node, taken directly from the `body` field or from the declarator's `value` for declaration-bound functions, or `None` when there is none (e.g. an expression-bodied arrow).
    """

    b = _node_field(node, "body")
    if b is not None and b.type == "statement_block":
        return b
    val = _node_field(node, "value")

    # Declaration-bound functions (`const f = ...`) nest the block one level down, under the declarator's value.
    if val is not None:
        b = _node_field(val, "body")
        if b is not None and b.type == "statement_block":
            return b

    return None


def _collect_body_js(body_node, source_lines: Chunk) -> Tuple[List[int], Dict[int, str], List[SegStatement]]:
    """
    Gather segmentation facts for one JavaScript function body.

    Recursively walks the body's statement tree, tallying where statements start, which lines may act as paragraph boundaries, return-statement merge anchors, and any forced break after a leading run of declarations. Nested function definitions are recorded but never descended into.

    Parameters:
    - `body_node`: The tree-sitter `statement_block` of the function being segmented.
    - `source_lines`: The pristine source lines of the file.

    Returns:
    - A tuple of sorted candidate boundary lines, a map from boundary line to its indentation string, and the `SegStatement` records in source order.
    """

    # Shared state filled in by the nested walker below.
    line_count: Dict[int, int] = {}
    line_col: Dict[int, int] = {}
    recorded: List[Tuple[int, int, object, bool]] = []  # (start, depth, node, first_in_scope)
    merge_map: Dict[int, int] = {}                       # return line -> anchor (preceding statement) line
    force_break_lines: set = set()                       # first real statement after a leading declaration run

    def walk(stmts: List, is_top: bool, depth: int) -> None:
        """
        Record segmentation facts for one statement list, recursing into nested suites.

        Mutates the enclosing collectors (`line_count`, `line_col`, `recorded`, `merge_map`, `force_break_lines`) rather than returning anything.

        Parameters:
        - `stmts`: The statements of the current suite, in source order.
        - `is_top`: `True` only for the function body's own statement list.
        - `depth`: Nesting depth of the suite, zero at the body itself.
        """

        # A body of one plain statement plus its return reads as a single thought, so mark the pair for merging.
        if (len(stmts) == 2 and stmts[1].type == "return_statement"
                and stmts[0].type not in _JS_BLOCK_OPENERS and not _js_is_def_stmt(stmts[0])):
            a = _to_1based(_row_of(stmts[0].start_point))
            r = _to_1based(_row_of(stmts[1].start_point))
            if a != r:
                merge_map[r] = a

        # A long enough leading run of declarations is cut off, so the first real statement opens a fresh paragraph.
        ndecl = 0
        while ndecl < len(stmts) and _is_js_decl(stmts[ndecl]):
            ndecl += 1
        if ndecl >= SEG_MIN_LEADING_DECLS and ndecl < len(stmts) and stmts[ndecl].type != "return_statement":
            force_break_lines.add(_to_1based(_row_of(stmts[ndecl].start_point)))

        for idx, stmt in enumerate(stmts):
            skip = (not is_top) and idx == 0  # the first line of an inner suite is never a block start
            start = _to_1based(_row_of(stmt.start_point))

            # Tally start lines: a line claimed by more than one statement can never be a clean boundary.
            if not skip:
                line_count[start] = line_count.get(start, 0) + 1
                line_col[start] = _col_of(stmt.start_point)

            # Nested function definitions are recorded but not entered - they are segmented on their own.
            recorded.append((start, depth, stmt, is_top and idx == 0))
            if _js_is_def_stmt(stmt):
                continue
            for sub in _js_statement_lists(stmt):
                walk(sub, False, depth + 1)

    # Walk the whole body first; everything below derives boundaries from the tallies.
    top_stmts = [c for c in _named_children(body_node) if _is_js_statement(c)]
    walk(top_stmts, True, 0)
    boundary_lines: List[int] = []
    indent_of: Dict[int, str] = {}

    # Only a line holding exactly one statement, with nothing but whitespace before it, may open a paragraph.
    for line, count in line_count.items():
        if count != 1 or not (1 <= line <= len(source_lines)):
            continue
        text = source_lines[line - 1]
        if text[: line_col[line]].strip() != "":
            continue
        indent_of[line] = text[: len(text) - len(text.lstrip())]
        boundary_lines.append(line)

    # Return-merge anchors must be boundaries even when the single-statement filter rejected them.
    for anchor in set(merge_map.values()):
        if 1 <= anchor <= len(source_lines) and anchor not in indent_of:
            text = source_lines[anchor - 1]
            indent_of[anchor] = text[: len(text) - len(text.lstrip())]
            boundary_lines.append(anchor)

    # Restore source order before deriving the per-statement segmentation facts.
    boundary_lines.sort()
    recorded.sort(key=lambda r: r[0])
    seg_statements: List[SegStatement] = []

    # Translate each recorded statement into the language-neutral form the segmenter consumes.
    for i, (start, depth, node, first_in_scope) in enumerate(recorded):
        end = _to_1based(_row_of(node.end_point))
        closed = 0

        # A dedent since the previous statement means a block just closed; measure its size.
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
    Extract the prose of any comment block sitting directly above a function header.

    Parameters:
    - `source_lines`: The source lines of the file.
    - `header_start`: 1-based line number of the function header.

    Returns:
    - The comment text with `/* */` or `//` decoration stripped, or an empty string when no comment precedes the header.
    """

    # Both comment styles are reduced to bare prose so the rewriter can reuse the wording.
    rng = _scan_existing_comment_block_above(source_lines, header_start)
    if rng is None:
        return ""
    s, e = rng
    block = "\n".join(source_lines[s - 1:e])
    if "/*" in block:
        return _extract_first_comment_block(block)
    return "\n".join(ln.strip()[2:].strip() if ln.strip().startswith("//") else ln.strip()
                     for ln in block.split("\n")).strip()


def file_doc_target_js(source_blob: str, source_lines: Chunk) -> Optional[FileDocTarget]:
    """
    Locate the top-of-file documentation zone in JavaScript source.

    Parameters:
    - `source_blob`: The full source text (unused; kept for interface parity with the other languages).
    - `source_lines`: The source split into lines.

    Returns:
    - A `FileDocTarget` describing the leading comment zone, or `None` if no suitable zone was found.
    """

    # The zone scan is shared with the other brace languages; only the slash comment styles here are JS-specific.
    return scan_brace_leading_zone(source_lines, SLASH_BLOCK_STYLE)


def iter_block_targets_js(source_blob: str, source_lines: Chunk) -> List[BlockTarget]:
    """
    Collect block-comment targets for every function in the JavaScript source.

    Each definition with a non-empty body yields a `BlockTarget` carrying its header and body line ranges, structural segment breaks, any existing doc comment, and a comment-stripped span hash used to re-bind the target after the file is edited.

    Parameters:
    - `source_blob`: The full JavaScript source text.
    - `source_lines`: The source split into lines.

    Returns:
    - A list of `BlockTarget` records, one per definition with at least one body statement.
    """

    tree, source_bytes = _parse_js(source_blob)
    targets: List[BlockTarget] = []

    # Build one target per definition that has at least one body statement; the recorded span hash is comment-stripped so later insertions still re-bind.
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
                sig=_js_span_hash(source_lines, info.header_start, info.end),
                segments=segments,
            )
        )

    return targets


# ---- Depth order -------------------------------------------------------------


def deepest_first_js(defs: List[DefInfoJS]) -> List[DefInfoJS]:
    """
    Order JavaScript definitions so the most deeply nested come first.

    Parameters:
    - `defs`: The definitions to order.

    Returns:
    - A new list sorted by descending depth, with later and more tightly enclosed spans ahead of those that contain them.
    """

    # Deeper, later spans sort first so inner bodies are patched before the regions that enclose them.
    return sorted(defs, key=lambda d: (d.depth, d.start, -d.end), reverse=True)


# ---- Snippet assembly (stubbing direct children) ----------------------------


def _render_jsdoc_block(text: str, base_indent: str) -> List[str]:
    """
    Render comment text as an indented JSDoc-style block.

    Parameters:
    - `text`: The comment body; each line becomes one gutter line.
    - `base_indent`: Whitespace prefix applied to every emitted line.

    Returns:
    - The complete block as a list of lines, delimiters included; empty text yields a `(no documentation)` placeholder.
    """

    lines = text.splitlines()
    out = [f"{base_indent}/**"]

    # Empty text still gets a visible placeholder line; the outer rstrip leaves blank body lines as a bare gutter with no trailing space.
    if lines:
        for ln in lines:
            out.append(f"{base_indent} * {ln.rstrip()}".rstrip())
    else:
        out.append(f"{base_indent} * (no documentation)")

    out.append(f"{base_indent} */")

    return out


def _render_js_line_comment(text: str, base_indent: str) -> List[str]:
    """
    Render comment text as `//` line comments at the given indentation.

    Parameters:
    - `text`: The comment body.
    - `base_indent`: Whitespace prefix for each emitted line.

    Returns:
    - One `//` line per input line, or a single `(no documentation)` placeholder when the text is empty.
    """

    # Empty text yields a placeholder rather than nothing; blank lines collapse to a bare `//` via the outer rstrip.
    lines = text.splitlines()
    if not lines:
        return [f"{base_indent}// (no documentation)"]
    return [f"{base_indent}// {ln.rstrip()}".rstrip() for ln in lines]


def _detect_doc_style_js(tree, source_bytes: bytes) -> str:
    """
    Detect whether the file's existing comments favour line or block style.

    The first comment in the file is treated as a banner and excluded, and block style wins unless every remaining comment is a `//` line.

    Parameters:
    - `tree`: The parsed tree-sitter tree for the source.
    - `source_bytes`: The source as bytes, used to inspect each comment's prefix.

    Returns:
    - `"line"` if only line comments are present, otherwise `"block"`.
    """

    # Note the position of any leading banner comment so a block-style licence header cannot decide the verdict.
    root = tree.root_node
    banner_start = (root.children[0].start_byte
                    if root.children and root.children[0].type == "comment" else None)
    has_line = has_block = False
    stack = [root]

    while stack:
        n = stack.pop()

        # The first two bytes decide the flavour; anything that is not `//` counts as a block comment.
        if n.type == "comment" and n.start_byte != banner_start:
            if source_bytes[n.start_byte:n.start_byte + 2] == b"//":
                has_line = True
            else:
                has_block = True

        for i in range(n.named_child_count):
            stack.append(n.named_child(i))

    # Block wins any mix: only a file whose comments are exclusively line-style reports `line`.
    return "line" if (has_line and not has_block) else "block"


def _strip_jsdoc_gutters(block: List[str]) -> str:
    """
    Strip the leading `*` gutters from the lines of a JSDoc comment body.

    Only one space after each `*` is removed, so indentation that belongs to the comment text is preserved.

    Parameters:
    - `block`: The comment lines, excluding the `/**` and `*/` delimiter rows.

    Returns:
    - The de-guttered text joined with newlines and trimmed of blank edges.
    """

    cleaned: List[str] = []

    for ln in block:
        s = ln.lstrip()

        # Only the gutter `*` and a single following space are eaten, so deliberate indentation inside the comment survives.
        if s.startswith("*"):
            s = s[1:]
            if s.startswith(" "):
                s = s[1:]

        cleaned.append(s.rstrip())

    # The final strip drops the blank edge lines left where the delimiters were removed.
    return "\n".join(cleaned).strip()


def _extract_first_comment_block(reply: str) -> str:
    """
    Extract the first comment block from a model reply.

    Tries, in order: a `/** ... */` JSDoc block (tolerating a missing terminator), a fenced code block, and finally the whole reply. Gutters are stripped from whichever form is found.

    Parameters:
    - `reply`: The raw model reply text.

    Returns:
    - The extracted comment text with gutters removed and blank edges trimmed.
    """

    # Dedent first so a uniformly indented reply still has its `/**` opener recognised.
    lines = textwrap.dedent(reply).split("\n")
    stripped = [ln.strip() for ln in lines]

    # First preference is a real JSDoc block; a missing opener raises ValueError and falls through to the fence path.
    try:
        start_idx = stripped.index("/**") + 1

        # An unterminated block is tolerated: everything to the end of the reply counts as the comment.
        for end_idx in range(start_idx, len(lines)):
            if "*/" in lines[end_idx]:
                break
        else:
            end_idx = len(lines)

        return _strip_jsdoc_gutters(lines[start_idx:end_idx])
    except ValueError:
        pass

    # Fenced code is the second choice; as a last resort the whole reply is taken, so a comment is never silently dropped.
    fence_idxs = [i for i, s in enumerate(stripped) if s.startswith("```")]
    if len(fence_idxs) >= 2:
        return _strip_jsdoc_gutters(lines[fence_idxs[0] + 1:fence_idxs[1]])
    return _strip_jsdoc_gutters(lines)


def assemble_snippet_for_js(
    info_by_id: Dict[int, DefInfoJS],
    source_lines: Chunk,
    node_id: int,
    docs_by_id: Dict[int, str]
) -> str:
    """
    Assemble the prompt snippet for a JS definition, eliding the bodies of its direct children.

    The definition's own header and loose body code are kept verbatim, but each direct child is collapsed to its already-generated JSDoc block plus its header line(s). Generation runs deepest-first, so every child's doc exists by the time its parent is assembled - the parent's prompt is informed by child documentation without re-sending child code.

    Parameters:
    - `info_by_id`: Map from `id()` of a definition's node to its `DefInfoJS`.
    - `source_lines`: The pristine source lines of the file.
    - `node_id`: The `id()` of the definition to assemble.
    - `docs_by_id`: Docs generated so far, keyed by `id()` of each node.

    Returns:
    - The assembled snippet text for the definition.
    """

    # Collect the routine's header and its direct children, sorted so one forward cursor can walk the body.
    info = info_by_id[node_id]
    header_text = _get_text_for_lines(source_lines, info.header_start, info.header_end)
    body_chunks: List[str] = []
    direct_children = [info_by_id[cid] for cid in info.children_ids]
    direct_children.sort(key=lambda d: d.start)  # body order: the forward cursor splices gap, stub, gap, stub...
    cursor = info.header_end + 1

    # Each child is reduced to its JSDoc plus header line - bodies are dropped to keep the parent's snippet small.
    for child in direct_children:
        if cursor <= child.start - 1:
            body_chunks.append(_get_text_for_lines(source_lines, cursor, child.start - 1))
        hdr_end = max(child.header_end, child.header_start)
        header_last_line = source_lines[hdr_end - 1] if 1 <= hdr_end <= len(source_lines) else ""
        indent = " " * _leading_spaces_count(header_last_line)
        child_doc = docs_by_id.get(id(child.node), "")
        jsdoc_block = "\n".join(_render_jsdoc_block(child_doc, indent))
        child_header = _get_text_for_lines(source_lines, child.header_start, hdr_end)
        body_chunks.append(jsdoc_block + ("\n" if jsdoc_block and child_header else "") + child_header)
        cursor = child.end + 1

    # Flush any code remaining after the last child.
    if cursor <= info.end:
        body_chunks.append(_get_text_for_lines(source_lines, cursor, info.end))
    parts: List[str] = [header_text]

    # Guard against gluing the body straight onto a header that lacks a trailing newline.
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
    Generate a JSDoc body for every JavaScript definition, deepest-first.

    Each definition is shown to the model as its header plus a body in which direct children are collapsed to their already-generated docs. Existing comments are ingested and updated rather than discarded, callee notes are appended when a provider is given, and every generation turn is popped from `messages` afterwards so the shared context never grows. When a verifier is supplied each doc must pass it; a doc that fails twice is still written but flagged for review, and a routine whose generation fails outright receives a visible placeholder.

    Parameters:
    - `llm`: The local chat model used for generation.
    - `cfg`: Generation settings for each turn.
    - `messages`: The primed chat history; restored to its original length after every turn.
    - `defs`: All definitions discovered in the file.
    - `source_blob`: The full source text.
    - `source_lines`: The pristine source lines used for snippet assembly.
    - `doc_order`: Optional qualname ordering that overrides the deepest-first default.
    - `callee_context`: Optional callback returning project-level notes about a routine's callees.
    - `on_doc`: Optional callback invoked with each qualname and doc as it is produced.
    - `verifier`: Optional verification floor applied to every generated doc.

    Returns:
    - A dict mapping `id()` of each definition's node to its generated doc text.
    """

    docs_by_node_id: Dict[int, str] = {}
    info_by_id: Dict[int, DefInfoJS] = {id(d.node): d for d in defs}

    # Deepest-first by default, so a child's doc always exists before its parent's snippet is assembled; a call-graph order can override this.
    if doc_order:
        ordered = apply_doc_order(defs, lambda d: d.qualname, doc_order, lambda d: (-d.depth, d.start, -d.end))
    else:
        ordered = deepest_first_js(defs)

    # Build the definition's snippet with child bodies collapsed to their docs, then crop it to the model's context budget.
    for info in ordered:
        node_id = id(info.node)
        snippet = assemble_snippet_for_js(info_by_id, source_lines, node_id, docs_by_node_id)
        header_lines = max(1, info.header_end - info.header_start + 1)
        snippet, omitted = fit_snippet(llm, cfg, messages, snippet, header_lines, MARKER_JS)
        if omitted:
            echo(f"[JS] Elided {omitted} body line(s) from '{info.qualname}' to fit the context window")
        echo("\n[JS] Snippet...\n")
        echo(snippet)

        # Classes get a dedicated prompt: summarise the abstraction as a whole rather than parroting one method.
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

        existing = _doc_above_header_js(source_lines, info.header_start)

        # Ingest-and-update: an existing comment is fed back so accurate institutional knowledge survives the rewrite.
        if existing:
            prompt += (
                "\nThe routine is already documented as follows. Ingest and update this existing comment - keep "
                "whatever is still accurate, correct anything stale, and reformat it to the requested style - rather "
                f"than writing from scratch:\n\n{existing}\n"
            )

        # Fold in project-layer notes about this routine's callees, when a provider was supplied.
        if callee_context is not None:
            notes = callee_context(info.qualname)
            if notes:
                prompt += "\n" + notes + "\n"

        # The generation turn is popped straight after the reply - the shared history never grows.
        messages.append({"role": "user", "content": prompt})
        reply = llm.generate(messages, cfg=cfg)
        echo(f"\n[JS] LLM output:\n\n{reply}")
        messages.pop()
        doc = _extract_first_comment_block(reply)

        # last_reply is a one-cell list so the retry closure can swap in each newer reply.
        if doc and verifier is not None:
            last_reply = [reply]

            def regenerate(feedback: str, _prompt: str = prompt) -> str:
                """
                Regenerate the doc-comment in response to verification feedback.

                Replays the original prompt and the previous reply, appends the feedback as a fresh user turn, then pops all three so the shared history is left untouched. A successful extraction becomes the new previous reply for any further round.

                Parameters:
                - `feedback`: The verifier's complaint for the model to address.
                - `_prompt`: The original generation prompt, bound at definition time.

                Returns:
                - The regenerated comment text, or an empty string if no comment block could be extracted.
                """

                # Replay the prompt, the previous reply, and the feedback as a throwaway exchange - all three turns are popped before returning.
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

            # Verification failure is non-fatal: the doc is still written, but flagged for human review.
            if not ok:
                error(f"[verify] '{info.qualname}': comment failed verification twice; writing it anyway - "
                      f"review this comment")

        # Record a visible placeholder rather than skipping the routine, then feed the doc to the run-level store.
        if not doc:
            doc = f"{info.kind} `{info.qualname}` - documentation generation failed."
        docs_by_node_id[node_id] = doc
        if on_doc is not None:
            on_doc(info.qualname, doc)

    return docs_by_node_id


# ---- Textual patcher (insert/replace above headers) -------------------------


def _scan_existing_comment_block_above(source_lines: Chunk, header_start_line_1b: int) -> Optional[Tuple[int, int]]:
    """
    Locate any existing comment block sitting directly above a definition header.

    Recognises either a `/* ... */` block whose closing line is immediately above the header, or an unbroken run of `//` line comments touching it; a blank line breaks adjacency, so detached comments are never claimed.

    Parameters:
    - `source_lines`: The lines of the file being patched.
    - `header_start_line_1b`: The 1-based line number of the definition header.

    Returns:
    - A 1-based inclusive `(start, end)` line range for the existing block, or `None` if there is none.
    """

    # Bail out when the header sits on the first line - there is nothing above it.
    i = header_start_line_1b - 2  # zero-based line just above header
    if i < 0:
        return None

    # A line ending in */ directly above means a block comment owns the slot.
    if source_lines[i].rstrip().endswith("*/"):
        j = i

        # Walk upwards to the matching /* opener; the whole block becomes the replace range.
        while j >= 0:
            if source_lines[j].lstrip().startswith("/*"):
                return (j + 1, i + 1)
            j -= 1

    # Otherwise look for an unbroken run of // line comments touching the header.
    j = i
    saw_slash = False

    while j >= 0:
        stripped = source_lines[j].lstrip()

        if stripped.startswith("//"):
            saw_slash = True
            j -= 1
            continue

        # A blank line (or any code) ends the run - only comments adjacent to the header count.
        if stripped == "":
            break  # blank breaks adjacency
        break

    if saw_slash:
        # The loop overshoots by one line, so +2 converts back to the 1-based first comment line.
        start_1b = j + 2
        return (start_1b, i + 1)

    return None


def patch_comments_textually_js(source_lines: Chunk, defs: List[DefInfoJS], doc_map: Dict[int, str],
                                style: str = "block") -> Chunk:
    """
    Insert or replace a comment block above each documented definition.

    Definitions are patched bottom-up so insertions never invalidate the line numbers of those still to be processed. A comment block already sitting against a header is replaced in place; otherwise the new block is inserted with a separating blank line. Indentation is copied from the header line, and definitions absent from `doc_map` are left untouched.

    Parameters:
    - `source_lines`: The pristine source lines.
    - `defs`: The definitions discovered in the file.
    - `doc_map`: Doc text keyed by `id()` of each definition's node.
    - `style`: `"line"` for `//` comments, anything else for JSDoc blocks.

    Returns:
    - A new list of lines with the comments applied; the input list is not modified.
    """

    # Pick the renderer matching the file's comment style, and patch a copy - never the input.
    render = _render_js_line_comment if style == "line" else _render_jsdoc_block
    out_lines = source_lines[:]

    # Patch bottom-up so earlier insertions cannot shift the line numbers of definitions still to come.
    for info in sorted(defs, key=lambda d: d.start, reverse=True):
        node_id = id(info.node)
        if node_id not in doc_map:
            continue
        doc = doc_map[node_id]
        header_line_text = source_lines[info.header_start - 1]
        indent = header_line_text[: len(header_line_text) - len(header_line_text.lstrip())]
        new_block_lines = render(doc, indent)
        existing = _scan_existing_comment_block_above(out_lines, info.header_start)

        # Replace an existing comment block in place; otherwise insert the new block with a separating blank line.
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
    Run the full JavaScript definition pass and return the patched lines.

    This is the shared per-language entry point: it parses the source with Tree-sitter, matches the file's existing doc-comment style, primes the chat by describing the imports, generates a JSDoc body for every definition, and finally patches the comments into the pristine source lines.

    Parameters:
    - `llm`: The local chat model used for generation.
    - `cfg`: Generation settings for each turn.
    - `messages`: The primed chat history shared across the run.
    - `source_blob`: The full source text handed to the parser.
    - `source_lines`: The pristine source lines that patching works from.
    - `doc_order`: Optional qualname ordering from the project call graph.
    - `callee_context`: Optional callback returning notes about a routine's callees.
    - `on_doc`: Optional callback invoked with each qualname and doc produced.
    - `verifier`: Optional verification floor passed through to generation.

    Returns:
    - The source lines with doc comments inserted or updated.
    """

    echo("Parsing JavaScript source with Tree-sitter...")
    tree, source_bytes = _parse_js(source_blob)
    defs = iter_defs_with_info_js(tree, source_bytes)
    echo(f"Found {len(defs)} JS definitions")
    style = _detect_doc_style_js(tree, source_bytes)
    echo(f"Doc-comment style for this file: {style}")
    echo("Identifying imports/requires...")
    describe_imports_js(llm, cfg, messages, tree, source_bytes)
    echo("Generating JSDoc comments...\n")
    doc_map = generate_comments_js(llm, cfg, messages, defs, source_blob, source_lines,
                                   doc_order=doc_order, callee_context=callee_context, on_doc=on_doc,
                                   verifier=verifier)
    echo("Applying JS patches...\n")

    # Patching works from the pristine source lines, never from anything the model emitted.
    return patch_comments_textually_js(source_lines, defs, doc_map, style=style)


# ---- Manifest emit (model-free) ----------------------------------------------


def _js_span_hash(source_lines: Chunk, header_start: int, end: int) -> str:
    """
    Hash a JavaScript routine's source span for manifest re-binding.

    The span is reduced to its comment-stripped code signature before hashing, so a routine still matches its manifest answer after comments are inserted or rewritten around it.

    Parameters:
    - `source_lines`: The full file as a list of lines.
    - `header_start`: 1-based line number of the routine's header.
    - `end`: 1-based inclusive line number of the routine's last line.

    Returns:
    - The hex digest string identifying the routine's code.
    """

    # Late import avoids a circular dependency; the slice bounds are 1-based and inclusive.
    from scale_blocks import _code_signature
    span = source_lines[header_start - 1:end]

    # Hash only the comment-stripped signature so inserting docs cannot break later re-binding.
    return routine_text_hash("\n".join(_code_signature(span, SLASH_BLOCK_STYLE)))


def collect_def_requests_js(source_blob: str, source_lines: Chunk, escalation) -> int:
    """
    Record one manifest def request for every JavaScript routine in the file.

    Each routine's verbatim text crosses into the manifest once, keyed by qualname and comment-stripped span hash so the apply phase can re-bind answers to unchanged code.

    Parameters:
    - `source_blob`: The full source text.
    - `source_lines`: The same source as a list of lines.
    - `escalation`: The run-manifest collector that receives the def requests.

    Returns:
    - The number of routines recorded.
    """

    tree, source_bytes = _parse_js(source_blob)
    defs = iter_defs_with_info_js(tree, source_bytes)

    # Key each routine by its comment-stripped span hash so the apply phase can re-bind answers to unchanged code.
    for info in defs:
        span = _get_text_for_lines(source_lines, info.header_start, info.end)
        escalation.record_def(qualname=info.qualname, kind=info.kind,
                              sig_hash=_js_span_hash(source_lines, info.header_start, info.end), snippet=span)

    # The count feeds the manifest completeness counter.
    return len(defs)


# ---- Manifest apply (model-free) ----------------------------------------------


def _clean_js_comment_answer(text: str) -> str:
    """
    Normalise a model's JavaScript comment reply to bare prose.

    Strips any surrounding Markdown code fence, then removes `//` or `/* ... */` comment delimiters if the model answered in comment syntax, leaving only the comment text for later re-wrapping.

    Parameters:
    - `text`: The raw model reply, possibly fenced or formatted as a JS comment.

    Returns:
    - The cleaned comment body; may be an empty string.
    """

    # Models often fence their reply; unwrap a single whole-body Markdown fence first.
    body = (text or "").strip()
    fence = re.match(r"^```[^\n]*\n(.*)\n```$", body, flags=re.DOTALL)
    if fence:
        body = fence.group(1).strip()

    # Accept replies already written as // or /* */ comments, reducing them to bare prose for re-wrapping.
    if body.startswith("/*") or body.startswith("//"):
        if body.startswith("//"):
            body = "\n".join(ln.strip()[2:].strip() if ln.strip().startswith("//") else ln.strip()
                             for ln in body.split("\n")).strip()
        else:
            body = _extract_first_comment_block(body)

    return body.strip()


def apply_manifest_js(source_lines: Chunk, manifest: dict) -> Chunk:
    """
    Apply a filled run manifest's answers to JavaScript source lines.

    Def answers are re-bound to routines by qualname plus comment-stripped span hash, so docs only attach to code unchanged since emit. Block answers become insertion-only edits, and a routine's edits are kept only if the preservation guard proves the code byte-identical; unmatched or unanswered requests leave their routines untouched.

    Parameters:
    - `source_lines`: The current source as a list of lines.
    - `manifest`: The filled manifest dictionary holding the `requests` list.

    Returns:
    - The patched list of source lines.
    """

    # Split the manifest into def and block requests; the two phases patch the same running line list in turn.
    from scale_blocks import SLASH_LINE_STYLE, _apply_edits, code_preserved, _parse_comment_reply
    width = int(manifest.get("line_length") or 0)   # comment wrap budget set at emit (0 = unwrapped)
    requests = manifest.get("requests", [])
    def_reqs = [r for r in requests if r.get("def") is not None]
    block_reqs = [r for r in requests if r.get("blocks") is not None]
    out_lines = source_lines

    # Def phase: re-parse the current text, detect the file's existing doc style, and index answered requests by qualname plus span hash.
    if def_reqs:
        tree, sb = _parse_js("\n".join(out_lines))
        records = iter_defs_with_info_js(tree, sb)
        style = _detect_doc_style_js(tree, sb)
        wanted = {(r["qualname"], r["sig_hash"]): r for r in def_reqs}
        doc_map: Dict[int, str] = {}
        used: set = set()
        matched: List[DefInfoJS] = []

        # Re-bind by recomputing each routine's comment-stripped hash; a miss means the code changed since emit.
        for info in records:
            key = (info.qualname, _js_span_hash(out_lines, info.header_start, info.end))
            req = wanted.get(key)
            if req is None or key in used:
                continue
            answer = req["def"].get("answer")

            # An empty answer leaves the routine untouched rather than splicing a blank doc.
            if not answer or not str(answer).strip():
                echo(f"[apply] Def request '{req['id']}' has no answer; leaving the definition untouched")
                continue

            doc = _clean_js_comment_answer(str(answer))

            # A literal NONE reply is the model declining; marking the key used stops duplicate spans matching twice.
            if doc and doc.upper() != "NONE":
                doc_map[id(info.node)] = doc
                used.add(key)
                matched.append(info)

        # All accepted docs go in via one insertion-only textual patch.
        out_lines = patch_comments_textually_js(out_lines, matched, doc_map, style=style)

    # Block phase: re-enumerate targets against the already def-patched text so boundary line numbers are current.
    if block_reqs:
        targets = iter_block_targets_js("\n".join(out_lines), out_lines)
        by_key = {(t.qualname, t.sig): t for t in targets}
        all_edits: List[Tuple[int, Optional[str], str]] = []

        # Re-bind each block request to its current target by qualname and signature hash.
        for req in block_reqs:
            target = by_key.get((req["qualname"], req["sig_hash"]))
            chunks = req["blocks"].get("chunks", [])

            # A missing target means the routine changed since emit; skip rather than guess a placement.
            if target is None:
                echo(f"[apply] No match for block request '{req['id']}'; skipping")
                continue

            # A wholly unanswered request is left alone so partially filled manifests stay harmless.
            if all(c.get("answer") is None for c in chunks):
                echo(f"[apply] Block request '{req['id']}' has no answers; leaving routine untouched")
                continue

            edits: List[Tuple[int, Optional[str], str]] = []

            # Map each chunk's index to the target's current boundary line; out-of-range indices are dropped silently.
            for chunk in chunks:
                bidx = chunk["bidx"]
                if not (0 <= bidx < len(target.boundary_lines)):
                    continue
                boundary = target.boundary_lines[bidx]
                comment = _parse_comment_reply(chunk.get("answer") or "", SLASH_LINE_STYLE)
                edits.append((boundary, comment, target.indent_of.get(boundary, "")))

            # Trial-apply this routine's edits in isolation so the guard can vet them.
            trial = _apply_edits(out_lines, edits, SLASH_LINE_STYLE, width)

            # Keep the edits only if the guard proves the code byte-identical; otherwise the whole routine's edits are dropped.
            if code_preserved(out_lines, trial, SLASH_LINE_STYLE):
                all_edits.extend(edits)
            else:
                echo(f"[apply] Skipped '{req['qualname']}': block edit would alter code; keeping original")

        # Apply every vetted edit in one batch so boundary line numbers stay valid.
        out_lines = _apply_edits(out_lines, all_edits, SLASH_LINE_STYLE, width)

    return out_lines
