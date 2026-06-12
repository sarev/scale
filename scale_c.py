#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The C language worker: it parses source with tree-sitter (tolerating both the 0.21 and 0.22 binding APIs), discovers
every function definition and prototype, and supplies the per-language pieces the shared pipeline needs - the
definition pass, the block-pass segmenter, the symbol scanner for the project call graph, and the file-doc target.

The definition pass gives each routine one bounded model turn, seeded with its existing comment and callee notes, then
splices the result above the header with `patch_comments_textually_c` - existing comment blocks are replaced in place
and code lines are never touched. `_collect_body_c` and `iter_block_targets_c` feed the structural segmenter, and a
doc-style detector picks `//` or `/* */` rendering to match the file.

The module also owns two C-specific extras: the run-wide doc-site plan (`CDocPlan` and `plan_doc_sites_c`), which
redirects a function's documentation from its definition to its header declaration when both are in the run, and the
online-mode pair `collect_def_requests_c` and `apply_manifest_c`, which record routines in the run manifest and
re-bind the filled answers by qualified name and span hash before patching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from scale_blocks import BlockTarget, SegStatement, structural_breaks, SEG_MIN_LEADING_DECLS, SLASH_BLOCK_STYLE
from scale_escalate import routine_text_hash
from scale_filedoc import FileDocTarget, scan_brace_leading_zone
from scale_llm import LocalChatModel, GenerationConfig, Messages, Chunk
from scale_log import echo, error
from scale_project import Symbol, apply_doc_order
from scale_text import fit_snippet, MARKER_C, PRIMING_ACK
from tree_sitter import Parser, Language  # type: ignore
from tree_sitter_c import language as c_language
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import re
import textwrap


# ---------------- Tree-sitter bootstrap (handles capsule vs Language object, and both Parser APIs)

def _load_c_language_and_parser() -> Tuple[Language, Parser]:
    """
    Construct the tree-sitter C `Language` and `Parser` across binding versions.

    Tolerates both the one-argument `Language(...)` API (tree-sitter >= 0.22) and the labelled two-argument 0.21 form, plus both the `set_language()` and constructor-argument `Parser` styles.

    Returns:
    - A `(Language, Parser)` pair ready to parse C source.
    """

    # What the grammar binding returns - a `Language` or a raw pointer - depends on its version.
    ptr_or_lang: Any = c_language()

    # Shim over the tree-sitter API split: one-argument `Language` from 0.22 onward, the labelled two-argument form on 0.21.
    if isinstance(ptr_or_lang, Language):
        lang = ptr_or_lang
    else:
        try:
            lang = Language(ptr_or_lang)        # new API (tree-sitter >= 0.22)
        except TypeError:
            lang = Language(ptr_or_lang, "C")   # old 0.21 API (second arg is a label)

    # Parser construction split likewise: older bindings use `set_language()`, newer ones take the language in the constructor.
    try:
        p = Parser()
        p.set_language(lang)
    except AttributeError:
        p = Parser(lang)

    return lang, p


C_LANGUAGE, C_PARSER = _load_c_language_and_parser()


def _parse_c(source_blob: str) -> Tuple[Any, bytes]:
    """
    Parse C source with tree-sitter and return the syntax tree plus the bytes it was parsed from.

    Line endings are normalised to LF before encoding, so node byte offsets index into the returned buffer rather than the original blob.

    Parameters:
    - `source_blob`: The C source text to parse.

    Returns:
    - A `(tree, source_bytes)` tuple: the tree-sitter parse tree and the UTF-8 bytes it indexes into.
    """

    # Normalise every line ending to LF before encoding, so tree-sitter rows and byte offsets match the line maths used downstream.
    norm = source_blob.replace("\r\n", "\n").replace("\r", "\n")
    source_bytes = norm.encode("utf-8", errors="replace")

    # Return the encoded bytes too: node byte offsets index this exact buffer, not the original blob.
    return C_PARSER.parse(source_bytes), source_bytes


# ---------------- Utilities

def _to_1based(row0: int) -> int:
    """
    Convert a 0-based tree-sitter row to a 1-based line number.

    Parameters:
    - `row0`: The 0-based row index.

    Returns:
    - The corresponding 1-based line number.
    """

    return row0 + 1


def _row_of(point) -> int:
    """
    Extract the row from a tree-sitter point of either supported shape.

    Older tree-sitter bindings give points as `(row, col)` tuples while newer ones use objects with a `.row` attribute; both are accepted.

    Parameters:
    - `point`: A tree-sitter point, as an object or a tuple.

    Returns:
    - The 0-based row of the point.
    """

    # Tolerate both tree-sitter binding generations: attribute-style points and plain tuples.
    try:
        return point.row  # object with .row
    except AttributeError:
        return point[0]   # tuple (row, col)


def _col_of(point) -> int:
    """
    Extract the column from a tree-sitter point of either supported shape.

    The tuple fallback mirrors `_row_of`, covering older tree-sitter bindings that give points as `(row, col)` tuples rather than objects with a `.column` attribute.

    Parameters:
    - `point`: A tree-sitter point, as an object or a tuple.

    Returns:
    - The 0-based column of the point.
    """

    # Tolerate both tree-sitter binding generations, as in the row accessor.
    try:
        return point.column  # object with .column
    except AttributeError:
        return point[1]      # tuple (row, col)


def _line_span_from_node(n) -> Tuple[int, int]:
    """
    Return a node's extent as a 1-based inclusive line span.

    Parameters:
    - `n`: The tree-sitter node.

    Returns:
    - A `(first, last)` tuple of 1-based line numbers covering the node.
    """

    return _to_1based(_row_of(n.start_point)), _to_1based(_row_of(n.end_point))


def _get_text_for_lines(source_lines: Chunk, a: int, b: int) -> str:
    """
    Join a 1-based inclusive line range into a single newline-separated string.

    Bounds are clamped to the file rather than treated as errors, and an empty or inverted range yields the empty string.

    Parameters:
    - `source_lines`: The file's lines, without trailing newlines.
    - `a`: First line of the range (1-based, inclusive).
    - `b`: Last line of the range (1-based, inclusive).

    Returns:
    - The selected lines joined with newlines, or an empty string for an empty range.
    """

    # Clamp out-of-range bounds to the file instead of raising; a range left inverted after clamping is treated as empty.
    a = max(1, a)
    b = min(len(source_lines), b)
    if a > b:
        return ""
    return "\n".join(source_lines[a - 1:b])


def _node_text(source_bytes: bytes, n) -> str:
    """
    Decode the source text covered by a tree-sitter node.

    Safe with `None`, so callers may slice optional child nodes without a guard.

    Parameters:
    - `source_bytes`: The encoded source that the node's byte offsets index into.
    - `n`: The tree-sitter node, or `None`.

    Returns:
    - The node's UTF-8 text (invalid bytes replaced), or an empty string when `n` is `None`.
    """

    # Safe with a missing node, so callers can fetch optional children without a guard.
    if n is None:
        return ""
    return source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")


def _leading_spaces_count(line: str) -> int:
    """
    Count the leading spaces on a line.

    Only spaces are counted: a leading tab stops the count rather than being expanded.

    Parameters:
    - `line`: The line to measure.

    Returns:
    - The number of leading space characters.
    """

    return len(line) - len(line.lstrip(" "))


def _scan_existing_comment_block_above(source_lines: Chunk, header_start_line_1b: int) -> Optional[Tuple[int, int]]:
    """
    Locate an existing comment block sitting directly above a C function header.

    Recognises either a `/* ... */` block whose closing delimiter ends the line immediately above the header, or an unbroken run of `//` lines touching the header. A blank line breaks adjacency, so detached comments are never claimed.

    Parameters:
    - `source_lines`: The source file as a list of lines.
    - `header_start_line_1b`: 1-based line number of the function header.

    Returns:
    - 1-based inclusive `(start, end)` line span of the comment block, or `None` if no adjacent comment exists.
    """

    # Index the line directly above the header; at the top of the file there is nothing to scan.
    i = header_start_line_1b - 2  # zero-based line just above header
    if i < 0:
        return None

    # A trailing */ immediately above the header signals a block comment to claim.
    if source_lines[i].rstrip().endswith("*/"):
        j = i

        # Walk upward to the matching /* opener and claim the whole block.
        while j >= 0:
            if source_lines[j].lstrip().startswith("/*"):
                return (j + 1, i + 1)
            j -= 1

    # No block comment - try a contiguous run of // lines instead.
    j = i
    saw_slash = False

    # Climb while lines still belong to the comment run.
    while j >= 0:
        stripped = source_lines[j].lstrip()

        if stripped.startswith("//"):
            saw_slash = True
            j -= 1
            continue

        # Blank lines and code both end the run - the comment must touch the header to count.
        if stripped == "":
            break  # blank breaks adjacency
        break

    # Only a non-empty run counts.
    if saw_slash:
        # j overshot to the line before the run, so +2 gives the 1-based first comment line.
        start_1b = j + 2
        return (start_1b, i + 1)

    return None


def file_doc_target_c(source_blob: str, source_lines: Chunk) -> Optional[FileDocTarget]:
    """
    Find the top-of-file description target zone for a C source file.

    Parameters:
    - `source_blob`: The full source text (unused; present for the shared provider signature).
    - `source_lines`: The source file as a list of lines.

    Returns:
    - A `FileDocTarget` for the leading comment zone, or `None` if no safe target was found.
    """

    return scan_brace_leading_zone(source_lines, SLASH_BLOCK_STYLE)


# ---------------- C DefInfo


@dataclass(frozen=True)
class DefInfoC:

    """
    Immutable record describing one function definition found in a C parse tree.

    Line numbers are 1-based and inclusive; `header_start`/`header_end` span the signature up to the line before the opening brace. `depth`, `parent_id` and `children_ids` mirror the shape used by the other language workers and stay at their defaults, as C functions do not nest.
    """

    # Line fields are 1-based; depth and parent/child links stay at their defaults because C functions do not nest - they exist for parity with the other language workers.
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
    Normalise an include target token for display.

    The token is stripped of surrounding whitespace; quoted and angle-bracket forms are returned verbatim, as is anything else.

    Parameters:
    - `tok`: The raw path token from a `#include` directive.

    Returns:
    - The stripped token text.
    """

    # Both branches return the stripped token unchanged - the quoted/angle-bracket test makes no difference here.
    tok = tok.strip()
    if (len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in ('"', "'")) or (tok.startswith("<") and tok.endswith(">")):
        return tok
    return tok


def _collect_includes_c(tree, source_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Collect every `#include` directive in a parsed C file as display-ready list items.

    The grammar's structured `path` field is preferred; malformed or macro-style directives fall back to regex extraction from the raw text, so every directive yields an entry.

    Parameters:
    - `tree`: The tree-sitter parse tree for the file.
    - `source_bytes`: The raw source bytes the tree was parsed from.

    Returns:
    - A list of `(line, text)` tuples in source order, where `line` is 1-based and `text` is a Markdown list item naming the include target.
    """

    out: List[Tuple[int, str]] = []
    root = tree.root_node

    def add(n, payload: str) -> None:
        """
        Append one include entry tagged with the node's 1-based source line.

        Parameters:
        - `n`: The tree-sitter node whose start line tags the entry.
        - `payload`: The display text to record.
        """

        ln = _to_1based(_row_of(n.start_point))
        out.append((ln, payload))

    stack = [root]

    # Iterative depth-first walk over the parse tree.
    while stack:
        n = stack.pop()
        t = n.type

        # An include directive: prefer the grammar's structured path field.
        if t == "preproc_include":
            path_node = n.child_by_field_name("path")

            # Take the parsed path when present, else fall back to regexes over the raw directive text.
            if path_node is not None:
                raw = _node_text(source_bytes, path_node).strip()
                target = _display_include_target(raw)
                add(n, f"- {target}")
            else:
                full = _node_text(source_bytes, n)
                m = re.search(r"<([^>]+)>", full)

                # Angle-bracket form first, then the quoted form.
                if m:
                    add(n, f"- <{m.group(1)}>")
                else:
                    m = re.search(r'"([^"\n\r]+)"', full)

                    # Last resorts: quoted form, then a macro-name include named by its identifier - every directive yields an entry.
                    if m:
                        add(n, f'- "{m.group(1)}"')
                    else:
                        m = re.search(r"#\s*include\s+([A-Za-z_]\w*)", full)
                        target = m.group(1) if m else "<unknown>"
                        add(n, f"- Includes {target}")

            # Includes cannot nest, so skip the node's children.
            continue

        # Push children reversed so the stack pops them in source order.
        for i in range(n.named_child_count - 1, -1, -1):
            stack.append(n.named_child(i))

    # Guarantee source-line order whatever the traversal produced.
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
    Prime the conversation with the file's `#include` list.

    The includes are gathered and appended to the chat history as a user turn answered by a canned acknowledgement, giving later generation turns sight of the file's dependencies. A file with no includes leaves the history untouched.

    Parameters:
    - `llm`: The local chat model (unused; present for the shared priming signature).
    - `cfg`: The generation configuration (unused; present for the shared priming signature).
    - `messages`: The chat history to receive the priming exchange.
    - `tree`: The tree-sitter parse tree for the file.
    - `source_bytes`: The raw source bytes the tree was parsed from.
    """

    # Prime the chat with the include list and a canned acknowledgement - context for later turns, no generation happens here.
    items = _collect_includes_c(tree, source_bytes)
    if not items:
        return
    lines = [text for _, text in items]
    payload = "\n".join(lines)
    echo(f"\n[C] Includes...\n{payload}")
    prompt = (
        "For additional context, here is a list of includes within this program:\n\n"
        f"{payload}"
    )
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": PRIMING_ACK})


# ---------------- Collect function definitions (ignores forward declarations)


def iter_defs_with_info_c(tree, source_bytes: bytes) -> List[DefInfoC]:
    """
    Scan a parsed C file and return a `DefInfoC` record for every function definition.

    Names are recovered by digging through the declarator to the underlying identifier, and header spans run to the line before the opening brace. Records are flat (depth 0, no parent links) since C functions do not nest.

    Parameters:
    - `tree`: The tree-sitter parse tree for the file.
    - `source_bytes`: The raw source bytes the tree was parsed from.

    Returns:
    - A list of `DefInfoC` records sorted by start line, then end line.
    """

    root = tree.root_node
    results: List[DefInfoC] = []

    # Header spans the signature up to the line before the opening brace; a bodiless definition collapses to its first line.
    def header_span_for_function(fn_node) -> Tuple[int, int]:
        """
        Compute the 1-based line span of a C function definition's header.

        The header runs from the definition's first line up to the line before the one on which the compound body opens; definitions without a parsable body degrade to a single-line span.

        Parameters:
        - `fn_node`: The tree-sitter `function_definition` node to measure.

        Returns:
        - A `(start, end)` tuple of 1-based inclusive line numbers covering the header.
        """

        # The body's start is what marks where the header text ends.
        body = fn_node.child_by_field_name("body")

        # With a real body, the header is every line before the one holding the opening brace.
        if body and body.type == "compound_statement":
            start, _ = _line_span_from_node(fn_node)
            header_end = _to_1based(_row_of(body.start_point)) - 1

            return start, header_end

        # No compound body (malformed or macro-mangled definition): fall back to the node's own first line.
        s, _e = _line_span_from_node(fn_node)

        return s, s

    # Dig through nested declarators (pointers, parameter lists) to the underlying identifier.
    def function_name(fn_node) -> str:
        """
        Extract a C function's name from its definition node.

        The identifier can be nested arbitrarily deep inside the declarator (pointer returns, parameter lists), so the declarator subtree is searched depth-first.

        Parameters:
        - `fn_node`: The tree-sitter `function_definition` node to name.

        Returns:
        - The function's identifier text, or `"<anonymous>"` when no identifier exists.
        """

        # The name hides somewhere inside the declarator subtree, so a missing declarator means no name at all.
        decl = fn_node.child_by_field_name("declarator")
        if decl is None:
            return "<anonymous>"
        stack = [decl]

        # Depth-first search; children are pushed in reverse so the leftmost identifier wins.
        while stack:
            n = stack.pop()
            if n.type == "identifier":
                return source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
            for i in range(n.named_child_count - 1, -1, -1):
                stack.append(n.named_child(i))

        # Declarator with no identifier anywhere (e.g. abstract): degrade rather than fail.
        return "<anonymous>"

    # Record every function flat - depth 0, no parent links - since C functions do not nest.
    def walk(n) -> None:
        """
        Recursively collect every C function definition beneath a syntax node.

        Each match is appended to the enclosing `results` list as a flat entry - C has no routine nesting, so depth and parentage are constant.

        Parameters:
        - `n`: The tree-sitter node at which to start the walk.
        """

        # Every definition is recorded flat: C has no nesting, so depth and parent are fixed.
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

        # Keep descending regardless, so definitions inside linkage or preprocessor wrappers are still found.
        for i in range(n.named_child_count):
            walk(n.named_child(i))

    walk(root)

    # Sorting guarantees source order whatever the walk produced.
    return sorted(results, key=lambda d: (d.start, d.end))


def _decl_function_declarator(decl_node):
    """
    Find the `function_declarator` inside a C declaration, if it has one.

    Pointer wrappers are peeled off on the way down, so prototypes for functions returning pointers are still recognised; any other declarator shape means the declaration is not a function prototype.

    Parameters:
    - `decl_node`: The tree-sitter `declaration` node to inspect.

    Returns:
    - The `function_declarator` node, or `None` when the declaration does not declare a function.
    """

    n = decl_node.child_by_field_name("declarator")

    # Unwrap until the parameter-list shape surfaces - that is what marks a function prototype.
    while n is not None:
        if n.type == "function_declarator":
            return n

        # Pointer wrappers (functions returning pointers) just add a layer: peel and keep looking.
        if n.type == "pointer_declarator":
            n = n.child_by_field_name("declarator")
            continue

        # Any other shape (variable, array, etc.) means this declaration is not a function.
        return None

    return None


def _prototype_name(decl_node, source_bytes: bytes) -> Optional[str]:
    """
    Extract the function name from a C prototype declaration.

    Pointer layers around the identifier are peeled off, but anything that does not bottom out at a plain identifier (function pointers, abstract declarators) is rejected.

    Parameters:
    - `decl_node`: The tree-sitter `declaration` node to inspect.
    - `source_bytes`: The raw source bytes the tree was parsed from.

    Returns:
    - The prototype's identifier text, or `None` when the declaration is not a plain named function prototype.
    """

    # Only plain named prototypes qualify: pointer layers are peeled, but function pointers and abstract declarators are turned away.
    fd = _decl_function_declarator(decl_node)
    if fd is None:
        return None
    inner = fd.child_by_field_name("declarator")
    while inner is not None and inner.type == "pointer_declarator":
        inner = inner.child_by_field_name("declarator")
    if inner is None or inner.type != "identifier":
        return None   # function pointer / abstract / unexpected -> not a prototype we redirect to
    return _node_text(source_bytes, inner)


def iter_decls_with_info_c(tree, source_bytes: bytes) -> List[DefInfoC]:
    """
    Collect every function prototype declaration in a parsed C file.

    Function bodies are never descended into, so local variable declarations cannot masquerade as prototypes; only declarations that bottom out at a plain named function declarator are kept.

    Parameters:
    - `tree`: The tree-sitter parse tree for the file.
    - `source_bytes`: The raw source bytes the tree was parsed from.

    Returns:
    - A list of `DefInfoC` records of kind `"declaration"`, sorted into source order.
    """

    results: List[DefInfoC] = []

    # Each named prototype becomes a flat entry whose header spans the whole declaration.
    def walk(n) -> None:
        """
        Recursively collect function prototype declarations beneath a syntax node.

        Function bodies are pruned outright so local variable declarations are never mistaken for prototypes; matches are appended to the enclosing `results` list.

        Parameters:
        - `n`: The tree-sitter node at which to start the walk.
        """

        if n.type in ("function_definition", "compound_statement"):
            return  # do not descend into bodies: their local variable declarations are not prototypes

        # A name comes back only for genuine function prototypes; everything else is ignored.
        if n.type == "declaration":
            name = _prototype_name(n, source_bytes)

            # A prototype is a single statement, so its header span is simply the whole declaration.
            if name:
                s, e = _line_span_from_node(n)
                results.append(DefInfoC(
                    qualname=name, node=n, kind="declaration",
                    start=s, end=e, header_start=s, header_end=e,
                    depth=0, parent_id=None, children_ids=tuple(),
                ))

            return  # a declaration has no nested prototypes

        # Descend through wrappers (linkage specs, preprocessor groups) that may still hold prototypes.
        for i in range(n.named_child_count):
            walk(n.named_child(i))

    walk(tree.root_node)

    # Sort by position so callers see prototypes in source order, whatever order the walk found them.
    return sorted(results, key=lambda d: (d.start, d.end))


def _collect_calls_c(node, source_bytes: bytes) -> List[Tuple[str, str]]:
    """
    Gather the direct function calls made inside a C syntax subtree.

    Only calls through a bare identifier are recorded - calls via function pointers or struct members cannot be resolved to a name. Each hit carries the callee name, the kind tag `"free"`, and its 1-based line number.

    Parameters:
    - `node`: The tree-sitter node whose subtree is scanned.
    - `source_bytes`: The raw source bytes the tree was parsed from.

    Returns:
    - A list of `(name, kind, line)` tuples, one per direct call.
    """

    # Each hit is recorded as (name, kind, 1-based line); an explicit stack keeps the walk iterative.
    calls: List[Tuple[str, str, int]] = []
    stack = [node]

    while stack:
        n = stack.pop()

        # Only direct calls through a bare identifier count - pointer and member calls cannot be named.
        if n.type == "call_expression":
            fn = n.child_by_field_name("function")
            if fn is not None and fn.type == "identifier":
                calls.append((_node_text(source_bytes, fn), "free", n.start_point[0] + 1))

        for i in range(n.named_child_count):
            stack.append(n.named_child(i))

    return calls


def iter_symbols(source_blob: str, source_lines: Chunk) -> List[Symbol]:
    """
    Extract every function definition and bare declaration from C source as `Symbol` records.

    Definitions are collected first, complete with signature text, any documentation block sitting above the header, and the names they call. Declarations (prototypes) are then added only for names with no in-file definition, so each symbol appears exactly once. Unparseable source yields an empty list rather than an error.

    Parameters:
    - `source_blob`: The full C source text to parse.
    - `source_lines`: The source split into lines, used to recover signature text.

    Returns:
    - A list of `Symbol` records, empty when the source cannot be parsed.
    """

    # Unparseable source means nothing to document: report no symbols rather than raising.
    try:
        tree, source_bytes = _parse_c(source_blob)
    except Exception:
        return []

    # Track definition names so prototypes of already-defined functions can be skipped below.
    symbols: List[Symbol] = []
    seen_defs: set = set()

    # Definitions carry the full record: signature text, any doc above the header, and the calls made inside.
    for d in iter_defs_with_info_c(tree, source_bytes):
        signature = "\n".join(source_lines[d.header_start - 1:d.header_end]) if d.header_end >= d.header_start else ""
        seen_defs.add(d.qualname)
        symbols.append(Symbol(
            qualname=d.qualname, kind=d.kind, signature=signature, start=d.start, end=d.end, depth=d.depth,
            parent_qualname=None, existing_doc=_doc_above_header(source_lines, d.header_start),
            calls=_collect_calls_c(d.node, source_bytes),
        ))

    # Prototypes are added only when the file holds no matching definition, and never carry call lists.
    for d in iter_decls_with_info_c(tree, source_bytes):
        if d.qualname in seen_defs:
            continue   # a definition in the same file already carries the symbol/contract for this name
        signature = "\n".join(source_lines[d.header_start - 1:d.header_end]) if d.header_end >= d.header_start else ""
        symbols.append(Symbol(
            qualname=d.qualname, kind="declaration", signature=signature, start=d.start, end=d.end, depth=d.depth,
            parent_qualname=None, existing_doc=_doc_above_header(source_lines, d.header_start),
            calls=[],
        ))

    return symbols


# ---------------- Within-function block targets (the `-b` block pass)


# Statements that open a brace block whose source-line span gates a paragraph break before/after it. C has no
# nested functions, so there is no "after a def" rule and no opaque-definition handling; a bare `{ ... }` scope
# block is treated as a block too.
_C_BLOCK_OPENERS = {"if_statement", "for_statement", "while_statement", "do_statement", "switch_statement",
                    "compound_statement"}


def _named_children(n) -> List:
    """
    Return a node's named tree-sitter children as a list.
    """

    return [n.named_child(i) for i in range(n.named_child_count)]


def _is_c_statement(n) -> bool:
    """
    Report whether a tree-sitter node is a C statement; declarations count as statements so they take part in segmentation.
    """

    return n.type == "declaration" or n.type.endswith("_statement")


def _c_suite(node) -> Optional[List]:
    """
    Normalise a C body node into its list of statement children.

    Covers the shapes tree-sitter produces for control-flow bodies: an `else` clause is unwrapped one level (handling both a braced block and a bare `else if`), a compound statement yields its statement children, and a brace-less single statement becomes a one-entry list.

    Parameters:
    - `node`: The tree-sitter node to normalise, or `None`.

    Returns:
    - A non-empty list of statement nodes, or `None` when there are no statements.
    """

    if node is None:
        return None

    # An `else` clause is only a wrapper: its single child holds the real body.
    if node.type == "else_clause":
        # Recursing one level covers both a braced block and a bare `else if`.
        kids = _named_children(node)
        return _c_suite(kids[0]) if kids else None

    # A braced block contributes its statement children.
    if node.type == "compound_statement":
        # An empty block reads as no suite at all (`None`), never an empty list.
        stmts = [c for c in _named_children(node) if _is_c_statement(c)]
        return stmts or None

    # A brace-less body is a single statement, still returned as a one-entry suite.
    if _is_c_statement(node):
        return [node]
    return None


def _c_statement_lists(stmt) -> List[List]:
    """
    Collect the statement suites nested directly inside one C statement.

    Each control-flow construct contributes its bodies: both branches of an `if`, the body of a loop, one list per `case` of a `switch`, and the contents of a bare braced block.

    Parameters:
    - `stmt`: The tree-sitter statement node to inspect.

    Returns:
    - A list of statement lists, one per non-empty nested suite; empty when the statement nests nothing.
    """

    t = stmt.type
    lists: List[List] = []

    # An `if` contributes both branches; a loop contributes its single body.
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

        # A `switch` yields one suite per case so each case body segments on its own; a bare braced block is itself a suite.
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
    """
    Return the number of source lines a node spans, counting both end lines.
    """

    return _row_of(node.end_point) - _row_of(node.start_point) + 1


def _c_closed_block(prev_node, resume_start: int, body_node) -> Optional[object]:
    """
    Find the outermost block construct that closed before the segmenter's resume line.

    Climbs the ancestor chain from the previous statement towards the function body; the highest block-opening ancestor that ends before `resume_start` wins, telling the caller how large a block has just been stepped out of.

    Parameters:
    - `prev_node`: The statement node preceding the resume point.
    - `resume_start`: The 1-based source line where processing resumes.
    - `body_node`: The function-body node that bounds the climb.

    Returns:
    - The outermost closed block node, or `None` if no enclosing block closed.
    """

    best = None
    n = prev_node.parent

    # Keep overwriting `best` while climbing so the outermost qualifying block wins.
    while n is not None and n is not body_node:
        if n.type in _C_BLOCK_OPENERS and _to_1based(_row_of(n.end_point)) < resume_start:
            best = n
        n = n.parent

    return best


def _collect_body_c(body_node, source_lines: Chunk) -> Tuple[List[int], Dict[int, str], List[SegStatement]]:
    """
    Walk a C function body and gather everything the structural segmenter needs.

    A recursive walk records every statement with its nesting depth, pairs a lone trailing `return` with the statement just above it so the two segment together, and forces a paragraph break after a long leading run of declarations. The recorded lines are then filtered to safe comment-insertion points (lines that start exactly one statement, preceded only by indentation) and each statement is turned into a `SegStatement` carrying its span, any block it opens, and the size of any block that has just closed before it.

    Parameters:
    - `body_node`: The tree-sitter compound-statement node for the function body.
    - `source_lines`: The full source as a list of lines, used for indentation checks.

    Returns:
    - A tuple of sorted boundary line numbers, a map from boundary line to its indentation string, and the `SegStatement` list in line order.
    """

    # State the recursive walk fills in: per-line tallies plus the merge and forced-break special cases.
    line_count: Dict[int, int] = {}
    line_col: Dict[int, int] = {}
    recorded: List[Tuple[int, int, object, bool]] = []  # (start, depth, node, first_in_scope)
    merge_map: Dict[int, int] = {}                       # return line -> anchor (preceding statement) line
    force_break_lines: set = set()                       # first real statement after a leading declaration run

    # The walk tags every statement with depth and scope, pairs a lone trailing return with its anchor, and forces a break after a leading declaration run.
    def walk(stmts: List, is_top: bool, depth: int) -> None:
        """
        Recursively record segmentation bookkeeping for one C statement list.

        Notes a work-then-return pair so the two merge into one paragraph, forces a break after a sufficiently long run of leading declarations, and tallies where each statement starts before descending into nested suites.

        Parameters:
        - `stmts`: The statement nodes at this nesting level.
        - `is_top`: `True` when this list is the function's top-level body.
        - `depth`: The current nesting depth, recorded alongside each statement.
        """

        # A short work-then-return body reads as one thought, so map the return's line back to merge the pair into a single paragraph.
        if (len(stmts) == 2 and stmts[1].type == "return_statement"
                and stmts[0].type not in _C_BLOCK_OPENERS):
            a = _to_1based(_row_of(stmts[0].start_point))
            r = _to_1based(_row_of(stmts[1].start_point))
            if a != r:
                merge_map[r] = a

        # A long enough run of leading declarations is a prologue in its own right: force a break at the first real statement (unless that is just the return).
        ndecl = 0
        while ndecl < len(stmts) and stmts[ndecl].type == "declaration":
            ndecl += 1
        if ndecl >= SEG_MIN_LEADING_DECLS and ndecl < len(stmts) and stmts[ndecl].type != "return_statement":
            force_break_lines.add(_to_1based(_row_of(stmts[ndecl].start_point)))

        # Catalogue every statement so the segmenter knows which lines genuinely begin one.
        for idx, stmt in enumerate(stmts):
            skip = (not is_top) and idx == 0  # the first line of an inner suite is never a block start
            start = _to_1based(_row_of(stmt.start_point))

            # Tally statement starts and columns per line so the segmenter can spot lines hosting more than one statement.
            if not skip:
                line_count[start] = line_count.get(start, 0) + 1
                line_col[start] = _col_of(stmt.start_point)

            # Remember the statement with its depth, then descend into any nested statement lists.
            recorded.append((start, depth, stmt, is_top and idx == 0))
            for sub in _c_statement_lists(stmt):
                walk(sub, False, depth + 1)

    top_stmts = [c for c in _named_children(body_node) if _is_c_statement(c)]
    walk(top_stmts, True, 0)
    boundary_lines: List[int] = []
    indent_of: Dict[int, str] = {}

    # Only lines that start exactly one statement, preceded by nothing but indentation, are safe comment-insertion points.
    for line, count in line_count.items():
        if count != 1 or not (1 <= line <= len(source_lines)):
            continue
        text = source_lines[line - 1]
        if text[: line_col[line]].strip() != "":
            continue
        indent_of[line] = text[: len(text) - len(text.lstrip())]
        boundary_lines.append(line)

    # Return-merge anchors must always be boundaries, even when the count filter rejected them.
    for anchor in set(merge_map.values()):
        if 1 <= anchor <= len(source_lines) and anchor not in indent_of:
            text = source_lines[anchor - 1]
            indent_of[anchor] = text[: len(text) - len(text.lstrip())]
            boundary_lines.append(anchor)

    boundary_lines.sort()
    recorded.sort(key=lambda r: r[0])
    seg_statements: List[SegStatement] = []

    # Second pass: turn each recorded node into the segmenter's per-statement facts.
    for i, (start, depth, node, first_in_scope) in enumerate(recorded):
        end = _to_1based(_row_of(node.end_point))
        opens = _c_span(node) if node.type in _C_BLOCK_OPENERS else 0
        closed = 0

        # A drop in depth means a block just closed; measure it so the segmenter can weigh the dedent.
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
    Return the prose of any comment block sitting immediately above a C function header.

    Both comment styles are handled: a `/* */` block yields its first comment body, while a run of `//` lines is stripped of its markers and rejoined.

    Parameters:
    - `source_lines`: The file's source lines.
    - `header_start`: 1-based line number of the function header.

    Returns:
    - The comment text with its delimiters stripped, or an empty string if no comment precedes the header.
    """

    # Cope with both comment styles: take the first /* */ body, or strip the // markers and rejoin the lines as prose.
    rng = _scan_existing_comment_block_above(source_lines, header_start)
    if rng is None:
        return ""
    s, e = rng
    block = "\n".join(source_lines[s - 1:e])
    if "/*" in block:
        return _extract_first_c_comment_block(block)
    return "\n".join(ln.strip()[2:].strip() if ln.strip().startswith("//") else ln.strip()
                     for ln in block.split("\n")).strip()


def iter_block_targets_c(
    source_blob: str,
    source_lines: Chunk,
    doc_override: Optional[Callable[[str], Optional[str]]] = None,
) -> List[BlockTarget]:
    """
    Build the block-pass targets for every function in a C source file.

    Each function with a compound body is structurally segmented; its doc is taken from `doc_override` when that supplies one, otherwise from any comment block above the header.

    Parameters:
    - `source_blob`: The full source text.
    - `source_lines`: The source split into lines.
    - `doc_override`: Optional callback mapping a qualname to a replacement doc; may return `None` to fall back.

    Returns:
    - A list of `BlockTarget` records, one per function with a body.
    """

    tree, source_bytes = _parse_c(source_blob)
    targets: List[BlockTarget] = []

    # An override doc takes precedence; otherwise fall back to whatever comment already sits above the header.
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
        doc = doc_override(info.qualname) if doc_override is not None else None
        if not doc:
            doc = _doc_above_header(source_lines, info.header_start)
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
                doc=doc,
                sig=routine_text_hash(_get_text_for_lines(source_lines, info.header_start, info.end)),
                segments=segments,
            )
        )

    return targets


# ---------------- Snippet assembly

def assemble_snippet_for_c(source_lines: Chunk, info: DefInfoC) -> str:
    """
    Reassemble a C function's full source snippet from its header and body line ranges.

    Parameters:
    - `source_lines`: The file's source lines.
    - `info`: The definition info giving the header and body line extents.

    Returns:
    - The header and body joined as a single string.
    """

    header_text = _get_text_for_lines(source_lines, info.header_start, info.header_end)
    body_text = _get_text_for_lines(source_lines, info.header_end + 1, info.end)
    parts: List[str] = [header_text]

    # Insert the joining newline only when the header lacks its own, so the reassembled snippet stays faithful to the source.
    if body_text:
        if header_text and not header_text.endswith("\n"):
            parts.append("\n")
        parts.append(body_text)

    return "".join(parts)


# ---------------- Header/implementation documentation site (the `--doc-site` plan)
#
# C's convention is to document an extern function where it is *declared* (the `.h` prototype), not where it is
# *defined* (the `.c` body). SCALE documents the definition, so when a project's headers and sources are annotated
# together the contract lands in the wrong place. This model-free planner decides, per function and across the whole
# run, where its doc should live: the prose is still generated from the definition's body (the model needs the body),
# but it can be *placed* above the prototype and the impl's own docstring *skipped*. The impl's block pass still runs,
# informed by the header doc (see `iter_block_targets_c`'s `doc_override`). Confident-only, mirroring the call graph:
# a name is only redirected when it has exactly one definition in the run (so `static` dupes across `.c`s are never
# touched). Only a *target* header is ever written; a read-only `--reference` file can still supply the body for prose.


@dataclass
class CFileDocPlan:

    """
    A single file's view of the run-wide C doc-site plan.

    `skip` and `header_names` hold this file's slices; snippet lookups and header-doc recording delegate to the shared `CDocPlan`, so anything recorded here is visible across the run.
    """

    # The sets are this file's slice of the run-wide plan; lookups go through the shared CDocPlan underneath.
    skip: Set[str]
    header_names: Set[str]
    _plan: "CDocPlan"

    def impl_snippet(self, name: str) -> Optional[str]:
        """
        Return the implementation snippet for a function via the shared plan.

        Parameters:
        - `name`: The function name to look up.

        Returns:
        - The implementation source snippet, or `None` if no body is recorded.
        """

        return self._plan.impl_snippet(name)

    def record_header_doc(self, name: str, doc: str) -> None:
        """
        Record a header doc for a function in the shared plan.

        Parameters:
        - `name`: The function name.
        - `doc`: The doc text to record.
        """

        self._plan.record_header_doc(name, doc)


@dataclass
class CDocPlan:

    """
    The run-wide plan for the C header/implementation doc-site.

    The skip, header-name and body maps are keyed per file, recording which names each file omits, which it declares, and where each implementation lives; `header_docs` accumulates the docs written during the run so headers and implementations can share them.
    """

    # The work maps are keyed per file, while header_docs accumulates run-wide as docs are written.
    skip: Dict[str, Set[str]]
    header_names: Dict[str, Set[str]]
    impl_body: Dict[str, Tuple[Chunk, DefInfoC]]
    impl_file: Dict[str, str]
    pairs: List[Tuple[str, str]]
    header_docs: Dict[str, str] = field(default_factory=dict)

    # Each file gets its own slice but keeps delegating to this shared plan, so recorded docs stay visible run-wide.
    def for_file(self, file_key: str) -> CFileDocPlan:
        """
        Build the per-file view of this plan for one run file.

        Parameters:
        - `file_key`: The run key of the file whose slice is wanted.

        Returns:
        - A `CFileDocPlan` carrying that file's skip and header-name sets; files the plan does not mention get empty sets.
        """

        # Unknown file keys collapse to empty sets, so files outside the plan get a harmless no-op view; `_plan` links back for shared lookups.
        return CFileDocPlan(
            skip=self.skip.get(file_key, set()),
            header_names=self.header_names.get(file_key, set()),
            _plan=self,
        )

    # Snippets are rebuilt on demand from the stored lines rather than kept as pre-rendered text.
    def impl_snippet(self, name: str) -> Optional[str]:
        """
        Return the assembled source snippet for a function's recorded implementation body.

        Parameters:
        - `name`: The function name to look up.

        Returns:
        - The snippet text, or `None` when the plan recorded no unique implementation body for the name.
        """

        # Only names with a single unique definition in the run were recorded, so a miss is expected for ambiguous names.
        body = self.impl_body.get(name)
        if body is None:
            return None
        lines, info = body

        return assemble_snippet_for_c(lines, info)

    def record_header_doc(self, name: str, doc: str) -> None:
        """
        Store the generated header documentation text for a function.

        Parameters:
        - `name`: The function name the documentation belongs to.
        - `doc`: The documentation text to keep for later splicing at the declaration site.
        """

        self.header_docs[name] = doc

    def header_doc(self, name: str) -> Optional[str]:
        """
        Return the stored header documentation for a function.

        Parameters:
        - `name`: The function name to look up.

        Returns:
        - The documentation text, or `None` when none has been recorded for the name.
        """

        return self.header_docs.get(name)

    # Only the skip and header-name sets constitute pending work; recorded docs alone mean nothing is left to do.
    def has_work(self) -> bool:
        """
        Report whether this plan redirects any documentation sites.

        Returns:
        - `True` if any definition docstrings are to be skipped or any header declarations need documenting, otherwise `False`.
        """

        return bool(self.skip or self.header_names)


def plan_doc_sites_c(files: List[Tuple[str, bool, str, Chunk]], policy: str = "auto") -> CDocPlan:
    """
    Decide where each C function's documentation should live across a multi-file run.

    Every definition and every target-file declaration in the run is indexed first. Under the `auto` policy, a name declared in a target file and defined exactly once anywhere in the run has its documentation redirected to the declaration (header) site, with the prose generated from the implementation body and the definition's own docstring suppressed. Names with several definitions are documented at every site instead, with a warning.

    Parameters:
    - `files`: The run's files as `(file_key, is_target, blob, lines)` tuples; non-target files still contribute definitions.
    - `policy`: Doc-site policy; only `"auto"` enables header redirection.

    Returns:
    - A `CDocPlan` holding the per-file skip sets, header names, captured implementation bodies, implementation files, and header-before-implementation ordering pairs.
    """

    # Run-wide indexes: definitions may live anywhere, but only target files contribute declaration sites.
    target_keys = {fk for fk, is_target, _blob, _lines in files if is_target}
    defs_by_name: Dict[str, List[Tuple[str, Chunk, DefInfoC]]] = {}
    target_decls: Dict[str, List[str]] = {}   # name -> target file keys that declare it (a prototype site)

    # A file that fails to parse simply contributes nothing rather than aborting the whole plan.
    for file_key, is_target, blob, lines in files:
        try:
            tree, sb = _parse_c(blob)
        except Exception:
            continue

        # Definitions from non-target files still count towards the unique-definition test.
        for d in iter_defs_with_info_c(tree, sb):
            defs_by_name.setdefault(d.qualname, []).append((file_key, lines, d))

        # Only declarations in target files become candidate header documentation sites.
        if is_target:
            for d in iter_decls_with_info_c(tree, sb):
                target_decls.setdefault(d.qualname, []).append(file_key)

    skip: Dict[str, Set[str]] = {}
    header_names: Dict[str, Set[str]] = {}
    impl_body: Dict[str, Tuple[Chunk, DefInfoC]] = {}
    impl_file: Dict[str, str] = {}
    pairs: List[Tuple[str, str]] = []

    # A name defined more than once cannot be redirected safely, so each site keeps its own documentation (with a warning).
    for name, decl_files in target_decls.items():
        defs = defs_by_name.get(name, [])
        unique_def = defs[0] if len(defs) == 1 else None
        if unique_def is None and len(defs) >= 2:
            echo(f"doc-site: '{name}' has {len(defs)} definitions across the run; documenting at each (ambiguous).")
        for dfile in set(decl_files):
            header_names.setdefault(dfile, set()).add(name)

        # Under `auto`, the header prose will be generated from the one unique implementation body.
        if policy == "auto" and unique_def is not None:
            def_file, def_lines, def_info = unique_def
            impl_body[name] = (def_lines, def_info)   # header prose is generated from the body
            impl_file[name] = def_file

            # Suppress the definition's own docstring only when its file is in the run, so the documentation is never written twice.
            if def_file in target_keys:
                skip.setdefault(def_file, set()).add(name)   # the redirected definition's docstring is skipped

                # Record the ordering constraint: a header must be documented before its implementation's block pass runs.
                for dfile in set(decl_files):
                    if dfile != def_file:
                        pairs.append((dfile, def_file))      # header documented before the impl's block pass

    return CDocPlan(skip=skip, header_names=header_names, impl_body=impl_body, impl_file=impl_file, pairs=pairs)


# ---------------- LLM exchange

def _render_c_block_comment(text: str, base_indent: str) -> List[str]:
    """
    Render documentation text as a C block comment at the given indentation.

    Parameters:
    - `text`: The prose to wrap; may be empty.
    - `base_indent`: Whitespace prefix applied to every emitted line.

    Returns:
    - The comment as a list of lines from the opening `/*` to the closing `*/`, with a placeholder body when `text` is empty.
    """

    lines = text.splitlines()
    out = [f"{base_indent}/*"]

    # Blank prose lines are trimmed to a bare ` *`, and empty text still yields a visible placeholder body.
    if lines:
        for ln in lines:
            out.append(f"{base_indent} * {ln.rstrip()}".rstrip())
    else:
        out.append(f"{base_indent} * (no documentation)")

    out.append(f"{base_indent} */")

    return out


def _render_c_line_comment(text: str, base_indent: str) -> List[str]:
    """
    Render documentation text as C `//` line comments at the given indentation.

    Parameters:
    - `text`: The prose to render; may be empty.
    - `base_indent`: Whitespace prefix applied to every line.

    Returns:
    - One `//` line per prose line, or a single placeholder line when `text` is empty.
    """

    # Empty text gets an explicit placeholder so the comment site never silently vanishes.
    lines = text.splitlines()
    if not lines:
        return [f"{base_indent}// (no documentation)"]
    return [f"{base_indent}// {ln.rstrip()}".rstrip() for ln in lines]


def _detect_doc_style_c(tree, source_bytes: bytes) -> str:
    """
    Detect whether a C file's comments favour line or block style.

    The file's opening banner comment is excluded from the tally, and the result defaults to `block` unless every remaining comment uses the `//` form.

    Parameters:
    - `tree`: The parsed Tree-sitter syntax tree for the file.
    - `source_bytes`: The raw source bytes the tree was parsed from.

    Returns:
    - `"line"` if all non-banner comments are line comments, otherwise `"block"`.
    """

    # Note a leading banner comment so it does not count towards the style tally.
    root = tree.root_node
    banner_start = (root.children[0].start_byte
                    if root.children and root.children[0].type == "comment" else None)
    has_line = has_block = False
    stack = [root]

    while stack:
        n = stack.pop()

        # Tally line versus block comments across the tree, ignoring the file banner.
        if n.type == "comment" and n.start_byte != banner_start:
            if source_bytes[n.start_byte:n.start_byte + 2] == b"//":
                has_line = True
            else:
                has_block = True

        for i in range(n.named_child_count):
            stack.append(n.named_child(i))

    # Default to block style unless the file uses line comments exclusively.
    return "line" if (has_line and not has_block) else "block"


def _extract_first_c_comment_block(reply: str) -> str:
    """
    Extract the body of the first C block comment from an LLM reply.

    The `/* ... */` delimiters and any leading `*` gutter are stripped so the result is bare prose; if no block comment is present, the whole reply is returned trimmed.

    Parameters:
    - `reply`: The raw model reply text.

    Returns:
    - The cleaned comment body, or the trimmed reply when no delimiters were found.
    """

    # Find the first delimited comment; the model may bury it in surrounding chatter.
    txt = textwrap.dedent(reply)
    start = txt.find("/*")
    end = txt.find("*/", start + 2) if start != -1 else -1

    # Keep only the comment's interior, discarding anything around it.
    if start != -1 and end != -1:
        inner = txt[start + 2:end]
        lines = inner.splitlines()
        cleaned = []

        for ln in lines:
            stripped = ln.lstrip()

            # Strip a decorative '*' gutter and the single space that usually follows it.
            if stripped.startswith("*"):
                stripped = stripped[1:]
                if stripped.startswith(" "):
                    stripped = stripped[1:]

            cleaned.append(stripped.rstrip())

        # Return bare prose, ready to be re-wrapped in whichever style the file uses.
        return "\n".join(cleaned).strip()

    # No delimiters found: treat the whole reply as the comment body.
    return txt.strip()


def generate_comments_c(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    defs: List[DefInfoC],
    source_blob: str,
    source_lines: Chunk,
    doc_order: Optional[List[str]] = None,
    callee_context: Optional[Callable[[str], str]] = None,
    on_doc: Optional[Callable[[str, str], None]] = None,
    doc_plan: Optional[CFileDocPlan] = None,
    decls: Optional[List[DefInfoC]] = None,
    verifier=None,
) -> Dict[Tuple[int, int], str]:
    """
    Generate a documentation comment for every C routine in a file.

    Each routine gets one bounded LLM turn (the prompt is popped straight after the reply), optionally seeded with its existing comment and callee contract notes, then passes through the verifier. Failures never drop a routine: an unverified comment is written with a warning, and a placeholder records total generation failure.

    Parameters:
    - `llm`: The local chat model used for generation.
    - `cfg`: Generation settings for each turn.
    - `messages`: The primed conversation; turns are appended and popped per routine.
    - `defs`: Function definitions discovered in the file.
    - `source_blob`: The full source text of the file.
    - `source_lines`: The pristine source lines used to assemble snippets.
    - `doc_order`: Optional leaf-first qualname ordering for generation.
    - `callee_context`: Optional callback returning contract notes for a routine's callees.
    - `on_doc`: Optional callback invoked with each qualname and doc as it is produced.
    - `doc_plan`: Optional header/impl doc-site plan; supplies implementation snippets for prototypes and the routines to skip.
    - `decls`: Header prototypes to document alongside the definitions.
    - `verifier`: Optional verifier applying the grounding gate and challenge turns.

    Returns:
    - A mapping of `(header_start, header_end)` line spans to generated comment bodies.
    """

    # Build the worklist: drop routines the doc-site plan owns, fold in header prototypes, honour any leaf-first ordering.
    doc_map: Dict[Tuple[int, int], str] = {}
    skip = doc_plan.skip if doc_plan is not None else set()
    records: List[DefInfoC] = [d for d in defs if d.qualname not in skip]
    if decls:
        records = records + list(decls)
    ordered = apply_doc_order(records, lambda d: d.qualname, doc_order, lambda d: d.start) if doc_order else records

    # Per routine: pick the snippet (the implementation body for prototypes), trim it to the context window, and frame the request.
    for info in ordered:
        is_decl = info.kind == "declaration"
        snippet = None
        if is_decl and doc_plan is not None:
            snippet = doc_plan.impl_snippet(info.qualname)
        if not snippet:
            snippet = assemble_snippet_for_c(source_lines, info)
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
        existing = _doc_above_header(source_lines, info.header_start)

        # Seed the prompt with the existing comment so accurate institutional detail survives the rewrite.
        if existing:
            prompt += (
                "\nThe function is already documented as follows. Ingest and update this existing comment - keep "
                "whatever is still accurate, correct anything stale, and reformat it to the requested style - rather "
                f"than writing from scratch:\n\n{existing}\n"
            )

        # Add callee contract notes when the call graph has any.
        if callee_context is not None:
            notes = callee_context(info.qualname)
            if notes:
                prompt += "\n" + notes + "\n"

        # One bounded turn: the prompt is popped straight after the reply so context never accumulates.
        messages.append({"role": "user", "content": prompt})
        reply = llm.generate(messages, cfg=cfg)
        echo(f"\n[C] LLM output:\n\n{reply}")
        messages.pop()
        body = _extract_first_c_comment_block(reply)

        # Box the reply in a one-element list so the retry closure can rebind it.
        if body and verifier is not None:
            last_reply = [reply]

            # Retry closure: replays the exchange plus the verifier's feedback, then unwinds all three turns.
            def regenerate(feedback: str, _prompt: str = prompt) -> str:
                """
                Regenerate the documentation comment after verifier feedback.

                Replays the original prompt and reply with the feedback appended, then pops all three turns so the shared context stays bounded. A usable retry becomes the new baseline for any further round.

                Parameters:
                - `feedback`: The verifier's complaint to feed back to the model.
                - `_prompt`: The original request, bound at definition time.

                Returns:
                - The extracted comment body, or an empty string if the retry produced none.
                """

                # Replay the original exchange plus the feedback, then pop all three turns to keep the context bounded.
                messages.append({"role": "user", "content": _prompt})
                messages.append({"role": "assistant", "content": last_reply[0]})
                messages.append({"role": "user", "content": feedback})
                retry = llm.generate(messages, cfg=cfg)
                for _ in range(3):
                    messages.pop()
                doc = _extract_first_c_comment_block(retry)
                if not doc:
                    return ""
                last_reply[0] = retry

                return doc

            # Grounding gate plus challenge turns; may substitute a regenerated comment.
            body, ok = verifier.verify_def(snippet, body, regenerate, label=info.qualname)

            # A twice-failed comment is still written - flagged for human review rather than dropped.
            if not ok:
                error(f"[verify] '{info.qualname}': comment failed verification twice; writing it anyway - "
                      f"review this comment")

        # Always record something for the span - a placeholder on total failure keeps the run complete - then feed the doc to the run model and doc-site plan.
        if not body:
            body = f"function `{info.qualname}` - documentation generation failed."
        doc_map[(info.header_start, info.header_end)] = body
        if on_doc is not None:
            on_doc(info.qualname, body)
        if is_decl and doc_plan is not None:
            doc_plan.record_header_doc(info.qualname, body)

    return doc_map


# ---------------- Textual patcher

def patch_comments_textually_c(source_lines: Chunk, defs: List[DefInfoC], doc_map: Dict[Tuple[int, int], str],
                               style: str = "block") -> Chunk:
    """
    Splice generated doc comments into the source lines above each C routine.

    Routines are patched bottom-up so insertions never invalidate the line numbers of spans still to be processed; an existing comment block above a header is replaced in place, otherwise the new block is inserted directly above it. Only comment lines are added or replaced - code lines are never touched.

    Parameters:
    - `source_lines`: The pristine source lines to patch.
    - `defs`: The routines (definitions and prototypes) whose headers anchor the comments.
    - `doc_map`: Comment bodies keyed by `(header_start, header_end)` span.
    - `style`: `"line"` or `"block"`; selects the rendering that matches the file.

    Returns:
    - A new list of lines with the comments spliced in.
    """

    # Match the file's prevailing comment style; patch a copy, never the input.
    render = _render_c_line_comment if style == "line" else _render_c_block_comment
    out_lines = source_lines[:]

    # Patch bottom-up so insertions never shift the line numbers still to be processed.
    for info in sorted(defs, key=lambda d: d.start, reverse=True):
        key = (info.header_start, info.header_end)
        if key not in doc_map:
            continue
        doc = doc_map[key].rstrip()
        header_line_text = source_lines[info.header_start - 1]
        indent = header_line_text[: len(header_line_text) - len(header_line_text.lstrip())]
        new_block_lines = render(doc, indent)
        existing = _scan_existing_comment_block_above(out_lines, info.header_start)

        # Replace an existing doc block in place; otherwise insert directly above the header.
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
    source_lines: Chunk,
    doc_order: Optional[List[str]] = None,
    callee_context: Optional[Callable[[str], str]] = None,
    on_doc: Optional[Callable[[str, str], None]] = None,
    doc_plan: Optional[CFileDocPlan] = None,
    verifier=None,
) -> Chunk:
    """
    Run the C definition pass: the shared per-language entry point.

    Parses the file with Tree-sitter, gathers definitions (plus any header prototypes redirected here by the doc-site plan), generates one doc comment per routine, and splices them in by patching - the code itself is never re-emitted.

    Parameters:
    - `llm`: The local chat model.
    - `cfg`: Generation settings.
    - `messages`: The primed conversation shared across the run.
    - `source_blob`: The full source text to parse.
    - `source_lines`: The pristine source lines used for snippets and patching.
    - `doc_order`: Optional leaf-first ordering of qualnames.
    - `callee_context`: Optional callback supplying callee contract notes.
    - `on_doc`: Optional callback invoked per generated doc.
    - `doc_plan`: Optional C header/impl doc-site plan.
    - `verifier`: Optional verifier enforcing the local quality floor.

    Returns:
    - The patched source lines with the new comments in place.
    """

    # Parse once with Tree-sitter and harvest the function definitions.
    echo("Parsing C source with Tree-sitter...")
    tree, source_bytes = _parse_c(source_blob)
    defs = iter_defs_with_info_c(tree, source_bytes)
    echo(f"Found {len(defs)} C function definition(s)")
    decls: List[DefInfoC] = []

    # The doc-site plan redirects selected prototypes here so their docs land at the header.
    if doc_plan is not None and doc_plan.header_names:
        decls = [d for d in iter_decls_with_info_c(tree, source_bytes) if d.qualname in doc_plan.header_names]
        echo(f"Documenting {len(decls)} prototype(s) at the header (doc-site redirect)")

    # Match the file's comment style, prime the model with #include context, then run the definition pass.
    style = _detect_doc_style_c(tree, source_bytes)
    echo(f"Doc-comment style for this file: {style}")
    echo("Identifying #includes...")
    describe_includes_c(llm, cfg, messages, tree, source_bytes)
    echo("Generating C comments...\n")
    doc_map = generate_comments_c(llm, cfg, messages, defs, source_blob, source_lines,
                                  doc_order=doc_order, callee_context=callee_context, on_doc=on_doc,
                                  doc_plan=doc_plan, decls=decls, verifier=verifier)
    echo("Applying C patches...\n")

    # Comments are spliced into the pristine lines by patching; code is never re-emitted.
    return patch_comments_textually_c(source_lines, defs + decls, doc_map, style=style)


# ---------------- Manifest emit (model-free)


def collect_def_requests_c(source_blob: str, source_lines: Chunk, escalation, doc_plan: Optional[CFileDocPlan] = None) -> int:
    """
    Collect every C routine in a file into the online run manifest.

    Model-free counterpart to the offline definition pass: the same skip set and doc-site redirects apply, and each routine is recorded once with its snippet and a span hash used to re-bind the answer at apply time.

    Parameters:
    - `source_blob`: The full source text to parse.
    - `source_lines`: The pristine source lines used to assemble snippets.
    - `escalation`: The manifest collector receiving one record per routine.
    - `doc_plan`: Optional doc-site plan supplying skips, header prototypes, and implementation snippets.

    Returns:
    - The number of routines recorded, used by the completeness counter.
    """

    # Mirror the offline worklist - same skip set and header redirects - so both modes cover identical routines.
    tree, source_bytes = _parse_c(source_blob)
    defs = iter_defs_with_info_c(tree, source_bytes)
    skip = doc_plan.skip if doc_plan is not None else set()
    records: List[DefInfoC] = [d for d in defs if d.qualname not in skip]
    if doc_plan is not None and doc_plan.header_names:
        records += [d for d in iter_decls_with_info_c(tree, source_bytes) if d.qualname in doc_plan.header_names]

    # Each routine crosses the wire once: its snippet plus a span hash to re-bind the answer at apply time.
    for info in records:
        snippet = None
        if info.kind == "declaration" and doc_plan is not None:
            snippet = doc_plan.impl_snippet(info.qualname)
        if not snippet:
            snippet = assemble_snippet_for_c(source_lines, info)
        span_hash = routine_text_hash(_get_text_for_lines(source_lines, info.header_start, info.end))
        escalation.record_def(qualname=info.qualname, kind=info.kind, sig_hash=span_hash, snippet=snippet)

    # The caller's completeness counter relies on this count.
    return len(records)


# ---------------- Manifest apply (model-free)


def _clean_c_comment_answer(text: str) -> str:
    """
    Normalise a manifest answer for a C routine to bare comment prose.

    Strips a wrapping Markdown code fence and, if the answer arrived as a `/* ... */` or `//` comment, removes the delimiters so only the body remains.

    Parameters:
    - `text`: The raw answer text from the manifest.

    Returns:
    - The cleaned comment body, possibly empty.
    """

    # Tolerate answers wrapped in code fences or comment delimiters; reduce them all to bare prose.
    body = (text or "").strip()
    fence = re.match(r"^```[^\n]*\n(.*)\n```$", body, flags=re.DOTALL)
    if fence:
        body = fence.group(1).strip()
    if body.startswith("/*") or body.startswith("//"):
        body = _extract_first_c_comment_block(body)
    return body.strip()


def apply_manifest_c(source_lines: Chunk, manifest: dict) -> Chunk:
    """
    Apply a filled run manifest's answers to C source, patching comments only.

    The def phase re-binds each answered request to its routine by qualified name plus a hash of the routine's current text, cleans the answer and splices the header docs textually. The block phase then inserts per-chunk comments at the recorded boundary lines, accepting a routine's edits only when the preservation guard confirms the code is otherwise byte-for-byte intact. Unanswered or unmatched requests leave their routines untouched.

    Parameters:
    - `source_lines`: The file's source as a list of lines.
    - `manifest`: The filled run-manifest dictionary for this file.

    Returns:
    - The patched list of source lines; only comments differ from the input.
    """

    # Split the manifest into def and block work; both phases patch the same evolving line list.
    from scale_blocks import SLASH_LINE_STYLE, _apply_edits, code_preserved, _parse_comment_reply
    requests = manifest.get("requests", [])
    def_reqs = [r for r in requests if r.get("def") is not None]
    block_reqs = [r for r in requests if r.get("blocks") is not None]
    out_lines = source_lines

    # Def phase: re-parse the current text and key the requests by (qualname, span hash) so answers re-bind by content, not stale line numbers.
    if def_reqs:
        tree, sb = _parse_c("\n".join(out_lines))
        records = iter_defs_with_info_c(tree, sb) + iter_decls_with_info_c(tree, sb)
        style = _detect_doc_style_c(tree, sb)
        wanted = {(r["qualname"], r["sig_hash"]): r for r in def_reqs}
        doc_map: Dict[Tuple[int, int], str] = {}
        used: set = set()
        matched: List[DefInfoC] = []

        # Re-hash each parsed routine to find its request; `used` stops one answer landing twice.
        for info in records:
            key = (info.qualname, routine_text_hash(_get_text_for_lines(out_lines, info.header_start, info.end)))
            req = wanted.get(key)
            if req is None or key in used:
                continue
            answer = req["def"].get("answer")

            # An unfilled answer leaves the routine untouched rather than inventing a doc.
            if not answer or not str(answer).strip():
                echo(f"[apply] Def request '{req['id']}' has no answer; leaving the record untouched")
                continue

            # Strip any code fences or comment delimiters the answering model may have added.
            doc = _clean_c_comment_answer(str(answer))

            # Bind the cleaned doc to its header span ready for the textual patcher.
            if doc:
                doc_map[(info.header_start, info.header_end)] = doc
                used.add(key)
                matched.append(info)

        # All matched docs land in one textual patch over the pristine lines.
        out_lines = patch_comments_textually_c(out_lines, matched, doc_map, style=style)

    # Block phase: re-locate the block targets in the now-patched text and gather edits file-wide.
    if block_reqs:
        targets = iter_block_targets_c("\n".join(out_lines), out_lines)
        by_key = {(t.qualname, t.sig): t for t in targets}
        all_edits: List[Tuple[int, Optional[str], str]] = []

        # Re-bind each block request to its routine by qualname and signature hash.
        for req in block_reqs:
            target = by_key.get((req["qualname"], req["sig_hash"]))
            chunks = req["blocks"].get("chunks", [])

            # A vanished or changed routine is skipped with a notice, never guessed at.
            if target is None:
                echo(f"[apply] No match for block request '{req['id']}'; skipping")
                continue

            # A wholly unanswered request leaves its routine untouched.
            if all(c.get("answer") is None for c in chunks):
                echo(f"[apply] Block request '{req['id']}' has no answers; leaving routine untouched")
                continue

            edits: List[Tuple[int, Optional[str], str]] = []

            # Turn each answered chunk into an insertion at its recorded boundary line, with matching indent; stale indices are dropped.
            for chunk in chunks:
                bidx = chunk["bidx"]
                if not (0 <= bidx < len(target.boundary_lines)):
                    continue
                boundary = target.boundary_lines[bidx]
                comment = _parse_comment_reply(chunk.get("answer") or "", SLASH_LINE_STYLE)
                edits.append((boundary, comment, target.indent_of.get(boundary, "")))

            # Trial-apply this routine's edits alone so a bad batch can be rejected without poisoning the rest.
            trial = _apply_edits(out_lines, edits, SLASH_LINE_STYLE)

            # The preservation guard is the gate: edits that would alter anything but comments are discarded wholesale.
            if code_preserved(out_lines, trial, SLASH_LINE_STYLE):
                all_edits.extend(edits)
            else:
                echo(f"[apply] Skipped '{req['qualname']}': block edit would alter code; keeping original")

        # One final pass lands every accepted edit at once.
        out_lines = _apply_edits(out_lines, all_edits, SLASH_LINE_STYLE)

    return out_lines
