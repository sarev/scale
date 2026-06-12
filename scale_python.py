#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The Python language worker: it parses source with the standard `ast` module, collects a `DefInfo` record for every
function, async function and class (dotted qualnames, header spans, nesting depth and parent/child links), and
supplies the per-language pieces the shared pipeline needs - the definition pass, the block-pass segmenter, the symbol
scanner, and the module-docstring file-doc target.

Docstring generation runs deepest-first so each parent's snippet can show its children as header-plus-docstring stubs,
with `elide_structurally` collapsing the deepest nested suites into model-written one-line summaries when a routine
exceeds the context budget. `patch_docstrings_textually` then splices the results in bottom-up, replacing existing
docstrings in place and never re-emitting code; signature hashes are computed over docstring-stripped ASTs, so editing
documentation never changes a routine's identity.

For the online mode, `collect_def_requests` records each routine in the run manifest and `apply_manifest` re-binds the
filled answers by qualname and signature hash, trial-applying each routine's block edits and rejecting them wholesale
if they would alter executable code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from scale_blocks import BlockTarget, SegStatement, structural_breaks
from scale_filedoc import FileDocTarget, PYTHON_DOC_STYLE
from scale_llm import LocalChatModel, GenerationConfig, Messages, Chunk
from scale_log import echo, error
from scale_project import Symbol, apply_doc_order
from scale_text import fit_snippet, summarise, LENGTH_LINE, MARKER_PYTHON, PRIMING_ACK
from typing import Callable, Dict, List, Optional, Tuple
import ast
import copy
import hashlib
import re
import textwrap


@dataclass(frozen=True)
class DefInfo:

    """
    Immutable record of one function, async function, or class definition discovered in the source.

    Line numbers are 1-based; `parent_id` and `children_ids` link nested definitions into a tree of indices.
    """

    # Header and body spans are 1-based line numbers; parent/children ids stitch nested definitions into a tree.
    qualname: str
    node: ast.AST
    start: int
    end: int
    def_line: int
    header_start: int
    header_end: int
    kind: str
    depth: int
    parent_id: Optional[int]
    children_ids: Tuple[int, ...] = field(default_factory=tuple)


def _is_def_node(n: ast.AST) -> bool:
    """
    Return whether an AST node is a function, async function, or class definition.

    Parameters:
    - `n`: The AST node to test.

    Returns:
    - `True` if the node introduces a definition, otherwise `False`.
    """

    return isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def _node_kind(n: ast.AST) -> str:
    """
    Return the source keyword that introduces a definition node.

    Parameters:
    - `n`: The definition AST node to classify.

    Returns:
    - One of `"class"`, `"async def"` or `"def"`.
    """

    if isinstance(n, ast.ClassDef):
        return "class"
    if isinstance(n, ast.AsyncFunctionDef):
        return "async def"
    return "def"


class _DocstringStripper(ast.NodeTransformer):

    """
    AST transformer that deletes the docstring from modules, classes and functions.

    Exists so `node_sig` can hash only executable structure: with docstrings stripped, editing documentation never changes a routine's signature hash.
    """

    def _strip(self, node: ast.AST) -> ast.AST:
        """
        Remove a leading docstring from `node`, then recurse so nested definitions lose theirs too.

        Parameters:
        - `node`: The AST node to strip; it is modified in place.

        Returns:
        - The same node, with docstrings removed throughout.
        """

        # Mutates the node in place and recurses, so nested defs are stripped in the same visit.
        body = getattr(node, "body", None)
        if body and _is_docstring_stmt(body[0]):
            node.body = body[1:]
        self.generic_visit(node)

        return node

    # Every docstring-bearing node type gets the same treatment, so alias rather than duplicate.
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip
    visit_Module = _strip


def node_sig(node: ast.AST) -> str:
    """
    Return a short structural signature for an AST node.

    Docstrings are stripped from a deep copy before hashing, so the signature changes only when executable structure changes - editing a docstring leaves it stable.

    Parameters:
    - `node`: The AST node to fingerprint.

    Returns:
    - The first 12 hex characters of a SHA-256 digest of the node's docstring-free dump.
    """

    # Hash a docstring-free deep copy, so documentation edits never disturb the signature.
    stripped = _DocstringStripper().visit(copy.deepcopy(node))
    return hashlib.sha256(ast.dump(stripped).encode("utf-8")).hexdigest()[:12]


def _header_span(n: ast.AST) -> Tuple[int, int]:
    """
    Return the 1-based inclusive line span of a definition's header.

    The span runs from the earliest decorator (ast's `lineno` points at the `def`/`class` keyword, not the decorators) down to the line just above the first body statement, so multi-line signatures are covered in full. An empty body falls back to the node's own end line.

    Parameters:
    - `n`: A function, async function or class definition node.

    Returns:
    - A `(start, end)` tuple of 1-based inclusive line numbers.
    """

    # ast's `lineno` is the `def`/`class` keyword's line; decorators sit above it and need separate handling.
    assert _is_def_node(n)
    start = n.lineno

    # Pull the start up to the earliest decorator so the whole header is covered.
    if getattr(n, "decorator_list", None):
        deco_starts = [d.lineno for d in n.decorator_list if hasattr(d, "lineno")]
        if deco_starts:
            start = min(deco_starts + [start])

    # End just above the first body statement - counting its decorators too, or a decorated first member would be swallowed into the header.
    if getattr(n, "body", None):
        first = n.body[0]
        first_line = first.lineno
        deco_starts = [d.lineno for d in getattr(first, "decorator_list", None) or [] if hasattr(d, "lineno")]
        if deco_starts:
            first_line = min(deco_starts + [first_line])
        end = first_line - 1
    else:
        end = getattr(n, "end_lineno", n.lineno)

    return start, end


def _property_accessor_role(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[Tuple[str, str]]:
    """
    Identify whether a function is a property accessor and which role it plays.

    Parameters:
    - `fn`: The function definition whose decorators are inspected.

    Returns:
    - A `(property_name, role)` tuple for `@x.getter`/`@x.setter`/`@x.deleter` decorated functions, or `None` if the function is not an accessor.
    """

    roles = {"getter", "setter", "deleter"}

    # Only the literal `@name.role` decorator shape counts; anything more elaborate is ignored.
    for deco in getattr(fn, "decorator_list", []) or []:
        if isinstance(deco, ast.Attribute) and isinstance(deco.value, ast.Name) and deco.attr in roles:
            return deco.value.id, deco.attr

    return None


def iter_defs_with_info(tree: ast.AST) -> List[DefInfo]:
    """
    Collect a `DefInfo` record for every function, async function and class in a parsed tree.

    Walks the tree depth-first, building dotted qualnames from the enclosing scopes (property accessors become `prop.role`), and records header/body line spans, nesting depth and parent/child links for each definition.

    Parameters:
    - `tree`: The parsed module's AST.

    Returns:
    - The `DefInfo` records, sorted by start line.
    """

    # Scope stacks give each def its dotted qualname and parent as the walk descends.
    results: List[DefInfo] = []
    children_map: Dict[int, List[int]] = {}  # parent_id -> [child_ids]
    scope_names: Chunk = []      # qualname parts
    scope_nodes: List[ast.AST] = []  # node stack (for parent tracking)

    # Parent/child links are keyed by node id so the immutable records can be completed after the walk.
    def add_child(parent_node: Optional[ast.AST], child_node: ast.AST) -> None:
        """
        Record `child_node` as a child of `parent_node` in the shared `children_map`.

        Parameters:
        - `parent_node`: The enclosing def/class node, or `None` for module-level defs, in which case nothing is recorded.
        - `child_node`: The nested def/class node to register.
        """

        # Module-level defs have no parent; entries are keyed by object identity to match `DefInfo.parent_id` later.
        if parent_node is None:
            return
        pid = id(parent_node)
        children_map.setdefault(pid, []).append(id(child_node))

    # Depth-first walk recording every def; property accessors are named `prop.role` so getter/setter pairs stay distinct.
    def walk(node: ast.AST) -> None:
        """
        Recursively record every def/class beneath `node` into the shared `results` list.

        Each definition found becomes a `DefInfo` (qualified name, span, depth, parent link) and is registered with its parent via `add_child`; the shared scope stacks are pushed and popped around the recursive descent so nested names come out fully qualified. Property accessors are named `prop.role` (e.g. `x.setter`) to keep same-named getter/setter/deleter defs distinct.

        Parameters:
        - `node`: The AST node whose children are scanned.
        """

        # Only def/class children get records; every other node is a transparent container to recurse through.
        for child in ast.iter_child_nodes(node):
            if _is_def_node(child):
                kind = _node_kind(child)
                name_part: str

                # Property accessor roles only make sense for (async) function defs; classes never qualify.
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    acc = _property_accessor_role(child)
                else:
                    acc = None

                # Accessors all share the property's name, so append the role to keep getter/setter/deleter distinct.
                if acc is not None and scope_names:
                    prop_name, role = acc  # e.g. ("analysis_sec", "setter")
                    name_part = f"{prop_name}.{role}"
                else:
                    name_part = child.name  # regular def/class name

                # Record the def, then descend with it pushed on the scope stacks; nested defs under an accessor keep the property's base name, and children_ids is patched in after traversal.
                qualname = ".".join(scope_names + [name_part]) if scope_names else name_part
                def_line = child.lineno
                start_line, header_end = _header_span(child)
                end_line = getattr(child, "end_lineno", def_line)
                parent_node = scope_nodes[-1] if scope_nodes else None
                depth = len(scope_nodes)
                info = DefInfo(
                    qualname=qualname,
                    node=child,
                    start=start_line,
                    end=end_line,
                    def_line=def_line,
                    header_start=start_line,
                    header_end=header_end,
                    kind=kind,
                    depth=depth,
                    parent_id=id(parent_node) if parent_node is not None else None,
                    children_ids=tuple(),  # filled after traversal
                )
                results.append(info)
                add_child(parent_node, child)
                scope_names.append(name_part if acc is None else prop_name)  # for nested defs, keep property base name
                scope_nodes.append(child)
                walk(child)
                scope_nodes.pop()
                scope_names.pop()
            else:
                walk(child)

    walk(tree)
    completed: List[DefInfo] = []

    # Children could not be known mid-walk, so patch each frozen record in a second pass.
    for info in results:
        child_ids = tuple(children_map.get(id(info.node), []))
        completed.append(replace(info, children_ids=child_ids))

    # Callers rely on source order, so sort by start line.
    return sorted(completed, key=lambda d: d.start)


def _collect_calls_py(node: ast.AST) -> List[Tuple[str, str, int]]:
    """
    Collect every call made directly in the body of a def/class node.

    Nested functions, classes, and lambdas are skipped, so their calls are not attributed to this routine.

    Parameters:
    - `node`: The AST node whose body statements are scanned.

    Returns:
    - A list of `(name, kind, line)` tuples, where kind is `"free"`, `"self"`, or `"method"`.
    """

    calls: List[Tuple[str, str, int]] = []

    # Recursive walker that stops at nested defs/classes/lambdas, so their calls are not attributed to this routine.
    def visit(n: ast.AST) -> None:
        """
        Recursively gather calls from `n` into the enclosing `calls` list, stopping at nested scopes.

        Each call is classified by its callee: a bare name is `"free"`, an attribute on `self` is `"self"`, and any other attribute call is `"method"`.

        Parameters:
        - `n`: The AST node to scan.
        """

        # Nested scopes own their calls; do not descend into them.
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return

        # Synthetic nodes may lack a line number; default to 0.
        if isinstance(n, ast.Call):
            f = n.func
            line = getattr(n, "lineno", 0)

            # Classify the callee: a bare name is a free call, `self.x` a method on this class, anything else an external method.
            if isinstance(f, ast.Name):
                calls.append((f.id, "free", line))
            elif isinstance(f, ast.Attribute):
                if isinstance(f.value, ast.Name) and f.value.id == "self":
                    calls.append((f.attr, "self", line))
                else:
                    calls.append((f.attr, "method", line))

        for child in ast.iter_child_nodes(n):
            visit(child)

    # Visit the body statements rather than the def node itself, which the walker would refuse as a nested scope.
    for stmt in getattr(node, "body", []):
        visit(stmt)
    return calls


def iter_symbols(source_blob: str, source_lines: Chunk) -> List[Symbol]:
    """
    Parse Python source and return a flat list of `Symbol` records for every def, async def, and class.

    Each symbol carries its verbatim header text (sliced from `source_lines`), span, depth, parent qualname, any existing docstring, and the calls made in its body.

    Parameters:
    - `source_blob`: The complete source text to parse.
    - `source_lines`: The same source split into lines, used to extract verbatim signatures.

    Returns:
    - A list of `Symbol` records, or an empty list if the source does not parse.
    """

    # Unparsable source yields an empty symbol list rather than raising.
    try:
        tree = ast.parse(source_blob)
    except SyntaxError:
        return []

    # The identity map turns each def's `parent_id` back into a parent qualname.
    defs = iter_defs_with_info(tree)
    qual_by_node_id: Dict[int, str] = {id(d.node): d.qualname for d in defs}
    symbols: List[Symbol] = []

    # Flatten each `DefInfo` into a `Symbol`; the signature is sliced verbatim from the source via the 1-based header span.
    for d in defs:
        signature = "\n".join(source_lines[d.header_start - 1:d.header_end]) if d.header_end >= d.header_start else ""
        parent_qualname = qual_by_node_id.get(d.parent_id) if d.parent_id is not None else None
        existing_doc = ast.get_docstring(d.node) or "" if _is_def_node(d.node) else ""
        symbols.append(Symbol(
            qualname=d.qualname, kind=d.kind, signature=signature, start=d.start, end=d.end, depth=d.depth,
            parent_qualname=parent_qualname, existing_doc=existing_doc, calls=_collect_calls_py(d.node),
        ))

    return symbols


def _is_module_docstring_expr(node: ast.AST) -> bool:
    """
    Report whether an AST statement is a bare string-constant expression, the form Python treats as a docstring.

    Parameters:
    - `node`: The AST statement to test.

    Returns:
    - `True` if the node is an expression statement wrapping a string constant, otherwise `False`.
    """

    # Only plain string constants qualify; bytes literals and f-strings are never docstrings.
    return (isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(node.value.value, str))


def _module_code_signature(tree: ast.AST) -> str:
    """
    Produce a canonical `ast.dump` of a module's AST with the module docstring excluded.

    Used to compare two parses for code equality while ignoring module-docstring-only edits.

    Parameters:
    - `tree`: The parsed module AST.

    Returns:
    - The dump string of the docstring-stripped module.
    """

    # Dropping the leading docstring means two trees differing only in the module docstring dump identically.
    body = list(getattr(tree, "body", []))
    if body and _is_module_docstring_expr(body[0]):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=list(getattr(tree, "type_ignores", []))))


def _py_doc_preserved(old_lines: Chunk, new_lines: Chunk, start: int, removed: int, added: int) -> bool:
    """
    Check that an edit to the source changed nothing but the module docstring.

    Lines outside the spliced window must match exactly, both versions must still parse, and their docstring-stripped AST dumps must be identical; any failure (including a syntax error) rejects the edit.

    Parameters:
    - `old_lines`: The original source lines.
    - `new_lines`: The edited source lines.
    - `start`: 0-based line index where the spliced window begins.
    - `removed`: Number of lines the edit removed at `start`.
    - `added`: Number of lines the edit inserted at `start`.

    Returns:
    - `True` if only the module docstring changed, otherwise `False`.
    """

    # Everything outside the spliced window must survive byte-for-byte; only the replaced span may differ.
    if old_lines[:start] != new_lines[:start]:
        return False
    if old_lines[start + removed:] != new_lines[start + added:]:
        return False

    # If either version no longer parses, fail safe and reject the edit.
    try:
        old_tree = ast.parse("\n".join(old_lines))
        new_tree = ast.parse("\n".join(new_lines))
    except SyntaxError:
        return False

    # Only the module docstring is exempt from the dump, so any other change makes the signatures differ.
    return _module_code_signature(old_tree) == _module_code_signature(new_tree)


def file_doc_target_py(source_blob: str, source_lines: Chunk) -> Optional[FileDocTarget]:
    """
    Locate where a Python module's top-of-file description lives or should be created.

    If the module already opens with a docstring, the returned target lists its prose lines as eligible for replacement - blank lines and any line carrying a docstring delimiter are excluded so they survive byte-for-byte - and fresh text is appended just before the closing delimiter. Without a docstring, the target requests a brand-new one above the first statement.

    Parameters:
    - `source_blob`: The complete source text, used for AST parsing.
    - `source_lines`: The source split into lines, consulted when collecting eligible docstring lines.

    Returns:
    - A `FileDocTarget` describing the update or fresh-insert site, or `None` if the source is empty or fails to parse.
    """

    # Empty or whitespace-only source has nothing worth a file description.
    if not source_lines or not any(ln.strip() for ln in source_lines):
        return None

    # Unparseable source cannot be located reliably, so decline rather than guess.
    try:
        tree = ast.parse(source_blob)
    except SyntaxError:
        return None

    # Detect whether the module already opens with a docstring.
    body = tree.body
    doc_node = body[0] if (body and _is_module_docstring_expr(body[0])) else None

    # Existing docstring: gather its prose lines so the description can be updated in place.
    if doc_node is not None:
        start_1b = doc_node.lineno
        end_1b = getattr(doc_node, "end_lineno", start_1b)
        eligible: List[Tuple[int, str, str]] = []
        append_prefix = ""

        # Only pure prose lines are eligible; blank and delimiter-bearing lines must survive byte-for-byte.
        for ln_no in range(start_1b, end_1b + 1):
            raw = source_lines[ln_no - 1]
            s = raw.strip()
            if not s:
                continue                                  # blank line - preserve
            if s in ('"""', "'''"):
                continue                                  # delimiter-only line - preserve
            if s.startswith(('"""', "'''")) or s.endswith(('"""', "'''")):
                continue                                  # opener/closer carrying content - preserve
            leading_ws = raw[: len(raw) - len(raw.lstrip())]
            eligible.append((ln_no, leading_ws, s))
            append_prefix = leading_ws

        # New text is appended just before the closing delimiter, reusing the prose indentation.
        return FileDocTarget(
            eligible=eligible,
            insert_index=end_1b - 1,                      # append before the closing delimiter line
            insert_prefix=append_prefix,
            insert_fresh=False,
            style=PYTHON_DOC_STYLE,
            indent="",
            has_zone=True,
            preserved=_py_doc_preserved,
        )

    # No docstring: a fresh one goes above the first statement, or at end-of-file when the body is empty.
    insert_index = (body[0].lineno - 1) if body else len(source_lines)

    # An empty eligible list with has_zone=False tells the splicer to create rather than update.
    return FileDocTarget(
        eligible=[], insert_index=insert_index, insert_fresh=True,
        style=PYTHON_DOC_STYLE, indent="", has_zone=False, preserved=_py_doc_preserved,
    )


def _format_from_source(module: Optional[str], level: int) -> str:
    """
    Render the source part of a `from ... import` statement as a dotted string.

    Parameters:
    - `module`: The module name, or `None` for purely relative imports.
    - `level`: The relative-import depth, rendered as that many leading dots.

    Returns:
    - The dotted source string, falling back to `"."` when both parts are empty.
    """

    # The trailing `or "."` covers the degenerate case of no dots and no module name.
    dots = "." * level
    return f"{dots}{module or ''}" or "."


class _ImportDescriber(ast.NodeVisitor):

    """
    AST visitor that collects one-line, human-readable descriptions of a module's imports.

    Each plain or `from ... import` statement is recorded as a Markdown bullet alongside its line number; `results()` returns the bullets in source order regardless of visit order.
    """

    def __init__(self) -> None:
        """
        Initialise the visitor with an empty list of `(line number, description)` pairs.
        """

        # Stored as (line number, text) pairs so `results()` can restore source order.
        self._items: List[tuple[int, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        """
        Record a Markdown bullet for every module named by a plain `import` statement.

        Parameters:
        - `node`: The `ast.Import` node; one bullet is appended per alias, noting any `as` rename.
        """

        for alias in node.names:
            if alias.asname:
                text = f"- Imports {alias.name} as {alias.asname}"
            else:
                text = f"- Imports {alias.name}"

            # The line number is kept so `results()` can emit bullets in source order.
            self._items.append((getattr(node, "lineno", 0), text))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Record a Markdown bullet for every name brought in by a `from ... import` statement.

        Star, renamed and relative imports each receive distinct phrasing.

        Parameters:
        - `node`: The `ast.ImportFrom` node; one bullet is appended per alias.
        """

        # The source (including any relative-import dots) is shared by every alias in the statement.
        src = _format_from_source(node.module, getattr(node, "level", 0) or 0)

        for alias in node.names:
            if alias.name == "*":
                text = f"- Imports everything from {src}"
            elif alias.asname:
                text = f"- Imports {alias.name} from {src} as {alias.asname}"
            else:
                text = f"- Imports {alias.name} from {src}"

            self._items.append((getattr(node, "lineno", 0), text))

    def results(self) -> List[str]:
        """
        Return the collected import descriptions in source order.

        Returns:
        - A list of Markdown bullet strings, ordered by the line number of the statement that produced each one.
        """

        # Sort by recorded line number so the bullets follow source order, however the tree was walked.
        self._items.sort(key=lambda t: t[0])
        return [text for _, text in self._items]


def describe_imports_from_tree(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    tree: ast.AST
) -> None:
    """
    Prime the conversation with a one-line description of each of the module's imports.

    The parsed tree is walked for `import` and `from ... import` statements; if any are found, the bullet list is appended to `messages` as a user turn followed by a canned assistant acknowledgement. Modules with no imports leave `messages` untouched.

    Parameters:
    - `llm`: The local chat model (unused here; present for the shared provider signature).
    - `cfg`: The generation configuration (unused here; present for the shared provider signature).
    - `messages`: The running chat history that receives the priming exchange.
    - `tree`: The parsed module AST; raises `TypeError` if it is not an `ast.AST`.
    """

    if not isinstance(tree, ast.AST):
        raise TypeError("tree must be an ast.AST")
    visitor = _ImportDescriber()
    visitor.visit(tree)
    imports = visitor.results()

    # Prime the chat with the import list as background, closing the exchange with a canned acknowledgement.
    if imports:
        imports = "\n".join(imports)
        echo(f"\n[Python] Imports...\n{imports}")
        prompt = (
            "For additional context, here is a list of imports within this program:\n\n"
            f"{imports}"
        )
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": PRIMING_ACK})


def deepest_first(defs: List[DefInfo]) -> List[DefInfo]:
    """
    Order definitions deepest-first for processing, so nested routines come before the ones that enclose them.

    Parameters:
    - `defs`: The definitions to reorder.

    Returns:
    - A new list sorted by descending nesting depth; the input list is left untouched.
    """

    # Processing children before parents lets enclosing routines see their nested defs' finished docs.
    return sorted(defs, key=lambda d: (d.depth, d.start, -d.end), reverse=True)


# ---------------------------- structural elision (for oversized routine snippets) ----------------------------


# Don't bother collapsing a body suite shorter than this (the one-line summary would barely shrink it and still costs a
# model call); and cap the total number of collapse calls so a pathological snippet can't trigger an unbounded run -
# whatever remains over budget after the cap is handled by the crude head/tail crop fallback.
STRUCTURAL_MIN_BODY_LINES = 3
STRUCTURAL_MAX_COLLAPSES = 24


def _is_elided_suite(body: List[ast.AST]) -> bool:
    """
    Test whether a statement suite is just a bare `...` placeholder.

    Used to skip suites that an earlier collapse has already replaced.

    Parameters:
    - `body`: The suite's statement list.

    Returns:
    - `True` when the suite is a single bare `Ellipsis` expression.
    """

    return (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and body[0].value.value is Ellipsis
    )


def _deepest_collapsible_suite(top: ast.AST, lines: Chunk):
    """
    Find the deepest, then longest, nested statement suite worth collapsing into a one-line summary.

    Depth beats length so the detail furthest from the routine's spine is sacrificed first. Suites shorter than `STRUCTURAL_MIN_BODY_LINES`, already-elided suites and nested definitions are never candidates.

    Parameters:
    - `top`: The parsed AST node of the routine being elided.
    - `lines`: The snippet's source lines, indexed 1-based by the AST line numbers.

    Returns:
    - A `(start, end, indent, body_text)` tuple for the chosen suite, or `None` when nothing qualifies.
    """

    best = None  # (sort_key, start, end, indent, body_text)

    # Depth-first scan: deeper suites outrank longer ones, so detail furthest from the routine's spine goes first.
    def visit(stmts: List[ast.AST], depth: int) -> None:
        """
        Recursively scan statement suites, recording the deepest (then longest) collapsible suite in the enclosing `best`.

        Nested definitions are skipped because they are already stubs in the assembled snippet.

        Parameters:
        - `stmts`: The statement list to scan.
        - `depth`: Nesting depth of `stmts`, the primary ranking term.
        """

        nonlocal best

        for stmt in stmts:
            if _is_def_node(stmt):
                continue  # nested definitions are opaque (already stubs in the assembled snippet)

            # Size each suite that is not already a collapsed placeholder; `end_lineno` can be absent, hence the fallback.
            for sub in _sub_statement_lists(stmt):
                if sub and not _is_elided_suite(sub):
                    start = sub[0].lineno
                    end = getattr(sub[-1], "end_lineno", None) or sub[-1].lineno
                    span = end - start + 1

                    # Only suites big enough to repay a summary line qualify; depth outranks span in the key.
                    if span >= STRUCTURAL_MIN_BODY_LINES and 1 <= start <= len(lines):
                        key = (depth + 1, span)

                        # Record the winner with its original indentation so the replacement `...` line aligns.
                        if best is None or key > best[0]:
                            indent = lines[start - 1][: len(lines[start - 1]) - len(lines[start - 1].lstrip())]
                            best = (key, start, end, indent, "\n".join(lines[start - 1:end]))

                # Recurse even after a hit - a deeper suite further in may still win.
                visit(sub, depth + 1)

    # Start from the routine's own body; no candidate means nothing met the size threshold.
    visit(getattr(top, "body", []), 0)
    if best is None:
        return None
    _key, start, end, indent, body_text = best

    # Drop the ranking key - callers only need the suite's location and text.
    return start, end, indent, body_text


def elide_structurally(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    snippet: str,
    header_line_count: int,
    marker: str = MARKER_PYTHON,
    safety: float = 0.9,
) -> Tuple[str, int]:
    """
    Shrink a routine snippet to fit the model's context budget by collapsing nested suites into one-line summaries.

    Each round re-parses the current text, picks the deepest collapsible suite and replaces it with an ellipsis line carrying a model-written summary, so the reader still sees what was removed. If structural collapse cannot reach the budget (or the text will not parse), a crude crop finishes the job.

    Parameters:
    - `llm`: The loaded chat model, used for token estimates and the suite summaries.
    - `cfg`: Generation settings for the summary calls.
    - `messages`: The current conversation, used to size the remaining snippet budget.
    - `snippet`: The assembled routine text to shrink.
    - `header_line_count`: Leading header lines the crop fallback must preserve.
    - `marker`: Elision marker passed to the crop fallback.
    - `safety`: Fraction of the raw budget to aim for, leaving headroom for estimate error.

    Returns:
    - A `(text, omitted)` tuple: the possibly elided snippet and the number of source lines removed.
    """

    # Take a safety margin off the token budget; most snippets already fit and pass through untouched.
    budget = int(llm.snippet_budget(messages, cfg) * safety)
    if llm.estimate_tokens(snippet) <= budget:
        return snippet, 0
    lines = snippet.split("\n")
    omitted = 0

    # Bounded collapse loop - stop as soon as the shrinking text fits.
    for _ in range(STRUCTURAL_MAX_COLLAPSES):
        if llm.estimate_tokens("\n".join(lines)) <= budget:
            break

        # Re-parse every round: the previous collapse changed the text.
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError:
            break  # can't reason structurally; let the crop fallback finish the job

        # Swap the chosen suite for a single `...` line carrying a model-written summary, so the elision stays informative.
        top = tree.body[0] if tree.body else None
        target = _deepest_collapsible_suite(top, lines) if top is not None else None
        if target is None:
            break  # nothing left worth collapsing
        start, end, indent, body_text = target
        summary = summarise(llm, cfg, body_text, LENGTH_LINE)
        summary = (summary.splitlines() or ["elided"])[0].strip() or "elided"
        lines[start - 1:end] = [f"{indent}...  # {summary}"]
        omitted += (end - start + 1) - 1

    text = "\n".join(lines)

    # Structural collapse may not reach the budget; the crude crop guarantees it.
    if llm.estimate_tokens(text) > budget:
        text, cropped = fit_snippet(llm, cfg, messages, text, header_line_count, marker)
        omitted += cropped

    return text, omitted


# Sharper re-ask used when the model's first reply is not a usable docstring (e.g. a bare "OK"). One nudge, then we
# either promote the routine to the stronger model (if a manifest is active) or fall back to a placeholder.
DOCSTRING_NUDGE = (
    "That was not a docstring. Output ONLY the docstring itself, triple-quoted, following the format you were shown - "
    "documenting every parameter and any return value. Do not reply 'OK' and do not add any text outside the docstring."
)


_ACK_EXACT = {
    "ok", "okay", "understood", "ready", "sure", "yes", "done", "got it", "acknowledged",
    PRIMING_ACK.strip().rstrip(".").lower(),   # the priming ack itself, parroted back verbatim
}


def _looks_like_ack(text: str) -> bool:
    """
    Heuristically detect a bare acknowledgement reply ("Understood.", "OK") masquerading as content.

    Parameters:
    - `text`: The candidate reply text; `None` is treated as empty.

    Returns:
    - `True` when the text is an exact stock acknowledgement, or a short single-line reply opening with one.
    """

    # The "ok,"/"ok " forms need their separator so ordinary words starting with "ok" cannot match.
    s = (text or "").strip().rstrip(".!").lower()
    if not s:
        return False
    if s in _ACK_EXACT:
        return True
    return (
        "\n" not in (text or "")
        and len(s) < 60
        and s.startswith(("understood", "acknowledged", "okay", "ok,", "ok ", "got it"))
    )


def _is_unusable_docstring(reply: str, docstring: str) -> bool:
    """
    Test whether a generation attempt produced nothing usable as a docstring.

    Parameters:
    - `reply`: The model's full reply text.
    - `docstring`: The docstring extracted from that reply.

    Returns:
    - `True` when the docstring is empty or either text is a bare acknowledgement.
    """

    # The raw reply is checked too: an acknowledgement can survive extraction and pose as content.
    return (not docstring) or _looks_like_ack(docstring) or _looks_like_ack(reply)


def generate_docstrings(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    defs: List[DefInfo],
    source_blob: str,
    source_lines: Chunk,
    doc_order: Optional[List[str]] = None,
    callee_context: Optional[Callable[[str], str]] = None,
    on_doc: Optional[Callable[[str, str], None]] = None,
    verifier=None,
) -> Dict[int, str]:
    """
    Generate a docstring for every definition and return them keyed by node identity.

    Definitions are processed deepest-first (or in `doc_order`), so each parent's snippet can show its children as header-plus-docstring stubs. Each snippet is elided to fit the context budget, sent through one generation turn with a single nudge retry for acknowledgement replies, optionally verified, and the conversation is restored to its original length before the next definition.

    Parameters:
    - `llm`: The loaded chat model.
    - `cfg`: Generation settings for every turn.
    - `messages`: The primed conversation; extended per definition and fully unwound afterwards.
    - `defs`: The definitions to document.
    - `source_blob`: The full source text (present for the shared worker signature).
    - `source_lines`: The pristine source lines that all snippets are sliced from.
    - `doc_order`: Optional qualname ordering (e.g. leaf-first from the call graph) replacing the default deepest-first sort.
    - `callee_context`: Optional callback returning contract notes for a qualname, appended to its prompt.
    - `on_doc`: Optional `(qualname, docstring)` hook invoked as each docstring is accepted.
    - `verifier`: Optional verification gate; a docstring failing twice is written anyway with a warning.

    Returns:
    - A mapping from `id(node)` to the generated docstring for each definition.
    """

    # Take the first triple-quoted or fenced block; with no delimiters the whole reply is used as-is.
    def extract_first_docstring(reply: str) -> str:
        """
        Extract the first delimited docstring body from a model reply.

        Recognises triple-quote and code-fence delimiters. A missing opener falls back to the whole reply, and a missing closer runs to the end; the result is dedented and trimmed.

        Parameters:
        - `reply`: The raw text of the model's reply.

        Returns:
        - The extracted docstring body, dedented and stripped of surrounding whitespace.
        """

        # Compare against whitespace-stripped lines so indented fences still match.
        lines = reply.split("\n")
        stripped = [ln.strip() for ln in lines]
        start_idx = None

        # Try each delimiter style in turn; the model may fence with quotes or backticks.
        for token in ('"""', "'''", '```'):
            try:
                start_idx = stripped.index(token) + 1
                break
            except ValueError:
                continue

        # No opening fence means a bare reply - treat the whole text as the body.
        if start_idx is None:
            start_idx = 0
        end_idx = None

        # Look for the matching closer only after the opener, in the same preference order.
        for token in ('"""', "'''", '```'):
            try:
                end_idx = stripped.index(token, start_idx)
                break
            except ValueError:
                continue

        # An unterminated fence runs to the end of the reply.
        if end_idx is None:
            end_idx = len(lines)
        block = "\n".join(lines[start_idx:end_idx])

        # Dedent so the patcher can apply its own indentation later.
        return textwrap.dedent(block).strip()

    # Clamped, 1-based slicing over the pristine source - patching always works from these lines.
    def get_text_for_lines(line_a: int, line_b: int) -> str:
        """
        Return the source text for an inclusive, 1-based line range.

        The range is clamped to the file's bounds; an empty or inverted range yields an empty string.

        Parameters:
        - `line_a`: The first line of the range (1-based, inclusive).
        - `line_b`: The last line of the range (1-based, inclusive).

        Returns:
        - The joined source text for the clamped range, or an empty string.
        """

        # Clamp the range to the file's bounds rather than raising on bad input.
        a = max(1, line_a)
        b = min(len(source_lines), line_b)
        if a > b:
            return ""
        return "\n".join(source_lines[a - 1:b])

    def get_statement_source(stmt: ast.AST) -> str:
        """
        Return the verbatim source text spanned by a statement node.

        Parameters:
        - `stmt`: The AST statement whose lines are wanted.

        Returns:
        - The source text for the statement's line range; nodes lacking position attributes fall back to line 1.
        """

        # Tolerate synthetic nodes that lack position attributes.
        s = getattr(stmt, "lineno", 1)
        e = getattr(stmt, "end_lineno", s)

        return get_text_for_lines(s, e)

    def leading_spaces_count(line: str) -> int:
        """
        Count the leading space characters on a line.

        Only spaces are counted; tabs are not.

        Parameters:
        - `line`: The line of text to measure.

        Returns:
        - The number of leading spaces.
        """

        return len(line) - len(line.lstrip(" "))

    docs_by_node_id: Dict[int, str] = {}    # node identity -> generated docstring (avoids qualname collisions)

    # Render a child def as header plus docstring only - the parent sees the contract, not the body.
    def make_child_stub(child_node_id: int) -> str:
        """
        Render a nested definition as a stub: its header plus a docstring-only body.

        Used when assembling a parent's snippet, so child bodies are elided while their contracts stay visible.

        Parameters:
        - `child_node_id`: The `id()` of the child definition node to stub.

        Returns:
        - The stub text: the header lines followed by an indented docstring, or a `(no docstring)` placeholder.
        """

        # Indent the stub's docstring one level beyond the child's header.
        child_info = info_by_node_id[child_node_id]
        header_text = get_text_for_lines(child_info.header_start, child_info.header_end)
        header_last_line = source_lines[child_info.header_end - 1] if 1 <= child_info.header_end <= len(source_lines) else ""
        base_indent = leading_spaces_count(header_last_line) + 4
        child_doc = docs_by_node_id.get(child_node_id, "")
        body_lines = [" " * base_indent + '"""']

        # A child with no docstring still gets a visible placeholder, so the elision is obvious.
        if child_doc:
            body_lines.extend(" " * base_indent + ln for ln in child_doc.splitlines())
        else:
            body_lines.append(" " * base_indent + "(no docstring)")

        body_lines.append(" " * base_indent + '"""')

        # Join header and body without introducing a stray leading newline.
        return header_text + ("\n" if header_text and body_lines else "") + "\n".join(body_lines)

    # Rebuild the routine's text with direct children collapsed to stubs; the cursor keeps interleaved code and gaps intact.
    def assemble_snippet_for(node_id: int) -> str:
        """
        Assemble the prompt snippet for one definition, collapsing direct child defs to docstring stubs.

        Text between statements (comments and the like) is carried over verbatim, so the snippet reads like the real source.

        Parameters:
        - `node_id`: The `id()` of the definition node to render.

        Returns:
        - The snippet text: the header plus the body, with nested definitions stubbed.
        """

        # The cursor tracks the next source line not yet emitted, so inter-statement text can be carried over.
        info = info_by_node_id[node_id]
        node = info.node
        header_text = get_text_for_lines(info.header_start, info.header_end)
        body_chunks: Chunk = []
        direct_children: set[int] = set(info.children_ids)
        cursor = info.header_end + 1  # first body line (1-based)

        # A nested def's span starts at its decorators, not the `def` line.
        for stmt in getattr(node, "body", []):
            stmt_start = _header_span(stmt)[0] if _is_def_node(stmt) else stmt.lineno

            # Carry over comment lines that sit between statements, skipping pure blanks.
            if cursor < stmt_start:
                gap = get_text_for_lines(cursor, stmt_start - 1)
                if gap.strip():
                    body_chunks.append(gap)

            stmt_id = id(stmt)

            # Collapse direct child definitions to docstring stubs; everything else is copied verbatim.
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and stmt_id in direct_children:
                body_chunks.append(make_child_stub(stmt_id))
                cursor = info_by_node_id[stmt_id].end + 1
            else:
                body_chunks.append(get_statement_source(stmt))
                cursor = getattr(stmt, "end_lineno", stmt.lineno) + 1

        parts: Chunk = [header_text]

        # Stitch header to body without doubling the newline between them.
        if body_chunks:
            if header_text and not header_text.endswith("\n"):
                parts.append("\n")
            parts.append("\n".join(body_chunks))

        return "".join(parts)

    info_by_node_id: Dict[int, DefInfo] = {id(info.node): info for info in defs}
    deepest_first_key = lambda d: (-d.depth, d.start, -d.end)

    # Deepest-first (or the caller's order) means child docstrings exist before their parent's snippet is assembled.
    if doc_order:
        defs_deepest_first = apply_doc_order(defs, lambda d: d.qualname, doc_order, deepest_first_key)
    else:
        defs_deepest_first = sorted(defs, key=lambda d: (d.depth, d.start, -d.end), reverse=True)

    # Assemble the child-stubbed snippet, then elide it to fit the context window before prompting.
    for info in defs_deepest_first:
        node_id = id(info.node)
        full_snippet = assemble_snippet_for(node_id)
        header_lines = max(1, info.header_end - info.header_start + 1)
        local_snippet, omitted = elide_structurally(llm, cfg, messages, full_snippet, header_lines, MARKER_PYTHON)
        if omitted:
            echo(f"[Python] Elided {omitted} body line(s) from '{info.qualname}' to fit the context window")
        echo("\n[Python] Snippet...\n")
        echo(local_snippet)

        # Classes get a variant prompt framing the nested method docstrings as what the class abstracts over.
        if info.kind == "class":
            prompt = (
                "Write exactly the docstring for this class, reformatting and updating any existing comment\n"
                "as required. Use the nested method docstrings to help but remember that they are nested so\n"
                f"the class is abstracting over all of them:\n\n{local_snippet}\n"
            )
        else:
            prompt = (
                "Write exactly the docstring for this program chunk, reformatting and updating any existing\n"
                f"comment as required:\n\n{local_snippet}\n"
            )

        # Callee contract notes ground any claims the docstring makes about routines it calls.
        if callee_context is not None:
            notes = callee_context(info.qualname)
            if notes:
                prompt += "\n" + notes + "\n"

        # First attempt; `appended` counts the turns so the conversation can be unwound exactly.
        appended = 0
        messages.append({"role": "user", "content": prompt})
        appended += 1
        reply = llm.generate(messages, cfg=cfg)
        echo(f"\n[Python] LLM output:\n\n{reply}")
        docstring = extract_first_docstring(reply)

        # One nudge retry when the model acknowledges instead of producing a docstring.
        if _is_unusable_docstring(reply, docstring):
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": DOCSTRING_NUDGE})
            appended += 2
            reply = llm.generate(messages, cfg=cfg)
            echo(f"\n[Python] LLM output (after nudge):\n\n{reply}")
            docstring = extract_first_docstring(reply)

        # Restore the conversation - no routine's turns may leak into the next.
        for _ in range(appended):
            messages.pop()

        # A still-unusable result becomes a visible placeholder rather than a silently missing docstring.
        if _is_unusable_docstring(reply, docstring):
            docstring = f"{info.kind} `{info.qualname}` - comment generation failed."
        elif verifier is not None:
            last_reply = [reply]

            # Verifier retry hook: replays the exchange plus feedback in-context, then unwinds those turns too.
            def regenerate(feedback: str, _prompt: str = prompt) -> str:
                """
                Retry docstring generation with verifier feedback layered onto the original exchange.

                The three turns appended for the retry are popped straight after, leaving the shared message list unchanged; a usable retry also replaces `last_reply` in place.

                Parameters:
                - `feedback`: The verifier's objection to feed back to the model.
                - `_prompt`: The original prompt, captured at definition time.

                Returns:
                - The regenerated docstring, or an empty string if the retry was unusable.
                """

                # Replay the original exchange plus the objection, then pop all three turns so the shared context stays bounded.
                messages.append({"role": "user", "content": _prompt})
                messages.append({"role": "assistant", "content": last_reply[0]})
                messages.append({"role": "user", "content": feedback})
                retry = llm.generate(messages, cfg=cfg)
                for _ in range(3):
                    messages.pop()
                doc = extract_first_docstring(retry)
                if _is_unusable_docstring(retry, doc):
                    return ""
                last_reply[0] = retry

                return doc

            docstring, ok = verifier.verify_def(local_snippet, docstring, regenerate, label=info.qualname)

            # A doubly-failed docstring is still written, but flagged loudly for human review.
            if not ok:
                error(f"[verify] '{info.qualname}': docstring failed verification twice; writing it anyway - "
                      f"review this docstring")

        # The `on_doc` hook lets callers capture each docstring as it is accepted (e.g. for the contract store).
        docs_by_node_id[node_id] = docstring
        if on_doc is not None:
            on_doc(info.qualname, docstring)

    return docs_by_node_id


def patch_docstrings_textually(source_lines: Chunk, defs: List[DefInfo], doc_map: Dict[int, str]) -> Chunk:
    """
    Splice generated docstrings into the source lines without re-emitting any code.

    Definitions are patched in reverse line order, so insertions never shift the positions of those still to be patched. Inline definitions whose body shares the header line are skipped. An existing docstring is replaced in place; otherwise the new one is inserted directly below the header.

    Parameters:
    - `source_lines`: The original source as a list of lines.
    - `defs`: The discovered definitions with their header and body positions.
    - `doc_map`: New docstring bodies keyed by the `id()` of each definition node.

    Returns:
    - A new list of lines with the docstrings applied; the input list is not modified.
    """

    out_lines = source_lines[:]  # mutable copy

    # Patch bottom-up so each insertion leaves earlier line numbers untouched.
    for info in sorted(defs, key=lambda d: d.start, reverse=True):
        if id(info.node) not in doc_map:
            continue
        doc = doc_map[id(info.node)]
        node = info.node
        has_body = bool(getattr(node, "body", []))

        # Check whether the body starts on the header line itself (a one-liner def).
        if has_body:
            first_stmt = node.body[0]
            prefix = source_lines[first_stmt.lineno - 1][:first_stmt.col_offset]

            # An inline body leaves no line to insert a docstring on - skip it.
            if prefix.strip() != "":
                echo(f"Skipping inline definition '{info.qualname}' (body shares the header line)")
                continue

        # Render the docstring block one indent level below the `def`, then probe for an existing docstring to replace.
        def_line_text = source_lines[info.def_line - 1]
        base_indent = def_line_text[: len(def_line_text) - len(def_line_text.lstrip())] + "    "
        new_doc_lines = [f'{base_indent}"""', *[f"{base_indent}{line}".rstrip() for line in doc.splitlines()], f'{base_indent}"""']
        existing_doc = (
            has_body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            and isinstance(getattr(node.body[0].value, "value", None), str)
        )

        # Replace an existing docstring in place; a fresh one goes just below the header with a separating blank.
        if existing_doc:
            ds_start = node.body[0].lineno - 1
            ds_end = (getattr(node.body[0], "end_lineno", node.body[0].lineno)) - 1
            out_lines[ds_start: ds_end + 1] = new_doc_lines
        else:
            insert_at = info.header_end  # 1-based → acts as 0-based insertion index
            out_lines[insert_at:insert_at] = new_doc_lines + [""]

    return out_lines


# ---------------------------- within-function block targets ----------------------------


def _is_docstring_stmt(stmt: ast.AST) -> bool:
    """
    Test whether a statement is a docstring expression (a bare string constant).

    Parameters:
    - `stmt`: The AST statement to test.

    Returns:
    - `True` if the statement is an expression wrapping a string constant, otherwise `False`.
    """

    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _sub_statement_lists(stmt: ast.AST) -> List[List[ast.AST]]:
    """
    Collect every nested statement list belonging to a compound statement.

    Covers the fixed `body`/`orelse`/`finalbody` attributes plus the indirectly nested blocks of `except` handlers and `match` cases.

    Parameters:
    - `stmt`: The compound AST statement to inspect.

    Returns:
    - A list of statement lists; empty if the statement nests nothing.
    """

    lists: List[List[ast.AST]] = []

    # The directly attached suites hang off a fixed trio of attribute names.
    for attr in ("body", "orelse", "finalbody"):
        block = getattr(stmt, attr, None)
        if block:
            lists.append(block)

    # Handler bodies are nested a level deeper, inside the except clauses rather than on the statement itself.
    for handler in getattr(stmt, "handlers", []) or []:
        if getattr(handler, "body", None):
            lists.append(handler.body)

    # `match` cases nest the same way, one suite per case.
    for case in getattr(stmt, "cases", []) or []:  # match-case
        if getattr(case, "body", None):
            lists.append(case.body)

    return lists


def _body_boundaries(node: ast.AST, source_lines: Chunk) -> Tuple[List[int], Dict[int, str]]:
    """
    Compute the candidate block-boundary lines inside a routine's body.

    A boundary is a line on which exactly one statement starts at the line's leading indent; semicolon-joined and mid-line statements are rejected, a leading docstring is ignored, and nested function or class definitions are recorded as single opaque boundaries rather than descended into.

    Parameters:
    - `node`: The AST node whose body is being segmented.
    - `source_lines`: The pristine source lines of the file, used for indent checks.

    Returns:
    - A tuple of the sorted 1-based boundary line numbers and a mapping from each boundary line to its leading whitespace.
    """

    # Per-line bookkeeping: lines hosting more than one statement are vetoed later.
    line_count: Dict[int, int] = {}   # line -> number of statements starting on it
    line_col: Dict[int, int] = {}     # line -> column of the (single) statement start
    nested_lines: set[int] = set()    # boundary lines that are opaque nested definitions

    # Tally a statement start, flagging nested defs so they survive the column check below.
    def record(line: int, col: int, is_nested: bool) -> None:
        """
        Tally one statement start on a line, noting its column and whether it is a nested definition.

        Parameters:
        - `line`: The 1-based line number the statement starts on.
        - `col`: The column offset of the statement start.
        - `is_nested`: `True` when the statement is a nested function or class definition.
        """

        line_count[line] = line_count.get(line, 0) + 1
        line_col[line] = col
        if is_nested:
            nested_lines.add(line)

    # Walk the body recursively; nested defs are kept opaque and counted as a single boundary at their header.
    def collect(stmts: List[ast.AST], is_top: bool) -> None:
        """
        Recursively record every statement start beneath a list of statements.

        Nested function and class definitions are recorded as single opaque boundaries without entering their bodies. The first statement of an inner suite is skipped as a boundary because it opens the suite rather than starting a new block, though its own sub-suites are still walked.

        Parameters:
        - `stmts`: The statements to walk.
        - `is_top`: `True` when `stmts` is the routine's top-level body, whose first statement does count as a block start.
        """

        for idx, stmt in enumerate(stmts):
            skip = (not is_top) and idx == 0  # first line of an inner suite is not a block start

            # Nested defs are opaque: record the header (decorators included) as one boundary and never enter the body.
            if _is_def_node(stmt):
                if not skip:
                    record(_header_span(stmt)[0], 0, True)
                continue

            # skip suppresses recording only - inner suites are still descended so their block starts are counted.
            if not skip:
                record(stmt.lineno, getattr(stmt, "col_offset", 0), False)
            for sub in _sub_statement_lists(stmt):
                collect(sub, False)

    # Segment the body with any leading docstring excluded - it is never a block start.
    body = list(getattr(node, "body", []))
    consider = body[1:] if body and _is_docstring_stmt(body[0]) else body
    collect(consider, True)
    boundary_lines: List[int] = []
    indent_of: Dict[int, str] = {}

    # Keep only lines hosting exactly one statement that starts at the line's first non-blank column; this rejects semicolon-joined and inline statements.
    for line, count in line_count.items():
        if count != 1 or not (1 <= line <= len(source_lines)):
            continue
        text = source_lines[line - 1]
        leading = text[: len(text) - len(text.lstrip())]
        if line not in nested_lines and text[: line_col[line]].strip() != "":
            continue
        boundary_lines.append(line)
        indent_of[line] = leading

    boundary_lines.sort()

    return boundary_lines, indent_of


# Compound statements that open a paragraph-worthy block (nested defs/classes included - a nested definition is its own
# paragraph). Used by the deterministic structural segmenter below.
_SEG_COMPOUND = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith,
                 ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
if hasattr(ast, "Match"):
    _SEG_COMPOUND = _SEG_COMPOUND + (ast.Match,)

# A block must span at least this many source lines to earn a paragraph break before it (or before the statement that
# resumes after it). Size is a better "is this trivial" gate than cognitive complexity, which is nesting-dominated and
# wrongly demotes long-but-flat blocks. 3 generalises across codebases (5 overfit this repo's style).
_SEG_MIN_BLOCK_LINES = 3


def _seg_span(node: ast.AST) -> int:
    """
    Return the number of source lines a statement spans.

    Definitions are measured from their first decorator line, and nodes without an `end_lineno` are treated as spanning a single line.

    Parameters:
    - `node`: The AST statement to measure.

    Returns:
    - The inclusive line count of the statement.
    """

    start = _header_span(node)[0] if _is_def_node(node) else node.lineno
    return getattr(node, "end_lineno", start) - start + 1


def _seg_records(node: ast.AST, source_lines: Chunk) -> List[Tuple[int, int, ast.AST, bool, bool]]:
    """
    Collect one record per statement in a routine's body for the structural segmenter.

    Each record carries the statement's start line (first decorator line for definitions), its indent width measured from the raw source text, the AST node, whether it opens an inner suite, and whether it opens the top-level body. A leading docstring is skipped and nested definitions are not descended into.

    Parameters:
    - `node`: The AST node whose body is being segmented.
    - `source_lines`: The pristine source lines, used to measure indentation.

    Returns:
    - A list of `(start_line, indent, node, is_suite_leader, is_body_leader)` tuples sorted by start line.
    """

    records: List[Tuple[int, int, ast.AST, bool, bool]] = []

    # Depth-first walk; definitions are kept opaque (recorded at their first decorator line) and their bodies never entered.
    def walk(stmts: List[ast.AST], is_top: bool) -> None:
        """
        Append a record for each statement, recursing into compound suites but never into nested definitions.

        Parameters:
        - `stmts`: The statements to record.
        - `is_top`: `True` when `stmts` is the routine's top-level body.
        """

        # The two trailing flags mark suite-opening and body-opening statements for the segmenter's merge rules.
        for idx, stmt in enumerate(stmts):
            start = _header_span(stmt)[0] if _is_def_node(stmt) else stmt.lineno
            text = source_lines[start - 1] if 1 <= start <= len(source_lines) else ""
            indent = len(text) - len(text.lstrip())
            records.append((start, indent, stmt, (not is_top) and idx == 0, is_top and idx == 0))

            # Definitions stay opaque - their bodies are never walked.
            if not _is_def_node(stmt):
                for sub in _sub_statement_lists(stmt):
                    walk(sub, False)

    # Walk the body without its docstring, then restore source-line order.
    body = list(getattr(node, "body", []))
    if body and _is_docstring_stmt(body[0]):
        body = body[1:]
    walk(body, True)
    records.sort(key=lambda r: r[0])

    return records


def _seg_closed_block(prev_node: ast.AST, resume_start: int, parents: Dict[int, ast.AST],
                      target: ast.AST) -> Optional[ast.AST]:
    """
    Find the outermost compound block that closed between two consecutive segment records.

    The parent chain is climbed from the previous statement up to (but not including) the target routine, keeping the highest compound ancestor whose last line falls before the resume line.

    Parameters:
    - `prev_node`: The statement immediately preceding the gap.
    - `resume_start`: The start line of the record being resumed at.
    - `parents`: Mapping from `id(node)` to that node's parent AST node.
    - `target`: The routine node bounding the climb.

    Returns:
    - The outermost closed compound block, or `None` if no enclosing block closed before `resume_start`.
    """

    best: Optional[ast.AST] = None
    n = parents.get(id(prev_node))

    # Climb the parent chain up to (but not including) the target routine; the outermost ancestor ending before the resume line wins.
    while n is not None and n is not target:
        if isinstance(n, _SEG_COMPOUND) and getattr(n, "end_lineno", 0) < resume_start:
            best = n                                 # keep climbing for the outermost closed block
        n = parents.get(id(n))

    return best


def _seg_merge_anchors(node: ast.AST) -> Dict[int, int]:
    """
    Map return-statement lines to the simple statement they should merge with.

    A suite consisting of exactly one non-compound statement followed by a `return` reads as a single paragraph, so the return is anchored to its predecessor's line rather than opening a block of its own. Nested definitions are not descended into.

    Parameters:
    - `node`: The AST node whose body is being scanned.

    Returns:
    - A dictionary mapping each such return's line number to the line number of its anchor statement.
    """

    anchors: Dict[int, int] = {}

    # A suite of exactly one simple statement plus a return reads as one paragraph, so anchor the return to its predecessor.
    def visit(stmts: List[ast.AST]) -> None:
        """
        Record merge anchors for one statement list, recursing into nested suites.

        A suite of exactly one non-compound statement followed by a `return` anchors the return's line to its predecessor's; nested definitions are not descended into.

        Parameters:
        - `stmts`: The statement list to scan.
        """

        # A simple setup line followed only by a return reads as one thought: anchor the return back to its setup line so the pair shares a segment.
        if (len(stmts) == 2 and isinstance(stmts[1], ast.Return)
                and not isinstance(stmts[0], _SEG_COMPOUND)):
            a, r = stmts[0].lineno, stmts[1].lineno
            if a != r:
                anchors[r] = a

        # Recurse into every nested suite, but skip nested defs - they are segmented as routines in their own right.
        for stmt in stmts:
            if _is_def_node(stmt):
                continue
            for sub in _sub_statement_lists(stmt):
                visit(sub)

    # Scan the whole body, ignoring any leading docstring.
    body = list(getattr(node, "body", []))
    if body and _is_docstring_stmt(body[0]):
        body = body[1:]
    visit(body)

    return anchors


def structural_segments(node: ast.AST, source_lines: Chunk, boundary_lines: List[int],
                        body_end: int) -> List[Tuple[int, int]]:
    """
    Compute the structural paragraph segments for one routine body.

    Each statement is reduced to a language-neutral `SegStatement` record - depth, span, the size of any block it opens or has just closed, and its return merge anchor - then handed to the shared `structural_breaks` engine, which applies the deterministic break rules.

    Parameters:
    - `node`: The routine's AST definition node.
    - `source_lines`: The file's source lines.
    - `boundary_lines`: Line numbers where a comment may legally be inserted.
    - `body_end`: The last line of the routine's body.

    Returns:
    - A list of `(start, end)` 1-based inclusive line ranges, one per segment; empty if the body has no segmentable statements.
    """

    # Bail out early when the body yields no segmentable statements.
    records = _seg_records(node, source_lines)
    if not records:
        return []
    parents: Dict[int, ast.AST] = {}

    # ast gives no parent links, so build a child-to-parent map for sizing just-closed blocks.
    for n in ast.walk(node):
        for c in ast.iter_child_nodes(n):
            parents[id(c)] = n

    # A leading docstring and the return merge anchors both feed the break rules.
    body = list(getattr(node, "body", []))
    has_doc = bool(body and _is_docstring_stmt(body[0]))
    merge = _seg_merge_anchors(node)
    stmts: List[SegStatement] = []

    # Reduce each statement to the language-agnostic record the shared break engine consumes.
    for i, (start, indent, stmt, _first_of_suite, first_in_scope) in enumerate(records):
        end = getattr(stmt, "end_lineno", start) or start
        opens = _seg_span(stmt) if isinstance(stmt, _SEG_COMPOUND) else 0
        prev = records[i - 1] if i > 0 else None
        closed = 0

        # A dedent from the previous statement means a block just closed; measure its span for the dedent rule.
        if prev is not None and prev[1] > indent:
            blk = _seg_closed_block(prev[2], start, parents, node)
            closed = _seg_span(blk) if blk is not None else 0

        # Merge anchors are only meaningful on return statements, so attach them there alone.
        stmts.append(SegStatement(
            start=start, end=end, depth=indent,
            is_return=isinstance(stmt, ast.Return), is_def=_is_def_node(stmt),
            opens_block=opens, first_in_scope=first_in_scope, closed_block=closed,
            merge_anchor=merge.get(start) if isinstance(stmt, ast.Return) else None,
        ))

    # Delegate the actual break decisions to the shared engine, passing Python's policy knobs.
    return structural_breaks(
        stmts, has_doc=has_doc, boundary_lines=tuple(boundary_lines), body_end=body_end,
        min_block_lines=_SEG_MIN_BLOCK_LINES, allow_after_def=True, allow_first_in_scope=True,
    )


def iter_block_targets(source_blob: str, source_lines: Chunk) -> List[BlockTarget]:
    """
    Build the block-pass target list for every routine in a Python source file.

    Each target is self-contained: header and body line ranges, the legal comment insertion points (boundary lines plus any return merge anchors, with their indentation), the existing docstring and signature, and the precomputed structural segments.

    Parameters:
    - `source_blob`: The full source text, parsed once for the whole file.
    - `source_lines`: The file's source lines.

    Returns:
    - A list of `BlockTarget` records, one per routine with a non-empty body.
    """

    tree = ast.parse(source_blob)
    targets: List[BlockTarget] = []

    # Only routines with a body can take block comments; the boundary scan maps their legal insertion points.
    for info in iter_defs_with_info(tree):
        body = list(getattr(info.node, "body", []))
        if not body:
            continue
        boundary_lines, indent_of = _body_boundaries(info.node, source_lines)

        # Return merge anchors may sit on lines the boundary scan skipped, so admit them as insertion points with their measured indentation.
        for anchor in set(_seg_merge_anchors(info.node).values()):
            if 1 <= anchor <= len(source_lines) and anchor not in indent_of:
                text = source_lines[anchor - 1]
                indent_of[anchor] = text[: len(text) - len(text.lstrip())]
                boundary_lines.append(anchor)

        # Bundle everything the block pass needs - ranges, insertion points, doc, signature and precomputed segments - into one self-contained target.
        boundary_lines = sorted(boundary_lines)
        targets.append(
            BlockTarget(
                qualname=info.qualname,
                kind=info.kind,
                header_start=info.header_start,
                header_end=info.header_end,
                body_start=body[0].lineno,
                body_end=info.end,
                boundary_lines=tuple(boundary_lines),
                indent_of=indent_of,
                depth=info.depth,
                doc=ast.get_docstring(info.node) or "",
                sig=node_sig(info.node),
                segments=structural_segments(info.node, source_lines, boundary_lines, info.end),
            )
        )

    return targets


def collect_def_requests(source_blob: str, source_lines: Chunk, escalation) -> int:
    """
    Record every routine in a Python source as a deferred definition request.

    Each def's verbatim source span is handed to the escalation collector exactly once.

    Parameters:
    - `source_blob`: The full source text to parse.
    - `source_lines`: The file's source lines, used to slice each span.
    - `escalation`: The run-manifest collector that records each request.

    Returns:
    - The number of definitions recorded.
    """

    defs = iter_defs_with_info(ast.parse(source_blob))

    # Each routine's verbatim span enters the manifest here - the one place its code crosses the wire.
    for info in defs:
        span = "\n".join(source_lines[info.header_start - 1:info.end])
        escalation.record_def(qualname=info.qualname, kind=info.kind, sig_hash=node_sig(info.node), snippet=span)

    # The count feeds the deterministic completeness check rather than trusting the model.
    return len(defs)


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
    Generate docstrings for every definition in a Python source file.

    This is the Python worker's shared entry point: it parses the source, describes the imports into the shared chat context, generates one docstring per definition, then patches the results into the original lines so executable code is preserved byte-for-byte.

    Parameters:
    - `llm`: The local chat model used for generation.
    - `cfg`: The generation settings for this run.
    - `messages`: The shared chat transcript carrying the priming context.
    - `source_blob`: The full source text to annotate.
    - `source_lines`: The same source split into lines, used for patching.
    - `doc_order`: Optional ordering of qualnames for docstring generation.
    - `callee_context`: Optional callback returning callee context for a qualname.
    - `on_doc`: Optional callback invoked with each qualname and docstring produced.
    - `verifier`: Optional verifier applied to each generated docstring.

    Returns:
    - The patched source lines with docstrings inserted or updated.
    """

    # Parse once and reuse the tree throughout; describing the imports first primes the shared chat context that the docstring turns build on.
    echo("Parsing Python source code...")
    tree = ast.parse(source_blob)
    echo("Identifying imports...")
    describe_imports_from_tree(llm, cfg, messages, tree)
    echo("Identifying definitions...")
    defs = iter_defs_with_info(tree)
    echo(f"Found {len(defs)} definitions")
    echo("Generating docstrings...\n")
    doc_map = generate_docstrings(llm, cfg, messages, defs, source_blob, source_lines,
                                  doc_order=doc_order, callee_context=callee_context, on_doc=on_doc,
                                  verifier=verifier)
    echo("Applying Python patches...\n")

    # Patch the pristine source lines textually - the model never re-emits code, so everything else survives byte-for-byte.
    return patch_docstrings_textually(source_lines, defs, doc_map)


# ---------------------------- manifest apply (model-free) ----------------------------


def _clean_docstring_answer(text: str) -> str:
    """
    Normalise a raw docstring answer into a bare body.

    Strips any Markdown code fence and any echoed triple-quote delimiters, then dedents, so the patcher receives plain prose it can re-indent and quote itself.

    Parameters:
    - `text`: The raw answer text from the model or manifest.

    Returns:
    - The cleaned docstring body, possibly empty.
    """

    # Models sometimes wrap the body in a Markdown fence despite instructions; unwrap it before anything else.
    body = text.strip()
    fence = re.match(r"^```[^\n]*\n(.*)\n```$", body, flags=re.DOTALL)
    if fence:
        body = fence.group(1).strip()

    # Strip echoed triple-quote delimiters too - only the bare body may reach the patcher, which adds its own quotes.
    for q in ('"""', "'''"):
        if body.startswith(q) and body.endswith(q) and len(body) >= 2 * len(q):
            body = body[len(q):-len(q)]
            break

    # Dedent fully; the patcher re-indents the body to the def's own level.
    return textwrap.dedent(body).strip()


def apply_manifest(source_lines: Chunk, manifest: dict) -> Chunk:
    """
    Apply a filled manifest's answers to the source lines.

    Def answers are matched to live AST nodes by qualname and signature hash and patched in first; block answers are then re-bound against the freshly segmented routines, with each routine's edits trial-applied and rejected wholesale if they would alter executable code. Unanswered or unmatched requests leave the source untouched.

    Parameters:
    - `source_lines`: The source lines to patch.
    - `manifest`: The filled manifest dictionary holding the requests and answers.

    Returns:
    - The patched source lines; unchanged wherever answers were missing or rejected.
    """

    # The local import avoids a circular dependency with scale_blocks; requests are split so docstrings land before block comments.
    from scale_blocks import PYTHON_STYLE, _apply_edits, code_preserved, _parse_comment_reply
    requests = manifest.get("requests", [])
    def_reqs = [r for r in requests if r.get("def") is not None]
    block_reqs = [r for r in requests if r.get("blocks") is not None]
    out_lines = source_lines

    # Def phase: re-parse the current text and key answers by qualname plus signature hash, not by line numbers the file may no longer share with emit time.
    if def_reqs:
        defs = iter_defs_with_info(ast.parse("\n".join(out_lines)))
        wanted = {(r["qualname"], r["sig_hash"]): r for r in def_reqs}
        doc_map: Dict[int, str] = {}
        used: set = set()

        # Walk the live defs so each answer binds at most one node; `used` stops a duplicate key claiming two routines.
        for info in defs:
            key = (info.qualname, node_sig(info.node))
            req = wanted.get(key)
            if req is None or key in used:
                continue
            answer = req["def"].get("answer")

            # An unanswered request leaves the existing docstring alone rather than blanking it.
            if not answer or not answer.strip():
                echo(f"[apply] Def request '{req['id']}' has no answer; leaving docstring untouched")
                continue

            doc = _clean_docstring_answer(answer)

            # Keyed by node identity, since qualnames alone can repeat within a file.
            if doc:
                doc_map[id(info.node)] = doc
                used.add(key)

        out_lines = patch_docstrings_textually(out_lines, defs, doc_map)

    # Block phase: re-segment only after the docstring patch, so boundary lines account for the freshly inserted docstrings.
    if block_reqs:
        targets = iter_block_targets("\n".join(out_lines), out_lines)
        by_key: Dict[Tuple[str, str], BlockTarget] = {(t.qualname, t.sig): t for t in targets}
        all_edits: List[Tuple[int, Optional[str], str]] = []

        for req in block_reqs:
            target = by_key.get((req["qualname"], req["sig_hash"]))
            chunks = req["blocks"].get("chunks", [])

            # A routine that changed since emit simply fails to match; skip it rather than guess at placement.
            if target is None:
                echo(f"[apply] No match for block request '{req['id']}'; skipping")
                continue

            # All-null answers mean the routine was never filled; distinguish that from chunks deliberately answered NONE.
            if all(c.get("answer") is None for c in chunks):
                echo(f"[apply] Block request '{req['id']}' has no answers; leaving routine untouched")
                continue

            edits: List[Tuple[int, Optional[str], str]] = []

            # Map each chunk's bidx back to its boundary line; out-of-range indices from a stale manifest are dropped silently.
            for chunk in chunks:
                bidx = chunk["bidx"]
                if not (0 <= bidx < len(target.boundary_lines)):
                    continue
                boundary = target.boundary_lines[bidx]
                comment = _parse_comment_reply(chunk.get("answer") or "", PYTHON_STYLE)
                edits.append((boundary, comment, target.indent_of.get(boundary, "")))

            # Trial-apply this routine's edits in isolation so one bad routine cannot taint the others.
            trial = _apply_edits(out_lines, edits, PYTHON_STYLE)

            # The preservation guard is the real safety net: any code drift rejects the whole routine's edits.
            if code_preserved(out_lines, trial, PYTHON_STYLE):
                all_edits.extend(edits)
            else:
                echo(f"[apply] Skipped '{req['qualname']}': block edit would alter code; keeping original")

        # Apply all surviving edits in a single pass so boundary line numbers stay valid.
        out_lines = _apply_edits(out_lines, all_edits, PYTHON_STYLE)

    return out_lines
