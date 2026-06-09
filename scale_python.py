#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

This program generates or updates docstrings in Python source code, ensuring that the generated documentation is precise
and aligned with the actual code structure.

**Key Functionalities:**

1. **DefInfo Data Structure**: The `DefInfo` class captures detailed information about each definition (function, class,
   property) in the source code.
2. **AST Traversal**: The program uses the Abstract Syntax Tree (AST) to traverse and analyse the source code.
3. **Docstring Generation**: The `generate_docstrings` function generates new or updated docstrings for each definition
   using a Large Language Model (LLM).
4. **Snippet Assembly**: The `assemble_snippet_for` function constructs the code snippet for each definition, including
   its header and body.
5. **Patch Application**: The `patch_docstrings_textually` function applies the generated or updated docstrings to the
   source code.

**Highlights:**

* **Precision in Definition Identification**: The program uses the `DefInfo` class to precisely identify and manipulate
  definitions within the AST.
* **Depth-First Processing**: Definitions are processed in a depth-first manner, ensuring that parent definitions are
  handled after their children.
* **LLM Integration**: The program leverages an LLM to generate docstrings based on the provided prompts and existing
  code snippets.
* **Stability during Updates**: The program ensures stability during updates by processing definitions in reverse order
  by start position.
* **Preservation of Comments and Formatting**: The program preserves all comments, blank lines, and formatting while
  updating docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from scale_blocks import BlockTarget
from scale_llm import LocalChatModel, GenerationConfig, Messages, Chunk
from scale_log import echo
from scale_text import fit_snippet, MARKER_PYTHON
from typing import Dict, List, Optional, Tuple
import ast
import textwrap


@dataclass(frozen=True)
class DefInfo:
    """
    A dataclass representing detailed information about a definition in the source code.

    This class captures attributes such as the qualified name, the corresponding AST node,
    line numbers, and more. It is used to precisely identify and manipulate definitions within
    the Abstract Syntax Tree (AST).

    Attributes:
    - `qualname`: The fully qualified name of the definition, e.g., "Foo.bar", "Foo.prop.setter", "outer.inner".
    - `node`: The corresponding AST node, which can be a FunctionDef, AsyncFunctionDef, or ClassDef.
    - `start`: The earliest line number including decorators (1-based).
    - `end`: The end line number (inclusive).
    - `def_line`: The line number of the 'def'/'class' keyword (1-based).
    - `header_start`: The start of the decorator and signature section.
    - `header_end`: The end of the decorator and signature section (line before the first body statement).
    - `kind`: The type of definition, which can be "class", "def", or "async def".
    - `depth`: The depth of the definition in the AST, with 0 indicating the module level.
    - `parent_id`: The ID of the parent node's AST node, or None if it is at the module level.
    - `children_ids`: A tuple of IDs for direct children nodes.
    """

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
    Determine whether the given AST node represents a function definition, async function definition, or class definition.

    Parameters:
    - `n`: The AST node to check.

    Returns:
    - `True` if the node is a function definition, async function definition, or class definition, otherwise `False`.
    """

    return isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))


def _node_kind(n: ast.AST) -> str:
    """
    Determine the kind of an AST node.

    This function identifies whether an AST node represents a class, an asynchronous function, or a regular function.

    Parameters:
    - n: The AST node to inspect.

    Returns:
    - A string indicating the kind of the node: "class", "async def", or "def".
    """

    if isinstance(n, ast.ClassDef):
        return "class"
    if isinstance(n, ast.AsyncFunctionDef):
        return "async def"
    return "def"


def _header_span(n: ast.AST) -> Tuple[int, int]:
    """
    Return the start and end lines for the node's header, including decorators and the full (possibly multi-line) signature,
    ending on the line before the first body statement.

    Parameters:
    - `n`: An Abstract Syntax Tree (AST) node representing a definition.

    Returns:
    - A tuple containing the start line and end line of the node's header.
    """

    assert _is_def_node(n)

    start = n.lineno
    if getattr(n, "decorator_list", None):
        deco_starts = [d.lineno for d in n.decorator_list if hasattr(d, "lineno")]
        if deco_starts:
            start = min(deco_starts + [start])

    if getattr(n, "body", None):
        end = n.body[0].lineno - 1
    else:
        end = getattr(n, "end_lineno", n.lineno)
    return start, end


def _property_accessor_role(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> Optional[Tuple[str, str]]:
    """
    Detect the role of a function as a property accessor.

    This function inspects the function's decorator list to identify whether it is a getter, setter, or deleter for a property.

    Parameters:
    - `fn`: The function definition to inspect, which can be either a synchronous or asynchronous function.

    Returns:
    - A tuple containing the property name and its corresponding role, where role ∈ {"getter", "setter", "deleter"}, or None
      if no matching decorator is found.
    """

    roles = {"getter", "setter", "deleter"}

    for deco in getattr(fn, "decorator_list", []) or []:
        # Match Attribute(Name(prop), role)
        if isinstance(deco, ast.Attribute) and isinstance(deco.value, ast.Name) and deco.attr in roles:
            return deco.value.id, deco.attr
    return None


def iter_defs_with_info(tree: ast.AST) -> List[DefInfo]:
    """
    Walk the AST using a scope stack to produce precise, nested definition info.

    This function traverses the Abstract Syntax Tree (AST) and returns a sorted list of `DefInfo` objects,
    each representing a definition in the code. The definitions are ordered by their start line.

    Parameters:
    - `tree`: The Abstract Syntax Tree (AST) to traverse and process.

    Returns:
    - List[DefInfo] sorted by start line ascending.
    """

    results: List[DefInfo] = []
    children_map: Dict[int, List[int]] = {}  # parent_id -> [child_ids]

    scope_names: Chunk = []      # qualname parts
    scope_nodes: List[ast.AST] = []  # node stack (for parent tracking)

    def add_child(parent_node: Optional[ast.AST], child_node: ast.AST) -> None:
        """
        Add a child node to the parent's list of children in the Abstract Syntax Tree (AST).

        Parameters:
        - `parent_node`: The parent node in the AST.
        - `child_node`: The child node to be added.

        Notes:
        If the parent node is `None`, no action is taken. Otherwise, the child node's ID is added to the parent's
        children map under its unique identifier.
        """

        if parent_node is None:
            return
        pid = id(parent_node)
        children_map.setdefault(pid, []).append(id(child_node))

    def walk(node: ast.AST) -> None:
        """
        Recursively traverse the Abstract Syntax Tree (AST) to process each child node, capturing relevant metadata for definitions.

        This function processes each child node of the given AST node, building a qualified name (qualname) and capturing relevant
        line information such as start, end, and definition lines. The qualname is constructed based on the node's type and any
        property accessors, and is stored in the `DefInfo` data structure along with other metadata.

        Parameters:
        - `node`: The current AST node to process.

        Returns:
        - None: This function does not return a value; it populates the `results` list with `DefInfo` objects as a side-effect.
        """

        for child in ast.iter_child_nodes(node):
            if _is_def_node(child):
                # Build qualname (handle property accessors specially)
                kind = _node_kind(child)
                name_part: str
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    acc = _property_accessor_role(child)
                else:
                    acc = None

                if acc is not None and scope_names:
                    prop_name, role = acc  # e.g. ("analysis_sec", "setter")
                    # Represent as Class.prop.role
                    name_part = f"{prop_name}.{role}"
                else:
                    name_part = child.name  # regular def/class name

                qualname = ".".join(scope_names + [name_part]) if scope_names else name_part

                # Lines
                def_line = child.lineno
                start_line, header_end = _header_span(child)
                end_line = getattr(child, "end_lineno", def_line)

                parent_node = scope_nodes[-1] if scope_nodes else None
                depth = len(scope_nodes)

                # Temporarily create DefInfo without children; we fill children later
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

                # Recurse into child scope
                scope_names.append(name_part if acc is None else prop_name)  # for nested defs, keep property base name
                scope_nodes.append(child)
                walk(child)
                scope_nodes.pop()
                scope_names.pop()
            else:
                walk(child)

    walk(tree)

    # Fill children_ids immutably
    completed: List[DefInfo] = []
    for info in results:
        child_ids = tuple(children_map.get(id(info.node), []))
        completed.append(replace(info, children_ids=child_ids))

    return sorted(completed, key=lambda d: d.start)


def _format_from_source(module: Optional[str], level: int) -> str:
    """
    Format the 'from' target with a specified number of leading dots.

    This function formats the `module` parameter with a specified number of leading dots (`level`).

    Parameters:
    - `module`: The module name to be formatted (optional).
    - `level`: The number of leading dots to include.

    Returns:
    - A string representing the formatted 'from' target.
    """

    dots = "." * level
    return f"{dots}{module or ''}" or "."


class _ImportDescriber(ast.NodeVisitor):
    """
    Abstract class for describing import statements in the Abstract Syntax Tree (AST).

    This class provides a way to record and process import statements, including `Import` and `ImportFrom` nodes.

    Parameters:
    - `_items`: A list of tuples containing line numbers and import text.

    Notes:
    - The class is designed to be subclassed and extended for specific use cases.
    - The `visit_Import` and `visit_ImportFrom` methods are responsible for iterating over the AST and extracting
      information about import statements.
    - The `results` method returns a sorted list of text results, ensuring robustness in case the visitor order
      differs from the source order.

    Subclasses should override the following methods to customise the import description process:

    - `visit_Import`: Process `Import` nodes.
    - `visit_ImportFrom`: Process `ImportFrom` nodes.
    - `results`: Return a sorted list of text results.
    """

    def __init__(self) -> None:
        """
        Initialise the object.

        Create an empty list to store items, where each item is a tuple containing an integer key and a string value.
        """

        self._items: List[tuple[int, str]] = []

    def visit_Import(self, node: ast.Import) -> None:
        """
        Record the import statement.

        This method iterates over the names imported by the `Import` node and appends a tuple containing the line
        number and import text to the `_items` list.

        Parameters:
        - `node`: The `Import` node being visited.
        """

        for alias in node.names:
            if alias.asname:
                text = f"- Imports {alias.name} as {alias.asname}"
            else:
                text = f"- Imports {alias.name}"
            self._items.append((getattr(node, "lineno", 0), text))
        # No children to visit for Import

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Process an import from statement.

        This method visits an `ImportFrom` node in the Abstract Syntax Tree (AST) and extracts information about the
        imported modules.

        Parameters:
        - `node`: The `ImportFrom` node to process.

        Returns:
        - A list of tuples containing the line number and a message describing each imported item.
        """

        src = _format_from_source(node.module, getattr(node, "level", 0) or 0)
        for alias in node.names:
            if alias.name == "*":
                text = f"- Imports everything from {src}"
            elif alias.asname:
                text = f"- Imports {alias.name} from {src} as {alias.asname}"
            else:
                text = f"- Imports {alias.name} from {src}"
            self._items.append((getattr(node, "lineno", 0), text))
        # No children to visit for ImportFrom

    def results(self) -> List[str]:
        """
        Return a sorted list of text results.

        The results are sorted based on the original source order to ensure robustness in case the visitor order differs.

        Returns:
        - A sorted list of text results.
        """

        self._items.sort(key=lambda t: t[0])
        return [text for _, text in self._items]


def describe_imports_from_tree(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    tree: ast.AST
) -> None:
    """
    Describe all imports found in an AST and feed them to the LLM as extra context.

    This function traverses the provided Abstract Syntax Tree (AST), builds a plain-English list of imported names in
    source order, and (if any are found) appends them to the conversation so the model has that context. The message
    list is mutated in place; nothing is returned.

    Parameters:
    - `llm`: The LocalChatModel instance.
    - `cfg`: The GenerationConfig instance.
    - `messages`: The Messages instance (mutated in place).
    - `tree`: A module AST produced by `ast.parse(...)`.

    Returns:
    - None
    """

    if not isinstance(tree, ast.AST):
        raise TypeError("tree must be an ast.AST")

    # Build the imports list
    visitor = _ImportDescriber()
    visitor.visit(tree)
    imports = visitor.results()

    # If we found anything, pass the list to the LLM to provide more context
    if imports:
        imports = "\n".join(imports)
        echo(f"\n[Python] Imports...\n{imports}")
        prompt = (
            "For additional context, here is a list of imports within this program:\n\n"
            f"{imports}\n\n"
            "Please respond by saying 'OK'. No other commentary is required at this time."
        )
        messages.append({"role": "user", "content": prompt})
        reply = llm.generate(messages, cfg=cfg)
        echo(f"\n[Python] LLM output:\n\n{reply}")
        messages.append({"role": "assistant", "content": reply})


def deepest_first(defs: List[DefInfo]) -> List[DefInfo]:
    """
    Sort the definitions in deepest-first order, ensuring stability by starting with ascending depths, then start positions,
    and finally descending end positions.

    Parameters:
    - `defs`: A list of `DefInfo` objects representing the definitions to be sorted.

    Returns:
    - A list of `DefInfo` objects sorted in deepest-first order.
    """

    return sorted(defs, key=lambda d: (d.depth, d.start, -d.end), reverse=True)


def generate_docstrings(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    defs: List[DefInfo],
    source_blob: str,
    source_lines: Chunk
) -> Dict[int, str]:
    """
    Generate or update docstrings for definitions in a Python source code AST.

    This function processes the deepest definitions first, assembling snippets that include headers and bodies while
    replacing direct child definitions with stubs containing their docstrings. Non-definition statements are included
    verbatim from the source.

    Parameters:
    - `llm`: The local chat model used to generate docstrings.
    - `cfg`: Configuration for the generation process.
    - `messages`: A list of messages for the chat model.
    - `defs`: A list of `DefInfo` objects representing definitions in the AST.
    - `source_blob`: The entire source code as a string.
    - `source_lines`: A chunk of source lines for context.

    Returns:
    - A dictionary mapping qualified names to generated or updated docstrings.
    """

    # ---------------- helpers ----------------

    def extract_first_docstring(reply: str) -> str:
        """
        Extract the first fenced docstring block from a reply string.

        This function searches for the first occurrence of a triple-quoted or triple-backtick block, and returns its
        contents, dedented and stripped. If no such block is found, it treats the entire reply as the candidate
        docstring.

        Parameters:
        - `reply`: The input string from which to extract the docstring.

        Returns:
        - The extracted docstring, dedented and stripped. Returns an empty string if no docstring is found.
        """

        lines = reply.split("\n")
        stripped = [ln.strip() for ln in lines]

        start_idx = None
        for token in ('"""', "'''", '```'):
            try:
                start_idx = stripped.index(token) + 1
                break
            except ValueError:
                continue
        if start_idx is None:
            start_idx = 0

        end_idx = None
        for token in ('"""', "'''", '```'):
            try:
                end_idx = stripped.index(token, start_idx)
                break
            except ValueError:
                continue
        if end_idx is None:
            end_idx = len(lines)

        block = "\n".join(lines[start_idx:end_idx])
        return textwrap.dedent(block).strip()

    def get_text_for_lines(line_a: int, line_b: int) -> str:
        """
        Return source text for an inclusive 1-based line range [line_a, line_b].

        Parameters:
        - `line_a`: The start of the line range (1-based).
        - `line_b`: The end of the line range (1-based).

        Returns:
        - A string containing the source text for the specified line range.

        Notes:
        - The function ensures that the line range is valid by clamping `line_a` to at least 1 and `line_b` to at most the
          number of source lines.
        """

        a = max(1, line_a)
        b = min(len(source_lines), line_b)
        if a > b:
            return ""
        return "\n".join(source_lines[a - 1:b])

    def get_statement_source(stmt: ast.AST) -> str:
        """
        Return the exact source code for a statement by line span, preserving indentation.

        Parameters:
        - stmt: The abstract syntax tree (AST) node representing a statement.

        Returns:
        - A string containing the exact source code for the statement, including preserved indentation.
        """

        s = getattr(stmt, "lineno", 1)
        e = getattr(stmt, "end_lineno", s)
        return get_text_for_lines(s, e)

    def leading_spaces_count(line: str) -> int:
        """
        Count the leading spaces in a given line.

        Parameters:
        - `line`: The input string representing a line of text.

        Returns:
        - The number of leading spaces in the line.
        """

        return len(line) - len(line.lstrip(" "))

    # ---------------- snippet assembly ----------------

    docs_by_node_id: Dict[int, str] = {}    # node identity -> generated docstring (avoids qualname collisions)

    def make_child_stub(child_node_id: int) -> str:
        """
        Generate a string containing the combined decorators, header, and child docstring for a direct child node.

        Parameters:
        - `child_node_id`: The ID of the child node for which to generate the stub.

        Returns:
        - A formatted string combining the decorators, header, and child docstring.
        """

        child_info = info_by_node_id[child_node_id]
        header_text = get_text_for_lines(child_info.header_start, child_info.header_end)

        # Base indent = indent of header’s final line + one level (4 spaces)
        header_last_line = source_lines[child_info.header_end - 1] if 1 <= child_info.header_end <= len(source_lines) else ""
        base_indent = leading_spaces_count(header_last_line) + 4

        child_doc = docs_by_node_id.get(child_node_id, "")
        body_lines = [" " * base_indent + '"""']
        if child_doc:
            body_lines.extend(" " * base_indent + ln for ln in child_doc.splitlines())
        else:
            body_lines.append(" " * base_indent + "(no docstring)")
        body_lines.append(" " * base_indent + '"""')

        return header_text + ("\n" if header_text and body_lines else "") + "\n".join(body_lines)

    def assemble_snippet_for(node_id: int) -> str:
        """
        Construct a code snippet for the given node.

        This function assembles a code snippet by including the node's header (decorators and signature) and body
        statements. Direct child definitions are replaced with stubs, containing only their headers and docstrings
        to avoid recursive expansion.

        Parameters:
        - `node_id`: The ID of the node for which to assemble the snippet.

        Returns:
        - A string containing the assembled snippet.
        """

        info = info_by_node_id[node_id]
        node = info.node

        header_text = get_text_for_lines(info.header_start, info.header_end)

        body_chunks: Chunk = []
        direct_children: set[int] = set(info.children_ids)

        for stmt in getattr(node, "body", []):
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stmt_id = id(stmt)
                if stmt_id in direct_children:
                    body_chunks.append(make_child_stub(stmt_id))
                    continue
            body_chunks.append(get_statement_source(stmt))

        parts: Chunk = [header_text]
        if body_chunks:
            if header_text and not header_text.endswith("\n"):
                parts.append("\n")
            parts.append("\n".join(body_chunks))
        return "".join(parts)

    # ---------------- indexes from DefInfo ----------------

    # Map node-id → DefInfo (unambiguous identity; avoids name collisions)
    info_by_node_id: Dict[int, DefInfo] = {id(info.node): info for info in defs}

    # ---------------- LLM loop (deepest-first using DefInfo.depth) ----------------

    defs_deepest_first: List[DefInfo] = sorted(defs, key=lambda d: (d.depth, d.start, -d.end), reverse=True)

    for info in defs_deepest_first:
        node_id = id(info.node)
        snippet = assemble_snippet_for(node_id)

        # Elide the body if this routine is too large for the context window (the patch is unaffected).
        header_lines = max(1, info.header_end - info.header_start + 1)
        snippet, omitted = fit_snippet(llm, cfg, messages, snippet, header_lines, MARKER_PYTHON)
        if omitted:
            echo(f"[Python] Elided {omitted} body line(s) from '{info.qualname}' to fit the context window")

        echo("\n[Python] Snippet...\n")
        echo(snippet)

        if info.kind == "class":
            prompt = (
                "Write exactly the docstring for this class, reformatting and updating any existing comment\n"
                "as required. Use the nested method docstrings to help but remember that they are nested so\n"
                f"the class is abstracting over all of them:\n\n{snippet}\n"
            )
        else:
            prompt = (
                "Write exactly the docstring for this program chunk, reformatting and updating any existing\n"
                f"comment as required:\n\n{snippet}\n"
            )
        messages.append({"role": "user", "content": prompt})

        reply = llm.generate(messages, cfg=cfg)
        echo(f"\n[Python] LLM output:\n\n{reply}")

        messages.pop()

        docstring = extract_first_docstring(reply)
        if not docstring:
            docstring = f"{info.kind} `{info.qualname}` - comment generation failed."

        docs_by_node_id[node_id] = docstring

    return docs_by_node_id


def patch_docstrings_textually(source_lines: Chunk, defs: List[DefInfo], doc_map: Dict[int, str]) -> Chunk:
    """
    Replace or insert docstrings in the source code, preserving comments, blank lines, and formatting.

    This function applies edits to the source code in reverse order by start position, ensuring that earlier slices remain valid.

    Parameters:
    - `source_lines`: A list of strings representing the source code lines.
    - `defs`: A list of `DefInfo` objects, each containing information about a definition in the source code.
    - `doc_map`: A dictionary mapping qualified names to docstrings.

    Returns:
    - A list of strings representing the modified source code lines.

    Notes:
    - Uses `DefInfo` to identify definitions and compute the indent for the docstring block.
    - If an existing docstring is present, it is replaced; otherwise, a new docstring is inserted immediately after the header block.
    - Preserves surrounding code as-is when replacing an existing docstring.
    - Applies edits in reverse order by start position to ensure stability during updates.
    """

    out_lines = source_lines[:]  # mutable copy

    # Process in reverse source order to keep indices stable
    for info in sorted(defs, key=lambda d: d.start, reverse=True):
        if id(info.node) not in doc_map:
            continue

        doc = doc_map[id(info.node)]

        node = info.node
        has_body = bool(getattr(node, "body", []))

        # A docstring can only be placed cleanly when the first body statement begins its own
        # line. For inline definitions (e.g. `def f(): return 1`, or a body that shares the
        # closing signature line) inserting or replacing would corrupt the code, so skip them.
        if has_body:
            first_stmt = node.body[0]
            prefix = source_lines[first_stmt.lineno - 1][:first_stmt.col_offset]
            if prefix.strip() != "":
                echo(f"Skipping inline definition '{info.qualname}' (body shares the header line)")
                continue

        # Compute indent for the docstring block: one level deeper than the def/class header
        def_line_text = source_lines[info.def_line - 1]
        base_indent = def_line_text[: len(def_line_text) - len(def_line_text.lstrip())] + "    "

        new_doc_lines = [f'{base_indent}"""', *[f"{base_indent}{line}" for line in doc.splitlines()], f'{base_indent}"""']

        existing_doc = (
            has_body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            and isinstance(getattr(node.body[0].value, "value", None), str)
        )

        if existing_doc:
            # Replace the existing docstring (preserve surrounding code as-is)
            ds_start = node.body[0].lineno - 1
            ds_end = (getattr(node.body[0], "end_lineno", node.body[0].lineno)) - 1
            out_lines[ds_start: ds_end + 1] = new_doc_lines
        else:
            # Insert a fresh docstring immediately after the header block.
            # header_end is the line BEFORE the first body statement; insert at 0-based index = header_end
            insert_at = info.header_end  # 1-based → acts as 0-based insertion index
            out_lines[insert_at:insert_at] = new_doc_lines + [""]

    return out_lines


# ---------------------------- within-function block targets ----------------------------


def _is_docstring_stmt(stmt: ast.AST) -> bool:
    """
    Report whether a statement is a docstring (a bare string-constant expression).

    Parameters:
    - `stmt`: The AST statement to test.

    Returns:
    - True if the statement is a string-constant expression, as a leading docstring would be.
    """

    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _sub_statement_lists(stmt: ast.AST) -> List[List[ast.AST]]:
    """
    Return the nested statement lists of a compound statement, so boundary collection can recurse into them.

    Covers the bodies and alternatives of every Python compound statement: `body`/`orelse`/`finalbody`, the body of
    each `except` handler, and the body of each `match` case.

    Parameters:
    - `stmt`: The AST statement to inspect.

    Returns:
    - A list of statement lists (empty for simple statements).
    """

    lists: List[List[ast.AST]] = []
    for attr in ("body", "orelse", "finalbody"):
        block = getattr(stmt, attr, None)
        if block:
            lists.append(block)
    for handler in getattr(stmt, "handlers", []) or []:
        if getattr(handler, "body", None):
            lists.append(handler.body)
    for case in getattr(stmt, "cases", []) or []:  # match-case
        if getattr(case, "body", None):
            lists.append(case.body)
    return lists


def _body_boundaries(node: ast.AST, source_lines: Chunk) -> Tuple[List[int], Dict[int, str]]:
    """
    Compute the lines within a routine body that may legally begin a block, plus their indentation.

    A line qualifies only if it begins exactly one statement at the line's first non-blank column. This naturally drops
    `a; b` lines (two statement starts), inline-compound lines such as `if x: return` (the inner statement shares the
    line), and continuation lines (which are not statement starts). Statement starts are collected at every nesting
    depth, except that a nested function/class is recorded as a single opaque boundary (its decorator-aware header
    line) and is not descended into - its own body is annotated when it is processed as its own target. A leading
    docstring is skipped so no comment is ever placed between a definition and its docstring.

    Parameters:
    - `node`: The function/async-function/class AST node whose body is examined.
    - `source_lines`: The full source split into lines (for indentation and the first-column check).

    Returns:
    - A tuple `(boundary_lines, indent_of)`: the sorted legal boundary lines and a map from each to its leading
      whitespace.
    """

    line_count: Dict[int, int] = {}   # line -> number of statements starting on it
    line_col: Dict[int, int] = {}     # line -> column of the (single) statement start
    nested_lines: set[int] = set()    # boundary lines that are opaque nested definitions

    def record(line: int, col: int, is_nested: bool) -> None:
        """Note that a statement starts at the given line/column."""
        line_count[line] = line_count.get(line, 0) + 1
        line_col[line] = col
        if is_nested:
            nested_lines.add(line)

    def collect(stmts: List[ast.AST], is_top: bool) -> None:
        """Walk a statement list, recording statement starts and recursing into non-definition compound bodies.

        The first statement of an *inner* suite (an `if`/`for`/`while`/`try`/`with`/`else`/`except` body) is never a
        boundary: a blank line as the first line of a suite reads badly. The first statement of the routine's own body
        stays eligible (a blank there just follows the signature/docstring, which is normal).
        """
        for idx, stmt in enumerate(stmts):
            skip = (not is_top) and idx == 0  # first line of an inner suite is not a block start
            if _is_def_node(stmt):
                # A nested definition is one opaque boundary at its decorator-aware header line; do not descend.
                if not skip:
                    record(_header_span(stmt)[0], 0, True)
                continue
            if not skip:
                record(stmt.lineno, getattr(stmt, "col_offset", 0), False)
            for sub in _sub_statement_lists(stmt):
                collect(sub, False)

    body = list(getattr(node, "body", []))
    consider = body[1:] if body and _is_docstring_stmt(body[0]) else body
    collect(consider, True)

    boundary_lines: List[int] = []
    indent_of: Dict[int, str] = {}
    for line, count in line_count.items():
        if count != 1 or not (1 <= line <= len(source_lines)):
            continue
        text = source_lines[line - 1]
        leading = text[: len(text) - len(text.lstrip())]
        # For a real statement start the column prefix must be all whitespace (nested-definition lines are trusted to
        # start their own line and bypass this check, since a decorator's column points past its '@').
        if line not in nested_lines and text[: line_col[line]].strip() != "":
            continue
        boundary_lines.append(line)
        indent_of[line] = leading

    boundary_lines.sort()
    return boundary_lines, indent_of


def iter_block_targets(source_blob: str, source_lines: Chunk) -> List[BlockTarget]:
    """
    Build the within-function block targets for a Python source file.

    Each function, method, and class body becomes one `BlockTarget` carrying its header/body line spans and the set of
    lines that may legally begin a block (see `_body_boundaries`). This is the Python implementation of the language-
    agnostic provider interface consumed by `scale_blocks.annotate_blocks`.

    Parameters:
    - `source_blob`: The complete source text (parsed with the standard library `ast`).
    - `source_lines`: The same source split into individual lines (for indentation and line text).

    Returns:
    - A list of `BlockTarget`, one per routine, in source order.
    """

    tree = ast.parse(source_blob)
    targets: List[BlockTarget] = []

    for info in iter_defs_with_info(tree):
        body = list(getattr(info.node, "body", []))
        if not body:
            continue

        boundary_lines, indent_of = _body_boundaries(info.node, source_lines)
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
            )
        )

    return targets


def generate_language_comments(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    source_blob: str,
    source_lines: Chunk
) -> Chunk:
    """
    Process Python source code and generate new/updated docstrings for each definition.

    This function takes in the language model, generation configuration, message history, source code text,
    and source code lines, and returns a patched source file with updated docstrings.

    Parameters:
    - llm: The language-model interface used to generate text.
    - cfg: Configuration object controlling generation parameters.
    - messages: The conversational message history to extend with new prompts and replies.
    - source_blob: The complete text of the source file as a single string (with original line endings).
    - source_lines: The source file split into individual lines.

    Returns:
    - A patched source file text, containing the new docstrings, split into individual lines.

    Notes:
    - This function processes the source code in multiple stages, including parsing, import identification,
      definition identification, docstring generation, and patch application.
    - The `echo` statements are used for debugging purposes only and can be removed in production builds.
    """

    # Parse the source file
    echo("Parsing Python source code...")
    tree = ast.parse(source_blob)

    # Provide a list of imports to the LLM (if there are any)
    echo("Identifying imports...")
    describe_imports_from_tree(llm, cfg, messages, tree)

    # Find all of the defs
    echo("Identifying definitions...")
    defs = iter_defs_with_info(tree)
    echo(f"Found {len(defs)} definitions")

    echo("Generating docstrings...\n")
    doc_map = generate_docstrings(llm, cfg, messages, defs, source_blob, source_lines)

    echo("Applying Python patches...\n")
    return patch_docstrings_textually(source_lines, defs, doc_map)
