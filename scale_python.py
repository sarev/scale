#!/usr/bin/env python3
"""
This program provides a robust framework for generating or updating docstrings in Python source code, ensuring that the generated
documentation is precise and aligned with the actual code structure.

This program is designed to generate or update docstrings for functions, classes, and properties in Python source code.

Here are the key functionalities and highlights of its internal workings:

1. **DefInfo Data Structure**: The `DefInfo` class captures detailed information about each definition (function, class, property)
   in the source code. It includes attributes like `qualname`, `node`, `start`, `end`, `def_line`, `header_start`, `header_end`,
   and more. This structure helps in precisely identifying and manipulating definitions within the AST.

2. **AST Traversal**: The program uses the Abstract Syntax Tree (AST) to traverse and analyze the source code. Functions like
   `_is_def_node`, `_node_kind`, and `_header_span` are used to identify and extract relevant information from the AST nodes.

3. **Docstring Generation**: The `generate_docstrings` function generates new or updated docstrings for each definition. It processes
   definitions in a depth-first manner, ensuring that parent definitions are handled before their children. The function uses an LLM
   (Large Language Model) to generate docstrings based on the provided prompts and existing code snippets.

4. **Snippet Assembly**: The `assemble_snippet_for` function constructs the code snippet for each definition, including its header
   and body. It replaces direct child definitions with stubs that include their docstrings, ensuring that the overall structure of the
   code is preserved while updating the docstrings.

5. **Docstring Extraction**: The `extract_first_docstring` function extracts the first fenced docstring block from the LLM's response.
   If no docstring is found, it generates a default message indicating that the docstring generation failed.

6. **Patch Application**: The `patch_docstrings_textually` function applies the generated or updated docstrings to the source code.
   It ensures that comments and blank lines are preserved while updating the docstrings. The function processes the definitions in
   reverse order to maintain index stability during updates.

7. **Main Functionality**: The `generate_language_comments` function orchestrates the entire process, from parsing the source code to
   applying the generated docstrings. It uses the other functions to identify definitions, generate docstrings, and apply the patches.

### Highlights

- **Precision in Definition Identification**: The program uses the `DefInfo` class to precisely identify and manipulate definitions
  within the AST.
- **Depth-First Processing**: Definitions are processed in a depth-first manner, ensuring that parent definitions are handled after
  their children.
- **LLM Integration**: The program leverages an LLM to generate docstrings based on the provided prompts and existing code snippets.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from scale_llm import LocalChatModel, GenerationConfig, Messages, Chunk
from scale_log import echo
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
    - n: The AST node to check.

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
    Return the start and end lines for the node's header, including decorators and the full (possibly multi-line) signature, ending on the line before the first body statement.

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
    Detect @prop.getter / @prop.setter / @prop.deleter on a function.

    Returns:
        (property_name, role) where role ∈ {"getter","setter","deleter"}, or None.
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
        Add a child node to the parent's list of children.

        Parameters:
        - `parent_node`: The parent node in the Abstract Syntax Tree (AST).
        - `child_node`: The child node to be added.

        If the parent node is `None`, no action is taken.
        """

        if parent_node is None:
            return
        pid = id(parent_node)
        children_map.setdefault(pid, []).append(id(child_node))

    def walk(node: ast.AST) -> None:
        """
        Traverse and process each child node of the given AST node.

        This function recursively traverses the Abstract Syntax Tree (AST) to process each child node. For each
        definition node, it builds a qualified name (qualname) and captures relevant line information such as
        start, end, and definition lines. The qualname is constructed based on the node's type and any property
        accessors, and is stored in the `DefInfo` data structure along with other metadata.

        Parameters:
        - `node`: The current AST node to process.

        Returns:
        - None: This function does not return a value; it populates the `results` list with `DefInfo` objects.
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


def deepest_first(defs: List[DefInfo]) -> List[DefInfo]:
    # depth desc; for stability: start asc, end desc
    """
    Sort the definitions in deepest-first order, ensuring stability by starting with ascending depths, then start positions, and finally descending end positions.

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
) -> Dict[str, str]:
    """
    Generate or update docstrings for definitions in a Python source code AST.

    This function processes the deepest definitions first, assembling snippets that include
    headers and bodies while replacing direct child definitions with stubs containing their
    docstrings. Non-definition statements are included verbatim from the source.

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
        Extract the first fenced docstring block (\"\"\", ''' or ```). If none,
        treat the entire reply as the candidate. Dedent and strip. Returns "" if empty.

        Parameters:
        - `reply`: The input string from which to extract the docstring.

        Returns:
        - The extracted docstring, dedented and stripped. Returns "" if no docstring is found.
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
        Return source text for inclusive 1-based line range [line_a, line_b].

        Parameters:
        - line_a: The start of the line range (1-based).
        - line_b: The end of the line range (1-based).

        Returns:
        - A string containing the source text for the specified line range.
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

    docs_by_qualname: Dict[str, str] = {}   # external return mapping
    docs_by_node_id: Dict[int, str] = {}    # internal disambiguation

    def make_child_stub(child_node_id: int) -> str:
        """
        Return 'decorators + header + child docstring' for a direct child.

        Parameters:
        - `child_node_id`: The ID of the child node for which to generate the stub.

        Returns:
        - A string containing the combined decorators, header, and child docstring.
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
        For the given node:
        - include header (decorators + signature),
        - include body statements verbatim except:
          direct child definitions are replaced by stubs (header + docstring).

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
    # Map qualname → node-id list (only used if you need to resolve duplicates externally)
    node_ids_by_qualname: Dict[str, List[int]] = {}
    for info in defs:
        node_ids_by_qualname.setdefault(info.qualname, []).append(id(info.node))

    # ---------------- LLM loop (deepest-first using DefInfo.depth) ----------------

    defs_deepest_first: List[DefInfo] = sorted(defs, key=lambda d: (d.depth, d.start, -d.end), reverse=True)

    for info in defs_deepest_first:
        node_id = id(info.node)
        snippet = assemble_snippet_for(node_id)

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

        docs_by_qualname[info.qualname] = docstring
        docs_by_node_id[node_id] = docstring

    return docs_by_qualname


def patch_docstrings_textually(source_lines: Chunk, defs: List[DefInfo], doc_map: Dict[str, str]) -> Chunk:
    """
    Return new source lines with docstrings replaced or inserted, preserving all comments,
    blank lines, and formatting.

    Parameters:
    - `source_lines`: A list of strings representing the source code lines.
    - `defs`: A list of `DefInfo` objects, each containing information about a definition in the source code.
    - `doc_map`: A dictionary mapping qualified names to docstrings.

    Returns:
    - A list of strings representing the modified source code lines.

    Notes:
    - Uses `DefInfo` (qualname, node, start/end, def_line, header_start/header_end, etc.) to identify definitions.
    - Applies edits in reverse order by start position so earlier slices remain valid.
    - If an existing docstring is present (first body expression is a string), it is replaced.
      Otherwise, a new docstring is inserted immediately after the header block.
    """

    out_lines = source_lines[:]  # mutable copy

    # Process in reverse source order to keep indices stable
    for info in sorted(defs, key=lambda d: d.start, reverse=True):
        qualname = info.qualname
        if qualname not in doc_map:
            continue

        doc = doc_map[qualname]

        # Compute indent for the docstring block: one level deeper than the def/class header
        def_line_text = source_lines[info.def_line - 1]
        base_indent = def_line_text[: len(def_line_text) - len(def_line_text.lstrip())] + "    "

        new_doc_lines = [f'{base_indent}"""', *[f"{base_indent}{line}" for line in doc.splitlines()], f'{base_indent}"""']

        node = info.node
        has_body = bool(getattr(node, "body", []))

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


def generate_language_comments(
    llm: LocalChatModel,
    cfg: GenerationConfig,
    messages: Messages,
    source_blob: str,
    source_lines: Chunk
) -> Chunk:
    """
    Process Python source code and generate new/updated docstrings for each def/class.

    Parameters:
    - llm: The language-model interface used to generate text.
    - cfg: Configuration object controlling generation parameters.
    - messages: The conversational message history to extend with new prompts and replies.
    - source_blob: The complete text of the source file as a single string (with original line endings).
    - source_lines: The source file split into individual lines.

    Returns:
    - A patched source file text, containing the new docstrings, split into individual lines.
    """

    # Parse the source file
    echo("Parsing Python source code...")
    tree = ast.parse(source_blob)

    # Find all of the defs
    echo("Identifying definitions...")
    defs = iter_defs_with_info(tree)
    echo(f"Found {len(defs)} definitions")

    echo("Generating docstrings...\n")
    doc_map = generate_docstrings(llm, cfg, messages, defs, source_blob, source_lines)

    echo("Applying Python patches...\n")
    return patch_docstrings_textually(source_lines, defs, doc_map)
