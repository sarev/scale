#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The project-context layer that sits above SCALE's per-file pipeline, giving each annotation run sight of the wider
codebase. It finds and distils the project's overview document into a short cached blurb, expands the CLI's path
patterns into a deduplicated file list, and scans every target and reference file into a keyed store of `RunFile`
records.

From those records `build_project_graph` constructs the run-wide call graph: call sites are resolved conservatively (a
missing edge always beats a wrong one), and an iterative Tarjan-plus-Kahn ordering places callees before callers, both
within a file (`doc_order`) and across files (`file_order`). The `ContractStore` keeps a one-line behavioural contract
per symbol - seeded from existing docstrings and refreshed as new ones are written - and renders capped callee-notes
blocks for generation prompts.

The module also renders the compact structural skeletons (header zone plus each symbol's signature and doc) used to
summarise files cheaply, and provides small shared utilities such as `apply_doc_order` and tolerant cached text I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple
import glob
import hashlib
import heapq

from scale_text import summarise, LENGTH_PARAGRAPH
from scale_log import echo
import re


# Source extensions used when a target/reference path is a directory (so a directory expands to just its source files,
# not every file). Explicitly named files and glob matches are taken as-is regardless of extension.
SOURCE_EXTS = (".py", ".c", ".h", ".js", ".mjs", ".cjs")


# Reply-length cap for the project blurb. It is background context shown above many files, so it is kept to a couple of
# sentences regardless of how large the source overview document is.
PROJECT_BLURB_MAX_TOKENS = 256

# Built-in default for the blurb instruction (overridable via scale-cfg/project.txt).
PROJECT_BLURB_INSTRUCTION = (
    "Summarise what this software project is, for a developer who is about to read its source files. In two or three "
    "sentences say what the project does, its domain, and any key concepts or terminology a reader should know. This "
    "is background shown above individual files, so keep it short and general: do not describe individual files, "
    "functions, or APIs, and do not pad with build/usage/installation detail. Plain prose - no headings, no lists."
)

# The cache lives alongside SCALE's summary cache (same directory convention, no import dependency on scale.py).
_CACHE_DIR = Path(__file__).resolve().parent / "__cache__"

# Project-overview document discovery. `CLAUDE.md` is preferred; otherwise a README with any common extension, any case.
_PREFERRED_NAME = "claude.md"
_README_STEM = "readme"
_DOC_SUFFIXES = ("", ".md", ".markdown", ".rst", ".txt")


def _read_text_bytes(path: Path) -> str:
    """
    Read a file as UTF-8 text, tolerating undecodable bytes.

    Parameters:
    - `path`: The file to read.

    Returns:
    - The decoded contents, or an empty string if the file cannot be read.
    """

    # A byte-level read with surrogateescape tolerates non-UTF-8 content; any I/O failure degrades to an empty string.
    try:
        return path.read_bytes().decode("utf-8", errors="surrogateescape")
    except OSError:
        return ""


def _read_optional(path: Path) -> Optional[str]:
    """
    Read a UTF-8 text file that may legitimately be absent.

    Parameters:
    - `path`: The file to read.

    Returns:
    - The file contents, or `None` if the file does not exist.
    """

    # Only a missing file maps to `None`; other I/O errors still propagate.
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def find_project_doc(start: Path, max_levels: int = 8) -> Optional[Path]:
    """
    Locate the project's overview document by walking up from a starting path.

    From the directory containing `start`, each parent level is searched in turn: a file with the preferred project-doc name wins outright, otherwise any README bearing a recognised documentation suffix is taken, Markdown first. The climb stops once the repository root (a directory holding `.git`, searched inclusively) or the filesystem root has been scanned.

    Parameters:
    - `start`: File or directory from which to begin the upward search.
    - `max_levels`: Maximum number of directory levels to examine.

    Returns:
    - The path of the chosen document, or `None` if no candidate was found.
    """

    # Anchor the climb at a directory, even when handed a file path.
    d = (start if start.is_dir() else start.parent).resolve()

    # Treat an unreadable directory as empty rather than aborting the climb.
    for _ in range(max_levels):
        try:
            entries = [p for p in d.iterdir() if p.is_file()]
        except OSError:
            entries = []

        # A dedicated project doc at this level beats any README.
        preferred = [p for p in entries if p.name.lower() == _PREFERRED_NAME]
        if preferred:
            return preferred[0]
        readmes = [p for p in entries if p.stem.lower() == _README_STEM and p.suffix.lower() in _DOC_SUFFIXES]

        # Failing that, fall back to READMEs with a recognised documentation suffix.
        if readmes:
            # Prefer Markdown, then name order, so the pick is deterministic.
            readmes.sort(key=lambda p: (p.suffix.lower() != ".md", p.name))
            return readmes[0]

        # Stop once the repository root has been scanned so the search never escapes the project.
        if (d / ".git").exists() or d.parent == d:   # repo root (inclusive) or filesystem root
            break
        d = d.parent

    return None


def _blurb_cache_path(doc_text: str) -> Path:
    """
    Derive the cache file path for a project blurb from the document it was distilled from.

    The name embeds a SHA-256 digest of the document text, so any edit to the document maps to a fresh cache entry.

    Parameters:
    - `doc_text`: The overview document text being distilled.

    Returns:
    - The blurb's cache file path under the shared cache directory.
    """

    digest = hashlib.sha256(doc_text.encode("utf-8", errors="surrogateescape")).hexdigest()
    return _CACHE_DIR / f"project-{digest}.txt"


def _read_cache(path: Path) -> Optional[str]:
    """
    Read a cached text entry, treating a missing file as a cache miss.

    Parameters:
    - `path`: The cache file to read.

    Returns:
    - The decoded file contents, or `None` if the file does not exist.
    """

    try:
        return path.read_bytes().decode("utf-8", errors="surrogateescape")
    except FileNotFoundError:
        return None


def _write_cache(path: Path, text: str) -> None:
    """
    Atomically write a text entry to the cache.

    The text lands in a sibling temporary file which is then renamed into place, so a concurrent reader never observes a partially written entry. Parent directories are created as needed.

    Parameters:
    - `path`: The destination cache file.
    - `text`: The text to store.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(text.encode("utf-8", errors="surrogateescape"))
    tmp.replace(path)


def _crop_to_budget(text: str, llm, budget_tokens: int) -> str:
    """
    Crop text to a token budget by keeping whole leading lines.

    Text that already fits is returned untouched. Otherwise lines are accumulated from the top, re-estimating the joined text at each step, and the line that tips over the budget is dropped; an explicit truncation marker is appended so the model knows the document was cut short.

    Parameters:
    - `text`: The document text to crop.
    - `llm`: Model handle supplying `estimate_tokens` for the budget checks.
    - `budget_tokens`: Maximum token count allowed for the result.

    Returns:
    - The original text if it fits, or the cropped prefix ending in a truncation notice.
    """

    # Fast path: the document already fits the budget.
    if llm.estimate_tokens(text) <= budget_tokens:
        return text
    kept: list[str] = []

    # Accumulate whole lines so the crop never splits mid-line.
    for line in text.splitlines():
        kept.append(line)

        # Re-estimate the full join each step - token counts are not additive per line - and drop the line that overshoots.
        if llm.estimate_tokens("\n".join(kept)) > budget_tokens:
            kept.pop()
            break

    # Flag the cut explicitly so the model knows the overview is incomplete.
    return "\n".join(kept) + "\n\n[... overview document truncated ...]"


def project_blurb(llm, cfg, scale_path: Path, doc_path: Path, no_cache: bool = False) -> str:
    """
    Distil a project's overview document into a short, cacheable blurb.

    The cache key is derived from the document's content, so the cached blurb is reused until the document changes; an empty document yields an empty blurb without invoking the model.

    Parameters:
    - `llm`: The local chat model used to summarise the document.
    - `cfg`: The run configuration passed through to `summarise`.
    - `scale_path`: The run-state directory; an optional `project.txt` there overrides the stock distillation instruction.
    - `doc_path`: Path of the project overview document (e.g. a README).
    - `no_cache`: When `True`, ignore any cached blurb and regenerate.

    Returns:
    - The distilled blurb, or an empty string if the document is empty.
    """

    # An empty overview document yields no blurb; the cache key hashes the document's content, so edits invalidate it.
    doc_text = _read_text_bytes(doc_path)
    if not doc_text.strip():
        return ""
    cache_path = _blurb_cache_path(doc_text)

    # Reuse a previous distillation unless the caller has forced a fresh one.
    if not no_cache:
        cached = _read_cache(cache_path)

        if cached is not None:
            echo(f"Loaded project blurb from cache ({doc_path.name})...")
            return cached

    # A project-local project.txt overrides the stock instruction, and the document is cropped so prompt plus reply fit the model's context window.
    echo(f"Distilling project overview from {doc_path.name}...")
    instruction = _read_optional(scale_path / "project.txt") or PROJECT_BLURB_INSTRUCTION
    budget = max(256, llm.n_ctx - llm.ctx_margin - PROJECT_BLURB_MAX_TOKENS - 64)
    cropped = _crop_to_budget(doc_text, llm, budget)
    blurb = summarise(llm, cfg, cropped, LENGTH_PARAGRAPH, subject="a software project's overview document",
                      max_tokens=PROJECT_BLURB_MAX_TOKENS, instruction=instruction)
    _write_cache(cache_path, blurb)

    return blurb


def gather_files(patterns: List[str], exts: Tuple[str, ...] = SOURCE_EXTS) -> List[Path]:
    """
    Expand a mixed list of path patterns into a deduplicated, sorted list of source files.

    Each pattern may be a glob (expanded recursively), a directory (walked recursively, keeping only files with the given extensions), or a literal file path; explicit paths bypass the extension filter.

    Parameters:
    - `patterns`: Glob patterns, directories, or file paths to expand.
    - `exts`: Lower-case file extensions accepted when walking directories.

    Returns:
    - The matching files, deduplicated by resolved path and sorted by path string for a deterministic order.
    """

    out: List[Path] = []
    seen = set()

    for pattern in patterns:
        p = Path(pattern)

        # Each pattern is interpreted as a glob, a directory to walk recursively (extension-filtered), or a literal file path.
        if any(ch in pattern for ch in "*?["):
            candidates = [Path(m) for m in glob.glob(pattern, recursive=True)]
        elif p.is_dir():
            candidates = [c for c in sorted(p.rglob("*")) if c.suffix.lower() in exts]
        else:
            candidates = [p]

        # Dedupe on the resolved path so the same file reached via different patterns is kept once.
        for c in candidates:
            if not c.is_file():
                continue
            key = c.resolve()

            if key not in seen:
                seen.add(key)
                out.append(c)

    # Sort by path string so the run order is deterministic across platforms.
    return sorted(out, key=lambda x: str(x))


@dataclass
class RunFile:

    """
    One source file participating in a multi-file run, holding its text, language, and parsed symbols.

    `is_target` distinguishes files being annotated from files retained purely as project context; `key` identifies the file in the run store.
    """

    # `key` identifies the file in the retained run store; `is_target` separates files to annotate from context-only ones.
    path: Path
    key: str
    is_target: bool
    source_blob: str
    source_lines: List[str]
    language: str
    symbols: List["Symbol"]


def scan_run_files(
    targets: List[Path],
    references: List[Path],
    load: Callable,
    provider_for: Callable,
) -> Dict[str, RunFile]:
    """
    Scan every target and reference file into a keyed store of `RunFile` records.

    Targets are walked before references, so a file named in both roles is recorded as a target. Entries are keyed (and deduplicated) by resolved absolute path; unreadable files and languages with no symbol provider are skipped silently.

    Parameters:
    - `targets`: Files this run will annotate.
    - `references`: Read-only context files.
    - `load`: Loader returning `(blob, lines, line_ending, language)` for a path.
    - `provider_for`: Maps a language name to a symbol-extraction callable, or `None` if the language is unsupported.

    Returns:
    - A dict mapping resolved path strings to `RunFile` records.
    """

    # Resolve target paths up front so each entry can record whether it is a target or merely a reference.
    target_keys = {str(t.resolve()) for t in targets}
    out: Dict[str, RunFile] = {}

    # Targets walk first, so a file listed in both roles lands as a target; resolved paths deduplicate aliases.
    for f in list(targets) + list(references):
        key = str(f.resolve())
        if key in out:
            continue

        # An unreadable file is skipped rather than aborting the whole run.
        try:
            blob, lines, _le, lang = load(f)
        except OSError:
            continue

        # Languages without a symbol provider are skipped; otherwise the file's symbols are parsed once and retained for the run.
        provider = provider_for(lang)
        if provider is None:
            continue
        out[key] = RunFile(path=f, key=key, is_target=key in target_keys,
                           source_blob=blob, source_lines=lines, language=lang, symbols=provider(blob, lines))

    return out


# ============================================================================
# The file skeleton (model-free distillation for the whole-file description)
#
# Summarising a whole source file pays for every body line, yet a file DESCRIPTION draws almost entirely on its
# signatures, its existing docs, and its header comments. The skeleton renderer distils a file to exactly that -
# leading comments, function/method signatures, class headers with their method prototypes, C declarations and
# top-level #defines, and each symbol's existing doc; NO bodies - typically a small fraction of the file, so the
# description is generated in a single call and map-reduce becomes rare. The guard is binary: a file with no symbols
# at all (no functions or classes - e.g. a data table or a config) keeps today's whole-file path, map-reduce included.
# ============================================================================


# A C top-level #define (the macro's first line carries its name and intent; a multi-line body is elided).
_C_DEFINE_RE = re.compile(r"^\s*#\s*define\b")


def _leading_zone(source_lines: List[str], language: str) -> List[str]:
    """
    Collect a file's leading shebang/comment/docstring zone, stopping at the first line of real code.

    The zone is taken verbatim: an optional shebang, blank lines, line comments, and any block comment or module docstring (followed through to its closing delimiter). Trailing blank lines are trimmed.

    Parameters:
    - `source_lines`: The file's lines, without line endings.
    - `language`: One of `python`, `c` or `js`; selects which comment syntaxes are recognised.

    Returns:
    - The zone's lines in order, or an empty list if the file opens straight with code.
    """

    out: List[str] = []
    n = len(source_lines)
    i = 0

    # A shebang belongs to the header zone even though it is not a comment.
    if i < n and source_lines[i].lstrip().startswith("#!"):
        out.append(source_lines[i])
        i += 1

    closer = None       # the token that ends the comment/docstring block we are inside, or None

    while i < n:
        line = source_lines[i]
        s = line.strip()

        # Inside a multi-line comment or docstring, take lines verbatim until the closing token appears.
        if closer is not None:
            out.append(line)
            if closer in s:
                closer = None
            i += 1
            continue

        # Blank lines within the zone are kept; trailing ones are trimmed at the end.
        if not s:
            out.append(line)
            i += 1
            continue

        # A module docstring counts as header material, tracked through to its closing delimiter.
        if language == "python" and (s.startswith('"""') or s.startswith("'''")):
            quote = s[:3]
            out.append(line)
            if not (s.count(quote) >= 2):      # an opening delimiter without its close on the same line
                closer = quote
            i += 1
            continue

        if language == "python" and s.startswith("#"):
            out.append(line)
            i += 1
            continue

        if language in ("c", "js") and s.startswith("//"):
            out.append(line)
            i += 1
            continue

        # Searching past the opener keeps a lone /*/ from counting as self-closing.
        if language in ("c", "js") and s.startswith("/*"):
            out.append(line)
            if "*/" not in s[2:]:
                closer = "*/"
            i += 1
            continue

        break                                  # the first real code line ends the zone

    while out and not out[-1].strip():
        out.pop()
    return out


def _skeleton_doc_lines(doc: str, indent: str, language: str) -> List[str]:
    """
    Render an existing doc-comment as skeleton lines in the language's native form.

    Parameters:
    - `doc`: The doc text; may be empty or `None`.
    - `indent`: The owning symbol's leading indentation.
    - `language`: `python` renders a docstring one level deeper than the signature; anything else renders a `/* ... */` block at the same indent.

    Returns:
    - The formatted lines, or an empty list when there is no doc text.
    """

    # A doc that is empty after trimming renders nothing at all.
    lines = [ln.rstrip() for ln in (doc or "").splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        return []

    if language == "python":
        # Python docs render as a docstring indented one level inside the body.
        inner = indent + "    "
        return [f'{inner}"""'] + [f"{inner}{ln}".rstrip() for ln in lines] + [f'{inner}"""']

    # C and JS docs render as a block comment at the signature's own indent.
    return [f"{indent}/*"] + [f"{indent} * {ln}".rstrip() for ln in lines] + [f"{indent} */"]


def render_skeleton(source_lines: List[str], language: str, symbols: List["Symbol"]) -> Optional[str]:
    """
    Render a compact skeleton of a file: its header zone, C `#define` lines, and each symbol's signature with any existing doc.

    Symbols appear in source order, indented by nesting depth; a prototype is skipped when its full definition is also present. Docs sit where the language expects them - below the signature for Python, above it for C/JS.

    Parameters:
    - `source_lines`: The file's lines, without line endings.
    - `language`: One of `python`, `c` or `js`.
    - `symbols`: The file's parsed symbols.

    Returns:
    - The skeleton as one newline-joined string, or `None` when there are no symbols.
    """

    # A file with no recognised symbols has no useful skeleton.
    if not symbols:
        return None
    parts: List[str] = []
    zone = _leading_zone(source_lines, language)

    # The file's own header comment zone opens the skeleton.
    if zone:
        parts.extend(zone)
        parts.append("")

    # C #define lines carry interface information, so they join the skeleton.
    if language == "c":
        defines = []

        # A multi-line macro is cut at its first line, with an ellipsis standing in for the continuation.
        for ln in source_lines:
            if _C_DEFINE_RE.match(ln):
                text = ln.rstrip()
                if text.endswith("\\"):
                    text = text.rstrip("\\").rstrip() + " ..."
                defines.append(text)

        if defines:
            parts.extend(defines)
            parts.append("")

    # Names with a full definition make their separate prototypes redundant.
    defined = {s.qualname for s in symbols if s.kind != "declaration"}

    # Emit symbols in source order, re-indenting continuation lines flush with the symbol's nesting depth.
    for sym in sorted(symbols, key=lambda s: s.start):
        if sym.kind == "declaration" and sym.qualname in defined:
            continue                            # the definition's signature already shows this prototype
        indent = "    " * max(0, sym.depth)
        signature = "\n".join(f"{indent}{ln.strip()}" if i else f"{indent}{ln.rstrip()}"
                              for i, ln in enumerate((sym.signature or "").splitlines()))
        doc = _skeleton_doc_lines(sym.existing_doc, indent, language)

        if language == "python":
            parts.append(signature)        # a Python doc reads as a docstring under its signature
            parts.extend(doc)
        else:
            parts.extend(doc)              # a C/JS doc comment sits above its signature
            parts.append(signature)

        parts.append("")

    while parts and not parts[-1].strip():
        parts.pop()
    return "\n".join(parts)


def resolve_project_doc(project_doc_arg: str, start: Path) -> Optional[Path]:
    """
    Resolve the project-doc CLI argument to a concrete document path.

    The literal `none` (case-insensitive) disables the project doc. An explicit path is used only if the file exists - a missing file yields `None` rather than an error. An empty argument falls back to auto-discovery via `find_project_doc`.

    Parameters:
    - `project_doc_arg`: The raw argument: a path, the literal `none`, or empty.
    - `start`: Directory from which auto-discovery starts when no path is given.

    Returns:
    - The document's path, or `None` when disabled or not found.
    """

    # The literal `none` lets the user disable the project doc outright.
    if project_doc_arg.strip().lower() == "none":
        return None

    if project_doc_arg:
        # An explicit path must exist; a missing file means no doc rather than an error.
        p = Path(project_doc_arg)
        return p if p.is_file() else None

    return find_project_doc(start)


# ============================================================================
# Call-graph-aware documentation
#
# SCALE documents routines in isolation, so a function's docstring cannot draw on what the functions it *calls* do, and
# routines are ordered by nesting rather than by call dependency. This layer adds a **model-free pre-pass** that parses
# every run file into symbols + calls, builds a call graph, and lets the definition pass (1) document callees before
# callers (leaf-first), and (2) inject the one-line contracts of each routine's resolved callees into its generation
# turn. Contracts come from existing docstrings (a seed), the docstrings the def pass itself generates (each one's
# first line updates the contract for later callers), and - for a *called but undocumented* routine - a one-liner the
# orchestrator generates lazily at the first caller that needs it (cached on the store; see scale.py). The
# byte-for-byte code guarantee is untouched; nothing here patches source.
#
# Call resolution is **confident-only** - it never guesses a receiver type. A call links only when it can be resolved
# safely: a free function by name (same-file first, else a run-wide unique match); `self`/`this` method calls to the
# enclosing class's own method; and `obj.m()` only when the method name `m` is defined by exactly one class across the
# whole run. Everything else (typed-receiver dispatch, function pointers, dynamic calls) stays unresolved and simply
# contributes no note.
# ============================================================================


# How many callee contracts to inject into a routine's generation turn (kept small so the per-turn context stays tight).
CALLEE_NOTES_CAP = 6


def apply_doc_order(items: List, qualname_of: Callable, doc_order: List[str], fallback_key: Callable) -> List:
    """
    Sort items so those named in the documentation order come first, in that order.

    Parameters:
    - `items`: The items to sort.
    - `qualname_of`: Maps an item to the qualified name looked up in `doc_order`.
    - `doc_order`: Qualified names in their desired output order.
    - `fallback_key`: Secondary key, ordering items absent from `doc_order` and breaking ties.

    Returns:
    - A new list sorted by doc-order position, then by the fallback key.
    """

    # The sentinel ranks anything not named in doc_order after everything that is.
    pos = {q: i for i, q in enumerate(doc_order)}
    sentinel = len(pos)

    # Listed items keep the documentation order; unlisted ones fall back to their natural key.
    return sorted(items, key=lambda it: (pos.get(qualname_of(it), sentinel), fallback_key(it)))


@dataclass
class Symbol:

    """
    One symbol (routine, class or declaration) parsed from a source file.

    `calls` records the `(name, kind)` call sites found in the body; `file` defaults to empty and is stamped when the project graph is built, not by the parser.
    """

    # `file` defaults to empty and is stamped later by the graph builder.
    qualname: str
    kind: str
    signature: str
    start: int
    depth: int
    parent_qualname: Optional[str]
    existing_doc: str
    calls: List[Tuple] = field(default_factory=list)
    file: str = ""
    end: int = 0


# A symbol key uniquely identifies a routine across the run: (file, qualname).
SymKey = Tuple[str, str]


def _first_line(text: str) -> str:
    """
    Return the first non-blank line of `text`, stripped of surrounding whitespace.

    Parameters:
    - `text`: The text to scan; `None` is tolerated and treated as empty.

    Returns:
    - The first non-empty line, stripped, or an empty string when there is none.
    """

    # The `or ""` guard lets a None value pass through as empty rather than raising.
    for line in (text or "").splitlines():
        s = line.strip()
        if s:
            return s

    return ""


def _tarjan_sccs(nodes: List, succ: Dict) -> List[List]:
    """
    Find the strongly connected components of a directed graph using an iterative Tarjan's algorithm.

    The depth-first search is driven by an explicit work stack rather than recursion, so arbitrarily deep graphs cannot hit Python's recursion limit. A component is emitted only after every component it has an edge to, i.e. in reverse topological order.

    Parameters:
    - `nodes`: All nodes in the graph; a search is started from each unvisited one.
    - `succ`: Mapping from a node to its successors; absent nodes have none.

    Returns:
    - A list of components, each a list of nodes, in reverse topological order.
    """

    # Tarjan bookkeeping: discovery index, low-link, and the stack holding the current component path.
    index: Dict = {}
    low: Dict = {}
    on_stack: Set = set()
    stack: List = []
    sccs: List[List] = []
    counter = 0

    # Open an explicit DFS frame for each node not already reached by an earlier search.
    for root in nodes:
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(succ.get(root, ())))]

        # Peek rather than pop: a frame stays live until all its successors are exhausted.
        while work:
            node, it = work[-1]
            descended = False

            # Descend into the first unvisited successor, suspending this frame mid-iteration.
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(succ.get(w, ()))))
                    descended = True
                    break

                # A successor still on the stack is a back-edge into the current component.
                if w in on_stack:
                    low[node] = min(low[node], index[w])

            # After a descent the child runs first; this frame resumes later with the same iterator.
            if descended:
                continue

            # A node whose low-link still equals its own index roots a component.
            if low[node] == index[node]:
                comp: List = []

                # Everything stacked above and including the root is one strongly connected component.
                while True:
                    x = stack.pop()
                    on_stack.discard(x)
                    comp.append(x)
                    if x == node:
                        break

                sccs.append(comp)

            work.pop()

            # Fold the finished child's low-link into its parent - the post-recursion step done by hand.
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])

    return sccs


def _leaf_first_order(nodes: List, succ: Dict, tiebreak: Callable) -> List:
    """
    Topologically order nodes so each precedes its successors, with cycles condensed and ties broken deterministically.

    Strongly connected components are computed first, then Kahn's algorithm runs over the acyclic component graph using a heap keyed by `tiebreak`, so the result never depends on dict iteration order. Members of a cycle are emitted together, sorted by the same key.

    Parameters:
    - `nodes`: All nodes to order.
    - `succ`: Mapping from a node to the nodes that must come after it.
    - `tiebreak`: Key function giving each node a sortable, deterministic priority.

    Returns:
    - Every node exactly once, in leaf-first order.
    """

    # Condense cycles first; ordering then runs on the acyclic component graph.
    sccs = _tarjan_sccs(nodes, succ)
    comp_of: Dict = {}

    # Map each node to its component for the edge collapse below.
    for i, comp in enumerate(sccs):
        for n in comp:
            comp_of[n] = i

    # Component-level adjacency and in-degrees, ready for Kahn's algorithm.
    cedges: Dict[int, Set[int]] = {i: set() for i in range(len(sccs))}
    indeg: Dict[int, int] = {i: 0 for i in range(len(sccs))}

    # Collapse the node-level edges onto the component graph.
    for n in nodes:
        cn = comp_of[n]

        for m in succ.get(n, ()):                      # n precedes m -> component cn precedes cm
            cm = comp_of[m]

            # Count each cross-component edge once so the in-degrees stay honest.
            if cn != cm and cm not in cedges[cn]:
                cedges[cn].add(cm)
                indeg[cm] += 1

    # Seed the heap with the dependency-free components, keyed for deterministic pops.
    comp_key = [min(tiebreak(n) for n in comp) for comp in sccs]
    heap = [(comp_key[i], i) for i in range(len(sccs)) if indeg[i] == 0]
    heapq.heapify(heap)
    out: List = []

    # Emit the smallest-keyed ready component, its members sorted by the same key.
    while heap:
        _, i = heapq.heappop(heap)
        out.extend(sorted(sccs[i], key=tiebreak))

        # Releasing a component may make its dependants ready.
        for cm in sorted(cedges[i]):
            indeg[cm] -= 1
            if indeg[cm] == 0:
                heapq.heappush(heap, (comp_key[cm], cm))

    return out


@dataclass
class ProjectGraph:

    """
    Project-wide call graph over every symbol parsed in the run.

    Keys are `(file, qualname)` pairs: `symbols` maps each key to its `Symbol`, `edges` lists each caller's resolved callees, `order` holds every symbol leaf-first (callees before callers), and `call_map` records which symbol each individual call site resolved to.
    """

    # Every map is keyed by `(file, qualname)`, so duplicate names across files stay distinct.
    symbols: Dict[SymKey, Symbol]
    edges: Dict[SymKey, List[SymKey]]
    order: List[SymKey]
    call_map: Dict[SymKey, Dict[Tuple[str, str], SymKey]] = field(default_factory=dict)

    def doc_order(self, file: str) -> List[str]:
        """
        Return the qualified names defined in one file, in leaf-first documentation order.

        Parameters:
        - `file`: The file whose symbols are wanted.

        Returns:
        - The qualnames from the run-wide order, callees before callers, restricted to `file`.
        """

        return [q for (f, q) in self.order if f == file]

    def file_order(self, target_files: List[str]) -> List[str]:
        """
        Order the run's files so that files defining callees come before the files that call into them.

        Symbol-level call edges are collapsed onto whole files and sorted with the shared leaf-first machinery; files that call into each other form a cycle and keep their relative input order. The caller-supplied order is the tiebreak, making the result deterministic.

        Parameters:
        - `target_files`: The run's files in their original (command-line) order.

        Returns:
        - The same files, reordered so callee files come first.
        """

        # Command-line position doubles as the deterministic tiebreak.
        idx = {f: i for i, f in enumerate(target_files)}
        tset = set(target_files)
        succ: Dict[str, Set[str]] = {f: set() for f in target_files}

        # Only callers inside the requested file set contribute ordering edges.
        for caller, callees in self.edges.items():
            cf = caller[0]
            if cf not in tset:
                continue

            # Cross-file calls collapse to file-level precedence; same-file calls are irrelevant here.
            for callee in callees:
                kf = callee[0]
                if kf != cf and kf in tset:
                    succ[kf].add(cf)   # the callee's file precedes the caller's file

        return _leaf_first_order(target_files, succ, tiebreak=lambda f: idx[f])


def build_project_graph(symbols_by_file: Dict[str, List[Symbol]]) -> ProjectGraph:
    """
    Build the run-wide call graph from each file's parsed symbols.

    Call sites are resolved conservatively - a name links only when its target is unambiguous - so a missing edge is always preferred to a wrong one. As a side effect, every `Symbol` is stamped with its owning file. The resulting order places callees before callers and nested routines before their parents.

    Parameters:
    - `symbols_by_file`: The parsed symbols of every file in the run, keyed by file path.

    Returns:
    - A `ProjectGraph` carrying the symbol table, resolved edges, leaf-first order and per-call-site resolution map.
    """

    symbols: Dict[SymKey, Symbol] = {}

    # Flatten into one `(file, qualname)` table, stamping each symbol with its owning file.
    for file, syms in symbols_by_file.items():
        for s in syms:
            s.file = file
            symbols[(file, s.qualname)] = s

    # Simple-name indexes that drive the call resolution below.
    free_funcs: Dict[str, List[SymKey]] = {}
    methods_by_name: Dict[str, List[SymKey]] = {}

    # Walk the table once to bucket resolution targets by unqualified name.
    for key, s in symbols.items():
        file, q = key
        simple = q.rsplit(".", 1)[-1]
        if s.kind == "declaration":
            continue   # a C prototype seeds a contract but is never a resolution target (the definition is)

        # Only methods of a real class are indexed; defs nested inside functions are never targets.
        if s.parent_qualname is None:
            free_funcs.setdefault(simple, []).append(key)
        else:
            parent = symbols.get((file, s.parent_qualname))
            if parent is not None and parent.kind == "class":
                methods_by_name.setdefault(simple, []).append(key)

    # Deliberately conservative: any ambiguity yields None - a missing edge beats a wrong one.
    def resolve(file: str, s: Symbol, name: str, ckind: str) -> Optional[SymKey]:
        """
        Resolve one recorded call site to its target's symbol key, or `None` when no safe match exists.

        Resolution is deliberately conservative: free calls prefer a unique same-file definition, then a run-wide unique one; `self` calls look only inside the caller's own class; bare method calls link only when the name is unique across the whole run. Any ambiguity yields `None` - a missing edge beats a wrong one.

        Parameters:
        - `file`: The calling symbol's file.
        - `s`: The calling symbol.
        - `name`: The simple (unqualified) name being called.
        - `ckind`: The call kind: `"free"`, `"self"` or `"method"`.

        Returns:
        - The target's `(file, qualname)` key, or `None` if unresolved or ambiguous.
        """

        # Free calls: a definition in the caller's own file shadows any others, so try same-file first.
        if ckind == "free":
            same = [k for k in free_funcs.get(name, []) if k[0] == file]
            if same:
                return same[0] if len(same) == 1 else None   # ambiguous same-file -> unresolved
            allk = free_funcs.get(name, [])

            # Otherwise accept a run-wide match only when the name is defined exactly once.
            return allk[0] if len(allk) == 1 else None        # else a run-wide unique match

        # `self` calls can only target the caller's own class, so build that qualname directly.
        if ckind == "self":
            if s.parent_qualname:
                cand = (file, f"{s.parent_qualname}.{name}")
                if cand in symbols:
                    return cand

            # No fallback for `self` calls - an unknown method stays unlinked rather than guessed.
            return None

        # Bare method calls carry no receiver type, so the name alone has to decide.
        if ckind == "method":
            cand = methods_by_name.get(name, [])
            return cand[0] if len(cand) == 1 else None        # unique method-name across the run -> safe to link

        # Unrecognised call kinds are left unresolved rather than guessed at.
        return None

    edges: Dict[SymKey, List[SymKey]] = {}
    call_map: Dict[SymKey, Dict[Tuple[str, str], SymKey]] = {}

    # Resolve every recorded call site into deduplicated edge lists.
    for key, s in symbols.items():
        file = key[0]
        resolved: List[SymKey] = []
        seen: Set[SymKey] = set()

        # Unresolved names and self-recursion contribute nothing.
        for call in s.calls:
            name, ckind = call[0], call[1]
            target = resolve(file, s, name, ckind)
            if target is None or target == key:
                continue
            call_map.setdefault(key, {})[(name, ckind)] = target

            # `seen` deduplicates the edge list while preserving first-call order.
            if target not in seen:
                seen.add(target)
                resolved.append(target)

        edges[key] = resolved

    # Build the 'must precede' edges that drive the leaf-first sort.
    succ: Dict[SymKey, Set[SymKey]] = {key: set() for key in symbols}

    for key, s in symbols.items():
        if s.parent_qualname:
            pkey = (key[0], s.parent_qualname)
            if pkey in symbols:
                succ[key].add(pkey)                # a nested child precedes its parent

    for caller, callees in edges.items():
        for callee in callees:
            succ[callee].add(caller)               # a callee precedes its caller

    # Tie-breaking on start line then name keeps the order stable across runs.
    order = _leaf_first_order(
        list(symbols.keys()), succ, tiebreak=lambda k: (symbols[k].start, k[1]))

    return ProjectGraph(symbols=symbols, edges=edges, order=order, call_map=call_map)


class ContractStore:

    """
    One-line behavioural contracts for every symbol in the run.

    A contract is the first line of a routine's docstring, keyed by `(file, qualname)`. The store seeds itself from docstrings that already exist, is refreshed as new ones are written during the run, and renders a capped callee-notes block for generation prompts.
    """

    # Seed contracts from docstrings already in the source, so existing knowledge is reused untouched.
    def __init__(self, graph: ProjectGraph) -> None:
        """
        Create the contract store, seeding it from docstrings already present in the project's source.

        Parameters:
        - `graph`: The project graph whose symbols and call edges back the store.
        """

        self._graph = graph
        self._contracts: Dict[SymKey, str] = {}

        # Seed with the first line of each pre-existing docstring so callers gain contracts before anything is regenerated.
        for key, s in graph.symbols.items():
            line = _first_line(s.existing_doc)
            if line:
                self._contracts[key] = line

    # Refresh the contract as soon as a new docstring is written, so later callers in the run see it.
    def update(self, file: str, qualname: str, docstring: str) -> None:
        """
        Record the first line of a freshly written docstring as the contract for a routine.

        Parameters:
        - `file`: Path of the file containing the routine.
        - `qualname`: Qualified name of the routine within that file.
        - `docstring`: The new docstring; only its first line is kept.
        """

        # Only the first line is kept - contracts are deliberately one-liners, never whole docstrings.
        line = _first_line(docstring)
        if line:
            self._contracts[(file, qualname)] = line

    def contract(self, key: SymKey) -> Optional[str]:
        """
        Return the one-line contract recorded for a symbol, or `None` if there is none.

        Parameters:
        - `key`: The `(file, qualname)` symbol key to look up.

        Returns:
        - The contract line, or `None` when the symbol has no recorded contract.
        """

        return self._contracts.get(key)

    # Callees still lacking a contract - the targets the run must summarise on demand.
    def missing_callee_contracts(self, file: str, qualname: str) -> List[SymKey]:
        """
        List the callees of a routine that do not yet have a recorded contract.

        Only calls resolved into the project graph are considered, so external or unresolved calls never count as missing.

        Parameters:
        - `file`: Path of the calling routine's file.
        - `qualname`: Qualified name of the calling routine.

        Returns:
        - Symbol keys of callees lacking a contract; empty when every callee is covered.
        """

        return [k for k in self._graph.edges.get((file, qualname), []) if k not in self._contracts]

    # Render a prompt-ready bullet list of callee contracts, capped so a large fan-out cannot swamp the context.
    def callee_notes(self, file: str, qualname: str, cap: int = CALLEE_NOTES_CAP) -> str:
        """
        Build a prompt-ready summary of the one-line contracts for the routines a given routine calls.

        Parameters:
        - `file`: Path of the calling routine's file.
        - `qualname`: Qualified name of the calling routine.
        - `cap`: Maximum number of callee entries to include.

        Returns:
        - A headed bullet list of callee contracts, or an empty string when none are known.
        """

        notes: List[Tuple[str, str]] = []

        # Gather up to `cap` contracts, keyed by the bare method name to keep the prompt compact.
        for callee in self._graph.edges.get((file, qualname), []):
            line = self._contracts.get(callee)
            if not line:
                continue
            notes.append((callee[1].rsplit(".", 1)[-1], line))
            if len(notes) >= cap:
                break

        # Return an empty string rather than a heading with no entries.
        if not notes:
            return ""
        body = "\n".join(f"- {name}: {contract}" for name, contract in notes)

        return "Functions/methods this routine calls:\n" + body
