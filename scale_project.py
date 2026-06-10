#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The project-context layer that sits above SCALE's per-file pipeline.

SCALE is otherwise single-file: it primes on one file's summary and annotates it in isolation, which makes file
descriptions read generically (an `error.c` that never mentions it belongs to a BASIC interpreter). This module gives
the per-file pipeline a *broader view* - but only as small, distilled facts, because the local model's context window
is tight. The byte-for-byte code guarantee is unaffected: nothing here patches source, it only produces context strings
that are fed into the existing priming.

It locates a project overview document (`CLAUDE.md` / `README.*`) near the files being annotated and distils it once
into a short, cached "project blurb" that is injected into every file's priming context.
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


# Source extensions used when a target/reference path is a directory (so a directory expands to just its source files,
# not every file). Explicitly named files and glob matches are taken as-is regardless of extension.
SOURCE_EXTS = (".py", ".c", ".h", ".js", ".mjs", ".cjs")

# Cap on how many read-only reference files are summarised into the shared context, so the injected one-liners stay
# small regardless of how large the --reference set is (the overflow is logged, not silently dropped).
MAX_REFERENCE_FILES = 16


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
    Read a file as text using the same surrogateescape decoding SCALE uses for source, returning "" if unreadable.

    Parameters:
    - `path`: The file to read.

    Returns:
    - The decoded contents, or "" on any read error.
    """

    try:
        return path.read_bytes().decode("utf-8", errors="surrogateescape")
    except OSError:
        return ""


def _read_optional(path: Path) -> Optional[str]:
    """Return a config file's text if it exists, otherwise None (a local copy so this module needs no scale.py import)."""

    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def find_project_doc(start: Path, max_levels: int = 8) -> Optional[Path]:
    """
    Locate the nearest project-overview document by walking up from `start`.

    From the directory of `start` (or `start` itself if it is a directory) and upwards, each directory is searched for
    `CLAUDE.md` (preferred) or a `README` with any common extension (`.md`/`.markdown`/`.rst`/`.txt`/none), matched
    case-insensitively. The first match wins (so a doc nearest the file beats one further up). The walk stops at a
    repository root (a directory containing `.git`, inclusive), the filesystem root, or after `max_levels` directories.

    Parameters:
    - `start`: A target file or directory to search from.
    - `max_levels`: The maximum number of directories to ascend.

    Returns:
    - The path to the chosen document, or None if none is found.
    """

    d = (start if start.is_dir() else start.parent).resolve()
    for _ in range(max_levels):
        try:
            entries = [p for p in d.iterdir() if p.is_file()]
        except OSError:
            entries = []

        preferred = [p for p in entries if p.name.lower() == _PREFERRED_NAME]
        if preferred:
            return preferred[0]

        readmes = [p for p in entries if p.stem.lower() == _README_STEM and p.suffix.lower() in _DOC_SUFFIXES]
        if readmes:
            # Prefer a Markdown README, then settle ties by name for determinism.
            readmes.sort(key=lambda p: (p.suffix.lower() != ".md", p.name))
            return readmes[0]

        if (d / ".git").exists() or d.parent == d:   # repo root (inclusive) or filesystem root
            break
        d = d.parent
    return None


def _blurb_cache_path(doc_text: str) -> Path:
    """Return the cache file for a blurb, keyed on a content hash of the source document (so edits invalidate it)."""

    digest = hashlib.sha256(doc_text.encode("utf-8", errors="surrogateescape")).hexdigest()
    return _CACHE_DIR / f"project-{digest}.txt"


def _read_cache(path: Path) -> Optional[str]:
    """Read a cached blurb, or None if absent."""

    try:
        return path.read_bytes().decode("utf-8", errors="surrogateescape")
    except FileNotFoundError:
        return None


def _write_cache(path: Path, text: str) -> None:
    """Atomically write a blurb to the cache (temp file + replace)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(text.encode("utf-8", errors="surrogateescape"))
    tmp.replace(path)


def _crop_to_budget(text: str, llm, budget_tokens: int) -> str:
    """
    Crop a document to a token budget, keeping the top (where a README/overview states what the project is).

    Parameters:
    - `text`: The document text.
    - `llm`: A model exposing `estimate_tokens`.
    - `budget_tokens`: The maximum estimated tokens to keep.

    Returns:
    - The original text if it fits, else its leading lines up to the budget plus a truncation marker.
    """

    if llm.estimate_tokens(text) <= budget_tokens:
        return text
    kept: list[str] = []
    for line in text.splitlines():
        kept.append(line)
        if llm.estimate_tokens("\n".join(kept)) > budget_tokens:
            kept.pop()
            break
    return "\n".join(kept) + "\n\n[... overview document truncated ...]"


def project_blurb(llm, cfg, scale_path: Path, doc_path: Path, no_cache: bool = False) -> str:
    """
    Distil a project-overview document into a short, cached "project blurb" for priming context.

    The blurb is a couple of sentences describing the project, its domain, and key terminology - background to be shown
    above individual files so the per-file passes stop reading as if each file stands alone. It is cached by the
    document's content hash (so editing the doc regenerates it). Large documents are cropped to a token budget first.

    Parameters:
    - `llm`: A model exposing `generate`, `estimate_tokens`, `n_ctx`, and `ctx_margin`.
    - `cfg`: The base generation configuration.
    - `scale_path`: The SCALE configuration directory (for the optional `project.txt` instruction override).
    - `doc_path`: The overview document to distil.
    - `no_cache`: When True, regenerate rather than loading a cached blurb.

    Returns:
    - The blurb text, or "" if the document is empty/unreadable.
    """

    doc_text = _read_text_bytes(doc_path)
    if not doc_text.strip():
        return ""

    cache_path = _blurb_cache_path(doc_text)
    if not no_cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            echo(f"Loaded project blurb from cache ({doc_path.name})...")
            return cached

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
    Expand a list of path patterns into a deduplicated, ordered list of existing files.

    Each pattern is resolved as: a glob (when it contains `*`/`?`/`[`, with `**` recursion); a directory (expanded
    recursively to files whose extension is in `exts`); or a single named file (taken as-is, any extension). Order is
    deterministic (sorted by path) and duplicates (by resolved path) are removed, so overlapping patterns are safe.

    Parameters:
    - `patterns`: The raw path/glob/dir strings (e.g. from the CLI).
    - `exts`: The source extensions a directory expands to.

    Returns:
    - The matching existing files, sorted, without duplicates.
    """

    out: List[Path] = []
    seen = set()
    for pattern in patterns:
        p = Path(pattern)
        if any(ch in pattern for ch in "*?["):
            candidates = [Path(m) for m in glob.glob(pattern, recursive=True)]
        elif p.is_dir():
            candidates = [c for c in sorted(p.rglob("*")) if c.suffix.lower() in exts]
        else:
            candidates = [p]
        for c in candidates:
            if not c.is_file():
                continue
            key = c.resolve()
            if key not in seen:
                seen.add(key)
                out.append(c)
    return sorted(out, key=lambda x: str(x))


def compose_project_context(blurb: str, related: List[Tuple[str, str]]) -> str:
    """
    Format the project blurb and a list of related-file one-liners into a single context string for priming.

    Parameters:
    - `blurb`: The project overview blurb (may be "").
    - `related`: `(filename, one_line_summary)` pairs for read-only reference files the run should be aware of.

    Returns:
    - A combined context string (possibly ""), suitable for injecting as a priming turn.
    """

    parts: List[str] = []
    if blurb.strip():
        parts.append(blurb.strip())
    related = [(name, summary.strip()) for name, summary in related if summary.strip()]
    if related:
        lines = "\n".join(f"- {name}: {summary}" for name, summary in related)
        parts.append("Related files in this project (read-only, for reference):\n" + lines)
    return "\n\n".join(parts)


def resolve_project_doc(project_doc_arg: str, start: Path) -> Optional[Path]:
    """
    Resolve the project-overview document from the `--project-doc` argument and the target location.

    Parameters:
    - `project_doc_arg`: The CLI value: "" to auto-detect, "none" to disable, or an explicit path.
    - `start`: A target file/directory to auto-detect from.

    Returns:
    - The resolved document path, or None when disabled or not found.
    """

    if project_doc_arg.strip().lower() == "none":
        return None
    if project_doc_arg:
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
# turn. It is cost-neutral: contracts come only from existing docstrings (a seed) and the docstrings the def pass itself
# generates (each one's first line updates the contract for later callers) - no extra model calls are made. The
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
    Order `items` by a pre-pass `doc_order` (qualnames), falling back to `fallback_key` for unlisted items and ties.

    A worker's def loop uses this to process routines callee/child-first per the call graph instead of its internal
    sort. Items whose qualname is in `doc_order` come first in that order; any item not listed (or sharing a qualname
    with another - a collision) is ordered by `fallback_key`, which should preserve the worker's own correctness
    constraint (e.g. deepest-first so a child is still documented before its parent).

    Parameters:
    - `items`: The worker's definition records.
    - `qualname_of`: Extracts an item's qualname.
    - `doc_order`: The qualnames in desired documentation order.
    - `fallback_key`: A sort key for unlisted items and intra-position ties.

    Returns:
    - The items reordered.
    """

    pos = {q: i for i, q in enumerate(doc_order)}
    sentinel = len(pos)
    return sorted(items, key=lambda it: (pos.get(qualname_of(it), sentinel), fallback_key(it)))


@dataclass
class Symbol:
    """
    One documentable routine in the run, as seen by the model-free call-graph pre-pass.

    A worker's `iter_symbols` emits these (without `file`, which `build_project_graph` stamps from the run's file key)
    by walking each routine's own body - not descending into nested definitions - so `calls` reflects only the
    routine's own call sites.

    Attributes:
    - `qualname`: The routine's qualified name (e.g. "foo", "Class.method", "outer.inner").
    - `kind`: The definition kind ("def"/"class"/"function"/"method"/...), as the worker reports it.
    - `signature`: The routine's header text (used only for context/debugging; not patched).
    - `start`: The 1-based start line of the routine (used as a deterministic ordering tiebreak).
    - `depth`: The nesting depth (0 at file scope).
    - `parent_qualname`: The qualname of the immediately enclosing definition, or None at file scope.
    - `existing_doc`: The first line of any documentation the routine already has (the seed contract).
    - `calls`: The routine's own call sites as `(name, kind)` pairs, kind ∈ {"free", "self", "method"}.
    - `file`: The run's file key (a resolved-path string); stamped by `build_project_graph`.
    """

    qualname: str
    kind: str
    signature: str
    start: int
    depth: int
    parent_qualname: Optional[str]
    existing_doc: str
    calls: List[Tuple[str, str]] = field(default_factory=list)
    file: str = ""


# A symbol key uniquely identifies a routine across the run: (file, qualname).
SymKey = Tuple[str, str]


def _first_line(text: str) -> str:
    """Return the first non-blank line of `text`, stripped (or "" if there is none) - a routine's one-line contract."""

    for line in (text or "").splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _tarjan_sccs(nodes: List, succ: Dict) -> List[List]:
    """
    Find the strongly-connected components of a directed graph, in reverse-topological order.

    An iterative Tarjan (so deep/recursive graphs cannot blow the Python stack). `succ[n]` is the set of nodes that `n`
    points to. The components are returned sink-first (a component is emitted after every component reachable from it),
    which callers reverse to get a source-first topological order. Cycles (recursion / mutual recursion) collapse into a
    single component, so the ordering never loops.

    Parameters:
    - `nodes`: All graph nodes (the iteration order seeds a deterministic DFS).
    - `succ`: Adjacency mapping each node to an iterable of its successors.

    Returns:
    - A list of components (each a list of nodes), in reverse-topological order.
    """

    index: Dict = {}
    low: Dict = {}
    on_stack: Set = set()
    stack: List = []
    sccs: List[List] = []
    counter = 0

    for root in nodes:
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(succ.get(root, ())))]
        while work:
            node, it = work[-1]
            descended = False
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(succ.get(w, ()))))
                    descended = True
                    break
                if w in on_stack:
                    low[node] = min(low[node], index[w])
            if descended:
                continue
            if low[node] == index[node]:
                comp: List = []
                while True:
                    x = stack.pop()
                    on_stack.discard(x)
                    comp.append(x)
                    if x == node:
                        break
                sccs.append(comp)
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return sccs


def _leaf_first_order(nodes: List, succ: Dict, tiebreak: Callable) -> List:
    """
    Order `nodes` leaf-first: a node appears before any node it points to, with cycles collapsed.

    `succ[n]` lists the nodes that must come *after* `n` (n "precedes" them). A deterministic Kahn topological sort over
    the SCC condensation: cycles (recursion / mutual recursion) collapse into one component, and a component is emitted
    only once everything that precedes it has been. Among components ready at the same time - and among unconstrained
    nodes, which all are - `tiebreak` decides, so the order is stable and (for independent nodes) follows their natural
    `tiebreak` order rather than an artefact of the DFS.

    Parameters:
    - `nodes`: The nodes to order.
    - `succ`: Adjacency where `n -> m` means n precedes m.
    - `tiebreak`: A sort key for ordering nodes within a component and for breaking ties between ready components.

    Returns:
    - The nodes in leaf-first order.
    """

    sccs = _tarjan_sccs(nodes, succ)
    comp_of: Dict = {}
    for i, comp in enumerate(sccs):
        for n in comp:
            comp_of[n] = i

    cedges: Dict[int, Set[int]] = {i: set() for i in range(len(sccs))}
    indeg: Dict[int, int] = {i: 0 for i in range(len(sccs))}
    for n in nodes:
        cn = comp_of[n]
        for m in succ.get(n, ()):                      # n precedes m -> component cn precedes cm
            cm = comp_of[m]
            if cn != cm and cm not in cedges[cn]:
                cedges[cn].add(cm)
                indeg[cm] += 1

    comp_key = [min(tiebreak(n) for n in comp) for comp in sccs]
    heap = [(comp_key[i], i) for i in range(len(sccs)) if indeg[i] == 0]
    heapq.heapify(heap)

    out: List = []
    while heap:
        _, i = heapq.heappop(heap)
        out.extend(sorted(sccs[i], key=tiebreak))
        for cm in sorted(cedges[i]):
            indeg[cm] -= 1
            if indeg[cm] == 0:
                heapq.heappush(heap, (comp_key[cm], cm))
    return out


@dataclass
class ProjectGraph:
    """
    The resolved call graph for a run: a symbol table, the resolved callee edges, and a leaf-first documentation order.

    Built by `build_project_graph` from every run file's symbols (targets and read-only references alike). `edges` maps
    each routine to the routines it calls *that were confidently resolved*; `order` is every symbol key in leaf-first
    order (callees and nested children before their callers/parents). References take part in resolution and ordering
    inputs but are never documented - the caller restricts documentation to its targets.
    """

    symbols: Dict[SymKey, Symbol]
    edges: Dict[SymKey, List[SymKey]]
    order: List[SymKey]

    def doc_order(self, file: str) -> List[str]:
        """Return the qualnames of `file`'s symbols in leaf-first documentation order."""

        return [q for (f, q) in self.order if f == file]

    def file_order(self, target_files: List[str]) -> List[str]:
        """
        Order the target files coarsely so a callee's file is documented before a caller's (leaf-first), by file.

        Only cross-file call edges constrain the order (nesting is intra-file); files with no constraint keep their
        input order. Cycles between files collapse so the order never loops. References are not included - only the
        targets are documented and reordered.

        Parameters:
        - `target_files`: The file keys to order (the run's targets).

        Returns:
        - The target file keys in leaf-first order.
        """

        idx = {f: i for i, f in enumerate(target_files)}
        tset = set(target_files)
        succ: Dict[str, Set[str]] = {f: set() for f in target_files}
        for caller, callees in self.edges.items():
            cf = caller[0]
            if cf not in tset:
                continue
            for callee in callees:
                kf = callee[0]
                if kf != cf and kf in tset:
                    succ[kf].add(cf)   # the callee's file precedes the caller's file
        return _leaf_first_order(target_files, succ, tiebreak=lambda f: idx[f])


def build_project_graph(symbols_by_file: Dict[str, List[Symbol]]) -> ProjectGraph:
    """
    Build the resolved call graph from every run file's symbols (the model-free pre-pass core).

    Indexes the symbols, resolves each routine's call sites by the confident-only rules (free-by-name with same-file
    preference then run-wide uniqueness; `self`/`this` to the enclosing class's own method; `obj.m()` only when `m` is
    unique to one class run-wide), and computes a leaf-first documentation order from two edge kinds - nesting
    (child precedes parent, so the existing child-stub mechanism still has children done first) and call (callee
    precedes caller) - condensed by SCC so recursion / mutual recursion cannot deadlock or loop.

    Parameters:
    - `symbols_by_file`: Each run file's symbols, keyed by the file key (a resolved-path string). Each symbol's `file`
      is stamped here.

    Returns:
    - The populated `ProjectGraph`.
    """

    symbols: Dict[SymKey, Symbol] = {}
    for file, syms in symbols_by_file.items():
        for s in syms:
            s.file = file
            symbols[(file, s.qualname)] = s

    # Name indices for resolution. `free_funcs`: bare-name -> top-level symbol keys (functions/classes at file scope).
    # `methods_by_name`: method simple-name -> keys of methods (a symbol whose immediate parent is a class).
    free_funcs: Dict[str, List[SymKey]] = {}
    methods_by_name: Dict[str, List[SymKey]] = {}
    for key, s in symbols.items():
        file, q = key
        simple = q.rsplit(".", 1)[-1]
        if s.kind == "declaration":
            continue   # a C prototype seeds a contract but is never a resolution target (the definition is)
        if s.parent_qualname is None:
            free_funcs.setdefault(simple, []).append(key)
        else:
            parent = symbols.get((file, s.parent_qualname))
            if parent is not None and parent.kind == "class":
                methods_by_name.setdefault(simple, []).append(key)

    def resolve(file: str, s: Symbol, name: str, ckind: str) -> Optional[SymKey]:
        """Resolve one call site to a callee key by the confident-only rules, or None when it cannot be linked safely."""

        if ckind == "free":
            same = [k for k in free_funcs.get(name, []) if k[0] == file]
            if same:
                return same[0] if len(same) == 1 else None   # ambiguous same-file -> unresolved
            allk = free_funcs.get(name, [])
            return allk[0] if len(allk) == 1 else None        # else a run-wide unique match
        if ckind == "self":
            if s.parent_qualname:
                cand = (file, f"{s.parent_qualname}.{name}")
                if cand in symbols:
                    return cand
            return None
        if ckind == "method":
            cand = methods_by_name.get(name, [])
            return cand[0] if len(cand) == 1 else None        # unique method-name across the run -> safe to link
        return None

    edges: Dict[SymKey, List[SymKey]] = {}
    for key, s in symbols.items():
        file = key[0]
        resolved: List[SymKey] = []
        seen: Set[SymKey] = set()
        for (name, ckind) in s.calls:
            target = resolve(file, s, name, ckind)
            if target is not None and target != key and target not in seen:
                seen.add(target)
                resolved.append(target)
        edges[key] = resolved

    # Documentation order: a "precedes" graph (n -> m means n is documented before m), then leaf-first topo over SCCs.
    succ: Dict[SymKey, Set[SymKey]] = {key: set() for key in symbols}
    for key, s in symbols.items():
        if s.parent_qualname:
            pkey = (key[0], s.parent_qualname)
            if pkey in symbols:
                succ[key].add(pkey)                # a nested child precedes its parent
    for caller, callees in edges.items():
        for callee in callees:
            succ[callee].add(caller)               # a callee precedes its caller

    order = _leaf_first_order(
        list(symbols.keys()), succ, tiebreak=lambda k: (symbols[k].start, k[1]))
    return ProjectGraph(symbols=symbols, edges=edges, order=order)


class ContractStore:
    """
    The run's evolving one-line contracts: each routine's first-line summary, used to feed callers their callees' gist.

    Seeded from every symbol's existing documentation (so a callee already documented - or a reference file - has a
    contract from the start), then refined as the definition pass writes each docstring (`update`). A caller's
    generation turn pulls its resolved callees' contracts via `callee_notes`. Contracts thus come only from existing or
    freshly-generated docs, so the whole layer adds no model calls.
    """

    def __init__(self, graph: ProjectGraph) -> None:
        """Seed a contract for every symbol that already has documentation (its first line)."""

        self._graph = graph
        self._contracts: Dict[SymKey, str] = {}
        for key, s in graph.symbols.items():
            line = _first_line(s.existing_doc)
            if line:
                self._contracts[key] = line

    def update(self, file: str, qualname: str, docstring: str) -> None:
        """Set a routine's contract to the first line of a freshly-generated docstring (ignored if it is blank)."""

        line = _first_line(docstring)
        if line:
            self._contracts[(file, qualname)] = line

    def callee_notes(self, file: str, qualname: str, cap: int = CALLEE_NOTES_CAP) -> str:
        """
        Format the contracts of a routine's resolved callees as a short context block (or "" when there are none).

        Callees without a contract yet (unresolved, or not documented and undocumented to begin with) are omitted, and
        the list is capped so the injected context stays small.

        Parameters:
        - `file`/`qualname`: The calling routine.
        - `cap`: The maximum number of callee contracts to include.

        Returns:
        - A "Functions/methods this routine calls:" block, or "" if no callee has a contract.
        """

        notes: List[Tuple[str, str]] = []
        for callee in self._graph.edges.get((file, qualname), []):
            line = self._contracts.get(callee)
            if not line:
                continue
            notes.append((callee[1].rsplit(".", 1)[-1], line))
            if len(notes) >= cap:
                break
        if not notes:
            return ""
        body = "\n".join(f"- {name}: {contract}" for name, contract in notes)
        return "Functions/methods this routine calls:\n" + body
