#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The online mode's manifest machinery: every routine's comments deferred to a stronger model.

SCALE runs in one of two modes. **Offline** (the default) the local model writes everything and no manifest exists.
**Online** (`--online --emit-manifest`) every routine's comment/docstring generation is deferred to a stronger model
(e.g. Claude Code): the routines are recorded as manifest *requests*, the stronger model fills in the answer text, and
SCALE patches the answers back in through the same insertion-only path as everything else - so the byte-for-byte code
guarantee is unchanged regardless of which model produced the words.

Two phases, coordinated by a single JSON manifest **per run** (covering every target file's routines):

- **emit**: a model-free pass (the GGUF is never loaded) parses each target and records one *request* per routine -
  its identity, the code, and what the stronger model needs - leaving the target byte-for-byte untouched. One
  `Escalation` object per target collects requests; `run_manifest` merges them.
- **apply**: a separate, model-free pass reads the manifest (now carrying the stronger model's `answer`s), re-parses
  each target, matches each request back to its routine by `(qualname, sig_hash)`, and patches the answers in.

The manifest is kept **lean** - in the worst case (a fully uncommented codebase) every routine's code crosses the wire
once, and nothing may make it cross twice: a routine deferred by both passes carries ONE `snippet` (its verbatim source
span), with the block chunks referencing line ranges *into* that snippet rather than duplicating the text; and any
request whose snippet is byte-identical to an earlier request's (e.g. a header prototype documented from an impl body
that is itself deferred) carries a `snippet_ref` to that request instead. `doc_style` is one copy per manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import json


MANIFEST_VERSION = 2


# ---------------------------- request identity ----------------------------


def request_id(qualname: str, sig_hash: str) -> str:
    """
    Build the stable identifier for a manifest request from its qualified name and signature hash.

    Parameters:
    - `qualname`: The routine's fully qualified name.
    - `sig_hash`: The routine's signature hash (`scale_python.node_sig`, or `routine_text_hash` for span-hashed
      languages).

    Returns:
    - The request identifier string, e.g. "fn:Foo.bar:ab12cd34ef56".
    """

    return f"fn:{qualname}:{sig_hash}"


def routine_text_hash(span_text: str) -> str:
    """
    Hash a routine's verbatim text span - a shift-proof identity for languages without a structural AST signature.

    A deferred routine is left byte-for-byte untouched between emit and apply, so its span text is identical even
    though its absolute line numbers move as other routines are annotated; hashing the text (not the position) lets
    the apply phase re-bind the request. Note that for C the doc comment sits ABOVE the header, outside the span, so
    the hash also survives the def-pass apply that precedes the block-pass apply.

    Parameters:
    - `span_text`: The routine's verbatim header-to-end text.

    Returns:
    - A short hex digest.
    """

    return hashlib.sha256(span_text.encode("utf-8", errors="surrogateescape")).hexdigest()[:16]


# ---------------------------- emit-phase collector ----------------------------


@dataclass
class Escalation:
    """
    The emit-phase request collector for the online mode (one instance per target file).

    An instance records what the stronger model needs to answer (`record_def` / `record_block`). Both passes record
    into the SAME per-routine request - keyed by `(qualname, sig_hash)` - which carries the routine's code once
    (`snippet`, its verbatim source span) plus a `def` slot and/or a `blocks` recipe. After every target has been
    collected, `run_manifest` merges the collectors into the run's manifest. This object never calls a model.

    Attributes:
    - `doc_style`: The house-style docstring template (e.g. `comment.<lang>.txt` + `guidelines.md`), carried into the
      manifest so the stronger model is told the required style when it writes deferred docstrings.
    - `requests`: The collected per-routine requests, in the order their routines were first recorded.
    """

    doc_style: str = ""
    requests: List[dict] = field(default_factory=list)
    _index: Dict[Tuple[str, str], dict] = field(default_factory=dict, init=False, repr=False)

    def _routine(self, qualname: str, kind: str, sig_hash: str, snippet: str) -> dict:
        """
        Find or create the per-routine request - the slimming pivot: one request, one snippet, both passes record into it.

        Parameters:
        - `qualname`/`kind`/`sig_hash`: The routine's identity.
        - `snippet`: The routine's verbatim source span (kept from the first recording that supplies one;
          `record_block` then overrides it, because its chunk line ranges index into its own snapshot).

        Returns:
        - The (possibly fresh) request dict, already registered in `requests`.
        """

        key = (qualname, sig_hash)
        req = self._index.get(key)
        if req is None:
            req = {
                "id": request_id(qualname, sig_hash),
                "qualname": qualname,
                "kind": kind,
                "sig_hash": sig_hash,
                "snippet": snippet,
            }
            self._index[key] = req
            self.requests.append(req)
        elif not req.get("snippet") and snippet:
            req["snippet"] = snippet
        return req

    def record_def(self, qualname: str, kind: str, sig_hash: str, snippet: str) -> None:
        """
        Record a deferred definition (docstring/header-comment) request on the routine's manifest entry.

        Parameters:
        - `qualname`: The routine's fully qualified name.
        - `kind`: The routine kind ("def", "async def", "class", "function", or "declaration").
        - `sig_hash`: The routine's signature hash.
        - `snippet`: The routine's verbatim source span (or, for a doc-site prototype, the implementation body the
          prose is to be written from).
        """

        self._routine(qualname, kind, sig_hash, snippet)["def"] = {"answer": None}

    def record_block(
        self,
        qualname: str,
        kind: str,
        sig_hash: str,
        doc_summary: str,
        length_note: str,
        chunks: List[dict],
        snippet: str = "",
    ) -> None:
        """
        Record a deferred within-function block request on the routine's manifest entry.

        Segmentation has already run locally (it is structural and deterministic); only the per-chunk comment *text*
        is deferred. Each chunk is identified by its boundary index - the position of its start line within the
        routine's sorted legal boundaries, stable across the line shifts between emit and apply because the deferred
        routine is left untouched - and carries `lines`, the chunk's 1-based line range INTO the routine's `snippet`
        (so the chunk text is never duplicated alongside the snippet). The snippet supplied here replaces any one a
        def-pass recording stored earlier: the chunk ranges were computed against the block pass's view of the source,
        which could differ from the def pass's view if the text changed in between.

        Parameters:
        - `qualname`: The routine's fully qualified name.
        - `kind`: The routine kind.
        - `sig_hash`: The routine's signature hash.
        - `doc_summary`: The routine's one-line docstring summary, as context for the comment.
        - `length_note`: The short/long length note SCALE would have used, so the model keeps the same bias.
        - `chunks`: One dict per chunk, each `{"bidx": int, "lines": [start, end]}` (snippet-relative, 1-based,
          inclusive); an `"answer": None` slot is added here.
        - `snippet`: The routine's verbatim source span (header through last body line).
        """

        req = self._routine(qualname, kind, sig_hash, snippet)
        # The chunk `lines` index into THIS snapshot of the routine, so a block recording's snippet must win over an
        # earlier def-pass one. The def answer is indifferent to which snapshot it reads.
        if snippet:
            req["snippet"] = snippet
        req["blocks"] = {
            "doc_summary": doc_summary,
            "length_note": length_note,
            "chunks": [{"bidx": c["bidx"], "lines": list(c.get("lines") or []), "answer": None} for c in chunks],
        }

    def to_manifest(self, source: str, language: str, line_ending: str) -> dict:
        """
        Serialise this single target's requests as a complete run manifest (the one-file convenience form).

        Parameters:
        - `source`: The source file path (recorded for provenance / the apply phase).
        - `language`: The resolved language identifier.
        - `line_ending`: The detected line ending of the source.

        Returns:
        - The manifest dictionary ready to be written as JSON.
        """

        return run_manifest([(source, language, line_ending, self)], self.doc_style)


def run_manifest(parts: List[Tuple[str, str, str, "Escalation"]], doc_style: str) -> dict:
    """
    Merge per-target escalation collectors into the run's single manifest.

    Each request is stamped with its `file` so the (run-level) apply phase can route it to the right target. The
    cross-request slimming happens here: a request whose `snippet` is byte-identical to an earlier request's (the
    classic case: a header prototype whose prose source is an impl body that is itself deferred for blocks) drops the
    duplicate text and carries a `snippet_ref` naming the request that holds it - the code crosses the wire once.

    Parameters:
    - `parts`: One `(source_path, language, line_ending, escalation)` tuple per target that collected requests.
    - `doc_style`: The house-style template (one copy per manifest).

    Returns:
    - The version-2 manifest dictionary.
    """

    le_name = {"\n": "lf", "\r\n": "crlf", "\r": "cr"}
    files: List[dict] = []
    requests: List[dict] = []
    seen_snippets: Dict[str, str] = {}
    for source, language, line_ending, esc in parts:
        files.append({"path": source, "language": language, "line_ending": le_name.get(line_ending, "lf")})
        for r in esc.requests:
            req = dict(r)
            req["file"] = source
            snippet = req.get("snippet")
            if snippet:
                prior = seen_snippets.get(snippet)
                if prior is not None:
                    req["snippet"] = None
                    req["snippet_ref"] = prior
                else:
                    seen_snippets[snippet] = req["id"]
            requests.append(req)

    return {
        "version": MANIFEST_VERSION,
        "tool": "scale",
        "files": files,
        "doc_style": doc_style,
        "requests": requests,
    }


# ---------------------------- completeness ----------------------------


def unfilled_answers(manifest: dict) -> List[str]:
    """
    Return an identifier for every answer slot in the manifest that has not been filled in.

    Completeness is enforced by this counter, never by trusting a model's diligence: the driver loops until the count
    is zero. An answer of `null` (or whitespace) is unfilled; the explicit string `"NONE"` is a deliberate decline and
    counts as filled.

    Parameters:
    - `manifest`: The parsed manifest dictionary.

    Returns:
    - One entry per unfilled slot, e.g. "fn:heavy:ab12:def" or "fn:heavy:ab12:block[2]".
    """

    out: List[str] = []
    for r in manifest.get("requests", []):
        d = r.get("def")
        if d is not None and not str(d.get("answer") or "").strip():
            out.append(f"{r.get('id', r.get('qualname', '?'))}:def")
        b = r.get("blocks")
        if b is not None:
            for i, chunk in enumerate(b.get("chunks", [])):
                if not str(chunk.get("answer") or "").strip():
                    out.append(f"{r.get('id', r.get('qualname', '?'))}:block[{i}]")
    return out


# ---------------------------- fragments (parallel fill protocol) ----------------------------


FRAGMENT_KEY = "fragment"


def _request_unfilled(req: dict) -> bool:
    """
    Report whether a request still has any unfilled answer slot.

    Parameters:
    - `req`: One manifest request dictionary.

    Returns:
    - True when the def answer or any block-chunk answer is null/whitespace ("NONE" counts as filled).
    """

    d = req.get("def")
    if d is not None and not str(d.get("answer") or "").strip():
        return True
    b = req.get("blocks")
    if b is not None:
        for chunk in b.get("chunks", []):
            if not str(chunk.get("answer") or "").strip():
                return True
    return False


def next_fragment_name(manifest: dict, master_name: str) -> str:
    """
    Choose the next fragment file name for a master manifest (e.g. "scale-manifest.frag-003.json").

    A monotonic `fragments_issued` counter on the master (bumped here, persisted by the caller) guarantees a name is
    never reused within one master, even after earlier fragments have been merged and their files deleted.

    Parameters:
    - `manifest`: The parsed master manifest (mutated: the issue counter is incremented).
    - `master_name`: The master manifest's file name (stem + suffix only; no directory).

    Returns:
    - The fragment file name (no directory).
    """

    stem, dot, suffix = master_name.rpartition(".")
    if not dot:
        stem, suffix = master_name, "json"
    issued = int(manifest.get("fragments_issued", 0)) + 1
    manifest["fragments_issued"] = issued
    return f"{stem}.frag-{issued:03d}.{suffix}"


def build_fragment(manifest: dict, size: int, fragment_name: str) -> Optional[dict]:
    """
    Check out the next batch of unfilled requests as a self-contained fragment manifest.

    The fragment is a valid version-2 manifest in its own right (same request shape, one `doc_style` copy), so a
    filling agent can read it directly and self-check it with the ordinary completeness counter. Each selected
    request is marked in the MASTER (mutated in place) with the fragment's name, so concurrent calls hand out
    disjoint work; the caller persists the master afterwards. A request whose `snippet` is a `snippet_ref` to a
    request outside this fragment gets the referenced text inlined, keeping every fragment self-contained.

    Parameters:
    - `manifest`: The parsed master manifest (mutated: selected requests gain a `fragment` marker).
    - `size`: The maximum number of requests to include.
    - `fragment_name`: The name to record on each selected request (the fragment's file name).

    Returns:
    - The fragment manifest dictionary, or None when every unfilled request is already checked out (or none remain).
    """

    pool = [r for r in manifest.get("requests", []) if _request_unfilled(r) and not r.get(FRAGMENT_KEY)]
    if not pool or size < 1:
        return None
    picked = pool[:size]
    for r in picked:
        r[FRAGMENT_KEY] = fragment_name

    # Resolve snippet_refs that point outside the fragment: the code must cross the wire once per READER, so a
    # fragment may never force its agent back to the master file.
    picked_ids = {r.get("id") for r in picked}
    snippets_by_id: Dict[str, str] = {}
    for r in manifest.get("requests", []):
        if r.get("snippet") and r.get("id") not in snippets_by_id:
            snippets_by_id[r["id"]] = r["snippet"]
    out_requests: List[dict] = []
    for r in picked:
        req = json.loads(json.dumps(r))  # deep copy; the fragment must not alias the master
        ref = req.get("snippet_ref")
        if not req.get("snippet") and ref and ref not in picked_ids:
            req["snippet"] = snippets_by_id.get(ref)
        req.pop(FRAGMENT_KEY, None)
        out_requests.append(req)

    used_files = {r.get("file") for r in out_requests}
    return {
        "version": MANIFEST_VERSION,
        "tool": manifest.get("tool", "scale"),
        "fragment_of": fragment_name,
        "files": [f for f in manifest.get("files", []) if f.get("path") in used_files],
        "doc_style": manifest.get("doc_style", ""),
        "requests": out_requests,
    }


def merge_fragment(manifest: dict, fragment: dict) -> int:
    """
    Fold a filled fragment's answers back into the master manifest and release its checked-out requests.

    Requests are matched by `(id, file)` (two files may legitimately carry the same id when their routines are
    byte-identical) and chunks by `bidx`. First write wins: a master slot that is already filled is never
    overwritten, so a stale or duplicated fragment cannot clobber good answers. Every master request the fragment
    covers has its checkout marker cleared - a slot the fragment left unfilled simply returns to the pile.

    Parameters:
    - `manifest`: The parsed master manifest (mutated in place).
    - `fragment`: The parsed fragment manifest carrying answers.

    Returns:
    - The number of answer slots newly filled in the master.
    """

    index: Dict[Tuple[str, str], dict] = {}
    for r in manifest.get("requests", []):
        index.setdefault((str(r.get("id")), str(r.get("file"))), r)

    filled = 0
    for fr in fragment.get("requests", []):
        mr = index.get((str(fr.get("id")), str(fr.get("file"))))
        if mr is None:
            continue
        md, fd = mr.get("def"), fr.get("def")
        if md is not None and fd is not None:
            if not str(md.get("answer") or "").strip() and str(fd.get("answer") or "").strip():
                md["answer"] = fd["answer"]
                filled += 1
        mb, fb = mr.get("blocks"), fr.get("blocks")
        if mb is not None and fb is not None:
            by_bidx = {c.get("bidx"): c for c in fb.get("chunks", [])}
            for chunk in mb.get("chunks", []):
                fc = by_bidx.get(chunk.get("bidx"))
                if fc is not None and not str(chunk.get("answer") or "").strip() \
                        and str(fc.get("answer") or "").strip():
                    chunk["answer"] = fc["answer"]
                    filled += 1
        mr.pop(FRAGMENT_KEY, None)
    return filled


def release_unfilled(manifest: dict) -> int:
    """
    Return every still-unfilled request to the pile by clearing its checkout marker.

    Called when a fill round ends incomplete (an agent died or skipped slots): the next `build_fragment` can then
    hand the remaining work out again.

    Parameters:
    - `manifest`: The parsed master manifest (mutated in place).

    Returns:
    - The number of requests released.
    """

    released = 0
    for r in manifest.get("requests", []):
        if r.get(FRAGMENT_KEY) and _request_unfilled(r):
            r.pop(FRAGMENT_KEY, None)
            released += 1
    return released


# ---------------------------- manifest I/O ----------------------------


def write_manifest(path: Path, manifest: dict) -> None:
    """
    Write a manifest dictionary to disk as indented JSON (UTF-8).

    Parameters:
    - `path`: The destination file path.
    - `manifest`: The manifest dictionary to serialise.
    """

    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _upgrade_v1(manifest: dict) -> dict:
    """
    Upgrade a version-1 (single-file, per-pass-request) manifest to the version-2 shape in memory.

    The v1 schema carried one request per pass ("def" with a top-level `answer`; "block" with per-chunk `text` and
    `answer`). Each becomes a v2 per-routine request stamped with the manifest's single `source` file; v1 block chunks
    keep their answers (their `text` is simply dropped - apply places by boundary index, not text).

    Parameters:
    - `manifest`: The parsed v1 manifest.

    Returns:
    - An equivalent v2 manifest dictionary.
    """

    source = manifest.get("source", "")
    requests: List[dict] = []
    for r in manifest.get("requests", []):
        req = {
            "id": request_id(r.get("qualname", "?"), r.get("sig_hash", "")),
            "qualname": r.get("qualname"),
            "kind": r.get("kind"),
            "sig_hash": r.get("sig_hash"),
            "snippet": r.get("snippet"),
            "file": source,
        }
        if r.get("pass") == "def":
            req["def"] = {"answer": r.get("answer")}
        elif r.get("pass") == "block":
            req["blocks"] = {
                "doc_summary": r.get("doc_summary", ""),
                "length_note": r.get("length_note", ""),
                "chunks": [{"bidx": c.get("bidx"), "lines": None, "answer": c.get("answer")}
                           for c in r.get("chunks", [])],
            }
        requests.append(req)

    return {
        "version": MANIFEST_VERSION,
        "tool": manifest.get("tool", "scale"),
        "files": [{"path": source, "language": manifest.get("language"),
                   "line_ending": manifest.get("line_ending", "lf")}],
        "doc_style": manifest.get("doc_style", ""),
        "requests": requests,
    }


def read_manifest(path: Path) -> dict:
    """
    Read and lightly validate a manifest file, returning it in the current (version-2) shape.

    A version-1 manifest is upgraded in memory so the apply phase needs to understand only one schema.

    Parameters:
    - `path`: The manifest file path.

    Returns:
    - The parsed manifest dictionary (version 2).

    Raises:
    - `ValueError`: If the file is not a SCALE manifest of a supported version.
    """

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") not in (1, MANIFEST_VERSION):
        raise ValueError(f"{path}: not a SCALE manifest of version 1 or {MANIFEST_VERSION}")
    if manifest.get("version") == 1:
        manifest = _upgrade_v1(manifest)
    manifest.setdefault("requests", [])
    manifest.setdefault("files", [])
    return manifest
