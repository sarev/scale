#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

The plumbing for SCALE's online mode: the run-level manifest through which every routine's comments are deferred to a
stronger model. The `Escalation` dataclass collects one request per routine in a file - keyed by qualified name plus
signature hash, with null answer slots for the def and block passes - and `run_manifest` merges each file's collector
into a single manifest in which identical snippet text crosses the wire only once.

Completeness is counted, never trusted: `unfilled_answers` lists every blank answer slot, so a run is finished only
when that list is empty. `build_fragment`, `merge_fragment` and `release_unfilled` support parallel filling - carving
self-contained fragments of unfilled requests out of the master, merging their answers back fill-only so nothing is
overwritten, and recovering requests stranded by fragments that never returned.

Reading and writing round out the module: manifests are serialised as pretty-printed UTF-8 JSON, and `read_manifest`
validates the file's identity and upgrades version-1 manifests to the current multi-file shape on load.
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
    Build the canonical manifest request id for a routine.

    Parameters:
    - `qualname`: The routine's qualified name.
    - `sig_hash`: The routine's signature hash.

    Returns:
    - The `fn:<qualname>:<sig_hash>` string used to identify the request.
    """

    return f"fn:{qualname}:{sig_hash}"


def routine_text_hash(span_text: str) -> str:
    """
    Hash a routine's span text into a short stable identifier.

    Encoding with `surrogateescape` mirrors the binary-safe file loader, so spans containing bytes that are not valid UTF-8 still hash deterministically.

    Parameters:
    - `span_text`: The routine's verbatim source text.

    Returns:
    - The first 16 hex digits of the SHA-256 digest of the text.
    """

    # `surrogateescape` mirrors the binary-safe loader, so undecodable bytes still hash stably.
    return hashlib.sha256(span_text.encode("utf-8", errors="surrogateescape")).hexdigest()[:16]


# ---------------------------- emit-phase collector ----------------------------


@dataclass
class Escalation:

    """
    Collector for one source file's deferred annotation requests.

    Each routine is recorded exactly once, keyed by qualified name plus signature hash, with null `answer` slots left for a stronger model to fill. `to_manifest` wraps the collected requests into a single-file run manifest.
    """

    # Requests keep emit order; the index dedupes by (qualname, sig_hash) so a routine never crosses the wire twice.
    doc_style: str = ""
    requests: List[dict] = field(default_factory=list)
    _index: Dict[Tuple[str, str], dict] = field(default_factory=dict, init=False, repr=False)

    def _routine(self, qualname: str, kind: str, sig_hash: str, snippet: str) -> dict:
        """
        Fetch or create the request record for a routine.

        The (qualname, sig_hash) key guarantees one record per routine; a later call with a non-empty snippet backfills a record created without one, but never replaces existing text.

        Parameters:
        - `qualname`: The routine's qualified name.
        - `kind`: The routine kind (e.g. function, method or class).
        - `sig_hash`: The routine's signature hash.
        - `snippet`: The routine's verbatim source, or empty if already captured.

        Returns:
        - The shared request dictionary, newly created or existing.
        """

        key = (qualname, sig_hash)
        req = self._index.get(key)

        # Create on first sight; later calls may only backfill a missing snippet, never replace existing text.
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
        Register a docstring request for a routine.

        Reuses or creates the routine's request record and attaches a `def` slot whose null answer is left for the stronger model to fill.

        Parameters:
        - `qualname`: The routine's qualified name.
        - `kind`: The routine kind (e.g. function, method or class).
        - `sig_hash`: The routine's signature hash.
        - `snippet`: The routine's verbatim source.
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
        Register the block-comment requests for a routine.

        Attaches a `blocks` section to the routine's request record, reducing each provider chunk to its boundary index, line range and a null answer slot for the stronger model to fill. A non-empty snippet replaces any text already stored.

        Parameters:
        - `qualname`: The routine's qualified name.
        - `kind`: The routine kind (e.g. function, method or class).
        - `sig_hash`: The routine's signature hash.
        - `doc_summary`: A one-line summary of the routine for the answering model.
        - `length_note`: Scoring guidance matched to the routine's length.
        - `chunks`: The provider's chunk records carrying `bidx` and line ranges.
        - `snippet`: The routine's verbatim source; optional if already captured.
        """

        # Chunks are reduced to boundary index, line range, the anchor line text and a null answer slot; any other provider extras are dropped.
        req = self._routine(qualname, kind, sig_hash, snippet)
        if snippet:
            req["snippet"] = snippet
        req["blocks"] = {
            "doc_summary": doc_summary,
            "length_note": length_note,
            "chunks": [{"bidx": c["bidx"], "lines": list(c.get("lines") or []),
                        "anchor": c.get("anchor", ""), "answer": None} for c in chunks],
        }

    def to_manifest(self, source: str, language: str, line_ending: str) -> dict:
        """
        Wrap this collector's requests into a single-file run manifest.

        Parameters:
        - `source`: The annotated file's path as recorded in the manifest.
        - `language`: The file's language name.
        - `line_ending`: The file's detected line-ending style.

        Returns:
        - The run-manifest dictionary covering just this file.
        """

        return run_manifest([(source, language, line_ending, self)], self.doc_style)


def run_manifest(parts: List[Tuple[str, str, str, "Escalation"]], doc_style: str) -> dict:
    """
    Build the run-level manifest dict from each processed file's collected escalation requests.

    Every request is tagged with its source file, and identical snippet text crosses the wire only once: the first request keeps the text and later duplicates carry a `snippet_ref` naming that first request's id.

    Parameters:
    - `parts`: One tuple per file of (source path, language, line-ending string, its `Escalation` collector).
    - `doc_style`: The house-style text embedded verbatim for the answering model.

    Returns:
    - The manifest dict (version, tool, files, doc_style and the merged request list), ready to serialise.
    """

    # Accumulators for the merged manifest; seen_snippets remembers which request first carried each snippet text.
    le_name = {"\n": "lf", "\r\n": "crlf", "\r": "cr"}
    files: List[dict] = []
    requests: List[dict] = []
    seen_snippets: Dict[str, str] = {}

    # One file record per part, with the line ending named symbolically so apply can restore it exactly.
    for source, language, line_ending, esc in parts:
        files.append({"path": source, "language": language, "line_ending": le_name.get(line_ending, "lf")})

        # Copy each request before tagging it with its file, leaving the per-file Escalation state untouched.
        for r in esc.requests:
            req = dict(r)
            req["file"] = source
            snippet = req.get("snippet")

            # Identical routine text must cross the wire only once, however many requests carry it.
            if snippet:
                prior = seen_snippets.get(snippet)

                # Later carriers of the same text drop it and point at the first carrier's id via snippet_ref.
                if prior is not None:
                    req["snippet"] = None
                    req["snippet_ref"] = prior
                else:
                    seen_snippets[snippet] = req["id"]

            requests.append(req)

    # Assemble the versioned envelope that the check, fragment and apply phases all consume.
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
    List every unfilled answer slot in a manifest, one label per slot.

    Null and whitespace-only answers both count as unfilled. Labels take the form `id:def` or `id:block[i]`, so the completeness report points at the exact slot.

    Parameters:
    - `manifest`: The manifest dict to scan.

    Returns:
    - A list of slot labels; empty when the manifest is complete.
    """

    out: List[str] = []

    # A def answer counts as filled only when it is a non-blank string; null and whitespace both flag it.
    for r in manifest.get("requests", []):
        d = r.get("def")
        if d is not None and not str(d.get("answer") or "").strip():
            out.append(f"{r.get('id', r.get('qualname', '?'))}:def")
        b = r.get("blocks")

        # Blank chunks are reported one label apiece, indexed by position, so the report names the exact slot.
        if b is not None:
            for i, chunk in enumerate(b.get("chunks", [])):
                if not str(chunk.get("answer") or "").strip():
                    out.append(f"{r.get('id', r.get('qualname', '?'))}:block[{i}]")

    return out


# ---------------------------- fragments (parallel fill protocol) ----------------------------


FRAGMENT_KEY = "fragment"


def _request_unfilled(req: dict) -> bool:
    """
    Report whether a request still has any blank answer slot.

    A blank def answer or any single blank block chunk makes the request unfilled; whitespace-only strings count as blank.

    Parameters:
    - `req`: The request dict to inspect.

    Returns:
    - `True` if at least one answer slot is empty, `False` once every slot is filled.
    """

    # Whitespace-only answers count as unfilled, the same rule the completeness report applies.
    d = req.get("def")
    if d is not None and not str(d.get("answer") or "").strip():
        return True
    b = req.get("blocks")

    # A single blank chunk is enough to keep the whole request in the claimable pool.
    if b is not None:
        for chunk in b.get("chunks", []):
            if not str(chunk.get("answer") or "").strip():
                return True

    return False


def next_fragment_name(manifest: dict, master_name: str) -> str:
    """
    Allocate the next fragment filename for a master manifest.

    The `fragments_issued` counter is stored in (and bumped on) the manifest itself, so numbering stays monotonic across separate emit runs. The fragment name is the master name with a zero-padded `.frag-NNN` inserted before its suffix.

    Parameters:
    - `manifest`: The master manifest dict; mutated to record the new counter.
    - `master_name`: The master manifest's filename.

    Returns:
    - The new fragment filename, e.g. `scale-manifest.frag-001.json`.
    """

    # The issue counter lives in the master manifest itself, so numbering stays monotonic across runs.
    stem, dot, suffix = master_name.rpartition(".")
    if not dot:
        stem, suffix = master_name, "json"
    issued = int(manifest.get("fragments_issued", 0)) + 1
    manifest["fragments_issued"] = issued

    return f"{stem}.frag-{issued:03d}.{suffix}"


def build_fragment(manifest: dict, size: int, fragment_name: str) -> Optional[dict]:
    """
    Carve a self-contained fragment of unfilled requests out of the master manifest.

    Up to `size` unfilled, unclaimed requests are stamped with the fragment name in the master - claiming them so concurrently issued fragments never overlap - then deep-copied into the fragment. A copy whose snippet carrier was not also picked has the snippet text inlined, so the fragment can be answered without sight of the master.

    Parameters:
    - `manifest`: The master manifest dict; picked requests are marked in place.
    - `size`: Maximum number of requests to include.
    - `fragment_name`: Recorded both as the claim stamp in the master and as the fragment's `fragment_of`.

    Returns:
    - The fragment manifest dict, or `None` when nothing is left to claim or `size` is below 1.
    """

    # Claim the first `size` unfilled, unclaimed requests by stamping them, so concurrently issued fragments never overlap.
    pool = [r for r in manifest.get("requests", []) if _request_unfilled(r) and not r.get(FRAGMENT_KEY)]
    if not pool or size < 1:
        return None
    picked = pool[:size]
    for r in picked:
        r[FRAGMENT_KEY] = fragment_name
    picked_ids = {r.get("id") for r in picked}
    snippets_by_id: Dict[str, str] = {}

    # Index every snippet carrier in the master so refs pointing outside this fragment can be resolved.
    for r in manifest.get("requests", []):
        if r.get("snippet") and r.get("id") not in snippets_by_id:
            snippets_by_id[r["id"]] = r["snippet"]

    out_requests: List[dict] = []

    # Deep-copy each pick; if its snippet carrier was not also picked, inline the text so the fragment is self-contained.
    for r in picked:
        req = json.loads(json.dumps(r))  # deep copy; the fragment must not alias the master
        ref = req.get("snippet_ref")
        if not req.get("snippet") and ref and ref not in picked_ids:
            req["snippet"] = snippets_by_id.get(ref)
        req.pop(FRAGMENT_KEY, None)
        out_requests.append(req)

    used_files = {r.get("file") for r in out_requests}

    # Mirror the master's envelope but list only the files these requests actually reference.
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
    Merge a completed fragment's answers back into the master manifest.

    The merge is fill-only: a fragment answer is copied only where the master slot is still blank, so existing answers are never overwritten. Requests are matched on (id, file) and chunks on `bidx`; rows the master does not know are ignored. Every matched master request has its fragment claim removed, so slots the fragment left blank become claimable again.

    Parameters:
    - `manifest`: The master manifest dict, updated in place.
    - `fragment`: The returned fragment dict.

    Returns:
    - The number of answer slots filled by this merge.
    """

    # Index master requests by (id, file); setdefault keeps the first row should duplicates ever occur.
    index: Dict[Tuple[str, str], dict] = {}
    for r in manifest.get("requests", []):
        index.setdefault((str(r.get("id")), str(r.get("file"))), r)
    filled = 0

    # Rows the master does not recognise are skipped, so a stale or foreign fragment cannot corrupt it.
    for fr in fragment.get("requests", []):
        mr = index.get((str(fr.get("id")), str(fr.get("file"))))
        if mr is None:
            continue
        md, fd = mr.get("def"), fr.get("def")

        # Fill-only def merge: an answer already present in the master is never overwritten.
        if md is not None and fd is not None:
            if not str(md.get("answer") or "").strip() and str(fd.get("answer") or "").strip():
                md["answer"] = fd["answer"]
                filled += 1

        mb, fb = mr.get("blocks"), fr.get("blocks")

        # Chunks are matched by their stable bidx, not by list position.
        if mb is not None and fb is not None:
            by_bidx = {c.get("bidx"): c for c in fb.get("chunks", [])}

            for chunk in mb.get("chunks", []):
                fc = by_bidx.get(chunk.get("bidx"))

                # Same fill-only rule for chunks, counting each transferred answer.
                if fc is not None and not str(chunk.get("answer") or "").strip() \
                        and str(fc.get("answer") or "").strip():
                    chunk["answer"] = fc["answer"]
                    filled += 1

        # Release the claim even if the fragment left slots blank, so leftovers become claimable again.
        mr.pop(FRAGMENT_KEY, None)

    return filled


def release_unfilled(manifest: dict) -> int:
    """
    Release the fragment claims on requests that are still unfilled.

    Recovers requests stranded by fragments that were issued but never merged back, returning them to the pool `build_fragment` draws from.

    Parameters:
    - `manifest`: The master manifest dict, updated in place.

    Returns:
    - The number of requests whose claim was released.
    """

    released = 0

    # Reclaim requests stranded by fragments that never came back, returning them to the claimable pool.
    for r in manifest.get("requests", []):
        if r.get(FRAGMENT_KEY) and _request_unfilled(r):
            r.pop(FRAGMENT_KEY, None)
            released += 1

    return released


# ---------------------------- manifest I/O ----------------------------


def write_manifest(path: Path, manifest: dict) -> None:
    """
    Write a manifest dict to disk as pretty-printed UTF-8 JSON with a trailing newline.

    Parameters:
    - `path`: Destination file path.
    - `manifest`: The manifest dict to serialise.
    """

    # Pretty-printed, non-ASCII-escaped JSON with a trailing newline keeps manifests readable and diff-friendly.
    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _upgrade_v1(manifest: dict) -> dict:
    """
    Convert a version-1 manifest into the current multi-file manifest shape.

    Version 1 described a single file with separate per-pass requests; the upgrade folds each request's pass-specific fields into the v2 `def`/`blocks` sub-objects and hoists the file metadata into a one-entry `files` list. Existing answers are carried across, but v1 stored no chunk line ranges, so those are left as `None`.

    Parameters:
    - `manifest`: The parsed version-1 manifest dictionary.

    Returns:
    - A new manifest dictionary in the current version's shape.
    """

    # A v1 manifest described exactly one file, so its single source path is stamped onto every request.
    source = manifest.get("source", "")
    requests: List[dict] = []

    # Rebuild each request in the v2 shape; v1 carried no per-request id, so derive one from the qualname and signature hash.
    for r in manifest.get("requests", []):
        req = {
            "id": request_id(r.get("qualname", "?"), r.get("sig_hash", "")),
            "qualname": r.get("qualname"),
            "kind": r.get("kind"),
            "sig_hash": r.get("sig_hash"),
            "snippet": r.get("snippet"),
            "file": source,
        }

        # Fold v1's separate per-pass requests into the v2 sub-objects; v1 stored no chunk line ranges, so those stay None.
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

    # Hoist the v1 top-level file metadata into v2's one-entry files list.
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
    Load and validate a SCALE manifest from disk.

    Version-1 manifests are upgraded in memory so callers always receive the current shape, and the `requests` and `files` keys are guaranteed to exist.

    Parameters:
    - `path`: Path to the manifest JSON file.

    Returns:
    - The manifest dictionary, in the current version's shape.

    Notes:
    Raises `ValueError` if the file is not a SCALE manifest of a recognised version.
    """

    # Only version 1 and the current version are accepted; v1 is upgraded in memory so callers only ever see one shape.
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") not in (1, MANIFEST_VERSION):
        raise ValueError(f"{path}: not a SCALE manifest of version 1 or {MANIFEST_VERSION}")
    if manifest.get("version") == 1:
        manifest = _upgrade_v1(manifest)
    manifest.setdefault("requests", [])
    manifest.setdefault("files", [])

    return manifest
