#!/usr/bin/env python3
"""
Copyright 2025 7th software Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific
language governing permissions and limitations under the License.

Selective escalation to a stronger model.

SCALE's local model is good enough for the bulk of the work - the whole-file summary, the structural segmentation of a
body into blocks, and the comments for simple routines. The comments that are *worth* a stronger model are those for the
genuinely involved routines, which is exactly where a small local model's prose gets unreliable. This module routes that
split: routines whose cognitive complexity exceeds a cutoff are deferred to a manifest, a stronger model (e.g. Claude
Code) fills in the comment text, and SCALE patches the answers back in through the same insertion-only path as everything
else - so the byte-for-byte code guarantee is unchanged regardless of which model produced the words.

Two phases, coordinated by a single JSON manifest:

- **emit**: SCALE runs both passes with the local model as usual, but for each routine above the cutoff it records a
  *request* (the routine's identity, its complexity, and what the stronger model needs to write a comment) instead of
  asking the local model, and leaves that routine untouched in the output. An `Escalation` object collects the requests;
  `to_manifest` serialises them.
- **apply**: a separate, model-free pass reads the manifest (now carrying the stronger model's `answer`s), re-parses the
  emit output, matches each request back to its routine by `(qualname, sig_hash)`, and patches the answers in.

The cutoff is `--escalate-cognitive`. Complexity is normally computed natively (see `scale_python.cognitive_complexity`)
so it lines up exactly with SCALE's own definition nodes; `--codestats-json` lets a precomputed report from the companion
`codestats` tool override it (useful for languages SCALE cannot score natively, or to use that tool as the source of
truth).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json


MANIFEST_VERSION = 1


# ---------------------------- complexity sources ----------------------------


def load_codestats_json(path: Path) -> Dict[str, int]:
    """
    Load a `codestats` JSON report and return a map of qualified name to cognitive complexity.

    `codestats` emits `{"functions": [{"function": "Foo.bar", "cognitive": N, ...}, ...]}`. Only the function name and
    its cognitive score are taken; everything else is ignored. When the same qualified name appears more than once
    (overloads, or two files scanned together), the highest score wins, so a routine is escalated if any namesake is
    complex.

    Parameters:
    - `path`: Path to the codestats JSON file.

    Returns:
    - A dictionary mapping each qualified name to its cognitive complexity.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, int] = {}
    for rec in data.get("functions", []):
        name = rec.get("function")
        if name is None:
            continue
        score = int(rec.get("cognitive", 0))
        out[name] = max(out.get(name, 0), score)
    return out


def request_id(pass_name: str, qualname: str, sig_hash: str) -> str:
    """
    Build the stable identifier for a manifest request from its pass, qualified name, and signature hash.

    Parameters:
    - `pass_name`: The pass that produced the request ("def" or "block").
    - `qualname`: The routine's fully qualified name.
    - `sig_hash`: The routine's signature hash (see `routine_sig_hash`).

    Returns:
    - The request identifier string, e.g. "block:Foo.bar:ab12cd34ef56".
    """

    return f"{pass_name}:{qualname}:{sig_hash}"


# ---------------------------- emit-phase collector + policy ----------------------------


@dataclass
class Escalation:
    """
    The emit-phase policy and request collector for selective escalation.

    An instance decides whether a routine is complex enough to defer to the stronger model (`should_escalate`) and, when
    it is, records what that model needs to answer (`record_def` / `record_block`). After both passes have run,
    `to_manifest` serialises the collected requests into the manifest dictionary written to disk. Only the local model
    runs during emit; this object never calls a model itself.

    Attributes:
    - `threshold`: The cognitive-complexity cutoff; a routine is escalated when its score is strictly greater.
    - `override`: Optional qualname to cognitive-complexity map (from `--codestats-json`) that overrides the native
      score for any routine it names; `None` means always use the native score supplied by the caller.
    - `doc_style`: The house-style docstring template (e.g. `comment.<lang>.txt` + `guidelines.md`), carried into the
      manifest so the stronger model is told the required style when it writes deferred docstrings.
    - `requests`: The collected escalation requests, in the order they were recorded.
    """

    threshold: int
    override: Optional[Dict[str, int]] = None
    doc_style: str = ""
    requests: List[dict] = field(default_factory=list)

    def score_for(self, qualname: str, native_score: int) -> int:
        """
        Return the cognitive score used for routing this routine.

        The native score (computed by the caller from SCALE's own AST) is used unless an override map was supplied and
        names this routine, in which case the override wins.

        Parameters:
        - `qualname`: The routine's fully qualified name.
        - `native_score`: The score SCALE computed natively for this routine.

        Returns:
        - The score to compare against the threshold.
        """

        if self.override is not None and qualname in self.override:
            return self.override[qualname]
        return native_score

    def should_escalate(self, qualname: str, native_score: int) -> bool:
        """
        Report whether a routine should be deferred to the stronger model.

        Parameters:
        - `qualname`: The routine's fully qualified name.
        - `native_score`: The score SCALE computed natively for this routine.

        Returns:
        - True when the routing score exceeds the threshold.
        """

        return self.score_for(qualname, native_score) > self.threshold

    def record_def(self, qualname: str, kind: str, sig_hash: str, cognitive: int, snippet: str) -> None:
        """
        Record a deferred definition (docstring/header-comment) request.

        Parameters:
        - `qualname`: The routine's fully qualified name.
        - `kind`: The routine kind ("def", "async def", or "class").
        - `sig_hash`: The routine's signature hash.
        - `cognitive`: The routing score that triggered escalation (for the manifest and logging).
        - `snippet`: The code snippet the local model would have been shown, given to the stronger model verbatim.
        """

        self.requests.append({
            "id": request_id("def", qualname, sig_hash),
            "pass": "def",
            "qualname": qualname,
            "kind": kind,
            "sig_hash": sig_hash,
            "cognitive": cognitive,
            "snippet": snippet,
            "answer": None,
        })

    def record_block(
        self,
        qualname: str,
        kind: str,
        sig_hash: str,
        cognitive: int,
        doc_summary: str,
        length_note: str,
        chunks: List[dict],
    ) -> None:
        """
        Record a deferred within-function block request.

        Segmentation has already run locally (it is structural and deterministic); only the per-chunk comment *text* is
        deferred. Each chunk is identified by its boundary index - the position of its start line within the routine's
        sorted legal boundaries - which is stable across the line shifts between emit and apply because the escalated
        routine is left untouched.

        Parameters:
        - `qualname`: The routine's fully qualified name.
        - `kind`: The routine kind ("def", "async def", or "class").
        - `sig_hash`: The routine's signature hash.
        - `cognitive`: The routing score that triggered escalation.
        - `doc_summary`: The routine's one-line docstring summary, as context for the comment.
        - `length_note`: The short/long length note SCALE would have used, so the model keeps the same bias.
        - `chunks`: One dict per chunk, each `{"bidx": int, "text": str}`; an `"answer": None` slot is added here.
        """

        self.requests.append({
            "id": request_id("block", qualname, sig_hash),
            "pass": "block",
            "qualname": qualname,
            "kind": kind,
            "sig_hash": sig_hash,
            "cognitive": cognitive,
            "doc_summary": doc_summary,
            "length_note": length_note,
            "chunks": [{"bidx": c["bidx"], "text": c["text"], "answer": None} for c in chunks],
        })

    def to_manifest(self, source: str, language: str, line_ending: str) -> dict:
        """
        Serialise the collected requests into the manifest dictionary.

        Parameters:
        - `source`: The source file path (recorded for provenance / the apply phase).
        - `language`: The resolved language identifier.
        - `line_ending`: The detected line ending of the source.

        Returns:
        - The manifest dictionary ready to be written as JSON.
        """

        return {
            "version": MANIFEST_VERSION,
            "tool": "scale",
            "source": source,
            "language": language,
            "line_ending": {"\n": "lf", "\r\n": "crlf", "\r": "cr"}.get(line_ending, "lf"),
            "escalate_cognitive": self.threshold,
            "doc_style": self.doc_style,
            "requests": self.requests,
        }


# ---------------------------- manifest I/O ----------------------------


def write_manifest(path: Path, manifest: dict) -> None:
    """
    Write a manifest dictionary to disk as indented JSON (UTF-8).

    Parameters:
    - `path`: The destination file path.
    - `manifest`: The manifest dictionary to serialise.
    """

    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict:
    """
    Read and lightly validate a manifest file.

    Parameters:
    - `path`: The manifest file path.

    Returns:
    - The parsed manifest dictionary.

    Raises:
    - `ValueError`: If the file is not a manifest of a supported version.
    """

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"{path}: not a SCALE manifest of version {MANIFEST_VERSION}")
    manifest.setdefault("requests", [])
    return manifest
