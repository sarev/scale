#!/usr/bin/env python3
"""
Online subdivision: a strong model may add finer block breaks/comments at statement-start lines (insertions).

The deterministic segmenter is conservative - a flat run of statements collapses into one coarse paragraph, so a
genuinely comment-worthy line (a cryptic shard-matching regex) is never even an addressable chunk. The online block
request now exposes every statement-start line (`stmt_lines`, suite leaders included) and an `insertions` slot the
model may fill with `{snippet_line: text}` - `""` for a paragraph break, text for a comment. Each is re-bound by
statement INDEX (not raw line number), so it survives the docstring line-shift between emit and apply exactly as a
chunk's bidx does, and lands through the same insertion-only patcher and code-preservation guard. No GGUF model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_python  # noqa: E402
from scale_python import iter_block_targets, apply_manifest  # noqa: E402
from scale_blocks import defer_block_targets, code_preserved, PYTHON_STYLE  # noqa: E402
from scale_escalate import Escalation, run_manifest, unfilled_answers  # noqa: E402


SRC = (
    "def fetch(repo_id, models_dir, include):\n"
    "    patterns = [include] if isinstance(include, str) else list(include)\n"
    "    out = build_path(models_dir, repo_id)\n"
    "    download(repo_id, out, patterns)\n"
    "    files = collect(out, patterns)\n"
    "    if not files:\n"
    "        raise FileNotFoundError(repo_id)\n"
    "    shard = next((p for p in files if matches(p)), None)\n"
)


def emit(offer_insertions=True):
    lines = SRC.split("\n")
    esc = Escalation(doc_style="s")
    defer_block_targets(esc, lines, iter_block_targets(SRC, lines), style=PYTHON_STYLE,
                        offer_insertions=offer_insertions)
    return lines, run_manifest([("t.py", "python", "\n", esc)], "s")


def main():
    lines, m = emit()
    blocks = m["requests"][0]["blocks"]
    snip = m["requests"][0]["snippet"].split("\n")

    # ---- 1. Every statement start is offered, suite leaders (the `raise`) included ----
    offered = {snip[L - 1].strip(): L for L in blocks["stmt_lines"]}
    for stmt in ("patterns = [include] if isinstance(include, str) else list(include)",
                 "shard = next((p for p in files if matches(p)), None)",
                 "raise FileNotFoundError(repo_id)"):
        assert stmt in offered, f"statement start not offered as an insertion point: {stmt!r}"

    # ---- 2. A docstring is added (def answer) AND the model subdivides; both land after the line-shift ----
    m["requests"][0]["def"] = {"answer": "Fetch a model and return its first shard."}
    for c in blocks["chunks"]:
        c["answer"] = "NONE"                         # decline the coarse baseline paragraph
    shard_line = offered["shard = next((p for p in files if matches(p)), None)"]
    download_line = offered["download(repo_id, out, patterns)"]
    blocks["insertions"] = {
        str(download_line): "",                      # a break-only insertion
        str(shard_line): "Pick the first shard, tolerating zero-padding",
    }

    assert unfilled_answers(m) == [], "insertions are optional and must not register as unfilled work"

    out = apply_manifest(lines, m)
    text = "\n".join(out)

    # The docstring landed (proving the shift happened) and the shard comment is right above the shard line.
    assert '"""Fetch a model' in text or "Fetch a model and return its first shard." in text, "def docstring missing"
    shard_idx = next(i for i, ln in enumerate(out) if ln.strip().startswith("shard ="))
    assert out[shard_idx - 1].strip() == "# Pick the first shard, tolerating zero-padding", \
        f"the shard comment must sit directly above the shard line; got {out[shard_idx - 1]!r}"
    # The break-only insertion produced a blank line above the download call.
    dl_idx = next(i for i, ln in enumerate(out) if ln.strip().startswith("download("))
    assert out[dl_idx - 1].strip() == "", "a break-only insertion must add a blank line above its statement"

    # ---- 3. Every original code line survives unchanged (a docstring is legitimately added on top) ----
    for cl in (l for l in SRC.split("\n") if l.strip()):
        assert cl in out, f"an original code line was lost or altered: {cl!r}"

    # ---- 4. Re-bind robustness: the insertion followed its statement past the inserted docstring ----
    # shard was snippet line 8 at emit; after a docstring is added it shifts down, yet the comment still lands on it.
    assert shard_idx > shard_line, "the shard line moved down after the docstring insert (shift really happened)"

    # ---- 5. A bad insertion (non-statement line) and a collision with a chunk are both dropped safely ----
    lines2, m2 = emit()
    b2 = m2["requests"][0]["blocks"]
    for c in b2["chunks"]:
        c["answer"] = "real comment"                 # chunk boundary now claims its line
    chunk_local = b2["chunks"][0]["lines"][0]
    b2["insertions"] = {str(chunk_local): "should be dropped (collides with the chunk)", "999": "no such line"}
    out2 = apply_manifest(lines2, m2)
    assert code_preserved(lines2, out2, PYTHON_STYLE), "dropped/garbage insertions must never corrupt code"
    assert not any("should be dropped" in ln for ln in out2), "an insertion colliding with a chunk must be dropped"

    # ---- 6. --no-subdivide (offer_insertions=False) leaves the slot off entirely ----
    _, m3 = emit(offer_insertions=False)
    assert "insertions" not in m3["requests"][0]["blocks"] and "stmt_lines" not in m3["requests"][0]["blocks"], \
        "opting out must leave the insertions slot off the manifest"

    print("PASS: online insertions add model-chosen breaks/comments at any statement start, index-rebound and guarded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
