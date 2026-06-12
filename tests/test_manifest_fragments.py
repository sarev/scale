#!/usr/bin/env python3
"""
The fragment protocol: SCALE owns the slot bookkeeping so parallel agents never share a file.

This guards (model-free):
- `build_fragment` checks out only unfilled, un-checked-out requests (disjoint batches across calls), marks the
  master, and produces a self-contained valid manifest - including inlining a `snippet_ref` whose target lives
  outside the fragment (an agent must never need the master),
- `merge_fragment` folds answers back by `(id, file)` + chunk `bidx`, never overwrites an already-filled master
  slot (first write wins), and releases the checkout markers,
- `release_unfilled` returns incomplete work to the pile so `build_fragment` hands it out again,
- `next_fragment_name` never reuses a name (monotonic issue counter persisted on the master),
- the whole flow round-trips through `write_manifest`/`read_manifest`.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale_escalate import (  # noqa: E402
    FRAGMENT_KEY, build_fragment, merge_fragment, next_fragment_name, read_manifest, release_unfilled,
    unfilled_answers, write_manifest,
)


def make_master():
    return {
        "version": 2,
        "tool": "scale",
        "files": [{"path": "a.py", "language": "python", "line_ending": "lf"},
                  {"path": "b.py", "language": "python", "line_ending": "lf"}],
        "escalate_cognitive": 10,
        "doc_style": "the style guide",
        "requests": [
            {"id": "fn:alpha:aaaa", "qualname": "alpha", "kind": "def", "sig_hash": "aaaa", "cognitive": 20,
             "snippet": "def alpha():\n    return 1", "file": "a.py",
             "def": {"answer": None},
             "blocks": {"doc_summary": "s", "length_note": "n",
                        "chunks": [{"bidx": 0, "lines": [1, 1], "answer": None},
                                   {"bidx": 2, "lines": [2, 2], "answer": None}]}},
            {"id": "fn:beta:bbbb", "qualname": "beta", "kind": "def", "sig_hash": "bbbb", "cognitive": 15,
             "snippet": None, "snippet_ref": "fn:alpha:aaaa", "file": "a.py",
             "def": {"answer": None}},
            {"id": "fn:gamma:cccc", "qualname": "gamma", "kind": "def", "sig_hash": "cccc", "cognitive": 12,
             "snippet": "def gamma():\n    return 3", "file": "b.py",
             "def": {"answer": "already answered"},
             "blocks": {"doc_summary": "s", "length_note": "n",
                        "chunks": [{"bidx": 1, "lines": [2, 2], "answer": None}]}},
        ],
    }


def test_checkout_is_disjoint_and_self_contained():
    m = make_master()
    f1 = build_fragment(m, 1, "m.frag-001.json")
    f2 = build_fragment(m, 1, "m.frag-002.json")
    f3 = build_fragment(m, 5, "m.frag-003.json")
    assert [r["id"] for r in f1["requests"]] == ["fn:alpha:aaaa"]
    assert [r["id"] for r in f2["requests"]] == ["fn:beta:bbbb"]
    assert [r["id"] for r in f3["requests"]] == ["fn:gamma:cccc"], "a partly-answered request is still in the pool"
    assert build_fragment(m, 5, "m.frag-004.json") is None, "everything is checked out"

    # The master carries the markers; the fragments do not.
    assert [r.get(FRAGMENT_KEY) for r in m["requests"]] == ["m.frag-001.json", "m.frag-002.json", "m.frag-003.json"]
    assert all(FRAGMENT_KEY not in r for f in (f1, f2, f3) for r in f["requests"])

    # beta's snippet_ref target (alpha) is outside its fragment, so the text is inlined; doc_style travels along.
    assert f2["requests"][0]["snippet"] == "def alpha():\n    return 1"
    assert f2["doc_style"] == "the style guide"
    assert [f["path"] for f in f2["files"]] == ["a.py"]

    # A fragment is a valid manifest: the ordinary counter works on it (agent self-check).
    assert unfilled_answers(f1) == ["fn:alpha:aaaa:def", "fn:alpha:aaaa:block[0]", "fn:alpha:aaaa:block[1]"]

    # When the ref target shares the fragment, the ref is kept and the text is NOT duplicated.
    m2 = make_master()
    both = build_fragment(m2, 2, "m.frag-001.json")
    assert both["requests"][1]["snippet"] is None and both["requests"][1]["snippet_ref"] == "fn:alpha:aaaa"


def test_merge_first_write_wins_and_releases():
    m = make_master()
    frag = build_fragment(m, 3, "m.frag-001.json")

    # The agent fills alpha fully, beta not at all, and tries to clobber gamma's already-filled def.
    fa, fb, fg = frag["requests"]
    fa["def"]["answer"] = "alpha doc"
    fa["blocks"]["chunks"][0]["answer"] = "why line 1"
    fa["blocks"]["chunks"][1]["answer"] = "NONE"
    fg["def"]["answer"] = "clobber attempt"
    fg["blocks"]["chunks"][0]["answer"] = "gamma chunk"

    filled = merge_fragment(m, frag)
    assert filled == 4, "alpha def + two chunks + gamma chunk; the clobber does not count"
    assert m["requests"][2]["def"]["answer"] == "already answered", "first write wins"
    assert all(FRAGMENT_KEY not in r for r in m["requests"]), "merge releases every covered request"

    assert unfilled_answers(m) == ["fn:beta:bbbb:def"], "beta went unanswered"
    again = build_fragment(m, 5, "m.frag-002.json")
    assert [r["id"] for r in again["requests"]] == ["fn:beta:bbbb"], "unanswered work is handed out again"


def test_release_unfilled_returns_work_to_pool():
    m = make_master()
    build_fragment(m, 3, "m.frag-001.json")
    assert build_fragment(m, 3, "m.frag-002.json") is None
    assert release_unfilled(m) == 3, "nothing was answered, so all three come back"
    assert build_fragment(m, 3, "m.frag-002.json") is not None


def test_names_and_io_round_trip():
    m = make_master()
    n1 = next_fragment_name(m, "scale-manifest.json")
    n2 = next_fragment_name(m, "scale-manifest.json")
    assert (n1, n2) == ("scale-manifest.frag-001.json", "scale-manifest.frag-002.json")

    frag = build_fragment(m, 1, n1)
    with tempfile.TemporaryDirectory() as tmp:
        mp, fp = Path(tmp) / "scale-manifest.json", Path(tmp) / n1
        write_manifest(mp, m)
        write_manifest(fp, frag)
        m_back, f_back = read_manifest(mp), read_manifest(fp)
        assert m_back["fragments_issued"] == 2 and m_back["requests"][0][FRAGMENT_KEY] == n1
        assert next_fragment_name(m_back, "scale-manifest.json") == "scale-manifest.frag-003.json"
        assert unfilled_answers(f_back) == unfilled_answers(frag), "the fragment survives the round trip"


def main():
    test_checkout_is_disjoint_and_self_contained()
    test_merge_first_write_wins_and_releases()
    test_release_unfilled_returns_work_to_pool()
    test_names_and_io_round_trip()
    print("PASS: fragments check out disjoint self-contained batches, merge first-write-wins, and release cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
