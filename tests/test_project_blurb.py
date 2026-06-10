#!/usr/bin/env python3
"""
Project-aware SCALE: locate a project-overview document and distil it into a short, cached "project blurb"
that is injected into every file's priming context.

Model-free (a fake LLM that returns a canned blurb). Guards:
- `find_project_doc` prefers CLAUDE.md, matches README variants/casing, walks up, and stops at a .git repo root;
- `resolve_project_doc` honours 'none' (disabled), an explicit path, and '' (auto-detect);
- `project_blurb` distils via the model and caches (a second call does no generation), with `no_cache` forcing regen;
- `prime_llm_for_comments` injects the blurb as a turn BEFORE the file-summary overview turn.
"""
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_project  # noqa: E402
from scale import prime_llm_for_comments, SummaryCache  # noqa: E402
from scale_llm import GenerationConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCALE_CFG = ROOT / "scale-cfg"


@dataclass
class _Cfg:
    temperature: float = 0.2
    max_new_tokens: int = 512


class _FakeLLM:
    n_ctx = 100_000
    ctx_margin = 0

    def __init__(self, reply="A BASIC interpreter; this is the distilled blurb."):
        self.reply = reply
        self.calls = 0

    def estimate_tokens(self, text):
        return len(text) // 4

    def count_tokens(self, messages):
        return sum(self.estimate_tokens(m["content"]) for m in messages)

    def generate(self, messages, *, cfg=None, stop=None):
        self.calls += 1
        return self.reply


def test_find_project_doc():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / ".git").mkdir()
        (root / "README.md").write_text("readme", encoding="utf-8")
        sub = root / "src" / "core"
        sub.mkdir(parents=True)

        # Nearest doc wins: a README two levels up is found from a deep file.
        f = sub / "thing.c"
        f.write_text("int main(void){return 0;}", encoding="utf-8")
        assert scale_project.find_project_doc(f) == root / "README.md"

        # CLAUDE.md is preferred over a README in the same directory.
        (sub / "README.txt").write_text("local", encoding="utf-8")
        (sub / "CLAUDE.md").write_text("claude", encoding="utf-8")
        assert scale_project.find_project_doc(f) == sub / "CLAUDE.md"

        # Casing/variant: a bare 'ReadMe' with no CLAUDE.md is matched.
        (sub / "CLAUDE.md").unlink()
        (sub / "README.txt").unlink()
        (sub / "ReadMe").write_text("variant", encoding="utf-8")
        assert scale_project.find_project_doc(f) == sub / "ReadMe"

    # No doc anywhere up to the repo root -> None.
    with tempfile.TemporaryDirectory() as tmp2:
        root = Path(tmp2)
        (root / ".git").mkdir()
        f = root / "a.c"
        f.write_text("x", encoding="utf-8")
        assert scale_project.find_project_doc(f) is None


def test_resolve_project_doc():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        doc = root / "CLAUDE.md"
        doc.write_text("proj", encoding="utf-8")
        f = root / "a.py"
        f.write_text("x = 1", encoding="utf-8")

        assert scale_project.resolve_project_doc("none", f) is None
        assert scale_project.resolve_project_doc(str(doc), f) == doc
        assert scale_project.resolve_project_doc("", f) == doc
        assert scale_project.resolve_project_doc(str(root / "missing.md"), f) is None


def test_project_blurb_generates_and_caches():
    with tempfile.TemporaryDirectory() as tmp:
        scale_project._CACHE_DIR = Path(tmp) / "__cache__"
        doc = Path(tmp) / "CLAUDE.md"
        doc.write_text("This project is a BBC BASIC interpreter with a tokenised program model.", encoding="utf-8")

        llm = _FakeLLM()
        blurb = scale_project.project_blurb(llm, _Cfg(), SCALE_CFG, doc, no_cache=False)
        assert blurb == "A BASIC interpreter; this is the distilled blurb." and llm.calls == 1

        # Second call hits the cache: no further generation.
        blurb2 = scale_project.project_blurb(_FakeLLM(reply="SHOULD NOT BE USED"), _Cfg(), SCALE_CFG, doc)
        assert blurb2 == blurb, "the cached blurb should be returned unchanged"

        # no_cache forces regeneration.
        llm3 = _FakeLLM(reply="regenerated blurb")
        assert scale_project.project_blurb(llm3, _Cfg(), SCALE_CFG, doc, no_cache=True) == "regenerated blurb"
        assert llm3.calls == 1


def test_blurb_injected_before_summary():
    BLURB = "PROJECT_BLURB_SENTINEL: a BBC BASIC interpreter."
    with tempfile.TemporaryDirectory() as tmp:
        SummaryCache._CACHE_DIR = Path(tmp)
        SummaryCache._CACHE_INDEX = Path(tmp) / "index.pkl"
        msgs = prime_llm_for_comments(
            _FakeLLM(reply="file summary"), GenerationConfig(), SCALE_CFG, Path("virtual.c"),
            source_blob="int main(void){return 0;}", language="c", no_cache=True, project_context=BLURB,
        )

    blurb_idx = next((i for i, m in enumerate(msgs) if BLURB in m["content"]), None)
    overview_idx = next((i for i, m in enumerate(msgs) if "here is an overview" in m["content"]), None)
    assert blurb_idx is not None, "the project blurb must be injected into the priming context"
    assert overview_idx is not None and blurb_idx < overview_idx, \
        "the blurb must come before the file-summary overview turn"


def main():
    test_find_project_doc()
    test_resolve_project_doc()
    test_project_blurb_generates_and_caches()
    test_blurb_injected_before_summary()
    print("PASS: project doc discovery, blurb distil+cache, and blurb injection before the file summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
