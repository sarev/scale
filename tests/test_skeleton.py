#!/usr/bin/env python3
"""
The model-free file skeleton and the two-pass description flow it feeds.

`scale_project.render_skeleton` distils a file to its leading comments, signatures, existing docs, and (for C)
top-level #defines - no bodies - so the whole-file description is generated from a fraction of the file. The guard is
binary: a file with no symbols at all returns None and keeps the whole-file summary path.

This guards (model-free):
- the Python skeleton: leading zone (module docstring/comments) + signatures + existing docstrings, body statements
  absent; methods indented under their class header,
- the C skeleton: #defines kept (multi-line macro bodies elided), prototypes kept, a prototype whose definition is in
  the same file deduplicated, the doc comment above a header kept,
- the binary guard (no symbols -> None),
- pass-1 priming uses the skeleton (the summary turn sees the skeleton, not the bodies),
- the pipeline order: the file-doc pass now runs LAST, so the published description is generated from the annotated
  text's skeleton (it sees the def pass's fresh docstrings).
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scale import prime_llm_for_comments, SummaryCache, generate_comments  # noqa: E402
from scale_llm import GenerationConfig  # noqa: E402
from scale_project import render_skeleton  # noqa: E402
from scale_python import iter_symbols as iter_symbols_py  # noqa: E402
import scale_c  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCALE_CFG = ROOT / "scale-cfg"


PY_SRC = '''"""Module overview prose."""

# A leading comment.


def helper(x):
    """Return x doubled."""
    secret_body_token = x * 2
    return secret_body_token


class Widget:
    """A widget."""

    def render(self):
        another_body_token = 1
        return another_body_token
'''


def test_python_skeleton():
    lines = PY_SRC.split("\n")
    skel = render_skeleton(lines, "python", iter_symbols_py(PY_SRC, lines))
    assert skel is not None
    assert "Module overview prose." in skel and "# A leading comment." in skel
    assert "def helper(x):" in skel and "Return x doubled." in skel
    assert "class Widget:" in skel and "    def render(self):" in skel, "methods sit indented under the class header"
    assert "secret_body_token" not in skel and "another_body_token" not in skel, "bodies must not appear"


C_SRC = (
    "/* Header banner. */\n"
    "#define MAX_LINES 100\n"
    "#define BIG_MACRO(x) \\\n"
    "    do_something(x)\n"
    "\n"
    "int helper(int x);\n"
    "int orphan_proto(void);\n"
    "\n"
    "/* Doubles x. */\n"
    "int helper(int x)\n"
    "{\n"
    "    int c_body_token = x * 2;\n"
    "    return c_body_token;\n"
    "}\n"
)


def test_c_skeleton():
    lines = C_SRC.split("\n")
    skel = render_skeleton(lines, "c", scale_c.iter_symbols(C_SRC, lines))
    assert skel is not None
    assert "Header banner." in skel
    assert "#define MAX_LINES 100" in skel
    assert "#define BIG_MACRO(x) ..." in skel, "a multi-line macro keeps its first line only"
    assert "do_something" not in skel, "the macro body is elided"
    assert "orphan_proto" in skel, "a declaration-only prototype is kept"
    assert skel.count("int helper(int x)") == 1, "a prototype with a same-file definition is deduplicated"
    assert "Doubles x." in skel, "the existing doc above the definition is kept"
    assert "c_body_token" not in skel, "bodies must not appear"


def test_guard_no_symbols_returns_none():
    src = "# just a comment\nDATA = [1, 2, 3]\n"
    assert render_skeleton(src.split("\n"), "python", iter_symbols_py(src, src.split("\n"))) is None
    assert render_skeleton([], "python", []) is None


class FakeLLM:
    """Stub model with a large window; records every summary-turn prompt so tests can see what was summarised."""
    n_ctx = 100_000
    ctx_margin = 0

    def __init__(self):
        self.summary_prompts = []

    def estimate_tokens(self, text):
        return len(text) // 4

    def count_tokens(self, messages):
        return sum(self.estimate_tokens(m["content"]) for m in messages)

    def snippet_budget(self, messages, cfg):
        return 100_000

    def generate(self, messages, *, cfg=None, stop=None):
        prompt = messages[-1]["content"]
        if "skeleton" in prompt or "source file" in prompt:
            self.summary_prompts.append(prompt)
            return "A description of the file."
        if "docstring" in prompt:
            return '"""\nFRESH_DOC: does the thing.\n"""'
        if "DESCRIPTION" in prompt:       # the file-doc classify turn
            return "NONE"
        return "OK"


def test_pass1_priming_summarises_the_skeleton():
    with tempfile.TemporaryDirectory() as tmp:
        SummaryCache._CACHE_DIR = Path(tmp)
        SummaryCache._CACHE_INDEX = Path(tmp) / "index.pkl"

        lines = PY_SRC.split("\n")
        skel = render_skeleton(lines, "python", iter_symbols_py(PY_SRC, lines))
        llm = FakeLLM()
        prime_llm_for_comments(llm, GenerationConfig(), SCALE_CFG, Path("virtual.py"),
                               source_blob=PY_SRC, language="python", no_cache=True, template="blocks",
                               skeleton=skel)
    assert llm.summary_prompts, "the priming must generate a summary"
    assert "secret_body_token" not in llm.summary_prompts[0], "the summary turn must see the skeleton, not bodies"
    assert "def helper(x):" in llm.summary_prompts[0]


def test_file_doc_runs_last_on_the_annotated_skeleton():
    # A def pass writes FRESH_DOC docstrings; the file-doc pass (last) generates the published description from the
    # CURRENT text's skeleton, so its summary turn must see FRESH_DOC.
    with tempfile.TemporaryDirectory() as tmp:
        SummaryCache._CACHE_DIR = Path(tmp)
        SummaryCache._CACHE_INDEX = Path(tmp) / "index.pkl"

        src = "def helper(x):\n    body_token = x * 2\n    return body_token\n"
        skel = render_skeleton(src.split("\n"), "python", iter_symbols_py(src, src.split("\n")))
        out_path = Path(tmp) / "out.py"
        llm = FakeLLM()
        rc = generate_comments(llm, GenerationConfig(), SCALE_CFG, Path(tmp) / "virtual.py", out_path,
                               src, src.split("\n"), "\n", "python", no_cache=True,
                               do_comment=True, do_blocks=False, do_file_doc=True, skeleton=skel)
        assert rc == 0
        out = out_path.read_text(encoding="utf-8")

    assert "FRESH_DOC: does the thing." in out, "the def pass docstring must be in the output"
    assert out.lstrip().startswith('"""'), "the file-doc pass must have inserted a module docstring"
    assert "A description of the file." in out
    published = [p for p in llm.summary_prompts if "FRESH_DOC" in p]
    assert published, "the published description must be generated from the ANNOTATED text's skeleton"
    assert all("body_token" not in p for p in llm.summary_prompts), "no summary turn may see function bodies"


def main():
    test_python_skeleton()
    test_c_skeleton()
    test_guard_no_symbols_returns_none()
    test_pass1_priming_summarises_the_skeleton()
    test_file_doc_runs_last_on_the_annotated_skeleton()
    print("PASS: the skeleton renderer distils signatures+docs (no bodies), guards no-symbol files, and the "
          "file-doc pass publishes from the annotated skeleton, last")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
