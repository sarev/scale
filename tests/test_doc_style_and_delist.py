#!/usr/bin/env python3
"""
Two output-quality mitigations:

1. File-description de-listing: a summary that comes back as a list/headings is detected (`_looks_listy`), reflowed
   into prose by one model turn (`_reflow_if_listy`), and - if it is still listy - has its markers stripped
   deterministically (`_strip_list_markers`), so a doc-comment never carries list/heading syntax.
2. Doc-comment style detection: a C/JS file whose body comments are `//` (ignoring the leading banner) has its
   generated docs rendered as `//`, while a file that uses `/* */` (or a mix) keeps the block form.

Model-free: the helpers are pure and the reflow uses a stub LLM.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale                       # noqa: E402
import scale_c as sc               # noqa: E402
import scale_javascript as sjs     # noqa: E402
from scale_llm import GenerationConfig  # noqa: E402

CFG = GenerationConfig()


class FakeLLM:
    def __init__(self, reply):
        self._reply = reply
        self.calls = 0

    def generate(self, messages, cfg=None):
        self.calls += 1
        return self._reply


# ---------------------------------------------------------------- de-listing

def test_looks_listy():
    assert scale._looks_listy("1. **Loading**: x.\n2. **Saving**: y.")          # numbered + bold
    assert scale._looks_listy("- one\n- two\n- three")                          # bullets
    assert scale._looks_listy("## Heading\n## Another")                          # headings
    # Flowing prose (even with an em-dash / a single hyphen) is NOT a list.
    assert not scale._looks_listy("This file does X. It also does Y - efficiently. And Z.")
    assert not scale._looks_listy("")


def test_strip_list_markers():
    out = scale._strip_list_markers("1. **Loading**: does x.\n- a bullet\n## Heading\nplain.")
    assert "**" not in out and out.splitlines()[0] == "Loading: does x."
    assert out.splitlines()[1] == "a bullet"
    assert out.splitlines()[2] == "Heading"


def test_reflow_listy_via_model():
    listy = "1. **A**: does a.\n2. **B**: does b."
    # The model returns flowing prose -> used as-is.
    llm = FakeLLM("This file does A and then B, in flowing prose.")
    out = scale._reflow_if_listy(llm, CFG, [], listy, 256)
    assert llm.calls == 1 and not scale._looks_listy(out) and out.startswith("This file does A")


def test_reflow_falls_back_to_strip():
    listy = "1. **A**: does a.\n2. **B**: does b."
    # The model stubbornly returns a list again -> deterministic strip guarantees no markers remain.
    llm = FakeLLM("1. **A**: still a.\n2. **B**: still b.")
    out = scale._reflow_if_listy(llm, CFG, [], listy, 256)
    assert not scale._looks_listy(out) and "**" not in out


def test_reflow_noop_on_prose():
    prose = "This is already two sentences of flowing prose. Nothing to do."
    llm = FakeLLM("SHOULD NOT BE CALLED")
    out = scale._reflow_if_listy(llm, CFG, [], prose, 256)
    assert out == prose and llm.calls == 0


# ---------------------------------------------------------------- doc-comment style

C_HEADER_LINE_DOCS = (
    "/**\n * @file foo.h\n * Banner block comment (the leading zone is ignored).\n */\n"
    "#if !defined(FOO_H)\n#define FOO_H\n\n"
    "// Does a thing.\nextern void a(void);\n\n"
    "// Does another.\nextern int b(int x);\n#endif\n"
)
C_MIXED = (
    "/* banner */\n"
    "/* a block doc */\nint a(void) { return 1; }\n"
    "// a line doc\nint b(void) { return 2; }\n"
)


def test_detect_doc_style_c():
    t, sb = sc._parse_c(C_HEADER_LINE_DOCS)
    assert sc._detect_doc_style_c(t, sb) == "line"      # banner excluded; body docs are //
    t, sb = sc._parse_c(C_MIXED)
    assert sc._detect_doc_style_c(t, sb) == "block"     # a mix prefers the block form
    t, sb = sc._parse_c("int a(void) { return 1; }\n")
    assert sc._detect_doc_style_c(t, sb) == "block"     # no body comments -> default block


def test_patch_line_style_c():
    # With style='line' the generated doc renders as // lines, not a /* */ block.
    src = "int clamp(int v);\n"
    lines = src.split("\n")
    t, sb = sc._parse_c(src)
    defs = sc.iter_decls_with_info_c(t, sb)
    doc_map = {(d.header_start, d.header_end): "Clamp v.\nReturns the clamped value." for d in defs}
    out = sc.patch_comments_textually_c(lines, defs, doc_map, style="line")
    text = "\n".join(out)
    assert "// Clamp v." in text and "// Returns the clamped value." in text
    assert "/*" not in text
    # Block style (default) still produces a /* */ block.
    out_b = sc.patch_comments_textually_c(lines, defs, doc_map, style="block")
    assert "/*" in "\n".join(out_b)


def test_detect_and_line_render_js():
    js_line = (
        "/** banner */\n"
        "// makes a thing\nfunction a() { return 1; }\n"
        "// makes another\nfunction b() { return 2; }\n"
    )
    t, sb = sjs._parse_js(js_line)
    assert sjs._detect_doc_style_js(t, sb) == "line"
    assert sjs._render_js_line_comment("Hello.\nWorld.", "  ") == ["  // Hello.", "  // World."]
    # A JSDoc-bodied file stays block.
    js_block = "/** banner */\n/** does a */\nfunction a() { return 1; }\n"
    t, sb = sjs._parse_js(js_block)
    assert sjs._detect_doc_style_js(t, sb) == "block"


def main():
    test_looks_listy()
    test_strip_list_markers()
    test_reflow_listy_via_model()
    test_reflow_falls_back_to_strip()
    test_reflow_noop_on_prose()
    test_detect_doc_style_c()
    test_patch_line_style_c()
    test_detect_and_line_render_js()
    print("PASS: file-description de-listing (detect/reflow/strip) and C/JS doc-comment style detection + line rendering")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
