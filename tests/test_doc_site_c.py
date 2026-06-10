#!/usr/bin/env python3
"""
The C header/implementation documentation site (`--doc-site`): collect prototypes, plan where each function's doc
lives (confident pairing), and have the C worker honour the plan - skip a redirected definition, document the
prototype from the implementation body, fire `on_doc`, preserve code byte-for-byte, and feed the header doc into the
implementation's block pass via `doc_override`. Also the header-before-impl target ordering.

Model-free: the LLM is stubbed and only the parsing / planning / patch logic runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_c as sc                       # noqa: E402
from scale import _order_header_before_impl  # noqa: E402

NL = "\n"


class FakeLLM:
    """Records each generation prompt; returns a canned C block comment for the extractor."""
    def __init__(self, reply):
        self.prompts = []
        self._reply = reply

    def snippet_budget(self, messages, cfg):
        return 100000      # never elide

    def estimate_tokens(self, text):
        return 1

    def generate(self, messages, cfg=None):
        self.prompts.append(messages[-1]["content"])
        return self._reply


def _lines(s):
    return s.split(NL)


def test_iter_decls_with_info_c():
    src = (
        "#include <stdio.h>\n"
        "\n"
        "int clamp(int v, int lo, int hi);\n"          # 3: plain prototype
        "const char *name_of(\n"                        # 4-6: multi-line, pointer return
        "    int id\n"
        ");\n"
        "int (*callback)(int);\n"                        # 7: function-pointer variable -> ignored
        "int global_var;\n"                             # 8: variable -> ignored
        "typedef struct { int x; } Point;\n"            # 9: typedef -> ignored
        "#ifdef FEATURE\n"
        "void only_feature(void);\n"                    # 11: prototype nested in #ifdef
        "#endif\n"
        "int clamp(int v, int lo, int hi) {\n"          # 13: definition (not a prototype)
        "    int local_decl;\n"                         # local declaration -> ignored
        "    return v;\n"
        "}\n"
    )
    tree, sb = sc._parse_c(src)
    decls = {d.qualname: d for d in sc.iter_decls_with_info_c(tree, sb)}
    assert set(decls) == {"clamp", "name_of", "only_feature"}, set(decls)
    assert (decls["clamp"].start, decls["clamp"].end) == (3, 3)
    assert (decls["name_of"].start, decls["name_of"].end) == (4, 6)         # multi-line span
    assert decls["name_of"].kind == "declaration"
    assert (decls["only_feature"].start, decls["only_feature"].end) == (11, 11)

    # iter_symbols emits the prototypes as `declaration` symbols, but a name with a definition in the same file keeps
    # its `function` symbol (no duplicate); the definition is the resolution target, the prototype only seeds context.
    syms = {s.qualname: s for s in sc.iter_symbols(src, _lines(src))}
    assert syms["clamp"].kind == "function"            # def present in this file -> not shadowed by its prototype
    assert syms["name_of"].kind == "declaration"
    assert syms["only_feature"].kind == "declaration"


def _two_file_run(policy, header_target=True, impl_target=True):
    header = "/* clamps a value */\nint clamp(int v, int lo, int hi);\n"
    impl = (
        "int clamp(int v, int lo, int hi) {\n"
        "    if (v < lo) return lo;\n"
        "    if (v > hi) return hi;\n"
        "    return v;\n"
        "}\n"
    )
    hk, ik = "/proj/foo.h", "/proj/foo.c"
    files = [
        (hk, header_target, header, _lines(header)),
        (ik, impl_target, impl, _lines(impl)),
    ]
    return sc.plan_doc_sites_c(files, policy), hk, ik, header, impl


def test_planner_auto_pairs_and_skips():
    plan, hk, ik, _h, _i = _two_file_run("auto")
    assert plan.skip == {ik: {"clamp"}}                        # the definition's docstring is skipped
    assert plan.header_names == {hk: {"clamp"}}                # the prototype is the doc site
    assert plan.impl_file["clamp"] == ik
    assert plan.pairs == [(hk, ik)]                            # header ordered before its impl
    assert "return v" in plan.impl_snippet("clamp")           # prose generated from the body
    assert plan.has_work()


def test_planner_impl_policy_never_skips():
    plan, hk, ik, _h, _i = _two_file_run("impl")
    assert plan.skip == {}                                     # legacy: the impl keeps its docstring
    assert plan.header_names == {hk: {"clamp"}}                # the prototype is still documented...
    assert plan.impl_snippet("clamp") is None                 # ...but from the prototype, not the body
    assert plan.pairs == []


def test_planner_decl_only_and_ambiguous():
    # Decl-only (no definition in the run): document the prototype from the prototype text alone.
    header = "void boot(void);\n"
    plan = sc.plan_doc_sites_c([("/p/x.h", True, header, _lines(header))], "auto")
    assert plan.header_names == {"/p/x.h": {"boot"}} and plan.skip == {}
    assert plan.impl_snippet("boot") is None

    # Ambiguous: two definitions of one name across the run -> never redirected (but the prototype is still a site).
    header = "int dup(void);\n"
    a = "int dup(void) { return 1; }\n"
    b = "int dup(void) { return 2; }\n"
    files = [("/p/d.h", True, header, _lines(header)),
             ("/p/a.c", True, a, _lines(a)),
             ("/p/b.c", True, b, _lines(b))]
    plan = sc.plan_doc_sites_c(files, "auto")
    assert plan.skip == {}                                     # ambiguous -> not skipped
    assert plan.header_names == {"/p/d.h": {"dup"}}
    assert plan.impl_snippet("dup") is None                   # no unique body to draw on


def test_planner_reference_never_written():
    # Header is a target but the implementation is a read-only reference: document the header from the body, but never
    # skip (a reference cannot be edited) and add no ordering pair (the reference's block pass does not run).
    plan, hk, ik, _h, _i = _two_file_run("auto", impl_target=False)
    assert plan.skip == {}
    assert plan.header_names == {hk: {"clamp"}}
    assert plan.impl_snippet("clamp") is not None             # the reference still supplies the body
    assert plan.pairs == []

    # A prototype that lives only in a reference file is never a documentation site.
    plan = sc.plan_doc_sites_c([("/p/r.h", False, "void z(void);\n", ["void z(void);", ""])], "auto")
    assert plan.header_names == {} and not plan.has_work()


def test_worker_skips_def_and_documents_prototype():
    plan, hk, ik, header, impl = _two_file_run("auto")

    # The header target: the prototype is documented, prose generated FROM THE IMPL BODY, doc lands above the
    # prototype, on_doc fires, and the full doc is recorded for the block pass.
    llm = FakeLLM("/*\n * Clamp v into [lo, hi].\n */")
    visited = []
    out_h = sc.generate_language_comments(
        llm, object(), [], header, _lines(header),
        doc_plan=plan.for_file(hk), on_doc=lambda q, d: visited.append(q))
    text_h = NL.join(out_h)
    assert "Clamp v into [lo, hi]." in text_h
    assert text_h.index("Clamp v into") < text_h.index("int clamp(")   # doc above the prototype
    assert "return v" in llm.prompts[-1]                                # generated from the body
    assert visited == ["clamp"]                                        # on_doc fired for the header function
    assert plan.header_docs["clamp"] == "Clamp v into [lo, hi]."        # recorded for the block pass

    # The implementation target: clamp is redirected, so the definition is left BYTE-FOR-BYTE untouched.
    llm2 = FakeLLM("/*\n * SHOULD NOT APPEAR\n */")
    out_i = sc.generate_language_comments(llm2, object(), [], impl, _lines(impl), doc_plan=plan.for_file(ik))
    assert out_i == _lines(impl)                                       # code (and absence of a doc) preserved
    assert "SHOULD NOT APPEAR" not in NL.join(out_i)


def test_block_pass_uses_header_doc_override():
    plan, hk, ik, header, impl = _two_file_run("auto")
    plan.record_header_doc("clamp", "Clamp v into [lo, hi].")
    # Without the override the block target's doc is whatever sits above the (bare) impl header - here nothing.
    bare = sc.iter_block_targets_c(impl, _lines(impl))
    assert bare[0].doc == ""
    # With the override the redirected routine's contract is restored into the block-comment context.
    overridden = sc.iter_block_targets_c(impl, _lines(impl), doc_override=lambda n: plan.header_doc(n))
    assert overridden[0].doc == "Clamp v into [lo, hi]."


def test_order_header_before_impl():
    # The reorder moves a header ahead of its paired impl while otherwise preserving the input order.
    h = Path("foo.h")
    c = Path("foo.c")
    other = Path("other.c")
    hk, ik = str(h.resolve()), str(c.resolve())
    ordered = _order_header_before_impl([other, c, h], [(hk, ik)])
    names = [p.name for p in ordered]
    assert names.index("foo.h") < names.index("foo.c"), names
    # No constraints -> order is unchanged.
    assert _order_header_before_impl([c, h], []) == [c, h]


def main():
    test_iter_decls_with_info_c()
    test_planner_auto_pairs_and_skips()
    test_planner_impl_policy_never_skips()
    test_planner_decl_only_and_ambiguous()
    test_planner_reference_never_written()
    test_worker_skips_def_and_documents_prototype()
    test_block_pass_uses_header_doc_override()
    test_order_header_before_impl()
    print("PASS: --doc-site collects prototypes, plans confident pairs, redirects docs to the header, preserves the "
          "impl, and feeds the header doc into the block pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
