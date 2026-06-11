#!/usr/bin/env python3
"""
The call-graph layer (`scale_project`): the confident-only resolver, leaf-first SCC ordering, the coarse file order,
and the `ContractStore`. All pure and model-free - it operates on hand-built `Symbol` records, so no parsing or LLM is
involved.

Guards: free calls resolve same-file-first then run-wide-unique (and ambiguous stays unresolved); `self`/`this` link to
the enclosing class's own method; `obj.m()` links only when `m` is unique to one class; documentation order is
leaf-first with nesting (child before parent) and call (callee before caller) edges; recursion / mutual recursion
collapse into an SCC and terminate; the file order puts a callee's file first; and the store seeds from existing docs,
updates on generation, and caps/omits contract-less callees.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_project as sp  # noqa: E402


def S(qualname, kind="def", parent=None, doc="", calls=None, start=1, depth=0):
    """Terse Symbol factory for the tests."""
    return sp.Symbol(qualname=qualname, kind=kind, signature="", start=start, depth=depth,
                     parent_qualname=parent, existing_doc=doc, calls=list(calls or []))


def test_resolver_free():
    # Same-file beats a run-wide match; a run-wide-unique match resolves; multiple candidates stay unresolved.
    a = [S("clamp", doc="A clamp."), S("user", calls=[("clamp", "free")])]
    b = [S("clamp", doc="B clamp.")]
    g = sp.build_project_graph({"a.py": a, "b.py": b})
    # user calls clamp: a same-file clamp exists, so it resolves to a.py's clamp (not b.py's).
    assert g.edges[("a.py", "user")] == [("a.py", "clamp")]

    # With no same-file clamp but two run-wide, the call is ambiguous -> unresolved.
    c = [S("user", calls=[("clamp", "free")])]
    g2 = sp.build_project_graph({"a.py": a[:1], "b.py": b, "c.py": c})
    assert g2.edges[("c.py", "user")] == []   # two clamps run-wide, none same-file -> unresolved

    # Exactly one run-wide match resolves cross-file.
    g3 = sp.build_project_graph({"a.py": a[:1], "c.py": c})
    assert g3.edges[("c.py", "user")] == [("a.py", "clamp")]


def test_resolver_self_and_method():
    syms = [
        S("Box", kind="class", start=1),
        S("Box.set", parent="Box", start=2, depth=1, calls=[("norm", "self"), ("paint", "method")]),
        S("Box.norm", parent="Box", start=4, depth=1, doc="Normalise."),
        # `paint` is unique to one class -> obj.paint() resolves; add a second `draw` defined twice -> ambiguous.
        S("Canvas", kind="class", start=10),
        S("Canvas.paint", parent="Canvas", start=11, depth=1, doc="Paint it."),
        S("Sketch", kind="class", start=20),
        S("Sketch.draw", parent="Sketch", start=21, depth=1),
        S("Plot", kind="class", start=30),
        S("Plot.draw", parent="Plot", start=31, depth=1),
        S("user", start=40, calls=[("draw", "method")]),
    ]
    g = sp.build_project_graph({"m.py": syms})
    # self -> enclosing class's own method; method -> unique-name class method.
    assert ("m.py", "Box.norm") in g.edges[("m.py", "Box.set")]
    assert ("m.py", "Canvas.paint") in g.edges[("m.py", "Box.set")]
    # `draw` is defined by two classes -> ambiguous -> unresolved.
    assert g.edges[("m.py", "user")] == []


def test_order_leaf_first_and_nesting():
    syms = [
        S("leaf", doc="Leaf.", start=1),
        S("mid", start=5, calls=[("leaf", "free")]),
        S("top", start=10, calls=[("mid", "free")]),
        S("Outer", kind="class", start=20),
        S("Outer.m", parent="Outer", start=21, depth=1),
    ]
    g = sp.build_project_graph({"f.py": syms})
    order = [q for _f, q in g.order]
    # call edges: leaf before mid before top.
    assert order.index("leaf") < order.index("mid") < order.index("top")
    # nesting: child before parent.
    assert order.index("Outer.m") < order.index("Outer")


def test_order_handles_cycles():
    # Direct recursion and mutual recursion must terminate and place the cycle members together.
    syms = [
        S("a", calls=[("a", "free"), ("b", "free")], start=1),
        S("b", calls=[("a", "free")], start=2),
    ]
    g = sp.build_project_graph({"f.py": syms})
    order = [q for _f, q in g.order]
    assert set(order) == {"a", "b"}   # both present, no loop/dedup loss
    # self-recursion does not create a self-edge.
    assert ("f.py", "a") not in g.edges[("f.py", "a")]


def test_file_order():
    util = [S("clamp", doc="Clamp.")]
    core = [S("run", calls=[("clamp", "free")])]
    g = sp.build_project_graph({"util.py": util, "core.py": core})
    # core.run calls util.clamp -> util's file is documented first.
    assert g.file_order(["core.py", "util.py"]) == ["util.py", "core.py"]
    # An unconstrained file keeps input order.
    g2 = sp.build_project_graph({"x.py": [S("f")], "y.py": [S("h")]})
    assert g2.file_order(["y.py", "x.py"]) == ["y.py", "x.py"]


def test_contract_store():
    syms = [
        S("clamp", doc="Clamp x to range.\nMore detail."),
        S("helper"),   # no existing doc -> no seed
        S("run", calls=[("clamp", "free"), ("helper", "free")]),
    ]
    g = sp.build_project_graph({"f.py": syms})
    store = sp.ContractStore(g)
    # Seeded from existing doc (first line only); helper has none, so it is omitted from the notes.
    notes = store.callee_notes("f.py", "run")
    assert "clamp: Clamp x to range." in notes
    assert "More detail" not in notes
    assert "helper" not in notes
    # update() refines the contract from a freshly-generated docstring's first line.
    store.update("f.py", "helper", "Assist the run.\nblah")
    assert "helper: Assist the run." in store.callee_notes("f.py", "run")
    # A blank docstring does not overwrite a contract.
    store.update("f.py", "clamp", "")
    assert "clamp: Clamp x to range." in store.callee_notes("f.py", "run")


def test_callee_notes_cap():
    callees = [(f"c{i}", "free") for i in range(10)]
    syms = [S(f"c{i}", doc=f"Does c{i}.") for i in range(10)] + [S("hub", calls=callees)]
    g = sp.build_project_graph({"f.py": syms})
    store = sp.ContractStore(g)
    notes = store.callee_notes("f.py", "hub", cap=3)
    assert notes.count("\n- ") == 3   # capped


def test_widened_call_tuples_and_call_map():
    # New-style `(name, kind, line)` records resolve exactly like the old pairs (the line is ignored by resolution,
    # and the older 2-tuple form - used by the hand-built symbols above - keeps working). The per-call-site
    # resolution is exposed as `call_map` so the block-pass annotator can re-find calls by (name, kind).
    syms = [
        S("clamp", doc="Clamp."),
        S("user", start=5, calls=[("clamp", "free", 6), ("mystery", "free", 7)]),
    ]
    g = sp.build_project_graph({"f.py": syms})
    assert g.edges[("f.py", "user")] == [("f.py", "clamp")]
    assert g.call_map[("f.py", "user")] == {("clamp", "free"): ("f.py", "clamp")}   # unresolved call -> no entry


def test_store_contract_and_missing():
    # `contract` reads one key's contract; `missing_callee_contracts` lists the resolved callees still lacking one
    # (the lazy one-liner generator's work list) and empties as contracts arrive.
    syms = [
        S("clamp", doc="Clamp."),
        S("helper"),
        S("user", start=5, calls=[("clamp", "free", 6), ("helper", "free", 7)]),
    ]
    g = sp.build_project_graph({"f.py": syms})
    store = sp.ContractStore(g)
    assert store.contract(("f.py", "clamp")) == "Clamp."
    assert store.contract(("f.py", "helper")) is None
    assert store.missing_callee_contracts("f.py", "user") == [("f.py", "helper")]
    store.update("f.py", "helper", "Help out.")
    assert store.missing_callee_contracts("f.py", "user") == []
    assert store.contract(("f.py", "helper")) == "Help out."


def test_apply_doc_order():
    items = [("a", 3), ("b", 1), ("c", 2), ("d", 9)]
    qof = lambda it: it[0]
    ordered = sp.apply_doc_order(items, qof, ["c", "a"], fallback_key=lambda it: it[1])
    # Listed first in doc_order ('c','a'), then unlisted by fallback ('b'=1 before 'd'=9).
    assert [q for q, _ in ordered] == ["c", "a", "b", "d"]


def main():
    test_resolver_free()
    test_resolver_self_and_method()
    test_order_leaf_first_and_nesting()
    test_order_handles_cycles()
    test_file_order()
    test_contract_store()
    test_callee_notes_cap()
    test_widened_call_tuples_and_call_map()
    test_store_contract_and_missing()
    test_apply_doc_order()
    print("PASS: call-graph resolver (2- and 3-tuple call records), call_map, leaf-first/SCC ordering, file order, "
          "ContractStore (incl. contract/missing accessors), and apply_doc_order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
