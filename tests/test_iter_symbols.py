#!/usr/bin/env python3
"""
The per-language `iter_symbols` call-graph extractors (Python / C / JS): each emits one `Symbol` per routine with its
parent qualname, existing-doc seed, and classified call sites (`free`/`self`/`method`), walking only the routine's own
body so nested definitions are opaque (their calls belong to them, not the enclosing routine).

Model-free: only the parsers run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scale_python as spy   # noqa: E402
import scale_c as sc         # noqa: E402
import scale_javascript as sjs  # noqa: E402


def _by_qual(symbols):
    return {s.qualname: s for s in symbols}


def test_python_symbols():
    src = (
        "def clamp(x):\n"
        '    """Clamp x."""\n'
        "    return x\n"
        "\n"
        "class Acc:\n"
        '    """An accumulator."""\n'
        "    def add(self, x):\n"
        "        y = clamp(x)\n"
        "        self.store(y)\n"
        "        return obj.transpose(y)\n"
        "    def store(self, y):\n"
        "        def inner():\n"
        "            return clamp(99)\n"
        "        pass\n"
    )
    syms = _by_qual(spy.iter_symbols(src, src.split("\n")))
    assert syms["clamp"].existing_doc.startswith("Clamp x")
    assert syms["Acc"].kind == "class" and syms["Acc"].calls == []          # class body: no own calls
    assert syms["Acc.add"].parent_qualname == "Acc"
    assert ("clamp", "free") in syms["Acc.add"].calls
    assert ("store", "self") in syms["Acc.add"].calls
    assert ("transpose", "method") in syms["Acc.add"].calls
    # The nested `inner` def is opaque: its clamp(99) call belongs to inner, not to Acc.store.
    assert syms["Acc.store"].calls == []
    assert ("clamp", "free") in syms["Acc.store.inner"].calls


def test_c_symbols():
    src = (
        "#include <stdio.h>\n"
        "\n"
        "/* Clamp v into range. */\n"
        "int clamp(int v, int lo, int hi) {\n"
        "    return v < lo ? lo : v;\n"
        "}\n"
        "\n"
        "int reduce(int *xs, int n) {\n"
        "    int acc = 0;\n"
        "    for (int i = 0; i < n; i++) acc += clamp(xs[i], 0, 9);\n"
        "    log_it(acc);\n"
        "    return acc;\n"
        "}\n"
    )
    syms = _by_qual(sc.iter_symbols(src, src.split("\n")))
    assert syms["clamp"].existing_doc.startswith("Clamp v")
    assert syms["clamp"].parent_qualname is None and syms["clamp"].depth == 0
    # C calls are all `free` (bare identifier); a function-pointer/field call would simply not be recorded.
    assert ("clamp", "free") in syms["reduce"].calls
    assert ("log_it", "free") in syms["reduce"].calls
    assert all(kind == "free" for _name, kind in syms["reduce"].calls)


def test_js_symbols():
    src = (
        "function clamp(x) { return x; }\n"
        "class Acc {\n"
        "  add(x) {\n"
        "    const y = clamp(x);\n"
        "    this.store(y);\n"
        "    return obj.transpose(y);\n"
        "  }\n"
        "  store(y) {\n"
        "    const inner = () => clamp(99);\n"
        "    return inner();\n"
        "  }\n"
        "}\n"
        "const reduce = (xs) => { let a = 0; for (const x of xs) a += clamp(x); return a; };\n"
    )
    syms = _by_qual(sjs.iter_symbols(src, src.split("\n")))
    assert syms["Acc"].kind == "class" and syms["Acc"].calls == []          # class body: no own calls
    assert ("clamp", "free") in syms["Acc.add"].calls
    assert ("store", "self") in syms["Acc.add"].calls          # this.store -> self
    assert ("transpose", "method") in syms["Acc.add"].calls    # obj.transpose -> method
    # The nested arrow `inner` is opaque: clamp(99) belongs to it, not Acc.store.
    assert ("clamp", "free") not in syms["Acc.store"].calls
    assert ("inner", "free") in syms["Acc.store"].calls
    assert ("clamp", "free") in syms["reduce"].calls           # a var_arrow is walked (its value is not a separate def)


def main():
    test_python_symbols()
    test_c_symbols()
    test_js_symbols()
    print("PASS: iter_symbols extracts parents, existing-doc seeds, and classified calls (nested defs opaque) in Py/C/JS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
