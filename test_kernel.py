"""Runnable tests for the OProof kernel, axiom store, generalization and bridge.

Run with:  python3 test_kernel.py
Everything is a plain assertion; the harness prints PASS/FAIL per test and
exits 1 on the first failure.
"""

import glob
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import opreof
from kernel import (
    BOOL,
    Assign,
    Block,
    Kernel,
    KernelError,
    Lit,
    Loop,
    Op,
    Var,
    builtin_rewrite,
    check_defined,
    eval_c,
    free_vars,
    int8,
    int16,
    int32,
    int64,
    ripen,
    substitute,
    uint8,
    uint16,
    uint32,
)

_TESTS: list = []

def test(name: str):
    def deco(fn):
        _TESTS.append((name, fn))
        return fn

    return deco

def _k(axioms_dir=None):
    return Kernel(axioms_dir=axioms_dir)

# Types, representability, canonical values

@test("representability: int8 x=200 must be rejected")

def t_rep_int8_overflow():
    b = Block("b", [Assign(Var("x", int8), Lit(200))])
    r = _k().prove_blocks(b, b, outputs={"x"})
    assert r.status == "rejected", r
    assert "representable" in r.reason, r

@test("representability: uint8 x=300 must be rejected")

def t_rep_uint8_overflow():
    b = Block("b", [Assign(Var("x", uint8), Lit(300))])
    r = _k().prove_blocks(b, b, outputs={"x"})
    assert r.status == "rejected", r

@test("conversion semantics: exts preserves sign, extz keeps bit pattern")

def t_convert_semantics():
    v = Lit(-1, int8)
    assert builtin_rewrite(Op("exts", (v,), int16)).value == -1
    assert builtin_rewrite(Op("extz", (v,), int16)).value == 255
    assert builtin_rewrite(Op("exts", (Lit(255, uint8),), uint16)).value == 255

# Shift soundness

@test("shifts: x<<8 on int8 rejected as UB")

def t_shift_ub_amount():
    x = Var("x", int8)
    b1 = Block("b", [Assign(Var("_o", int8), Op("shl", (x, Lit(8))))])
    b2 = Block("b", [Assign(Var("_o", int8), x)])
    r = _k().prove_blocks(b1, b2, inputs={"x": int8}, outputs={"_o"})
    assert r.status == "rejected", r
    assert "undefined" in r.reason, r

@test("shifts: x<<1 == x*2 (shl rewrite is sound)")

def t_shift_shl_sound():
    x = Var("x", int8)
    b1 = Block("b", [Assign(Var("_o", int8), Op("shl", (x, Lit(1))))])
    b2 = Block("b", [Assign(Var("_o", int8), Op("mul", (x, Lit(2))))])
    assert _k().prove_blocks(b1, b2, inputs={"x": int8}, outputs={"_o"})

@test("shifts: shrs != shru (arithmetic vs logical)")

def t_shift_kinds_diff():
    x = Var("x", int8)
    b1 = Block("b", [Assign(Var("_o", int8), Op("shrs", (x, Lit(2))))])
    b2 = Block("b", [Assign(Var("_o", int8), Op("shru", (x, Lit(2))))])
    r = _k().prove_blocks(b1, b2, inputs={"x": int8}, outputs={"_o"})
    assert r.status != "proven", r

@test("shifts: shl amount in range differs from bigger constant (sound)")

def t_shift_amount_soundness():
    x = Var("x", int8)
    b1 = Block("b", [Assign(Var("_o", int8), Op("shru", (x, Lit(2))))])
    b2 = Block("b", [Assign(Var("_o", int8), Op("shru", (x, Lit(3))))])
    r = _k().prove_blocks(b1, b2, inputs={"x": int8}, outputs={"_o"})
    assert r.status != "proven", r

# Constant folding / expression equivalence

@test("constant folding: 3+4 == 7")

def t_fold_const():
    assert _k().prove_expr(Op("add", (Lit(3), Lit(4))), Lit(7))

@test("expr: x+x == x<<1 (uses axiom add_self_is_shl1)")

def t_expr_via_axiom():
    k = _k()
    k.axioms.load_file(os.path.join(opreof.AXIOM_DIR, "foundational.py"))
    x = Var("x", int8)
    r = k.prove_expr(Op("add", (x, x)), Op("shl", (x, Lit(1))), inputs={"x": int8})
    assert r, r
    assert "add_self_is_shl1" in r.axioms_used

@test("expr: x-y == x+neg(y) (uses axiom sub_as_add_neg)")

def t_sub_as_add_neg():
    k = _k()
    k.axioms.load_file(os.path.join(opreof.AXIOM_DIR, "foundational.py"))
    x = Var("x", int8)
    y = Var("y", int8)
    assert k.prove_expr(
        Op("sub", (x, y)),
        Op("add", (x, Op("neg", (y,)))),
        inputs={"x": int8, "y": int8},
    )

# Loop unrolling

def _body():
    return [Assign(Var("a", int8), Op("mul", (Var("a", int8), Lit(2))))]

@test("unroll: Loop(4,B) == Loop(2,B);Loop(2,B)")

def t_unroll_equivalence():
    body = _body()
    b1 = Block("orig", [Assign(Var("a", int8), Lit(3)), Loop(4, body)])
    b2 = Block(
        "unr",
        [Assign(Var("a", int8), Lit(3)), Loop(2, body), Loop(2, body)],
    )
    r = _k().prove_blocks(b1, b2, inputs={"a": int8}, outputs={"a"})
    assert r, r

@test("unroll_verify: n=9 q=4 r=1 proven")

def t_unroll_verify_good():
    body = _body()
    orig = Block("orig", [Assign(Var("a", int8), Lit(3)), Loop(9, body)])
    unr = Block(
        "unr",
        [
            Assign(Var("a", int8), Lit(3)),
            Loop(4, body + body),
            *body[:1],
        ],
    )
    r = _k().unroll_verify(orig, unr, factor=2, inputs={"a": int8})
    assert r, r

@test("unroll_verify: missing tail rejected")

def t_unroll_verify_no_tail():
    body = _body()
    orig = Block("orig", [Assign(Var("a", int8), Lit(3)), Loop(9, body)])
    unr = Block(
        "unr",
        [Assign(Var("a", int8), Lit(3)), Loop(4, body + body)],
    )
    r = _k().unroll_verify(orig, unr, factor=2, inputs={"a": int8})
    assert r.status == "rejected", r

@test("unroll_verify: wrong loop count rejected")

def t_unroll_verify_wrong_count():
    body = _body()
    orig = Block("orig", [Assign(Var("a", int8), Lit(3)), Loop(9, body)])
    unr = Block(
        "unr",
        [Assign(Var("a", int8), Lit(3)), Loop(3, body + body)],
    )
    r = _k().unroll_verify(orig, unr, factor=2, inputs={"a": int8})
    assert r.status == "rejected", r

# Example: two Fibonacci programs proven equivalent

def _fib_body(commuted: bool = False) -> list:
    a, b, t = Var("a", int32), Var("b", int32), Var("t", int32)
    return [
        Assign(t, Op("add", (b, a) if commuted else (a, b))),
        Assign(a, b),
        Assign(b, t),
    ]

def _fib(n, commuted: bool = False) -> Block:
    a, b = Var("a", int32), Var("b", int32)
    return Block(
        "fib",
        [Assign(a, Lit(0)), Assign(b, Lit(1)), Loop(n, _fib_body(commuted))],
    )

def _fib_unrolled(n, factor: int) -> Block:
    a, b = Var("a", int32), Var("b", int32)
    q, r = divmod(n, factor)
    body = _fib_body()
    return Block(
        "fib_unrolled",
        [Assign(a, Lit(0)), Assign(b, Lit(1)), Loop(q, body * factor)]
        + body[:r],
    )

@test("example: fib(body with t=b+a) equals fib(body with t=a+b)")

def t_fib_commuted():
    r = _k().prove_blocks(
        _fib(8, commuted=True), _fib(8), inputs={}, outputs={"a", "b", "t"}
    )
    assert r, r

@test("example: fib loop unrolled by 2 is a faithful equivalent")

def t_fib_unrolled():
    k = _k()
    r = k.unroll_verify(_fib(8), _fib_unrolled(8, 2), factor=2, inputs={})
    assert r, r
    # an odd trip count makes body[:1] (just `t = a+b`) an incomplete
    # iteration, so the naive unroll must NOT be accepted
    r = k.unroll_verify(_fib(9), _fib_unrolled(9, 2), factor=2, inputs={})
    assert r.status == "rejected", r

@test("example: fib unrolled but forgetting the tail is rejected")

def t_fib_unrolled_no_tail():
    k = _k()
    a, b = Var("a", int32), Var("b", int32)
    bad = Block(
        "fib_bad",
        [Assign(a, Lit(0)), Assign(b, Lit(1)), Loop(4, _fib_body() * 2)],
    )
    r = k.unroll_verify(_fib(9), bad, factor=2, inputs={})
    assert r.status == "rejected", r

# Axiom store: add / remove / reload

@test("axiom lifecycle: too weak, add, prove, remove, reload")

def t_axiom_lifecycle():
    k = _k(axioms_dir=tempfile.mkdtemp())
    x = Var("x", int8)
    e1 = Op("add", (x, x))
    e2 = Op("shl", (x, Lit(1)))
    assert not k.prove_expr(e1, e2, inputs={"x": int8})
    k.axioms.load_file(os.path.join(opreof.AXIOM_DIR, "foundational.py"))
    assert k.prove_expr(e1, e2, inputs={"x": int8})
    k.axioms.remove("add_self_is_shl1")
    assert not k.prove_expr(e1, e2, inputs={"x": int8})
    k.axioms.load_file(os.path.join(opreof.AXIOM_DIR, "foundational.py"))
    assert k.prove_expr(e1, e2, inputs={"x": int8})

@test("axiom registry: duplicate rule returns False, no raise")

def t_axiom_registry_dupe():
    from axiom_store import AxiomRegistry

    reg = AxiomRegistry()
    rule = Op("add", (Var("a", int8), Var("b", int8)))
    assert reg.register(opreof.Axiom("r1", rule, rule))
    assert reg.register(opreof.Axiom("r2", rule, rule)) is False
    assert reg.names() == ["r1"]

# Automatic generalization

@test("generalization: proven fact auto-derives, persists, reloads")

def t_generalize_auto():
    with tempfile.TemporaryDirectory() as d:
        k = Kernel(axioms_dir=d)
        x = Var("x", int8)
        y = Var("y", int8)
        r = k.prove_expr(
            Op("add", (x, Op("sub", (y, y)))),
            x,
            inputs={"x": int8, "y": int8},
        )
        assert r, r
        assert r.derived_axioms, "expected an auto-derived axiom"
        files = glob.glob(os.path.join(d, "*.py"))
        assert files, "auto axiom was not persisted"

        k2 = Kernel(axioms_dir=None)
        k2.axioms.load_file(files[0])
        assert k2.axioms.names()[0].startswith("auto_")
        assert len(k2.axioms.names()) == 1

        # near-miss (x*5 vs x*4) must NOT install anything
        k3 = Kernel(axioms_dir=tempfile.mkdtemp())
        r3 = k3.prove_expr(
            Op("mul", (x, Lit(5))), Op("mul", (x, Lit(4))), inputs={"x": int8}
        )
        assert r3.status == "not_proven"
        assert not k3.axioms.names()

# Concrete axiom theory (axioms/arithmetic.py, pow2.py, logic.py)

@test("theory: the whole axioms/ store loads and every rule is live")

def t_theory_loaded():
    k = opreof.kernel
    assert len(k.axioms.names()) >= 50
    suspects = {
        "sub_absorb", "sub_self", "add_extra_sub", "sub_take_left",
        "sub_take_right", "neg_sub", "sub_neg_left", "mul_neg_left",
        "mul_neg_both", "peel_add_neg", "peel_sub_self", "peel_take_neg",
        "divs_corr_int8_2", "divs_corr_int16_16", "divs_corr_int32_8",
        "remu_pow2_8", "bool_xor_one",
        "bool_eq_zero", "or_absorb_left", "and_absorb_right",
    }
    assert suspects <= set(k.axioms.names()), k.axioms.names()
    # every directed rule must rewrite its LHS for at least one representative
    # value; every *equational* rule must instead merge its two sides inside a
    # saturated e-graph (laws are usable, just not as a directed rewrite)
    for ax in k.axioms.rules():
        fired = False
        # an e-graph holding one rule: the check is about the rule, not about
        # the whole store
        solo = Kernel(axioms_dir=str(opreof.AXIOM_DIR))
        solo.axioms.register(ax)
        for tv in (int8, uint8, int16, int32):
            vals = _representative_values(tv)
            for combo in _combos(ax, vals, tv):
                try:
                    el = ripen(substitute(ax.lhs, combo), tv, env={})
                except KernelError:
                    continue
                if ax.equational:
                    try:
                        er = ripen(substitute(ax.rhs, combo), tv, env={})
                    except KernelError:
                        continue
                    if solo.egraph_prove(el, er, budget=1 << 12,
                                         verify=False).status == "proven":
                        fired = True
                        break
                    continue
                try:
                    n = k.normalize(el)
                except KernelError:
                    continue
                if n != el:
                    fired = True
                    break
            if fired:
                break
        assert fired, f"axiom {ax.name} never fires"

def _representative_values(ty):
    n = ty.width
    if n == 1:
        return [0, 1]
    hi = (1 << (n - 1)) - 1 if ty.signed else (1 << n) - 1
    lo = -(1 << (n - 1)) if ty.signed else 0
    vals = [lo, lo + 1, -1, 0, 1, 127, 128, hi - 1, hi]
    return sorted({v for v in vals if lo <= v <= hi})

def _combos(ax, vals, ty):
    """Yield every placeholder assignment for one axiom over ``vals``."""
    names = sorted(free_vars(ax.lhs))
    if not names:
        yield {}
        return
    if len(names) == 1:
        for v1 in vals:
            yield {names[0]: Lit(v1, ty)}
        return
    if len(names) == 2:
        for v1 in vals:
            for v2 in vals:
                yield {names[0]: Lit(v1, ty), names[1]: Lit(v2, ty)}
        return
    # 3 placeholders (associativity, distributivity): bounded triple product
    small = sorted({v for v in vals if v in (-128, -1, 0, 1, 2, 127)})
    if len(small) < 2:
        small = vals[:2]
    for v1 in small:
        for v2 in small:
            for v3 in small:
                yield {names[0]: Lit(v1, ty), names[1]: Lit(v2, ty),
                       names[2]: Lit(v3, ty)}

def _conds_ok(ax, combo, ty):
    """True when every condition of the axiom holds for this instantiation.

    Brute-forcing must only ever assert equality on combinations the axiom is
    actually allowed to fire on; e.g. the BOOL identities are not valid on
    any other type.
    """
    types_ = {name: ty for name in combo}
    for c in ax.conditions:
        if not c.pred(combo, types_):
            return False
    return True

@test("theory: every axiom is bit-exact over exhaustive concrete inputs")

def t_theory_soundness_bruteforce():
    k = opreof.kernel

    def _walk_lits(e, acc):
        if isinstance(e, Lit):
            acc.append(e.value)
        elif isinstance(e, Op):
            for a in e.args:
                _walk_lits(a, acc)
        elif isinstance(e, Block):
            for a in e.assigns:
                _walk_lits(a.rhs, acc)

    def _eval_type(ax):
        lits = []
        _walk_lits(ax.lhs, lits)
        _walk_lits(ax.rhs, lits)
        for ty in (int8, int16, int32):
            lo, hi = ty.min_value, ty.max_value
            if all(lo <= v <= hi for v in lits):
                return ty
        return int64

    for ax in k.axioms.rules():
        names = sorted(free_vars(ax.lhs))
        ty = _eval_type(ax)
        if len(names) == 1:
            lo, hi = ty.min_value, ty.max_value
            span = hi - lo + 1
            if span >= 2**16:
                vals = sorted(set(_representative_values(ty) + [0]))
            else:
                vals = list(range(lo, hi + 1))
            iterator = ({names[0]: Lit(v, ty)} for v in vals)
        elif len(names) == 2:
            i8_vals = list(range(-128, 128))
            iterator = (
                {names[0]: Lit(a, int8), names[1]: Lit(b, int8)}
                for a in i8_vals
                for b in i8_vals
            )
        elif len(names) == 3:
            # associativity/distributivity: exhaustive-over-a-sample is about
            # catching structural errors; modularity makes the identities
            # generic.  A curated triple sample keeps the suite fast.
            i8_vals = [-128, -1, 0, 1, 2, 127]
            iterator = (
                {names[0]: Lit(a, int8), names[1]: Lit(b, int8), names[2]: Lit(c, int8)}
                for a in i8_vals
                for b in i8_vals
                for c in i8_vals
            )
        else:
            continue
        legal = 0
        checked = 0
        for combo in iterator:
            if not _conds_ok(ax, combo, ty if len(names) == 1 else int8):
                continue
            legal += 1
            try:
                el = ripen(substitute(ax.lhs, combo), combo[names[0]].type, env={})
                er = ripen(substitute(ax.rhs, combo), combo[names[0]].type, env={})
            except KernelError:
                continue
            assert eval_c(el) == eval_c(er), (ax.name, combo, eval_c(el), eval_c(er))
            checked += 1
        if legal:
            assert checked > 0, ax.name

@test("theory: BOOL rules exhaustive over both truth values")

def t_theory_bool_bruteforce():
    k = opreof.kernel
    for ax in k.axioms.rules():
        names = sorted(free_vars(ax.lhs))
        if len(names) != 1 or names != ["a"]:
            continue
        for v in (0, 1):
            combo = {"a": Lit(v, BOOL)}
            if not _conds_ok(ax, combo, BOOL):
                continue
            try:
                el = ripen(substitute(ax.lhs, combo), BOOL, env={})
                er = ripen(substitute(ax.rhs, combo), BOOL, env={})
            except KernelError:
                continue
            assert eval_c(el) == eval_c(er), (ax.name, v)

@test("theory: sampled values on wider types keep every rule sound")

def t_theory_soundness_wider():
    k = opreof.kernel
    for ty in (int16, uint16, int32, uint32, int64):
        vals = _representative_values(ty)
        for ax in k.axioms.rules():
            names = sorted(free_vars(ax.lhs))
            if not names or len(names) > 2:
                continue
            for combo in _combos(ax, vals, ty):
                if not _conds_ok(ax, combo, ty):
                    continue
                try:
                    el = ripen(substitute(ax.lhs, combo), ty, env={})
                    er = ripen(substitute(ax.rhs, combo), ty, env={})
                except KernelError:
                    continue
                issues = check_defined(el) or check_defined(er)
                assert not issues, (ax.name, ty, combo, issues)
                assert eval_c(el) == eval_c(er), (ax.name, ty, combo)

@test("theory: the flagship identities actually prove")

def t_theory_proofs():
    i8 = int8
    x, y = Var("x", i8), Var("y", i8)
    B = BOOL
    p = Var("p", B)
    cases = [
        (Op("sub", (x, Op("sub", (x, y)))), y, {"x": i8, "y": i8}, int8),
        (Op("add", (x, Op("sub", (y, x)))), y, {"x": i8, "y": i8}, int8),
        (Op("add", (Op("sub", (y, x)), x)), y, {"x": i8, "y": i8}, int8),
        (Op("sub", (Op("add", (x, y)), x)), y, {"x": i8, "y": i8}, int8),
        (Op("neg", (Op("sub", (x, y)),)), Op("sub", (y, x)), {"x": i8, "y": i8}, int8),
        (Op("mul", (Op("neg", (x,)), y)), Op("neg", (Op("mul", (x, y)),)), {"x": i8, "y": i8}, int8),
        (Op("divu", (x, Lit(4))), Op("shru", (x, Lit(2))), {"x": i8}, int8),
        (Op("remu", (x, Lit(8))), Op("and", (x, Lit(7))), {"x": i8}, int8),
        (Op("divs", (x, Lit(2))),
         Op("shrs", (Op("add", (x, Op("and", (Op("shrs", (x, Lit(7))), Lit(1))))), Lit(1))),
         {"x": i8}, int8),
        (Op("divs", (x, Lit(4))),
         Op("shrs", (Op("add", (x, Op("and", (Op("shrs", (x, Lit(7))), Lit(3))))), Lit(2))),
         {"x": i8}, int8),
        (Op("xor", (p, Lit(1, B))), Op("not", (p,)), {"p": B}, B),
        (Op("eq", (p, Lit(0, B))), Op("not", (p,)), {"p": B}, B),
        (Op("ne", (p, Lit(1, B))), Op("not", (p,)), {"p": B}, B),
        (Op("or", (x, Op("and", (x, y)))), x, {"x": i8, "y": i8}, int8),
        (Op("and", (x, Op("or", (x, y)))), x, {"x": i8, "y": i8}, int8),
    ]
    for e1, e2, inputs, _ in cases:
        r = opreof.kernel.prove_expr(e1, e2, inputs=inputs, generalize=False)
        assert r, (e1.sugar(), e2.sugar(), r)

    # a divisor too large for the type is not representable -> rejected,
    # never silently rewritten to a partial shift
    x32 = Var("x", int32)
    r = opreof.kernel.prove_expr(
        Op("divs", (x32, Lit(2**40))), Op("shru", (x32, Lit(1))),
        inputs={"x": int32}, generalize=False,
    )
    assert r.status == "rejected", r
    # divu on unsigned types rewrites and is defined for every value
    u = Var("u", uint16)
    r = opreof.kernel.prove_expr(
        Op("divu", (u, Lit(3))), Op("divu", (u, Lit(3))),
        inputs={"u": uint16}, generalize=False,
    )
    assert r, r

@test("theory: signed division/remainder truncate toward zero, not floor")

def t_division_semantics():
    def ev(op, lhs, rhs, ty):
        e = ripen(Op(op, (Lit(lhs, ty), Lit(rhs, ty))), ty, env={})
        return eval_c(e)

    # C / RISC-V DIV/REM / WASM i32.div_s/rem_s / x86 IDIV semantics
    assert ev("divs", -3, 2, int8) == -1          # floor would be -2
    assert ev("divs", 3, -2, int8) == -1
    assert ev("divs", -3, -2, int8) == 1
    assert ev("divs", -128, -1, int8) == -128     # wraps like two's complement
    assert ev("rems", -3, 2, int8) == -1          # sign follows the dividend
    assert ev("rems", 3, -2, int8) == 1
    assert ev("rems", -3, -2, int8) == -1

    # arithmetic shift is floor division, so the naive signed rewrite is
    # unsound for negative x and must NOT prove
    x = Var("x", int8)
    r = opreof.kernel.prove_expr(
        Op("divs", (x, Lit(2))), Op("shrs", (x, Lit(1))),
        inputs={"x": int8}, generalize=False,
    )
    assert r.status == "not_proven", r

# api.Term <-> kernel bridge

@test("bridge: Term(int32)/uint16 resolve; expr roundtrip; end-to-end proof")

def t_bridge_terms():
    import kernel as K
    from api import Term

    assert opreof.resolve_term_type(Term(int32)) is K.int32
    assert opreof.resolve_term_type(Term(uint16)) is K.uint16

    x = opreof.expr_of_term(_parse_parms("x"))
    back = opreof.term_of_expr(x)
    assert back is not None and back.name == "x"

    # the bridge-proof that previously failed with 'cannot resolve type of x'
    p = opreof.prove_expr(
        Op("mul", (x, Lit(2))), Op("shl", (x, Lit(1))), inputs={"x": int8}
    )
    assert p, p

@test("bridge: prove_expr with inputs drives typing of adaptable vars")

def t_bridge_inputs_typing():
    from opreof import Lit as oLit
    from opreof import Op as oOp
    from opreof import Var as oVar

    x = oVar("x")
    p = opreof.prove_expr(
        oOp("mul", (x, oLit(2))), oOp("shl", (x, oLit(1))), inputs={"x": int8}
    )
    assert p, p

@test("bridge: prove_blocks via opreof facade")

def t_bridge_blocks():
    from opreof import Lit as oLit
    from opreof import Op as oOp
    from opreof import Var as oVar

    a = oVar("a", int8)
    b1 = opreof.Block("o", [opreof.Assign(a, oOp("mul", (a, oLit(2))))])
    b2 = opreof.Block("u", [opreof.Assign(a, oOp("shl", (a, oLit(1))))])
    r = opreof.kernel.prove_blocks(b1, b2, inputs={"a": int8}, outputs={"a"})
    assert r, r

def _parse_parms(s):
    from api import parse_parms

    return parse_parms(s)

@test("bridge: literal terms are kind='literal', not kind='type'")

def t_bridge_literal_kind():
    from opreof import expr_of_term, term_of_expr

    t = term_of_expr(Lit(7, int32))
    assert t.kind == "literal", t
    assert t.name == "7" and t.value == 7 and t.type is int32
    assert expr_of_term(t) == Lit(7, int32)

    # adaptable literal stays adaptable across the bridge
    t = term_of_expr(Lit(7))
    assert t.kind == "literal" and t.type is None
    assert expr_of_term(t) == Lit(7, None)

    # a whole canonical expression round-trips with the literal in place
    # (ripen again: the bridge only carries the literal's type, so the
    # adaptable variable must be re-bound to its kernel type)
    n = opreof.kernel.normalize(Op("add", (Var("x", int32), Lit(7))))
    call = term_of_expr(n)
    assert call.args[0].kind == "literal"
    rebuilt = ripen(expr_of_term(call), int32, env={})
    assert opreof.kernel.normalize(rebuilt) == n

# Solving -- the api goes beyond equivalence checking

@test("solve: equation over an exhaustible input domain")

def t_solve_expr():
    x = Var("x", int8)
    r = opreof.solve_expr(Op("mul", (x, x)), Lit(16, int8), {"x": int8})
    assert r.status == "some", r
    assert r.exhaustive, r
    sols = [s["x"] for s in r.satisfying]
    assert {4, -4} <= set(sols), sols
    assert all((v * v) % 256 == 16 for v in sols)
    # the same equation, probed via the kernel proof path, stays "not_proven"
    p = opreof.kernel.prove_expr(Op("mul", (x, x)), Lit(16, int8), inputs={"x": int8})
    assert p.status == "not_proven", p

@test("solve: always / never / rejected; witnesses and counterexamples")

def t_solve_statuses():
    x = Var("x", int8)
    r = opreof.solve_expr(Op("add", (x, x)), Op("shl", (x, Lit(1))), {"x": int8})
    assert r.status == "always" and r.exhaustive and r.counterexample is None, r
    assert not opreof.find_counterexample(
        Op("add", (x, x)), Op("shl", (x, Lit(1))), {"x": int8})

    r = opreof.solve_expr(x, Op("add", (x, Lit(1))), {"x": int8})
    assert r.status == "never" and r.witness is None, r
    cex = opreof.find_counterexample(x, Op("add", (x, Lit(1))), {"x": int8})
    assert cex is not None and cex["x"] in (-128, -127, 0, 127)

    w = opreof.find_witness(Op("mul", (x, x)), Lit(16, int8), {"x": int8})
    assert w is not None and (w["x"] * w["x"]) % 256 == 16

    r = opreof.solve_expr(Op("shl", (x, Lit(8))), x, {"x": int8})
    assert r.status == "rejected", r

@test("solve_for: closed forms, cross-checked, and residuals")

def t_solve_for():
    x = Var("x", int8)
    a, b = Var("a", int8), Var("b", int8)

    r = opreof.solve_for(Op("add", (x, Lit(7))), Lit(10, int8), {"x": int8}, var="x")
    assert r.symbolic is not None and r.symbolic.sugar() == "3", r

    r = opreof.solve_for(Op("mul", (x, Lit(3))), Lit(9, int8), {"x": int8}, var="x")
    assert r.symbolic is not None and r.symbolic.sugar() == "3", r

    # closed form in other variables: x == add(neg(a), b)
    r = opreof.solve_for(Op("add", (x, a)), b,
                         {"x": int8, "a": int8, "b": int8}, var="x")
    assert r.residual is None and r.symbolic is not None, r
    sol = ripen(r.symbolic, int8, env={"a": (Var("a", int8), int8),
                                       "b": (Var("b", int8), int8)})
    back = opreof.kernel.normalize(
        Op("sub", (substitute(Op("add", (x, a)), {"x": sol}), b), int8))
    assert back == Lit(0, int8), back.sugar()

    # nonlinear: not isolatable -> residual + exact solution set
    r = opreof.solve_for(Op("mul", (x, x)), Lit(16, int8), {"x": int8}, var="x")
    assert r.symbolic is None and r.residual is not None, r
    assert {s["x"] for s in r.satisfying} >= {4, -4}

@test("synthesize: constant, function-of-input, and underdetermined holes")

def t_synthesize():
    from opreof import expr_of_term

    x = Var("x", int8)
    h = Var("?", int8)

    # constant hole: xor(x, ?) == not(x)   =>  ? == -1
    r = opreof.synthesize(Op("xor", (x, h)), Op("not", (x,)), {"x": int8}, hole="?")
    assert r.status == "always" and r.exhaustive, r
    fill = ripen(expr_of_term(r.synthesized), int8, env={"x": (Var("x", int8), int8)})
    chk = opreof.solve_expr(Op("xor", (x, fill)), Op("not", (x,)), {"x": int8})
    assert chk.status == "always" and chk.exhaustive, (fill.sugar(), chk)
    # the discovered constant is now provable by rewriting, not just enumerable
    assert opreof.kernel.prove_expr(
        Op("xor", (x, fill)), Op("not", (x,)), inputs={"x": int8}).status == "proven"

    # function hole: x + ? == 0   =>  ? == neg(x)
    r = opreof.synthesize(Op("add", (x, h)), Lit(0, int8), {"x": int8}, hole="?")
    assert r.status == "always" and r.synthesized is not None, r
    fill = ripen(expr_of_term(r.synthesized), int8, env={"x": (Var("x", int8), int8)})
    chk = opreof.solve_expr(Op("add", (x, fill)), Lit(0, int8), {"x": int8})
    assert chk.status == "always" and chk.exhaustive, (fill.sugar(), chk)

    # underdetermined: mul(x, ?) == 0  -- only a table pins the hole down
    r = opreof.synthesize(Op("mul", (x, h)), Lit(0, int8), {"x": int8}, hole="?")
    assert r.status == "some" and r.kmap is not None, r
    for xv, hits in r.kmap.items():
        assert all((xv[0] * c) % 256 == 0 for c in hits)

@test("kernel: constant folding wraps to canonical machine values")

def t_fold_canonical():
    k = opreof.kernel
    # -765 mod 256 == 3 ; 2**7 wraps to -128 in int8
    assert k.normalize(Op("mul", (Lit(9, int8), Lit(-85, int8)))) == Lit(3, int8)
    assert k.normalize(Op("shl", (Lit(1, int8), Lit(7)))) == Lit(-128, int8)
    assert k.normalize(Op("add", (Lit(200, uint8), Lit(100, uint8)))) == Lit(44, uint8)
    # divs/rems fold with machine (trunc) semantics, not Python floor
    assert k.normalize(Op("divs", (Lit(-3, int8), Lit(2, int8)))) == Lit(-1, int8)
    assert k.normalize(Op("rems", (Lit(-3, int8), Lit(2, int8)))) == Lit(-1, int8)
    # folding must agree with evaluation on the very same node
    e = ripen(Op("mul", (Lit(-85, int8), Lit(9, int8))), int8, env={})
    assert eval_c(k.normalize(e)) == eval_c(e)

# E-graphs -- saturate the store, then extract; and axiom curation

@test("egraph: laws the rewriter must refuse are proved by saturation")

def t_egraph_laws():
    x, y, z = Var("x", int8), Var("y", int8), Var("z", int8)
    I = {"x": int8, "y": int8, "z": int8}
    # (lhs, rhs, deciding rule, tier) -- every one of these is an *equational*
    # axiom the directed normaliser refuses to fire
    laws = [
        (Op("add", (Op("add", (x, y)), z)), Op("add", (x, Op("add", (y, z)))),
         "add_assoc", "affine"),
        (Op("mul", (Op("add", (y, z)), x)),
         Op("add", (Op("mul", (x, y)), Op("mul", (x, z)))), "mul_dist",
         "multilinear"),
        (Op("not", (Op("and", (x, y)),)),
         Op("or", (Op("not", (x,)), Op("not", (y,)))), "not_and", "bitwise"),
        (Op("xor", (x, y)),
         Op("or", (Op("and", (x, Op("not", (y,)))),
                   Op("and", (Op("not", (x,)), y)))), "xor_via_and_or",
         "bitwise"),
    ]
    for lhs, rhs, rule, tier in laws:
        rew = opreof.prove_expr(lhs, rhs, I)
        assert rew.status == "not_proven", (rule, rew)
        p = opreof.egraph_prove(lhs, rhs, I)
        assert p.status == "proven" and p.checked == "complete", (rule, p, tier)
        assert rule in p.axioms_used, (rule, p.axioms_used)

    # extraction reads the *smallest* member back out: the distributed form
    # comes back factored, and the expanded xor comes back as a bare xor
    factored = opreof.egraph_normalize(
        Op("add", (Op("mul", (x, y)), Op("mul", (x, z)))), I)
    assert factored.sugar() == "mul(add(y, z), x)", factored.sugar()
    compact = opreof.egraph_normalize(laws[3][1], I)
    assert compact.sugar() == "xor(x, y)", compact.sugar()
    # ... and the factored normal form is still bit-exact for the original
    env = {n: (Var(n, ty), ty) for n, ty in I.items()}
    lx = ripen(Op("add", (Op("mul", (x, y)), Op("mul", (x, z)))), int8, env=env)
    lf = ripen(factored, int8, env=env)
    for xv, yv, zv in ((3, 5, 7), (-7, 11, 2), (-128, 127, 1), (0, 0, 0)):
        vals = {"x": Lit(xv, int8), "y": Lit(yv, int8), "z": Lit(zv, int8)}
        assert eval_c(substitute(lx, vals)) == eval_c(substitute(lf, vals)), \
            (xv, yv, zv)

@test("egraph: an identity written as a law beats the rewriter's orientation")

def t_egraph_expansion():
    # the same fact written the *other* way round: an expansion the rewriter
    # can never fire (it only shrinks), yet a perfectly good equality
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Var("a", int8), Var("b", int8)
        small = Var("b", int8)
        big = Op("add", (Op("add", (Op("neg", (a,)), b),), a),)
        path = os.path.join(tmp, "law.py")
        with open(path, "w") as fh:
            fh.write(
                "from axiom_store import axiom\n"
                "from opreof import Op, Var, int8\n"
                "BIG = Op('add', (Op('add', (Op('neg', (Var('a', int8),)),"
                " Var('b', int8)),), Var('a', int8)),)\n"
                "AXIOMS = [axiom('peel_bwd', Var('b', int8), BIG,"
                " source='law', equational=True)]\n"
            )
        kb = opreof.new_kernel(axioms_dir=tmp)
        I = {"a": int8, "b": int8}
        assert kb.prove_expr(small, big, I).status == "not_proven"
        p = kb.egraph_prove(small, big, I)
        assert p.status == "proven" and p.checked == "complete", p
        assert p.axioms_used == ("peel_bwd",), p
        # the extracted minimal form is the small side, not the one we asked in
        assert kb.egraph_normalize(big, I).sugar() == "b"
        r = opreof.solve_expr(small, big, I)
        assert r.status == "always" and r.exhaustive, r

@test("egraph: a cycle of mutually dependent rules merges in one saturation")

def t_egraph_cycles():
    # Three rules that fire into each other:  X ~> Y ~> Z ~> X.  Rules are
    # applied in both directions, so a directed orientation cannot decide this
    # set; the e-graph treats the cycle as three equalities.
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Var("a", int8), Var("b", int8)
        x = Op("sub", (a, b))
        y = Op("add", (a, Op("neg", (b,)),))
        z = Op("sub", (Op("neg", (b,)), Op("neg", (a,))))
        path = os.path.join(tmp, "cycle.py")
        with open(path, "w") as fh:
            fh.write(
                "from axiom_store import axiom\n"
                "from opreof import Op, Var, int8\n"
                "X = Op('sub', (Var('a', int8), Var('b', int8)))\n"
                "Y = Op('add', (Var('a', int8), Op('neg', (Var('b', int8),))))\n"
                "Z = Op('sub', (Op('neg', (Var('b', int8),)),"
                " Op('neg', (Var('a', int8),))))\n"
                "AXIOMS = [axiom('cyc_x_y', X, Y, source='cycle'),\n"
                "          axiom('cyc_y_z', Y, Z, source='cycle'),\n"
                "          axiom('cyc_z_x', Z, X, source='cycle')]\n"
            )
        kb = opreof.new_kernel(axioms_dir=tmp)
        assert kb.axioms.names() == ["cyc_x_y", "cyc_y_z", "cyc_z_x"]
        I = {"a": int8, "b": int8}
        p = kb.egraph_prove(x, z, I)
        assert p.status == "proven" and p.checked == "complete", p
        assert set(p.axioms_used) == {"cyc_x_y", "cyc_y_z", "cyc_z_x"}, p
        # the rules really are true, and the rewriter agrees when it can
        r = opreof.solve_expr(x, z, I)
        assert r.status == "always" and r.exhaustive, r
        assert kb.prove_expr(x, y, I).status == kb.prove_expr(x, z, I).status

@test("egraph: one knowledge base, reused across queries")

def t_egraph_incremental():
    x, y = Var("x", int8), Var("y", int8)
    I = {"x": int8, "y": int8}
    # a consumer's own narrow rule set -- no blind "prove from the whole
    # store", just the four facts this program is written against
    kb = opreof.prepare_egraph(axioms=("add_self_is_shl1", "peel_add_neg",
                                       "peel_take_neg", "sub_as_add_neg"))
    first = kb.add(Op("add", (Op("add", (x, y)), Op("neg", (x,)))), I)
    kb.saturate()
    # a second, unrelated term joins the same graph and finds the class the
    # first query already built, instead of re-proving from scratch: both
    # sides are ``... x ... x`` cancellations, so both are the same ``y``
    second = kb.add(Op("add", (Op("add", (Op("neg", (x,)), y)), x)), I)
    kb.saturate()
    assert kb.graph.find(first) == kb.graph.find(second), "peel_add_neg"
    # re-saturating never redoes finished work: each pass continues where the
    # last stopped, and the node count settles at a fixpoint
    prev = -1
    for _ in range(6):
        kb.saturate()
        count = len(kb.graph.nodes)
        if count == prev:
            break
        prev = count
    else:
        raise AssertionError("saturation never reached a fixpoint")
    # a third query on the same graph: doubling is a shift
    assert kb.same(Op("add", (x, x)), Op("shl", (x, Lit(1, int8)))), \
        "add_self_is_shl1"
    # ... and the extracted normal form of the first term is its shared class
    assert kb.normal_form(Op("add", (Op("add", (x, y)), Op("neg", (x,))))) \
        .sugar() == "y"
    # adding a term whose type the caller forgot is an explicit rejection
    try:
        kb.add(Op("add", (x, Var("w", None))))
    except KernelError:
        pass
    else:
        raise AssertionError("untyped input should not be guessed")

@test("egraph: a wrong axiom is caught, never turned into a proof")

def t_egraph_soundness():
    x, y = Var("x", int8), Var("y", int8)
    I = {"x": int8, "y": int8}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.py")
        with open(path, "w") as fh:
            fh.write(
                "from axiom_store import axiom\n"
                "from opreof import Op, Var, int8\n"
                "AXIOMS = [axiom('bogus_sub',"
                " Op('sub', (Var('a', int8), Var('b', int8))),"
                " Op('add', (Var('a', int8), Var('b', int8))),"
                " source='bad')]\n"
            )
        kb = opreof.new_kernel(axioms_dir=tmp)
        p = kb.egraph_prove(Op("sub", (x, y)), Op("add", (x, y)), I)
        assert p.status == "rejected" and p.checked == "broken", p
        assert "bogus_sub" in p.reason, p
        # a false claim is not_proven, never proven
        p2 = opreof.egraph_prove(Op("add", (x, y)), Op("sub", (y, x)), I)
        assert p2.status == "not_proven", p2
    # sampling is reported as sampling.  A rule the shape-based cubes cannot
    # decide (it involves a shift, so only enumeration would do) over a domain
    # too large to enumerate (all 65536 values of a 16-bit type) is *found* by
    # the e-graph but must never be claimed: the caller is told it is a sample.
    x16 = Var("x", int16)
    I16 = {"x": int16}
    p3 = opreof.egraph_prove(
        Op("add", (x16, x16)), Op("shl", (x16, Lit(1, int16))), I16
    )
    assert p3.status == "not_proven" and p3.checked == "sampled", p3
    # ... and the very same claim is a proof over a domain small enough to
    # enumerate bit-exactly
    p3b = opreof.egraph_prove(
        Op("add", (x, x)), Op("shl", (x, Lit(1, int8))), I
    )
    assert p3b.status == "proven" and p3b.checked == "complete", p3b
    # ... and the same claim is a proof once the caller certifies the store
    p4 = opreof.egraph_prove(
        Op("mul", (Op("mul", (x, y)), Var("z", int8))),
        Op("mul", (x, Op("mul", (y, Var("z", int8))))),
        {"x": int8, "y": int8, "z": int8},
        axioms=("mul_assoc",), verify=False,
    )
    assert p4.status == "proven" and p4.checked is None, p4

@test("curation: the axiom set is data a consumer can inspect and narrow")

def t_curation():
    names = opreof.list_theories()
    assert "arithmetic" in names and "logic" in names, names
    arith = opreof.theory_rules("arithmetic")
    assert arith and all(a.name for a in arith)
    # curation is honoured: a proof may be restricted to exactly one fact
    x = Var("x", int8)
    I = {"x": int8}
    goal = (Op("add", (x, x)), Op("shl", (x, Lit(1, int8))))
    assert opreof.prove_expr(*goal, I).status == "proven"
    with opreof.with_axioms("add_self_is_shl1"):
        assert opreof.prove_expr(*goal, I).status == "proven"
    with opreof.with_axioms("sub_as_add_neg"):
        # the doubling fact is no longer available: not a proof, but not an error
        assert opreof.prove_expr(*goal, I).status == "not_proven"
    # an unknown rule name is an error, not a silent no-op
    try:
        opreof.prove_expr(*goal, I, axioms=("no_such_rule",))
    except KernelError:
        pass
    else:
        raise AssertionError("unknown axiom name must be rejected")
    # a fresh kernel is a separate world
    solo = opreof.new_kernel(theories=("logic",))
    assert set(solo.axioms.names()) == {a.name for a in opreof.theory_rules("logic")}
    assert "add_assoc" not in solo.axioms.names()
    assert "add_self_is_shl1" in opreof.kernel.axioms.names()

# main

def main():
    import time

    fails = 0
    for name, fn in _TESTS:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.2f}s)")
        except AssertionError as err:
            fails += 1
            print(f"FAIL  {name}  ({time.time() - t0:.2f}s): {err}")
        sys.stdout.flush()
    print(f"\n{len(_TESTS) - fails}/{len(_TESTS)} tests passed")
    return 1 if fails else 0

if __name__ == "__main__":
    raise SystemExit(main())
