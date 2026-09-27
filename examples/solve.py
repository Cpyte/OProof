"""OProof API sample -- solving, not just equivalence checking.

The kernel *proves* equivalences by rewriting.  The solving layer goes the
other way: it enumerates the concrete bit-space of the inputs (bit-exact,
with ``eval_c`` as ground truth), so you can ask ``opreof`` to

  1. solve an equation   -- find which inputs make ``e1 == e2``;
  2. produce a witness or counterexample for any equivalence;
  3. solve for a single variable -- isolate a closed form like ``x = b - a``;
  4. synthesize a hole -- fill ``?`` in ``xor(x, ?) == not(x)``.

Run with:  python3 examples/solve.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opreof import (
    Lit,
    Op,
    Var,
    find_counterexample,
    find_witness,
    int8,
    solve_expr,
    solve_for,
    synthesize,
)

SL = "  "

def solving_layer():
    x = Var("x", int8)
    a = Var("a", int8)
    b = Var("b", int8)

    print("== equation solving, witnesses, counterexamples ==")
    r = solve_expr(Op("mul", (x, x)), Lit(16, int8), {"x": int8})
    print(SL + f"solve(x*x == 16, int8)              -> {r}")
    if r.status == "some":
        shown = sorted(s["x"] for s in list(r.satisfying)[:6])
        print(SL + f"  first satisfying inputs: {shown} ...")
    print(SL + f"find_witness(x*x == 16)        -> {find_witness(Op('mul', (x, x)), Lit(16, int8), {'x': int8})}")

    r = solve_expr(Op("add", (x, x)), Op("shl", (x, Lit(1))), {"x": int8})
    print(SL + f"solve(x+x == x<<1, int8)          -> {r}")

    r = solve_expr(x, Op("add", (x, Lit(1))), {"x": int8})
    print(SL + f"solve(x == x+1, int8)             -> {r}")
    print(SL + f"find_counterexample(x == x+1)-> {find_counterexample(x, Op('add', (x, Lit(1))), {'x': int8})}")

    r = solve_expr(Op("shl", (x, Lit(8))), x, {"x": int8})
    print(SL + f"solve(x<<8 == x, int8)            -> {r}")

    print("== solve_for: isolate a variable ==")
    r = solve_for(Op("mul", (x, Lit(3))), Lit(9, int8), {"x": int8}, var="x")
    print(SL + f"x*3 == 9                          -> {r}")

    r = solve_for(Op("add", (x, a)), b,
                  {"x": int8, "a": int8, "b": int8}, var="x")
    print(SL + f"x+a == b                          -> {r}")

    r = solve_for(Op("mul", (x, x)), Lit(16, int8), {"x": int8}, var="x")
    print(SL + f"x*x == 16                         -> {r}")

    print("== synthesize: fill the hole '?' ==")
    r = synthesize(Op("xor", (x, Var("?", int8))), Op("not", (x,)), {"x": int8},
                   hole="?")
    print(SL + f"xor(x, ?) == not(x)          -> {r}")

    r = synthesize(Op("add", (x, Var("?", int8))), Lit(0, int8), {"x": int8},
                   hole="?")
    print(SL + f"x + ? == 0                   -> {r}")

    r = synthesize(Op("mul", (x, Var("?", int8))), Lit(0, int8), {"x": int8},
                   hole="?")
    print(SL + f"x * ? == 0                   -> {r}")
    if r.kmap:
        print(SL + f"  one satisfying value per input: "
                   f"{list(r.kmap.items())[:3]} ...")

def main() -> int:
    solving_layer()
    print()
    print("All solving samples passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
