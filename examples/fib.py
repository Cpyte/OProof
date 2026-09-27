"""OProof API sample -- a Fibonacci loop, an equivalent program, and tests.

This file has three parts:

  1. ``api_sample()``  -- the api.Term/api.Call layer and the explicit bridge
     (resolve_term_type, expr_of_term, term_of_expr) to the kernel.
  2. fibonacci programs  -- the classic iterative Fibonacci loop written as an
     OProof ``Block`` (Assign/Loop over typed int32 variables), plus an
     equivalent program produced by unrolling it by a factor of two.
  3. tests  -- prove the two Fibonacci forms bit-for-bit equivalent, verify the
     unrolling is faithful, and show that a *wrong* unrolling is rejected.

Run with:  python3 examples/fib.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import Call, Term
from opreof import (
    Assign,
    Block,
    Lit,
    Loop,
    Op,
    Var,
    expr_of_term,
    int32,
    kernel,
    list_axioms,
    prove,
    prove_expr,
    resolve_term_type,
    term_of_expr,
    unroll_verify,
)

# 1. The api sample

def api_sample():
    from api import parse_parms

    print("== api layer and bridge ==")
    # A Term can denote a kernel machine type.  Term(int32) resolves to the
    # kernel's true ``int32`` through the explicit bridge -- the two layers
    # stay separate, this just lets the api *denote* a machine type.
    t32 = Term(int32)
    print("  resolve_term_type(Term(int32))      ->", resolve_term_type(t32))
    print("  parse_parms(int32).name             ->", parse_parms(int32).name)

    # A variable Term + a Call tree translate into kernel Vars and Ops.
    e = expr_of_term(Call(Term("add"), (Term(name="x"), Term(name="y"))))
    print("  expr_of_term(add(x, y))             ->", e.sugar())

    # Expressions come back to the api layer through term_of_expr.  Carrying
    # the kernel's canonical, sorted form: x + 7 normalises to add(7, x).
    # Note a kernel literal surfaces as kind='literal' (a value, not a type).
    n = kernel.normalize(Op("add", (Var("x", int32), Lit(7))))
    back = term_of_expr(n)
    print("  term_of_expr(normalize(x + 7))      ->", back)
    assert back.args[0].kind == "literal"
    # and the round-trip back into the kernel is lossless for the literal
    print("  expr_of_term(term_of_expr(Lit(7, int32))) == Lit(7, int32) ->",
          expr_of_term(term_of_expr(Lit(7, int32))) == Lit(7, int32))

    # the live axiom store (ordinary files in axioms/), auditing every proof
    print(f"  axioms loaded in the running kernel -> {len(list_axioms())}")

# 2. The programs

I = int32  # Fibonacci counters/locals live in 32-bit machine ints.

def _fib_body(commuted: bool = False) -> list:
    """Body B of the loop: { t = a + b; a = b; b = t }.

    ``commuted`` swaps the add operands (``t = b + a``) -- semantically the
    same body, used to prove addition is commutative through the kernel.
    """
    a, b, t = Var("a", I), Var("b", I), Var("t", I)
    return [
        Assign(t, Op("add", (b, a) if commuted else (a, b))),
        Assign(a, b),
        Assign(b, t),
    ]

def fibonacci(n, commuted: bool = False) -> Block:
    """Classic iterative Fibonacci: a = 0, b = 1, repeat n times B."""
    a, b = Var("a", I), Var("b", I)
    return Block(
        "fib",
        [Assign(a, Lit(0)), Assign(b, Lit(1)), Loop(n, _fib_body(commuted))],
    )

def fibonacci_unrolled(n, factor: int) -> Block:
    """Equivalent program: Loop(n//f, B*f) then the remainder tail B[0:r]."""
    a, b = Var("a", I), Var("b", I)
    q, r = divmod(n, factor)
    body = _fib_body()
    return Block(
        "fib_unrolled",
        [Assign(a, Lit(0)), Assign(b, Lit(1)), Loop(q, body * factor)]
        + body[:r],
    )

# 3. Tests

def tests():
    print("\n== programs ==")
    orig = fibonacci(8)
    print("  original:", orig.stmts[-1])
    print("  unrolled:", fibonacci_unrolled(8, 2).stmts[-1])

    print("\n== proofs ==")

    # (a) the commuted-argument Fibonacci is the same program.
    r = prove(fibonacci(8, commuted=True), orig, inputs={}, generalize=False)
    print(
        f"  fib(commuted) == fib(original)        : {r.status} "
        f"(use/add {', '.join(r.axioms_used) or 'none'})"
    )
    assert r, r.reason

    # (b) the unrolled-by-2 Fibonacci is a faithful equivalent.
    # n=8 = 4 iteration-pairs, so the remainder tail is empty (r = 0).
    r = unroll_verify(orig, fibonacci_unrolled(8, 2), factor=2, inputs={})
    print(f"  unroll_verify(fib, fib[:, 2], f=2)    : {r.status} -- {r.reason}")
    assert r, r.reason

    # A loop body of three statements cannot be split as B[0:r] for an odd
    # trip count (B[:1] = "t = a + b" is not a full Fibonacci iteration), so
    # a naive 2x unroll of an odd trip count is NOT a faithful equivalent:
    # the verifier simulates both blocks and refuses it.
    r = unroll_verify(fibonacci(9), fibonacci_unrolled(9, 2), factor=2, inputs={})
    print(
        f"  unroll_verify(fib, fib[:, 2], f=2, n=9): {r.status} "
        f"-- {r.reason}"
    )
    assert r.status == "rejected", r

    # (c) a *wrong* unrolling must be rejected: n=9 needs a remainder tail, and
    # this unrolled program forgets it, dropping one full iteration.
    bad = Block(
        "fib_unrolled_broken",
        [
            Assign(Var("a", I), Lit(0)),
            Assign(Var("b", I), Lit(1)),
            Loop(4, _fib_body() * 2),
        ],
    )
    r = unroll_verify(fibonacci(9), bad, factor=2, inputs={})
    print(f"  unroll_verify(fib, broken, f=2)      : {r.status} -- {r.reason}")
    assert r.status == "rejected", r

    # (d) expression-level: fib(a, b) step is just a+b folded by the kernel.
    r = prove_expr(
        Op("add", (Lit(21, I), Lit(13, I))), Lit(34, I), inputs={}
    )
    print(f"  prove_expr(21 + 13 == 34)             : {r.status}")
    assert r, r.reason

    print("\nAll fib tests passed.")

if __name__ == "__main__":
    api_sample()
    tests()
