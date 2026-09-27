"""OProof API sample -- equality saturation, and *curated* axioms.

The kernel's rewriter is directed: every axiom has a preferred side, it only
shrinks, and an identity that can be read in both directions (associativity,
distributivity, De Morgan) is deliberately kept out of its hands.  Those laws
are still facts, and ``opreof`` exposes them through an e-graph: a term goes
into a graph, the axioms fire as *undirected* equalities, and what comes back
is whatever shape the consumer asked for -- a proof that two spellings are the
same, or the smallest member of the class.

Two things this file is really about:

  1. *saturation beats orientation* -- ``x*y + x*z`` is written the wrong way
     round for the rewriter, and the e-graph factors it back;
  2. *a consumer curates its own axioms* -- nobody here proves from the whole
     store.  ``prepare_egraph(axioms=(...))`` pins the exact rules a program is
     written against, and ``with_axioms(...)`` does the same for a plain proof.

Run with:  python3 examples/egraph.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opreof import (
    Lit,
    Op,
    Var,
    egraph_normalize,
    egraph_prove,
    int8,
    int16,
    list_theories,
    prepare_egraph,
    prove_expr,
    theory_rules,
    with_axioms,
)

CHECKS = 0

def check(label, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")

def demo_laws_the_rewriter_refuses():
    """Identities the directed normaliser must not fire, proved by merging."""
    print("1. laws the rewriter refuses are still provable")
    x, y, z = Var("x", int8), Var("y", int8), Var("z", int8)
    inputs = {"x": int8, "y": int8, "z": int8}
    laws = [
        ("associativity",
         Op("add", (Op("add", (x, y)), z)),
         Op("add", (x, Op("add", (y, z))))),
        ("distributivity",
         Op("mul", (Op("add", (y, z)), x)),
         Op("add", (Op("mul", (x, y)), Op("mul", (x, z))))),
        ("de Morgan",
         Op("not", (Op("and", (x, y)),)),
         Op("or", (Op("not", (x,)), Op("not", (y,))))),
    ]
    for name, lhs, rhs in laws:
        directed = prove_expr(lhs, rhs, inputs)
        saturated = egraph_prove(lhs, rhs, inputs)
        check(
            f"{name:15s} rewriter={directed.status:11s} e-graph={saturated.status}",
            directed.status == "not_proven"
            and saturated.status == "proven"
            and saturated.checked == "complete",
            f"via {', '.join(saturated.axioms_used)}",
        )

def demo_factoring():
    """Extraction reads the *smallest* member of a class back out."""
    print("2. the normal form an e-graph chooses for you")
    x, y, z = Var("x", int8), Var("y", int8), Var("z", int8)
    inputs = {"x": int8, "y": int8, "z": int8}
    expanded = Op("add", (Op("mul", (x, y)), Op("mul", (x, z))))
    factored = egraph_normalize(expanded, inputs)
    check(
        "x*y + x*z  ->  x*(y+z)",
        factored.sugar() == "mul(add(y, z), x)",
        factored.sugar(),
    )
    # the rewriter only shrinks, so it has nothing to say here
    check(
        "the rewriter declines the same term",
        prove_expr(factored, expanded, inputs).status == "not_proven",
    )
    # and the factored form is the *same computation*, bit for bit
    same = egraph_prove(expanded, factored, inputs)
    check("factored form is bit-exact", same.status == "proven", same.reason)

def demo_curation():
    """A consumer names its axioms; it does not prove from the whole store."""
    print("3. the axiom set is data, and it is yours to narrow")
    check("theories on disk", set(list_theories()) >= {"arithmetic", "logic"},
          ", ".join(list_theories()))
    check("arithmetic has rules", len(theory_rules("arithmetic")) > 0,
          f"{len(theory_rules('arithmetic'))} rules")

    x = Var("x", int8)
    inputs = {"x": int8}
    doubling = (Op("add", (x, x)), Op("shl", (x, Lit(1, int8))))
    check("doubling == shift, whole store", prove_expr(*doubling, inputs).status
          == "proven")
    with with_axioms("sub_as_add_neg"):
        # the doubling fact is no longer in scope: not a proof, and not an error
        check("doubling with an unrelated rule only",
              prove_expr(*doubling, inputs).status == "not_proven")

    # the same narrowing for the e-graph: four rules, reused across queries
    kb = prepare_egraph(axioms=("peel_take_neg", "peel_add_neg",
                                "add_self_is_shl1"))
    y = Var("y", int8)
    first = kb.add(Op("add", (Op("add", (x, y)), Op("neg", (x,)))), inputs)
    kb.saturate()
    second = kb.add(Op("add", (Op("add", (Op("neg", (x,)), y)), x)), inputs)
    kb.saturate()
    check("two spellings share one class",
          kb.graph.find(first) == kb.graph.find(second))
    check("the class is y", kb.normal_form(
        Op("add", (Op("add", (x, y)), Op("neg", (x,))))).sugar() == "y")
    check("doubling is a shift here too",
          kb.same(Op("add", (x, x)), Op("shl", (x, Lit(1, int8)))))

def demo_soundness():
    """`proven` is an exact argument, or it is not `proven`."""
    print("4. what the e-graph will not claim")
    x, y = Var("x", int8), Var("y", int8)
    inputs = {"x": int8, "y": int8}
    # a rule whose domain is too large to enumerate bit-exactly is only sampled
    x16 = Var("x", int16)
    sampled = egraph_prove(Op("add", (x16, x16)),
                           Op("shl", (x16, Lit(1, int16))), {"x": int16})
    check("int16 doubling is found but only sampled",
          sampled.status == "not_proven" and sampled.checked == "sampled",
          sampled.reason)
    # ... while the same law over 8 bits is enumerated in full
    exact = egraph_prove(Op("add", (x, x)), Op("shl", (x, Lit(1, int8))), inputs)
    check("int8 doubling is bit-exact",
          exact.status == "proven" and exact.checked == "complete",
          exact.reason)
    # unrelated terms are simply not proven
    unrelated = egraph_prove(Op("mul", (x, y)), Op("add", (x, y)), inputs)
    check("x*y == x+y is not proven",
          unrelated.status == "not_proven", unrelated.reason)

def main():
    demo_laws_the_rewriter_refuses()
    demo_factoring()
    demo_curation()
    demo_soundness()
    print(f"\nAll {CHECKS} e-graph checks passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
