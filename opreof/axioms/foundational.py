"""Foundational user axioms for OProof.

Each axiom is an ordinary Axiom object; the kernel only ever applies a rule
whose side conditions the checker has verified for the concrete match, and
every result is re-normalized and re-checked.  These two rules are purely
algebraic (valid for every fixed-width two's-complement type) and exist to
demonstrate the add / remove / load cycle -- not to extend the sound core.
"""

from opreof import Axiom, Condition, Lit, Op, Var

_INTEGER_TYPES_COND = Condition(
    lambda subs, types: (
        "a" in types and types["a"] is not None and not types["a"].is_bool()
    ),
    "a is an integer type",
)

# a + a  ~>  a << 1        (valid for all integer widths: 2a mod 2^w)
ADD_SELF_IS_SHL1 = Axiom(
    "add_self_is_shl1",
    Op("add", (Var("a"), Var("a"))),
    Op("shl", (Var("a"), Lit(1))),
    conditions=(_INTEGER_TYPES_COND,),
    source="axioms/foundational.py",
)

# x - y  ~>  x + (-y)      (valid for all integer widths)
SUB_AS_ADD_NEG = Axiom(
    "sub_as_add_neg",
    Op("sub", (Var("x"), Var("y"))),
    Op("add", (Var("x"), Op("neg", (Var("y"),))), None),
    conditions=(Condition(
        lambda sub, types: True, "always"
    ),),
    source="axioms/foundational.py",
)

AXIOMS = [
    ADD_SELF_IS_SHL1,
    SUB_AS_ADD_NEG,
]
