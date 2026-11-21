"""Arithmetic structure theory for OProof.

Every rule below is a universal bit-exact identity for fixed-width
two's-complement arithmetic (modular ``add``/``sub``/``mul``/``neg``), so it
is sound on *any* kernel integer type.  These rules flatten and re-orient the
symbolic arithmetic so later rewrites and comparisons meet a canonical
structure; the kernel re-normalizes and re-checks every application.

Orientation is deliberate and away from duplicate/shifted structure; no two
rules in this store reverse each other, so normalisation always decreases.
"""

from opreof import Axiom, Condition, Op, Var

_INT_COND = Condition(
    lambda subs, types: (
        types.get("a") is not None and not types["a"].is_bool()
    ),
    "a is an integer type",
)

# x + (y - x)  ~>  y
SUB_ABSORB = Axiom(
    "sub_absorb",
    Op("add", (Op("sub", (Var("b"), Var("a"))), Var("a"))),
    Var("b"),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
)

# x - (x - y)  ~>  y
SUB_SELF = Axiom(
    "sub_self",
    Op("sub", (Var("a"), Op("sub", (Var("a"), Var("b"))))),
    Var("b"),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
)

# (x - y) + y  ~>  x
ADD_EXTRA_SUB = Axiom(
    "add_extra_sub",
    Op("add", (Op("sub", (Var("a"), Var("b"))), Var("b"))),
    Var("a"),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
)

# (x + y) - x  ~>  y
SUB_TAKE_LEFT = Axiom(
    "sub_take_left",
    Op("sub", (Op("add", (Var("a"), Var("b"))), Var("a"))),
    Var("b"),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
)

# (x + y) - y  ~>  x
SUB_TAKE_RIGHT = Axiom(
    "sub_take_right",
    Op("sub", (Op("add", (Var("a"), Var("b"))), Var("b"))),
    Var("a"),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
)

# -(x - y)  ~>  y - x
NEG_SUB = Axiom(
    "neg_sub",
    Op("neg", (Op("sub", (Var("a"), Var("b"))),)),
    Op("sub", (Var("b"), Var("a"))),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
)

# (-x) - y  ~>  -(x + y)
SUB_NEG_LEFT = Axiom(
    "sub_neg_left",
    Op("sub", (Op("neg", (Var("a"),)), Var("b"))),
    Op("neg", (Op("add", (Var("a"), Var("b"))),)),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
)

# (-x) * y  ~>  -(x * y)
MUL_NEG_LEFT = Axiom(
    "mul_neg_left",
    Op("mul", (Op("neg", (Var("a"),)), Var("b"))),
    Op("neg", (Op("mul", (Var("a"), Var("b"))),)),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
)

# (-x) * (-y)  ~>  x * y
MUL_NEG_BOTH = Axiom(
    "mul_neg_both",
    Op("mul", (Op("neg", (Var("a"),)), Op("neg", (Var("b"),)))),
    Op("mul", (Var("a"), Var("b"))),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
)

# (a - b) + b  ~>  a;  b + (a - b)  ~>  a
# Normalisation runs bottom-up, so the inner ``sub`` of ``a + (b - a)`` is
# already rewritten into its canonical ``add(a, neg(b))`` form before an
# axiom could match ``sub`` at the root.  The peeling rules below therefore
# work directly on the canonical add/neg orientation.
PEEL_ADD_SUB = Axiom(
    "peel_add_neg",
    Op("add", (Op("add", (Op("neg", (Var("a"),)), Var("b"))), Var("a"))),
    Var("b"),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
)

# a - (a - b)  ~>  b   (canonical form: add(neg(add(neg(b), a)), a))
PEEL_SUB_SELF = Axiom(
    "peel_sub_self",
    Op("add", (Op("neg", (Op("add", (Op("neg", (Var("a"),)), Var("b"))),)), Var("b"))),
    Var("a"),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
)

# (a + b) - a  ~>  b   (canonical form: add(add(a, b), neg(a)))
PEEL_TAKE_ADD = Axiom(
    "peel_take_neg",
    Op("add", (Op("add", (Var("a"), Var("b"))), Op("neg", (Var("a"),)))),
    Var("b"),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
)

# -(a - b)  ~>  b - a.  The inner ``sub`` is already canonicalised away, so
# this matches the surviving shape  neg(add(neg(a), b))  and hands back the
# sorted  add(neg(b), a)  form of ``b - a``.
NEG_ADD_NEG = Axiom(
    "neg_add_neg",
    Op("neg", (Op("add", (Op("neg", (Var("a"),)), Var("b"))),)),
    Op("add", (Op("neg", (Var("b"),)), Var("a"))),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
)

# (a + b) + c  ~>  a + (b + c)   -- modular addition is associative.
# Normalisation does not reassociate on its own, so this is the axiom that
# flattens chains of sums into a canonical (right-nested) shape.
ADD_ASSOC = Axiom(
    "add_assoc",
    Op("add", (Op("add", (Var("a"), Var("b"))), Var("c"))),
    Op("add", (Var("a"), Op("add", (Var("b"), Var("c"))))),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
    equational=True,
)

# (a * b) * c  ~>  a * (b * c)   -- modular multiplication is associative.
MUL_ASSOC = Axiom(
    "mul_assoc",
    Op("mul", (Op("mul", (Var("a"), Var("b"))), Var("c"))),
    Op("mul", (Var("a"), Op("mul", (Var("b"), Var("c"))))),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
    equational=True,
)

# a * (b + c)  ~>  a * b + a * c   -- distributivity of the modular ring.
# Normalisation sorts commutative operands first, so the product of a sum is
# seen as ``mul(add(b, c), a)``; the right-hand side is the canonical,
# fully-distributed form.  The e-graph re-uses the same rule *undirected* to
# factor the other way.
MUL_DIST = Axiom(
    "mul_dist",
    Op("mul", (Op("add", (Var("b"), Var("c"))), Var("a"))),
    Op("add", (Op("mul", (Var("a"), Var("b"))), Op("mul", (Var("a"), Var("c"))))),
    conditions=(_INT_COND,),
    source="axioms/arithmetic.py",
    equational=True,
)

AXIOMS = [
    SUB_ABSORB,
    SUB_SELF,
    ADD_EXTRA_SUB,
    SUB_TAKE_LEFT,
    SUB_TAKE_RIGHT,
    NEG_SUB,
    SUB_NEG_LEFT,
    MUL_NEG_LEFT,
    MUL_NEG_BOTH,
    PEEL_ADD_SUB,
    PEEL_SUB_SELF,
    PEEL_TAKE_ADD,
    NEG_ADD_NEG,
    ADD_ASSOC,
    MUL_ASSOC,
    MUL_DIST,
]
