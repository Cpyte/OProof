"""Boolean and bit-logic theory for OProof.

Two families:

1. ``BOOL`` identities against the constants ``0``/``1`` (equivalences with and
   without the literal on either side), gated on the operand being a bool.
2. Absorption laws of the bitwise ops (valid for any integer type since they
   are pure bitwise identities):

       x | (x & y) == x
       x & (x | y) == x

3. The all-ones mask: ``xor(x, -1) == not(x)`` (two's-complement ``-1`` is
   the all-ones mask, so it flips every bit), valid for any integer width.
4. De Morgan / complement: ``not(x & y) == not(x) | not(y)``, the exclusive-or
   decomposition ``x ^ y == (x & ~y) | (~x & y)``, and the complement
   annihilators ``x & ~x == 0``, ``x | ~x == -1``, ``x & -1 == x``,
   ``x | -1 == -1``.

Both ``and``/``or``/``xor`` sort their operands during normalisation, so the
absorption and all-ones rules are written in both operand orders to fire on
either sort.
"""

from opreof import BOOL, Axiom, Condition, Lit, Op, Var, cond_type

_BOOL_COND = cond_type("a", BOOL)

AND_ONE = Axiom(
    "bool_and_one",
    Op("and", (Lit(1), Var("a"))),
    Var("a"),
    conditions=(_BOOL_COND,),
    source="axioms/logic.py",
)

OR_ONE = Axiom(
    "bool_or_one",
    Op("or", (Lit(1), Var("a"))),
    Lit(1, BOOL),
    conditions=(_BOOL_COND,),
    source="axioms/logic.py",
)

XOR_ONE = Axiom(
    "bool_xor_one",
    Op("xor", (Lit(1), Var("a"))),
    Op("not", (Var("a"),)),
    conditions=(_BOOL_COND,),
    source="axioms/logic.py",
)

EQ_ZERO = Axiom(
    "bool_eq_zero",
    Op("eq", (Var("a"), Lit(0))),
    Op("not", (Var("a"),)),
    conditions=(_BOOL_COND,),
    source="axioms/logic.py",
)

EQ_ZERO_R = Axiom(
    "bool_eq_zero_r",
    Op("eq", (Lit(0), Var("a"))),
    Op("not", (Var("a"),)),
    conditions=(_BOOL_COND,),
    source="axioms/logic.py",
)

NE_ZERO = Axiom(
    "bool_ne_zero",
    Op("ne", (Var("a"), Lit(0))),
    Var("a"),
    conditions=(_BOOL_COND,),
    source="axioms/logic.py",
)

NE_ZERO_R = Axiom(
    "bool_ne_zero_r",
    Op("ne", (Lit(0), Var("a"))),
    Var("a"),
    conditions=(_BOOL_COND,),
    source="axioms/logic.py",
)

EQ_ONE = Axiom(
    "bool_eq_one",
    Op("eq", (Var("a"), Lit(1))),
    Var("a"),
    conditions=(_BOOL_COND,),
    source="axioms/logic.py",
)

EQ_ONE_R = Axiom(
    "bool_eq_one_r",
    Op("eq", (Lit(1), Var("a"))),
    Var("a"),
    conditions=(_BOOL_COND,),
    source="axioms/logic.py",
)

NE_ONE = Axiom(
    "bool_ne_one",
    Op("ne", (Var("a"), Lit(1))),
    Op("not", (Var("a"),)),
    conditions=(_BOOL_COND,),
    source="axioms/logic.py",
)

NE_ONE_R = Axiom(
    "bool_ne_one_r",
    Op("ne", (Lit(1), Var("a"))),
    Op("not", (Var("a"),)),
    conditions=(_BOOL_COND,),
    source="axioms/logic.py",
)

_ABS_COND = Condition(
    lambda subs, types: (
        types.get("a") is not None and not types["a"].is_bool()
    ),
    "a is an integer type",
)

OR_ABSORB_L = Axiom(
    "or_absorb_left",
    Op("or", (Op("and", (Var("a"), Var("b"))), Var("a"))),
    Var("a"),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
)

OR_ABSORB_R = Axiom(
    "or_absorb_right",
    Op("or", (Op("and", (Var("b"), Var("a"))), Var("a"))),
    Var("a"),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
)

AND_ABSORB_L = Axiom(
    "and_absorb_left",
    Op("and", (Op("or", (Var("a"), Var("b"))), Var("a"))),
    Var("a"),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
)

AND_ABSORB_R = Axiom(
    "and_absorb_right",
    Op("and", (Op("or", (Var("b"), Var("a"))), Var("a"))),
    Var("a"),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
)

#
# In two's complement ``-1`` is the all-ones mask, so xoring with it flips
# every bit -- exactly the machine ``not``.  Valid for every integer width
# (no width assumption), written in both operand orders because xor sorts
# its operands during normalisation.

XOR_ALL_ONES = Axiom(
    "xor_all_ones",
    Op("xor", (Var("a"), Lit(-1))),
    Op("not", (Var("a"),)),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
)

XOR_ALL_ONES_R = Axiom(
    "xor_all_ones_r",
    Op("xor", (Lit(-1), Var("a"))),
    Op("not", (Var("a"),)),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
)

# --- De Morgan / complement families (any integer type) ---------------------
#
# Bitwise negation and the boolean ops are pure bit-pattern identities, so
# De Morgan, absorption of ``a ^ ~a``, and the ``-1`` (all-ones) annihilators
# are valid for every two's-complement width.  Written in the orientation the
# normaliser produces (``not(...)`` sorts before a bare variable).

NOT_AND = Axiom(
    "not_and",
    Op("not", (Op("and", (Var("a"), Var("b"))),)),
    Op("or", (Op("not", (Var("a"),)), Op("not", (Var("b"),)))),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
    # an identity, not a step: the reverse direction undoes it, so the
    # directed rewriter must leave it to the e-graph
    equational=True,
)

NOT_OR = Axiom(
    "not_or",
    Op("not", (Op("or", (Var("a"), Var("b"))),)),
    Op("and", (Op("not", (Var("a"),)), Op("not", (Var("b"),)))),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
    equational=True,
)

XOR_NOT_LEFT = Axiom(
    "xor_not_left",
    Op("not", (Op("xor", (Var("a"), Var("b"))),)),
    Op("xor", (Op("not", (Var("a"),)), Var("b"))),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
)

XOR_VIA_AND_OR = Axiom(
    "xor_via_and_or",
    Op("xor", (Var("a"), Var("b"))),
    Op("or", (Op("and", (Var("a"), Op("not", (Var("b"),)))),
              Op("and", (Op("not", (Var("a"),)), Var("b")))),
        ),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
    equational=True,
)

AND_NOT_ZERO = Axiom(
    "and_not_zero",
    Op("and", (Op("not", (Var("a"),)), Var("a"))),
    Lit(0),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
)

OR_NOT_ONES = Axiom(
    "or_not_ones",
    Op("or", (Op("not", (Var("a"),)), Var("a"))),
    Lit(-1),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
)

AND_ALL_ONES_ID = Axiom(
    "and_all_ones_id",
    Op("and", (Lit(-1), Var("a"))),
    Var("a"),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
)

OR_ALL_ONES_ABS = Axiom(
    "or_all_ones_abs",
    Op("or", (Lit(-1), Var("a"))),
    Lit(-1),
    conditions=(_ABS_COND,),
    source="axioms/logic.py",
)

AXIOMS = [
    AND_ONE,
    OR_ONE,
    XOR_ONE,
    EQ_ZERO,
    EQ_ZERO_R,
    NE_ZERO,
    NE_ZERO_R,
    EQ_ONE,
    EQ_ONE_R,
    NE_ONE,
    NE_ONE_R,
    OR_ABSORB_L,
    OR_ABSORB_R,
    AND_ABSORB_L,
    AND_ABSORB_R,
    XOR_ALL_ONES,
    XOR_ALL_ONES_R,
    NOT_AND,
    NOT_OR,
    XOR_NOT_LEFT,
    XOR_VIA_AND_OR,
    AND_NOT_ZERO,
    OR_NOT_ONES,
    AND_ALL_ONES_ID,
    OR_ALL_ONES_ABS,
]
