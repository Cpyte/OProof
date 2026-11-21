"""Power-of-two arithmetic for OProof.

Machine semantics (see kernel.py): signed ``divs``/``rems`` truncate toward
zero (C, RISC-V, WASM, x86), while an arithmetic right shift (``shrs``) is
*floor* division by ``2**k``.  Therefore::

    divs(a, 2**k) == shrs(a, k)          holds ONLY for a >= 0

so the naive signed divide-by-power-of-two is NOT promoted globally.  The
identity that is sound for every ``a`` is the compiler's sign-corrected
strength reduction::

    divs(a, 2**k) == shrs(a + (shrs(a, w-1) & (2**k - 1)), k)   (w = width)

which this module promotes as one rule per (machine type, k).  The unsigned
forms are globally sound (they are pure bit-pattern operations, where
division, logical shift and mask coincide):

    divu(a, 2**k) == shru(a, k)
    remu(a, 2**k) == and(a, 2**k - 1)

EVERY rule independently asserts ``k < width(a)`` in its own condition, so
its RHS shift can never be out of range no matter what type the dividend or
divisor carries -- the invariant is part of the axiom, not something a later
ripening stage happens to enforce.
"""

from opreof import Axiom, Condition, Lit, Op, Var, cond_type, int8, int16, int32

_INT_COND = Condition(
    lambda subs, types: (
        types.get("a") is not None and not types["a"].is_bool()
    ),
    "a is an integer type",
)

def _within_width(k: int) -> Condition:
    """Explicit guard: the power-of-two / shift in question fits width(a)."""
    return Condition(
        lambda subs, types, k=k: (
            types.get("a") is not None and k < types["a"].width
        ),
        f"2**{k} fits width(a)",
    )

def _pow2_shift(op: str, shiftop: str, divisor: int, k: int) -> Axiom:
    return Axiom(
        f"{op}_pow2_{divisor}",
        Op(op, (Var("a"), Lit(divisor))),
        Op(shiftop, (Var("a"), Lit(k))),
        conditions=(_INT_COND, _within_width(k)),
        source="axioms/pow2.py",
    )

def _pow2_remu(mask: int, k: int) -> Axiom:
    return Axiom(
        f"remu_pow2_{1 << k}",
        Op("remu", (Var("a"), Lit(1 << k))),
        Op("and", (Var("a"), Lit(mask))),
        conditions=(_INT_COND, _within_width(k)),
        source="axioms/pow2.py",
    )

def _divs_corr(ty, k: int) -> Axiom:
    """Sign-corrected signed strength reduction, parameterised per type."""
    mask = (1 << k) - 1
    divisor = 1 << k
    rhs = Op(
        "shrs",
        (
            Op(
                "add",
                (
                    Var("a"),
                    Op(
                        "and",
                        (Op("shrs", (Var("a"), Lit(ty.width - 1))), Lit(mask)),
                    ),
                ),
            ),
            Lit(k),
        ),
    )
    return Axiom(
        f"divs_corr_{ty.name}_{divisor}",
        Op("divs", (Var("a"), Lit(divisor))),
        rhs,
        conditions=(cond_type("a", ty), _within_width(k)),
        source="axioms/pow2.py",
    )

AXIOMS = []
for _k in range(1, 9):
    _d = 1 << _k
    AXIOMS.append(_pow2_shift("divu", "shru", _d, _k))
    AXIOMS.append(_pow2_remu(_d - 1, _k))
for _ty in (int8, int16, int32):
    for _k in range(1, _ty.width - 1):
        AXIOMS.append(_divs_corr(_ty, _k))
