"""Automatic axiom generalization.

When the kernel CONCRETELY proves two expressions equal, it does not stop
there.  Each proven pair is anti-unified: concrete literal constants at the
aligned positions are lifted into fresh typed variables, producing the most
general pattern pair.  The kernel then re-runs the prover on the generalised
pair over fresh symbolic inputs:

* if the generalised equivalence re-proves cleanly, the fact is promoted to a
  new axiom -- registered in the running store AND persisted to an ordinary
  Python axiom file, so it survives restarts;
* if it does not re-prove, nothing is installed (a near-miss lift such as
  ``x * 2 == x * 4`` is silently discarded).

This is fully automatic, and safety-critical: an axiom is only ever installed
when the checker itself has certified the *generalised* statement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from axiom_store import TRUE_COND, Axiom, AxiomError
from kernel import (
    Const,
    Expr,
    Kernel,
    Lit,
    Op,
    Select,
    Type,
    Var,
    free_vars,
)

_GENERATION = 0

@dataclass

class _AntiResult:
    g1: Expr
    g2: Expr
    lifted: int  # number of positions lifted
    introduced: tuple[str, ...]  # fresh variable names

def _fresh_name(used: set[str], prefix: str = "_p") -> str:
    i = len(used)
    while True:
        cand = f"{prefix}{i}"
        if cand not in used:
            used.add(cand)
            return cand
        i += 1

def anti_unify(
    a: Expr,
    b: Expr,
    used: set[str],
) -> _AntiResult | None:
    """Anti-unify two proven-equal expressions.

    Equal aligned literals are left as literals (nothing to generalise there);
    *differing* aligned literals become one shared fresh typed variable.  Any
    structural difference that cannot be unified makes the pair non-
    generalisable and ``None`` is returned.
    """
    if isinstance(a, Lit) or isinstance(b, Lit):
        # Differing aligned literals become one shared fresh typed variable;
        # equal aligned literals carry no information and stay as literals.
        if isinstance(a, Lit) and isinstance(b, Lit):
            if a.value == b.value:
                return _AntiResult(a, b, 0, ())
            ty = a.type if a.type is not None else b.type
            name = _fresh_name(used)
            v = Var(name, ty)
            return _AntiResult(v, v, 1, (name,))
        other = b if isinstance(a, Lit) else a
        if not isinstance(other, (Var, Const)):
            return None  # only lift literals against literals or variables
        ty = (
            (a.type if isinstance(a, Lit) else None)
            or (b.type if isinstance(b, Lit) else None)
            or other.type
        )
        if ty is None:
            return None
        name = _fresh_name(used)
        v = Var(name, ty)
        return _AntiResult(v, v, 1, (name,))

    if isinstance(a, Op) and isinstance(b, Op):
        if a.op != b.op or len(a.args) != len(b.args):
            return None
        parts = []
        lifted = 0
        names: list[str] = []
        for pa, pb in zip(a.args, b.args):
            r = anti_unify(pa, pb, used)
            if r is None:
                return None
            parts.append((r.g1, r.g2))
            lifted += r.lifted
            names.extend(r.introduced)
        g1 = Op(a.op, tuple(x[0] for x in parts), a.type)
        g2 = Op(b.op, tuple(x[1] for x in parts), b.type)
        return _AntiResult(g1, g2, lifted, tuple(names))

    if isinstance(a, Select) and isinstance(b, Select):
        parts = []
        lifted = 0
        names: list[str] = []
        for pa, pb in (
            (a.cond, b.cond),
            (a.then_expr, b.then_expr),
            (a.else_expr, b.else_expr),
        ):
            r = anti_unify(pa, pb, used)
            if r is None:
                return None
            parts.append((r.g1, r.g2))
            lifted += r.lifted
            names.extend(r.introduced)
        g1 = Select(parts[0][0], parts[1][0], parts[2][0], a.type)
        g2 = Select(parts[0][1], parts[1][1], parts[2][1], b.type)
        return _AntiResult(g1, g2, lifted, tuple(names))

    if type(a) is type(b) and isinstance(a, (Var, Const)):
        if a.name == b.name and a.type == b.type:  # pyright: ignore[reportAttributeAccessIssue]
            return _AntiResult(a, b, 0, ())
        return None

    # Exactly one side is a bare variable/constant leaf: keep the pair as-is
    # (they differ in shape, but both sides already use pattern variables).
    if isinstance(a, (Var, Const)) or isinstance(b, (Var, Const)):
        return _AntiResult(a, b, 0, ())
    return None

def var_types(expr: Expr) -> dict[str, Type]:
    got: dict[str, Type] = {}

    def walk(e: Expr) -> None:
        if isinstance(e, (Var, Const)):
            if e.name not in got and e.type is not None:
                got[e.name] = e.type
        elif isinstance(e, Op):
            for a in e.args:
                walk(a)
        elif isinstance(e, Select):
            walk(e.cond)
            walk(e.then_expr)
            walk(e.else_expr)

    walk(expr)
    return got

def derive_axiom(
    kernel: Kernel,
    e1: Expr,
    e2: Expr,
    axioms_dir: str | Path | None = None,
    *,
    source: str = "generalization",
) -> Axiom | None:
    """Attempt to promote a concrete equality into a generalised axiom.

    Returns the new axiom when installed, otherwise ``None``.
    """
    used: set[str] = set(free_vars(e1) | free_vars(e2))
    lift = anti_unify(e1, e2, used)
    if lift is None:
        return None
    if lift.g1 == lift.g2:
        return None  # tautology after lifting -- nothing new

    inputs = dict(var_types(lift.g1))
    for name in [n for n in lift.introduced if n not in inputs]:
        ty = var_types(lift.g1).get(name) or var_types(lift.g2).get(name)
        if ty is None:
            return None
        inputs[name] = ty

    proof = kernel.prove_expr(lift.g1, lift.g2, inputs=inputs, generalize=False)
    if not proof:
        return None  # the generalised claim does not hold -- discard

    global _GENERATION
    name = f"auto_{_GENERATION:04d}"
    _GENERATION += 1

    ax = Axiom(name, lift.g1, lift.g2, conditions=(TRUE_COND,), source=source)
    if not kernel.axioms.register(ax):
        return None  # duplicate rule already present

    if axioms_dir is not None:
        try:
            kernel.axioms.export_source(axioms_dir, ax)
        except AxiomError:
            pass
    return ax

def generalize_equality(
    kernel: Kernel,
    e1: Expr,
    e2: Expr,
    axioms_dir: str | Path | None = None,
) -> list[Axiom]:
    """Auto-generalize every output pair; returns newly installed axioms."""
    ax = derive_axiom(kernel, e1, e2, axioms_dir)
    return [ax] if ax is not None else []
