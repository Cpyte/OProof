"""Solving for OProof -- go beyond equivalence checking.

Equivalence checking answers "do these two expressions agree on EVERY input?"
(as proven / not_proven / rejected).  Solving answers the richer questions:

    solve_expr(e1, e2, inputs)       -- the exact SET of inputs on which the
                                        two sides produce equal bits
    find_witness(e1, e2, inputs)     -- one concrete input where they agree
    find_counterexample(e1, e2, ...) -- one concrete input where they differ
    solve_for(e1, e2, inputs, var)   -- isolate ``var`` (closed form when the
                                        equation is invertible, else residual)
    synthesize(e1, e2, inputs, ?)    -- fill a hole ``?`` with a term making
                                        ``e1 == e2`` hold for all inputs

Semantics are exactly the kernel's: bit-exact over the typed (wrapping)
machine domains with ``eval_c`` as ground truth.  Enumeration is complete
(``SolveResult.exhaustive``) when the Cartesian product of the input domains
fits in ``EXHAUSTIVE_CAP``; larger products are resolved on a dense/edge
sample and flagged ``exhaustive=False``.

``solve_for``'s symbolw isolation is modular-algebra sound: on the 2**w ring
only the injective operations (add/sub/xor/neg, valid shifts, multiplication
by an odd constant) admit a unique inverse; anything else is reported as a
residual, and the exhaustive solution set still pins the variable down.
"""

from dataclasses import dataclass, field
from itertools import product

from kernel import (
    Const,
    Expr,
    KernelError,
    Lit,
    Op,
    Select,
    Type,
    Var,
    check_defined,
    eval_c,
    free_vars,
    ripen,
    substitute,
)

EXHAUSTIVE_CAP = 1 << 16  # fully enumerate products up to this many combos

# Domains

def _full_values(ty: Type) -> list[int]:
    n = ty.width
    if ty.signed:
        lo, hi = -(1 << (n - 1)), (1 << (n - 1)) - 1
    else:
        lo, hi = 0, (1 << n) - 1
    return list(range(lo, hi + 1))

def _rep_values(ty: Type) -> list[int]:
    n = ty.width
    if ty.signed:
        lo, hi = -(1 << (n - 1)), (1 << (n - 1)) - 1
    else:
        lo, hi = 0, (1 << n) - 1
    vals = [lo, lo + 1, -1, 0, 1, 127, 128, hi - 1, hi]
    return sorted({v for v in vals if lo <= v <= hi})

def _domain(ty: Type) -> tuple[list[int], bool]:
    """Return ``(values, exhaustive)`` for a machine type."""
    if (1 << ty.width) <= EXHAUSTIVE_CAP:
        return _full_values(ty), True
    return sorted(set(_rep_values(ty) + [0])), False

def _enumerate(inputs: dict) -> tuple[object, bool]:
    """Yield input combinations; return ``(iterator, exhaustive)``.

    Combos are yielded lazily; when the Cartesian product would exceed
    ``EXHAUSTIVE_CAP`` the enumeration degrades to a dense/edge sample on the
    largest domain and stops early, so a solver never blows up on wide types.
    """
    names = list(inputs)
    domains = {n: _domain(t) for n, t in inputs.items()}
    size = 1
    for vals, ex in domains.values():
        size *= len(vals)
    exhaustive = size <= EXHAUSTIVE_CAP
    if exhaustive:
        return (dict(zip(names, c)) for c in product(*[v for v, _ in domains.values()])), True

    # degrade: replace every full domain that is itself small with its full
    # values, and sample the rest; hard-stop at the cap for safety
    values = []
    for n in names:
        vals, ex = domains[n]
        if ex:
            values.append(vals)
        else:
            values.append(_rep_values(inputs[n]))
    seen = 0
    def gen():
        nonlocal seen
        for c in product(*values):
            yield dict(zip(names, c))
            seen += 1
            if seen >= EXHAUSTIVE_CAP:
                break
    return gen(), False

# Result type

@dataclass

class SolveResult:
    """Outcome of a solve / solve_for / synthesize request.

    ``status`` is one of ``always``, ``some``, ``never``, ``rejected``.
    ``satisfying`` lists the (sampled or exhaustive) inputs on which the two
    sides agree; ``exhaustive`` records whether that list is complete.
    """

    status: str = "rejected"
    kind: str = "equation"                      # equation | solve_for | synthesis
    lhs: Expr | None = None
    rhs: Expr | None = None
    inputs: dict = field(default_factory=dict)
    satisfying: list = field(default_factory=list)
    exhaustive: bool = False
    witness: dict | None = None
    counterexample: dict | None = None
    undefined: int = 0
    # solve_for
    solved_var: str | None = None
    symbolic: Expr | None = None
    residual: Expr | None = None
    # synthesis
    synthesized: object | None = None
    kmap: dict | None = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.status == "always"

    def __repr__(self) -> str:
        head = f"<OProof solve {self.kind} {self.status}"
        if self.satisfying:
            n = len(self.satisfying)
            head += f" (exhaustive {n})" if self.exhaustive else f" (sampled {n})"
        if self.symbolic is not None:
            head += f" {self.solved_var} == {self.symbolic.sugar()}"
        if self.residual is not None:
            head += f" residual={self.residual.sugar()}"
        if self.synthesized is not None:
            head += f" ? = {self.synthesized}"
        if self.reason:
            head += f": {self.reason}"
        return head + ">"

# Shared preparation

def _first_var_type(expr, name):
    if isinstance(expr, (Var, Const)) and expr.name == name:
        return expr.type
    if isinstance(expr, Op):
        for a in expr.args:
            t = _first_var_type(a, name)
            if t is not None:
                return t
    if isinstance(expr, Select):
        return (
            _first_var_type(expr.then_expr, name)
            or _first_var_type(expr.else_expr, name)
        )
    return None

def _prepare(kernel, e1, e2, inputs):
    """Ripen both sides and reconcile the input types.

    Returns ``(r1, r2, inputs, error_reason_or_None)``.  Mirrors
    ``kernel.prove_expr``: both sides must resolve to one concrete type, and
    every free variable needs a resolvable type.
    """
    inputs = dict(inputs or {})
    env = {n: (Var(n, t), t) for n, t in inputs.items()}
    r1 = ripen(e1, env=env)
    r2 = ripen(e2, env=env)
    if r1.type is None or r2.type is None:
        ty = r1.type or r2.type
        if ty is None:
            return None, None, inputs, "cannot resolve an expression type"
        r1 = ripen(e1, ty, env=env)
        r2 = ripen(e2, ty, env=env)
    if r1.type is not None and r2.type is not None and r1.type != r2.type:
        return None, None, inputs, (
            f"types differ, so the sides can never be equal: {r1.type} vs {r2.type}"
        )
    for nm in (free_vars(r1) | free_vars(r2)) - set(inputs):
        found = _first_var_type(r1, nm) or _first_var_type(r2, nm)
        if found is None:
            return None, None, inputs, f"cannot resolve type of input {nm!r}"
        inputs[nm] = found
    return r1, r2, inputs, None

def _eval_subst(kernel, ripened, subs) -> int:
    inst = substitute(ripened, subs)
    if check_defined(inst):
        raise KernelError("undefined on this input")
    return eval_c(inst)

# solve_expr

def solve_expr(kernel, e1, e2, inputs=None, *, target: str | None = None) -> SolveResult:
    """Return the inputs on which ``e1`` and ``e2`` evaluate to equal bits."""
    res = SolveResult(
        kind="solve_for" if target else "equation",
        lhs=e1, rhs=e2, inputs=dict(inputs or {}) if inputs else {},
    )
    try:
        r1, r2, inputs, error = _prepare(kernel, e1, e2, inputs)
        if error:
            res.status = "never" if "types differ" in error else "rejected"
            res.reason = error
            return res
        res.inputs = dict(inputs)
    except KernelError as err:
        res.status = "rejected"
        res.reason = err.message
        return res

    iterator, exhaustive = _enumerate(res.inputs)
    res.exhaustive = exhaustive

    satisfying: list[dict] = []
    legal = 0
    undefined = 0
    witness = None
    counterexample = None

    for combo in iterator:
        subs = {nm: Lit(v, res.inputs[nm]) for nm, v in combo.items()}
        try:
            i1 = substitute(r1, subs)
            i2 = substitute(r2, subs)
        except KernelError:
            continue
        if check_defined(i1) or check_defined(i2):
            undefined += 1
            continue
        legal += 1
        if eval_c(i1) == eval_c(i2):
            satisfying.append(combo)
            if witness is None:
                witness = combo
        elif counterexample is None:
            counterexample = combo

    res.undefined = undefined
    res.satisfying = satisfying
    res.witness = witness
    res.counterexample = counterexample

    if legal == 0:
        res.status = "rejected"
        res.reason = "ill-defined on every input"
    elif satisfying and len(satisfying) == legal:
        res.status = "always"
    elif satisfying:
        res.status = "some"
    else:
        res.status = "never"

    if target is not None:
        if target not in free_vars(r1) | free_vars(r2):
            res.reason = f"variable {target!r} does not occur in the equation"
            return res
        res.solved_var = target
        res.symbolic, res.residual = _isolate(kernel, r1, r2, target)
    return res

def find_witness(kernel, e1, e2, inputs=None) -> dict | None:
    """One input where ``e1`` and ``e2`` agree (``None`` when never)."""
    return solve_expr(kernel, e1, e2, inputs).witness

def find_counterexample(kernel, e1, e2, inputs=None) -> dict | None:
    """One input where ``e1`` and ``e2`` differ (``None`` when always)."""
    return solve_expr(kernel, e1, e2, inputs).counterexample

# solve_for -- symbolic isolation of one variable

def _modulus(ty: Type) -> int:
    return 1 << ty.width

def _lit_inverse(c: int, ty: Type) -> Lit | None:
    """Multiplicative inverse of ``c`` on the 2**w ring, if odd (invertible)."""
    m = _modulus(ty)
    v = c % m
    if v % 2 == 0 or v == 0:
        return None
    inv = pow(v, -1, m)
    if ty.signed and inv >= (1 << (ty.width - 1)):
        inv -= m
    return Lit(inv, ty)

def _static_lit(e: Expr) -> int | None:
    return e.value if isinstance(e, Lit) else None

def _var_only(e: Expr, var: str) -> bool:
    return isinstance(e, Var) and e.name == var

def _isolate(kernel, r1, r2, var: str) -> tuple[Expr | None, Expr | None]:
    """Derive ``var == f(other vars)`` if the equation is invertible in ``var``.

    Returns ``(closed_form, residual)``; exactly one is non-None.  Sound under
    the wrapping 2**w semantics: only injective operations (add/sub/xor/neg,
    valid shifts, multiplication by an odd constant, xor with a constant) are
    inverted.  The variable appearing in a non-invertible position (or on both
    sides multiplicatively) yields a residual and the caller keeps the
    enumeration-based solution set.
    """
    ty = r1.type or r2.type
    if ty is None:
        return None, None
    try:
        d = kernel.normalize(Op("sub", (r1, r2), ty))
    except KernelError:
        return None, None

    open_expr = d
    solved = Lit(0, ty)

    def contains(e: Expr) -> bool:
        return var in free_vars(e)

    while isinstance(open_expr, Op) and contains(open_expr):
        op, args = open_expr.op, open_expr.args
        try:
            if op in ("add", "xor") and contains(args[0]) and not contains(args[1]):
                kwop = "sub" if op == "add" else "xor"
                open_expr, solved = args[0], kernel.normalize(
                    Op(kwop, (solved, args[1]), ty))
            elif op == "add" and contains(args[1]) and not contains(args[0]):
                open_expr, solved = args[1], kernel.normalize(
                    Op("sub", (solved, args[0]), ty))
            elif op == "sub" and contains(args[0]) and not contains(args[1]):
                open_expr, solved = args[0], kernel.normalize(
                    Op("add", (solved, args[1]), ty))
            elif op == "sub" and contains(args[1]) and not contains(args[0]):
                open_expr, solved = args[1], kernel.normalize(
                    Op("sub", (args[0], solved), ty))
            elif op == "neg":
                open_expr, solved = args[0], kernel.normalize(
                    Op("neg", (solved,), ty))
            elif op == "mul" and contains(args[0]) and not contains(args[1]):
                inv = _lit_inverse(_static_lit(args[1]), ty)
                if inv is None:
                    return None, d
                open_expr, solved = args[0], kernel.normalize(
                    Op("mul", (solved, inv), ty))
            elif op == "mul" and contains(args[1]) and not contains(args[0]):
                inv = _lit_inverse(_static_lit(args[0]), ty)
                if inv is None:
                    return None, d
                open_expr, solved = args[1], kernel.normalize(
                    Op("mul", (solved, inv), ty))
            elif op == "shl" and contains(args[0]) and _static_lit(args[1]) is not None:
                k = _static_lit(args[1])
                if not (0 <= k < ty.width):
                    return None, d
                open_expr, solved = args[0], kernel.normalize(
                    Op("shru", (solved, Lit(k, ty)), ty))
            else:
                # non-injective position (shru/shrs/div/rem/and/cmp/select)
                # or the variable on both sides: cannot isolate by inversion
                return None, d
        except KernelError:
            return None, d

        # normalize open_expr after each move so the solver never loops
        try:
            open_expr = kernel.normalize(open_expr)
        except KernelError:
            return None, d

    if _var_only(open_expr, var):
        return kernel.normalize(solved), None
    return None, d

# synthesize -- fill a hole so the equivalence becomes an identity

def synthesize(kernel, e1, e2, inputs=None, *, hole: str = "?") -> SolveResult:
    """Find a term for the hole ``hole`` making ``e1 == e2`` on every input.

    ``hole`` is a free variable name (default ``"?"``) that may be a constant
    placeholder or a function of the other inputs.  When the required hole
    value is unique per input, a closed term is synthesized with a bounded
    search over the kernel's own operation vocabulary; otherwise the exact
    k-map (inputs -> satisfying hole values) is returned in ``.kmap``.
    """
    res = SolveResult(kind="synthesis", lhs=e1, rhs=e2)
    inputs = dict(inputs or {})
    try:
        r1, r2, inputs, error = _prepare(kernel, e1, e2, inputs)
        if error:
            res.status = "never" if "types differ" in error else "rejected"
            res.reason = error
            return res
    except KernelError as err:
        res.status = "rejected"
        res.reason = err.message
        return res

    fv = free_vars(r1) | free_vars(r2)
    if hole not in fv:
        res.status = "rejected"
        res.reason = f"hole {hole!r} does not occur in the equation"
        return res

    ty = r1.type or r2.type
    hole_ty = inputs.get(hole) or ty
    others = sorted(fv - {hole})

    ovals = [_domain(inputs[n])[0] for n in others]
    hvals, hexhaustive = _domain(hole_ty)
    res.exhaustive = all(_domain(inputs[n])[1] for n in others) and hexhaustive
    combos = [dict(zip(others, c)) for c in product(*ovals)] if others else [{}]
    req_hits = {tuple(c.get(n) for n in others): [] for c in combos}

    for combo in combos:
        key = tuple(combo[n] for n in others)
        osubs = {n: Lit(combo[n], inputs[n]) for n in others}
        hits = []
        for cv in hvals:
            subs = {**osubs, hole: Lit(cv, hole_ty)}
            try:
                if _eval_subst(kernel, r1, subs) == _eval_subst(kernel, r2, subs):
                    hits.append(cv)
            except KernelError:
                continue
        req_hits[key] = hits

    res.kmap = {combo: hits for combo, hits in req_hits.items()}
    feasible = all(hits for hits in req_hits.values())

    if not others:
        # the hole is a plain constant
        res.satisfying = [{"?": cv} for hits in req_hits.values() for cv in hits]
        res.witness = res.satisfying[0] if res.satisfying else None
        if feasible and res.satisfying:
            res.status = "always"
            from api import Literal
            res.synthesized = Literal(res.satisfying[0]["?"], hole_ty)
        elif feasible:
            res.status = "never"
            res.reason = "no constant makes the identity hold on every input"
        else:
            res.status = "never"
            res.reason = "no hole value works for every input"
        return res

    # a function of the other inputs: unique required value per input?
    if feasible and all(len(h) == 1 for h in req_hits.values()):
        requirement = {combo: h[0] for combo, h in req_hits.items()}
        term = _synth_candidate(kernel, requirement, others, ty)
        if term is not None:
            res.status = "always"
            res.synthesized = term
            res.satisfying = [{**dict(zip(others, combo)), hole: val}
                              for combo, val in requirement.items()]
            res.witness = res.satisfying[0] if res.satisfying else None
            return res

    res.status = "some"
    res.reason = (
        "hole value is not unique per input (or no closed term found); "
        "see .kmap for the exact satisfying table"
    )
    return res

def _synth_candidate(kernel, requirement, others, ty):
    """Search a bounded term vocabulary for an exact match of ``requirement``."""
    from opreof import term_of_expr  # local import avoids a cycle

    base = [Var(n, ty) for n in others] + [Lit(0, ty), Lit(1, ty)]
    if ty.signed:
        base.append(Lit(-1, ty))
    pool = list(base)
    for e in base:
        pool.append(Op("neg", (e,), ty))
        pool.append(Op("not", (e,), ty))
        if isinstance(e, Var):
            pool.append(Op("shl", (e, Lit(1, ty)), ty))
    for a in base:
        for b in base:
            for op in ("add", "sub", "mul", "and", "or", "xor"):
                pool.append(Op(op, (a, b), ty))

    env = {n: (Var(n, ty), ty) for n in others}
    for cand in pool:
        try:
            rip = ripen(cand, ty, env=env)
        except KernelError:
            continue
        ok = True
        for combo, want in requirement.items():
            subs = {n: Lit(v, ty) for n, v in zip(others, combo)}
            try:
                got = _eval_subst(kernel, rip, subs)
            except KernelError:
                ok = False
                break
            if got != want:
                ok = False
                break
        if ok:
            folded = kernel.normalize(rip)
            if isinstance(folded, Lit):
                from api import Literal
                return Literal(folded.value, ty)
            return term_of_expr(folded)
    return None
