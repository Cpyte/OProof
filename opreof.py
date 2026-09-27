"""OProof -- public API.

This module is the single entry point ("the api interface") for the OProof
system.  It deliberately keeps three layers separate:

    api.Term / api.Call          the proof-language / API representation
        |  expr_of_term / term_of_expr        (explicit bridge)
        v
    kernel.Expr / kernel.Op      the kernel's typed expression language
        |  resolve_term_type
        v
    kernel.Type                  the machine semantic type

Axioms are ordinary Python files under ``axioms/``; they are loaded at import,
and may be added or removed at any time through this interface.  Proven facts
are automatically generalized and promoted to new axioms (persisted back to
``axioms/``).

Consumer software is never forced to accept the whole store: ``list_theories``
/``theory_rules``/``with_axioms``/``new_kernel`` expose the axiom set as data
so a caller can decide which facts a proof may use, and the e-graph entry
points (``egraph_prove``, ``egraph_normalize``, ``prepare_egraph``) let it
saturate those facts -- including the purely *equational* ones such as
associativity and distributivity that a directed rewriter must not fire --
and then extract a minimal, independently verified form.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from api import (
    Call,
    Literal,
    Term,
    _parse_parms,
    parse_parms,
)
from api import (
    Type as APIType,
)
from axiom_store import (
    TRUE_COND,
    Axiom,
    AxiomError,
    AxiomRegistry,
    Condition,
    axiom,
    cond_lit,
    cond_lit_range,
    cond_type,
)
from egraphs import EGraph, EGraphRunner
from kernel import (
    BOOL,
    Assign,
    Block,
    Const,
    DefinitionError,
    Expr,
    Kernel,
    KernelError,
    Lit,
    Loop,
    Nop,
    Op,
    OverFlowError,
    Proof,
    Select,
    Type,
    TypeError_,
    UndefinedError,
    Var,
    builtin_rewrite,
    check_defined,
    eval_c,
    free_vars,
    int8,
    int16,
    int32,
    int64,
    parse_type,
    ripen,
    uint8,
    uint16,
    uint32,
    uint64,
)
from solving import (
    SolveResult,
)
from solving import (
    find_counterexample as _find_counterexample,
)
from solving import (
    find_witness as _find_witness,
)
from solving import (
    solve_expr as _solve_expr,
)
from solving import (
    synthesize as _synthesize,
)

AXIOM_DIR = Path(__file__).resolve().parent / "axioms"

#: The live kernel.  One axiom store; every proof is audited by it.
kernel = Kernel(axioms_dir=str(AXIOM_DIR))

def load_axioms(directory: str | Path | None = None) -> list[Axiom]:
    """(Re)load every axiom file from ``axioms/`` (or a given directory)."""
    directory = Path(directory) if directory is not None else AXIOM_DIR
    return kernel.axioms.load_dir(directory)

def register_axiom(ax: Axiom) -> bool:
    """Add an axiom to the running kernel."""
    return kernel.axioms.register(ax)

def remove_axiom(name: str) -> bool:
    """Remove an axiom from the running kernel (by name)."""
    return kernel.axioms.remove(name)

def list_axioms() -> list[str]:
    return kernel.axioms.names()

def get_axiom(name: str) -> Axiom | None:
    return kernel.axioms.get(name)

def prove(
    b1: Block,
    b2: Block,
    inputs: dict[str, Type] | None = None,
    outputs: set[str] | None = None,
    *,
    generalize: bool = True,
    axioms: tuple[str, ...] | None = None,
) -> Proof:
    """Prove two sequences of code produce EXACTLY the same output bits.

    ``axioms`` curates the rules the proof may use -- see ``list_theories``,
    ``theory_rules`` and ``with_axioms``.
    """
    return kernel.prove_blocks(
        b1, b2, inputs=inputs, outputs=outputs, generalize=generalize, axioms=axioms
    )

def prove_expr(
    e1: Expr,
    e2: Expr,
    inputs: dict[str, Type] | None = None,
    *,
    axioms: tuple[str, ...] | None = None,
) -> Proof:
    """Prove two expressions are bit-identical for every input."""
    return kernel.prove_expr(e1, e2, inputs, axioms=axioms)

# Axiom curation -- consumer code chooses what a proof may draw on

def list_theories() -> list[str]:
    """Names of the loaded theory files (``arithmetic``, ``logic``, ...)."""
    return kernel.theories()

def theory_rules(name: str) -> list[Axiom]:
    """Every axiom contributed by one theory file."""
    return kernel.theory_rules(name)

def load_theory(name: str) -> list[Axiom]:
    """Load a single theory file into the live store (by name, no ``.py``)."""
    return kernel.load_theory(name)

@contextmanager

def with_axioms(*names: str) -> Iterator[None]:
    """Restrict every proof inside the block to exactly ``names``.

    >>> with opreof.with_axioms("add_self_is_shl1"):
    ...     opreof.prove_expr(a, b)          # only that fact is available
    """
    with kernel.scope(*names):
        yield

def new_kernel(
    theories: tuple[str, ...] | None = None,
    axioms_dir: str | Path | None = None,
) -> Kernel:
    """A fresh, isolated kernel -- its own store, so experiments cannot leak.

    With ``theories`` given, only those theory files are loaded; otherwise the
    whole ``axioms/`` directory is.  Use this to explore a curated proof
    search space (e.g. "what if the prover only knew about associativity?").
    """
    k = Kernel(axioms_dir=str(Path(axioms_dir) if axioms_dir is not None
                              else AXIOM_DIR))
    if theories:
        for name in theories:
            k.load_theory(name)
    else:
        k.axioms.load_dir(k.axioms_dir)
    return k

# E-graphs -- saturate the axiom store, then extract a minimal form

def prepare_egraph(
    terms: list[Expr] | None = None,
    inputs: dict[str, Type] | None = None,
    *,
    axioms: tuple[str, ...] | None = None,
    budget: int = 1 << 12,
) -> EGraphRunner:
    """Build a reusable knowledge base over ``terms`` (e.g. a whole program).

    The runner keeps its e-graph across ``add``/``saturate`` calls, so a
    consumer can feed it one function at a time and ask about the union --
    shared subterms are matched against the saturated graph instead of being
    re-normalized from scratch.
    """
    runner = EGraphRunner(rules=kernel._axiom_rules(axioms), budget=budget)
    for t in terms or []:
        runner.add(t, inputs)
    return runner

def egraph_normalize(
    e: Expr,
    inputs: dict[str, Type] | None = None,
    *,
    axioms: tuple[str, ...] | None = None,
    budget: int = 1 << 12,
) -> Expr:
    """Minimal representative of ``e`` up to the axiom store, laws included.

    Unlike ``normalize`` this honours *equational* rules (associativity,
    distributivity, xor decomposition) and returns the smallest member of the
    e-class rather than whatever a directed rewrite happened to reach.
    """
    return kernel.egraph_normalize(e, inputs, axioms=axioms, budget=budget)

def egraph_prove(
    e1: Expr,
    e2: Expr,
    inputs: dict[str, Type] | None = None,
    *,
    axioms: tuple[str, ...] | None = None,
    budget: int = 1 << 12,
    verify: bool = True,
) -> Proof:
    """Prove ``e1 == e2`` by saturating the axiom store undirected.

    With ``verify`` (the default) the extracted representatives are re-checked
    by concrete evaluation, so ``proven`` still means "bit-exact on every
    input" -- the e-graph may only *suggest* the proof, never assert it.
    """
    return kernel.egraph_prove(e1, e2, inputs, axioms=axioms, budget=budget,
                               verify=verify)

def unroll_verify(
    original: Block,
    unrolled: Block,
    factor: int,
    inputs: dict[str, Type] | None = None,
) -> Proof:
    """Verify that ``unrolled`` is a faithful unrolling of ``original``."""
    return kernel.unroll_verify(original, unrolled, factor, inputs)

def normalize(e: Expr) -> Expr:
    return kernel.normalize(e)

# Solving -- beyond equivalence checking

def solve_expr(e1: Expr, e2: Expr, inputs: dict[str, Type] | None = None, *,
               target: str | None = None) -> SolveResult:
    """Return the inputs on which ``e1`` and ``e2`` evaluate to equal bits.

    Status is ``always`` / ``some`` / ``never`` / ``rejected``; the exact
    satisfying set is in ``.satisfying`` (complete when ``.exhaustive``).
    """
    return _solve_expr(kernel, e1, e2, inputs, target=target)

solve = solve_expr

def solve_for(e1: Expr, e2: Expr, inputs: dict[str, Type] | None = None,
              var: str | None = None) -> SolveResult:
    """Isolate ``var``: produce a closed form (or the residual equation)."""
    return _solve_expr(kernel, e1, e2, inputs, target=var)

def find_witness(e1: Expr, e2: Expr, inputs: dict[str, Type] | None = None):
    """One input where the two sides agree (``None`` when they never do)."""
    return _find_witness(kernel, e1, e2, inputs)

def find_counterexample(e1: Expr, e2: Expr, inputs: dict[str, Type] | None = None):
    """One input where the two sides differ (``None`` when they always agree)."""
    return _find_counterexample(kernel, e1, e2, inputs)

def synthesize(e1: Expr, e2: Expr, inputs: dict[str, Type] | None = None, *,
               hole: str = "?") -> SolveResult:
    """Fill the hole ``hole`` with a term that makes ``e1 == e2`` on every input."""
    return _synthesize(kernel, e1, e2, inputs, hole=hole)

# Bridge: api.Term  <->  kernel.Expr / kernel.Type

_OP_NAMES = {
    "add", "sub", "mul", "neg", "and", "or", "xor", "not",
    "shl", "shru", "shrs", "divu", "divs", "remu", "rems",
    "eq", "ne", "ltu", "lteu", "gtu", "gteu", "lts", "ltes", "gts", "gtes",
    "extz", "exts", "trunc",
}

def resolve_term_type(term: Term) -> Type:
    """Resolve an ``api.Term`` to the kernel machine type it denotes.

    ``op.Term(int32)`` (equivalently ``parse_parms(int32)``) resolves to the
    kernel's ``int32``; a term wrapping the string ``"int32"`` resolves the
    same way.
    """
    if not isinstance(term, Term):
        raise KernelError(f"expected an api.Term, got {term!r}")
    definition = term.definition
    if isinstance(definition, Type):
        return definition
    if isinstance(definition, str):
        try:
            return parse_type(definition)
        except KernelError:
            pass
    if isinstance(definition, Term):
        return resolve_term_type(definition)
    if definition is None:
        try:
            return parse_type(term.name)
        except KernelError:
            pass
    raise TypeError_(f"term {term.name!r} does not denote a kernel type")

def expr_of_term(term: Term) -> Expr:
    """Translate an ``api.Term``/``api.Call`` tree into a kernel expression."""
    if not isinstance(term, Term):
        raise KernelError(f"expected an api.Term, got {term!r}")

    if isinstance(term, Literal):
        return Lit(term.value, term.type)

    if isinstance(term, Call):
        callee = term.callee
        op_name = callee.name.lower()
        if op_name in _OP_NAMES:
            args = tuple(expr_of_term(a) for a in term.args)
            if op_name in {"extz", "exts", "trunc"} and term.kwargs.get("to"):
                target = term.kwargs["to"]
                ty = target if isinstance(target, Type) else resolve_term_type(target)
                return Op(op_name, args, ty)
            return Op(op_name, args, None)
        raise TypeError_(f"unknown kernel operation {callee.name!r}")

    definition = term.definition
    if isinstance(definition, Type):
        raise TypeError_(
            f"term {term.name!r} is a type, not a value; wrap it in Var()"
        )
    if definition is None or isinstance(definition, str):
        return Var(term.name, None)
    raise TypeError_(f"cannot translate term {term!r} to a kernel expression")

def term_of_expr(expr: Expr) -> Term:
    """Translate a kernel expression back into the ``api.Term`` layer."""
    if isinstance(expr, (Var, Const)):
        return Term(name=expr.name, metadata={"kernel": "var"})
    if isinstance(expr, Lit):
        return Literal(expr.value, expr.type)
    if isinstance(expr, Op):
        callee = Term(name=expr.op)
        return Call(callee, tuple(term_of_expr(a) for a in expr.args))
    if isinstance(expr, Select):
        return Call(
            Term(name="select"),
            (term_of_expr(expr.cond), term_of_expr(expr.then_expr), term_of_expr(expr.else_expr)),
        )
    raise KernelError(f"cannot translate kernel expression {expr!r}")

__all__ = [  # noqa: RUF022  (grouped by layer, not alphabetical)
    # api layer
    "Term", "Call", "Literal", "parse_parms", "APIType", "_parse_parms",
    # kernel layer
    "Kernel", "Proof", "Block", "Assign", "Loop", "Nop",
    "Var", "Const", "Lit", "Op", "Select", "Expr", "Type",
    "BOOL", "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "parse_type",
    "ripen", "eval_c", "free_vars", "check_defined", "builtin_rewrite",
    "KernelError", "TypeError_", "DefinitionError", "OverFlowError",
    "UndefinedError",
    # axioms
    "Axiom", "Condition", "AxiomRegistry", "AxiomError", "axiom",
    "TRUE_COND", "cond_lit", "cond_lit_range", "cond_type",
    "register_axiom", "remove_axiom", "list_axioms", "get_axiom",
    "load_axioms", "list_theories", "theory_rules", "load_theory",
    "with_axioms", "new_kernel",
    # proofs
    "kernel", "prove", "prove_expr", "unroll_verify", "normalize",
    # e-graphs
    "EGraph", "EGraphRunner", "prepare_egraph",
    "egraph_normalize", "egraph_prove",
    # solving
    "SolveResult", "solve_expr", "solve", "solve_for",
    "find_witness", "find_counterexample", "synthesize",
    # bridge
    "resolve_term_type", "expr_of_term", "term_of_expr",
]

# Auto-load the axiom files in axioms/ (ordinary Python modules exporting
# an ``AXIOMS`` list).  This runs after every name above is bound so the
# files can ``from opreof import ...``.
load_axioms()
