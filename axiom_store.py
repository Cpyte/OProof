"""OProof axiom store.

Axioms are the *reusable* facts the kernel may invoke during normalization.
They are ordinary Python objects assembled in ordinary Python files; the
registry can add, remove, and reload them at run time.

An axiom rewrites an expression:

    lhs  (pattern, with placeholders)  ~>  rhs

A ``Var``/``Const`` node in ``lhs`` is a placeholder.  A ``Lit`` is an exact
literal (or a wildcard if ``value`` is ``None``).  Side conditions are
callables inspected per match, so a rule only fires when it is provably
applicable -- the kernel re-normalizes and re-checks every result, so an
unsound axiom can never manufacture a proof.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from kernel import (
    Const,
    Expr,
    Lit,
    Op,
    Select,
    Type,
    Var,
)


class AxiomError(Exception):
    pass

@dataclass(frozen=True)

class Condition:
    """A side condition: ``pred(subs, types) -> bool``.

    ``subs``  maps each bound placeholder name to the matched subexpression.
    ``types`` maps each bound placeholder name to its resolved kernel type.
    """

    pred: Callable[[Mapping[str, Expr], Mapping[str, Type]], bool] = field(
        default=lambda s, t: True
    )
    desc: str = ""

    def __call__(
        self, subs: Mapping[str, Expr], types: Mapping[str, Type]
    ) -> bool:
        return bool(self.pred(subs, types))

TRUE_COND = Condition(desc="always")

def cond_type(name: str, *types: Type) -> Condition:
    def pred(subs: Mapping[str, Expr], types_: Mapping[str, Type]) -> bool:
        t = types_.get(name)
        return t is not None and any(t == want for want in types)

    lst = ", ".join(t.name for t in types) or "*"
    return Condition(pred, f"placeholder {name!r} is of type {lst}")

def cond_lit(name: str) -> Condition:
    def pred(subs: Mapping[str, Expr], _types: Mapping[str, Type]) -> bool:
        e = subs.get(name)
        return isinstance(e, Lit)

    return Condition(pred, f"placeholder {name!r} is a literal")

def cond_lit_range(name: str, lo: int, hi: int) -> Condition:
    """Placeholder must be a literal value in ``[lo, hi)``."""

    def pred(subs: Mapping[str, Expr], _types: Mapping[str, Type]) -> bool:
        e = subs.get(name)
        return isinstance(e, Lit) and lo <= e.value < hi

    return Condition(pred, f"placeholder {name!r} is a literal in [{lo}, {hi})")

def _match_pat(
    pat: Expr,
    expr: Expr,
    subs: Mapping[str, Expr],
    types: Mapping[str, Type],
) -> tuple[Mapping[str, Expr], Mapping[str, Type]] | None:
    if isinstance(pat, (Var, Const)):
        if pat.type is not None and expr.type is not None and pat.type != expr.type:
            return None
        name = pat.name
        if name in subs:
            return (subs, types) if subs[name] == expr else None
        ns = dict(subs)
        ns[name] = expr
        nt = dict(types)
        nt[name] = expr.type
        return ns, nt

    if isinstance(pat, Lit):
        if not isinstance(expr, Lit):
            return None
        if pat.type is not None and expr.type is not None and pat.type != expr.type:
            return None
        if pat.type is not None and expr.type is None:
            return None
        if pat.value is not None and pat.value != expr.value:
            return None
        return subs, types

    if isinstance(pat, Op):
        if not isinstance(expr, Op) or pat.op != expr.op:
            return None
        if len(pat.args) != len(expr.args):
            return None
        cur = (subs, types)
        for p, a in zip(pat.args, expr.args):
            cur = _match_pat(p, a, cur[0], cur[1])
            if cur is None:
                return None
        return cur

    if isinstance(pat, Select):
        if not isinstance(expr, Select):
            return None
        cur = _match_pat(pat.cond, expr.cond, subs, types)
        if cur is None:
            return None
        cur = _match_pat(pat.then_expr, expr.then_expr, cur[0], cur[1])
        if cur is None:
            return None
        return _match_pat(pat.else_expr, expr.else_expr, cur[0], cur[1])

    return None

class Axiom:
    """A rewrite rule ``lhs ~> rhs`` guarded by side conditions.

    ``equational`` marks a rule that is a *law*, not a directed rewrite: it
    holds in both directions, so firing it one way may re-create the shape it
    matched (associativity fights the canonical operand sort, distributivity
    and xor-decomposition grow the term).  The normaliser skips such rules --
    they are available to the e-graph, which saturates them undirected and
    then extracts a minimal form.  Contractive laws (De Morgan, ``x & -1 == x``)
    stay directed because they strictly reduce the term.
    """

    __slots__ = ("conditions", "equational", "lhs", "name", "rhs", "source")

    def __init__(
        self,
        name: str,
        lhs: Expr,
        rhs: Expr,
        conditions: tuple[Condition, ...] = (TRUE_COND,),
        source: str = "user",
        equational: bool = False,
    ):
        self.name = name
        self.lhs = lhs
        self.rhs = rhs
        self.conditions = tuple(conditions or (TRUE_COND,))
        self.source = source
        self.equational = equational

    def match(self, target: Expr) -> Iterator[Mapping[str, Expr]]:
        """Yield each successful substitution; conditions must all hold."""
        got = _match_pat(self.lhs, target, {}, {})
        if got is None:
            return
        subs, types = got
        if all(c(subs, types) for c in self.conditions):
            yield {"subs": subs, "types": types}

    def rule_key(self) -> tuple:
        return (self.lhs.key(), self.rhs.key())

    def __repr__(self) -> str:
        conds = ",".join(c.desc for c in self.conditions if c.desc)
        return (
            f"Axiom({self.name!r}, {self.lhs.sugar()} == {self.rhs.sugar()}"
            f"{'| ' + conds if conds else ''}{', equational' if self.equational else ''})"
        )

def axiom(
    name: str,
    lhs: Expr,
    rhs: Expr,
    conditions: tuple[Condition, ...] = (TRUE_COND,),
    source: str = "user",
    equational: bool = False,
) -> Axiom:
    return Axiom(name, lhs, rhs, conditions, source, equational)

class AxiomRegistry:
    """Add / remove / query axioms; load them from ordinary Python files."""

    def __init__(self) -> None:
        self._axioms: dict[str, Axiom] = {}

    def register(self, ax: Axiom) -> bool:
        if ax.name in self._axioms:
            existing = self._axioms[ax.name]
            if existing.rule_key() == ax.rule_key():
                return False
            raise AxiomError(
                f"axiom name {ax.name!r} is already registered with a "
                f"different rule"
            )
        for cur in self._axioms.values():
            if cur.rule_key() == ax.rule_key():
                return False  # rule already installed under another name
        self._axioms[ax.name] = ax
        return True

    def remove(self, name: str) -> bool:
        return self._axioms.pop(name, None) is not None

    def get(self, name: str) -> Axiom | None:
        return self._axioms.get(name)

    def names(self) -> list[str]:
        return sorted(self._axioms)

    def rules(self) -> list[Axiom]:
        return list(self._axioms.values())

    def __len__(self) -> int:
        return len(self._axioms)

    def load_file(self, path: str | Path) -> list[Axiom]:
        """Import one axiom file and register every ``AXIOMS`` it exports."""
        path = Path(path)
        root = str(path.parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        modname = f"opreof_axiom_{abs(hash(str(path))) & 0xFFFFFFFF:x}"
        spec = importlib.util.spec_from_file_location(modname, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[modname] = module
        spec.loader.exec_module(module)
        raw = getattr(module, "AXIOMS", None)
        if raw is None:
            raw = getattr(module, "axioms", None)
        if raw is None:
            raise AxiomError(f"axiom file {path.name} exports no AXIOMS list")
        loaded: list[Axiom] = []
        for item in raw:
            if isinstance(item, Axiom):
                ax = item
            else:
                raise AxiomError(
                    f"axiom file {path.name} contains a non-Axiom entry: {item!r}"
                )
            self.register(ax)
            loaded.append(ax)
        return loaded

    def load_dir(self, dirpath: str | Path) -> list[Axiom]:
        """Load every ``*.py`` (except ``__init__``) from a directory."""
        dirpath = Path(dirpath)
        loaded: list[Axiom] = []
        for child in sorted(dirpath.iterdir()):
            if child.name == "__init__.py" or not child.name.endswith(".py"):
                continue
            loaded.extend(self.load_file(child))
        return loaded

    def export_source(self, dirpath: str | Path, ax: Axiom) -> Path:
        """Serialize ``ax`` into a new ordinary Python axiom file."""
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        idx = 0
        while True:
            cand = dirpath / f"auto_{idx:04d}.py"
            if not cand.exists():
                break
            idx += 1

        def _t_repr(t: Type) -> str:
            single = {
                "bool": "BOOL",
                "int8": "int8",
                "uint8": "uint8",
                "int16": "int16",
                "uint16": "uint16",
                "int32": "int32",
                "uint32": "uint32",
                "int64": "int64",
                "uint64": "uint64",
            }
            return single[t.name]

        def expr_repr(e: Expr) -> str:
            if isinstance(e, (Var, Const)):
                ty = "None" if e.type is None else _t_repr(e.type)
                cls = "Var" if isinstance(e, Var) else "Const"
                return f"{cls}({e.name!r}, {ty})"
            if isinstance(e, Lit):
                ty = "None" if e.type is None else _t_repr(e.type)
                return f"Lit({e.value!r}, {ty})"
            if isinstance(e, Op):
                args = ", ".join(expr_repr(a) for a in e.args)
                ty = "None" if e.type is None else _t_repr(e.type)
                return f"Op({e.op!r}, ({args},), {ty})"
            if isinstance(e, Select):
                return (
                    f"Select({expr_repr(e.cond)}, {expr_repr(e.then_expr)}, "
                    f"{expr_repr(e.else_expr)}, None)"
                )
            raise AxiomError(f"cannot serialize {e!r}")

        dy, dm, dd = __import__("datetime").date.today().timetuple()[0:3]
        lines = [
            "from opreof import Axiom, Op, Var, Const, Lit, Select",
            "from opreof import (",
            "    int8, uint8, int16, uint16, int32, uint32,",
            "    int64, uint64, BOOL,",
            ")",
            "",
            f"# Auto-derived axiom: {ax.name}",
            f"# source: {ax.source}   date: {dy:04d}-{dm:02d}-{dd:02d}",
            f"#   {ax.lhs.sugar()}  ~>  {ax.rhs.sugar()}",
            "AXIOMS = [",
            f"    Axiom({ax.name!r},",
            f"           {expr_repr(ax.lhs)},",
            f"           {expr_repr(ax.rhs)},",
            f"           source={ax.source!r},",
            "    ),",
            "]",
            "",
        ]
        out = cand
        out.write_text("\n".join(lines))
        return out
