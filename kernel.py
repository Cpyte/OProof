"""OProof kernel.

Sound proof-checking kernel for proving that two sequences of code are
EXACTLY the same (same types, same bit patterns, same definedness, for every
input).
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:  # axiom_store imports the kernel, so this is one-way
    from axiom_store import Axiom

# Canonical value invariant (nailed down BEFORE any rewrite rule)
#
# Every concrete value ``v`` of a fully typed ``Lit(v, t)`` is stored in the
# *canonical* form for ``t``:
#
#   * ``t.signed``   -> ``t.min_value <= v <= t.max_value``
#                      (two's-complement signed interpretation)
#   * ``t.signed`` F -> ``0 <= v < 2**t.width``
#                      (plain unsigned bit pattern)
#
# ``t.wrap(raw)`` brings any integer into canonical form and is the ONLY
# place bits are re-interpreted.  It follows from these semantics that:
#
#   * ``exts`` (sign-extension) preserves the signed value: on a literal it
#     is ``v -> v`` for signed destinations and ``v -> v mod 2**w`` for
#     unsigned destinations;
#   * ``extz`` (zero-extension) keeps the source unsigned bit pattern:
#     ``v -> v mod 2**srcw``;
#   * a right shift is *arithmetic* for signed types (``shrs``) and
#     *logical* for unsigned ones (``shru``) -- never interchangeable;
#   * every other operator wraps modulo ``2**t.width`` exactly like the bit
#     pattern of the underlying machine type.
#
# Literals are stored "adaptable" (``type is None``) until ripening binds
# them to a type; ripening then wraps them into canonical form.  Because
# canonical form is unique per bit pattern, ``key()`` equality between two
# sides of a proof is exactly bit-for-bit equality.
#
# Division semantics (an explicit decision, not an accident of Python):
#
#   ``divs``/``rems`` are *signed machine* division and remainder: for
#   negative operands they truncate toward zero (this is what C, RISC-V
#   DIV/REM, WASM i32.div_s/rem_s and x86 IDIV do), and the remainder takes
#   the sign of the dividend::
#
#       divs(-3, 2) == -1          # NOT floor division (that would be -2)
#       rems(-3, 2) == -1          # sign follows the dividend
#
#   An arithmetic right shift (``shrs``) is *floor* division by ``2**k``
#   (two's complement), so ``divs(x, 2**k) == shrs(x, k)`` holds only for
#   ``x >= 0``.  The identity sound for every ``x`` is the compiler's
#   sign-corrected strength reduction, promoted in ``axioms/pow2.py``.

class KernelError(Exception):
    """Base error raised for ill-formed programs or proof steps."""

    kind = "error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class TypeError_(KernelError):
    kind = "type"

class DefinitionError(KernelError):
    kind = "definition"

class OverFlowError(KernelError):
    kind = "overflow"

class UndefinedError(KernelError):
    kind = "undefined"

# Data types -- "data types control and resolvance"

class Type:
    """A fixed-width scalar machine type (two's complement)."""

    __slots__ = ("name", "signed", "width")

    def __init__(self, name: str, width: int, signed: bool):
        self.name = name
        self.width = width
        self.signed = signed

    @property
    def modulus(self) -> int:
        return 1 << self.width

    @property
    def mask(self) -> int:
        return self.modulus - 1

    @property
    def min_value(self) -> int:
        return -(1 << (self.width - 1)) if self.signed else 0

    @property
    def max_value(self) -> int:
        if self.signed:
            return (1 << (self.width - 1)) - 1
        return self.modulus - 1

    def wrap(self, raw: int) -> int:
        v = raw % self.modulus
        if self.signed and v >= (1 << (self.width - 1)):
            v -= self.modulus
        return v

    def fits(self, raw: int) -> bool:
        return self.min_value <= raw <= self.max_value

    def is_bool(self) -> bool:
        return self.width == 1 and not self.signed

    def __repr__(self) -> str:
        return self.name

    __str__ = __repr__

    def __hash__(self) -> int:
        return hash((self.name, self.width, self.signed))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Type)
            and self.name == other.name
            and self.width == other.width
            and self.signed == other.signed
        )

BOOL = Type("bool", 1, False)
int8 = INT8 = Type("int8", 8, True)
uint8 = UINT8 = Type("uint8", 8, False)
int16 = INT16 = Type("int16", 16, True)
uint16 = UINT16 = Type("uint16", 16, False)
int32 = INT32 = Type("int32", 32, True)
uint32 = UINT32 = Type("uint32", 32, False)
int64 = INT64 = Type("int64", 64, True)
uint64 = UINT64 = Type("uint64", 64, False)

_TYPES_BY_NAME: dict[str, Type] = {
    t.name: t
    for t in (BOOL, int8, uint8, int16, uint16, int32, uint32, int64, uint64)
}

_ACODE_IMPORTS = {
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

def parse_type(name: str) -> Type:
    try:
        return _TYPES_BY_NAME[name]
    except KeyError:
        raise TypeError_(
            f"unknown type {name!r}; known types: {', '.join(_TYPES_BY_NAME)}"
        )

# Expressions

class Expr:
    """Immutable symbolic expression node carrying its resolved type."""

    __slots__ = ("type",)

    def __init__(self, ty: Type | None = None):
        self.type = ty

    def key(self) -> tuple:
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self.key())

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Expr)
            and type(self) is type(other)
            and self.key() == other.key()
        )

    def __repr__(self) -> str:
        return self.sugar()

    def sugar(self) -> str:
        raise NotImplementedError

class Var(Expr):
    """A variable. Doubles as the pattern placeholder inside axiom LHS."""

    __slots__ = ("name",)

    def __init__(self, name: str, ty: Type | None = None):
        super().__init__(ty)
        self.name = name

    def key(self):
        t = self.type
        return ("v", self.name, t.name if t else None)

    def sugar(self) -> str:
        return self.name

class Const(Expr):
    """A named symbolic constant (never substituted, part of the input)."""

    __slots__ = ("name",)

    def __init__(self, name: str, ty: Type | None = None):
        super().__init__(ty)
        self.name = name

    def key(self):
        t = self.type
        return ("c", self.name, t.name if t else None)

    def sugar(self) -> str:
        return self.name

class Lit(Expr):
    """An integer literal.

    ``type`` may be ``None`` while the literal is still "adaptable": its type
    (and so its wrap) is resolved from context during ripening.
    """

    __slots__ = ("value",)

    def __init__(self, value: int, ty: Type | None = None):
        super().__init__(ty)
        self.value = int(value)

    def key(self):
        t = self.type
        return ("lit", self.value, t.name if t else None)

    def sugar(self) -> str:
        return str(self.value)

class Op(Expr):
    __slots__ = ("args", "op")

    def __init__(self, op: str, args: tuple[Expr, ...], ty: Type | None = None):
        super().__init__(ty)
        self.op = op
        self.args = tuple(args)

    def key(self):
        t = self.type
        return ("op", self.op, tuple(a.key() for a in self.args), t.name if t else None)

    def sugar(self) -> str:
        return f"{self.op}({', '.join(a.sugar() for a in self.args)})"

class Select(Expr):
    __slots__ = ("cond", "else_expr", "then_expr")

    def __init__(
        self,
        cond: Expr,
        then_expr: Expr,
        else_expr: Expr,
        ty: Type | None = None,
    ):
        super().__init__(ty)
        self.cond = cond
        self.then_expr = then_expr
        self.else_expr = else_expr

    def key(self):
        t = self.type
        return (
            "sel",
            self.cond.key(),
            self.then_expr.key(),
            self.else_expr.key(),
            t.name if t else None,
        )

    def sugar(self) -> str:
        return (
            f"sel({self.cond.sugar()} ? {self.then_expr.sugar()}"
            f" : {self.else_expr.sugar()})"
        )

# Statements / blocks -- "two sequences of code"

class Stmt:
    __slots__ = ()

class Assign(Stmt):
    __slots__ = ("rhs", "var")

    def __init__(self, var: Var, rhs: Expr):
        if not isinstance(var, Var) or var.type is None:
            raise TypeError_("assignment target must be a typed variable")
        self.var = var
        self.rhs = rhs

    def __repr__(self) -> str:
        return f"{self.var.name} = {self.rhs.sugar()}"

class Loop(Stmt):
    """Structured loop with a statically known or shared symbolic bound."""

    __slots__ = ("body", "count")

    def __init__(self, count: int | Expr, body: list[Stmt]):
        if isinstance(count, int) and count < 0:
            raise TypeError_("loop bound must be >= 0")
        self.count = count
        self.body = list(body)

    def __repr__(self) -> str:
        return f"loop({self.count}) {{ {self.body} }}"

class Nop(Stmt):
    __slots__ = ()

    def __repr__(self) -> str:
        return "nop"

class Block:
    __slots__ = ("name", "stmts")

    def __init__(self, name: str = "block", stmts: list[Stmt] | None = None):
        self.name = name
        self.stmts = list(stmts or [])

    def __repr__(self) -> str:
        return f"Block({self.name!r}, {self.stmts})"

# Ripening -- type resolution ("resolvance") with literal adaptation

_DEFAULT_INT = int32

_BINARY_INT_OPS = {"add", "sub", "mul", "and", "or", "xor"}
_SHIFT_OPS = {"shl", "shru", "shrs"}
_DIV_OPS = {"divu", "divs", "remu", "rems"}
_CMP_OPS = {
    "eq", "ne",
    "ltu", "lteu", "gtu", "gteu",
    "lts", "ltes", "gts", "gtes",
}
_UNARY_OPS = {"neg", "not"}
_CONV_OPS = {"extz", "exts", "trunc"}

_ALL_OPS = (
    _BINARY_INT_OPS | _SHIFT_OPS | _DIV_OPS | _CMP_OPS | _UNARY_OPS | _CONV_OPS
)

_TRUE = Op("eq", (Lit(1), Lit(1)), BOOL)
_FALSE = Op("eq", (Lit(1), Lit(0)), BOOL)

def _truth(v: int) -> bool:
    return v != 0

def _resolve_literal(raw: int, ty: Type) -> int:
    return ty.wrap(raw)

def ripen(
    expr: Expr,
    expected: Type | None = None,
    *,
    top: bool = False,
    env: Mapping[str, Any] | None = None,
) -> Expr:
    """Fully type-resolve an expression (bottom-up) into a typed tree.

    Adaptable literals adopt ``expected`` (propagated top-down); where that is
    unknown the sibling operand or ``int32`` decides.  When ``top`` is set the
    outermost literal must be representable in its type -- this is the
    "data types control" guard that rejects silent narrowing on assignment.
    """
    if isinstance(expr, (Var, Const)):
        if env is not None and expr.name in env:
            return env[expr.name][0]
        ty = expr.type or expected or _DEFAULT_INT
        if expr.type is not None:
            return expr
        # adaptable variable: adopt context (input/expected type, else int32)
        return type(expr)(expr.name, ty)

    if isinstance(expr, Lit):
        if expr.type is not None:
            return expr
        ty = expected or _DEFAULT_INT
        wrapped = _resolve_literal(expr.value, ty)
        if top and not ty.fits(expr.value):
            raise OverFlowError(
                f"literal {expr.value} is not representable in {ty.name}"
            )
        return Lit(wrapped, ty)

    if isinstance(expr, Select):
        cond = ripen(expr.cond, BOOL, env=env)
        if cond.type is not None and not cond.type.is_bool():
            raise TypeError_("select condition must be of type bool")
        if isinstance(cond, Lit) and cond.type is not None:
            cond = Lit(_truth(cond.value), BOOL)
        a = ripen(expr.then_expr, expected, env=env)
        b = ripen(expr.else_expr, expected, env=env)
        _require_same_type(a, b, "select branches")
        return Select(cond, a, b, a.type)

    if isinstance(expr, Op):
        return _ripen_op(expr, expected, top=top, env=env)

    raise TypeError_(f"cannot ripen {expr!r}")

def _require_same_type(a: Expr, b: Expr, what: str) -> None:
    ta, tb = a.type, b.type
    if ta is not None and tb is not None and ta != tb:
        raise TypeError_(f"{what} have different types: {ta} vs {tb}")

def _effective_type(arg: Expr, env: Mapping[str, Any] | None) -> Type | None:
    if arg.type is not None:
        return arg.type
    if isinstance(arg, (Var, Const)) and env is not None and arg.name in env:
        return env[arg.name][1]
    if isinstance(arg, Op):
        # a compound expression built raw carries no type yet; look inside
        # (e.g. shrs(add(x, ...), k) must see that x is int8, not default
        # to int32 and then collide with the env binding)
        for a in arg.args:
            t = _effective_type(a, env)
            if t is not None:
                return t
    return None

def _pick_type(
    *args: Expr,
    expected: Type | None,
    what: str,
    env: Mapping[str, Any] | None = None,
) -> Type:
    if expected is not None:
        return expected
    for a in args:
        t = _effective_type(a, env)
        if t is not None:
            return t
    return _DEFAULT_INT

def _ripen_op(expr: Op, expected: Type | None, *, top: bool, env) -> Expr:
    op = expr.op
    if op not in _ALL_OPS:
        raise TypeError_(f"unknown operation {op!r}")

    if op in _BINARY_INT_OPS:
        args = list(expr.args)
        if len(args) != 2:
            raise TypeError_(f"{op} takes exactly two arguments")
        ty = _pick_type(args[0], args[1], expected=expected, what=op, env=env)
        a = ripen(args[0], ty, env=env)
        b = ripen(args[1], ty, env=env)
        _require_same_type(a, b, f"operands of {op}")
        return Op(op, (a, b), ty)

    if op in _SHIFT_OPS:
        lhs, amount = expr.args
        ty = _pick_type(lhs, expected=expected, what=op, env=env)
        ra = ripen(lhs, ty, env=env)
        rn = ripen(amount, int32, env=env)
        if rn.type is not None and rn.type.is_bool():
            raise TypeError_(f"shift amount for {op} must be an integer, not bool")
        return Op(op, (ra, rn), ty)

    if op in _DIV_OPS:
        args = list(expr.args)
        ty = _pick_type(args[0], args[1], expected=expected, what=op, env=env)
        a = ripen(args[0], ty, env=env)
        b = ripen(args[1], ty, env=env)
        _require_same_type(a, b, f"operands of {op}")
        return Op(op, (a, b), ty)

    if op in _CMP_OPS:
        a = ripen(expr.args[0], expected, env=env)
        b = ripen(expr.args[1], expected, env=env)
        _require_same_type(a, b, f"operands of {op}")
        return Op(op, (a, b), BOOL)

    if op in _UNARY_OPS:
        a = ripen(expr.args[0], expected, env=env)
        if op == "neg" and a.type is not None and a.type.is_bool():
            raise TypeError_("neg is not defined on bool")
        if op == "not" and a.type is not None and a.type.is_bool():
            return Op("not", (a,), a.type)
        return Op(op, (a,), a.type)

    if op in _CONV_OPS:
        a = ripen(expr.args[0], None, env=env)
        ty = expr.type or expected or a.type or _DEFAULT_INT
        if a.type is not None and a.type.width > ty.width:
            raise TypeError_(
                f"{op} cannot widen {a.type.name} to the smaller {ty.name}"
            )
        return Op(op, (a,), ty)

    raise TypeError_(f"cannot ripen {op}")

# Concrete evaluation (bit-for-bit, wrapping to the operand type)

def eval_c(expr: Expr) -> int:
    """Evaluate a fully ripened expression over exact two's-complement values."""
    if isinstance(expr, Lit):
        return expr.value
    if isinstance(expr, (Var, Const)):
        raise TypeError_(f"cannot evaluate symbolic {expr.sugar()!r}")
    if isinstance(expr, Select):
        return eval_c(expr.then_expr) if _truth(eval_c(expr.cond)) else eval_c(expr.else_expr)
    if not isinstance(expr, Op):
        raise TypeError_(f"cannot evaluate {expr!r}")
    op, args = expr.op, expr.args
    ty = expr.type
    vals = [eval_c(a) for a in args]
    x, y = (vals[0], vals[1]) if len(vals) == 2 else (vals[0], None)

    def total(v: int) -> int:
        return v if (ty is not None and ty.signed) else (v % ty.modulus if ty else v)

    if op == "add":
        r = x + y
    elif op == "sub":
        r = x - y
    elif op == "mul":
        r = x * y
    elif op == "and":
        r = x & y
    elif op == "or":
        r = x | y
    elif op == "xor":
        r = x ^ y
    elif op == "neg":
        r = -x
    elif op == "not":
        r = ~x
    elif op == "shl":
        r = x << y
    elif op == "shru":
        r = (x % ty.modulus) >> y
    elif op == "shrs":
        r = x >> y
    elif op == "divu":
        r = (x % ty.modulus) // (y % ty.modulus)
    elif op == "divs":
        # signed division truncates toward zero (C/RISC-V/WASM/x86), NOT
        # Python's floor division
        q = abs(x) // abs(y)
        r = -q if (x < 0) != (y < 0) else q
    elif op == "remu":
        r = (x % ty.modulus) % (y % ty.modulus)
    elif op == "rems":
        # signed remainder takes the sign of the dividend: x - y*divs(x,y)
        q = abs(x) // abs(y)
        if (x < 0) != (y < 0):
            q = -q
        r = x - y * q
    elif op == "eq":
        return int(total(x) == total(y))
    elif op == "ne":
        return int(total(x) != total(y))
    elif op == "ltu":
        return int((x % ty.modulus) < (y % ty.modulus))
    elif op == "lteu":
        return int((x % ty.modulus) <= (y % ty.modulus))
    elif op == "gtu":
        return int((x % ty.modulus) > (y % ty.modulus))
    elif op == "gteu":
        return int((x % ty.modulus) >= (y % ty.modulus))
    elif op == "lts":
        return int(x < y)
    elif op == "ltes":
        return int(x <= y)
    elif op == "gts":
        return int(x > y)
    elif op == "gtes":
        return int(x >= y)
    elif op == "extz":
        return x % (1 << args[0].type.width)
    elif op == "exts":
        return x
    elif op == "trunc":
        return x % (1 << ty.width)
    else:
        raise TypeError_(f"cannot evaluate {op}")

    return ty.wrap(r)

def free_vars(expr: Expr) -> set[str]:
    got: set[str] = set()
    _collect_vars(expr, got)
    return got

def _collect_vars(expr: Expr, got: set[str]) -> None:
    if isinstance(expr, (Var, Const)):
        got.add(expr.name)
    elif isinstance(expr, Op):
        for a in expr.args:
            _collect_vars(a, got)
    elif isinstance(expr, Select):
        _collect_vars(expr.cond, got)
        _collect_vars(expr.then_expr, got)
        _collect_vars(expr.else_expr, got)

def substitute(expr: Expr, subs: Mapping[str, Expr]) -> Expr:
    if isinstance(expr, (Var, Const)):
        return subs.get(expr.name, expr)
    if isinstance(expr, Lit):
        return expr
    if isinstance(expr, Op):
        return Op(expr.op, tuple(substitute(a, subs) for a in expr.args), expr.type)
    if isinstance(expr, Select):
        return Select(
            substitute(expr.cond, subs),
            substitute(expr.then_expr, subs),
            substitute(expr.else_expr, subs),
            expr.type,
        )
    return expr

# Definedness / "proper bit shifts check"

def check_defined(expr: Expr) -> list[tuple[str, str]]:
    """Return (severity, message) issues: ('reject', ...) or ('cannot', ...).

    - ``shl``/``shru``/``shrs``: a literal shift amount outside ``[0, width)``
      is definitely undefined  -> ``reject``;  a symbolic amount cannot be
      verified  -> ``cannot``.
    - division/remainder by a literal zero is definitely undefined -> reject;
      a symbolic divisor cannot be verified -> cannot.
    """
    issues: list[tuple[str, str]] = []

    def walk(e: Expr) -> None:
        if isinstance(e, Op):
            if e.op in _SHIFT_OPS:
                lhs, n = e.args
                width = lhs.type.width if lhs.type else 0
                if isinstance(n, Lit):
                    v = n.value
                    if not (0 <= v < width):
                        issues.append(
                            ("reject",
                             f"{e.op} by {v} on {width}-bit value is undefined"))
                else:
                    issues.append(
                        ("cannot",
                         f"{e.op} amount is symbolic, range not verified"))
            elif e.op in _DIV_OPS:
                _, d = e.args
                if isinstance(d, Lit) and d.value == 0:
                    issues.append(("reject", f"division by literal zero in {e.op}"))
                elif not isinstance(d, Lit):
                    issues.append(
                        ("cannot", f"divisor of {e.op} is symbolic, may be zero"))
            for a in e.args:
                walk(a)
        elif isinstance(e, Select):
            walk(e.cond)
            walk(e.then_expr)
            walk(e.else_expr)

    walk(expr)
    return issues

# Normalization -- sound algebra + registered axioms

_ACOMM = {"add", "mul", "and", "or", "xor"}

def builtin_rewrite(e: Expr) -> Expr | None:
    """One sound algebraic rewrite at the root (args already normalized).

    Every rule here is provably sound under two's-complement wraparound of
    the operand types; no rule hides undefined behaviour.
    """
    if not isinstance(e, Op):
        return None
    op, ty = e.op, e.type
    args = e.args

    if op in _ACOMM and len(args) == 2:
        a, b = args
        if isinstance(a, Lit) and isinstance(b, Lit) and ty is not None:
            return Lit(_fold(op, a.value, b.value, ty), ty)
        if a.key() > b.key():
            return Op(op, (b, a), ty)
        if op == "add":
            if isinstance(b, Lit) and b.value == 0:
                return a
            if isinstance(a, Lit) and a.value == 0:
                return b
            if isinstance(a, Op) and a.op == "neg" and b == a.args[0]:
                return Lit(0, ty)
            if isinstance(b, Op) and b.op == "neg" and a == b.args[0]:
                return Lit(0, ty)
        if op == "mul":
            if isinstance(b, Lit) and b.value == 1:
                return a
            if isinstance(b, Lit) and b.value == 0:
                return Lit(0, ty)
        if op == "and":
            if isinstance(b, Lit) and b.value == 0:
                return Lit(0, ty)
            if isinstance(b, Lit) and b.value == ty.mask:
                return a
            if a == b:
                return a
        if op == "or":
            if isinstance(b, Lit) and b.value == 0:
                return a
            if isinstance(b, Lit) and b.value == ty.mask:
                return b
            if a == b:
                return a
        if op == "xor":
            if isinstance(b, Lit) and b.value == 0:
                return a
            if a == b:
                return Lit(0, ty)
        return None

    if op in {"add", "mul", "and", "or", "xor"}:
        return None

    if op == "sub":
        a, b = args
        if isinstance(a, Lit) and isinstance(b, Lit) and ty is not None:
            return Lit(_fold("sub", a.value, b.value, ty), ty)
        if isinstance(b, Lit) and b.value == 0:
            return a
        if a == b:
            return Lit(0, ty)
        if isinstance(a, Lit) and a.value == 0 and ty is not None:
            return Op("neg", (b,), ty)
        return None

    if op == "neg":
        (a,) = args
        if isinstance(a, Lit) and ty is not None:
            return Lit(ty.wrap(-a.value), ty)
        if isinstance(a, Op) and a.op == "neg":
            return a.args[0]
        return None

    if op == "not":
        (a,) = args
        if isinstance(a, Lit) and ty is not None:
            return Lit(ty.wrap(~a.value), ty)
        if isinstance(a, Op) and a.op == "not":
            return a.args[0]
        return None

    if op in _SHIFT_OPS:
        x, n = args
        if isinstance(n, Lit):
            sh = n.value
            width = x.type.width if x.type else 0
            if sh == 0:
                return x
            if not (0 <= sh < width):
                raise UndefinedError(
                    f"shift {op} by {sh} on {width}-bit value is undefined")
            if op == "shl":
                # x << n == x * 2**n  (mod 2**w) holds unconditionally here.
                return Op("mul", (x, Lit(x.type.wrap(pow(2, sh)), x.type)), x.type)
            if isinstance(x, Lit):
                return Lit(_fold(op, x.value, sh, x.type), x.type)
        return None

    if op in _DIV_OPS:
        a, b = args
        if isinstance(a, Lit) and isinstance(b, Lit) and ty is not None:
            if b.value == 0:
                raise UndefinedError("division by literal zero")
            return Lit(_fold(op, a.value, b.value, ty), ty)
        if isinstance(b, Lit):
            if b.value == 1:
                return a
            if op in {"remu", "rems"} and b.value == 1:
                return Lit(0, ty)
        return None

    if op in _CMP_OPS:
        a, b = args
        if isinstance(a, Lit) and isinstance(b, Lit) and ty is not None:
            t = a.type
            va = a.value % t.modulus if not t.signed else a.value
            vb = b.value % t.modulus if not t.signed else b.value
            res = _cmp(op, va, vb, t)
            return Lit(int(res), BOOL)
        if op == "eq" and a == b:
            return Lit(1, BOOL)
        if op == "ne" and a == b:
            return Lit(0, BOOL)
        return None

    if op in _CONV_OPS:
        (a,) = args
        if isinstance(a, Lit) and ty is not None:
            if op == "extz":
                # zero-extension: keep the source's unsigned bit pattern,
                # widened into the destination type.
                return Lit(a.value % (1 << a.type.width), ty)
            if op == "exts":
                if ty.signed:
                    # sign-extension preserves the signed value.
                    return Lit(a.value, ty)
                return Lit(a.value % (1 << ty.width), ty)
            return Lit(a.value % (1 << ty.width), ty)
        if a.type == ty:
            return a
        return None

    return None

def _fold(op: str, x: int, y: int, ty: Type) -> int:
    """Fold a literal application to a canonical machine value.

    Every result is wrapped into the operand type's canonical range (for
    signed types that is ``[-2**(w-1), 2**(w-1)-1]``), and ``divs``/``rems``
    use the kernel's explicit machine semantics (truncate toward zero,
    remainder sign follows the dividend) -- never Python's floor behavior.
    """
    if op == "add":
        r = x + y
    elif op == "sub":
        r = x - y
    elif op == "mul":
        r = x * y
    elif op == "and":
        r = x & y
    elif op == "or":
        r = x | y
    elif op == "xor":
        r = x ^ y
    elif op == "shru":
        r = (x % ty.modulus) >> y
    elif op == "shrs":
        r = x >> y
    elif op == "shl":
        r = x << y
    elif op == "divu":
        r = (x % ty.modulus) // (y % ty.modulus)
    elif op == "divs":
        q = abs(x) // abs(y)
        r = -q if (x < 0) != (y < 0) else q
    elif op == "remu":
        r = (x % ty.modulus) % (y % ty.modulus)
    elif op == "rems":
        q = abs(x) // abs(y)
        if (x < 0) != (y < 0):
            q = -q
        r = x - y * q
    else:
        raise TypeError_(f"unexpected fold {op}")
    return ty.wrap(r)

def _cmp(op: str, x: int, y: int, t: Type) -> bool:
    if op == "eq":
        return x == y
    if op == "ne":
        return x != y
    if not t.signed:
        if op == "ltu":
            return x < y
        if op == "lteu":
            return x <= y
        if op == "gtu":
            return x > y
        if op == "gteu":
            return x >= y
    else:
        if op == "lts":
            return x < y
        if op == "ltes":
            return x <= y
        if op == "gts":
            return x > y
        if op == "gtes":
            return x >= y
    raise TypeError_(f"mismatched comparison {op} for {t.name}")

# Proof result

@dataclass

class Proof:
    """Outcome of a kernel check.

    ``status`` is one of ``"proven"``, ``"not_proven"``, ``"rejected"``.

    * ``proven``    -- asserted only when the kernel has an exact argument.
    * ``not_proven``-- the kernel could not establish the claim; never an
                       equality, so it can never corrupt a proof chain.
    * ``rejected``  -- the claim is ill-formed (type error, UB, unrepresentable
                       literal, undefined read, ...).  Rejection is final.
    """

    status: str = "not_proven"
    kind: str = "equivalence"
    reason: str = ""
    lhs: Any = None
    rhs: Any = None
    axioms_used: tuple = ()
    derived_axioms: tuple = ()
    #: How exactly the ``axioms_used`` were checked: ``"complete"`` (every
    #: domain decided, bit-exact), ``"sampled"`` (the axioms were spot-checked
    #: but the domain is too large to enumerate) or ``None`` (no axiom was
    #: consulted, or the caller opted out of checking).
    checked: str | None = None

    def __bool__(self) -> bool:
        return self.status == "proven"

    def __repr__(self) -> str:
        return (
            f"<OProof {self.kind} {self.status}"
            + (f": {self.reason}" if self.reason else "")
            + ">"
        )

PROVEN = Proof("proven")

# Symbolic execution of blocks

class _SymbolicLoop(Exception):
    """Raised internally when a loop's trip count is not statically known."""

class _Cell(NamedTuple):
    """One variable's value during symbolic execution.

    ``norm`` is the normalized (canonical) expression, ``ty`` its type, and
    ``raw`` the pre-normalization ripened expression (kept so generalization
    can lift the concrete constants the programmer actually wrote).
    """

    norm: Expr
    ty: Type
    raw: Expr

State = dict[str, _Cell]

def _static_int(v: Any) -> int | None:
    if isinstance(v, int):
        return v
    if isinstance(v, Lit):
        return v.value
    return None

def _interface(block: Block) -> dict[str, Type]:
    """Read-before-write variables are the interface (inputs) of a block."""
    inputs: dict[str, Type] = {}
    written: set[str] = set()

    def scan_expr(e: Expr) -> None:
        if isinstance(e, (Var, Const)):
            if e.name not in written and e.name not in inputs and e.type is not None:
                inputs[e.name] = e.type
        elif isinstance(e, Op):
            for a in e.args:
                scan_expr(a)
        elif isinstance(e, Select):
            scan_expr(e.cond)
            scan_expr(e.then_expr)
            scan_expr(e.else_expr)

    def scan(stmts: list[Stmt]) -> None:
        for st in stmts:
            if isinstance(st, Assign):
                scan_expr(st.rhs)
                written.add(st.var.name)
            elif isinstance(st, Loop):
                if isinstance(st.count, Expr):
                    scan_expr(st.count)
                scan(st.body)
            elif isinstance(st, Nop):
                pass

    scan(block.stmts)
    return inputs

def _writes(block: Block) -> set[str]:
    got: set[str] = set()

    def scan(stmts: list[Stmt]) -> None:
        for st in stmts:
            if isinstance(st, Assign):
                got.add(st.var.name)
            elif isinstance(st, Loop):
                scan(st.body)

    scan(block.stmts)
    return got

def _cc_expr(e: Expr | None) -> str:
    if e is None:
        return ""
    return repr(e.sugar())

def _exec_stmt(st: Stmt, state: State, kernel: Kernel) -> State:
    if isinstance(st, Assign):
        name, ty = st.var.name, st.var.type
        rhs = ripen(st.rhs, ty, top=True, env=state)
        if rhs.type is not None and rhs.type != ty:
            raise TypeError_(
                f"assignment to {name}: expression resolves to {rhs.type}, "
                f"expected {ty}"
            )
        state = dict(state)
        state[name] = _Cell(kernel.normalize(rhs), ty, rhs)
        return state

    if isinstance(st, Nop):
        return state

    if isinstance(st, Loop):
        n = _static_int(st.count)
        if n is None:
            raise _SymbolicLoop()
        for _ in range(n):
            for inner in st.body:
                state = _exec_stmt(inner, state, kernel)
        return state

    raise TypeError_(f"unsupported statement {st!r}")

def _exec_block(block: Block, inputs: dict[str, Type], kernel: Kernel) -> State:
    state: State = {
        name: _Cell(Var(name, ty), ty, Var(name, ty)) for name, ty in inputs.items()
    }
    for st in block.stmts:
        state = _exec_stmt(st, state, kernel)
    return state

# The kernel

_AXIS_BUDGET = 200
_EGRAPH_BUDGET = 1 << 12
_NORMALIZE_DEPTH = 64

_TYPE_FILL_OPS = _BINARY_INT_OPS | _SHIFT_OPS | _DIV_OPS | {"neg", "not"}

def _retag(e: Expr) -> Expr:
    """Canonical type recovery over a normalised tree.

    Rewrites (in particular axiom substitution) can leave ops *and* their
    literal leaves with ``None`` types where the pattern ops were untyped.
    The normal form must be independent of how the types were written, so:

    * arithmetic ops inherit the first operand's (recovered) type;
    * a ``None``-typed literal leaf adopts the type ``ripen`` would have
      given it -- the sibling operand's type, or ``int32`` for a shift
      amount;
    * conversions and ``Select`` keep their own explicit target.
    """
    if isinstance(e, Op):
        args = tuple(_retag(a) for a in e.args)
        ty = e.type
        if ty is None and e.op in _TYPE_FILL_OPS and args:
            ty = args[0].type
        new_args = list(args)
        for i, a in enumerate(args):
            if isinstance(a, Lit) and a.type is None:
                want = None
                sibling = args[1 - i] if len(args) == 2 else None
                if e.op in _SHIFT_OPS and i == 1:
                    want = int32
                elif sibling is not None and sibling.type is not None:
                    want = sibling.type
                elif ty is not None:
                    want = ty
                if want is not None:
                    new_args[i] = Lit(a.value, want)
        return Op(e.op, tuple(new_args), ty)
    if isinstance(e, Select):
        return Select(_retag(e.cond), _retag(e.then_expr), _retag(e.else_expr), e.type)
    return e

class Kernel:
    """The OProof proof-checking kernel.

    A single instance owns one axiom store; every proof step is audited by
    :meth:`normalize`, which combines the provably-sound built-in algebra with
    the registered axioms (loaded from ordinary Python files, add/remove).
    """

    def __init__(self, axioms_dir: str | None = None) -> None:
        from axiom_store import AxiomRegistry

        self.axioms = AxiomRegistry()
        self.axioms_dir = axioms_dir
        self._axiom_uses: set[str] = set()
        self._enabled: set[str] | None = None

    @contextmanager
    def _axiom_trace(self) -> Iterator[set[str]]:
        saved = self._axiom_uses
        self._axiom_uses = set()
        try:
            yield self._axiom_uses
        finally:
            self._axiom_uses = saved

    @contextmanager
    def scope(self, *names: str) -> Iterator[None]:
        """Temporarily restrict normalization to exactly the named axioms.

        ``with kernel.scope("add_self_is_shl1", ...): prove_expr(...)`` proves
        with a curated rule set, not the whole store -- consumer code decides
        which facts a proof may draw on.  Nested scopes restore the outer one.
        """
        saved = self._enabled
        try:
            self._enabled = set(names)
            yield
        finally:
            self._enabled = saved

    def _axiom_rules(self, names: list[str] | tuple[str, ...] | None):
        """The registered rules a proof may use, curated by ``names``.

        ``None`` selects the whole store (minus any enclosing ``scope``);
        unknown names are rejected rather than silently ignored.
        """
        rules = self.axioms.rules()
        if names:
            wanted = set(names)
            missing = wanted - {a.name for a in self.axioms.rules()}
            if missing:
                raise KernelError(
                    f"unknown axiom(s): {', '.join(sorted(missing))}"
                )
            rules = [a for a in rules if a.name in wanted]
        if self._enabled is not None:
            rules = [a for a in rules if a.name in self._enabled]
        return rules

    def normalize(
        self,
        e: Expr,
        _seen: set[tuple] | None = None,
        _depth: int = 0,
    ) -> Expr:
        """Reduce ``e`` to a canonical normal form.

        Sound by construction: only bit-exact modular-algebra rules and the
        registered axioms fire, every axiom application is re-normalized and
        re-checked, and rewrite cycles are detected and abandoned.  Rules
        marked ``equational`` (associativity, distributivity, xor
        decomposition) are *not* fired here -- they hold in both directions and
        would fight the canonical operand sort -- so they are reachable only
        through ``egraph_normalize``/``egraph_prove``, which saturate them
        undirected.  ``_depth`` bounds the rewrite descent so a hostile rule
        set can waste time but never blow the stack.
        """
        if _seen is None:
            _seen = set()
        if _depth > _NORMALIZE_DEPTH:
            return e
        e = _retag(e)

        if isinstance(e, Op):
            args = tuple(self.normalize(a, _seen, _depth + 1) for a in e.args)
            e = Op(e.op, args, e.type)
        elif isinstance(e, Select):
            e = Select(
                self.normalize(e.cond, _seen, _depth + 1),
                self.normalize(e.then_expr, _seen, _depth + 1),
                self.normalize(e.else_expr, _seen, _depth + 1),
                e.type,
            )

        # built-in algebra until stable
        for _ in range(_AXIS_BUDGET):
            r = builtin_rewrite(e)
            if r is None:
                break
            e = r

        # user axioms at the root, until stable or until a cycle is detected
        for _ in range(_AXIS_BUDGET):
            fired = False
            for axiom in self.axioms.rules():
                if axiom.equational:
                    continue
                if self._enabled is not None and axiom.name not in self._enabled:
                    continue
                for match in axiom.match(e):
                    subs = match["subs"]
                    rhs = substitute(axiom.rhs, subs)
                    ek = e.key()
                    # record the node *before* descending, so a rule that
                    # re-creates a shape an ancestor already rewrote is
                    # refused instead of recursing forever
                    _seen.add(ek)
                    nxt = self.normalize(rhs, _seen, _depth + 1)
                    nk = nxt.key()
                    if nk == ek or nk in _seen:
                        continue
                    e = nxt
                    self._axiom_uses.add(axiom.name)
                    fired = True
                    break
                if fired:
                    break
            if not fired:
                break

        return _retag(e)

    def check_defined_pair(self, e1: Expr, e2: Expr) -> tuple[str, str]:
        """Return an overall ``(status, reason)`` for a comparison window.

        Reject any definite UB anywhere; if unverifiable (symbolic) issues are
        not mirrored identically on both sides, refuse to prove.
        """
        is1 = {m for m, _ in check_defined(e1)}
        is2 = {m for m, _ in check_defined(e2)}
        if "reject" in is1 or "reject" in is2:
            return ("rejected", "a shift or division is definitely undefined")
        if "cannot" in is1 and "cannot" not in is2:
            return ("not_proven", "definedness not verifiable on the left block")
        if "cannot" in is2 and "cannot" not in is1:
            return ("not_proven", "definedness not verifiable on the right block")
        return ("ok", "")

    def prove_expr(
        self,
        e1: Expr,
        e2: Expr,
        inputs: Mapping[str, Type] | None = None,
        *,
        generalize: bool = True,
        axioms: tuple[str, ...] | list[str] | None = None,
    ) -> Proof:
        """Prove two expressions yield identical bits for every input.

        ``axioms`` optionally curates the rule set a proof may use; without
        it the whole store (minus any enclosing ``scope``) is fair game.
        """
        if axioms is not None:
            self._axiom_rules(axioms)  # reject unknown names before scoping
            with self.scope(*axioms):
                return self.prove_expr(e1, e2, inputs, generalize=generalize)
        inputs = dict(inputs) if inputs is not None else None
        try:
            env = {n: (Var(n, t), t) for n, t in (inputs or {}).items()}
            r1 = ripen(e1, env=env)
            r2 = ripen(e2, env=env)
            if r1.type is None or r2.type is None:
                ty = r1.type or r2.type or _DEFAULT_INT
                r1 = ripen(e1, ty, env=env)
                r2 = ripen(e2, ty, env=env)
            if r1.type is not None and r2.type is not None and r1.type != r2.type:
                return Proof(
                    "not_proven",
                    reason=f"types differ: {r1.type} vs {r2.type}",
                )
        except KernelError as err:
            return Proof("rejected", reason=err.message)

        if inputs is None:
            names = free_vars(r1) | free_vars(r2)
            inputs = {}
            for nm in names:
                found = _first_var_type(r1, nm) or _first_var_type(r2, nm)
                if found is None:
                    return Proof(
                        "rejected", reason=f"cannot resolve type of input {nm!r}"
                    )
                inputs[nm] = found

        b1 = Block("lhs", [Assign(Var("_out", r1.type), r1)])
        b2 = Block("rhs", [Assign(Var("_out", r2.type), r2)])
        return self.prove_blocks(
            b1, b2, inputs=inputs, outputs={"_out"}, generalize=generalize
        )

    def prove_blocks(
        self,
        b1: Block,
        b2: Block,
        inputs: Mapping[str, Type] | None = None,
        outputs: set[str] | None = None,
        *,
        generalize: bool = False,
        axioms: tuple[str, ...] | list[str] | None = None,
    ) -> Proof:
        """Prove two sequences of code produce EXACTLY the same output bits.

        ``inputs`` is the interface (name -> type); when omitted it is read
        off the blocks themselves.  ``outputs`` names the variables whose final
        values are compared; when omitted every variable written by both
        sides is compared.  ``axioms`` curates the rule set used by the proof.
        """
        if axioms is not None:
            self._axiom_rules(axioms)  # reject unknown names before scoping
            with self.scope(*axioms):
                return self.prove_blocks(
                    b1, b2, inputs, outputs, generalize=generalize
                )
        try:
            inputs = dict(inputs) if inputs is not None else None
            if inputs is None:
                merged = dict(_interface(b1))
                merged.update({k: v for k, v in _interface(b2).items() if k not in merged})
                inputs = merged
            self._check_input_types(b1, inputs)
            self._check_input_types(b2, inputs)

            if outputs is None:
                outputs = _writes(b1) & _writes(b2)

            with self._axiom_trace() as used:
                return self._prove_blocks_exec(
                    b1, b2, inputs, outputs, generalize, used
                )

        except KernelError as err:
            return Proof("rejected", reason=err.message)

    def _prove_blocks_exec(
        self,
        b1: Block,
        b2: Block,
        inputs: dict[str, Type],
        outputs: set[str],
        generalize: bool,
        used: set[str],
    ) -> Proof:
        try:
            s1 = _exec_block(b1, inputs, self)
            s2 = _exec_block(b2, inputs, self)
        except _SymbolicLoop:
            return self._prove_transitional(
                b1, b2, inputs, outputs, generalize=generalize, used=used
            )

        confirmed: list[tuple[Expr, Expr]] = []
        for name in sorted(outputs):
            has1 = name in s1
            has2 = name in s2
            if has1 != has2:
                return Proof(
                    "not_proven",
                    reason=f"output {name!r} is written on only one side",
                )
            c1 = s1[name]
            c2 = s2[name]
            e1, t1, raw1 = c1.norm, c1.ty, c1.raw
            e2, t2, raw2 = c2.norm, c2.ty, c2.raw
            if t1 != t2:
                return Proof(
                    "rejected",
                    reason=f"output {name!r} has types {t1} vs {t2}",
                )
            st, why = self.check_defined_pair(e1, e2)
            if st != "ok":
                return Proof(st, reason=why)
            if self.normalize(e1) != self.normalize(e2):
                return Proof(
                    "not_proven",
                    reason=(
                        f"output {name!r} normalizes to "
                        f"{self.normalize(e1).sugar()} vs "
                        f"{self.normalize(e2).sugar()}"
                    ),
                )
            confirmed.append((raw1, raw2))

        result = Proof(
            "proven",
            kind="equivalence",
            reason=f"{len(outputs)} output(s) proven bit-exact",
            axioms_used=tuple(sorted(used)),
        )
        if generalize:
            result = self._maybe_derive_axioms(result, confirmed, inputs)
        return result

    def _prove_transitional(
        self,
        b1: Block,
        b2: Block,
        inputs: dict[str, Type],
        outputs: set[str],
        *,
        generalize: bool,
        used: set[str],
    ) -> Proof:
        """Prove blocks with equal symbolic loop bounds by single-step induction."""
        s1 = {name: (Var(name, ty), ty) for name, ty in inputs.items()}
        s2 = {name: (Var(name, ty), ty) for name, ty in inputs.items()}
        ok = self._compare_stmts(b1.stmts, b2.stmts, s1, s2, inputs)
        if not ok:
            return Proof(
                "not_proven",
                reason=(
                    "symbolic loop counts present; single-step transitions "
                    "could not be matched"
                ),
            )
        result = Proof(
            "proven",
            kind="equivalence",
            reason="transitional equivalence by induction",
            axioms_used=tuple(sorted(used)),
        )
        return result

    def _compare_stmts(
        self,
        s1: list[Stmt],
        s2: list[Stmt],
        state1: State,
        state2: State,
        inputs: dict[str, Type],
    ) -> bool:
        """Positional, step-equivalent comparison used for symbolic loops."""

        def flip(s: State) -> State:
            return dict(s)

        for a, b in zip(s1, s2):
            ta, tb = type(a), type(b)
            if ta is not tb:
                return False
            if isinstance(a, Nop):
                continue
            if isinstance(a, Assign):
                if a.var.name != b.var.name or a.var.type != b.var.type:
                    return False
                ea = ripen(a.rhs, a.var.type, env=flip(state1))
                eb = ripen(b.rhs, b.var.type, env=flip(state2))
                if self.normalize(ea) != self.normalize(eb):
                    return False
                state1 = dict(state1)
                state2 = dict(state2)
                state1[a.var.name] = (self.normalize(ea), a.var.type)
                state2[a.var.name] = (self.normalize(eb), b.var.type)
            elif isinstance(a, Loop):
                if a.count != b.count:
                    # allow int equivalences
                    na, nb = _static_int(a.count), _static_int(b.count)
                    if na is None or nb is None or na != nb:
                        return False
                if not self._compare_stmts(a.body, b.body, state1, state2, inputs):
                    return False
        return len(s1) == len(s2)

    def _raw(self, e: Expr) -> Expr:
        return e

    def _check_input_types(self, block: Block, inputs: dict[str, Type]) -> None:
        iface = _interface(block)
        for name, ty in inputs.items():
            got = iface.get(name)
            if got is not None and got != ty:
                raise TypeError_(
                    f"input {name!r} used as {got} but declared {ty}"
                )

    def _maybe_derive_axioms(
        self, result: Proof, pairs: list[tuple[Expr, Expr]], inputs: dict[str, Type]
    ) -> Proof:
        try:
            from generalization import generalize_equality

            derived: list = []
            for e1, e2 in pairs:
                axs = generalize_equality(self, e1, e2, self.axioms_dir)
                derived.extend(axs)
            if derived:
                result = Proof(
                    "proven",
                    kind="equivalence",
                    reason=result.reason,
                    lhs=result.lhs,
                    rhs=result.rhs,
                    axioms_used=result.axioms_used,
                    derived_axioms=tuple(derived),
                )
        except Exception:  # noqa: BLE001, S110
            # Promoting new axioms is an optimisation, never part of the
            # verdict.  It writes into axioms_dir, which is read-only for an
            # installed package, and a failed promotion must not turn a
            # `proven` into anything else.
            pass
        return result

    def egraph_normalize(
        self,
        e: Expr,
        inputs: Mapping[str, Type] | None = None,
        *,
        axioms: tuple[str, ...] | list[str] | None = None,
        budget: int = _EGRAPH_BUDGET,
    ) -> Expr:
        """Extract a size-minimal representative of ``e`` from a saturated
        e-graph built over the (curated) axiom store."""
        from egraphs import EGraphRunner

        inputs = dict(inputs) if inputs is not None else {}
        env = {n: (Var(n, ty), ty) for n, ty in inputs.items()}
        re = ripen(e, env=env)
        if re.type is None:
            re = ripen(e, _DEFAULT_INT, env=env)
        runner = EGraphRunner(rules=self._axiom_rules(axioms), budget=budget)
        return runner.normal_form(re)

    def egraph_prove(
        self,
        e1: Expr,
        e2: Expr,
        inputs: Mapping[str, Type] | None = None,
        *,
        axioms: tuple[str, ...] | list[str] | None = None,
        budget: int = _EGRAPH_BUDGET,
        verify: bool = True,
    ) -> Proof:
        """Prove ``e1 == e2`` by equality saturation of the axiom store.

        Rules fire as *undirected* equalities (no orientation, no cycle
        rejection), so the associativity/distributivity laws the rewriter must
        refuse are available here.  Soundness comes from local verification
        rather than from the search: the merge is justified by a chain of
        axiom applications, and every rule that contributed is re-checked
        bit-exactly on a domain that *decides* it (``egraphs.rule_tier``), so
        ``proven`` remains an exact argument without enumerating 2**(w*k)
        inputs.  A rule whose domain is too large to enumerate is only sampled
        and the proof is downgraded to ``not_proven`` (pass ``verify=False`` to
        take the store's word for it).
        """
        from egraphs import EGraphRunner, verify_rules

        inputs = dict(inputs) if inputs is not None else None
        try:
            env = {n: (Var(n, ty), ty) for n, ty in (inputs or {}).items()}
            r1 = ripen(e1, env=env)
            r2 = ripen(e2, env=env)
            if r1.type is None or r2.type is None:
                ty = r1.type or r2.type or _DEFAULT_INT
                r1 = ripen(e1, ty, env=env)
                r2 = ripen(e2, ty, env=env)
            if r1.type is not None and r2.type is not None and r1.type != r2.type:
                return Proof(
                    "not_proven",
                    kind="egraph",
                    reason=f"types differ: {r1.type} vs {r2.type}",
                )
        except KernelError as err:
            return Proof("rejected", kind="egraph", reason=err.message)

        if inputs is None:
            inputs = {}
            for nm in sorted(free_vars(r1) | free_vars(r2)):
                found = _first_var_type(r1, nm) or _first_var_type(r2, nm)
                if found is None:
                    return Proof(
                        "rejected", kind="egraph",
                        reason=f"cannot resolve type of input {nm!r}",
                    )
                inputs[nm] = found

        rules = self._axiom_rules(axioms)
        runner = EGraphRunner(rules=rules, budget=budget)
        ca = runner.graph.add(r1)
        cb = runner.graph.add(r2)
        runner.saturate()
        used = tuple(sorted(runner.used))
        if runner.graph.find(ca) != runner.graph.find(cb):
            nf1 = runner.graph.extract(ca)
            nf2 = runner.graph.extract(cb)
            return Proof(
                "not_proven",
                kind="egraph",
                reason=(
                    f"no shared e-class; extracted {nf1.sugar()} vs "
                    f"{nf2.sugar()}"
                ),
                axioms_used=used,
                checked=None,
            )
        if not verify:
            return Proof(
                "proven", kind="egraph",
                reason="equal in the saturated e-graph (axioms not re-checked)",
                axioms_used=used, checked=None,
            )
        fired = [r for r in rules if r.name in runner.used]
        hold, complete, unsound = verify_rules(fired, runner.graph.match_types)
        if not hold:
            return Proof(
                "rejected", kind="egraph",
                reason=(
                    "the axiom store is unsound: "
                    + ", ".join(unsound)
                    + " failed bit-exact checking"
                ),
                axioms_used=used, checked="broken",
            )
        nf = runner.graph.extract(ca)
        if not complete:
            return Proof(
                "not_proven", kind="egraph",
                reason=(
                    f"equal in the saturated e-graph, but the {len(fired)} "
                    "rule(s) involved cannot be decided on a finite domain "
                    "(multiplication/shift laws): the check was a sample"
                ),
                axioms_used=used, checked="sampled",
            )
        return Proof(
            "proven", kind="egraph",
            reason=(
                f"equal in the saturated e-graph; {len(fired)} rule(s) "
                f"bit-exactly re-checked on the types they fired at, "
                f"minimal form {nf.sugar()}"
            ),
            axioms_used=used, checked="complete",
        )

    def load_theory(self, name: str) -> list[Axiom | Any]:
        """Load one axiom file by name (``"arithmetic"`` -> ``arithmetic.py``)."""
        from axiom_store import AxiomError

        if not self.axioms_dir:
            raise AxiomError("kernel has no axioms_dir to load theories from")
        path = os.path.join(self.axioms_dir, f"{name}.py")
        return self.axioms.load_file(path)

    def theories(self) -> list[str]:
        """Names of the theory files contributing axioms (e.g. ``arithmetic``)."""
        names: set[str] = set()
        for ax in self.axioms.rules():
            src = (ax.source or "").replace("\\", "/")
            stem = src.rsplit("/", 1)[-1]
            if stem.endswith(".py"):
                names.add(stem[:-3])
        return sorted(names)

    def theory_rules(self, name: str) -> list[Axiom | Any]:
        """Every rule whose declared source is ``name``'s theory file."""
        src = (name if name.endswith(".py") else f"{name}.py")
        return [ax for ax in self.axioms.rules() if (ax.source or "").endswith(src)]

    def unroll_verify(
        self,
        original: Block,
        unrolled: Block,
        factor: int,
        inputs: Mapping[str, Type] | None = None,
    ) -> Proof:
        """Verify that ``unrolled`` is a faithful unrolling of ``original``.

        The unrolled shape must be exactly::

            original  : [ ..., Loop(n, B) ]
            unrolled  : [ ..., Loop(q, B*f), B[0:r], ..., B[r-1] ]

        The kernel checks every required invariant itself -- the trip-count
        arithmetic ``n == q*factor + r``, the body multiplicity ``B*f``, the
        tail ``B[0:r]``, and (belt and braces) bit-exact simulation of the two
        whole blocks from identical inputs.
        """
        if factor <= 0:
            return Proof("rejected", reason="unroll factor must be >= 1")
        if not (original.stmts and isinstance(original.stmts[-1], Loop)):
            return Proof("not_proven", reason="original block does not end with a loop")
        if not any(isinstance(s, Loop) for s in unrolled.stmts):
            return Proof(
                "rejected", reason="unrolled block does not contain a main loop"
            )

        inputs = dict(inputs) if inputs is not None else {}
        if not inputs:
            merged = dict(_interface(original))
            merged.update(
                {k: v for k, v in _interface(unrolled).items() if k not in merged}
            )
            inputs = merged

        loop = original.stmts[-1]
        n_static = _static_int(loop.count)
        if n_static is None:
            return Proof(
                "not_proven", reason="cannot verify unrolling of a symbolic trip count"
            )
        unrolled_loop_idx = next(
            (i for i, s in enumerate(unrolled.stmts) if isinstance(s, Loop)), -1
        )
        if unrolled_loop_idx < 0:
            return Proof("rejected", reason="unrolled block has no loop")
        unrolled_loop = unrolled.stmts[unrolled_loop_idx]

        q = n_static // factor
        r = n_static % factor
        if n_static != q * factor + r:
            return Proof(
                "rejected",
                reason=f"trip-count arithmetic fails: {n_static} != {q}*{factor}+{r}",
            )
        if _static_int(unrolled_loop.count) != q:
            return Proof(
                "rejected",
                reason=f"unrolled loop count must be n//factor = {q}",
            )

        body = loop.body
        expected_block: list[Stmt] = list(body) * factor
        if not self._stmts_equal(expected_block, unrolled_loop.body):
            return Proof(
                "rejected",
                reason=f"unrolled body is not body repeated exactly {factor} times",
            )

        after = unrolled.stmts[unrolled_loop_idx + 1:]
        expected_tail = list(body[:r])
        if not self._stmts_equal(expected_tail, after):
            return Proof(
                "rejected",
                reason=f"tail after the unrolled loop must be body[0:{r}] "
                       f"({len(expected_tail)} statement(s)), got {len(after)}",
            )

        if not self._verify_simulation(original, unrolled, inputs):
            return Proof(
                "rejected", reason="simulation of original and unrolled disagree"
            )
        return Proof(
            "proven",
            kind="unroll",
            reason=f"unroll-by-{factor} verified (n={n_static}, "
                   f"q={q}, r={r}); bodies and tail match exactly",
            axioms_used=tuple(sorted(self._axiom_uses)),
        )

    def _stmts_equal(self, a: list[Stmt], b: list[Stmt]) -> bool:
        if len(a) != len(b):
            return False

        def key(st: Stmt) -> Any:
            if isinstance(st, Assign):
                return ("a", st.var.name, st.var.type.name, st.rhs.key())
            if isinstance(st, Loop):
                cnt = _static_int(st.count)
                return ("l", cnt, tuple(key(x) for x in st.body))
            if isinstance(st, Nop):
                return ("n",)
            return ("?", str(type(st)))

        return all(key(x) == key(y) for x, y in zip(a, b))

    def _clone_stmt(self, st: Stmt) -> Stmt:
        if isinstance(st, Assign):
            return Assign(Var(st.var.name, st.var.type), st.rhs)
        if isinstance(st, Loop):
            return Loop(st.count, [self._clone_stmt(x) for x in st.body])
        return st

    def _verify_simulation(
        self,
        original: Block,
        unrolled: Block,
        inputs: dict[str, Type],
    ) -> bool:
        try:
            s1 = _exec_block(original, inputs, self)
            s2 = _exec_block(unrolled, inputs, self)
        except KernelError:
            return False
        names = sorted(set(s1) | set(s2))
        if len(s1) != len(s2):
            return False
        for k in names:
            if self.normalize(s1[k].norm) != self.normalize(s2[k].norm):
                return False
        return True

def _first_var_type(expr: Expr, name: str) -> Type | None:
    if isinstance(expr, (Var, Const)) and expr.name == name:
        return expr.type
    if isinstance(expr, Op):
        for a in expr.args:
            t = _first_var_type(a, name)
            if t is not None:
                return t
    if isinstance(expr, Select):
        return (
            _first_var_type(expr.cond, name)
            or _first_var_type(expr.then_expr, name)
            or _first_var_type(expr.else_expr, name)
        )
    return None

def proof_status(status: str) -> Proof:
    if status == "proven":
        return PROVEN
    if status == "rejected":
        return Proof("rejected", reason="rejected")
    return Proof("not_proven")
