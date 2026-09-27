import inspect
from typing import Any, get_args, get_origin


def _definition_name(value: Any) -> str:
    if isinstance(value, Term):
        return value.name
    if isinstance(value, str):
        return value.lower()
    origin = get_origin(value)
    if origin is not None:
        args = ", ".join(_definition_name(arg) for arg in get_args(value))
        return f"{_definition_name(origin)}[{args}]"
    name = getattr(value, "__name__", None) or getattr(value, "__qualname__", None)
    if name:
        return str(name).rsplit(".", 1)[-1].lower()
    return type(value).__name__.lower()

class Term:
    """Runtime type/callable descriptor.

    A term may represent a bare type name, a Python type, a function, a lambda,
    or any object implementing ``__call__``.  It retains the resolved
    signature and annotations so callers can inspect and validate it without
    invoking the definition.
    """

    __slots__ = (
        "annotations",
        "definition",
        "kind",
        "metadata",
        "name",
        "params",
        "return_term",
    )

    def __init__(
        self,
        definition: Any = None,
        params: Any = None,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        resolved_name, resolved = _resolve_definition(definition)
        self.definition = resolved
        self.name = str(name or resolved_name or "type").lower()
        self.kind = self._kind_of(resolved)
        self.params = _normalize_params(params, resolved)
        self.annotations = dict(getattr(resolved, "__annotations__", {}) or {})
        return_annotation = self.annotations.get("return")
        self.return_term = _annotation_term(return_annotation)
        self.metadata = dict(metadata or {})

    @staticmethod
    def _kind_of(definition: Any) -> str:
        if definition is None:
            return "bare"
        if isinstance(definition, Term):
            return "term"
        if isinstance(definition, type):
            return "type"
        if get_origin(definition) is not None:
            return "type"
        if all(hasattr(definition, a) for a in ("name", "width", "signed")):
            return "type"
        if getattr(definition, "__name__", None) == "<lambda>":
            return "lambda"
        return "callable"

    @property
    def signature(self) -> inspect.Signature | None:
        if self.definition is None:
            return None
        if isinstance(self.definition, Term):
            return self.definition.signature
        try:
            return inspect.signature(self.definition)
        except (TypeError, ValueError):
            return None

    @property
    def is_callable(self) -> bool:
        return isinstance(self.definition, Term) or callable(self.definition)

    def resolve(self) -> Any:
        """Return the underlying definition, or this term for a bare term."""
        return self.definition if self.definition is not None else self

    def accepts(self, *args: Any, **kwargs: Any) -> bool:
        signature = self.signature
        if signature is None:
            return not self.is_callable
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            return False
        return True

    def bind(self, *args: Any, **kwargs: Any) -> inspect.BoundArguments:
        signature = self.signature
        if signature is None:
            raise TypeError(f"term {self.name!r} has no callable signature")
        return signature.bind(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return Call(self, args, kwargs)

    def __repr__(self) -> str:
        return f"Term({self.name!r}, kind={self.kind!r}, params={self.params!r})"

    def __hash__(self) -> int:
        return hash((self.name, self.kind, _value_key(self.definition),
                     _value_key(self.params)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Term):
            return NotImplemented
        return (
            self.name,
            self.kind,
            _value_key(self.definition),
            _value_key(self.params),
        ) == (
            other.name,
            other.kind,
            _value_key(other.definition),
            _value_key(other.params),
        )

def _callable_name(value: Any) -> str:
    """Return a stable, lowered name for a type or callable definition."""
    name = getattr(value, "__name__", None)
    if not name:
        name = getattr(value, "__qualname__", None)
    if not name:
        name = type(value).__name__
    return str(name).rsplit(".", 1)[-1].lower()

def _resolve_definition(definition: Any) -> tuple[str, Any]:
    """Resolve aliases and callable objects without requiring a function type."""
    if isinstance(definition, Term):
        return definition.name, definition
    if definition is None:
        return "", None
    if isinstance(definition, str):
        return definition.lower(), None
    if isinstance(definition, type) or get_origin(definition) is not None:
        return _definition_name(definition), definition
    # OProof kernel type descriptor (a Type carrying name/width/signed).
    # Admitted here so ``Term(int32)`` can resolve to the kernel's ``int32``
    # through the OProof bridge -- the two layers stay separate, this just
    # lets the api *denote* a kernel machine type.
    if (
        not callable(definition)
        and all(hasattr(definition, a) for a in ("name", "width", "signed"))
    ):
        return str(definition.name).lower(), definition
    if not callable(definition):
        raise TypeError(
            "definition must be None, a Term, a type, a parameterized type, "
            "or a callable object"
        )
    return _callable_name(definition), definition

def _annotation_term(annotation: Any) -> Term | None:
    if annotation is None or annotation is inspect.Signature.empty:
        return None
    if isinstance(annotation, Term):
        return annotation
    if isinstance(annotation, type) or get_origin(annotation) is not None:
        return Term(annotation)
    if callable(annotation):
        return Term(annotation)
    return Term(name=_definition_name(annotation), metadata={"annotation": annotation})

def _value_key(value: Any) -> Any:
    """Build a hash-safe key without invoking arbitrary equality methods."""
    if isinstance(value, Term):
        return (
            type(value),
            value.name,
            value.kind,
            _value_key(value.definition),
            _value_key(value.params),
        )
    if isinstance(value, tuple):
        return tuple(_value_key(item) for item in value)
    if isinstance(value, list):
        return ("list", tuple(_value_key(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return ("identity", type(value), id(value))
    return ("value", type(value), value)

def _normalize_params(params: Any, definition: Any) -> tuple:
    if params is None:
        if definition is None:
            return ()
        if isinstance(definition, Term):
            return definition.params
        try:
            return tuple(inspect.signature(definition).parameters.values())
        except (TypeError, ValueError):
            return ()
    if isinstance(params, str):
        return (params,)
    if isinstance(params, tuple):
        return params
    if isinstance(params, list):
        return tuple(params)
    return (params,)

class Call(Term):
    """Proof-term representation of applying a term to arguments."""

    __slots__ = ("args", "callee", "kwargs")

    def __init__(
        self,
        callee: Term,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ):
        if not isinstance(callee, Term):
            raise TypeError("Call callee must be a Term")
        super().__init__(name=f"{callee.name}()")
        self.kind = "call"
        self.return_term = callee.return_term
        self.callee = callee
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})

    @property
    def term(self) -> Term:
        return self.callee

    def __repr__(self) -> str:
        suffix = ", ".join(
            [*(repr(arg) for arg in self.args), *(
                f"{key}={value!r}" for key, value in self.kwargs.items()
            )]
        )
        return f"Call({self.callee!r}{', ' if suffix else ''}{suffix})"

    def __hash__(self) -> int:
        return hash((
            type(self),
            _value_key(self.callee),
            _value_key(self.args),
            _value_key(self.kwargs),
        ))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Call):
            return NotImplemented
        return (
            _value_key(self.callee),
            _value_key(self.args),
            _value_key(self.kwargs),
        ) == (
            _value_key(other.callee),
            _value_key(other.args),
            _value_key(other.kwargs),
        )

class Literal(Term):
    """A concrete literal value at the api layer.

    ``7`` is a value, not a type: a literal Term carries ``kind == 'literal'``
    with its value directly in ``definition`` (and, when known, the kernel
    ``Type`` it belongs to in ``type``).  This is what ``term_of_expr``
    produces for kernel ``Lit`` nodes, so the round-trip
    ``expr_of_term(term_of_expr(Lit))`` is lossless instead of mislabelling
    the constant as ``kind='type'``.
    """

    __slots__ = ("type", "value")

    def __init__(self, value: Any, type_: "Type | None" = None):
        super().__init__(definition=None, name=str(value))
        self.value = value
        self.type = type_
        self.definition = value
        self.kind = "literal"

    def __repr__(self) -> str:
        suffix = "" if self.type is None else f", {self.type.name}"
        return f"Literal({self.value!r}{suffix})"

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(("literal", self.value, self.type))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Literal)
            and self.value == other.value
            and self.type == other.type
        )

def parse_parms(
    definition: Any = None,
    params: Any = None,
    *,
    name: str | None = None,
) -> Term:
    """Lower any definition into a fully introspectable :class:`Term`.

    ``definition`` may be absent, a normal type, a lambda, a function, or any
    object implementing ``__call__``.  ``params`` may be omitted, in which case
    callable signatures are resolved with :func:`inspect.signature`.

    Examples::

        parse_parms()                         # bare/anonymous type
        parse_parms(int)                      # type definition
        parse_parms(Term(lambda x: x + 1))  # callable term
        parse_parms(handler_object)          # arbitrary callable instance
        parse_parms(int, ("value",))         # explicit parameters
    """
    if isinstance(definition, Term) and params is None and name is None:
        return definition
    return Term(
        definition,
        params,
        name=name,
    )

# Backward-compatible constructor retained for callers using the previous API:
# Type("name", params, definition).

class Type(Term):
    def __init__(
        self,
        name: str,
        params: Any = None,
        definition: Any = None,
    ):
        super().__init__(definition, params, name=name)

_parse_parms = parse_parms
