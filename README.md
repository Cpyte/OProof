# OProof

Bit-exact equivalence proving for machine integers. OProof decides whether two
computation blocks over `bool`, `int8`…`int64`, and `uint8`…`uint64` compute the
same outputs, using the arithmetic the target machine actually performs —
truncation, wraparound, sign extension, and defined shifts.

It is not a floating-point or real-arithmetic prover. `x + 1` overflowing
`int8` is a different function from unbounded `x + 1`, and OProof reports the
overflow rather than proving something convenient.

## Install

```sh
pip install opreof
```

Pure Python, no dependencies, requires Python 3.11+.

## Prove an equivalence

```python
from opreof import Block, Assign, Var, int8, prove

a = Block("a", [Assign(Var("x", int8), Var("y", int8))])
b = Block("b", [Assign(Var("x", int8), Var("y", int8))])

r = prove(a, b, outputs={"x"})
print(r.status)   # proven
```

`status` is one of:

| status | meaning |
| --- | --- |
| `proven` | a derivation was found and every step checked |
| `not_proven` | the prover stopped without finding one |
| `rejected` | the question is ill-posed (e.g. `200` is not an `int8`) |

`not_proven` is not a refutation. Only `proven` is a positive answer, and the
kernel re-verifies each step of the derivation it returns.

## What it understands

Rules live in plain Python files under `axioms/`, so the fact base is data you
can read, narrow, or replace:

Theories are files; axioms are the individual named rules inside them.
Curation selects **axiom names**, and an unknown name is an error rather than
a silently ignored typo:

```python
from opreof import list_theories, theory_rules, list_axioms, with_axioms, prove

list_theories()             # ['arithmetic', 'foundational', 'logic', 'pow2']
theory_rules("logic")       # every rule from that file
list_axioms()               # 109 names, e.g. 'add_assoc'

with with_axioms("add_self_is_shl1"):   # only that fact is available
    prove(a, b, outputs={"x"})

prove(a, b, outputs={"x"}, axioms=("add_assoc", "add_self_is_shl1"))  # per call
```

Facts proven during a session are generalized and promoted to new axioms,
persisted back to `axioms/`, so a proof you already paid for is reused.

A rule declared with `equational=True` (associativity, distributivity, De
Morgan) is deliberately kept out of directed normalization. A rewriter that
fired those would loop or pick arbitrary orientations. OProof hands them to the
e-graph instead.

## Equality saturation

Directed rewriting answers "can `a` become `b`". Equality saturation builds an
e-graph, applies every rule in both directions until nothing new merges, then
extracts the cheapest representative:

The e-graph entry points take kernel expressions (`Op`/`Var`), plus the
input-type map, rather than `Block`s:

```python
from opreof import Var, Op, int8, egraph_normalize, egraph_prove

I = {"x": int8, "y": int8, "z": int8}
x, y, z = Var("x", int8), Var("y", int8), Var("z", int8)

# associativity is equational, so only saturation can use it
left  = Op("add", (Op("add", (x, y)), z))
right = Op("add", (x, Op("add", (y, z))))
print(egraph_prove(left, right, I).status)   # proven, via add_assoc

# extraction returns the smallest member, so a sum of products comes back factored
f = Op("add", (Op("mul", (x, y)), Op("mul", (x, z))))
print(egraph_normalize(f, I).sugar())        # mul(add(y, z), x)
```

Extraction is a min-plus relaxation over the final equivalence classes, and each
extracted term is checked locally against the class it came from. Saturation
runs under node and work budgets; when a budget cuts the search short the
result carries `truncated=True` and is a saturated *subset* — fewer merges,
never a wrong one.

## What it will not do

- A rule whose side is a bare variable is rejected, because e-matching it would
  bind an unconstrained placeholder and merge arbitrary terms.
- Shifts that lose bits or start negative are `rejected` as undefined, not
  silently modelled.
- Literals outside the range of their declared type are `rejected`.
- Budget exhaustion is reported, never rounded down to a confident `proven`.

## Run the tests

```sh
python test_kernel.py
```

41 tests, including an exhaustive bit-exact sweep of every axiom over
concrete inputs for the small types.

## Examples

```sh
python examples/egraph.py   # equality saturation
python examples/solve.py    # synthesis
python examples/fib.py      # block equivalence
```

## License

MIT. See [LICENSE](LICENSE).
