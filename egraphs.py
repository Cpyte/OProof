"""E-graph based equality reasoning for OProof.

The kernel's ``normalize`` is a *directed* rewrite system: an axiom is applied
only as ``lhs ~> rhs``, its orientation is baked into a canonical order, and a
rewrite that would loop is abandoned.  That is exactly what a consumer does
*not* want when they hand a prover a pile of facts: they want the facts, not a
scripted direction.

An e-graph (equality saturation graph) treats every axiom as an undirected
equivalence:

    * one ``e-node`` per term, hash-consed; ``e-classes`` are unions of nodes
      the rules proved equal (union-find + congruence closure), so if ``a``
      and ``b`` share a class then ``f(a) == f(b)`` is implied automatically;
    * rules fire by *e-matching*: bind each pattern placeholder to a concrete
      node, check the rule's side conditions against that concrete node (its
      kernel type is known), then union the matched class with the right-hand
      side -- no orientation, no cycle rejection (a cycle is just another
      equality);
    * saturation iterates the rule set until fixpoint or a node/matching
      budget, closing congruences once per round;
    * ``extract`` picks a size-minimal representative out of a class, so a
      normal form can be read back out of a saturated graph.

Two rules keep that honest, because an e-matching loop is where a prover goes
wrong quietly:

    * a side that is a *bare variable* (``and(-1, a) == a`` written the other
      way round, ``b == add(sub(b, a), a)``) is a directed hint, not an
      equality.  E-matching it would bind the variable to any node at all and
      merge that node with its own expansion -- the axiom proving itself.  Only
      the side with real structure is matched;
    * a body placeholder the match did not pin down is never invented: the
      other side is e-matched too, with what is known pinned, and if it still
      cannot be pinned the firing is dropped rather than guessed.

Any kernel ``Axiom`` works; `EGraphRunner` ties a hand-curated rule list to a
persistent graph and is a reusable knowledge base: add more terms and saturate
again without recomputing what is already merged.  ``Kernel.egraph_prove``
re-checks every rule the merge depended on (``rule_tier`` decides a rule on a
small cube when its shape decides it, and enumerates the domain only when the
domain is small enough), so an ill-advised rule can never manufacture a proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from axiom_store import Axiom
from kernel import (
    Const,
    Expr,
    KernelError,
    Lit,
    Op,
    Select,
    Type,
    Var,
    free_vars,
    ripen,
    substitute,
)

_VAR = "v"
_CONST = "c"
_LIT = "l"
_OP = "o"
_ROOT = "\x00root"

class _OutOfWork(Exception):
    """Internal: the match budget for one saturation round is spent."""

_SINGLE_TYPES = {
    "bool": None,
    "int8": None,
    "uint8": None,
    "int16": None,
    "uint16": None,
    "int32": None,
    "uint32": None,
    "int64": None,
    "uint64": None,
}
from kernel import BOOL, int8, int16, int32, int64, uint8, uint16, uint32, uint64

_SINGLE_TYPES.update(
    {
        "bool": BOOL,
        "int8": int8,
        "uint8": uint8,
        "int16": int16,
        "uint16": uint16,
        "int32": int32,
        "uint32": uint32,
        "int64": int64,
        "uint64": uint64,
    }
)

def _type_of(name: str | None) -> Type | None:
    if name is None:
        return None
    return _SINGLE_TYPES.get(name)

class EGraph:
    """An equality saturation graph over kernel terms.

    Nodes are anonymous tuples; an ``Op``/``Select`` node's ``args`` always
    hold *e-class ids* (the ids are canonicalised through ``find`` on the way
    in).  ``rebuild`` re-canonicalises every node through the union-find and
    merges congruent nodes, so the graph stays hash-consed after unions.
    """

    def __init__(self) -> None:
        #: union-find parents/sizes over e-class ids (size in classes[c]<0)
        self._uf: list[int] = []
        #: parallel to ``nodes``: e-class id each node lives in
        self._class_of: list[int] = []
        #: canonical e-class id -> the node ids that live in it.  e-matching
        #: asks for the members of a class once per pattern argument per node,
        #: so this has to be an index, not a scan.
        self._members: dict[int, set[int]] = {}
        self._member_cache: dict[int, list[int]] = {}
        #: one e-node per entry; ``('v', name, ty)``, ``('c', name, ty)``,
        #: ``('l', value, ty)``, ``('o', op, arg-class-tuple, ty)``
        self.nodes: list[tuple] = []
        #: structural key -> node id (hash-cons)
        self._mem: dict[tuple, int] = {}
        #: rule name -> concrete placeholder types of its last firing, so the
        #: caller can re-check that rule on exactly the types it was used at
        self.match_types: dict[str, dict] = {}
        #: depth cutoff for *derived* nodes (0 = follow the input terms).
        #: Rules are applied in both directions, so a contracting law such as
        #: ``neg(neg(x)) == x`` also *expands*; composed with a two-variable
        #: law that manufactures ``neg(neg(neg(neg(a))))``, ``...`` forever.
        #: Refusing terms deeper than a few levels past what the caller handed
        #: in costs no equality anyone asked about and makes saturation reach a
        #: fixpoint instead of a budget.
        self.max_depth: int = 0
        self._max_seen_depth: int = 0
        #: e-matching attempt budget for the whole saturation, and the counter.
        #: e-matching every pattern at every node of every e-class is
        #: quadratic-to-exponential in the rule set, so the engine spends a
        #: fixed budget and reports truncation instead of running unbounded.
        self.work_budget: int = 1 << 21
        self.work: int = 0
        #: set when a node/work budget cut the search short (a saturated
        #: *subset*: fewer merges, never a wrong one)
        self.truncated: bool = False

    @staticmethod
    def expr_depth(e: Expr) -> int:
        if isinstance(e, Op):
            return 1 + max((EGraph.expr_depth(a) for a in e.args), default=0)
        if isinstance(e, Select):
            return 1 + max(
                EGraph.expr_depth(x)
                for x in (e.cond, e.then_expr, e.else_expr)
            )
        return 0

    def _depth_limit(self, depth: int) -> int:
        if not self.max_depth:
            self.max_depth = max(8, self._max_seen_depth + 2)
        return self.max_depth

    def _within_depth(self, e: Expr) -> bool:
        return self.expr_depth(e) <= self._depth_limit(self.expr_depth(e))

    def _new_class(self) -> int:
        self._uf.append(-1)
        return len(self._uf) - 1

    def find(self, c: int) -> int:
        root = c
        while self._uf[root] >= 0:
            root = self._uf[root]
        while c != root:
            nxt = self._uf[c]
            self._uf[c] = root
            c = nxt
        return root

    def _union(self, a: int, b: int) -> bool:
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self._uf[a] > self._uf[b]:
            a, b = b, a
        self._uf[a] += self._uf[b]
        self._uf[b] = a
        # keep the class -> members index in step with the union-find
        merged = self._members.pop(a, set()) | self._members.pop(b, set())
        self._members[a] = merged
        self._member_cache.pop(a, None)
        self._member_cache.pop(b, None)
        return True

    def _add_node(self, key: tuple) -> int:
        # hash-cons on the *canonical* form, so a term added after a merge
        # still finds its node instead of growing a duplicate
        key = self._canon_key(key)
        got = self._mem.get(key)
        if got is not None:
            return got
        nid = len(self.nodes)
        self.nodes.append(key)
        cls = self._new_class()
        self._class_of.append(cls)
        self._members.setdefault(cls, set()).add(nid)
        self._mem[key] = nid
        return nid

    def add(self, e: Expr) -> int:
        """Hash-cons ``e`` into the graph; return its e-class id.

        Terms the *caller* supplies always go in, and they set the depth cutoff
        for everything the rules derive from them.
        """
        if isinstance(e, Var):
            nid = self._add_node((_VAR, e.name, _ty(e.type)))
        elif isinstance(e, Const):
            nid = self._add_node((_CONST, e.name, _ty(e.type)))
        elif isinstance(e, Lit):
            nid = self._add_node((_LIT, e.value, _ty(e.type)))
        elif isinstance(e, Select):
            args = tuple(self.find(self.add(a)) for a in (e.cond, e.then_expr, e.else_expr))
            nid = self._add_node((_OP, "select", args, _ty(e.type)))
        elif isinstance(e, Op):
            args = tuple(self.find(self.add(a)) for a in e.args)
            nid = self._add_node((_OP, e.op, args, _ty(e.type)))
        else:
            raise TypeError(f"cannot add {e!r} to an e-graph")
        self._max_seen_depth = max(self._max_seen_depth, self.expr_depth(e))
        return self._class_of[nid]

    def _canon_key(self, node: tuple) -> tuple:
        if node[0] != _OP:
            return node
        return (node[0], node[1], tuple(self.find(a) for a in node[2]), node[3])

    def rebuild(self) -> bool:
        """Merge congruent nodes; loop to a fixpoint.  Returns ``changed``."""
        changed = False
        while True:
            round_changed = False
            by_key: dict[tuple, int] = {}
            for nid, node in enumerate(self.nodes):
                self._class_of[nid] = self.find(self._class_of[nid])
                k = self._canon_key(node)
                other = by_key.get(k)
                if other is None:
                    by_key[k] = nid
                elif self._union(self._class_of[nid], self._class_of[other]):
                    self._class_of[nid] = self.find(self._class_of[nid])
                    round_changed = True
                    changed = True
            if not round_changed:
                break
        return changed

    def union(self, a: int, b: int) -> bool:
        """Merge two classes (congruence-closing), returns ``changed``."""
        if not self._union(a, b):
            return False
        self.rebuild()
        return True

    def class_members(self, cls: int) -> list[int]:
        """The node ids living in ``cls`` (cached; the cache dies with unions)."""
        root = self.find(cls)
        got = self._member_cache.get(root)
        if got is None:
            got = sorted(self._members.get(root, ()))
            self._member_cache[root] = got
        return got

    def ematch(self, pat: Expr, pre: dict | None = None) -> list[dict]:
        """Bind the pattern's placeholders to concrete nodes (name -> node id).

        Every node is a candidate root, so the rule fires wherever its LHS
        shape appears; repeated placeholders must land in the same e-class.
        The reserved key ``_root`` holds the matched root node id.  Names in
        ``pre`` are *pinned*: a match that would put them elsewhere is not a
        match, which is how a rule side with a bare variable
        (``b == add(sub(b, a), a)``) gets its remaining placeholders filled in
        from the graph instead of from thin air.

        Matching is *acyclic*: a placeholder may not be bound to a node whose
        e-class is already on the descent path.  Without that, a rule with
        free variables would happily build ever-deeper variants of a term
        (``f(f(a, b), a)``, ``f(f(f(a, b), a), a)``, ...) that are all equal
        and none of which the consumer asked for -- the classic e-matching
        blowup.  Skipping those bindings costs no equality that has finite
        depth, and it is what makes saturation reach a fixpoint.
        """
        pinned = pre or {}
        path = frozenset(
            self.find(self._class_of[v]) for v in pinned.values()
        )
        out: list[dict] = []
        try:
            for nid in range(len(self.nodes)):
                self._match_at(
                    pat, nid, {**pinned, _ROOT: nid}, out, path
                )
        except _OutOfWork:
            self.work = self.work_budget
            self.truncated = True
        seen: set[frozenset] = set()
        res: list[dict] = []
        for b in out:
            key = frozenset((n, self.find(self._class_of[v])) for n, v in b.items())
            if key not in seen:
                seen.add(key)
                res.append(b)
        return res

    def _match_at(
        self, pat: Expr, nid: int, bind: dict, out: list, path: frozenset
    ) -> None:
        self.work += 1
        if self.work >= self.work_budget:
            raise _OutOfWork
        node = self.nodes[nid]
        if isinstance(pat, (Var, Const)):
            if (
                pat.type is not None
                and node[0] in (_VAR, _CONST)
                and node[2] != _ty(pat.type)
            ):
                return
            if self.find(self._class_of[nid]) in path:
                return  # cyclic match -- would only build deeper duplicates
            name = pat.name
            cur = bind.get(name)
            if cur is not None:
                if self.find(self._class_of[cur]) == self.find(self._class_of[nid]):
                    out.append(bind)
                return
            nb = dict(bind)
            nb[name] = nid
            out.append(nb)
            return
        if isinstance(pat, Lit):
            if node[0] != _LIT:
                return
            if pat.type is not None and node[2] != _ty(pat.type):
                return
            if pat.value is not None and pat.value != node[1]:
                return
            out.append(bind)
            return
        if isinstance(pat, Select):
            self._match_op("select", (pat.cond, pat.then_expr, pat.else_expr),
                           node, nid, bind, out, path)
            return
        if isinstance(pat, Op):
            self._match_op(pat.op, pat.args, node, nid, bind, out, path)
            return

    def _match_op(
        self, op: str, pats: tuple, node: tuple, nid: int, bind: dict,
        out: list, path: frozenset
    ) -> None:
        if node[0] != _OP or node[1] != op or len(node[2]) != len(pats):
            return
        cur = [bind]
        here = path | {self.find(self._class_of[nid])}
        for pa, acls in zip(pats, node[2]):
            nxt: list[dict] = []
            for cb in cur:
                for a_nid in self.class_members(acls):
                    self._match_at(pa, a_nid, cb, nxt, here)
            cur = nxt
            if not cur:
                return
        out.extend(cur)

    def _merge(self, rule: Axiom, bind: dict, build: Expr) -> bool:
        """Rewrite ``build`` under ``bind`` and merge it with the matched root.

        Refuses (returns ``False``, adding nothing) unless the match pinned
        down *every* placeholder the body mentions.  A body placeholder that
        the match left free would otherwise enter the graph as a floating
        ``Var('b')`` -- a node that the next rule can bind by name, which
        quietly equates terms nothing said were equal.
        """
        subs = {n: self._node_expr(v) for n, v in bind.items() if n != _ROOT}
        types = {n: e.type for n, e in subs.items()}
        if not free_vars(build) <= set(subs):
            return False
        try:
            if not all(c(subs, types) for c in rule.conditions):
                return False
            other_expr = substitute(build, subs)
            # the rule body carries no types (patterns are adaptable);
            # the rewrite engine re-ripens after substitution, the
            # e-graph must too, or typed and untyped copies never
            # collide.
            root_ty = _root_type(self.nodes[bind[_ROOT]])
            if root_ty is not None:
                other_expr = ripen(other_expr, root_ty, env={})
        except (KernelError, TypeError, ValueError, RecursionError):
            # a rule that will not type/ripen for this binding simply does not
            # fire here; it is not evidence of anything
            return False
        if not self._within_depth(other_expr):
            return False  # deeper than the caller ever wrote: skip
        other_cls = self.add(other_expr)
        matched_cls = self.find(self._class_of[bind[_ROOT]])
        if not self._union(other_cls, matched_cls):
            return False
        self.match_types.setdefault(rule.name, types)
        return True

    def apply_rule(self, rule: Axiom, budget: int = 1 << 20) -> bool:
        """Fire one rule as an equality; returns ``True`` on any new union.

        A rule is an *equation*, so both of its sides are e-matched: matching
        the left-hand side rewrites right, and matching the right-hand side
        rewrites left.  That is what lets a consumer factor as well as expand
        (``x*y + x*z`` finds ``x*(y+z)``) without the rule being written twice.

        Side conditions run against the concrete matched nodes (their kernel
        types are known), exactly as the rewrite engine does.  The concrete
        placeholder types of every firing are recorded in ``self.match_types``
        so the caller can re-check the rule on exactly those types.
        """
        changed = False
        for pat, other in ((rule.lhs, rule.rhs), (rule.rhs, rule.lhs)):
            if isinstance(pat, (Var, Const)):
                # A bare variable on one side (``and(-1, a) == a`` written the
                # other way round, or ``b == add(sub(b, a), a)``) is a
                # *directional* hint: it says a term equals one of its
                # expansions.  E-matching it would bind the variable to any
                # node at all and then merge that node with its own expansion
                # -- proving the axiom from itself.  Only the side with real
                # structure is matched.
                continue
            for bind in self.ematch(pat):
                if self._merge(rule, bind, other):
                    changed = True
                    if len(self.nodes) >= budget:
                        return True
                    continue
                # the match did not pin the whole body down, which happens when
                # the matched side is a bare variable (``b == add(sub(b, a),
                # a)``): complete the binding by matching the *other* side too,
                # with everything already known pinned.
                pinned = {n: v for n, v in bind.items() if n != _ROOT}
                if not free_vars(other) - set(pinned):
                    continue
                for bind2 in self.ematch(other, pre=pinned):
                    if self._merge(rule, bind2, pat):
                        changed = True
                        if len(self.nodes) >= budget:
                            return True
        return changed

    def saturate(self, rules, budget: int = 1 << 12, limit: int = 32) -> set[str]:
        """Apply every rule to a fixpoint (or until a budget runs out).

        Returns the names of the rules that contributed at least one union --
        e-graph use is audited exactly like rewrite use is.  Running out of
        node or matching budget sets ``self.truncated``: the graph is then a
        saturated *subset*, which can only cost merges, never invent them.
        """
        used: set[str] = set()
        rules = list(rules)
        for _ in range(limit):
            changed = False
            for rule in rules:
                if self.apply_rule(rule, budget):
                    used.add(rule.name)
                    changed = True
                if len(self.nodes) >= budget:
                    self.truncated = True
                    return used
                if self.work >= self.work_budget:
                    self.truncated = True
                    return used
            # congruence closure once per round, not once per union: the pass
            # is linear in the graph, and doing it per merge is what turns a
            # saturating e-graph into a quadratic one
            if self.rebuild():
                changed = True
            if not changed:
                break
        else:
            self.truncated = True
        return used

    def _node_expr(self, nid: int) -> Expr:
        node = self.nodes[nid]
        ty = _type_of(node[3] if node[0] == _OP else node[2])
        if node[0] == _VAR:
            return Var(node[1], ty)
        if node[0] == _CONST:
            return Const(node[1], ty)
        if node[0] == _LIT:
            return Lit(node[1], ty)
        args = tuple(self._rep_expr(a) for a in node[2])
        if node[1] == "select":
            return Select(args[0], args[1], args[2], ty)
        return Op(node[1], args, ty)

    def _rep_expr(self, nid: int, path: frozenset[int] = frozenset()) -> Expr:
        """*A* term for the class of ``nid`` -- any one will do.

        e-matching only needs a concrete value to substitute for a matched
        placeholder, so this never searches for a small one: it rebuilds the
        node it was handed.  (Searching for the smallest term here is the
        classic way to make an e-graph exponentially slow, because
        extraction is called once per match per argument.)  Cyclic classes
        collapse to a literal, which is a member of every arithmetic class.
        """
        node = self.nodes[nid]
        ty = _type_of(node[3] if node[0] == _OP else node[2])
        if node[0] == _VAR:
            return Var(node[1], ty)
        if node[0] == _CONST:
            return Const(node[1], ty)
        if node[0] == _LIT:
            return Lit(node[1], ty)
        cls = self.find(self._class_of[nid])
        if cls in path:
            return Lit(0, ty)
        path = path | {cls}
        args = tuple(self._rep_expr(a, path) for a in node[2])
        if node[1] == "select":
            return Select(args[0], args[1], args[2], ty)
        return Op(node[1], args, ty)

    def _costs(self) -> dict[int, int]:
        """Fewest-nodes cost per e-class, by relaxation to a fixpoint.

        The textbook min-plus fixpoint over the class graph, iterated a bounded
        number of times so that a cyclic graph cannot spin.  Monotone
        decreasing over integers, so it terminates.
        """
        cost: dict[int, int] = {}
        for nid, node in enumerate(self.nodes):
            if node[0] != _OP:
                cost[self.find(self._class_of[nid])] = 1
        for _ in range(len(self.nodes) + 1):
            changed = False
            for nid, node in enumerate(self.nodes):
                if node[0] != _OP:
                    continue
                cls = self.find(self._class_of[nid])
                total = 1
                for a in node[2]:
                    total += cost.get(self.find(a), 1)
                if total < cost.get(cls, total + 1):
                    cost[cls] = total
                    changed = True
            if not changed:
                break
        return cost

    def _best_node(
        self, cls: int, cost: dict[int, int], path: frozenset[int]
    ) -> int | None:
        """Cheapest member of ``cls`` whose arguments stay off ``path``."""
        safe: int | None = None
        safe_cost: int | None = None
        any_node: int | None = None
        any_cost: int | None = None
        for m in self.class_members(cls):
            node = self.nodes[m]
            c = (
                1
                if node[0] != _OP
                else 1 + sum(cost.get(self.find(a), 1) for a in node[2])
            )
            if any_cost is None or c < any_cost:
                any_cost, any_node = c, m
            blocked = node[0] == _OP and any(
                self.find(a) == cls or self.find(a) in path for a in node[2]
            )
            if not blocked and (safe_cost is None or c < safe_cost):
                safe_cost, safe = c, m
        return safe if safe is not None else any_node

    def extract(self, cls: int) -> Expr:
        """Smallest representative term of the e-class ``cls``."""
        cost = self._costs()
        chosen = self._best_node(self.find(cls), cost, frozenset())
        if chosen is None:
            return Lit(0, None)
        return self._build(chosen, frozenset(), cost)

    def _build(self, nid: int, path: frozenset[int], cost: dict[int, int]) -> Expr:
        node = self.nodes[nid]
        cls = self.find(self._class_of[nid])
        if cls in path:
            return Lit(0, _type_of(node[3] if node[0] == _OP else node[2]))
        path = path | {cls}
        ty = _type_of(node[3] if node[0] == _OP else node[2])
        if node[0] == _VAR:
            return Var(node[1], ty)
        if node[0] == _CONST:
            return Const(node[1], ty)
        if node[0] == _LIT:
            return Lit(node[1], ty)
        args = tuple(self._build_arg(a, path, cost) for a in node[2])
        if node[1] == "select":
            return Select(args[0], args[1], args[2], ty)
        return Op(node[1], args, ty)

    def _build_arg(self, acls: int, path: frozenset[int], cost: dict[int, int]) -> Expr:
        n = self._best_node(self.find(acls), cost, path)
        if n is None:
            return Lit(0, None)
        return self._build(n, path, cost)

def _ty(t: Type | None) -> str | None:
    return t.name if t is not None else None

# Local axiom verification -- the e-graph's soundness contract
#
# An e-graph proves ``a == b`` by *chaining* axiom applications plus congruence.
# Chaining is sound as soon as every axiom application is sound, so the e-graph
# does not have to enumerate the whole input space: it only has to check the
# finite set of rules it actually fired.  That check is cheap when the rule's
# *shape* decides it:
#
#   affine  -- every operation is add/sub/neg.  The two sides are affine forms
#              over Z(2**w) whose coefficients are in {0,+-1,+-2}, so they can
#              only differ if some coefficient differs; testing the cube
#              {0,1,-1}**k (which contains the decision basis 0, +-e_i) decides
#              the identity completely, for any width.
#   bitwise -- every operation is and/or/xor/not, i.e. a per-bit Boolean
#              function of the k placeholders.  Equality of two such functions
#              is decided by the whole cube {0,1}**k.
#   enumerated -- anything else (mul, shifts, division).  There the identity
#              must be decided by evaluating it, which is only *complete* while
#              the domain fits the cap; beyond that the check is a sample and
#              the caller is told so instead of being handed a false "proven".

_AFFINE_OPS = frozenset({"add", "sub", "neg"})
_BITWISE_OPS = frozenset({"and", "or", "xor", "not"})
_MUL_OPS = frozenset({"add", "sub", "neg", "mul"})

#: how many concrete evaluations one rule may get before its check counts as
#: a sample rather than a decision
VERIFY_CAP = 1 << 14

#: set OPROOF_VERIFY_DEBUG=1 to see why a rule failed its local check
_DEBUG_VERIFY = __import__("os").environ.get("OPROOF_VERIFY_DEBUG") == "1"

#: value sets that *decide* the corresponding tier
_TIER_VALUES = {
    "affine": (0, 1, -1),
    "bitwise": (0, 1),
    "multilinear": (0, 1),
}

def _rule_ops(e: Expr, out: set[str]) -> None:
    if isinstance(e, Op):
        out.add(e.op)
        for a in e.args:
            _rule_ops(a, out)
    elif isinstance(e, Select):
        out.add("select")
        _rule_ops(e.cond, out)
        _rule_ops(e.then_expr, out)
        _rule_ops(e.else_expr, out)

def _degrees(e: Expr) -> dict | None:
    """Per-placeholder degree of ``e``; ``None`` once a variable reaches 2."""
    if isinstance(e, Lit):
        return {}
    if isinstance(e, (Var, Const)):
        return {e.name: 1}
    if isinstance(e, Op):
        if e.op in ("add", "sub", "neg"):
            acc: dict = {}
            for a in e.args:
                d = _degrees(a)
                if d is None:
                    return None
                for k, v in d.items():
                    acc[k] = max(acc.get(k, 0), v)
            return acc
        if e.op == "mul":
            total: dict = {}
            for a in e.args:
                d = _degrees(a)
                if d is None:
                    return None
                for k, v in d.items():
                    if k in total or v > 1:
                        return None  # a variable twice over: not multilinear
                    total[k] = 1
            return total
    return None

def _multilinear(e: Expr) -> bool:
    return _degrees(e) is not None

def rule_tier(rule: Axiom) -> str:
    """Which decision procedure decides ``rule``.

    * ``affine``      -- add/sub/neg only; the ``{0,1,-1}`` cube decides it
    * ``bitwise``     -- and/or/xor/not only; the ``{0,1}`` cube decides it
    * ``multilinear`` -- ring ops with every variable of degree <= 1, so
      distributivity's ``x*(y+z) == x*y + x*z`` is a finite difference away
      from a decision on the ``{0,1}`` cube
    * ``enumerated``  -- anything else (reassociating products, shifts): only
      full enumeration decides it
    """
    ops: set[str] = set()
    _rule_ops(rule.lhs, ops)
    _rule_ops(rule.rhs, ops)
    if "select" in ops:
        return "enumerated"
    if not ops or ops <= _AFFINE_OPS:
        return "affine"
    if ops <= _BITWISE_OPS:
        return "bitwise"
    if ops <= _MUL_OPS and _multilinear(rule.lhs) and _multilinear(rule.rhs):
        return "multilinear"
    return "enumerated"

def rule_holds(rule: Axiom, types: dict | None = None,
               cap: int = VERIFY_CAP) -> tuple[bool, bool]:
    """Check one rule on concrete inputs; returns ``(holds, complete)``.

    ``complete`` is False when the domain was too large to enumerate, so the
    caller can distinguish "the rule is bit-exact" from "the rule survived a
    sample".
    """
    from kernel import KernelError, eval_c

    names = sorted(free_vars(rule.lhs) | free_vars(rule.rhs))
    if not names:
        # a closed rule (all literals): it is decided by folding the sides
        try:
            same = ripen(rule.lhs, int8, env={}) == ripen(rule.rhs, int8, env={})
        except KernelError:
            return False, True
        return same, True
    tier = rule_tier(rule)
    env: dict = {}
    for nm in names:
        t = (types or {}).get(nm) or int8
        env[nm] = (Var(nm, t), t)
    root_ty = env[names[0]][1]
    if tier in _TIER_VALUES:
        # the tier's value cube *is* the decision procedure
        values = _TIER_VALUES[tier]
        complete = True
    else:
        width = root_ty.width
        domain = 1 << width
        complete = domain ** len(names) <= cap
        if complete:
            values = tuple(range(domain))
            if root_ty.signed:
                half = domain >> 1
                values = tuple(v - domain if v >= half else v for v in values)
        elif width == 1:
            values = (0, 1)
        elif root_ty.signed:
            values = (-(1 << (width - 1)), -1, 0, 1, 2, (1 << (width - 1)) - 1)
        else:
            values = (0, 1, 2, (1 << width) - 1)
    combos: list[dict] = [{}]
    for nm in names:
        nxt: list[dict] = []
        for c in combos:
            for v in values:
                c2 = dict(c)
                c2[nm] = v
                nxt.append(c2)
        combos = nxt
        if len(combos) > cap:
            break
    for vals in combos:
        lits = {nm: Lit(v, root_ty) for nm, v in vals.items()}
        try:
            lv = eval_c(ripen(substitute(rule.lhs, lits), root_ty, env=env))
            rv = eval_c(ripen(substitute(rule.rhs, lits), root_ty, env=env))
        except (KernelError, TypeError, ValueError) as err:
            if _DEBUG_VERIFY:
                print(f"  [verify] {rule.name} {vals} raised {err!r}")
            return False, complete
        if lv != rv:
            if _DEBUG_VERIFY:
                print(f"  [verify] {rule.name} {vals}: {lv} != {rv}")
            return False, complete
    return True, complete

def verify_rules(rules, types_by_rule: dict | None = None,
                 cap: int = VERIFY_CAP) -> tuple[bool, bool, list[str]]:
    """Check every rule a merge depended on.

    Returns ``(all_hold, all_complete, unsound)``: ``all_hold`` False means a
    rule is *wrong* (the store is unsound and the proof must be refused),
    ``all_complete`` False means the check was only a sample.
    """
    hold = True
    complete = True
    unsound: list[str] = []
    for r in rules:
        ok, done = rule_holds(r, (types_by_rule or {}).get(r.name), cap)
        if not ok:
            hold = False
            unsound.append(r.name)
        if not done:
            complete = False
    return hold, complete, unsound

def _root_type(node: tuple) -> Type | None:
    """The kernel type carried by an e-node (result type for ops)."""
    if node[0] == _OP:
        return _type_of(node[3])
    return _type_of(node[2])

def _untyped_vars(e: Expr) -> set[str]:
    """Names of variables in ``e`` the term gives no type for."""
    out: set[str] = set()

    def walk(x: Expr) -> None:
        if isinstance(x, Var):
            if x.type is None:
                out.add(x.name)
        elif isinstance(x, Op):
            for a in x.args:
                walk(a)
        elif isinstance(x, Select):
            walk(x.cond)
            walk(x.then_expr)
            walk(x.else_expr)

    walk(e)
    return out

@dataclass

class EGraphRunner:
    """A reusable equality knowledge base.

    ``EGraphRunner(rules)`` pre-saturates nothing until you add terms; ``add``
    more terms and ``saturate`` again any time -- already-merged classes stay
    merged, so later queries reuse earlier work instead of proving from scratch.
    """

    rules: list[Axiom] = field(default_factory=list)
    graph: EGraph = field(default_factory=EGraph)
    budget: int = 1 << 12
    used: set[str] = field(default_factory=set)

    def reset_budget(self) -> None:
        """Start a fresh matching budget for another saturation pass.

        The budget bounds *one* pass, not the life of the graph: a long-lived
        knowledge base must still be able to make progress on the term it is
        handed next.
        """
        self.graph.work = 0
        self.graph.truncated = False

    def add(self, e: Expr, inputs: dict | None = None) -> int:
        """Hash-cons ``e`` (typing it first when ``inputs`` is given)."""
        if inputs is not None:
            env = {n: (Var(n, t), t) for n, t in inputs.items()}
            e = ripen(e, env=env)
            if e.type is None:
                e = ripen(e, int8, env=env)
        else:
            # no input types: a variable the term does not already type is a
            # caller mistake, not something to guess
            unknown = sorted(_untyped_vars(e))
            if unknown:
                raise KernelError(
                    f"cannot resolve type of input {unknown[0]!r}"
                )
        return self.graph.add(e)

    def saturate(self, budget: int | None = None) -> set[str]:
        self.reset_budget()
        self.used |= self.graph.saturate(self.rules, budget or self.budget)
        return self.used

    def same(self, a: Expr, b: Expr) -> bool:
        """True when the two terms are provably equal in this knowledge base."""
        ca, cb = self.graph.add(a), self.graph.add(b)
        self.saturate()
        return self.graph.find(ca) == self.graph.find(cb)

    def normal_form(self, e: Expr) -> Expr:
        c = self.graph.add(e)
        self.saturate()
        return self.graph.extract(c)
