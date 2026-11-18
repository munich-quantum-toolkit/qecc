# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Polish passes that keep a design FT: depth-aware coupling order, shared flags, redundancy removal, flag takeover.

They compress a design and never add new structure. ``PolishSolver`` runs them, then the circuit stage.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import operator
import time
from typing import TYPE_CHECKING

from ..bellred import prune_ancillas, reduce_front
from ..circuit import asap, metrics, ops_depth
from ..design import Design, Node, compile_node
from ..evaluation import ft_cached, score_cached
from ..front import pareto
from ..schedule import schedule
from .common import removable, remove
from .edit import OrderSolver

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..circuit import Circuit
    from ..config import Context
    from ..verify import CSSState


class DepthOrderSolver(OrderSolver):
    """The moves of ``OrderSolver`` on an FT design, kept if the design stays FT and its depth drops."""

    objective = "depth"
    name = "orderdepth"


class SharedFlagSolver:
    """Shared flags (Chao & Reichardt, Chamberland style): merge two flags of the scope's checks into one flag qubit.

    Each merge saves a measurement and a qubit. Merges are applied greedily while the scope's weighted failure count
    does not grow; the exact check rejects merges that let two flagged faults cancel. Flags merge only among checks in
    the preparation circuit or only among late checks.
    """

    ctx_deps = ()
    name = "sharedflag"

    def propose(  # ruff: ignore[no-self-use]  (solver interface)
        self,
        design: Design,
        path: tuple[int, ...],
        ctx: Context,  # ruff: ignore[unused-method-argument]  (solver interface)
    ) -> Iterator[tuple[str, Node]]:
        """Yield the scope with shared flags."""
        node = design.node(path)
        if node.opaque:
            return
        t = design.t

        def score(gs: list) -> float | None:
            trial = copy.copy(node)
            trial.gadgets = gs
            return score_cached(compile_node(trial), node.target, t)

        def merges(gs: list) -> Iterator[tuple[bool, int, int]]:
            """All (late, a, b) flag pairs that may merge: same group (in the preparation circuit or late) and type."""
            by: dict = {}
            for g in gs:
                for *_, fid in g.flags:
                    by.setdefault((g.late, g.kind), set()).add(fid)
            for (late, _kind), fset in by.items():
                ids = sorted(fset)
                for i_ in range(len(ids)):
                    for k_ in range(i_ + 1, len(ids)):
                        yield late, ids[i_], ids[k_]

        def merged(gs: list, late: bool, a: int, b: int) -> list:
            return [
                g
                if g.late != late
                else dataclasses.replace(g, flags=[(j, k, a if f == b else f) for j, k, f in g.flags])
                for g in gs
            ]

        gs = list(node.gadgets)
        base = score(gs)
        if base is None:
            return
        n_merges, changed = 0, True
        while changed:
            changed = False
            for late, a, b in merges(gs):
                cand = merged(gs, late, a, b)
                sc = score(cand)
                if sc is not None and sc <= base:
                    gs, n_merges, changed = cand, n_merges + 1, True
                    break
        if n_merges:
            nd = copy.copy(node)
            nd.gadgets = gs
            yield f"sharedflag[-{n_merges} flag qubits]", nd


class PruneSolver:
    """Redundancy removal on FT designs: later checks can make earlier ones unnecessary.

    Tries deleting each element of ``removable``, largest saving first, and keeps a deletion if the whole design stays FT.
    """

    ctx_deps = ()
    name = "prune"

    def propose(  # ruff: ignore[no-self-use]  (solver interface)
        self,
        design: Design,
        path: tuple[int, ...],
        ctx: Context,  # ruff: ignore[unused-method-argument]  (solver interface)
    ) -> Iterator[tuple[str, Node]]:
        """Yield the design with the redundant elements removed."""

        def is_ft(d: Design) -> bool:
            return ft_cached(d.circuit(), d.root.target, d.t)

        if not is_ft(design):
            return
        cur = design.clone()
        removed = []
        ops0 = metrics(cur.circuit())["ops"]
        while True:
            cands = removable(cur.node(path))
            done = False
            for _, lab, elem in sorted(cands, key=lambda c: -c[0]):
                trial = cur.clone()
                remove(trial.node(path), elem)
                if is_ft(trial):
                    cur, done = trial, True
                    removed.append(lab)
                    break
            if not done:
                break
        if not removed:
            return
        yield f"prune[-{ops0 - metrics(cur.circuit())['ops']} ops: {', '.join(removed)}]", cur.node(path)


TAKEOVER_TRIALS = 60  # flag takeover: exact checks per call at most
POLISH_ROUNDS = 3  # rounds of each polish stage (t >= 3: at most 2)
POLISH_TRIES = 6  # reschedules per circuit in the circuit stage (t >= 3: a third)


class TakeoverSolver:
    """Flag takeover on FT designs (flag at origin, Forlivesi & Amaro, arXiv:2508.14200).

    Between its two CNOTs a flag holds a copy of its check's ancilla, so the flag can drive couplings of the check in
    its span (``Gadget.drive``): same gates, but the fan-out runs on two qubits at once. Greedy: apply the takeover that
    lowers (depth, sum of finish times) most and keeps the design FT; repeat. At most ``TAKEOVER_TRIALS`` exact checks.
    """

    ctx_deps = ()
    name = "takeover"

    @staticmethod
    def _shape(c: Circuit) -> tuple[int, int]:
        fin: dict[int, int] = {}
        for op, (_, e) in zip(c.ops, asap(c), strict=True):
            for q in op.qubits:
                fin[q] = max(fin.get(q, 0), e)
        return max(fin.values(), default=0), sum(fin.values())

    def propose(
        self,
        design: Design,
        path: tuple[int, ...],
        ctx: Context,  # ruff: ignore[unused-method-argument]  (solver interface)
    ) -> Iterator[tuple[str, Node]]:
        """Yield the scope with flag takeovers applied."""
        node = design.node(path)
        if node.opaque or not any(g.flags for g in node.gadgets):
            return
        if not ft_cached(design.circuit(), design.root.target, design.t):
            return
        users: dict[int, int] = {}
        for g in node.gadgets:
            for *_, fid in g.flags:
                users[fid] = users.get(fid, 0) + 1
        cur, shape0 = design, self._shape(design.circuit())
        shape, n_moves, trials = shape0, 0, 0
        while trials < TAKEOVER_TRIALS:
            nd = cur.node(path)
            cands = []
            for gi, g in enumerate(nd.gadgets):
                taken = g.driven()
                for fj, fk, fid in g.flags:
                    if users[fid] > 1:
                        continue
                    for j in range(fj, min(fk, len(g.support))):
                        if j in taken:
                            continue
                        new = copy.copy(nd)
                        new.gadgets = [
                            *nd.gadgets[:gi],
                            dataclasses.replace(g, drive=(*g.drive, (j, fid))),
                            *nd.gadgets[gi + 1 :],
                        ]
                        d = cur.with_node(path, new)
                        s = self._shape(d.circuit())
                        if s < shape:
                            cands.append((s, d))
            accepted = False
            for s, d in sorted(cands, key=operator.itemgetter(0)):
                if trials >= TAKEOVER_TRIALS:
                    break
                trials += 1
                if ft_cached(d.circuit(), d.root.target, d.t):
                    cur, shape, accepted = d, s, True
                    n_moves += 1
                    break
            if not accepted:
                break
        if shape[0] < shape0[0]:
            yield f"takeover[{n_moves} couplings; depth {shape0[0]}->{shape[0]}]", cur.node(path)


class PolishSolver:
    """The FT-preserving polish: a structured stage on the design's scopes, then a circuit stage on the circuit."""

    @staticmethod
    def polish(design: Design, ctx: Context, passes: list, stop: float = math.inf) -> Design:
        """Structured stage: ``passes`` on every scope, repeated while anything improves, until ``stop`` (wall clock)."""
        cur = design
        for _ in range(POLISH_ROUNDS if design.t <= 2 else min(POLISH_ROUNDS, 2)):
            changed = False
            for sv in passes:
                for path in [q for q, _ in cur.nodes()]:
                    if time.time() > stop:
                        return cur
                    try:
                        node = cur.node(path)  # an earlier pass (prune) may have removed this scope
                    except (IndexError, AttributeError):
                        continue
                    if node.opaque:
                        continue
                    for _, nd in sv.propose(cur, path, ctx):
                        cand = cur.with_node(path, nd)
                        if ft_cached(cand.circuit(), cand.root.target, cand.t):
                            m0, m1 = metrics(cur.circuit()), metrics(cand.circuit())
                            if (m1["ops"], m1["depth"]) <= (m0["ops"], m0["depth"]) and m1 != m0:
                                cur, changed = cand, True
                        break
            if not changed:
                break
        return cur

    @staticmethod
    def circuit_variants(
        circ: Circuit,
        state: CSSState,
        t: int,
        stop: float = math.inf,
    ) -> list:
        """Circuit stage: rescheduled, ancilla-pruned and Bell-reduced FT versions of a circuit, itself included."""

        def ok(c: Circuit) -> bool:
            return ft_cached(c, state, t)

        # t >= 3: fewer reschedules, a smaller Bell beam and two rounds bound the time
        tries, beam, rounds = (
            (POLISH_TRIES, 4, POLISH_ROUNDS) if t <= 2 else (max(2, POLISH_TRIES // 3), 2, min(POLISH_ROUNDS, 2))
        )

        def resched(c: Circuit) -> Circuit:
            best, bd = c, metrics(c)["depth"]
            for keep in (True, False):
                for k in range(tries):
                    if time.time() > stop:
                        return best
                    try:
                        s_ = schedule(c, seed=k, noise=0.0 if k == 0 else 0.9, keep_prep_order=keep)
                    except Exception:  # ruff: ignore[blind-except, try-except-continue]  (skip a failed schedule)
                        continue
                    if metrics(s_)["depth"] < bd and ok(s_):
                        best, bd = s_, metrics(s_)["depth"]
            return best

        def bell(c: Circuit) -> list[Circuit]:
            return [x for x, _ in reduce_front(c, ok, beam=beam)] if time.time() < stop else []

        def cprune(c: Circuit) -> list[Circuit]:
            return [prune_ancillas(c, ok)[0]] if time.time() < stop else []

        pool, frontier = {str(circ.to_stim()): circ}, [circ]
        for _ in range(rounds):
            new = []
            for c in frontier:
                if time.time() > stop:
                    break
                for v in [resched(c), *cprune(c), *bell(c)]:
                    key = str(v.to_stim())
                    if key not in pool:
                        pool[key] = v
                        new.append(v)
            if not new:
                break
            # next round only from the (ops, depth) front of the new circuits, at most 4
            frontier = pareto(new, ops_depth)[:4]
        return list(pool.values())
