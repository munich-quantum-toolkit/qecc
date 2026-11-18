# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Edit moves on existing designs: coupling order, removal of checks, flags or blocks, moving a prep CNOT."""

from __future__ import annotations

import copy
import dataclasses
import itertools
import operator
import random
from typing import TYPE_CHECKING

import numpy as np

from ..circuit import Circuit, metrics
from ..config import REMOVE_K, REMOVE_PAIRS
from ..design import Design, Node, compile_node, scope_name
from ..evaluation import score_cached
from ..verify import check_ft, weighted_failures
from .common import removable, remove

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from ..circuit import Op
    from ..config import Context
    from ..gauge import Gadget


ORDER_TRIES = 24  # coupling order: random restarts per check and sweep
ORDER_SWEEPS = 3
ORDER_EXHAUSTIVE = 6  # equal-time groups up to this size are enumerated fully


class OrderSolver:
    """Coupling order inside the scope's bare checks, which decides the hooks a check produces.

    Local search per check of weight >= 3 over permutations inside its equal-time groups (small groups exhaustively,
    then pairwise swaps and random restarts), plus moves of whole checks in the check list. A change is kept if the
    scope's weighted failure count drops.
    """

    ctx_deps = ()
    objective = "failures"  # DepthOrderSolver: "depth"

    name = "order"

    def propose(
        self,
        design: Design,
        path: tuple[int, ...],
        ctx: Context,  # ruff: ignore[unused-method-argument]  (solver interface)
    ) -> Iterator[tuple[str, Node]]:
        """Yield the scope with a better coupling order."""
        node = design.node(path)
        late = node.late_gadgets
        if node.opaque or not any(len(g.support) >= 3 for g in node.prep_gadgets):
            return
        t = design.t
        rng = np.random.default_rng(0)
        trial = copy.copy(node)

        depth_mode = self.objective == "depth"

        def score(gs: list, bound: float | None = None) -> float | None:
            """The order's score (lower is better); None if not deterministic or, in depth mode, not FT or not below ``bound``.

            The depth bound is tested before the costly exact check.
            """
            trial.gadgets = [*gs, *late]
            c = compile_node(trial)
            if depth_mode:
                dep = float(metrics(c)["depth"])
                if bound is not None and dep >= bound:
                    return None
            try:
                c.to_stim().detector_error_model()
            except ValueError:
                return None
            rep = check_ft(c, node.target, t)
            if depth_mode:  # only FT orders count; minimise depth
                return dep if rep.ok else None
            return weighted_failures(rep.raw_failures, t)

        best = [g.copy() for g in node.prep_gadgets]
        bs = score(best)
        if bs is None or (bs == 0 and not depth_mode):
            return
        s0 = bs

        def reordered(g: Gadget, idx: Sequence[int]) -> Gadget:
            sl = g.coupling_slots()
            return dataclasses.replace(
                g,
                support=[g.support[j] for j in idx],
                flags=list(g.flags),
                slots=[sl[j] for j in idx] if g.slots is not None else None,
                ticks=[g.ticks[j] for j in idx] if g.ticks is not None else None,
            )

        def groups_of(g: Gadget) -> list[list[int]]:
            sl = g.coupling_slots()
            return [[j for j in range(len(sl)) if sl[j] == v] for v in sorted(set(sl))]

        for _ in range(ORDER_SWEEPS):
            improved = False
            for gi in range(len(best)):
                if len(best[gi].support) < 3:
                    continue
                exhaust = True
                while exhaust and bs > 0:
                    exhaust = False
                    g = best[gi]
                    for grp in groups_of(g):
                        if not 2 <= len(grp) <= ORDER_EXHAUSTIVE:
                            continue
                        for perm in itertools.permutations(grp):
                            idx = list(range(len(g.support)))
                            for slot_j, src in zip(grp, perm, strict=True):
                                idx[slot_j] = src
                            cand = list(best)
                            cand[gi] = reordered(g, idx)
                            sc = score(cand, bs)
                            if sc is not None and sc < bs:
                                best, bs, exhaust, improved = cand, sc, True, True
                                g = best[gi]
                                break
                # hill climbing over pairwise swaps inside equal-time groups (first improvement), then random restarts
                climbing = True
                while climbing and bs > 0:
                    climbing = False
                    g = best[gi]
                    for grp in groups_of(g):
                        for a_ in range(len(grp)):
                            for b_ in range(a_ + 1, len(grp)):
                                idx = list(range(len(g.support)))
                                idx[grp[a_]], idx[grp[b_]] = idx[grp[b_]], idx[grp[a_]]
                                cand = list(best)
                                cand[gi] = reordered(g, idx)
                                sc = score(cand, bs)
                                if sc is not None and sc < bs:
                                    best, bs, climbing, improved = cand, sc, True, True
                                    break
                            if climbing:
                                break
                        if climbing:
                            break
                for _ in range(ORDER_TRIES):
                    if bs == 0:
                        break
                    g = best[gi]
                    idx = [j for grp in groups_of(g) for j in rng.permutation(grp)]
                    cand = list(best)
                    cand[gi] = reordered(g, idx)
                    sc = score(cand, bs)
                    if sc is not None and sc < bs:
                        best, bs, improved = cand, sc, True
            # interleaving between checks: at a shared time point couplings run in check-list order, so try
            # moving each check to every other list position (first improvement)
            moved = True
            while moved and bs > 0:
                moved = False
                for a_ in range(len(best)):
                    for b_ in range(len(best)):
                        if a_ == b_:
                            continue
                        cand = list(best)
                        cand.insert(b_, cand.pop(a_))
                        sc = score(cand, bs)
                        if sc is not None and sc < bs:
                            best, bs, moved, improved = cand, sc, True, True
                            break
                    if moved:
                        break
            if not improved or bs == 0:
                break
        if bs < s0:
            nd = copy.copy(node)
            nd.gadgets = [*best, *late]
            yield f"{self.name if depth_mode else 'order'}[{s0:.0f}->{bs:.0f}]", nd


class NeighbourhoodSolver:
    """Large-neighbourhood move: remove one or two elements anywhere in the design so that it is no longer FT.

    The search then completes it again, which leads out of plateaus of near-identical designs. Proposes the
    ``REMOVE_K`` removals that save most, single ones first, then ``REMOVE_PAIRS`` sampled pairs; removals that keep
    the design FT are left to ``PruneSolver``.
    """

    ctx_deps = ()
    name = "remove"

    def propose(  # ruff: ignore[no-self-use]  (solver interface)
        self,
        design: Design,
        path: tuple[int, ...],
        ctx: Context,  # ruff: ignore[unused-method-argument]  (solver interface)
    ) -> Iterator[tuple[str, Node]]:
        """Yield the design with one or two elements removed."""
        if path or design.root.opaque:
            return
        elems = []  # (saving, label, (scope path, element))
        for sp, nd in design.nodes():
            if not nd.opaque:
                where = scope_name(sp)
                found = removable(nd, block_saving=lambda spec: 3 * len(spec.support))
                elems += [(sv, f"{lab}@{where}", (sp, el)) for sv, lab, el in found]
        if not elems:
            return
        rng = random.Random(design.key())  # ruff: ignore[suspicious-non-cryptographic-random-usage]  (sampling only)
        combos = [[e] for e in sorted(elems, key=lambda e: -e[0])]
        others = [e for e in elems if e[2][1][0] != "f"]
        for _ in range(REMOVE_PAIRS if len(others) > 1 else 0):
            a, b = rng.sample(others, 2)
            if a[2][0] != b[2][0] or a[2][1] != b[2][1]:
                combos.append([a, b])

        def apply(d: Design, sel: list) -> Design | None:
            new = d.clone()
            # remove from the highest index down so earlier indices stay valid; blocks may contain other targets
            for _, _, (sp, el) in sorted(sel, key=lambda e: (len(e[2][0]), e[2][1]), reverse=True):
                try:
                    nd = new.node(sp)
                except (IndexError, AttributeError):
                    return None
                if not remove(nd, el):
                    return None
            return new

        n_out = 0
        for sel in combos:
            if n_out >= REMOVE_K:
                break
            d = apply(design, sel)
            if d is None:
                continue
            sc = score_cached(d.circuit(), d.root.target, d.t)
            if sc is None or sc == 0.0:  # ruff: ignore[float-equality-comparison]  (0.0 exactly when FT)
                continue
            n_out += 1
            yield f"remove[-{', -'.join(e[1] for e in sel)}]", d.root


def _commute_ops(a: Op, b: Op) -> bool:
    if a.name != "CX" or b.name != "CX":
        return not (set(a.qubits) & set(b.qubits))
    (c1, t1), (c2, t2) = a.qubits, b.qubits
    return c1 != t2 and c2 != t1


def prep_moves(base: Circuit) -> list[tuple[int, int]]:
    """Moves (i, j), i < j: CNOT j moved to just before op i, past ops it commutes with.

    Only moves past a CNOT with the same control or target, as only those change how faults spread.
    """
    out = []
    for j, b in enumerate(base.ops):
        if b.name != "CX":
            continue
        for i in range(j - 1, -1, -1):
            a = base.ops[i]
            if not _commute_ops(a, b):
                break
            if a.name == "CX" and (a.qubits[0] == b.qubits[0] or a.qubits[1] == b.qubits[1]):
                out.append((i, j))
    return out


def apply_prep_move(node: Node, move: tuple[int, int]) -> Node:
    """A copy of ``node`` with the move applied; gadgets and mid-prep blocks stay at the ops they sat before."""
    i, j = move
    new = copy.deepcopy(node)
    ops = list(new.base.ops)
    ops.insert(i, ops.pop(j))
    new.base = Circuit(ops, new.base.num_qubits, list(new.base.data), new.base.detectors)
    new.layered = False

    def shift(p: int) -> int:
        return p + 1 if i < p <= j else p

    new.gadgets = [
        g
        if g.late
        else dataclasses.replace(
            g, pos=shift(g.pos), slots=[shift(x) for x in g.slots] if g.slots is not None else None
        )
        for g in new.gadgets
    ]
    new.blocks = [
        (dataclasses.replace(spec, pos=shift(spec.pos)) if spec.pos is not None else spec, ch)
        for spec, ch in new.blocks
    ]
    return new


PREP_K = 4  # prep moves: branches per design
PREP_MOVES = 300  # prep moves tried at most


class PrepMoveSolver:
    """Move a CNOT of the data preparation circuit earlier and keep the checks: the state stays, the fault spread changes.

    Partial design: the ``PREP_K`` moves that lower the weighted failure count most. FT design: moves that keep it
    FT, each followed by ``PruneSolver``, the ``PREP_K`` cheapest under the run's weights. At most ``PREP_MOVES``
    moves tried.
    """

    ctx_deps = ("lam", "lams")
    name = "prepmove"

    def propose(  # ruff: ignore[no-self-use]  (solver interface)
        self, design: Design, path: tuple[int, ...], ctx: Context
    ) -> Iterator[tuple[str, Node]]:
        """Yield the scope with a moved prep CNOT."""
        node = design.node(path)
        if node.opaque:
            return
        moves = prep_moves(node.base)
        if not moves:
            return
        if len(moves) > PREP_MOVES:
            step = len(moves) / PREP_MOVES
            moves = [moves[int(i * step)] for i in range(PREP_MOVES)]

        def weighted(d: Design) -> float | None:
            return score_cached(d.circuit(), d.root.target, d.t)

        w0 = weighted(design)
        if w0 is None:
            return
        scored = []
        if w0 > 0:  # partial design: fewer failures
            for mv in moves:
                d = design.with_node(path, apply_prep_move(node, mv))
                w = weighted(d)
                if w is not None and w < w0:
                    scored.append((w, mv, d, f"failures {w0:.0f}->{w:.0f}"))
        else:  # FT design: keep FT, then prune; cheaper or shallower under any of the run's λs
            from .polish import PruneSolver  # ruff: ignore[import-outside-top-level]  (polish imports this module)

            m0 = metrics(design.circuit())
            for mv in moves:
                d = design.with_node(path, apply_prep_move(node, mv))
                if weighted(d) != 0.0:  # ruff: ignore[float-equality-comparison]  (0.0 exactly when FT)
                    continue
                pr = list(PruneSolver().propose(d, path, ctx))
                if pr:
                    d = d.with_node(path, pr[0][1])
                m = metrics(d.circuit())
                if any(ctx.cost(m, lam) < ctx.cost(m0, lam) for lam in ctx.all_lams()):
                    scored.append((ctx.cost(m), mv, d, f"-> {m['ops']}/{m['depth']}" + (" after prune" if pr else "")))
            if scored:  # the best per λ first, then the rest by main cost
                per_lam = ctx.best_per_lam(scored, lambda x: metrics(x[2].circuit()))
                scored = per_lam + [
                    x for x in sorted(scored, key=operator.itemgetter(0)) if not any(x is y for y in per_lam)
                ]
        for _, mv, d, note in (sorted(scored, key=operator.itemgetter(0)) if w0 > 0 else scored)[:PREP_K]:
            yield f"prepmove[CX {mv[1]} before op {mv[0]}; {note}]", d.node(path)
