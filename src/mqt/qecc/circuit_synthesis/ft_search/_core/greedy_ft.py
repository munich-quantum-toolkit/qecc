# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Efficiency-greedy fault-tolerant verification: add the check with the best net gain per cost.

Each step ranks the candidates (stabilizer checks at every layer boundary, flags on existing checks) by their coverage
of the failing combinations, inserts the best ``top_k`` for an exact trial, and keeps the one with the best
gain / (Δops + λ · max(Δdepth, 0) + MU · new failing classes); the gain is sum_s W^(t-s) times the drop of the raw
failures of order s (an order whose count rises adds 0). Repeat until FT. The scope's blocks stay as they are.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .circuit import metrics
from .gauge import CheckCoverage, Gadget, apply_candidate, build, candidate_space
from .verify import CSSState, check_added, check_ft, check_noiseless

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .circuit import Circuit

logger = logging.getLogger(__name__)

WEIGHT_BASE = 1000.0  # weight of a failing combination of s faults: WEIGHT_BASE ** (t - s)
MU = 3.0  # cost of each new failing class (about a flag)
W2_MAX = 4  # largest weight of a two-time check


def greedy_ft(
    base_in: Circuit,
    state: CSSState,
    t: int,
    *,
    lam: float = 0.0,
    top_k: int = 30,
    max_steps: int = 80,
    blocks: list | None = None,
    gadgets0: list[Gadget] | None = None,
    layer: bool = True,
    positions: list[int] | None = None,
    late: bool = False,
    by_order: bool = False,
) -> tuple[Circuit, list[Gadget], bool]:
    """Add checks and flags to the prep greedily until it is FT; returns the circuit, its gadgets and whether it is FT.

    ``gadgets0``: start from these gadgets (relative to the layered base, see ``gauge.promote``). ``positions``:
    candidate positions instead of every layer boundary. ``late``: checks at the end of the prep become late checks
    (after all blocks). Candidates are ranked on a sample of the failing sets and the ``top_k`` best get an exact
    trial; every flag that the ranking cannot score gets one. ``by_order``: fault orders one at a time (Criger's
    incremental schedule), so most checks are chosen against the short lists of the low orders; flags on checks of
    the prep are ranked with the checks (by frame probes), and the trials go further down the ranking while none has
    gained; trials by ``verify.check_added`` where it applies.
    """
    space = candidate_space(base_in, state, layer=layer, positions=positions)
    base, end, positions = space.base, space.end, space.positions
    nd = len(base.data)
    dpos = {d: i for i, d in enumerate(base.data)}

    def images(kind: str, p1: int) -> Iterator[tuple[int, np.ndarray]]:
        """Forward images of single-qubit Paulis from ``p1`` to each later position p2.

        Yields (p2, matrix): row q of the (nd x nd) matrix is the image of P_q at p2. The matrix is one buffer, updated
        in place: the images of all position pairs can take several GB.
        """
        cur = np.eye(nd, dtype=np.uint8)
        k = p1
        for p2 in positions[positions.index(p1) + 1 :]:
            while k < p2:
                op = base.ops[k]
                if op.name == "CX" and op.qubits[0] in dpos and op.qubits[1] in dpos:
                    c, tq = dpos[op.qubits[0]], dpos[op.qubits[1]]
                    if kind == "X":  # X on control spreads to target
                        cur[:, tq] ^= cur[:, c]
                    else:  # Z on target spreads to control
                        cur[:, c] ^= cur[:, tq]
                k += 1
            yield p2, cur

    def full(gs: list[Gadget]) -> tuple[Circuit, dict[int, int]]:
        return build(base, gs, blocks)

    def evaluate(gs: list[Gadget]) -> tuple[dict, int, dict, Circuit] | None:
        """A trial's raw failures, its debt (the increase of failing key sets), metrics and circuit; None if invalid."""
        c, _ = full(gs)
        if any(g.slots is not None for g in gs):
            try:
                # couplings at several times may cross non-commuting couplings of other checks
                check_noiseless(c, state)
            except Exception:  # ruff: ignore[blind-except]
                return None
        delta = check_added(c, state, t_cur, circ, rep) if by_order else None
        if delta is not None:
            raw2, fails2 = delta["raw"], delta["failures"]
        else:
            r2 = check_ft(c, state, t_cur)
            raw2, fails2 = dict(r2.raw_failures), r2.failures
        debt = sum(max(fails2.get(s, 0) - rep.failures.get(s, 0), 0) for s in range(1, t_cur + 1))
        return raw2, debt, metrics(c), c

    def efficiency(raw0: dict, raw2: dict, debt: int, d_ops: int, d_depth: int) -> tuple[float, float]:
        """Gain (weighted removed raw failures) and efficiency; each new failing class costs ``MU``."""
        gain = sum(WEIGHT_BASE ** (t_cur - s) * max(raw0.get(s, 0) - raw2.get(s, 0), 0) for s in range(1, t_cur + 1))
        return gain, gain / (d_ops + lam * d_depth + MU * debt)

    gadgets: list[Gadget] = [g.copy() for g in gadgets0 or []]
    next_fid = 1 + max((fid for g in gadgets for *_, fid in g.flags), default=-1)
    t_cur = 1 if by_order else t
    for step in range(max_steps + 1):
        circ, where = full(gadgets)
        rep = check_ft(circ, state, t_cur, collect=True)
        while rep.ok and t_cur < t:
            t_cur += 1
            rep = check_ft(circ, state, t_cur, collect=True)
        m = metrics(circ)
        if rep.ok:
            return circ, gadgets, True
        if step == max_steps:
            break
        # pre-filter by coverage; flags are scored on checks of the prep only (elsewhere their ops differ)
        tags = set(circ.tags or ())
        flaggable = [
            gi
            for gi, g in enumerate(gadgets)
            if by_order
            and g.block is None
            and not g.late
            and not g.drive
            and all(("c", gi, j) in tags for j in range(len(g.support)))
        ]
        cov = CheckCoverage(space, circ, where, gadgets, state, rep, flag_on=flaggable, seed=step)
        cb = cov.combos  # possibly a sample: the weights keep the gain unbiased
        sw = np.array([WEIGHT_BASE ** (t_cur - int(n)) for n in cb.ns]) * cb.scale

        # (gain / cost, candidate)
        pre = [
            (g / c, cand)
            for cands, comb, cost in cov.stabilizers()
            for cand, g, c in zip(cands, sw @ comb, cost, strict=True)
        ]
        for kind in ("Z", "X"):
            comp = {p: cov.frame(p, kind).astype(np.int64) for p in positions}
            for p1 in positions:
                for p2, mat in images(kind, p1):
                    w = 1 + mat.sum(axis=1)
                    okq = np.nonzero(w <= W2_MAX)[0]
                    if not len(okq):
                        continue
                    b1 = comp[p1][:, okq]  # parity of P_q at p1 = frame component on q
                    b2 = (comp[p2] @ mat[okq].T.astype(np.int64)) % 2
                    gain = sw @ cb.combine((b1 ^ b2) % 2)
                    for col in np.nonzero(gain > 0)[0]:
                        q = int(okq[col])
                        pre.append((gain[col] / (w[q] + 1), ("T", kind, p1, p2, q)))
        for cands, comb, cost in cov.flags(cost=5):  # a flag: reset, 2 CNOTs, measurement
            pre.extend((g / c, cand) for cand, g, c in zip(cands, sw @ comb, cost, strict=True))
        pre.sort(key=lambda x: -x[0])
        seen, shortlist = set(), []
        for _, c in pre:
            if c not in seen:
                seen.add(c)
                shortlist.append(c)
        # the flags the probes cannot score: an exact trial each, every step
        unscored = [
            ("F", gi, j, k)
            for gi, g in enumerate(gadgets)
            if gi not in flaggable
            for j in range(len(g.support) - 1)
            for k in range(j + 2, len(g.support) + 1)
            if not any((j, k) == (a, b) for a, b, _ in g.flags)
        ]

        def apply(gs: list[Gadget], c: tuple, fid: int) -> list[Gadget]:
            img = None
            if c[0] == "T":
                _, kind, p1, p2, _ = c
                img = {kind: {p1: {p2: next(m.copy() for p, m in images(kind, p1) if p == p2)}}}
            return apply_candidate(gs, c, fid, late_at=end if late else None, img=img)

        raw0 = dict(rep.raw_failures)
        best, scored = None, []
        # exact trials in rank order: the ``top_k`` best, and (``by_order``) further ones while none has gained
        for idx, c in enumerate(shortlist + unscored):
            if idx < len(shortlist) and (
                idx >= (10 * top_k if by_order else top_k) or (idx >= top_k and best is not None)
            ):
                continue
            trial = apply(gadgets, c, next_fid)
            ev = evaluate(trial)
            if ev is None:
                continue
            raw2, debt, m2, _ = ev
            d_ops = m2["ops"] - m["ops"]
            d_depth = max(m2["depth"] - m["depth"], 0)
            gain, eff = efficiency(raw0, raw2, debt, d_ops, d_depth)
            scored.append((eff, c))
            if gain <= 0:
                continue
            if best is None or eff > best[0]:
                best = (eff, [c], trial, d_ops, d_depth)
        if best is None:
            # stuck: two-step lookahead over the best single candidates (e.g. a check plus a flag for its hooks)
            scored.sort(key=lambda x: -x[0])
            top = [c for _, c in scored[:12]]
            for i, c1 in enumerate(top):
                t1 = apply(gadgets, c1, next_fid)
                fid2 = next_fid + (c1[0] == "F")
                # second move: any top candidate, or a flag on the gadget just added
                seconds = top[i + 1 :]
                if c1[0] in {"S", "T", "G"}:
                    gi = len(t1) - 1
                    nsup = len(t1[gi].support)
                    seconds += [("F", gi, j, k) for j in range(nsup - 1) for k in range(j + 2, nsup + 1)]
                for c2 in seconds:
                    t2 = apply(t1, c2, fid2)
                    ev = evaluate(t2)
                    if ev is None:
                        continue
                    raw2, debt, m2, _ = ev
                    d_ops = m2["ops"] - m["ops"]
                    d_depth = max(m2["depth"] - m["depth"], 0)
                    gain, eff = efficiency(raw0, raw2, debt, d_ops, d_depth)
                    if gain <= 0:
                        continue
                    if best is None or eff > best[0]:
                        best = (eff, [c1, c2], t2, d_ops, d_depth)
        if best is None:
            logger.debug("greedy: no candidate (or pair) with positive net gain")
            break
        _, cs, gadgets, _, _ = best
        next_fid += sum(1 for c in cs if c[0] == "F")
    circ, _ = full(gadgets)
    return circ, gadgets, False
