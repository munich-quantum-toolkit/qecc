# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bell reduction (Jeon, Lee, Kim, arXiv:2610.09749, Eq. 9): a fault-equivalent ZX rewrite as a circuit post-pass.

Pattern: ancillas c (fresh |+>) and t (fresh |0>) joined by CX(c, t), which share no other gate and are each reset
and measured once. Then one of them, x, can be removed, since (<m|_x ⊗ 1)(V_x ⊗ U_y)|Φ+> = U_y V_x^T |m>: the kept
qubit y starts in the state x was measured in, runs x's gates in reverse order (CX, CZ and H are invariant under
partial transposition), then its own gates. The new circuit is the old one's m_x = 0 branch, so every old detector
stays deterministic once x's record is dropped from it. Saves a CX, a measurement and a qubit. The pair's CX may sit
anywhere in the leading runs of CNOTs that share c as control and t as target (such CNOTs commute). The caller checks
every result exactly.
"""

from __future__ import annotations

import heapq
import itertools
import operator
from typing import TYPE_CHECKING

from .circuit import MEASUREMENTS, RESETS, Circuit, Op, ops_depth
from .front import pareto

if TYPE_CHECKING:
    from collections.abc import Callable


def _safe(ok: Callable[[Circuit], bool], c: Circuit) -> bool:
    """``ok`` with every check error (non-deterministic detector, wrong state) counted as not OK."""
    try:
        return bool(ok(c))
    except (AssertionError, ValueError):
        return False


def _per_qubit(circ: Circuit) -> dict[int, list[int]]:
    """Per qubit the indices of its ops."""
    seq: dict[int, list[int]] = {}
    for i, op in enumerate(circ.ops):
        for q in op.qubits:
            seq.setdefault(q, []).append(i)
    return seq


def normalize(circ: Circuit) -> Circuit:
    """Reset directly followed by H on the same qubit -> the other reset (fewer fault locations, same ops)."""
    drop, repl = set(), {}
    for q, s in _per_qubit(circ).items():
        if len(s) >= 2 and circ.ops[s[0]].name in RESETS and circ.ops[s[1]].name == "H":
            drop.add(s[1])
            repl[s[0]] = Op("RX" if circ.ops[s[0]].name == "R" else "R", (q,))
    ops = [repl.get(k, op) for k, op in enumerate(circ.ops) if k not in drop]
    return Circuit(ops, circ.num_qubits, list(circ.data), circ.detectors)


def _in_lead_run(circ: Circuit, seq: list[int], i: int, role: int) -> bool:
    """Whether op ``i`` is in the run of CNOTs (qubit in place ``role``) right after the qubit's reset."""
    q = circ.ops[seq[0]].qubits[0]
    for k in seq[1:]:
        op = circ.ops[k]
        if op.name != "CX" or op.qubits[role] != q:
            return False
        if k == i:
            return True
    return False


def _standalone_records(circ: Circuit) -> set[int]:
    """Measurement records that are a detector on their own (a check or flag read directly)."""
    if circ.detectors is None:
        return set(range(circ.num_measurements))
    return {d[0] for d in circ.detectors if len(d) == 1}


def candidates(circ: Circuit) -> list[tuple[int, int]]:
    """(op index of the Bell-pair CX, qubit to remove) for every applicable pattern (both qubits of each pair).

    The side condition of Jeon et al.: a qubit whose record is a detector on its own is skipped (its check would be
    lost; ``prune_ancillas`` removes redundant checks more cheaply).
    """
    seq = _per_qubit(circ)
    alone = _standalone_records(circ)
    rec = {k: r for r, k in enumerate(k for k, op in enumerate(circ.ops) if op.name in MEASUREMENTS)}
    data = set(circ.data)
    out = []
    for i, op in enumerate(circ.ops):
        if op.name != "CX":
            continue
        c, t = op.qubits
        if c in data or t in data:
            continue
        sc, st = seq[c], seq[t]
        if circ.ops[sc[0]].name != "RX" or circ.ops[st[0]].name != "R":
            continue
        if not (_in_lead_run(circ, sc, i, 0) and _in_lead_run(circ, st, i, 1)):
            continue
        if any(
            circ.ops[s[-1]].name not in {"M", "MX"} or any(circ.ops[k].name in RESETS | MEASUREMENTS for k in s[1:-1])
            for s in (sc, st)
        ):
            continue  # one reset, one final measurement each
        if any(k != i and set(circ.ops[k].qubits) >= {c, t} for k in sc[1:]):
            continue  # they share another gate
        out += [(i, x) for x in (t, c) if rec[seq[x][-1]] not in alone]
    return out


def apply(circ: Circuit, i: int, x: int) -> Circuit | None:
    """Remove qubit ``x`` of the Bell pair created by op ``i``; None if the result has no consistent time order."""
    seq = _per_qubit(circ)
    c, t = circ.ops[i].qubits
    y = c if x == t else t
    sx, sy = seq[x], seq[y]
    v_ops = [k for k in sx[1:-1] if k != i]  # x's gates other than the Bell CX (which commutes to the front)
    meas = circ.ops[sx[-1]].name
    removed = {sx[0], i, sx[-1]}
    ops: dict[int, Op] = {}
    for k, op in enumerate(circ.ops):
        if k in removed:
            continue
        if k == sy[0]:
            ops[k] = Op("R" if meas == "M" else "RX", (y,))
        elif k in v_ops:
            ops[k] = Op(op.name, tuple(y if q == x else q for q in op.qubits))
        else:
            ops[k] = op
    new_seq = {q: [k for k in s if k not in removed] for q, s in seq.items() if q not in {x, y}}
    new_seq[y] = [sy[0], *reversed(v_ops), *[k for k in sy[1:] if k != i]]
    # time order from the per-qubit sequences (stable: lowest original index first)
    succ: dict[int, set[int]] = {k: set() for k in ops}
    indeg = dict.fromkeys(ops, 0)
    for s in new_seq.values():
        for a, b in itertools.pairwise(s):
            if b not in succ[a]:
                succ[a].add(b)
                indeg[b] += 1
    heap = [k for k, d in indeg.items() if d == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        k = heapq.heappop(heap)
        order.append(k)
        for b in succ[k]:
            indeg[b] -= 1
            if indeg[b] == 0:
                heapq.heappush(heap, b)
    if len(order) != len(ops):
        return None
    old_rec = {k: r for r, k in enumerate(k for k, op in enumerate(circ.ops) if op.name in MEASUREMENTS)}
    new_rec = {old_rec[k]: r for r, k in enumerate(k for k in order if ops[k].name in MEASUREMENTS)}
    dets = circ.detectors if circ.detectors is not None else [[j] for j in range(circ.num_measurements)]
    new_dets = []
    for d in dets:
        s: set[int] = set()
        for j in d:
            if j in new_rec:
                s ^= {new_rec[j]}
        if s:
            new_dets.append(sorted(s))
    return Circuit([ops[k] for k in order], circ.num_qubits, list(circ.data), new_dets)


def reduce_front(
    circ: Circuit, ok: Callable[[Circuit], bool], beam: int = 3, max_tries: int = 400
) -> list[tuple[Circuit, int]]:
    """Beam search over sequences of Bell reductions, keeping only circuits with ``ok``.

    Returns the (ops, depth) Pareto set of everything reached, excluding the start, as (circuit, number of reductions).
    """

    # items are (circuit, reductions, key): the key is computed once per circuit
    def front(items: list) -> list:
        return pareto(items, operator.itemgetter(2))

    c0 = normalize(circ)
    if not _safe(ok, c0):
        c0 = circ
    archive, level, seen, tries = [], [(c0, 0, ops_depth(c0))], set(), 0
    while level and tries < max_tries:
        nxt = []
        for c, n, _ in level:
            for i, x in candidates(c):
                r = apply(c, i, x)
                if r is None:
                    continue
                s = str(r.to_stim())
                if s in seen:
                    continue
                seen.add(s)
                tries += 1
                if _safe(ok, r):
                    nxt.append((r, n + 1, ops_depth(r)))
        archive = front(archive + nxt)
        keep = front(nxt) + sorted(nxt, key=operator.itemgetter(2))[:beam]
        level = list({id(p): p for p in keep}.values())[: 2 * beam]
    return [(c, n) for c, n, _ in archive]


def remove_qubit(circ: Circuit, q: int) -> Circuit:
    """Delete every op on ancilla ``q`` and every detector that reads one of its records."""
    meas = [op for op in circ.ops if op.name in MEASUREMENTS]
    gone = {r for r, op in enumerate(meas) if q in op.qubits}
    old_to_new = {r: n for n, r in enumerate(r for r in range(len(meas)) if r not in gone)}
    dets = circ.detectors if circ.detectors is not None else [[i] for i in range(circ.num_measurements)]
    new_dets = [[old_to_new[i] for i in d] for d in dets if not any(i in gone for i in d)]
    ops = [op for op in circ.ops if q not in op.qubits]
    return Circuit(ops, circ.num_qubits, list(circ.data), new_dets)


def prune_ancillas(circ: Circuit, ok: Callable[[Circuit], bool]) -> tuple[Circuit, list[int]]:
    """Remove measured ancillas (most ops first, with their gates and detectors) while ``ok`` holds.

    Ancillas left without a two-qubit gate go too. Returns the circuit and the removed qubits.
    """
    removed: list[int] = []
    while True:
        seq = _per_qubit(circ)
        data = set(circ.data)
        anc = [q for q, s in seq.items() if q not in data and any(circ.ops[k].name in MEASUREMENTS for k in s)]
        done = False
        for q in sorted(anc, key=lambda q: -len(seq[q])):
            trial = remove_qubit(circ, q)
            idle = [
                a
                for a, s in _per_qubit(trial).items()
                if a not in data and not any(len(trial.ops[k].qubits) == 2 for k in s)
            ]
            for a in idle:
                trial = remove_qubit(trial, a)
            if _safe(ok, trial):
                circ, done = trial, True
                removed += [q, *idle]
                break
        if not done:
            return circ, removed
