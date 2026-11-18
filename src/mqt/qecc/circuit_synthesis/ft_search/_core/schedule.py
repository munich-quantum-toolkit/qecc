# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Depth scheduling of a circuit by reordering commuting gates.

Two CNOTs sharing a qubit commute if that qubit is the control of both or the target of both. On every qubit the ops
form *groups*: maximal runs of CNOTs with the same role there; every other op is a group of its own. Any order that
keeps the group order on every qubit gives the same circuit. Moving gates changes where faults sit, so the caller
checks the result again.
"""

from __future__ import annotations

import numpy as np

from .circuit import MEASUREMENTS, Circuit


def _groups(
    circ: Circuit, fixed: set[int] | None = None
) -> tuple[dict[int, list[list[int]]], dict[int, dict[int, int]]]:
    """Per qubit its groups (op indices), and op -> {qubit: group index}; ops in ``fixed`` keep their order."""
    fixed = fixed or set()
    seq: dict[int, list[list[int]]] = {}
    gid: dict[int, dict[int, int]] = {}
    last_role: dict[int, str | None] = {}
    has_fixed: dict[tuple[int, int], bool] = {}
    for i, op in enumerate(circ.ops):
        gid[i] = {}
        for k, q in enumerate(op.qubits):
            role = ("c" if k == 0 else "t") if op.name == "CX" else None
            gs = seq.setdefault(q, [])
            can_merge = role is not None and gs and last_role.get(q) == role
            if can_merge and i in fixed and has_fixed.get((q, len(gs) - 1)):
                can_merge = False
            if can_merge:
                gs[-1].append(i)
                if i in fixed:
                    has_fixed[q, len(gs) - 1] = True
            else:
                gs.append([i])
                has_fixed[q, len(gs) - 1] = i in fixed
            last_role[q] = role
            gid[i][q] = len(gs) - 1
    return seq, gid


def schedule(circ: Circuit, *, seed: int = 0, noise: float = 0.0, keep_prep_order: bool = False) -> Circuit:
    """List schedule of ``circ`` with critical-path priorities, perturbed by ``noise`` (seeded).

    Each layer takes non-conflicting ready two-qubit gates by priority; other ops take no time. Detectors follow the
    new measurement order. ``keep_prep_order``: data-data CNOTs that share a qubit keep their relative order (the fault
    structure of the preparation circuit is unchanged).
    """
    rng = np.random.default_rng(seed)
    ops = circ.ops
    n = len(ops)
    data = set(circ.data)
    fixed = {i for i, op in enumerate(ops) if keep_prep_order and op.name == "CX" and set(op.qubits) <= data}
    seq, gid = _groups(circ, fixed)
    two = np.array([op.name in {"CX", "CZ"} for op in ops], dtype=np.int64)
    # critical path (in two-qubit gates) from each op to the end, through later groups on its qubits
    tail = np.zeros(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        best = 0
        for q, g in gid[i].items():
            if g + 1 < len(seq[q]):
                best = max(best, *(tail[j] for j in seq[q][g + 1]))
        tail[i] = two[i] + best
    prio = tail + noise * rng.random(n)
    cur = dict.fromkeys(seq, 0)  # current group per qubit
    left = {(q, g): len(seq[q][g]) for q in seq for g in range(len(seq[q]))}
    done = np.zeros(n, dtype=bool)
    out: list[int] = []

    def ready(i: int) -> bool:
        return all(cur[q] == g for q, g in gid[i].items())

    def finish(i: int) -> None:
        done[i] = True
        out.append(i)
        for q, g in gid[i].items():
            left[q, g] -= 1
            if left[q, g] == 0:
                cur[q] = g + 1

    while len(out) < n:
        # zero-time ops first
        progressed = True
        while progressed:
            progressed = False
            for i in range(n):
                if not done[i] and not two[i] and ready(i):
                    finish(i)
                    progressed = True
        cand = [i for i in range(n) if not done[i] and two[i] and ready(i)]
        if not cand and len(out) < n:
            msg = "scheduler stuck"
            raise RuntimeError(msg)
        cand.sort(key=lambda i: -prio[i])
        busy: set[int] = set()
        layer = []
        for i in cand:
            if busy.isdisjoint(ops[i].qubits):
                layer.append(i)
                busy.update(ops[i].qubits)
        for i in layer:
            finish(i)
    new_ops = [ops[i] for i in out]
    old_rec = {}
    r = 0
    for i, op in enumerate(ops):
        if op.name in MEASUREMENTS:
            old_rec[i] = r
            r += 1
    new_rec = {}
    r = 0
    for i in out:
        if ops[i].name in MEASUREMENTS:
            new_rec[old_rec[i]] = r
            r += 1
    dets = None if circ.detectors is None else [[new_rec[x] for x in d] for d in circ.detectors]
    if circ.detectors is None:
        # implicit one-detector-per-record: keep the old meaning explicitly
        dets = [[new_rec[k]] for k in range(r)]
    return Circuit(new_ops, circ.num_qubits, list(circ.data), dets)
