# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Preparation circuits by greedy column elimination, and ancilla preps guided against cancellation.

Greedy column elimination: column operations col_t ^= col_c (a CNOT c->t read backwards) on the generator matrix until
only rank(h) columns are nonzero; the prep is H on the surviving columns followed by the elimination reversed.

Guided ancilla preps (after Weilandt, Peham & Wille, arXiv:2601.13313): ancilla faults of the type a block detects
only flip its outcomes, so together with a dangerous data error of the same block syndrome they leave the error
undetected (cancellation). The guided prep rejects each CNOT whose new ancilla fault, alone or with one earlier fault,
has the block syndrome of a dangerous data error of the faults before the block, and backtracks if no CNOT passes.
"""

from __future__ import annotations

import collections
import copy
from typing import TYPE_CHECKING

import numpy as np

from .circuit import Circuit, Op
from .design import compile_node
from .faults import fault_table
from .verify import _pack, gf2_row_basis, leader_weights

if TYPE_CHECKING:
    from collections.abc import Callable

    from .blocks import BlockSpec
    from .design import Node
    from .verify import CSSState


def guided_prep(
    h: np.ndarray,
    qvals: list[int] | None = None,
    bad: Callable[[int, int], bool] | None = None,
    track_z: bool = False,
    *,
    seed: int = 0,
    layered: bool = True,
    eps: float = 0.0,
    max_back: int = 400,
    pairs: bool = True,
) -> Circuit | None:
    """Prep of the state with X-type generator ``h`` (uniform superposition over rowspan(h)) by column elimination.

    Args:
        h: The X-type generator matrix.
        qvals: Per qubit k the value (an int, XOR-linear) of a tracked fault on k at the end of the prep.
        bad: ``bad(v, s)``: is a combination of s tracked faults with value v forbidden. None: no fault is tracked
            (greedy column elimination).
        track_z: The tracked faults are Z (they spread from target to control), else X.
        seed: Seed of the tie breaking and the random moves.
        layered: Prefer CNOTs on columns not used in the current layer.
        eps: The probability that a step takes a random improving CNOT instead of the best one.
        max_back: Maximum number of backtracking steps.
        pairs: Also reject a CNOT whose fault forms a forbidden pair with an earlier fault.

    Returns:
        The circuit, or None if the search fails.
    """
    rng = np.random.default_rng(seed)
    m = gf2_row_basis(np.asarray(h, dtype=np.uint8) % 2).astype(np.int64)
    r, n = m.shape
    vals = list(qvals) if qvals is not None else [0] * n  # tracked fault value per qubit (the prep is built backwards)
    faults = list(vals)
    red: list[tuple[int, int]] = []
    stack: list = []
    forbid: list[set] = [set()]
    used: set = set()
    backs = 0
    while int(m.any(axis=0).sum()) > r:
        if len(red) > 10 * n:
            return None
        w = m.sum(axis=0)
        gain = w[None, :] - (m[:, :, None] ^ m[:, None, :]).sum(axis=0)
        np.fill_diagonal(gain, -(10**9))
        gain[:, w == 0] = -(10**9)
        gain[w == 0, :] = -(10**9)
        cs, ts = np.nonzero(gain > 0)
        if len(cs):
            if eps > 0 and rng.random() < eps:
                order = list(rng.permutation(len(cs)))
            else:
                tie = rng.random(len(cs))
                order = sorted(
                    range(len(cs)),
                    key=lambda k: ((layered and (cs[k] in used or ts[k] in used)), -gain[cs[k], ts[k]], tie[k]),
                )
            cands = [(int(cs[k]), int(ts[k])) for k in order]
        else:  # Gaussian steps: any pair inside a row with >= 2 ones
            cands = []
            for i in rng.permutation(r):
                ones = np.nonzero(m[i])[0]
                if len(ones) >= 2:
                    cands += [(int(a), int(b)) for a in ones for b in ones if a != b]
            rng.shuffle(cands)
        chosen = None
        for c, t in cands:
            if (c, t) in forbid[-1]:
                continue
            # CX(c, t) placed before everything so far: an X on c spreads to t, a Z on t spreads to c
            k, other = (t, c) if track_z else (c, t)
            nf = vals[k] ^ vals[other]
            if bad is not None and (bad(nf, 1) or (pairs and any(bad(nf ^ g, 2) for g in faults))):
                forbid[-1].add((c, t))
                continue
            chosen = (c, t, k, nf)
            break
        if chosen is None:
            if not stack or backs >= max_back:
                return None
            backs += 1
            m, vals, faults, red, forbid, used = stack.pop()
            continue
        c, t, k, nf = chosen
        stack.append((
            m.copy(),
            list(vals),
            list(faults),
            list(red),
            [set(x) for x in forbid[:-1]] + [forbid[-1] | {(c, t)}],
            set(used),
        ))
        used = {c, t} if (c in used or t in used) else used | {c, t}
        m[:, t] ^= m[:, c]
        vals[k] = nf
        faults.append(nf)
        red.append((c, t))
        forbid.append(set())
    alive = [int(q) for q in np.nonzero(m.any(axis=0))[0]]
    ops = (
        [Op("R", (q,)) for q in range(n)]
        + [Op("H", (q,)) for q in alive]
        + [Op("CX", (c, t)) for c, t in reversed(red)]
    )
    return Circuit(ops, n, list(range(n)))


def greedy_candidates(h: np.ndarray, k: int, *, seed: int = 0, eps: float = 0.0) -> list[Circuit]:
    """Up to ``k`` distinct preps by greedy column elimination (``guided_prep`` without tracked faults).

    The first uses the given column order, the others random column permutations; fewest CNOTs first. ``eps``: as
    for ``guided_prep``.
    """
    rng = np.random.default_rng(seed)
    n = h.shape[1]
    out, seen = [], set()
    for i in range((4 if eps == 0 else 8) * k):
        perm = np.arange(n) if i == 0 else rng.permutation(n)
        c = guided_prep(h[:, perm], seed=int(rng.integers(1 << 30)), layered=False, eps=eps)
        if c is None:
            continue
        ops = [Op(op.name, tuple(int(perm[q]) for q in op.qubits)) for op in c.ops]
        circ = Circuit(ops, n, list(range(n)))
        key = str(circ.to_stim())
        if key in seen:
            continue
        seen.add(key)
        out.append(circ)
        if len(out) == k:
            break
    return sorted(out, key=lambda c: sum(op.name == "CX" for op in c.ops))


def cancel_reference(before: Circuit, spec: BlockSpec, state: CSSState, t: int) -> tuple[dict, dict]:
    """Largest data-error weight per block syndrome, for 1 and 2 undetected faults of ``before``.

    ``before``: everything before the block. The data errors are of the type the block detects.
    """
    tab = fault_table(before)
    nd = len(before.data)
    err = (tab.data_z if spec.kind == "X" else tab.data_x)[:, :nd]
    stab = gf2_row_basis(state.sx if spec.kind == "X" else state.sz)  # judges Z (X) errors by their X (Z) syndrome
    lw = leader_weights(stab, t + 1)
    cw, cm = _pack(stab.T), _pack(np.asarray(spec.checks, dtype=np.uint8).T % 2)
    rows = [np.nonzero(e)[0] for e in err]
    ws = np.array([int(np.bitwise_xor.reduce(cw[r])) if len(r) else 0 for r in rows], dtype=np.int64)  # ty: ignore[no-matching-overload]  (numpy stub gap)
    ms = np.array([int(np.bitwise_xor.reduce(cm[r])) if len(r) else 0 for r in rows], dtype=np.int64)  # ty: ignore[no-matching-overload]  (numpy stub gap)
    det = (tab.flips.astype(np.int64) @ before.detector_matrix().T.astype(np.int64)) % 2  # detector bits per fault
    flips = [bytes(f) if f.any() else b"" for f in det.astype(np.uint8)]
    one: dict = {}
    two: dict = {}
    sel = [i for i in range(len(rows)) if ms[i] or ws[i]]
    for i in sel:
        if not flips[i] and ms[i]:
            one[int(ms[i])] = max(one.get(int(ms[i]), 0), int(lw[ws[i]]))
    if t >= 2:
        groups = collections.defaultdict(list)
        for i in sel:
            groups[flips[i]].append(i)
        for members in groups.values():
            ks = np.array(members)
            w_, m_ = ws[ks], ms[ks]
            for i in range(len(ks) - 1):
                w2 = lw[w_[i] ^ w_[i + 1 :]]
                m2 = m_[i] ^ m_[i + 1 :]
                for mm, ww in zip(m2[m2 != 0].tolist(), w2[m2 != 0].tolist(), strict=True):
                    if ww > two.get(mm, 0):
                        two[mm] = ww
    return one, two


def guided_block_preps(
    node: Node, index: int, t: int, *, k: int = 4, seeds: int = 40, layered: bool = True
) -> list[Circuit]:
    """Up to ``k`` distinct cancellation-guided preps for the ancilla of end block ``index`` of ``node``."""
    spec = node.blocks[index][0]
    before = copy.deepcopy(node)
    for j in reversed(range(index, len(before.blocks))):  # the block and later ones go with their gadgets
        before.remove_block(j)
    one, two = cancel_reference(compile_node(before), spec, node.state, t)

    def bad(v: int, s: int) -> bool:
        if not v:
            return False
        # s ancilla faults + b data faults (s + b <= t) are dangerous if the data error has weight > s + b
        return any(w > s + b for b, w in ((1, one.get(v, 0)), (2, two.get(v, 0))) if s + b <= t)

    ht = spec.restricted() % 2
    cols = _pack(ht.T)
    qvals = [int(cols[a]) for a in range(spec.n_anc)]
    h = spec.ancilla_generator()
    out, seen = [], set()
    for sd in range(seeds):
        c = guided_prep(h, qvals, bad, track_z=spec.kind == "X", seed=sd, layered=layered)
        if c is None:
            continue
        key = str(c.to_stim())
        if key not in seen:
            seen.add(key)
            out.append(c)
        if len(out) >= k:
            break
    return out
