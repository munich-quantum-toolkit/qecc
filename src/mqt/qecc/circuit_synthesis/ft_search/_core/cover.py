# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Stabilizer group elements as candidate checks, and robust covers of undetected dangerous fault combinations.

Elements are coefficient masks over the basis rows (bit i = row i). Groups with at most ``EXACT_LIMIT`` nonzero
elements are enumerated; larger ones are sampled (seeded): light elements by information-set decoding plus random
ones. A sampled set can only make a cover costlier, never wrong, since every design is checked exactly.

Robust covers choose elements so that every given undetected dangerous fault combination stays detected even under
light outcome errors (ILP). The block solvers use them to choose which checks a transversal block measures.
"""

from __future__ import annotations

import itertools
import operator
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from . import memo
from .setcover import solve_cover
from .verify import _pack, leader_weights

if TYPE_CHECKING:
    from .verify import SparseLeader


EXACT_LIMIT = 1 << 16


def _weights(g: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Weight of each element mask in ``g``."""
    r = basis.shape[0]
    out = np.zeros(len(g), dtype=np.int64)
    chunk = 1 << 14
    for i in range(0, len(g), chunk):
        gg = g[i : i + chunk]
        out[i : i + chunk] = ((((gg[:, None] >> np.arange(r)) & 1).astype(np.uint8) @ basis) % 2).sum(axis=1)
    return out


def exact_elements(basis: np.ndarray, max_elems: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """All nonzero elements (masks, weights); with ``max_elems`` only the lightest that many."""
    r = basis.shape[0]
    g = np.arange(1, 1 << r, dtype=np.int64)
    w = _weights(g, basis)
    if max_elems is not None and len(g) > max_elems:
        keep = np.argsort(w, kind="stable")[:max_elems]
        keep.sort()
        g, w = g[keep], w[keep]
    return g, w


def isd_elements(basis: np.ndarray, n: int, *, seed: int = 0) -> dict[int, int]:
    """Light elements by information-set decoding: mask -> weight, about ``n`` of the lightest found.

    Each round permutes the columns at random and eliminates: every row of the systematic form and every sum of two
    rows is a light element.
    """
    rng = np.random.default_rng(seed)
    r, m = basis.shape
    found: dict[int, int] = {}
    pw = np.left_shift(np.int64(1), np.arange(r, dtype=np.int64))
    for _ in range(400):
        perm = rng.permutation(m)
        a = np.concatenate([basis[:, perm] % 2, np.eye(r, dtype=np.uint8)], axis=1).astype(np.uint8)
        row = 0
        for col in range(m):
            piv = np.nonzero(a[row:, col])[0]
            if not len(piv):
                continue
            p = row + piv[0]
            if p != row:
                a[[row, p]] = a[[p, row]]
            others = np.nonzero(a[:, col])[0]
            others = others[others != row]
            a[others] ^= a[row]
            row += 1
            if row == r:
                break
        vec, coef = a[:, :m], a[:, m:]
        masks = coef.astype(np.int64) @ pw
        wts = vec.sum(axis=1).astype(np.int64)
        for i in range(r):
            found.setdefault(int(masks[i]), int(wts[i]))
        # sums of two systematic rows (Stern-like): still light, many more elements
        i, j = np.triu_indices(r, k=1)
        wij = (vec[i] ^ vec[j]).sum(axis=1)
        for mk, wv in zip((masks[i] ^ masks[j]).tolist(), wij.tolist(), strict=True):
            found.setdefault(mk, wv)
        if len(found) >= 4 * n:
            break
    found.pop(0, None)
    keep = sorted(found.items(), key=operator.itemgetter(1, 0))[:n]
    return dict(keep)


def group_elements(basis: np.ndarray, max_elems: int = 8192, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Candidate group elements (masks, weights), sorted by mask.

    The lightest ``max_elems`` for small groups, else 80 % of ``max_elems`` ISD-light plus 20 % random elements
    (seeded).
    """
    basis = np.asarray(basis, dtype=np.uint8) % 2
    r = basis.shape[0]
    if r == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if (1 << r) - 1 <= EXACT_LIMIT:
        return exact_elements(basis, max_elems)
    key = memo.digest(basis.tobytes(), basis.shape, max_elems, seed)
    hit = memo.get("elements", key)
    if not isinstance(hit, tuple):
        n_rand = int(0.2 * max_elems)
        light = isd_elements(basis, max_elems - n_rand, seed=seed)
        rng = np.random.default_rng(seed + 7919)
        rand = rng.integers(1, 1 << r, size=2 * n_rand, dtype=np.int64)
        g = np.unique(np.concatenate([np.fromiter(light.keys(), dtype=np.int64, count=len(light)), rand]))
        g = g[g != 0][: max_elems + n_rand]
        hit = (g, _weights(g, basis))
        memo.put("elements", key, hit, limit=256)
    return hit


def anticommuting_element(target: int, r: int, rng: np.random.Generator) -> int:
    """A random element mask with odd overlap with syndrome ``target`` (it detects that syndrome); 0 if target is 0."""
    if target == 0:
        return 0
    m = int(rng.integers(1, 1 << r))
    if (m & target).bit_count() % 2 == 0:
        m ^= target & -target  # flip the lowest set bit of the target: parity toggles
    return m


def parity_matrix(targets: np.ndarray, elems: np.ndarray) -> np.ndarray:
    """Whether each element mask anticommutes with an error of each syndrome target (targets x elements, int8)."""
    tg = np.asarray(targets, dtype=np.int64)
    el = np.asarray(elems, dtype=np.int64)
    out = np.empty((len(tg), len(el)), dtype=np.int8)
    step = max(1, (1 << 22) // max(len(el), 1))
    for i in range(0, len(tg), step):
        x = np.bitwise_and(tg[i : i + step, None], el[None, :])
        if hasattr(np, "bitwise_count"):
            out[i : i + step] = np.bitwise_count(x) & 1
            continue
        x ^= x >> 32
        x ^= x >> 16
        x ^= x >> 8
        x ^= x >> 4
        x ^= x >> 2
        x ^= x >> 1
        out[i : i + step] = x & 1
    return out


def robust_cover_from_failures(
    basis: np.ndarray,
    fails: list[tuple[int, int]],
    t_max: int,
    *,
    time_limit: float = 600.0,
    cost_jitter: float = 0.0,
    seed: int = 0,
    lazy_targets: int = 20000,
    lw: np.ndarray | SparseLeader | None = None,
) -> list[int]:
    """Robust cover of undetected dangerous combinations (syndrome w.r.t. ``basis`` rows, number of faults s).

    Returns element masks such that every syndrome stays detected after adding any outcome error of weight
    k <= t_max - s under which the combination stays dangerous. Each element costs its weight + 1. ``cost_jitter`` > 0:
    costs multiplied by exp(cost_jitter * N(0, 1)) (seeded).
    More than ``lazy_targets`` targets are covered lazily (a subset first, then the ones the cover misses). ``lw``: the
    leader-weight table of the target (``verify.state_leader``; default ``leader_weights(basis)``).
    """
    r, n = basis.shape
    if lw is None:
        lw = leader_weights(basis, t_max)
    cols = _pack(basis.T).astype(np.int64)
    light = {0: np.array([0], dtype=np.int64)}
    for k in range(1, t_max):
        light[k] = np.unique(
            np.array(
                [np.bitwise_xor.reduce(cols[list(c)]) for c in itertools.combinations(range(n), k)],  # ty: ignore[no-matching-overload]  (numpy stub gap)
                dtype=np.int64,
            )
        )
    if not fails:
        return []
    sigs = np.array([f for f, _ in fails], dtype=np.int64)
    ss = np.array([sv for _, sv in fails], dtype=np.int64)
    parts = []
    for k, lk in light.items():
        sel = (ss + k <= t_max) & (np.asarray(lw[sigs]) > ss + k)
        if sel.any():
            parts.append((sigs[sel, None] ^ lk[None, :]).ravel())
    tg = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)
    tg = tg[tg != 0]
    if not len(tg):
        return []
    rng = np.random.default_rng(seed + 31337)

    def solve(tg: np.ndarray) -> list[int]:
        g, w = group_elements(basis, seed=seed)
        a = parity_matrix(tg, g)
        if not a.any(axis=1).all() and len(g) < (1 << r) - 1:  # the lightest elements cannot cover every target
            g, w = group_elements(basis, max_elems=1 << 16, seed=seed)
            a = parity_matrix(tg, g)
        if not a.any(axis=1).all():  # sampled set (large group): add an element detecting each uncovered target
            extra = np.unique([anticommuting_element(int(x), r, rng) for x in tg[~a.any(axis=1)]])
            extra = extra[~np.isin(extra, g)]
            g = np.concatenate([g, extra])
            w = np.concatenate([w, _weights(extra, basis)])
            a = parity_matrix(tg, g)
        useful = a.any(axis=0)
        g, w, a = g[useful], w[useful], a[:, useful]
        cost = (w + 1).astype(float)
        if cost_jitter:
            cost *= np.exp(cost_jitter * np.random.default_rng(seed).standard_normal(len(cost)))
        sol = solve_cover(csr_matrix(a), cost, time_limit=time_limit, seed=seed)
        if sol is None:
            msg = "cover failed (infeasible or no solution within the time limit)"
            raise RuntimeError(msg)
        return [int(x) for x in g[sol]]

    if len(tg) <= lazy_targets:
        return solve(tg)
    # lazily: cover a subset (the failures themselves first), add the ones the cover misses, repeat
    base_t = np.intersect1d(np.unique(sigs), tg)
    active = base_t if len(base_t) <= lazy_targets else rng.choice(base_t, lazy_targets, replace=False)
    rest = np.setdiff1d(tg, active)
    if len(active) < lazy_targets and len(rest):
        active = np.union1d(active, rng.choice(rest, min(lazy_targets - len(active), len(rest)), replace=False))
    for _ in range(20):
        chosen = solve(active)
        miss = tg[~parity_matrix(tg, np.array(chosen, dtype=np.int64)).any(axis=1)] if chosen else tg
        if not len(miss):
            return chosen
        add = miss if len(miss) <= lazy_targets // 2 else rng.choice(miss, lazy_targets // 2, replace=False)
        active = np.union1d(active, add)
    msg = "lazy cover did not converge"
    raise RuntimeError(msg)
