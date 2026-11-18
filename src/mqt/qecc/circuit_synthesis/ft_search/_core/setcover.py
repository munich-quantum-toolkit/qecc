# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Weighted set cover for every cover problem of the search (block covers, promotion rounds).

Rows are failing combinations, columns candidate checks; a column covers the rows it detects. ``conflicts``: column
sets that must not all be chosen (checks that do not commute in time).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, csr_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence


def ilp_cover(
    a: np.ndarray | csr_matrix,
    cost: Sequence[float] | np.ndarray,
    *,
    conflicts: Sequence[Sequence[int]] = (),
    time_limit: float = 300.0,
) -> list[int] | None:
    """Exact cover by the ILP (scipy ``milp`` / HiGHS); None if infeasible or no solution in time."""
    a = csr_matrix(a)
    cons = [LinearConstraint(a, lb=1, ub=np.inf)]
    if conflicts:
        cm = np.zeros((len(conflicts), a.shape[1]))
        for r, c in enumerate(conflicts):
            cm[r, list(c)] = 1
        cons.append(
            LinearConstraint(csr_matrix(cm), lb=-np.inf, ub=np.array([len(c) - 1 for c in conflicts], dtype=float))
        )
    res = milp(
        c=np.asarray(cost, dtype=float),
        integrality=np.ones(a.shape[1]),
        bounds=Bounds(0, 1),
        constraints=cons,
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    if res.x is None:
        return None
    return [int(i) for i in np.nonzero(res.x > 0.5)[0]]


def ls_cover(
    a: np.ndarray | csr_matrix,
    cost: Sequence[float] | np.ndarray,
    *,
    conflicts: Sequence[Sequence[int]] = (),
    time_limit: float = 10.0,
    seed: int = 0,
    patience: int = 2000,
) -> list[int] | None:
    """Greedy cover plus local search (remove a few columns, re-cover greedily with noise, drop redundant ones).

    None if some row cannot be covered. Stops after ``patience`` iterations without improvement (or at the time limit).
    """
    rng = np.random.default_rng(seed)
    sp = csr_matrix(a).astype(bool)
    n_rows, n_cols = sp.shape
    cost = np.asarray(cost, dtype=float)
    if n_rows == 0:
        return []
    if not np.asarray(sp.sum(axis=1)).ravel().all():
        return None
    dense = sp.nnz > 0.05 * n_rows * n_cols  # parity coverage matrices are ~50 % dense: numpy is much faster
    a_r = sp.toarray() if dense else sp
    if dense:
        col_rows = [np.nonzero(a_r[:, j])[0] for j in range(n_cols)]
    else:
        a_c = csc_matrix(sp)
        col_rows = [a_c.indices[a_c.indptr[j] : a_c.indptr[j + 1]] for j in range(n_cols)]
    conf_of: dict[int, list[set]] = {}
    for c in conflicts:
        s = {int(x) for x in c}
        for j in s:
            conf_of.setdefault(j, []).append(s)

    def allowed(j: int, sel: set) -> bool:
        return all(not (s - {j}) <= sel for s in conf_of.get(j, ()))

    def cover_greedy(sel: set, count: np.ndarray, noisy: bool) -> bool:
        unc = np.nonzero(count == 0)[0]
        while len(unc):
            gain = np.asarray(a_r[unc].sum(axis=0)).ravel().astype(float)  # rows newly covered per column
            score = gain / cost
            if noisy:
                score *= np.exp(0.3 * rng.standard_normal(n_cols))
            score[gain == 0] = -1
            for j in np.argsort(-score):
                if score[j] <= 0:
                    return False
                if j not in sel and allowed(int(j), sel):
                    break
            else:
                return False
            j = int(j)
            sel.add(j)
            count[col_rows[j]] += 1
            unc = unc[count[unc] == 0]
        return True

    def prune(sel: set, count: np.ndarray) -> None:
        for j in sorted(sel, key=lambda j: -cost[j]):
            if (count[col_rows[j]] > 1).all():
                sel.discard(j)
                count[col_rows[j]] -= 1

    sel: set = set()
    count = np.zeros(n_rows, dtype=np.int64)
    if not cover_greedy(sel, count, noisy=False):
        # a conflict blocked a column some row needs: again, with the only column of each such row chosen first
        sel, count = set(), np.zeros(n_rows, dtype=np.int64)
        for j in np.unique(sp[np.asarray(sp.sum(axis=1)).ravel() == 1].indices).tolist():
            if not allowed(j, sel):
                return None
            sel.add(j)
            count[col_rows[j]] += 1
        if not cover_greedy(sel, count, noisy=False):
            return None
    prune(sel, count)
    best, best_cost = set(sel), cost[list(sel)].sum()
    t0 = time.time()
    last_gain = 0
    for it in range(100000):
        if time.time() - t0 > time_limit or it - last_gain > patience:
            break
        cur, cnt = set(best), np.zeros(n_rows, dtype=np.int64)
        for j in cur:
            cnt[col_rows[j]] += 1
        # remove 1-3 chosen columns (expensive ones more likely), re-cover greedily with noise, drop redundant ones
        k = int(rng.integers(1, min(3, len(cur)) + 1)) if cur else 0
        if k:
            lst = np.array(sorted(cur))
            p = cost[lst] / cost[lst].sum()
            for j in rng.choice(lst, size=k, replace=False, p=p):
                cur.discard(int(j))
                cnt[col_rows[int(j)]] -= 1
        if not cover_greedy(cur, cnt, noisy=True):
            continue
        prune(cur, cnt)
        c = cost[list(cur)].sum()
        if c < best_cost - 1e-9:
            best, best_cost, last_gain = cur, c, it
    return sorted(best)


def solve_cover(
    a: np.ndarray | csr_matrix,
    cost: Sequence[float] | np.ndarray,
    *,
    conflicts: Sequence[Sequence[int]] = (),
    time_limit: float = 300.0,
    seed: int = 0,
) -> list[int] | None:
    """Columns of a cheap cover, or None if there is none in time.

    The exact ILP (at most 30 s) up to 10^6 matrix entries, else local search (at most 5 s), which does better than a
    time-limited ILP on large problems.
    """
    if a.shape[0] * a.shape[1] <= 10**6:
        return ilp_cover(a, cost, conflicts=conflicts, time_limit=min(time_limit, 30.0))
    return ls_cover(a, cost, conflicts=conflicts, time_limit=min(time_limit, 5.0), seed=seed)
