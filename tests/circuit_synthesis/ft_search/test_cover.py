# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for candidate stabilizer elements, leader tables and robust covers.

Candidate stabilizer elements: exact for small groups (unchanged), ISD + random samples for large ones (elements of
the group, light, seed-dependent); the sparse leader table equals the dense one; the sampled pulled-back groups lie in
the pulled-back group; a robust cover on a large group still covers every target.
"""

from __future__ import annotations

import itertools

import numpy as np

from mqt.qecc.circuit_synthesis.ft_search._core import cover
from mqt.qecc.circuit_synthesis.ft_search._core.circuit import Circuit, Op
from mqt.qecc.circuit_synthesis.ft_search._core.cover import parity_matrix, robust_cover_from_failures
from mqt.qecc.circuit_synthesis.ft_search._core.gauge import pulled_back_groups
from mqt.qecc.circuit_synthesis.ft_search._core.verify import (
    CSSState,
    SparseLeader,
    _pack,
    gf2_row_basis,
    leader_weights,
)


def _surface_hx(d: int) -> np.ndarray:
    """X-type checks of the rotated surface code (weight-4 plaquettes + weight-2 boundaries)."""
    rows = []
    q = lambda r, c: r * d + c
    for r in range(-1, d):
        for c in range(-1, d):
            if (r + c) % 2 != 0:
                continue
            supp = [q(rr, cc) for rr in (r, r + 1) for cc in (c, c + 1) if 0 <= rr < d and 0 <= cc < d]
            if len(supp) == 4 or (len(supp) == 2 and (r in {-1, d - 1})):
                v = np.zeros(d * d, dtype=np.uint8)
                v[supp] = 1
                rows.append(v)
    return gf2_row_basis(np.array(rows))


def _in_rowspace(basis: np.ndarray, vecs: np.ndarray) -> bool:
    r = gf2_row_basis(basis).shape[0]
    return all(gf2_row_basis(np.vstack([basis, v[None]])).shape[0] == r for v in vecs)


def test_exact_path_unchanged() -> None:
    """Small groups use the exact elements."""
    basis = _surface_hx(5)
    g, w = cover.group_elements(basis, 100)
    _allg, allw = cover.exact_elements(basis)
    assert (1 << basis.shape[0]) - 1 <= cover.EXACT_LIMIT
    assert len(g) == 100
    assert sorted(w.tolist()) == sorted(allw.tolist())[:100]


def test_isd_sample_on_surface_d7() -> None:
    """Large groups get a light ISD sample of group elements."""
    basis = _surface_hx(7)
    r = basis.shape[0]
    assert r == 24
    assert (1 << r) - 1 > cover.EXACT_LIMIT
    g, w = cover.group_elements(basis, 2000, seed=1)
    assert len(g) >= 1500
    assert len(set(g.tolist())) == len(g)
    vecs = (((g[:200, None] >> np.arange(r)) & 1).astype(np.uint8) @ basis) % 2
    assert _in_rowspace(basis, vecs)
    assert (vecs.sum(axis=1) == w[:200]).all()
    assert (w == 2).sum() >= 6
    assert (w == 4).sum() >= 18
    g2, _ = cover.group_elements(basis, 2000, seed=2)
    assert set(g.tolist()) != set(g2.tolist())  # another seed, another sample


def test_sparse_leader_matches_dense() -> None:
    """The sparse leader table equals the dense one."""
    basis = _surface_hx(5)
    dense = leader_weights(basis, 3)
    assert isinstance(dense, np.ndarray)
    sparse = SparseLeader(basis, 3)
    idx = np.arange(len(dense))
    assert (sparse[idx] == dense[idx]).all()
    assert sparse[5] == dense[5]
    assert isinstance(leader_weights(_surface_hx(7), 2), SparseLeader)


def test_robust_cover_on_large_group() -> None:
    """A robust cover on a large group covers every target."""
    basis = _surface_hx(7)
    rng = np.random.default_rng(0)
    fails = [(int(x), 1) for x in rng.integers(1, 1 << basis.shape[0], size=30)]
    elems = robust_cover_from_failures(basis, fails, 2, time_limit=60)
    a = parity_matrix(np.array([f for f, _ in fails]), np.array(elems))
    assert a.any(axis=1).all()


def test_pulled_back_sample_in_group() -> None:
    """The sampled pulled-back elements lie in the group."""
    basis = _surface_hx(7)
    n = basis.shape[1]
    state = CSSState(basis, np.zeros((0, n), dtype=np.uint8))
    ops = [Op("R", (q,)) for q in range(n)] + [Op("CX", (q, q + 1)) for q in range(0, n - 1, 3)]
    base = Circuit(ops, n, list(range(n)), None)
    g = pulled_back_groups(base, state, [0, len(ops)], max_elems=500)
    assert len(g[len(ops)]["X"]) >= 300
    assert _in_rowspace(basis, g[len(ops)]["X"][:50])


def test_lazy_cover_equals_full_cover_coverage() -> None:
    """With few allowed targets per round, the lazy cover still detects every target (here: many t = 3 targets)."""
    basis = _surface_hx(5)
    rng = np.random.default_rng(1)
    fails = [
        (int(x), int(s))
        for x, s in zip(rng.integers(1, 1 << basis.shape[0], size=60), rng.integers(1, 3, size=60), strict=True)
    ]
    full = robust_cover_from_failures(basis, fails, 3, time_limit=60, lazy_targets=10**9)
    lazy = robust_cover_from_failures(basis, fails, 3, time_limit=60, lazy_targets=300)

    cols = _pack(basis.T).astype(np.int64)
    lw = leader_weights(basis, 3)
    tg = set()
    for f, sv in fails:  # targets exactly as in robust_cover_from_failures: still dangerous after the light error
        for k in range(3 - sv + 1):
            for c in itertools.combinations(range(basis.shape[1]), k):
                x = f ^ (int(np.bitwise_xor.reduce(cols[list(c)])) if c else 0)  # ty: ignore[no-matching-overload]  (numpy stub gap)
                if x and lw[f] > sv + k:
                    tg.add(x)
    tg = np.array(sorted(tg), dtype=np.int64)
    assert parity_matrix(tg, np.array(lazy)).any(axis=1).all()
    w = lambda es: sum(int(((((e >> np.arange(basis.shape[0])) & 1) @ basis) % 2).sum()) + 1 for e in es)
    assert w(lazy) <= 1.5 * w(full)
