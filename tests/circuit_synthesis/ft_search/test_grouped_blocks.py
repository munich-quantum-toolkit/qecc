# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for grouped blocks.

Grouped blocks (``BlockSpec.groups``, Huang & Brown's merge (ii)) on the [[17,1,5]] color code: they compile to
noiseless-correct circuits, transversal specs are unchanged, and the weighted ancilla target judges ancilla errors by
their weight on the data (brute force, and against the global check).
"""

from __future__ import annotations

import itertools
import pickle  # ruff: ignore[suspicious-pickle-import]  (round trips of objects made here)

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.ft_search._core.blocks import BlockSpec
from mqt.qecc.circuit_synthesis.ft_search._core.design import Design, ancilla_states, local_check, make_node
from mqt.qecc.circuit_synthesis.ft_search._core.faults import fault_table
from mqt.qecc.circuit_synthesis.ft_search._core.verify import (
    CSSState,
    WeightedTarget,
    check_ft,
    check_noiseless,
    gf2_row_basis,
    state_leader,
)
from mqt.qecc.circuit_synthesis.ft_search.synth import candidates

from .helpers import color_17_1_5


@pytest.fixture(scope="module")
def code():
    """The [[17,1,5]] color code state and its prep."""
    prep, state = color_17_1_5("gate")
    return state, prep


def _pair(state: CSSState, kind: str) -> np.ndarray:
    """Two overlapping weight-4 stabilizers of that type."""
    rows = gf2_row_basis(state.sz if kind == "Z" else state.sx)
    els = {}
    for m in range(1, 1 << rows.shape[0]):
        v = np.bitwise_xor.reduce(rows[[i for i in range(rows.shape[0]) if m >> i & 1]], axis=0)  # ty: ignore[no-matching-overload]  (numpy stub gap)
        if v.sum() == 4:
            els[v.tobytes()] = v
    vs = list(els.values())
    for a, b in itertools.combinations(vs, 2):
        if (a & b).sum() == 2:
            return np.array([a, b], dtype=np.uint8)
    raise AssertionError


def _grouped(checks: np.ndarray, kind: str) -> BlockSpec:
    """A grouped block with one ancilla qubit per pair (only A, both, only B)."""
    a, b = checks
    only_a = [int(q) for q in np.nonzero(a & (1 - b))[0]]
    both = [int(q) for q in np.nonzero(a & b)[0]]
    only_b = [int(q) for q in np.nonzero(b & (1 - a))[0]]
    ht = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    return BlockSpec(kind, checks, groups=(tuple(only_a), tuple(both), tuple(only_b)), htilde=ht)


def _design(state, prep, specs) -> Design:
    root = make_node("D", prep, state)
    for i, sp in enumerate(specs):
        full, target = ancilla_states(sp)
        c = candidates(sp.ancilla_generator(), "greedy", 1)[0]
        root.add_block(sp, make_node(f"D.{sp.kind}{i}", c, full, target))
    return Design(root, 2)


@pytest.mark.parametrize("kind", ["Z", "X"])
def test_grouped_block_compiles_correctly(code, kind) -> None:
    """A grouped block compiles to a noiseless-correct circuit and pickles."""
    state, prep = code
    checks = _pair(state, kind)
    sp = _grouped(checks, kind)
    assert sp.grouped
    assert sp.n_anc == 3
    d = _design(state, prep, [sp])
    check_noiseless(d.circuit(), state)
    pickle.loads(pickle.dumps(d))  # ruff: ignore[suspicious-pickle-usage]


def test_transversal_round_trip_and_singleton_groups(code) -> None:
    """A transversal spec survives a pickle round trip, and singleton groups give the transversal form."""
    state, _ = code
    checks = _pair(state, "Z")
    old = BlockSpec("Z", checks)
    back = pickle.loads(pickle.dumps(old))  # ruff: ignore[suspicious-pickle-usage]
    assert (back.kind, back.pos, back.gamma, back.groups) == (old.kind, old.pos, old.gamma, old.groups)
    assert np.array_equal(back.checks, old.checks)
    single = BlockSpec("Z", checks, groups=tuple((q,) for q in old.support), htilde=old.restricted())
    assert single.gamma == tuple(old.support)
    assert single.groups is None


@pytest.mark.parametrize("kind", ["Z", "X"])
def test_weighted_target_brute_force(code, kind) -> None:
    """The weighted ancilla target equals a brute-force weight on the data."""
    state, _ = code
    sp = _grouped(_pair(state, kind), kind)
    _, target = ancilla_states(sp)
    assert isinstance(target, WeightedTarget)
    t = 2
    tab = state_leader(target, kind, t)
    rows = gf2_row_basis(target.sx if kind == "Z" else target.sz)
    hd_span = [np.zeros(target.gamma.shape[0], dtype=np.uint8)]
    for h in target.hd:
        hd_span += [(x + h) % 2 for x in hd_span]
    for m in range(1 << sp.n_anc):
        e = np.array([m >> a & 1 for a in range(sp.n_anc)], dtype=np.uint8)
        w = min(int(((target.gamma @ e + h) % 2).sum()) for h in hd_span)
        key = sum(int((rows[i] @ e) % 2) << i for i in range(rows.shape[0]))
        assert int(tab[key]) == min(w, t + 1)


def test_single_ancilla_fault_is_dangerous(code) -> None:
    """A grouped ancilla is not FT on its own (a single Z / X spreads to a pair), the local and the global check agree."""
    state, prep = code
    for kind in "ZX":
        d = _design(state, prep, [_grouped(_pair(state, kind), kind)])
        anc = d.root.blocks[0][1]
        assert not local_check(anc, 2).ok
        circ = d.circuit()
        rep = check_ft(circ, state, 2, collect=True)
        table = fault_table(circ)
        assert circ.tags is not None
        # the bare prep fails on its own: require a failing single fault in the ancilla circuit (provenance tag "a")
        assert any(len(r) == 1 and circ.tags[int(table.op_index[r[0]])][0] == "a" for r in rep.failing_reps)


def test_weighted_target_needs_its_matrices() -> None:
    """A weighted target cannot be made without the data map ``gamma`` and the measured checks ``hd``."""
    empty = np.zeros((0, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        WeightedTarget(empty, empty)  # ty: ignore[missing-argument]  (on purpose)
