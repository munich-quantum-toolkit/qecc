# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for scope-level behaviour.

Scope-level behaviour on the [[17,1,5]] color code with an empty memo: late checks protect unprotected ancillas
jointly, slack-aware part costs, and gadgets inside a base that has its own ancillas and detectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.ft_search._core.circuit import Circuit, Op
from mqt.qecc.circuit_synthesis.ft_search._core.config import Context
from mqt.qecc.circuit_synthesis.ft_search._core.design import Design, make_node
from mqt.qecc.circuit_synthesis.ft_search._core.evaluation import evaluate_cached, proposals
from mqt.qecc.circuit_synthesis.ft_search._core.gauge import Gadget, build, promote, pulled_back_groups
from mqt.qecc.circuit_synthesis.ft_search._core.solvers import (
    BlockSolver,
    CrigerSolver,
    part_cost,
)
from mqt.qecc.circuit_synthesis.ft_search._core.verify import check_ft, check_noiseless

from .helpers import color_17_1_5

CTX = Context(lam=2, lams=(0.0, 2.0, 10.0), synth=[("greedy", 1), ("qecc_depth", 1)], cover_time=30, pool=1)


@pytest.fixture
def start():
    """A design of the bare [[17,1,5]] color prep and its state."""
    prep, state = color_17_1_5("gate")
    return Design(make_node("D", prep, state), 2), state


@pytest.mark.slow
def test_late_greedy_protects_unprotected_ancillas_jointly(start) -> None:
    """Late checks protect an unprotected block ancilla together with the data."""
    d, _state = start
    bd = proposals(BlockSolver("Z", "weight"), d, (), CTX)[0][1]
    assert (0,) in evaluate_cached(bd).open_scopes  # the block's ancilla is not FT on its own
    outs = proposals(CrigerSolver("greedy", 80, late=True), bd, (), CTX)
    assert outs
    x = outs[0][1]
    assert evaluate_cached(x).ok
    assert all(g.late for g in x.root.gadgets[len(bd.root.gadgets) :])


def test_part_cost_counts_depth_beyond_slack() -> None:
    """The part cost counts only the depth beyond the slack."""
    m = {"cnot": 10, "meas": 2, "depth": 7, "qubits": 9}
    assert part_cost(CTX, m, 10.0, None) == CTX.cost(m, 10.0)
    assert part_cost(CTX, m, 10.0, 7) == 12
    assert part_cost(CTX, m, 10.0, 5) == 12 + 20


def _checked_prep(state) -> Circuit:
    """The bare d = 5 prep with one Z stabilizer measured twice on two own ancillas, and declared detectors."""
    prep, _ = color_17_1_5("gate")
    row = min(state.sz, key=lambda r: int(r.sum()) if 0 < r.sum() < 17 else 99)
    ops, n = list(prep.ops), prep.num_qubits
    for a in (n, n + 1):
        ops += [Op("R", (a,)), *[Op("CX", (prep.data[q], a)) for q in np.flatnonzero(row)], Op("M", (a,))]
    c = Circuit(ops, n + 2, list(prep.data))
    c.detectors = [[0, 1], [1]]  # declared, with a parity of two records (renumbered when gadgets come between them)
    check_noiseless(c, state)
    return c


def test_gadgets_inside_a_base_with_detectors(start) -> None:
    """Gadgets inside a base with its own ancillas and detectors compile correctly."""
    _, state = start
    base = _checked_prep(state)
    inside = [p for p in range(1, len(base.ops)) if base.ops[p - 1].name == "CX"]
    groups = pulled_back_groups(base, state, inside)
    n_checked = 0
    for p in inside[:: max(1, len(inside) // 12)]:
        for kind in ("Z", "X"):
            rows = groups[p][kind]
            if len(rows) == 0:
                continue
            g = Gadget(kind, p, [int(i) for i in np.flatnonzero(rows[np.argmin(rows.sum(axis=1))])])
            c, _ = build(base, [g])
            check_noiseless(c, state)  # the base's renumbered detectors and the new check all deterministic 0
            n_checked += 1
    assert n_checked > 4


def test_promote_inside_a_base_with_detectors_is_ft(start) -> None:
    """The promotion inside a base with its own ancillas and detectors ends FT."""
    _, state = start
    base = _checked_prep(state)
    inside = list(range(len(base.ops) // 2, len(base.ops)))  # no end position: every check inside the base
    circ, gadgets, plog = promote(
        base, state, 2, positions=inside, layer=False, incremental=True, time_limit=60, log=lambda *_a, **_k: None
    )
    assert plog.ok
    assert gadgets
    assert all(p < len(base.ops) for g in gadgets for p in g.coupling_slots())
    check_noiseless(circ, state)
    assert check_ft(circ, state, 2).ok
