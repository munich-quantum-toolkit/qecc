# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Flag takeover (``Gadget.drive``, ``TakeoverSolver``): build semantics, pickles, and the solver on recorded designs."""

from __future__ import annotations

import pickle  # ruff: ignore[suspicious-pickle-import]  (round trips of objects made here)

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.ft_search._core.circuit import metrics
from mqt.qecc.circuit_synthesis.ft_search._core.gauge import Gadget, build
from mqt.qecc.circuit_synthesis.ft_search._core.verify import check_noiseless, gf2_row_basis

from .helpers import color_17_1_5


def _color():
    return color_17_1_5("gate")


@pytest.mark.parametrize("kind", ["X", "Z"])
def test_drive_semantics(kind) -> None:
    """A driven coupling keeps the state and moves the coupling to the flag."""
    prep, state = _color()
    rows = gf2_row_basis(state.sx if kind == "X" else state.sz)
    row = max(rows, key=lambda r: int(r.sum()))  # a weight-8 check
    sup = [int(q) for q in np.nonzero(row)[0]]
    end = len(prep.ops)
    plain, _ = build(prep, [Gadget(kind, end, sup, [(1, 5, 0)])])
    drv, _ = build(prep, [Gadget(kind, end, sup, [(1, 5, 0)], drive=((2, 0), (4, 0)))])
    check_noiseless(plain, state)
    check_noiseless(drv, state)  # same state, all checks deterministic
    a, f = prep.num_qubits, prep.num_qubits + 1
    cx = [op.qubits for op in drv.ops if op.name == "CX" and set(op.qubits) & {a, f} and set(op.qubits) - {a, f}]
    by_flag = [qs for qs in cx if f in qs]
    assert len(by_flag) == 2  # couplings 2 and 4 driven by the flag
    for qs in by_flag:
        assert qs == ((f, qs[1]) if kind == "X" else (qs[0], f))
    assert metrics(drv)["cnot"] == metrics(plain)["cnot"]
    assert metrics(drv)["meas"] == metrics(plain)["meas"]
    # invalid entries are ignored: outside the flag's span, unknown flag, duplicate coupling
    ign, _ = build(prep, [Gadget(kind, end, sup, [(1, 5, 0)], drive=((0, 0), (5, 0), (3, 7)))])
    assert str(ign.to_stim()) == str(plain.to_stim())


def test_drive_ignored_on_shared_flags() -> None:
    """The drive of a shared flag is ignored."""
    prep, state = _color()
    rows = gf2_row_basis(state.sx)
    end = len(prep.ops)
    g1 = [int(q) for q in np.nonzero(rows[0])[0]]
    g2 = [int(q) for q in np.nonzero(rows[1])[0]]
    shared = [Gadget("X", end, g1, [(1, 3, 0)], drive=((1, 0),)), Gadget("X", end, g2, [(1, 3, 0)])]
    plain = [Gadget("X", end, g1, [(1, 3, 0)]), Gadget("X", end, g2, [(1, 3, 0)])]
    assert str(build(prep, shared)[0].to_stim()) == str(build(prep, plain)[0].to_stim())


def test_pickle_round_trip() -> None:
    """A gadget with or without drive survives a pickle round trip (jobs travel to worker processes)."""
    g = Gadget("Z", 3, [1, 2, 3], [(0, 2, 0)])
    h = pickle.loads(pickle.dumps(g))  # ruff: ignore[suspicious-pickle-usage]
    assert h == g
    assert h.drive == ()
    gd = Gadget("Z", 3, [1, 2, 3], [(0, 2, 0)], drive=((1, 0),), block=1, anc=2)
    assert pickle.loads(pickle.dumps(gd)) == gd  # ruff: ignore[suspicious-pickle-usage]
