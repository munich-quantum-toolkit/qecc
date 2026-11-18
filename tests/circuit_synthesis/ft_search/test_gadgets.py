# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for gadgets.

One gadget type: the ancilla qubits of a block are gadgets of their scope (``Node.add_block``), so flags work on a
grouped block's couplings like on a bare check; removing a block keeps the other blocks' gadgets right;
designs carry a structure class.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from mqt.qecc.circuit_synthesis.ft_search._core.blocks import BlockSpec
from mqt.qecc.circuit_synthesis.ft_search._core.design import Design, ancilla_states, make_node
from mqt.qecc.circuit_synthesis.ft_search._core.faults import fault_table
from mqt.qecc.circuit_synthesis.ft_search._core.gauge import Gadget
from mqt.qecc.circuit_synthesis.ft_search._core.verify import (
    check_noiseless,
    state_leader,
)
from mqt.qecc.circuit_synthesis.ft_search.synth import candidates

from .helpers import table_keys
from .test_grouped_blocks import _grouped, _pair, code  # ruff: ignore[unused-import]  (fixture)


def _with(state, prep, specs) -> Design:
    root = make_node("D", prep, state)
    for i, sp in enumerate(specs):
        full, target = ancilla_states(sp)
        c = candidates(sp.ancilla_generator(), "greedy", 1)[0]
        root.add_block(sp, make_node(f"D.{sp.kind}{i}", c, full, target))
    return Design(root, 2)


def _bad_coupling_faults(circ, state) -> int:
    """Single faults on the couplings of block gadgets that go undetected and leave a data error of weight > 1."""
    tab = fault_table(circ)
    kx, kz, kc, _, _ = table_keys(tab, state, circ.detector_matrix())
    wx, wz = state_leader(state, "X", 2), state_leader(state, "Z", 2)
    gadgets_c = {i for i, tg in enumerate(circ.tags) if tg[0] == "c"}
    n = 0
    for f in range(len(tab)):
        if tab.op_index[f] in gadgets_c and kc[f] == 0 and max(int(wx[int(kx[f])]), int(wz[int(kz[f])])) > 1:
            n += 1
    return n


def test_flags_on_grouped_couplings(code) -> None:  # ruff: ignore[redefined-while-unused]
    """A flag on each grouped ancilla qubit catches the interaction hooks.

    A flag on each grouped ancilla qubit (across its couplings) catches the interaction hooks: single faults on the
    couplings that leave a weight-2 data error undetected exist without flags and are all caught with them.
    """
    state, prep = code
    d = _with(state, prep, [_grouped(_pair(state, "Z"), "Z")])
    gs = d.root.gadgets
    assert [(g.block, g.anc, len(g.support)) for g in gs] == [(0, 0, 2), (0, 1, 2), (0, 2, 2)]
    assert _bad_coupling_faults(d.circuit(), state) > 0
    flagged = d.clone()
    flagged.root.gadgets = [dataclasses.replace(g, flags=[(0, 2, i)]) for i, g in enumerate(gs)]
    check_noiseless(flagged.circuit(), state)
    assert _bad_coupling_faults(flagged.circuit(), state) == 0


def test_remove_block_keeps_the_others(code) -> None:  # ruff: ignore[redefined-while-unused]
    """Removing a block keeps the gadgets of the other blocks right."""
    state, prep = code
    z, x = _pair(state, "Z"), _pair(state, "X")
    d = _with(state, prep, [BlockSpec("Z", z), BlockSpec("X", x)])
    only_x = _with(state, prep, [BlockSpec("X", x)])
    d.root.remove_block(0)
    assert {g.block for g in d.root.gadgets} == {0}
    assert str(d.circuit().to_stim()) == str(only_x.circuit().to_stim())


def test_structure_classes(code) -> None:  # ruff: ignore[redefined-while-unused]
    """Designs get the right structure class."""
    state, prep = code
    z = _pair(state, "Z")
    assert _with(state, prep, []).structure() == "separate"
    assert _with(state, prep, [BlockSpec("Z", z[:1])]).structure() == "separate"  # a cat
    assert _with(state, prep, [BlockSpec("Z", z)]).structure() == "shared"
    assert _with(state, prep, [_grouped(z, "Z")]).structure() == "shared+grouped"
    single = BlockSpec(
        "Z",
        z[:1],
        groups=(
            (int(np.flatnonzero(z[0])[0]), int(np.flatnonzero(z[0])[1])),
            *((int(q),) for q in np.flatnonzero(z[0])[2:]),
        ),
        htilde=np.ones((1, 3), dtype=np.uint8),
    )
    assert _with(state, prep, [single]).structure() == "grouped"


def test_takeover_is_grouped(code) -> None:  # ruff: ignore[redefined-while-unused]
    """A check partly driven by a flag (takeover) is measured by two ancilla qubits on several data qubits: grouped."""
    state, prep = code
    root = make_node("D", prep, state)
    sz = sorted(int(q) for q in np.flatnonzero(_pair(state, "Z")[0]))
    root.gadgets = [Gadget("Z", len(root.base.ops), sz, flags=[(0, 4, 0)], drive=((2, 0), (3, 0)))]
    d = Design(root, 2)
    check_noiseless(d.circuit(), state)
    assert d.structure() == "grouped"
