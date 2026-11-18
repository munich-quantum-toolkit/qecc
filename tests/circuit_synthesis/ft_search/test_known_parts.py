# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for known parts.

Known parts (``parts``): a completion records its FT ancilla, the rollout installs it in an open scope, and the
block + known protection move (``parts.with_known_parts``) offers it. [[17,1,5]] color code.
"""

from __future__ import annotations

import pytest
import stim

from mqt.qecc.circuit_synthesis.ft_search._core import parts
from mqt.qecc.circuit_synthesis.ft_search._core.circuit import Circuit
from mqt.qecc.circuit_synthesis.ft_search._core.complete import (
    _attach_parts,
    complete,
)
from mqt.qecc.circuit_synthesis.ft_search._core.config import Context
from mqt.qecc.circuit_synthesis.ft_search._core.design import Design, compile_node, make_node
from mqt.qecc.circuit_synthesis.ft_search._core.evaluation import check_cached, evaluate_cached, proposals
from mqt.qecc.circuit_synthesis.ft_search._core.solvers import BlockSolver
from mqt.qecc.circuit_synthesis.ft_search._core.verify import CSSState, check_ft, check_noiseless

from .helpers import color_17_1_5

CTX = Context(lam=2, lams=(0.0, 2.0, 10.0), synth=[("greedy", 1)], cover_time=30, pool=1)


@pytest.fixture
def start():
    """A design of the bare [[17,1,5]] color prep and its state, with no known parts."""
    parts.clear()
    prep, state = color_17_1_5("gate")
    return Design(make_node("D", prep, state), 2), state


def _block_design(d: Design) -> tuple[list, Design]:
    props = proposals(BlockSolver("Z", "weight"), d, (), CTX)
    assert props
    return props, props[0][1]


def _fresh_ok(d: Design, state: CSSState) -> bool:
    c = Circuit.from_stim(stim.Circuit(str(d.circuit().to_stim())), data=list(range(17)))
    check_noiseless(c, state)
    return check_ft(c, state, 2).ok


@pytest.mark.slow
def test_known_parts_attached_by_rollout_and_move(start) -> None:
    """A completion records its FT ancilla as a part, and the rollout and the move install it."""
    d, _state = start
    props, bd = _block_design(d)
    # a completion records its locally FT ancilla as a part
    comps = [x for _, x, _ in complete(bd, CTX, with_circuit=False, polish_budget=60)]
    assert all(evaluate_cached(x).ok for x in comps)
    local = [x for x in comps if check_cached(compile_node(x.node((0,))), x.node((0,)).target, 2)[0] is True]
    assert local  # some completion protects the ancilla on its own (others may catch its faults elsewhere)
    anc = local[0].node((0,))
    assert parts.state_key(anc, 2) in {k for k, _ in parts.snapshot()}
    assert any(k == parts.state_key(anc, 2) for k, _ in parts.take_new())
    # rollout recipe: the block design's open ancilla gets the known part
    d2, n = _attach_parts(bd, CTX, 2.0)
    assert n >= 1
    a2 = d2.node((0,))
    assert check_cached(compile_node(a2), a2.target, 2)[0] is True
    # move: block proposals plus variants with the known part on the new scope
    extra = parts.with_known_parts(d, props, CTX)
    assert extra
    assert all(lab.endswith("+parts") for lab, _ in extra)
    keys = {e["key"] for skey, e in parts.snapshot() if skey == parts.state_key(anc, 2)}
    assert all(parts.design_key(x.node((0,))) in keys for _, x in extra)
    assert all(x.history[-1].startswith("0:parts[") for _, x in extra)


def test_parts_travel_between_processes() -> None:
    """What one worker found (``take_new``) reaches another through ``merge``; ``snapshot`` keeps the (ops, depth)-front."""
    parts.clear()
    entry = {"key": "k1", "node": None, "metrics": {"ops": 10, "depth": 5}}
    worse = {"key": "k2", "node": None, "metrics": {"ops": 12, "depth": 6}}
    other = {"key": "k3", "node": None, "metrics": {"ops": 8, "depth": 9}}
    parts.merge([("s", entry), ("s", worse), ("s", other)])
    assert sorted(e["key"] for _, e in parts.snapshot()) == ["k1", "k3"]
    assert parts.take_new() == []
    parts.clear()
    assert parts.snapshot() == []
