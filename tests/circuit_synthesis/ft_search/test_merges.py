# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for rectangle partitions and the merge moves.

Rectangle partitions and the merge moves (``solvers.merge``, ``MergeSolver``) on two overlapping checks of the
[[17,1,5]] color code: the special cases (cats, bare, block, grouped block) come out as expected, and every
proposal of every merge move compiles to a noiseless-correct circuit.
"""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.ft_search._core.blocks import BlockSpec
from mqt.qecc.circuit_synthesis.ft_search._core.config import Context
from mqt.qecc.circuit_synthesis.ft_search._core.design import Design, make_node
from mqt.qecc.circuit_synthesis.ft_search._core.gauge import Gadget
from mqt.qecc.circuit_synthesis.ft_search._core.solvers import MergeSolver
from mqt.qecc.circuit_synthesis.ft_search._core.solvers import merge as rx
from mqt.qecc.circuit_synthesis.ft_search._core.verify import check_noiseless

from .test_gadgets import _with
from .test_grouped_blocks import _pair, code  # ruff: ignore[unused-import]  (fixture)


def _kinds(parts):
    return sorted(
        "bare" if w == "bare" else ("grouped" if x.grouped else ("split" if x.split else "plain")) for w, x in parts
    )


def test_special_cases(code) -> None:  # ruff: ignore[redefined-while-unused]
    """The special cases of the rectangle partitions."""
    state, _ = code
    checks = _pair(state, "Z")
    cats_r = rx.unshare_max(rx.rects_of(BlockSpec("Z", checks)))
    assert rx.valid(checks, cats_r)
    assert len(cats_r) == 8
    assert _kinds(rx.realize("Z", checks, cats_r, None)) == ["plain", "plain"]  # two one-row blocks = two cats
    bare = rx.group_to(cats_r, None)
    assert _kinds(rx.realize("Z", checks, bare, None)) == ["bare", "bare"]
    block = rx.share_max(cats_r)
    (part,) = rx.realize("Z", checks, block, None)
    assert part[0] == "block"
    assert isinstance(part[1], BlockSpec)
    assert not part[1].split  # the plain transversal block: old keys and targets
    gb = rx.group_to(block, None)
    (part,) = rx.realize("Z", checks, gb, None)
    assert isinstance(part[1], BlockSpec)
    assert part[1].grouped
    assert part[1].n_anc == 3


@pytest.mark.parametrize("op", ["macro", "group"])
def test_merge_moves_compile(code, op) -> None:  # ruff: ignore[redefined-while-unused]
    """Every proposal of the merge moves compiles to a noiseless-correct circuit."""
    state, prep = code
    checks = _pair(state, "Z")
    root = make_node("D", prep, state)
    end = len(root.base.ops)
    for row in checks:  # two bare checks at the end of the prep
        root.gadgets.append(Gadget("Z", end, [int(q) for q in np.nonzero(row)[0]]))
    d = Design(root, 2)
    ctx = Context(synth=[("greedy", 1)])
    # the same two checks as bare checks, as two cats and as one block (macros of the joined gadget)
    macros = dict(MergeSolver("macro").propose(d, (), ctx))
    starts = [d, *(d.with_node((), macros[f"blockmerge:macro[g0+g1:{m}]"]) for m in ("cats", "block"))]
    props = [(s, nd) for s in starts for _, nd in MergeSolver(op).propose(s, (), ctx)]
    assert props  # every step applies to one of them
    for s, nd in props:
        check_noiseless(s.with_node((), nd).circuit(), state)


def test_all_gadgets_joined(code, monkeypatch) -> None:  # ruff: ignore[redefined-while-unused]
    """All bare checks of a type at one time point form one unit.

    All bare checks of a type at one time point form one unit: one macro gives the Steane-like block on all of them,
    one the grouped block, without a chain of pairwise shares.
    """
    state, prep = code
    # three overlapping Z-type stabilizers of the state (they share qubits 0, 1 and 4)
    supports = [[0, 1, 2, 4, 6], [0, 1, 3, 4, 7], [4, 5, 8, 9, 10]]
    root = make_node("D", prep, state)
    end = len(root.base.ops)
    for support in supports:
        root.gadgets.append(Gadget("Z", end, support))
    d = Design(root, 2)
    ctx = Context(synth=[("greedy", 1)])
    monkeypatch.setattr(rx, "MAX_PROPS", 64)
    props = dict(MergeSolver("macro").propose(d, (), ctx))
    for m in ("block", "groupedblock"):
        nd = props[f"blockmerge:macro[g0+g1+g2:{m}]"]
        assert len(nd.blocks) == 1
        assert np.asarray(nd.blocks[0][0].checks).shape[0] == 3
        assert not [g for g in nd.gadgets if g.block is None]
        check_noiseless(d.with_node((), nd).circuit(), state)


def test_every_gadget_gets_a_proposal(code, monkeypatch) -> None:  # ruff: ignore[redefined-while-unused]
    """Every gadget gets a proposal under a small proposal cap.

    With a small proposal cap the large gadget (a block on all Z stabilizers) still gets one: proposals go round the
    gadgets in turn.
    """
    state, prep = code

    z = _pair(state, "Z")
    d = _with(state, prep, [BlockSpec("Z", state.sz % 2)])
    end = len(d.root.base.ops)
    for row in z:
        d.root.gadgets.append(Gadget("Z", end, [int(q) for q in np.nonzero(row)[0]]))
    ctx = Context(synth=[("greedy", 1)])
    monkeypatch.setattr(rx, "MAX_PROPS", 4)
    labs = [lab for lab, _ in MergeSolver("macro").propose(d, (), ctx)]
    assert len(labs) == 4
    assert any("[b0:" in lab for lab in labs)
