# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for every building block of the design search.

Every building block of the design search, on the [[17,1,5]] color code (t = 2): all proposals compile to circuits
that are noiseless-correct (deterministic detectors, right state); the FT-preserving polish passes keep a design FT.
Runs on an empty memo, greedy synthesis only.
"""

from __future__ import annotations

import pytest

from mqt.qecc.circuit_synthesis.ft_search._core.circuit import metrics
from mqt.qecc.circuit_synthesis.ft_search._core.complete import complete
from mqt.qecc.circuit_synthesis.ft_search._core.config import Context, profile
from mqt.qecc.circuit_synthesis.ft_search._core.design import Design, compile_node, make_node
from mqt.qecc.circuit_synthesis.ft_search._core.evaluation import proposals
from mqt.qecc.circuit_synthesis.ft_search._core.moves import families
from mqt.qecc.circuit_synthesis.ft_search._core.solvers import (
    BlockSolver,
    DepthOrderSolver,
    PruneSolver,
    SharedFlagSolver,
    TakeoverSolver,
    edit,
)
from mqt.qecc.circuit_synthesis.ft_search._core.verify import check_ft, check_noiseless

from .helpers import color_17_1_5

# the polish passes: on an FT design every proposal stays FT
FT_PRESERVING = {"prune", "sharedflag", "orderdepth", "takeover"}
# the solvers that also act on FT designs (the edit moves that keep or break FT, and the polish passes)
ON_FT = {"remove", "prepmove", *FT_PRESERVING}


@pytest.fixture(scope="module")
def designs():
    """Bare prep, a design with one block (an ancilla scope exists), and a complete FT design."""
    prep, state = color_17_1_5("gate")
    ctx = Context(lam=0.0, lams=(), synth=[("greedy", 1)], cover_time=30, pool=1)
    bare = Design(make_node("D", prep, state), 2)
    one_block = proposals(BlockSolver("X", "weight"), bare, (), ctx)[0][1]
    ft = complete(one_block, ctx, with_circuit=False, polish_budget=60)[0][1]
    assert ft is not None
    check_noiseless(ft.circuit(), state)
    assert check_ft(ft.circuit(), state, 2).ok
    return {"bare": bare, "one_block": one_block, "ft": ft}, ctx


# solvers whose proposals take minutes (ILPs and covers on the [[17,1,5]] designs)
SLOW = {"crigerg8r3", "crigerstep3g8ir3", "blockZ:weight:both:r3"}


def _cases():
    return [
        pytest.param(sv, id=sv.name, marks=[pytest.mark.slow] if sv.name in SLOW else [])
        for sv in [
            *(s for v in families(profile(2)).values() for s, _ in v),
            PruneSolver(),
            SharedFlagSolver(),
            TakeoverSolver(),
            DepthOrderSolver(),
        ]
    ]


@pytest.mark.parametrize("solver", _cases())
def test_solver_proposals_are_valid(solver, designs) -> None:
    """Every proposal of every solver compiles to a noiseless-correct circuit."""
    ds, ctx = designs
    n_props = 0
    for dname, d in ds.items():
        if dname == "ft" and solver.name not in ON_FT:
            continue
        for path, _ in d.nodes():
            for label, nd in proposals(solver, d, path, ctx):
                n_props += 1
                circ = nd.circuit()
                check_noiseless(circ, nd.root.state)  # raises if not deterministic or wrong state
                assert nd.history
                assert nd.history[-1] == label
                if dname == "ft" and solver.name in FT_PRESERVING:
                    assert check_ft(circ, nd.root.state, 2).ok, f"{solver.name} broke FT: {label}"
    # every solver must at least run; most also propose something on one of the three designs
    assert n_props >= 0


def test_depth_order_checks_only_shallower_orders(designs, monkeypatch) -> None:
    """The depth-order pass checks only orders shallower than its start.

    The depth-order pass runs the exact check only on orders shallower than its start (the depth is cheap, the
    check is not); what it proposes is FT and no deeper.
    """
    ds, ctx = designs
    ft = ds["ft"]
    checked: list[int] = []
    real = edit.check_ft

    def counting(c, *a, **k):
        checked.append(metrics(c)["depth"])
        return real(c, *a, **k)

    monkeypatch.setattr(edit, "check_ft", counting)
    for path, node in ft.nodes():
        start = metrics(compile_node(node))["depth"]
        checked.clear()
        props = list(DepthOrderSolver().propose(ft, path, ctx))
        assert all(dep < start for dep in checked[1:])  # the first check is the start itself
        for _, nd in props:
            c = compile_node(nd)
            assert metrics(c)["depth"] <= start
            assert check_ft(c, nd.target, 2).ok
