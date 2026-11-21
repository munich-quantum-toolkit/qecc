# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for greedy column elimination, cancellation-guided ancilla preps and the rollout's block step.

Greedy column elimination on the Steane code; cancellation-guided ancilla preps (``guided``) and the block step of the
rollout on the [[17,1,5]] color code (depth-optimized prep).
"""

from __future__ import annotations

import copy

import pytest

from mqt.qecc.circuit_synthesis.ft_search._core import moves
from mqt.qecc.circuit_synthesis.ft_search._core.blocks import dual
from mqt.qecc.circuit_synthesis.ft_search._core.circuit import metrics
from mqt.qecc.circuit_synthesis.ft_search._core.complete import block_repairs, complete
from mqt.qecc.circuit_synthesis.ft_search._core.config import Context, profile
from mqt.qecc.circuit_synthesis.ft_search._core.design import Design, ancilla_states, local_check, make_node
from mqt.qecc.circuit_synthesis.ft_search._core.evaluation import evaluate_cached, proposals
from mqt.qecc.circuit_synthesis.ft_search._core.faults import fault_table
from mqt.qecc.circuit_synthesis.ft_search._core.guided import greedy_candidates, guided_block_preps, guided_prep
from mqt.qecc.circuit_synthesis.ft_search._core.solvers import BlockSolver, GuideSolver
from mqt.qecc.circuit_synthesis.ft_search._core.verify import CSSState, check_ft, check_noiseless, weighted_failures

from .helpers import color_17_1_5, load_code

CTX = Context(lam=10, lams=(2.0, 10.0), synth=[("greedy", 1)], cover_time=30, pool=2)


@pytest.fixture
def start():
    """A design of the bare [[17,1,5]] color prep and its state."""
    prep, state = color_17_1_5("depth")
    return Design(make_node("D", prep, state), 2), state


@pytest.mark.parametrize("eps", [0.0, 0.5])
def test_greedy_candidates_prepare_the_state(eps: float) -> None:
    """Greedy column elimination gives distinct circuits that prepare the state, the same ones for the same seed."""
    h = load_code("steane").hx
    state = CSSState(h, dual(h))
    circs = greedy_candidates(h, 3, seed=5, eps=eps)
    assert circs
    assert len({str(c.to_stim()) for c in circs}) == len(circs)
    for c in circs:
        check_noiseless(c, state)
    assert [str(c.to_stim()) for c in greedy_candidates(h, 3, seed=5, eps=eps)] == [str(c.to_stim()) for c in circs]
    cnots = [metrics(c)["cnot"] for c in circs]
    assert cnots == sorted(cnots)


def _zx(d: Design) -> Design:
    d1 = proposals(BlockSolver("Z", "weight"), d, (), CTX)[0][1]
    return proposals(BlockSolver("X", "weight"), d1, (), CTX)[0][1]


def test_guided_prep_prepares_the_state_and_layered_is_shallower(start) -> None:
    """Guided preps prepare the ancilla state, and the layered ones are not deeper."""
    d, _ = start
    spec = _zx(d).root.blocks[1][0]
    h = spec.ancilla_generator()
    full, _ = ancilla_states(spec)
    q = [1 << a for a in range(spec.n_anc)]
    deps = {}
    for layered in (False, True):
        cs = [guided_prep(h, q, lambda _v, _s: False, track_z=True, seed=sd, layered=layered) for sd in range(6)]
        circs = [c for c in cs if c is not None]
        assert len(circs) == len(cs)
        for c in circs:
            check_noiseless(c, full)
        deps[layered] = min(metrics(c)["depth"] for c in circs)
    assert deps[True] <= deps[False]
    assert guided_prep(h, q, lambda v, _s: v != 0, track_z=True, seed=0, max_back=20) is None  # nothing allowed


def _cancel_pairs(design: Design) -> int:
    """Failing pairs of (X-block ancilla prep fault that does not reach the data, data prep fault): cancellation.

    The ops are told apart by their provenance tags (``gauge.build``: "a" = a block's ancilla circuit, "b" = the data
    prep).
    """
    root = design.root
    circ = design.circuit()
    table = fault_table(circ)
    rep = check_ft(circ, root.state, design.t, collect=True)
    n = 0
    assert circ.tags is not None
    for combo in rep.failing_reps:
        kinds = []
        for k in combo:
            tg = circ.tags[int(table.op_index[k])]
            reach = bool(table.data_x[k].any() or table.data_z[k].any())
            if tg[:2] == ("a", 1) and not reach:
                kinds.append("cancel")
            elif tg[0] == "b":
                kinds.append("data")
            else:
                kinds.append("other")
        n += len(combo) == 2 and sorted(kinds) == ["cancel", "data"]
    return n


def test_guided_preps_remove_cancellation(start) -> None:
    """Guided preps leave no cancelling fault pairs."""
    d, _ = start
    dzx = _zx(d)
    spec, ch = dzx.root.blocks[1]
    full, target = ancilla_states(spec)
    preps = guided_block_preps(dzx.root, 1, 2, k=3)
    assert preps
    for c in preps:
        check_noiseless(c, full)
        nd = copy.deepcopy(dzx.root)
        nd.blocks[1] = (spec, make_node(ch.name, c, full, target))
        assert _cancel_pairs(Design(nd, 2)) == 0


@pytest.mark.slow
def test_block_step_of_the_rollout(start) -> None:
    """The block step on a single unprotected Z block.

    A single unprotected Z block: the rollout's block step gives chains of an X block beside it (the Steane-like
    pair) with its ancilla protected by a nested Z block, the ancilla plain or cancellation-guided. One chain per depth
    weight is to be finished, the others follow ranked by remaining failures; ``complete`` hands all of them over as
    children and completes one to an FT design with both blocks.
    """
    d, state = start
    one = proposals(BlockSolver("Z", "weight"), d, (), CTX)[0][1]
    reps = block_repairs(one, CTX)
    assert reps
    assert all(tuple(spec.kind for spec, _ in x.root.blocks) == ("Z", "X") for _, x, _ in reps)  # beside
    assert all([s.kind for s, _ in x.root.blocks[1][1].blocks] == ["Z"] for _, x, _ in reps)  # nested
    assert any("+guided" in lab for lab, _, _ in reps)  # guided ancillas among the candidates
    assert sorted(lam for _, _, lams in reps for lam in lams) == sorted(CTX.extreme_lams())  # one finish per λ
    assert CTX.lam in reps[0][2]  # the main λ's chain first: the rollout follows it past its budget
    rest = [weighted_failures(evaluate_cached(x).raw, 2) for _, x, lams in reps if not lams]
    assert rest == sorted(rest)  # the others by promise
    for _, x, _ in reps:
        check_noiseless(x.circuit(), state)
    partials: list = []
    comps = complete(one, CTX, with_circuit=False, polish_budget=30, partials=partials)
    assert len(partials) == len(reps)
    assert any(len(x.root.blocks) == 2 for _, x, _ in comps if not x.root.opaque)


@pytest.mark.slow
def test_pair_from_scratch_in_the_rollout(start) -> None:
    """The block step gives a design without blocks the whole pair.

    A design without blocks (here the bare prep) gets the whole pair from the block step: a first block on the
    root, a block of the other type beside it and a nested block in that one's ancilla, chains of both seed types
    ranked together.
    """
    d, state = start
    reps = block_repairs(d, CTX)
    assert reps
    kinds = {tuple(spec.kind for spec, _ in x.root.blocks) for _, x, _ in reps}
    assert kinds <= {("Z", "X"), ("X", "Z")}
    assert ("Z", "X") in kinds
    assert all(len(x.root.blocks[1][1].blocks) == 1 for _, x, _ in reps)  # nested
    assert sorted(lam for _, _, lams in reps for lam in lams) == sorted(CTX.extreme_lams())
    rest = [weighted_failures(evaluate_cached(x).raw, 2) for _, x, lams in reps if not lams]
    assert rest == sorted(rest)
    for _, x, _ in reps:
        check_noiseless(x.circuit(), state)


def test_later_rounds_nest_a_block_in_the_ancilla(start) -> None:
    """The later rounds of the block step nest a block in the ancilla.

    The later rounds of the block step (``nest``, t >= 3) protect an unprotected block's ancilla from inside: a
    block of the other type in its ancilla scope, not a second block beside it. One round at t = 2, more from t = 3.
    """
    d, state = start
    one = proposals(BlockSolver("Z", "weight"), d, (), CTX)[0][1]
    reps = block_repairs(one, CTX, nest=True)
    assert reps
    for _, x, _ in reps:
        assert [spec.kind for spec, _ in x.root.blocks] == ["Z"]  # nothing beside
        assert [spec.kind for spec, _ in x.root.blocks[0][1].blocks] == ["X"]  # inside its ancilla
        check_noiseless(x.circuit(), state)
    assert sorted(lam for _, _, lams in reps for lam in lams) == sorted(CTX.extreme_lams())
    # the block that leaves the ancilla with the fewest local failures (over every cover and circuit) comes first
    props = proposals(BlockSolver("X", "weight"), one, (0,), CTX)
    fewest = min(sum(local_check(x.node((0,)), 2).raw_failures.values()) for _, x in props)
    assert all(sum(local_check(x.node((0,)), 2).raw_failures.values()) == fewest for _, x, _ in reps)


def test_guide_move(start) -> None:
    """Guided preps are a move on any plain block ancilla."""
    d, state = start
    one = proposals(BlockSolver("Z", "weight"), d, (), CTX)[0][1]
    props = proposals(GuideSolver(), one, (), CTX)
    assert props
    for _, x in props:
        assert x.root.blocks[0][1].origin == "guided"
        check_noiseless(x.circuit(), state)


@pytest.mark.slow
def test_block_children_one_per_cover_first(start) -> None:
    """Block moves release one child per cover first.

    Block moves release one child per cover before a second circuit of any cover (``moves.by_cover``), so a full
    cover is not crowded out by the cheaper lowest-order covers.
    """
    assert [lab for _, lab, _ in moves.by_cover([(1, "a[1]:x", None), (2, "a[1]:y", None), (3, "b[2]:x", None)])] == [
        "a[1]:x",
        "b[2]:x",
        "a[1]:y",
    ]
    d, _ = start
    fams = moves.families(profile(2))
    idx = next(i for i, (sv, _) in enumerate(fams["ancilla"]) if sv.name.startswith("blockZ:weight"))
    children, errors = moves.run(d, [()], "ancilla", idx, CTX, profile(2))
    assert children
    assert not errors
    covers = [lab.split("]")[0] for lab, _ in children]
    n = len(set(covers))
    assert len(set(covers[:n])) == n  # every cover once before any repeats
