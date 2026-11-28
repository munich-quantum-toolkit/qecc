# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for parts computed ahead.

Parts computed ahead (``Profile.precompute_parts``): the ancillas of the first blocks on a prep, each completed as a
design of its own, become parts that the rollout installs. [[17,1,5]] color code, depth-optimized prep.
"""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.ft_search._core import parts
from mqt.qecc.circuit_synthesis.ft_search._core.complete import (
    _attach_parts,
    ancilla_seeds,
)
from mqt.qecc.circuit_synthesis.ft_search._core.config import Context, profile
from mqt.qecc.circuit_synthesis.ft_search._core.design import Design, ancilla_states, make_node
from mqt.qecc.circuit_synthesis.ft_search._core.evaluation import evaluate_cached, proposals
from mqt.qecc.circuit_synthesis.ft_search._core.search import part_job, seed_job
from mqt.qecc.circuit_synthesis.ft_search._core.solvers import BlockSolver

from .helpers import color_17_1_5

CTX = Context(lam=10, lams=(2.0, 10.0), synth=[("greedy", 1)], cover_time=30, pool=2)


@pytest.fixture
def start():
    """A design of the bare [[17,1,5]] color prep, with no known parts."""
    parts.clear()
    prep, state = color_17_1_5("depth")
    return Design(make_node("D", prep, state), 2)


def test_seeds_are_the_first_blocks_ancillas(start) -> None:
    """One seed per ancilla state of the kept block proposals of each type; each seed is that ancilla alone."""
    seeds = ancilla_seeds(start, CTX)
    assert {lab.split(":")[1] for lab, _ in seeds} == {"blockZ", "blockX"}
    keys = [parts.state_key(d.root, d.t) for _, d in seeds]
    assert len(keys) == len(set(keys))
    _lab, d1 = proposals(BlockSolver("Z", "weight"), start, (), CTX)[0]
    spec, ch = d1.root.blocks[-1]
    full, target = ancilla_states(spec)
    assert parts.state_key(ch, 2) in keys
    seed = next(d for _, d in seeds if parts.state_key(d.root, d.t) == parts.state_key(ch, 2))
    assert np.array_equal(seed.root.state.sx, full.sx)
    assert np.array_equal(seed.root.target.sx, target.sx)


@pytest.mark.slow
def test_part_job_stores_a_part_the_rollout_installs(start) -> None:
    """A part job on a seed whose ancilla is not FT records FT protections of that state.

    The rollout then installs one in the same block's ancilla scope.
    """
    out = seed_job((start, CTX, profile(2)))
    assert not out["errors"]
    seeds = out["seeds"]
    open_seeds = [d for _, d in seeds if not evaluate_cached(d).ok]
    assert open_seeds
    seed = open_seeds[0]
    assert not parts.known(seed.root, 2)
    res = part_job((seed, CTX, profile(2)))
    assert not res["errors"]
    assert parts.known(seed.root, 2)
    # the same ancilla state as a block's ancilla in a design: the rollout's parts step fills it
    key = parts.state_key(seed.root, 2)
    for kind in ("Z", "X"):
        for _, d1 in proposals(BlockSolver(kind, "weight"), start, (), CTX):
            if parts.state_key(d1.root.blocks[-1][1], 2) == key:
                d2, n = _attach_parts(d1, CTX, CTX.lam)
                assert n >= 1
                assert any(":parts[" in h for h in d2.history)
                return
    pytest.fail("no block proposal has the seed's ancilla state")
