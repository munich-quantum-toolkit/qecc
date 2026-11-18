# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Two-tier node evaluation (``Profile.quick_budget``): the node job's quick rollout has no circuit stage and no polish.

The deep job is the full rollout. [[17,1,5]] color code, depth-optimized prep.
"""

from __future__ import annotations

import dataclasses

import pytest

from mqt.qecc.circuit_synthesis.ft_search._core.config import Context, profile
from mqt.qecc.circuit_synthesis.ft_search._core.design import Design, make_node
from mqt.qecc.circuit_synthesis.ft_search._core.search import node_job
from mqt.qecc.circuit_synthesis.ft_search._core.verify import check_ft

from .helpers import color_17_1_5

CTX = Context(lam=10, lams=(2.0, 10.0), synth=[("greedy", 1)], cover_time=30, pool=2)


@pytest.fixture
def start():
    """A design of the bare [[17,1,5]] color prep and its state."""
    prep, state = color_17_1_5("depth")
    return Design(make_node("D", prep, state), 2), state


@pytest.mark.slow
def test_quick_and_deep_node_jobs(start) -> None:
    """The quick node job has no polish; the deep job has it."""
    d, state = start
    prof = dataclasses.replace(profile(2), quick_budget=30.0)
    quick = node_job((d, CTX, prof))
    assert not quick["errors"]
    assert quick["completions"]
    labels = [lab for lab, _, _ in quick["completions"]]
    assert not any("polish" in lab for lab in labels)  # neither polish nor circuit stage
    assert not any(x.root.opaque for _, x, _ in quick["completions"])
    deep = node_job((d, CTX, prof, True))
    assert not deep["errors"]
    assert any("polish" in lab for lab, _, _ in deep["completions"])
    for _, x, _ in quick["completions"] + deep["completions"]:
        assert check_ft(x.circuit(), state, 2).ok
