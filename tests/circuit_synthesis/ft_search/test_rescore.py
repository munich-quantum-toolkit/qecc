# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""``search.rescore``: node values rebuilt from the recorded visits against the current front."""

from __future__ import annotations

import pytest

from mqt.qecc.circuit_synthesis.ft_search._core.front import DMAX, Front, depth_rewards
from mqt.qecc.circuit_synthesis.ft_search._core.search import TNode, rescore


def _chain() -> tuple[TNode, TNode, TNode, TNode]:
    """Root -> a -> c and root -> b -> c (a DAG: c has two parents)."""
    root, a, b, c = (TNode(x, None, "partial") for x in "rabc")
    for p, q in ((root, a), (root, b), (a, c), (b, c)):
        p.children.append(q)
        q.parents.append(p)
    return root, a, b, c


def _score(front: Front):
    return lambda ops, depth: depth_rewards(front, ops, depth, 0.05, 0.2, 0.2)


def test_each_ancestor_counts_a_visit_once() -> None:
    """A visit is backed up to each ancestor once."""
    root, a, b, c = _chain()
    c.found = [((60, 9),), ()]  # one visit with a new design, one without
    a.found = [()]
    front = Front()
    front.add(60, 9)
    rescore([root, a, b, c], _score(front))
    assert (root.rn, a.rn, b.rn, c.rn) == (3, 3, 2, 2)
    assert [x.rsum[9] for x in (root, a, b, c)] == pytest.approx([1.0] * 4)
    assert all(c.rsum[T] == pytest.approx(0.0) for T in range(9))


def test_a_better_front_lowers_old_rewards() -> None:
    """Old visits lose value when the front improves."""
    root, a, b, c = _chain()
    c.found = [((60, 9),)]
    front = Front()
    front.add(60, 9)
    rescore([root, a, b, c], _score(front))
    assert c.rsum[9] == pytest.approx(1.0)
    front.add(50, 9)
    rescore([root, a, b, c], _score(front))
    assert c.rsum[9] == pytest.approx(0.0)  # 20 % above the front: beyond the tail
    assert c.rn == 1


def test_a_visit_counts_its_best_design_per_depth() -> None:
    """A visit scores its best design per target depth."""
    _, _, _, c = _chain()
    c.parents = []
    c.found = [((58, 9), (70, 7))]
    front = Front()
    front.add(58, 9)
    front.add(70, 7)
    rescore([c], _score(front))
    assert c.rsum[7] == pytest.approx(1.0)
    assert c.rsum[8] == pytest.approx(1.0)
    assert c.rsum[9] == pytest.approx(1.0)
    assert len(c.rsum) == DMAX + 1


def test_timed_out_visits_are_neutral() -> None:
    """A timed-out visit (None) neither adds value nor counts as a visit, so it cannot pull a node's mean to 0."""
    root, a, b, c = _chain()
    c.found = [((60, 9),), None]
    front = Front()
    front.add(60, 9)
    rescore([root, a, b, c], _score(front))
    assert c.rn == 1
    assert c.rsum[9] == pytest.approx(1.0)
