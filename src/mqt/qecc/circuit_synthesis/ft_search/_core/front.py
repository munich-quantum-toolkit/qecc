# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""The (ops, depth) front of a run and the search reward: the relative ops excess over the front."""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_T = TypeVar("_T")


def pareto(items: Sequence[_T], point: Callable[[_T], tuple[int, int]], *, ties: bool = False) -> list[_T]:
    """The items whose (ops, depth) point no other item beats, by increasing ops.

    Args:
        items: The items.
        point: The (ops, depth) of an item.
        ties: Keep every item of a front point; by default only the first one in ``items``.

    Returns:
        The items on the (ops, depth) front.
    """
    pts = sorted(((point(x), i, x) for i, x in enumerate(items)), key=operator.itemgetter(0, 1))
    out, best_d, last = [], math.inf, None
    for pt, _, x in pts:
        if pt[1] < best_d:
            out.append(x)
            best_d, last = pt[1], pt
        elif ties and pt == last:
            out.append(x)
    return out


class Front:
    """The (ops, depth) front of the designs found so far."""

    def __init__(self) -> None:
        """Start with an empty front."""
        self.best: dict[int, int] = {}  # depth -> fewest ops found at exactly that depth

    def ops_at(self, depth: int) -> int | None:
        """Fewest ops of a front point with depth <= ``depth`` (None: nothing that shallow)."""
        return min((o for d, o in self.best.items() if d <= depth), default=None)

    def excess(self, ops: int, depth: int) -> float:
        """Relative excess over the front (< 0: the point improves it; -1 if nothing is that shallow)."""
        b = self.ops_at(depth)
        return -1.0 if b is None else (ops - b) / b

    def add(self, ops: int, depth: int) -> bool:
        """Add a point; True if it improved the front."""
        if self.excess(ops, depth) >= 0:
            return False
        self.best[depth] = min(ops, self.best.get(depth, ops))
        return True

    def points(self) -> list[tuple[int, int]]:
        """The front points as (ops, depth), by increasing depth."""
        out, b = [], math.inf
        for d in sorted(self.best):
            if self.best[d] < b:
                out.append((self.best[d], d))
                b = self.best[d]
        return out

    def near(self, ops: int, depth: int, rel: float) -> bool:
        """Whether (ops, depth) is at most ``rel`` above the front."""
        return self.excess(ops, depth) <= rel

    def targets(self, margin: int = 0) -> list[int]:
        """The target depths of the reward: from the front's shallowest point to ``margin`` past its deepest."""
        pts = self.points()
        return list(range(pts[0][1], min(DMAX, pts[-1][1] + margin) + 1)) or [DMAX] if pts else [DMAX]

    def dominated(self, ops: int, depth: int) -> bool:
        """Some other front point is at least as good in ops and depth."""
        return any(o <= ops and d <= depth and (o, d) != (ops, depth) for o, d in self.points())

    def max_depth(self) -> int:
        """Depth of the deepest front point (``DMAX`` if the front is empty)."""
        pts = self.points()
        return pts[-1][1] if pts else DMAX


DMAX = 64  # per-depth rewards: target depths 0..DMAX (deeper designs score 0 everywhere)


def _score(e: float, scale: float, tail: float, tail_scale: float) -> float:
    if e <= 0:
        return 1.0
    return max(0.0, 1.0 - e / scale, tail * (1.0 - e / tail_scale))


def depth_rewards(
    front: Front, ops: int, depth: int, scale: float, tail: float = 0.0, tail_scale: float = 0.2
) -> list[float]:
    """The reward of a design per target depth T = 0..DMAX.

    0 for T < depth, else from the ops excess over the front's best point of depth <= T: 1 on or beyond the front,
    falling linearly to 0 at relative excess ``scale``, plus a tail (``tail`` at the front, 0 at ``tail_scale``). A
    dominated design is scored as one op worse than the front.
    """
    out = [0.0] * (DMAX + 1)
    dom = front.dominated(ops, depth)
    for td in range(depth, DMAX + 1):
        b = front.ops_at(td)
        if b is None:
            out[td] = 1.0
            continue
        e = (ops - b) / b
        out[td] = _score(1 / b if dom and e <= 0 else e, scale, tail, tail_scale)
    return out
