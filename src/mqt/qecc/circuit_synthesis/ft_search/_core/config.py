# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run settings: one profile per fault distance t (code size), plus the run's cost weights (``Context``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# Search constants.
PW = 2.0  # progressive widening: a node may have ceil(PW * visits^0.5) children
C_UCT = 0.5
REWARD_SCALE = 0.05  # relative excess over the front at which the reward reaches 0
REWARD_TAIL = 0.2  # ... plus a tail of at most this, reaching 0 at ``REWARD_TAIL_SCALE``
REWARD_TAIL_SCALE = 0.2
CLASS_REWARD = 0.5  # weight of the reward against the front of the design's structure class (``Design.structure``)
TARGET_MARGIN = 3  # per-depth reward: target depths up to this far past the front's deepest point
NEAR_POLISH = 0.03  # completions within this relative excess also get a full polish and remove moves
POLISH_BACKLOG = 4  # near-front designs are polished while fewer polish jobs than this are waiting
MAX_MOVES = 12  # tree depth limit (moves from a preparation circuit)
MAX_REMOVES = 2  # designs with this many removes in their history get no further remove moves
REMOVE_K = 6  # edit move remove: single removals per design
REMOVE_PAIRS = 8  # ... and pairs
PREP_SHARE = 0.5  # every preparation circuit gets at least this fraction of an equal share of the root's visits
FAMILY_SHARE = 0.5  # at each node every move family gets at least this fraction of an equal share of the jobs
SOLVER_SHARE = 0.3  # no solver gets new move jobs while it has used more than this share of the move-job time
SOLVER_SHARE_WARM = 600.0  # ... after this many worker-seconds
RESTART_K = 2  # partial designs per front depth in the restart group
K_CRIGER = 3  # Criger move: ILP solutions per round
FRONT_SHARE = 0.25  # share of the workers for the front group's subtree (refining front designs) + polish
PART_SHARE = 0.5  # share of the workers for the seed and part jobs (``Profile.precompute_parts``)
DEEP_SHARE = 0.5  # share of the workers for deep jobs
RESTART_SHARE = 0.35  # share of the workers for descents from the restart group
RESTART_EVERY = 1200.0  # seconds between refreshes of the restart group
RESCORE_EVERY = 300.0  # seconds between re-scorings of all visits against the current front
COVER_TIME = 120.0  # seconds per cover ILP
ILP_TIME = 300.0  # seconds per promotion ILP (complete Criger promotion, late checks); step mode: min(120, this)

# Default run settings (``Context``).
LAM = 4.0
LAMS = (0.0, 1.0, 2.0, 4.0, 10.0, 20.0)
SYNTH = (("qecc_heuristic", 1), ("greedy", 1), ("random", 1))
POOL = 2


@dataclass
class Context:
    """What the solvers need of a run: the depth weights, the synthesis modes, the ILP limits."""

    lam: float = LAM  # weight of depth (ops per layer): cost = CNOTs + measurements + lam * depth
    synth: list[tuple[str, int]] = field(default_factory=lambda: list(SYNTH))
    synthesizers: dict = field(default_factory=dict)  # further synthesis modes by name: f(h, k) -> circuits
    pool: int = POOL  # synthesize pool * k circuits per mode and offer k chosen by their error properties
    lams: tuple[float, ...] = LAMS  # all depth weights of the run; () = just ``lam``
    cover_time: float = COVER_TIME
    ilp_time: float = ILP_TIME  # tightened to ``Profile.rollout_ilp_time`` inside the completion

    def cost(self, m: dict, lam: float | None = None) -> float:
        """Cost of a circuit's metrics (``circuit.metrics``) under the run's weights (``lam`` overrides the depth weight)."""
        lam = self.lam if lam is None else lam
        return m["cnot"] + m["meas"] + lam * m["depth"]

    def all_lams(self) -> tuple[float, ...]:
        """The run's depth weights incl. the main one, sorted."""
        return tuple(sorted({self.lam, *self.lams}))

    def extreme_lams(self) -> tuple[float, ...]:
        """Main λ plus the smallest and largest of the run: the λs rollouts and post-passes optimize for."""
        a = self.all_lams()
        return tuple(sorted({a[0], self.lam, a[-1]}))

    def best_per_lam(
        self, items: list, metrics_of: Callable[[Any], dict], lams: tuple[float, ...] | None = None
    ) -> list:
        """The items that are cheapest under at least one λ (default ``all_lams``), each once, the main λ's best first."""
        lams = self.all_lams() if lams is None else lams
        out = []
        for lam in sorted(lams, key=lambda x: x != self.lam):
            if items:
                best = min(items, key=lambda it: self.cost(metrics_of(it), lam))
                if not any(best is o for o in out):
                    out.append(best)
        return out


@dataclass
class Profile:
    """The limits of a run (by fault distance t)."""

    node_limit: float  # seconds per job (a move, or scoring a node)
    polish_limit: float  # seconds per full-polish job
    samples: int  # perturbed-cost samples of the cover / ILP moves
    rollout_ilp_time: float  # seconds per ILP inside the completion
    rollout_budget: float | None = None  # seconds for the finishers beyond the main depth weight (None: no limit)
    greedy_top_k: int = 30  # candidates the greedy finisher checks exactly per step
    greedy_by_order: bool = False  # greedy finisher: one fault order at a time (``greedy_ft.greedy_ft``)
    finishers: tuple[str, ...] = ("greedy",)  # how ``complete`` finishes a design, in order: "greedy" or "criger"
    block_rounds: int = 1  # rounds of the rollout's block step (``complete.block_repairs``)
    precompute_parts: bool = False  # protect the ancillas of the first blocks as parts at the start (seed jobs)
    # two-tier evaluation: a node job runs a quick rollout with this budget (no polish, no circuit stage); nodes whose
    # quick completions come within ``near_deep`` of the front get a deep job, the full rollout. None: one tier
    quick_budget: float | None = None
    near_deep: float = 0.15  # relative excess over the front that earns a node a deep job


PROFILES = {
    2: Profile(node_limit=1200, polish_limit=900, samples=3, rollout_ilp_time=60.0),
    3: Profile(
        node_limit=2400,
        polish_limit=1800,
        samples=2,
        rollout_ilp_time=120.0,
        rollout_budget=300.0,
        greedy_top_k=8,
        greedy_by_order=True,
        finishers=("criger", "greedy"),
        block_rounds=4,
        precompute_parts=True,
        quick_budget=60.0,
        near_deep=0.4,
    ),
}


def profile(t: int) -> Profile:
    """The limits for fault distance ``t`` (t = 1 runs with the limits of t = 2)."""
    return PROFILES[min(max(t, min(PROFILES)), max(PROFILES))]
