# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""The search loop: a DAG of designs (one node per design key), asynchronous over a worker pool.

The root's children are one node per preparation circuit and the front group, whose children (FT designs on the
front) get the edit moves. Jobs: node jobs (exact evaluation and ``complete``), deep jobs (the full rollout for nodes
whose quick rollout came near the front), move jobs (one solver on a node, ``moves.run``), polish jobs (front
designs) and seed and part jobs (``Profile.precompute_parts``).

Selection: UCB on the mean visit reward of a node's subtree for the descent's target depth; a visit earns the best
reward (``front.depth_rewards``) of the designs it found first, so a subtree that only repeats itself loses value.
Progressive widening allows ceil(PW * visits^0.5) children per node; each slot goes to a move family (minimum share,
then yield), within it to the solver used least at that node. A restart group of good partial designs and the front
group each get a share of the workers.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from . import moves, parts, polish
from .circuit import metrics, ops_depth
from .complete import ancilla_seeds, complete
from .config import (
    C_UCT,
    CLASS_REWARD,
    DEEP_SHARE,
    FAMILY_SHARE,
    FRONT_SHARE,
    MAX_REMOVES,
    NEAR_POLISH,
    PART_SHARE,
    POLISH_BACKLOG,
    PREP_SHARE,
    PW,
    RESCORE_EVERY,
    RESTART_EVERY,
    RESTART_K,
    RESTART_SHARE,
    REWARD_SCALE,
    REWARD_TAIL,
    REWARD_TAIL_SCALE,
    SOLVER_SHARE,
    SOLVER_SHARE_WARM,
    TARGET_MARGIN,
)
from .design import Design
from .evaluation import evaluate_cached
from .front import DMAX, Front, depth_rewards
from .kernels import warm
from .pool import JobTimeout, job_time_limit, kill_pool, process_pool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import ProcessPoolExecutor

    from .config import (
        Context,
        Profile,
    )


@dataclass(eq=False)
class TNode:
    """A node of the search tree."""

    key: str
    design: Design | None
    kind: str  # "root", "prep", "partial", "ft" (front design: edit moves), "front" (group)
    family: str = ""
    label: str = ""
    status: str = "new"  # new -> running -> ready | dead
    visits: int = 0
    children: list = field(default_factory=list)
    parents: list = field(default_factory=list)
    open_scopes: list = field(default_factory=list)
    todo: list = field(default_factory=list)  # move jobs not run yet: (family, solver index)
    moving: bool = False  # a move job is running
    pending: dict = field(default_factory=dict)  # (family, solver index) -> [(label, design)] not released yet
    rr: int = 0  # round-robin position over the pending families
    used: dict = field(default_factory=dict)  # family -> children released + move jobs run here (minimum share)
    used_arm: dict = field(default_factory=dict)  # (family, solver index) -> the same per solver
    next_move: tuple = ()  # (family, solver index) of the move job ``widen`` picked
    route: tuple = ()  # move families on the way here (front points count for all of them)
    rsum: list = field(default_factory=lambda: [0.0] * (DMAX + 1))  # per target depth: sum of visit rewards
    rn: int = 0
    found: list = field(default_factory=list)  # per visit: (ops, depth, structure class) of the designs it found first
    side: bool = False  # in the front group's subtree (counts against its worker share)
    exhausted: bool = False
    prep_key: str = ""  # key of the preparation circuit this node grew from
    points: list = field(default_factory=list)  # (ops, depth) of its completions
    via_restart: bool = False  # evaluated from a restart-group descent


def rescore(tnodes: list[TNode], score: Callable[..., list]) -> None:
    """Rebuild ``rsum`` and ``rn`` of ``tnodes`` from their recorded visits.

    ``score(ops, depth, structure class)`` is the reward per target depth; a visit counts the best reward of its designs,
    for the node and each ancestor once.
    """
    memo: dict = {}
    sums = {id(x): np.zeros(DMAX + 1) for x in tnodes}
    counts = dict.fromkeys(sums, 0)
    for n in tnodes:
        visits = [pts for pts in n.found if pts is not None]  # None: a timed-out visit, neutral (see ``node_ready``)
        if not visits:
            continue
        vec = np.zeros(DMAX + 1)
        for pts in visits:
            for p in pts:
                if p not in memo:
                    memo[p] = np.asarray(score(*p))
            if pts:
                vec += np.max([memo[p] for p in pts], axis=0)
        stack, seen = [n], set()
        while stack:
            x = stack.pop()
            if id(x) in seen:
                continue
            seen.add(id(x))
            sums[id(x)] += vec
            counts[id(x)] += len(visits)
            stack += x.parents
    for x in tnodes:
        x.rsum, x.rn = sums[id(x)].tolist(), counts[id(x)]


def _timed(limit: float, out: dict, body: Callable[[], object]) -> dict:
    """Run ``body`` (which fills ``out``) under the job time limit; add its error and its seconds to ``out``."""
    out.setdefault("errors", [])
    t0 = time.time()
    try:
        with job_time_limit(limit):
            body()
    except (Exception, JobTimeout):  # ruff: ignore[blind-except]
        out["errors"].append(traceback.format_exc(limit=-4))
    out["seconds"] = time.time() - t0
    return out


def node_job(args: tuple) -> dict:
    """Exact evaluation and rollout of a node; ``args``: (design, ctx, profile[, deep]).

    With a quick budget in the profile the rollout is the quick one unless ``deep``. On a timeout the completions are
    those verified so far.
    """
    design, ctx, prof, *rest = args
    quick = prof.quick_budget is not None and not (rest and rest[0])
    found: list = []
    out: dict = {"completions": found, "ok": False, "open": [], "partials": []}

    def body() -> None:
        ev = evaluate_cached(design)
        out["ok"], out["open"] = ev.ok, ev.open_scopes
        out["completions"] = complete(
            design,
            ctx,
            with_circuit=not quick,
            polish_budget=0.0 if quick else prof.node_limit / 8,
            partials=out["partials"],
            quick_budget=prof.quick_budget if quick else None,
            found=found,
        )

    return _timed(prof.node_limit, out, body)


def move_job(args: tuple) -> dict:
    """Run one move (solver configuration) on a design."""
    design, open_scopes, fam, idx, ctx, prof = args
    out: dict = {"children": [], "errors": [], "family": fam, "idx": idx}

    def body() -> None:
        out["children"], errors = moves.run(design, open_scopes, fam, idx, ctx, prof)
        out["errors"] += errors

    return _timed(prof.node_limit, out, body)


def polish_job(args: tuple) -> dict:
    """Fully polish a front design."""
    design, ctx, prof = args
    out: dict = {"designs": []}

    def body() -> None:
        # on a timeout, out["designs"] holds the candidates verified so far
        front = polish.full(design, ctx, budget=0.8 * prof.polish_limit, found=out["designs"])
        out["designs"] = [d for d in front if evaluate_cached(d).ok]

    return _timed(prof.polish_limit, out, body)


def seed_job(args: tuple) -> dict:
    """The ancillas of the first blocks on a preparation circuit, each as a design of its own (``ancilla_seeds``)."""
    design, ctx, prof = args
    out: dict = {"seeds": []}

    def body() -> None:
        out["seeds"] = ancilla_seeds(design, ctx)

    return _timed(prof.node_limit, out, body)


def part_job(args: tuple) -> dict:
    """Complete one ancilla as a design of its own: ``evaluate_cached`` records its FT versions as parts of its state."""
    design, ctx, prof = args
    out: dict = {}

    def body() -> None:
        complete(design, ctx, with_circuit=False, polish_budget=prof.node_limit / 8)

    return _timed(prof.node_limit, out, body)


def _point(d: Design) -> tuple[int, int]:
    return ops_depth(d.circuit())


def _job(fn: Callable[[tuple], dict], args: tuple, known_parts: list) -> dict:
    """Run one job in a worker with the run's known parts; return its result with the parts it found."""
    parts.merge(known_parts)
    out = fn(args)
    out["new_parts"] = parts.take_new()
    return out


def search(
    start: Design | list[Design],
    ctx: Context,
    prof: Profile,
    *,
    workers: int,
    time_limit: float,
) -> list[dict]:
    """Run the search from the designs ``start`` for ``time_limit`` seconds on ``workers`` processes.

    Ctrl-C stops the search early.

    Returns:
        A record per FT design on the final front (several designs can share a front point), by ops: its metrics and
        stim circuit.
    """
    warm()  # compile the numba kernels before the clock starts
    deadline = time.time() + time_limit
    front = Front()
    class_fronts: dict[str, Front] = {}  # structure class -> its own front (``Design.structure``)
    nodes: dict[str, TNode] = {}
    root = TNode("root", None, "root", status="ready")
    fgroup = TNode("front", None, "front", status="ready")
    rgroup = TNode("restart", None, "front", status="ready")  # not under the root: its own share of the descents
    from_restart: set = set()  # running jobs started by a restart-group descent
    solutions: dict[str, dict] = {}  # design key -> record of every FT design found
    polished: set[str] = set()
    front_members: set[str] = set()  # structured designs handed to the front group
    stats: dict[str, list] = {}  # job family -> [jobs, seconds, new front points]
    queue: list = []  # (job kind, args, tnode) waiting for a worker (prep nodes, front designs, polish)
    running: dict = {}
    counter = {"t": 0}  # descents so far (they cycle over the target depths)
    oom: set = set()  # (prep key, family, solver index): that solver ran out of memory on that preparation circuit
    move_arm: dict = {}  # running move job (future) -> its (family, solver index)

    def stat(name: str, seconds: float, new: int = 0) -> None:
        st = stats.setdefault(name, [0, 0.0, 0])
        st[0] += 1
        st[1] += seconds
        st[2] += new

    def value(n: TNode, td: int) -> float:
        """Mean visit reward of the subtree for target depth ``td``."""
        return n.rsum[td] / n.rn if n.rn else 0.0

    def backup(n: TNode, r: list, pts: tuple | None) -> None:
        """Add one visit's rewards to ``n`` and each ancestor once; ``pts`` (None: a stand-in) for ``rescore``."""
        n.found.append(pts)
        hot = [(td, v) for td, v in enumerate(r) if v]
        stack, seen = [n], set()
        while stack:
            x = stack.pop()
            if id(x) in seen:
                continue
            seen.add(id(x))
            for td, v in hot:
                x.rsum[td] += v
            x.rn += 1
            stack += x.parents

    zero = [0.0] * (DMAX + 1)

    def rewards(ops: int, depth: int, cls: str | None = None) -> list:
        """A new design's reward per target depth.

        The larger of the reward against the front and ``CLASS_REWARD`` times that against its class front ``cls``.
        """
        r = depth_rewards(front, ops, depth, REWARD_SCALE, REWARD_TAIL, REWARD_TAIL_SCALE)
        if cls is None or cls not in class_fronts or not CLASS_REWARD:
            return r
        rc = depth_rewards(class_fronts[cls], ops, depth, REWARD_SCALE, REWARD_TAIL, REWARD_TAIL_SCALE)
        return [max(a, CLASS_REWARD * b) for a, b in zip(r, rc, strict=True)]

    def vmax(vs: list) -> list:
        return [max(x) for x in zip(*vs, strict=False)] if vs else zero

    def near(ops: int, depth: int, rel: float) -> bool:
        """Within ``rel`` of the front and no deeper than the front (a dominated deep design is not near it)."""
        return front.near(ops, depth, rel) and depth <= front.max_depth() + TARGET_MARGIN

    def link(parent: TNode, child: TNode) -> None:
        if child not in parent.children:
            parent.children.append(child)
        if parent not in child.parents:
            child.parents.append(parent)

    def record(d: Design, label: str, source: TNode, base: Design | None = None) -> tuple[int, list, tuple | None]:
        """Record an FT design found; ``base``: the structured design an opaque one came from.

        Returns (1 if it improved the front, its reward per target depth, its point if it is new). Only new designs
        earn a reward.
        """
        m = metrics(d.circuit())
        k = d.key()
        new, r, pt = 0, zero, None
        src = base if d.root.opaque and base is not None else d
        cls = src.structure()
        if k not in solutions:
            r, pt = rewards(m["ops"], m["depth"], cls), (m["ops"], m["depth"], cls)
            solutions[k] = {"metrics": m, "stim": str(d.circuit().to_stim())}
        is_near = near(m["ops"], m["depth"], NEAR_POLISH)
        class_fronts.setdefault(cls, Front()).add(m["ops"], m["depth"])
        if front.add(m["ops"], m["depth"]):
            new = 1
            logger.info("front point: %d operations at depth %d (%s)", m["ops"], m["depth"], label)
        # full polish of front designs (near-front ones while the backlog is small)
        gate = new or (is_near and sum(q[0] == "polish" for q in queue) < POLISH_BACKLOG)
        if k not in polished and gate:
            polished.add(k)
            queue.append(("polish", (d, ctx, prof), source))
        if gate and d.root.opaque and base is not None and not base.root.opaque and base.key() not in polished:
            polished.add(base.key())
            queue.append(("polish", (base, ctx, prof), source))
        struct = d if not d.root.opaque else base
        if new and struct is not None and not struct.root.opaque and moves.removes(struct) < MAX_REMOVES:
            sk = struct.key()
            if sk not in front_members and sk not in nodes:
                front_members.add(sk)
                fgroup.pending.setdefault(("edit", -1), []).append((
                    label,
                    Design(struct.root, struct.t, [], struct.full_history()),
                ))
                if struct is not d and sk not in polished:
                    polished.add(sk)
                    queue.append(("polish", (struct, ctx, prof), source))
        return new, r, pt

    def ucb(parent: TNode, child: TNode, td: int) -> float:
        return value(child, td) + C_UCT * math.sqrt(math.log(parent.visits + 1) / (child.visits + 1))

    def fam_yield(fam: str) -> float:
        """Front points per worker time of a move family's nodes so far (unseen families first)."""
        _jobs, secs, new = stats.get(fam, [0, 0.0, 0])
        return (new + 1) / (1 + secs / 600)

    def best_fam(fams: list, rr: int) -> str:
        top = max(fam_yield(f) for f in fams)
        best = [f for f in fams if fam_yield(f) >= top - 1e-9]
        return best[rr % len(best)]

    def pick_fam(n: TNode, fams: list) -> str:
        """The family for the next child or move job at ``n``: one below its minimum share, else the best yield."""
        total = sum(n.used.values())
        low = [f for f in fams if n.used.get(f, 0) < FAMILY_SHARE * total / len(fams)]
        f = min(low, key=lambda f: n.used.get(f, 0)) if low else best_fam(fams, n.rr)
        n.used[f] = n.used.get(f, 0) + 1
        return f

    def pick_arm(n: TNode, arms: list) -> tuple:
        """The solver (family, index) for the next child or move job at ``n``: ``pick_fam``, then its least used."""
        fam = pick_fam(n, list(dict.fromkeys(a[0] for a in arms)))
        arm = min((a for a in arms if a[0] == fam), key=lambda a: (n.used_arm.get(a, 0), a[1]))
        n.used_arm[arm] = n.used_arm.get(arm, 0) + 1
        return arm

    def release(n: TNode, arm: tuple, strict: bool = False) -> TNode | None:
        """Release the next pending child of ``n`` from ``arm`` (then the others, unless ``strict``); link known ones."""
        while any(n.pending.values()):
            if not n.pending.get(arm):
                if strict:
                    return None
                arm = pick_arm(n, [a for a in n.pending if n.pending[a]])
            n.rr += 1
            label, d = n.pending[arm].pop(0)
            k = d.key()
            if k in nodes:
                link(n, nodes[k])
                continue
            fam = arm[0]
            c = TNode(
                k,
                d,
                "ft" if n is fgroup else "partial",
                family=fam,
                label=label,
                side=n.side or n is fgroup,
                route=tuple(dict.fromkeys((*n.route, fam))),
                prep_key=n.prep_key,
            )
            nodes[k] = c
            link(n, c)
            return c
        return None

    def over_share(arm: tuple) -> bool:
        """Whether solver ``arm`` used more than ``SOLVER_SHARE`` of the move-job time, running jobs included."""
        now = time.time()
        live = [(a, now - t0) for f_, (k_, _, t0) in running.items() if k_ == "move" for a in [move_arm.get(f_)]]
        tot = sum(v[1] for k_, v in stats.items() if k_.startswith("solver:")) + sum(s for _, s in live)
        st = stats.get(f"solver:{arm[0]}/{arm[1]}", [0, 0.0, 0])[1] + sum(s for a, s in live if a == arm)
        return tot > SOLVER_SHARE_WARM and st > SOLVER_SHARE * tot

    def open_todo(n: TNode) -> list:
        """The move jobs left at ``n``: not of a solver that ran out of memory on this preparation circuit."""
        return [tuple(a) for a in n.todo if (n.prep_key, *a) not in oom]

    def todo_arms(n: TNode) -> list:
        """The move jobs ``n`` may start now: also not of a solver over ``SOLVER_SHARE``."""
        return [a for a in open_todo(n) if not over_share(a)]

    def widen(n: TNode) -> tuple[str, TNode] | None:
        """A new child to evaluate, or a move job to run; None if neither is possible now.

        The solver is picked first (``pick_arm``), then its pending child or else its move job, so solvers with many
        children do not starve the others.
        """
        arms = [a for a in n.pending if n.pending[a]] + (todo_arms(n) if not n.moving else [])
        if not arms:
            return None
        arm = pick_arm(n, list(dict.fromkeys(arms)))
        if not n.pending.get(arm):
            n.next_move = arm
            return ("move", n)
        c = release(n, arm)
        if c is not None:
            return ("node", c)
        free = todo_arms(n)
        if free and not n.moving:
            n.next_move = free[0]
            return ("move", n)
        return None

    def select(start: TNode, td: int) -> tuple[str, TNode] | None:
        """One descent from ``start`` for target depth ``td``: ("node", new child), ("move", node) or None."""
        n = start
        for _ in range(64):
            n.visits += 1
            if n.kind == "root":
                cands = [
                    c
                    for c in n.children
                    if not c.exhausted
                    and (c.status == "ready" if c.kind != "front" else bool(c.children or any(c.pending.values())))
                ]
                # every preparation circuit gets at least ``PREP_SHARE`` of an equal share of the root's visits
                roots = [c for c in cands if c.kind == "prep"]
                low = [c for c in roots if c.visits < PREP_SHARE * n.visits / len(roots)] if roots else []
                if low:
                    n = min(low, key=lambda c: c.visits)
                    continue
            else:
                lim = math.ceil(PW * n.visits**0.5)
                if n.status == "ready" and len(n.children) < lim:
                    w = widen(n)
                    if w is not None:
                        return w
                cands = [c for c in n.children if not c.exhausted and c.status == "ready"]
                if not cands and n.status == "ready":
                    w = widen(n)
                    if w is not None:
                        return w
            if not cands:
                if (
                    n.kind not in {"root", "front"}
                    and n.status == "ready"
                    and not open_todo(n)
                    and not n.moving
                    and not any(n.pending.values())
                    and all(c.exhausted or c.status == "dead" for c in n.children)
                ):
                    n.exhausted = True
                return None
            n = max(cands, key=lambda c: ucb(n, c, td))
        return None

    def credit(n: TNode, found: list, name: str, out: dict, *, timed_out: bool = False) -> int:
        """Record the FT designs ``found`` (label, design, base) of a job on ``n`` as one visit; return new front points.

        ``timed_out``: the parent's mean value stands in for the visit (a zero would bury the node); re-scoring drops it.
        """
        recs = [record(d, lab, n, base) for lab, d, base in found]
        new = sum(x for x, _, _ in recs)
        if timed_out:
            p = n.parents[0]
            backup(n, [v / max(p.rn, 1) for v in p.rsum], None)
        else:
            backup(n, vmax([r for _, r, _ in recs]), tuple(p for _, _, p in recs if p))
        stat(name, out.get("seconds", 0.0), new)
        return new

    def node_ready(n: TNode, out: dict) -> None:
        comps = out.get("completions", [])
        timed_out = not comps and bool(n.parents) and any("JobTimeout" in e for e in out.get("errors", []))
        new = credit(n, comps, n.family or n.kind, out, timed_out=timed_out)
        if n.via_restart:
            stat("restart", out.get("seconds", 0.0), new)
        for f in n.route:  # the moves that led here share the credit
            if f != n.family:
                stats.setdefault(f, [0, 0.0, 0])[2] += new
        n.open_scopes = out.get("open", [])
        if out.get("partials"):
            n.pending.setdefault(("repair", -1), []).extend(out["partials"])
        assert n.design is not None, "a node job runs on a node with a design"
        n.todo = (
            moves.plan(n.design, n.kind == "ft" or out.get("ok", False), prof)
            if (n.kind == "ft" or not out.get("ok"))
            else []
        )
        n.status = "ready" if (comps or n.todo or any(n.pending.values())) else "dead"
        pts = sorted({_point(d) for _, d, _ in comps})
        n.points = pts
        # two-tier evaluation: a quick rollout near the front earns the node the full one
        if (
            prof.quick_budget is not None
            and n.key not in deep_done
            and (new or any(near(o, dp, prof.near_deep) for o, dp in pts))
        ):
            deep_done.add(n.key)
            queue.append(("deep", (n.design, ctx, prof, True), n))

    def deep_ready(n: TNode, out: dict) -> None:
        """Record the full rollout of a node as a further visit; its partial designs become children."""
        comps = out.get("completions", [])
        credit(n, comps, "deep", out)
        if out.get("partials"):
            n.pending.setdefault(("repair", -1), []).extend(out["partials"])
        n.points = sorted(set(n.points) | {_point(d) for _, d, _ in comps})

    def refresh_restarts() -> None:
        """Link the restart group to the best partial designs.

        Per front depth T the ``RESTART_K`` partial designs with the fewest completion ops at depth <= T; the others
        are unlinked.
        """
        cands = [
            n for n in nodes.values() if n.kind == "partial" and n.status == "ready" and not n.exhausted and n.points
        ]
        pick: list = []
        for td in front.targets(TARGET_MARGIN):
            sc = sorted(
                (
                    (min(o for o, d in n.points if d <= td), n.visits, i)
                    for i, n in enumerate(cands)
                    if any(d <= td for _, d in n.points)
                )
            )
            for _, _, i in [s for s in sc if cands[s[2]] not in pick][:RESTART_K]:
                pick.append(cands[i])
        for c in list(rgroup.children):
            if c not in pick:
                rgroup.children.remove(c)
                c.parents.remove(rgroup)
        for c in pick:
            link(rgroup, c)

    def result() -> list[dict]:
        pts = set(front.points())
        sols = sorted(solutions.values(), key=lambda s: (s["metrics"]["ops"], s["metrics"]["depth"]))
        return [s for s in sols if (s["metrics"]["ops"], s["metrics"]["depth"]) in pts]

    # the preparation circuits: a tree node each, one front for all
    parts.clear()
    parts_version = [0]  # bumped when a job returns new parts
    snap: list = [[], 0]  # the parts sent with each job and the version they belong to
    pool = process_pool(workers)
    for i, st in enumerate(start if isinstance(start, list) else [start]):
        k = st.key()
        if k in nodes:
            continue
        n = TNode(k, st, "prep", family="prep", label=st.root.origin or f"prep{i}", status="running", prep_key=k)
        nodes[k] = n
        link(root, n)
        queue.append(("node", (st, ctx, prof), n))
        if prof.precompute_parts:
            queue.append(("seed", (st, ctx, prof), n))
    link(root, fgroup)
    part_states: set = set()  # ancilla states with a part job (one per state over all preps)
    deep_done: set = set()  # nodes with a deep job (one each)

    def submit_next() -> None:
        while len(running) < workers:
            n_pol = sum(k_ == "polish" for k_, _, _ in running.values())
            # polish: a quarter of the workers, half while a backlog waits
            n_wait = sum(q[0] == "polish" for q in queue)
            cap = max(1, workers // 2 if n_wait > workers // 4 else workers // 4)
            # seed and part jobs: at most ``PART_SHARE`` of the workers, the tree search keeps the rest
            n_part = sum(k_ in {"seed", "part"} for k_, _, _ in running.values())
            cap_part = max(1, round(PART_SHARE * workers))
            n_deep = sum(k_ == "deep" for k_, _, _ in running.values())
            cap_deep = max(1, round(DEEP_SHARE * workers))
            j = next(
                (
                    i
                    for i, q in enumerate(queue)
                    if (q[0] != "polish" or n_pol < cap)
                    and (q[0] not in {"seed", "part"} or n_part < cap_part)
                    and (q[0] != "deep" or n_deep < cap_deep)
                ),
                None,
            )
            if j is not None:
                kind, args, n = queue.pop(j)
            else:
                busy = sum(1 for k_, n_, _ in running.values() if k_ == "polish" or n_.side)
                busy_r = sum(1 for f_ in running if f_ in from_restart)
                s, via_r = None, False
                tg = front.targets(TARGET_MARGIN)
                counter["t"] += 1
                td = tg[counter["t"] % len(tg)]
                # a few descents: one can end at a node whose solvers are all over their share
                for _ in range(8):
                    if busy < max(1, round(FRONT_SHARE * workers)) and (
                        fgroup.children or any(fgroup.pending.values())
                    ):
                        s = select(fgroup, td)
                    if s is None and rgroup.children and busy_r < round(RESTART_SHARE * workers):
                        s = select(rgroup, td)
                        via_r = s is not None
                    if s is None:
                        s = select(root, td)
                    if s is not None:
                        break
                if s is None:
                    return
                kind, n = s
                if kind == "node":
                    n.status = "running"
                    n.via_restart = via_r
                    args = (n.design, ctx, prof)
                else:  # the move job of the family ``widen`` picked
                    i = next((i for i, a in enumerate(n.todo) if tuple(a) == n.next_move), 0)
                    fam, idx = n.todo.pop(i)
                    n.moving = True
                    args = (n.design, n.open_scopes, fam, idx, ctx, prof)
            fn = {
                "node": node_job,
                "move": move_job,
                "polish": polish_job,
                "seed": seed_job,
                "part": part_job,
                "deep": node_job,
            }[kind]
            if snap[1] != parts_version[0]:
                snap[:] = [parts.snapshot(), parts_version[0]]
            try:
                fut = pool.submit(_job, fn, args, snap[0])
            except BrokenProcessPool:  # the job goes to the new pool
                queue.insert(0, (kind, args, n))
                raise
            running[fut] = (kind, n, time.time())
            if j is None and via_r:
                from_restart.add(fut)
            if kind == "move":
                move_arm[fut] = (args[2], args[3])  # ty: ignore[index-out-of-bounds]  (a move job's args: 6 items)

    def restart(old: ProcessPoolExecutor) -> ProcessPoolExecutor:
        """A worker died: end the old pool, mark its running jobs as lost and start a new pool."""
        kill_pool(old)
        for k_, n_, _ in running.values():
            lost(k_, n_)
        running.clear()
        move_arm.clear()
        from_restart.clear()
        return process_pool(workers)

    def lost(kind: str, n: TNode) -> None:
        """Mark the node of a job that died with its worker, so the search does not wait for it.

        A lost deep job leaves its node as it is: its quick evaluation stands.
        """
        if kind == "node":
            n.status = "dead"
        elif kind == "move":
            n.moving = False

    last_restart = last_rescore = time.time()
    rescored: list = []  # the front of the last re-scoring
    rescore_cost = 0.0  # seconds of the last re-scoring
    try:  # ruff: ignore[too-many-statements-in-try-clause]  (Ctrl-C and the pool clean-up cover the whole loop)
        while time.time() < deadline:
            if time.time() - last_restart > RESTART_EVERY:
                refresh_restarts()
                last_restart = time.time()
            # re-scoring takes at most 5 % of the main loop's time
            if time.time() - last_rescore > max(RESCORE_EVERY, 20 * rescore_cost) and front.points() != rescored:
                t1 = time.time()
                rescore([root, fgroup, rgroup, *nodes.values()], rewards)
                rescored, last_rescore = front.points(), time.time()
                rescore_cost = last_rescore - t1
            try:
                submit_next()
            except BrokenProcessPool:  # a worker died since the last wait
                pool = restart(pool)
                continue
            if not running:
                if not queue and all(c.exhausted or c.status == "dead" for c in root.children if c.kind == "prep"):
                    break
                time.sleep(1)
                continue
            done, _ = wait(
                list(running), timeout=max(1.0, min(30.0, deadline - time.time())), return_when=FIRST_COMPLETED
            )
            broken = False
            for f in done:
                kind, n, t0 = running.pop(f)
                move_arm.pop(f, None)
                from_restart.discard(f)
                out: dict[str, Any]
                try:
                    out = f.result()
                    if out.get("new_parts"):
                        parts.merge(out.pop("new_parts"))
                        parts_version[0] += 1
                except BrokenProcessPool:
                    broken = True
                    lost(kind, n)
                    continue
                except Exception as e:  # ruff: ignore[blind-except]
                    out = {"errors": [repr(e)], "seconds": time.time() - t0}
                if kind == "node":
                    node_ready(n, out)
                elif kind == "deep":
                    deep_ready(n, out)
                elif kind == "seed":
                    for _, d in out.get("seeds", []):
                        k = parts.state_key(d.root, d.t)
                        if k not in part_states:
                            part_states.add(k)
                            queue.append(("part", (d, ctx, prof), n))
                    stat("seed", out.get("seconds", 0.0))
                elif kind == "part":
                    stat("part", out.get("seconds", 0.0))
                elif kind == "move":
                    n.moving = False
                    ch = out.get("children", [])
                    if ch:
                        n.pending.setdefault((out["family"], out["idx"]), []).extend(ch)
                    stat(f"move:{out.get('family', '?')}", out.get("seconds", 0.0))
                    stat(f"solver:{out.get('family', '?')}/{out.get('idx', '?')}", out.get("seconds", 0.0))
                    if any("MemoryError" in e for e in out.get("errors", [])) and "idx" in out:
                        oom.add((n.prep_key, out["family"], out["idx"]))
                else:
                    credit(n, [("polish:full", d, None) for d in out.get("designs", [])], "polish", out)
            if broken:  # a worker died: the other running jobs are lost, finished results above were kept
                pool = restart(pool)
    except KeyboardInterrupt:
        logger.info("search interrupted: returning the front found so far")
    except Exception:
        logger.exception("search failed: returning the front found so far")
    finally:
        kill_pool(pool)
    return result()
