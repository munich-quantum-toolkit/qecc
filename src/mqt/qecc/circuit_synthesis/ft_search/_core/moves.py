# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""The search moves: each move family is a fixed set of solver configurations.

Partial designs get "criger" (checks and flags by Criger's ILP promotion, plus the subsets of each bundle) and
"ancilla" (a transversal block on a cover of what is still dangerous, or a cancellation-guided ancilla prep). Partial
and FT designs get "merge" (Huang & Brown's two merges, ``solvers.merge``) and "edit" (remove one or two elements,
place a check elsewhere, coupling order, move a CNOT of the data preparation circuit). Moves are generated lazily,
one solver configuration per job, when the search widens a node.
"""

from __future__ import annotations

import copy
import dataclasses
import operator
from typing import TYPE_CHECKING, TypeVar

from .circuit import metrics
from .config import K_CRIGER, MAX_MOVES, MAX_REMOVES
from .evaluation import proposals
from .parts import with_known_parts
from .solvers import (
    BlockSolver,
    CrigerSolver,
    GuideSolver,
    MergeSolver,
    NeighbourhoodSolver,
    OrderSolver,
    PrepMoveSolver,
)
from .verify import check_noiseless

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .config import (
        Context,
        Profile,
    )
    from .design import Design, Node
    from .evaluation import Solver

_T = TypeVar("_T")


def families(prof: Profile) -> dict[str, list[tuple[Solver, str]]]:
    """Family -> [(solver, where)]; where: "open" = every open scope, "root" = the data scope only."""
    s = prof.samples
    return {
        # step: one promotion round per move; complete: all checks of a scope in one ILP solve
        "criger": [
            (CrigerSolver("step", K_CRIGER, gauge_w=8, samples=s), "open"),
            (CrigerSolver("complete", gauge_w=8, samples=s), "open"),
        ],
        "ancilla": [
            (BlockSolver("Z", "weight", "both", samples=s), "open"),
            (BlockSolver("X", "weight", "both", samples=s), "open"),
            (BlockSolver("Z", "full"), "open"),
            (BlockSolver("X", "full"), "open"),
            (GuideSolver(), "open"),
        ],
        "merge": [(MergeSolver(op), "open") for op in ("macro", "group")],
        "edit": [
            (NeighbourhoodSolver(), "root"),
            (PlaceSolver(), "open"),
            (OrderSolver(), "open"),
            (PrepMoveSolver(), "root"),
        ],
    }


REMOVE = ("edit", 0)


def removes(design: Design) -> int:
    """Number of removal moves in the design's history."""
    return sum(":remove[" in h for h in design.full_history())


BLOCK_MAKERS = ("block",)  # moves that create ancilla scopes (known protections apply): block, blockmerge


FT_FAMILIES = ("merge", "edit")  # the families FT designs get


def plan(design: Design, ft: bool, prof: Profile) -> list[tuple[str, int]]:
    """The move jobs of a node as (family, solver index), round-robin over the families; FT designs: merge and edit."""
    if len(design.history) >= MAX_MOVES:
        return []
    fams = {k: v for k, v in families(prof).items() if not ft or k in FT_FAMILIES}
    queues = {
        k: [i for i in range(len(v)) if (k, i) != REMOVE or removes(design) < MAX_REMOVES] for k, v in fams.items()
    }
    out = []
    while any(queues.values()):
        out.extend((k, queues[k].pop(0)) for k in queues if queues[k])
    return out


def split(design: Design, path: tuple, props: list[tuple[str, Design]]) -> list[tuple[str, Design]]:
    """Subsets of the Criger bundles: each new check alone (also bare, at its best place), each bundle without one."""
    old = design.node(path).gadgets
    out = []
    for lab, d in props:
        gs = d.node(path).gadgets
        new = list(range(len(old), len(gs)))  # the engines append new checks; old gadgets (maybe with new flags) stay
        fids = [f[2] for g in gs for f in g.flags]
        if len(new) < 2 or len(fids) != len(set(fids)):  # nothing to split / shared flags tie checks together
            continue
        subsets = [(f"/{j}", {j}) for j in new] + ([(f"/-{j}", set(new) - {j}) for j in new] if len(new) >= 3 else [])
        for sfx, keep in subsets:
            nd = copy.deepcopy(d.node(path))
            nd.gadgets = [g for i, g in enumerate(gs) if i not in new or i in keep]
            out.append((lab + sfx, design.with_node(path, nd, lab + sfx)))
            if len(keep) == 1:  # the single check also bare at its best place in the preparation circuit
                out.extend(
                    (lab + sfx + "p", design.with_node(path, x, lab + sfx + "p"))
                    for x in place(design, path, nd, next(i for i, g in enumerate(nd.gadgets) if g not in old))
                )
    return out


def place(design: Design, path: tuple, nd: Node, i: int, *, bare: bool = True, n: int = 1) -> list:
    """Copies of scope ``nd`` with gadget ``i`` at one position in the preparation circuit.

    The ``n`` valid positions with the lowest distinct (depth, ops); ``bare``: without flags.
    """
    best: dict = {}  # (depth, ops) -> node
    for p in range(len(nd.base.ops) + 1):
        x = copy.deepcopy(nd)
        g = x.gadgets[i]
        x.gadgets[i] = dataclasses.replace(
            g, pos=p, slots=None, ticks=None, **({"flags": [], "drive": ()} if bare else {})
        )
        if x.gadgets[i] == g:
            continue
        circ = design.with_node(path, x).circuit()
        m = metrics(circ)
        c = (m["depth"], m["ops"])
        if c in best or (len(best) >= n and c >= max(best)):
            continue
        try:
            check_noiseless(circ, design.root.state)
        except (AssertionError, ValueError):  # not a stabilizer there (stim: non-deterministic detectors)
            continue
        best[c] = x
        if len(best) > n:
            del best[max(best)]
    return [best[c] for c in sorted(best)]


class PlaceSolver:
    """Move one bare check of the scope (flags kept) to its two best other positions by (depth, ops).

    The ILP places checks for fault coverage and cannot see depth.
    """

    ctx_deps = ()
    name = "place"

    def propose(  # ruff: ignore[no-self-use]  (solver interface)
        self,
        design: Design,
        path: tuple,
        ctx: Context,  # ruff: ignore[unused-method-argument]  (solver interface)
    ) -> Iterator[tuple[str, Node]]:
        """Yield each bare check at its best other positions."""
        nd = design.node(path)
        if nd.opaque:
            return
        for i, g in enumerate(nd.gadgets):
            if g.late:
                continue
            for k, x in enumerate(place(design, path, nd, i, bare=False, n=2)):
                yield f"place{i}#{k}", x


def by_cover(items: list[tuple[float, str, _T]]) -> list[tuple[float, str, _T]]:
    """Order cost-sorted block children round-robin over their covers (the label up to the block's shape).

    Sorted by cost alone, the cheap covers of the lowest fault order would take all first widening slots.
    """
    groups: dict[str, list] = {}
    for x in items:
        groups.setdefault(x[1].split("]")[0], []).append(x)
    rows = list(groups.values())
    return [g[k] for k in range(max(map(len, rows), default=0)) for g in rows if k < len(g)]


def run(
    design: Design, open_scopes: list, fam: str, idx: int, ctx: Context, prof: Profile
) -> tuple[list[tuple[str, Design]], list[str]]:
    """Run one move job: the children of ``design`` from one solver configuration, cheapest first, and errors.

    Criger bundles and their subsets (``split``) come interleaved, block children by ``by_cover``.
    """
    sv, where = families(prof)[fam][idx]
    scopes = [()] if where == "root" or not open_scopes else open_scopes
    bundles, subsets, errors, seen = [], [], [], set()
    for path in scopes:
        try:
            props = proposals(sv, design, path, ctx)
            if sv.name.startswith(BLOCK_MAKERS):
                props += with_known_parts(design, props, ctx)
            more = (
                split(design, path, props) if isinstance(sv, CrigerSolver) and sv.mode in {"step", "complete"} else []
            )
        except Exception as e:  # ruff: ignore[blind-except]  (one failing scope must not stop the others)
            errors.append(f"{sv.name}@{path}: {type(e).__name__}: {str(e)[:200]}")
            continue
        for dst, items in ((bundles, props), (subsets, more)):
            for lab, d in items:
                k = d.key()
                if k not in seen:
                    seen.add(k)
                    dst.append((ctx.cost(metrics(d.circuit())), lab, d))
    bundles.sort(key=operator.itemgetter(0))
    subsets.sort(key=operator.itemgetter(0))
    if sv.name.startswith(BLOCK_MAKERS):
        bundles = by_cover(bundles)
    mixed = (
        [x for pair in zip(bundles, subsets, strict=False) for x in pair]
        + bundles[len(subsets) :]
        + subsets[len(bundles) :]
    )
    return [(lab, d) for _, lab, d in mixed], errors
