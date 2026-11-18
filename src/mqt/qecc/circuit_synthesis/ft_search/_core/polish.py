# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Two-stage polish of FT designs.

* structured: redundancy removal, shared flags, flag takeover, depth-aware coupling order
  (``solvers.PolishSolver.polish``); the result stays a structured design.
* circuit: rescheduling, ancilla removal and Bell reduction on the compiled circuit
  (``PolishSolver.circuit_variants``); the results are opaque designs.

Structure comes first, since the circuit stage makes a design opaque.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import math
import time
from typing import TYPE_CHECKING

from .circuit import ops_depth
from .design import Design, Node
from .evaluation import evaluate_cached
from .front import pareto
from .solvers import DepthOrderSolver, PolishSolver, PruneSolver, SharedFlagSolver, TakeoverSolver

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .config import Context


def _passes() -> list:
    return [PruneSolver(), SharedFlagSolver(), TakeoverSolver(), DepthOrderSolver()]


def structured(design: Design, ctx: Context, stop: float = math.inf) -> Design:
    """Structured stage (no-op on opaque designs)."""
    if design.root.opaque:
        return design
    return PolishSolver.polish(design, ctx, _passes(), stop=stop)


def circuit(design: Design, stop: float = math.inf) -> list[Design]:
    """Circuit stage: the FT circuit variants of the compiled design (opaque designs), without the design itself."""
    cur = str(design.circuit().to_stim())
    out = []
    for c in PolishSolver.circuit_variants(design.circuit(), design.root.target, design.t, stop=stop):
        if str(c.to_stim()) == cur:
            continue
        node = Node(
            design.root.name,
            c,
            design.root.state,
            design.root.target,
            opaque=True,
            origin=design.root.origin + "+polish",
        )
        out.append(Design(node, design.t, [*design.history, "root:polish:circuit"], list(design.origin)))
    return out


def front_of(designs: list[Design]) -> list[Design]:
    """The designs on the (ops, depth) front of the list (one per point)."""
    return pareto(designs, lambda d: ops_depth(d.circuit()))


def quick(design: Design, ctx: Context, *, with_circuit: bool, budget: float) -> tuple[Design, list[Design]]:
    """Polish inside ``complete``: the structured stage, and with ``with_circuit`` the front of the circuit stage."""
    stop = time.time() + budget
    s = structured(design, ctx, stop=time.time() + budget / 2)
    if not with_circuit or time.time() > stop:
        return s, []
    return s, front_of(circuit(s, stop=stop))


def coupling_variants(design: Design) -> Iterator[Design]:
    """Unchecked copies of ``design`` with the coupling order of one bare check changed.

    All orders up to weight 4, else the rotations of the order and of its reverse; deepest scopes first, round-robin
    over the checks. Which order suits the circuit stage only shows after it.
    """
    per_check = []
    for path, nd in sorted(design.nodes(), key=lambda x: -len(x[0])):
        if nd.opaque:
            continue
        for i, g in enumerate(nd.gadgets):
            if g.flags or g.drive or len(g.support) < 2:
                continue
            sup = list(g.support)
            if len(sup) <= 4:
                orders = [list(p) for p in itertools.permutations(sup)]
            else:
                orders = [r[k:] + r[:k] for r in (sup, sup[::-1]) for k in range(len(sup))]
            slots = [g.slots, None] if g.slots is not None and not g.late else [None]
            out = []
            for o in orders:
                for sl in slots:
                    v = dataclasses.replace(g, support=o, slots=sl, ticks=None)
                    if v != g:
                        out.append((path, i, v))
            per_check.append(out)
    for k in range(max((len(x) for x in per_check), default=0)):
        for out in per_check:
            if k < len(out):
                path, i, v = out[k]
                nd = copy.copy(design.node(path))
                nd.gadgets = [v if j == i else x for j, x in enumerate(nd.gadgets)]
                yield design.with_node(path, nd, f"{'.'.join(map(str, path)) or 'root'}:polish:couple{i}")


def full(design: Design, ctx: Context, budget: float) -> list[Design]:
    """Full polish of a new front design: the front of everything found, the input excluded.

    The structured stage, then the circuit stage on its result, on the design itself and on its FT coupling-order
    variants.
    """
    stop = time.time() + budget
    s = structured(design, ctx, stop=time.time() + budget / 3)
    cands = [s] if s is not design else []
    for base in [s, design] if s is not design else [design]:
        if time.time() > stop:
            break
        cands += circuit(base, stop=stop)
    seen = {design.key(), s.key()}
    if not s.root.opaque:
        for v in coupling_variants(s):
            if time.time() > stop:
                break
            k = v.key()
            if k in seen:
                continue
            seen.add(k)
            try:
                ok = evaluate_cached(v).ok
            except Exception:  # ruff: ignore[blind-except]
                ok = False
            if ok:
                cands += [v, *circuit(v, stop=stop)]
    return front_of(cands) if cands else []
