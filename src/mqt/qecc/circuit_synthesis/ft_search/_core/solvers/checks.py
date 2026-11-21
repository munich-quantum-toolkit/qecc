# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Add checks: bare checks and flags in the preparation circuit, or late checks after the blocks.

Two engines: ``gauge.promote`` (ILP rounds) and ``greedy_ft`` (efficiency greedy).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from ..config import profile
from ..design import Design, Node, scope_name
from ..gauge import promote
from ..greedy_ft import greedy_ft
from .common import JITTER, _blocks_of

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config import Context


class CrigerSolver:
    """Criger-style bare checks and flags on the scope.

    ``mode`` "complete": ``gauge.promote`` until the scope is FT (Criger's incremental schedule); "step": one promotion
    round, the ``k`` best distinct ILP solutions each a proposal; "greedy": ``k`` steps of ``greedy_ft``. ``late``:
    new checks only after all blocks, where they also catch what unprotected ancillas send back. ``gauge_w`` > 0:
    gauge-operator candidates up to that weight (e.g. flags at the origin of a cat-state prep). ``samples`` > 1: the
    promotion again with perturbed ILP costs (``JITTER``), each distinct result a further proposal.
    """

    def __init__(
        self,
        mode: str = "complete",
        k: int = 1,
        *,
        gauge_w: int = 0,
        samples: int = 1,
        late: bool = False,
    ) -> None:
        """Configure the check engine."""
        assert mode in {"complete", "step", "greedy"}, mode
        self.late = late
        self.mode, self.k, self.gauge_w = mode, k, gauge_w
        self.samples = samples if mode != "greedy" else 1
        r = f"r{self.samples}" if self.samples > 1 else ""
        # complete mode: early output checks with gauge candidates (not for late checks: they sit after the blocks)
        self.asap = not late and gauge_w > 0
        self.ctx_deps = ("lam",) if mode == "greedy" else ("ilp_time",)
        if late:
            self.name = "latecheck" if mode == "complete" else "lategreedy"
        elif mode == "complete":
            self.name = "criger" + (f"g{gauge_w}" if gauge_w else "") + r
        elif mode == "greedy":
            self.name = f"greedy{k}"
        else:
            self.name = f"crigerstep{k}g{gauge_w}i{'' if self.asap else 'n'}{r}"

    def cache_extra(self, design: Design, path: tuple[int, ...]) -> tuple:
        """Extra memo key: in step mode, the number of steps this scope has had."""
        return (_steps_done(design, path),) if self.mode == "step" else ()

    def propose(self, design: Design, path: tuple[int, ...], ctx: Context) -> Iterator[tuple[str, Node]]:
        """Yield the scope with the added checks."""
        node = design.node(path)
        if node.opaque:
            return
        if self.late and not node.blocks:
            return
        blocks = _blocks_of(node) or None
        # the engines start from every gadget of the scope (one flag id space); ``late`` only says where new checks go
        prep = node.gadgets
        end = len(node.base.ops)
        common: dict[str, Any] = {
            "blocks": blocks,
            "gadgets0": prep,
            "layer": node.layered,
            "late": self.late,
        }
        variants = []
        seen: set[str] = set()

        def add(tag: str, gadgets: list) -> None:
            key = repr([(g.kind, g.pos, g.support, g.flags, g.slots, g.ticks, g.drive) for g in gadgets])
            if key not in seen:
                seen.add(key)
                variants.append((tag, gadgets))

        for s in range(self.samples):
            jit: dict[str, Any] = {"cost_jitter": JITTER if s else 0.0, "seed": s}
            sfx = f"~{s}" if s else ""
            if self.mode == "complete":
                _, gadgets, plog = promote(
                    node.base,
                    node.target,
                    design.t,
                    incremental=True,
                    time_limit=ctx.ilp_time,
                    gauge_w=self.gauge_w,
                    asap_end=self.asap,
                    rounds=10 if design.t <= 2 else 25,
                    positions=[end] if self.late else None,
                    **jit,
                    **common,
                )
                if plog.ok:
                    add("inc" + sfx, gadgets)
            elif self.mode == "step":
                _, _, plog = promote(
                    node.base,
                    node.target,
                    design.t,
                    incremental=True,
                    gauge_w=self.gauge_w,
                    asap_end=self.asap,
                    alternatives=self.k,
                    stage_round0=_steps_done(design, path),
                    time_limit=min(ctx.ilp_time, 120.0),
                    **jit,
                    **common,
                )
                for i, gadgets in enumerate(plog.alternatives):
                    add(f"#{i}{sfx}", gadgets)
        if self.mode == "greedy":
            prof = profile(design.t)
            _, gadgets, _ = greedy_ft(
                node.base,
                node.target,
                design.t,
                lam=ctx.lam,
                max_steps=self.k,
                top_k=prof.greedy_top_k,
                by_order=prof.greedy_by_order,
                positions=[end] if self.late else None,
                **common,
            )
            variants.append(("", gadgets))
        n_g, n_f = len(prep), sum(len(g.flags) for g in prep)
        for tag, gadgets in variants:
            dg, df = len(gadgets) - n_g, sum(len(g.flags) for g in gadgets) - n_f
            if self.mode != "step" and dg == 0 and df == 0:
                continue
            nd = copy.copy(node)
            nd.gadgets = gadgets
            yield (f"{self.name}[+{dg}g,{df}f]{tag}" if self.mode == "step" else f"{self.name}{tag}[+{dg}]"), nd


def _steps_done(design: Design, path: tuple[int, ...]) -> int:
    return sum(h.startswith(f"{scope_name(path)}:crigerstep") for h in design.full_history())
