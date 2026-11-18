# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Add block: measure a set of same-type checks through one transversal block."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from ..blocks import BlockSpec
from ..config import profile
from ..cover import robust_cover_from_failures
from ..design import Design, Node, ancilla_states, local_check, make_node
from ..guided import guided_block_preps
from ..verify import gf2_row_basis, state_leader
from .common import CHOICE_CTX_DEPS, JITTER, block_slack, circuit_choices

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..config import Context


class BlockSolver:
    """One transversal block on the scope (Huang & Brown, arXiv:2012.15403); its ancilla becomes a child scope.

    ``checks`` "full": all stabilizers of the type (Steane copy); "weight": a robust cover of the scope's remaining
    failures. ``subset`` "all", "lowest" (lowest fault order only) or "both". One proposal per cover and ancilla circuit.
    """

    ctx_deps = (*CHOICE_CTX_DEPS, "cover_time")

    def __init__(self, kind: str, checks: str = "weight", subset: str = "all", samples: int = 1) -> None:
        """``samples`` > 1: also covers with randomly perturbed ILP costs (``JITTER``), each a further proposal."""
        assert checks in {"full", "weight"}, checks
        self.kind, self.checks, self.subset, self.samples = kind, checks, subset, samples
        self.name = (
            f"block{kind}:{checks}"
            + ("" if subset == "all" else f":{subset}")
            + (f":r{samples}" if samples > 1 else "")
        )

    def _covers(self, node: Node, basis: np.ndarray, t: int, ctx: Context) -> list[tuple[str, np.ndarray]]:
        rep = local_check(node, t)
        if self.checks == "full":  # a whole copy only where this type still fails
            return [("", basis)] if len(rep.failing_x if self.kind == "Z" else rep.failing_z) else []
        synd = rep.failing_x if self.kind == "Z" else rep.failing_z
        lw = state_leader(node.target, "X" if self.kind == "Z" else "Z", t)
        fails = sorted({(int(x), len(r)) for x, r in zip(synd, rep.failing_reps, strict=True) if lw[int(x)] > len(r)})
        if not fails:
            return []
        out, seen = [], set()
        for sub in ("all", "lowest") if self.subset == "both" else (self.subset,):
            fs = fails
            if sub == "lowest":
                s_min = min(sv for _, sv in fails)
                fs = [f for f in fails if f[1] == s_min]
            for s in range(min(self.samples, profile(t).samples)):  # the run's limit
                try:
                    elems = robust_cover_from_failures(
                        basis,
                        fs,
                        t,
                        time_limit=ctx.cover_time,
                        cost_jitter=JITTER if s else 0.0,
                        seed=s,
                        lw=lw,
                    )
                except RuntimeError:  # no cover in time: a perturbed sample keeps the covers found so far
                    if not s:
                        raise
                    continue
                if not elems or tuple(sorted(elems)) in seen:
                    continue
                seen.add(tuple(sorted(elems)))
                bits = np.array([[(g >> i) & 1 for i in range(basis.shape[0])] for g in elems], dtype=np.uint8)
                out.append((("" if sub == "all" else ":lowest") + (f"~{s}" if s else ""), (bits @ basis) % 2))
        return out

    def propose(self, design: Design, path: tuple[int, ...], ctx: Context) -> Iterator[tuple[str, Node]]:
        """Yield one block per cover and ancilla prep circuit."""
        node = design.node(path)
        basis = gf2_row_basis(node.target.sz if self.kind == "Z" else node.target.sx)
        if basis.shape[0] == 0:
            return
        t = design.t
        for sub, checks in self._covers(node, basis, t, ctx):
            spec = BlockSpec(self.kind, checks)
            full, target = ancilla_states(spec)
            h = spec.ancilla_generator()
            for tag, c in circuit_choices(h, target, t, ctx, slack=block_slack(node, spec)):
                nd = copy.deepcopy(node)
                nd.add_block(spec, make_node(f"{node.name}.{self.kind}{len(node.blocks)}", c, full, target, origin=tag))
                yield f"block{self.kind}:{self.checks}{sub}[{checks.shape[0]}x{len(spec.support)}]:{tag}", nd


class GuideSolver:
    """Replace a plain end-block ancilla prep by up to two cancellation-guided ones (``guided``).

    A guided prep avoids CNOTs whose ancilla faults would mask a dangerous data error; the exact check still decides
    fault tolerance. The local checks judge only the error type an ancilla sends back, so no other move targets
    masking.
    """

    ctx_deps = ()
    name = "guide"

    def propose(  # ruff: ignore[no-self-use]  (solver interface)
        self,
        design: Design,
        path: tuple[int, ...],
        ctx: Context,  # ruff: ignore[unused-method-argument]  (solver interface)
    ) -> Iterator[tuple[str, Node]]:
        """Yield the scope with one block ancilla replaced by a guided prep."""
        node = design.node(path)
        if node.opaque or not node.blocks:
            return
        for i, (spec, ch) in enumerate(node.blocks):
            if ch.opaque or ch.gadgets or ch.blocks or spec.pos is not None:
                continue
            full, target = ancilla_states(spec)
            for j, c in enumerate(guided_block_preps(node, i, design.t, k=2)):
                nd = copy.deepcopy(node)
                nd.blocks[i] = (spec, make_node(ch.name, c, full, target, origin="guided"))
                yield f"guide[{spec.kind}{i}#{j}]", nd
