# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helpers shared by the solvers: ancilla circuit choice, scope slack, and the verified circuits of MQT QECC."""

from __future__ import annotations

import copy
import dataclasses
import operator
from typing import TYPE_CHECKING

import stim

from ....circuit_utils import qiskit_to_stim_circuit
from ....circuits import CNOTCircuit
from ....state_prep import FaultyStatePrepCircuit, heuristic_verification_circuit
from ...synth import candidates
from ..circuit import Circuit, metrics, ready_times
from ..design import Design, Node, compile_node
from ..verify import check_ft, weighted_failures

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy as np
    from qiskit import QuantumCircuit

    from ..blocks import BlockSpec
    from ..config import Context
    from ..verify import CSSState


def circuit_choices(
    h: np.ndarray, target: CSSState, t: int, ctx: Context, slack: int | None = None
) -> list[tuple[str, Circuit]]:
    """Preparation circuits for the state with X-type generator ``h``, as (label, circuit).

    Per synthesis mode (mode, k): with ``ctx.pool`` > 1, pool * k candidates are synthesized and k chosen: first the
    one with the fewest dangerous fault combinations under ``target``, then, among the better half, the ones whose
    dangerous combinations overlap least with those chosen. ``slack`` (the ancilla's depth budget): the candidate
    with the fewest CNOTs plus λ per layer beyond the slack comes first.
    """
    out, seen = [], set()
    everything = []  # (mode, i, circuit): every synthesized candidate, for the slack choice
    for mode, k in ctx.synth:
        try:
            pool = candidates(h, mode, k * max(ctx.pool, 1), synthesizers=ctx.synthesizers)
        except Exception:  # ruff: ignore[blind-except, try-except-continue]  (external tool missing / failed)
            continue
        everything += [(mode, i, c) for i, c in enumerate(pool)]
        pool = [(i, c) for i, c in enumerate(pool) if str(c.to_stim()) not in seen]
        if ctx.pool > 1 and len(pool) > k:
            info = []
            for i, c in pool:
                rep = check_ft(c, target, t, collect=True)
                dang = {
                    (int(x), int(z), len(r))
                    for x, z, r in zip(rep.failing_x, rep.failing_z, rep.failing_reps, strict=True)
                }
                w = weighted_failures(rep.raw_failures, t)
                info.append((i, c, dang, w, sum(op.name == "CX" for op in c.ops)))
            info.sort(key=operator.itemgetter(3, 4))
            chosen, covered = [info[0]], set(info[0][2])
            rest = info[1 : max(k, (len(info) + 1) // 2)]  # diversity only among the better half
            while len(chosen) < k and rest:
                rest.sort(key=lambda x: (len(x[2] & covered) / max(len(x[2]), 1), x[3], x[4]))
                nxt = rest.pop(0)
                chosen.append(nxt)
                covered |= nxt[2]
            pool = [(i, c) for i, c, *_ in chosen]
        else:
            pool = pool[:k]
        for i, c in pool:
            seen.add(str(c.to_stim()))
            out.append((f"{mode}{i}", c))
    if slack is not None:
        for mode in ("greedy", "random"):
            everything += [(mode, i, c) for i, c in enumerate(candidates(h, mode, 4))]
        if everything:

            def fit_cost(c: Circuit) -> tuple:
                m = metrics(c)
                return (m["cnot"] + ctx.lam * max(0, m["depth"] - slack), m["depth"])

            mode, i, best = min(everything, key=lambda x: fit_cost(x[2]))
            key = str(best.to_stim())
            out = [(f"{mode}{i}~fit{slack}", best)] + [(lab, c) for lab, c in out if str(c.to_stim()) != key]
    return out


CHOICE_CTX_DEPS = ("synth", "synthesizers", "pool", "lam")  # the ``Context`` fields ``circuit_choices`` reads
JITTER = 0.3  # the further samples of a cover or promotion: ILP costs times exp(JITTER * N(0, 1))


def block_slack(node: Node, spec: BlockSpec) -> int:
    """Depth budget of a block's ancilla prep: the earliest time any of its data qubits is free where it couples."""
    c = compile_node(node)
    upto = c.base_index[spec.pos] if spec.pos is not None else None
    rt = ready_times(c, upto)
    return min((rt.get(c.data[q], 0) for q in spec.support), default=0)


def scope_slack(design: Design, path: tuple[int, ...]) -> int | None:
    """Depth budget of the ancilla scope at ``path`` (``block_slack``); None for the root or if unknown."""
    if not path:
        return None
    try:
        parent = design.node(path[:-1])
        spec = parent.blocks[path[-1]][0]
        return block_slack(parent.truncated(path[-1]), spec)
    except Exception:  # ruff: ignore[blind-except]
        return None


def part_cost(ctx: Context, m: dict, lam: float, slack: int | None) -> float:
    """Cost of a known protection (a part) for a scope with depth budget ``slack``: only depth beyond it counts."""
    if slack is None:
        return ctx.cost(m, lam)
    return m["cnot"] + m["meas"] + lam * max(0, m["depth"] - slack)


def removable(node: Node, block_saving: Callable[[BlockSpec], int] | None = None) -> list[tuple[int, str, tuple]]:
    """The elements of ``node`` that a solver may remove, as (saving estimate, label, element).

    Elements: ("g", gi) a check outside blocks with its flags, ("f", gi, fi) one flag, ("b", bi) a whole block with
    its ancilla subtree. A block saves ``block_saving(spec)`` (default three per ancilla qubit).
    """
    out: list[tuple[int, str, tuple]] = []
    for gi, g in enumerate(node.gadgets):
        name = f"{g.kind}{len(g.support)}"
        if g.block is None:  # a block's ancilla qubits go only with the whole block
            out.append((
                len(g.support) + 1 + 3 * len(g.flags),
                f"{'late check' if g.late else 'check'} {name}",
                ("g", gi),
            ))
        out.extend((3, f"flag {name}.{fi}", ("f", gi, fi)) for fi in range(len(g.flags)))
    for bi, (spec, _) in enumerate(node.blocks):
        saving = 3 * spec.n_anc if block_saving is None else block_saving(spec)
        out.append((saving, f"block {spec.kind}{spec.checks.shape[0]}x{spec.n_anc}", ("b", bi)))
    return out


def remove(node: Node, elem: tuple) -> bool:
    """Remove an element of ``removable`` from ``node`` in place; False if its index is out of range."""
    kind, i, *rest = elem
    if kind == "b":
        if i >= len(node.blocks):
            return False
        node.remove_block(i)
        return True
    if i >= len(node.gadgets):
        return False
    if kind == "g":
        node.gadgets = [g for j, g in enumerate(node.gadgets) if j != i]
    else:
        g = node.gadgets[i]
        flags = [f for j, f in enumerate(g.flags) if j != rest[0]]
        node.gadgets = [*node.gadgets[:i], dataclasses.replace(g, flags=flags), *node.gadgets[i + 1 :]]
    return True


def _blocks_of(node: Node) -> list:
    return [(spec, compile_node(ch)) for spec, ch in node.blocks]


def from_qiskit(qc: QuantumCircuit, n_data: int) -> Circuit:
    """A qiskit circuit of MQT QECC (data on the first ``n_data`` qubits) as a :class:`Circuit`."""
    return Circuit.from_stim(qiskit_to_stim_circuit(qc), data=list(range(n_data)))


def peham_verified(sp: FaultyStatePrepCircuit, *, find_coset_leaders: bool = False) -> Circuit:
    """Preparation circuit and heuristic verification of MQT QECC (Peham et al.), as a :class:`Circuit`."""
    qc = heuristic_verification_circuit(sp, find_coset_leaders=find_coset_leaders)
    return from_qiskit(qc, sp.num_qubits)


def to_qecc_prep(circ: Circuit, t: int) -> FaultyStatePrepCircuit:
    """A non-fault-tolerant preparation circuit as a ``FaultyStatePrepCircuit`` of MQT QECC."""
    return FaultyStatePrepCircuit(to_cnot_circuit(circ), t, t)


def to_cnot_circuit(circ: Circuit) -> CNOTCircuit:
    """A non-fault-tolerant preparation circuit (resets, H before any CX, CX) as a ``CNOTCircuit`` (R + H -> RX)."""
    plus, touched = set(), set()
    for op in circ.ops:
        if op.name == "H" and op.qubits[0] not in touched:
            plus.add(op.qubits[0])
        elif op.name == "CX":
            touched.update(op.qubits)
        elif op.name not in {"R", "RX", "H"}:
            msg = f"unexpected {op.name} in a CNOT preparation circuit"
            raise ValueError(msg)
    s = stim.Circuit()
    for op in circ.ops:
        if op.name == "RX":
            plus.add(op.qubits[0])
    for q in range(circ.num_qubits):
        s.append("RX" if q in plus else "R", [q])
    for op in circ.ops:
        if op.name == "CX":
            s.append("CX", list(op.qubits))
    return CNOTCircuit.from_stim_circuit(s)


class QeccSolver:
    """Synthesis and verification by MQT QECC (``peham_verified``) on a bare preparation circuit."""

    ctx_deps = ()
    name = "qecc"

    def propose(  # ruff: ignore[no-self-use]  (solver interface)
        self,
        design: Design,
        path: tuple[int, ...],
        ctx: Context,  # ruff: ignore[unused-method-argument]  (solver interface)
    ) -> Iterator[tuple[str, Node]]:
        """Yield the scope with MQT QECC's verified prep."""
        node = design.node(path)
        if node.opaque or node.gadgets or node.blocks or node.base.data != list(range(len(node.base.data))):
            return
        try:
            circ = peham_verified(to_qecc_prep(node.base, design.t))
        except Exception:  # ruff: ignore[blind-except]
            return
        nd = copy.copy(node)
        nd.base, nd.opaque, nd.origin = circ, True, node.origin + "+qecc"
        yield "qecc", nd
