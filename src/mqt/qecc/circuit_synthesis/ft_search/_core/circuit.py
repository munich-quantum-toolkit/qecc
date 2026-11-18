# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Minimal circuit model for fault analysis of CSS state-preparation gadgets.

Gate set: resets ``R`` (|0>) and ``RX`` (|+>), ``H``, ``CX``, ``CZ``, and measurements ``M``, ``MX``, ``MR``, ``MRX``.
Each detector must be deterministic without faults; any flip rejects the shot (postselection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import stim

RESETS = {"R", "RX"}
ONE_QUBIT = {"H"}
TWO_QUBIT = {"CX", "CZ"}
MEASUREMENTS = {"M", "MX", "MR", "MRX"}
IGNORED = {"TICK", "I", "QUBIT_COORDS", "SHIFT_COORDS", "OBSERVABLE_INCLUDE"}


@dataclass(frozen=True)
class Op:
    """One instruction acting on one qubit (resets, H, measurements) or a pair (CX, CZ)."""

    name: str
    qubits: tuple[int, ...]


@dataclass
class Circuit:
    """A gate list with a designated set of data qubits.

    Attributes:
        ops: Instructions in time order.
        num_qubits: Total qubits (data + ancillas).
        data: Data qubit indices; the output state lives on these, in this order.
        detectors: Check bits as XORs of measurement-record indices; ``None`` = every record is its
            own check (each measurement must then be deterministic 0).
        tags: Provenance of every op, set by ``gauge.build`` (``None`` on other circuits).
        base_index: Op index of each base op, set by ``design.compile_node``.
    """

    ops: list[Op]
    num_qubits: int
    data: list[int] = field(default_factory=list)
    detectors: list[list[int]] | None = None
    tags: list[tuple] | None = field(default=None, init=False, compare=False, repr=False)
    base_index: dict[int, int] = field(default_factory=dict, init=False, compare=False, repr=False)

    @classmethod
    def from_stim(cls, circ: stim.Circuit, data: list[int] | None = None) -> Circuit:
        """Convert a stim circuit; qubits whose first use is not a reset get an implicit ``R`` at the start.

        Args:
            circ: Circuit using only the supported gate set (annotations and identity gates are dropped).
            data: Data qubits. Defaults to all qubits that are never measured.

        Raises:
            ValueError: An unsupported gate (for example a Pauli gate), a target that is not a qubit (for example a
                measurement record), an inverted measurement target or a measurement with a flip probability.
        """
        ops: list[Op] = []
        dets: list[list[int]] = []
        nmeas = 0
        for item in circ.flattened():
            inst = cast("stim.CircuitInstruction", item)  # flattened(): no repeat blocks
            name = inst.name
            if name == "DETECTOR":
                dets.append([nmeas + t.value for t in inst.targets_copy()])
                continue
            if name in IGNORED:
                continue
            if any(t.is_inverted_result_target for t in inst.targets_copy()):
                msg = f"inverted measurement targets are not supported: {inst}"
                raise ValueError(msg)
            if not all(t.is_qubit_target for t in inst.targets_copy()):
                msg = f"only qubit targets are supported: {inst}"
                raise ValueError(msg)
            if any(inst.gate_args_copy()):
                msg = f"noisy measurements are not supported: {inst}"
                raise ValueError(msg)
            qs = [t.value for t in inst.targets_copy()]
            if name in MEASUREMENTS:
                nmeas += len(qs)
            if name in TWO_QUBIT:
                ops.extend(Op(name, (qs[i], qs[i + 1])) for i in range(0, len(qs), 2))
            elif name in RESETS | ONE_QUBIT | MEASUREMENTS:
                ops.extend(Op(name, (q,)) for q in qs)
            else:
                msg = f"unsupported instruction {name}"
                raise ValueError(msg)
        n = max(circ.num_qubits, max(data, default=-1) + 1 if data is not None else 0)
        first: dict[int, str] = {}
        for op in ops:
            for q in op.qubits:
                first.setdefault(q, op.name)
        implicit = [Op("R", (q,)) for q in range(n) if first.get(q) not in RESETS]
        ops = implicit + ops
        if data is None:
            measured = {op.qubits[0] for op in ops if op.name in MEASUREMENTS}
            data = [q for q in range(n) if q not in measured]
        return cls(ops, n, list(data), dets or None)

    def to_stim(self) -> stim.Circuit:
        """Return the equivalent stim circuit."""
        c = stim.Circuit()
        for op in self.ops:
            c.append(op.name, list(op.qubits))
        if self.detectors is not None:
            m = self.num_measurements
            for d in self.detectors:
                c.append("DETECTOR", [stim.target_rec(i - m) for i in d])
        return c

    def detector_matrix(self) -> np.ndarray:
        """(checks, records) 0/1 matrix: identity if no detectors are declared."""
        m = self.num_measurements
        if self.detectors is None:
            return np.eye(m, dtype=np.uint8)
        out = np.zeros((len(self.detectors), m), dtype=np.uint8)
        for k, d in enumerate(self.detectors):
            for i in d:
                out[k, i] ^= 1
        return out

    @property
    def num_measurements(self) -> int:
        """Number of measurement records (= check bits)."""
        return sum(op.name in MEASUREMENTS for op in self.ops)


@dataclass
class Layering:
    """ASAP CNOT layering. ``slice_of_op[i]`` = first slice at which a fault after op i exists."""

    gates: list[list[tuple[int, int]]]  # gates[t] for t = 1..D (index 0 unused)
    slice_of_op: list[int]

    @property
    def depth(self) -> int:
        """Number of CNOT layers."""
        return len(self.gates) - 1


def layering(circ: Circuit) -> Layering:
    """ASAP CNOT layering of ``circ`` (the ``asap`` schedule: CNOT layer k ends at time k)."""
    for op in circ.ops:
        if op.name not in {"CX", "R", "RX", "H"}:
            msg = f"layering supports CX/R/RX/H only, got {op.name}"
            raise ValueError(msg)
    sched = asap(circ)
    gates: list[list[tuple[int, int]]] = [[] for _ in range(max((e for _, e in sched), default=0) + 1)]
    for op, (_, end) in zip(circ.ops, sched, strict=True):
        if op.name == "CX":
            c, t = op.qubits
            gates[end].append((c, t))
    return Layering(gates, [end for _, end in sched])


def asap(circ: Circuit) -> list[tuple[int, int]]:
    """As-soon-as-possible schedule: (start, end) per op; two-qubit gates take one step, every other op none."""
    ready = [0] * circ.num_qubits
    out = []
    for op in circ.ops:
        start = max(ready[q] for q in op.qubits)
        end = start + (1 if op.name in TWO_QUBIT else 0)
        for q in op.qubits:
            ready[q] = end
        out.append((start, end))
    return out


def ready_times(circ: Circuit, upto: int | None = None) -> dict[int, int]:
    """Time at which each qubit is free after the first ``upto`` ops (default: all) in the ``asap`` schedule."""
    ready: dict[int, int] = {}
    for op, (_, end) in zip(circ.ops[:upto], asap(circ), strict=False):
        for q in op.qubits:
            ready[q] = end
    return ready


def metrics(circ: Circuit) -> dict[str, int]:
    """Counts of a circuit: cnot, meas, ops (= cnot + meas), depth (two-qubit gate depth of ``asap``), qubits."""
    cnot = sum(op.name in TWO_QUBIT for op in circ.ops)
    meas = sum(op.name in MEASUREMENTS for op in circ.ops)
    return {
        "cnot": cnot,
        "meas": meas,
        "ops": cnot + meas,
        "depth": max((e for _, e in asap(circ)), default=0),
        "qubits": len({q for op in circ.ops for q in op.qubits}),
    }


def ops_depth(circ: Circuit) -> tuple[int, int]:
    """The (ops, depth) point of a circuit (``metrics``)."""
    m = metrics(circ)
    return m["ops"], m["depth"]
