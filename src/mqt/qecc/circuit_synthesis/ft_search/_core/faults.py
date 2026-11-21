# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Elementary fault locations and their linear response.

Noise model (circuit level, no idling): after every reset and ``H`` one of the 3 single-qubit Paulis, after every
``CX``/``CZ`` one of the 15 two-qubit Paulis, and at every measurement a flip of its outcome (for ``MR``/``MRX`` also a
Pauli after the reset). The circuit is Clifford, so the response of a fault set (data X error, data Z error, flipped
records) is the XOR of the elementary responses. The compiled propagation is ``kernels._propagate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .circuit import MEASUREMENTS, RESETS, Circuit

if TYPE_CHECKING:
    from collections.abc import Sequence

_P1 = ["X", "Y", "Z"]
_P2 = [a + b for a in "IXYZ" for b in "IXYZ" if a + b != "II"]


@dataclass
class FaultTable:
    """Response of every elementary fault.

    Attributes:
        op_index: Instruction index the fault follows, shape (F,).
        label: Pauli label per fault ("XZ" on a two-qubit gate, "X" on one qubit, "flip" for a measurement).
        data_x: X part of the output error on the data qubits, (F, n_data) uint8.
        data_z: Z part of the output error on the data qubits, (F, n_data) uint8.
        flips: Flipped measurement records, (F, m) uint8.
    """

    op_index: np.ndarray
    label: list[str]
    data_x: np.ndarray
    data_z: np.ndarray
    flips: np.ndarray
    frames: dict | None = None  # p -> (x, z) frames right before op p, over the qubits ``frame_cols`` (default all)
    frame_cols: list | None = None
    probe: np.ndarray | None = None  # (F, probes): one frame bit per requested (op index, qubit, "x" | "z")

    def frame(self, p: int, qubits: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """The frames before op ``p`` restricted to ``qubits`` (columns in that order)."""
        assert self.frames is not None, "frames are recorded only with ``record_at``"
        x, z = self.frames[p]
        cols = list(qubits) if self.frame_cols is None else [self.frame_cols.index(q) for q in qubits]
        if self.frame_cols is not None and cols == list(range(len(self.frame_cols))):
            return x, z
        return x[:, cols], z[:, cols]

    def __len__(self) -> int:
        """Number of elementary faults."""
        return len(self.label)


_OPCODE = {"R": 0, "RX": 1, "H": 2, "CX": 3, "CZ": 4, "M": 5, "MX": 6, "MR": 7, "MRX": 8}  # op codes of the kernels
_FAULTS_PER_OP = {"M": 1, "MX": 1, "MR": 4, "MRX": 4, "CX": 15, "CZ": 15}


def _first_fault(circ: Circuit) -> np.ndarray:
    """Index of the first elementary fault of each op (order of ``_fault_list``), one more entry for the end."""
    per = np.array([_FAULTS_PER_OP.get(op.name, 3) for op in circ.ops], dtype=np.int64)
    return np.concatenate([[0], np.cumsum(per)])


def _op_arrays(circ: Circuit) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Op codes and qubits of ``circ`` for the kernels."""
    names = np.array([_OPCODE[op.name] for op in circ.ops], dtype=np.uint8)
    q0 = np.array([op.qubits[0] for op in circ.ops], dtype=np.int64)
    q1 = np.array([op.qubits[1] if len(op.qubits) == 2 else -1 for op in circ.ops], dtype=np.int64)
    return names, q0, q1


def _fault_list(circ: Circuit) -> list[tuple[int, str]]:
    """(op index, Pauli label or "flip") of every elementary fault, in the order of the fault keys."""
    out: list[tuple[int, str]] = []
    for i, op in enumerate(circ.ops):
        if op.name in MEASUREMENTS:
            out.append((i, "flip"))
            if op.name in {"MR", "MRX"}:
                out.extend((i, p) for p in _P1)
        elif len(op.qubits) == 2:
            out.extend((i, p) for p in _P2)
        else:
            out.extend((i, p) for p in _P1)
    return out


def fault_table(
    circ: Circuit,
    record_at: list[int] | None = None,
    frame_qubits: list[int] | None = None,
    probes: list[tuple[int, int, str]] | None = None,
) -> FaultTable:
    """Propagate every elementary fault to the end of the circuit (all faults in parallel).

    ``record_at``: op indices p; the frames of all faults right before op p (p = len(ops): at the end) are kept, on the
    qubits ``frame_qubits`` only if given (read them with ``FaultTable.frame``). ``probes``: (p, qubit, "x" | "z")
    single frame bits right before op p, in ``FaultTable.probe``.
    """
    fq = None if frame_qubits is None else list(frame_qubits)
    probes = list(probes or [])
    probe_at: dict[int, list[int]] = {}
    for k, (p, _, _) in enumerate(probes):
        probe_at.setdefault(p, []).append(k)
    faults = _fault_list(circ)
    f = len(faults)
    n = circ.num_qubits
    x = np.zeros((f, n), dtype=np.uint8)
    z = np.zeros((f, n), dtype=np.uint8)
    flips = np.zeros((f, circ.num_measurements), dtype=np.uint8)
    probe = np.zeros((f, len(probes)), dtype=np.uint8)

    def take(i: int) -> None:
        for k in probe_at.get(i, ()):
            _, qq, b = probes[k]
            probe[:, k] = (x if b == "x" else z)[:, qq]

    by_op: dict[int, list[int]] = {}
    for k, (i, _) in enumerate(faults):
        by_op.setdefault(i, []).append(k)

    rec = 0
    want = set(record_at or [])
    frames: dict = {}
    for i, op in enumerate(circ.ops):
        if i in want:
            frames[i] = (x.copy(), z.copy()) if fq is None else (x[:, fq].copy(), z[:, fq].copy())
        take(i)
        q = op.qubits
        name = op.name
        # 1. propagate existing frames through the instruction
        if name == "CX":
            c, t = q
            x[:, t] ^= x[:, c]
            z[:, c] ^= z[:, t]
        elif name == "CZ":
            a, b = q
            z[:, a] ^= x[:, b]
            z[:, b] ^= x[:, a]
        elif name == "H":
            (a,) = q
            x[:, a], z[:, a] = z[:, a].copy(), x[:, a].copy()
        elif name in RESETS:
            (a,) = q
            x[:, a] = 0
            z[:, a] = 0
        elif name in MEASUREMENTS:
            (a,) = q
            flips[:, rec] ^= x[:, a] if name in {"M", "MR"} else z[:, a]
            if name in {"MR", "MRX"}:
                x[:, a] = 0
                z[:, a] = 0
        # 2. inject this instruction's faults
        for k in by_op.get(i, []):
            label = faults[k][1]
            if label == "flip":
                flips[k, rec] ^= 1
                continue
            for qq, p in zip(q, label, strict=True):
                if p in "XY":
                    x[k, qq] ^= 1
                if p in "ZY":
                    z[k, qq] ^= 1
        if name in MEASUREMENTS:
            rec += 1

    if len(circ.ops) in want:
        frames[len(circ.ops)] = (x.copy(), z.copy()) if fq is None else (x[:, fq].copy(), z[:, fq].copy())
    take(len(circ.ops))
    d = circ.data
    return FaultTable(
        op_index=np.array([i for i, _ in faults], dtype=np.int64),
        label=[lab for _, lab in faults],
        data_x=x[:, d].copy(),
        data_z=z[:, d].copy(),
        flips=flips,
        frames=frames if record_at is not None else None,
        frame_cols=fq,
        probe=probe if probes else None,
    )
