# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Named MQT QECC codes, the [[17,1,5]] color code fixtures and non-FT preps for the tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import stim

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import FaultyStatePrepCircuit, heuristic_prep_circuit
from mqt.qecc.circuit_synthesis.ft_search._core.circuit import Circuit
from mqt.qecc.circuit_synthesis.ft_search._core.solvers.common import from_qiskit
from mqt.qecc.circuit_synthesis.ft_search._core.verify import CSSState, _pack, gf2_row_basis, state_of_code
from mqt.qecc.codes import SquareOctagonColorCode

if TYPE_CHECKING:
    from mqt.qecc.circuit_synthesis.ft_search._core.faults import FaultTable

RESULTS = Path(__file__).resolve().parent / "results"


@dataclass
class Code:
    """CSS code matrices (uint8)."""

    name: str
    hx: np.ndarray
    hz: np.ndarray
    lx: np.ndarray
    lz: np.ndarray
    d: int
    qecc: CSSCode

    @property
    def n(self) -> int:
        """Number of data qubits."""
        return self.hx.shape[1]

    def zero_state(self) -> CSSState:
        """The logical |0...0>."""
        return CSSState.zero_state(self.hx, self.hz, self.lz)


def load_code(name: str) -> Code:
    """A named code from mqt.qecc."""
    c = CSSCode.from_code_name(name)
    u8 = lambda a: np.asarray(a, dtype=np.uint8) % 2
    return Code(name, u8(c.Hx), u8(c.Hz), u8(c.Lx), u8(c.Lz), int(c.distance), c)


def color_17_1_5_state() -> CSSState:
    """The logical |0> of the [[17,1,5]] square-octagon color code."""
    return state_of_code(SquareOctagonColorCode(5))


def color_17_1_5(prep: str) -> tuple[Circuit, CSSState]:
    """A stored non-FT preparation circuit of the [[17,1,5]] color code's logical |0>, and that state.

    Args:
        prep: "gate" (fewest CNOTs) or "depth" (fewest CNOT layers).

    Returns:
        The circuit (data qubits 0..16) and the state.
    """
    return load_circuit(RESULTS / "inputs" / f"color_17_1_5_{prep}.stim", 17), color_17_1_5_state()


def load_circuit(path: Path, n_data: int) -> Circuit:
    """A stim circuit file as a :class:`Circuit` with data qubits 0..n_data-1."""
    return Circuit.from_stim(stim.Circuit(path.read_text(encoding="utf-8")), data=list(range(n_data)))


def prep_circuit(sp: FaultyStatePrepCircuit) -> Circuit:
    """The non-FT preparation circuit as a :class:`Circuit`."""
    return from_qiskit(sp.circ.to_qiskit_circuit(), sp.num_qubits)


def heuristic_prep(code: Code, *, optimize_depth: bool = False) -> FaultyStatePrepCircuit:
    """Qecc's heuristic |0_L> preparation."""
    return heuristic_prep_circuit(code.qecc, optimize_depth=optimize_depth, state="all_zero")


def table_keys(
    table: FaultTable, state: CSSState, det: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fault keys from a ``FaultTable``: a reference for the compiled keys of the exact check.

    Returns the X syndrome, Z syndrome and detector signature of each fault, and the row bases of ``state.sx`` and
    ``state.sz``. ``det``: detectors as rows over the records (default: each record).
    """
    sz = gf2_row_basis(state.sz)
    sx = gf2_row_basis(state.sx)
    kx = _pack((table.data_x.astype(np.int64) @ sz.T.astype(np.int64)) % 2)
    kz = _pack((table.data_z.astype(np.int64) @ sx.T.astype(np.int64)) % 2)
    flips = table.flips
    if det is not None:
        flips = (flips.astype(np.int64) @ det.T.astype(np.int64)) % 2
    if flips.shape[1] > 64:
        proj = np.random.default_rng(0).integers(0, 2, size=(flips.shape[1], 64), dtype=np.int64)
        flips = (flips.astype(np.int64) @ proj) % 2
    return kx, kz, _pack(flips), sx, sz
