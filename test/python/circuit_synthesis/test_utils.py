# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test utility functions for the circuit synthesis module."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest
import stim
from ldpc import mod2
from qiskit import AncillaRegister, ClassicalRegister, QuantumCircuit, QuantumRegister
from stim import Flow, PauliString

from mqt.qecc.circuit_synthesis.circuit_utils import compact_stim_circuit, qiskit_to_stim_circuit
from mqt.qecc.circuit_synthesis.state_prep import final_matrix_constraint
from mqt.qecc.circuit_synthesis.synthesis_utils import (
    gaussian_elimination_min_column_ops,
    gaussian_elimination_min_parallel_eliminations,
    measure_flagged,
    measure_stab_unflagged,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from qiskit import AncillaQubit, ClBit, Qubit


class MatrixTest(NamedTuple):
    """Test matrix and expected results."""

    matrix: npt.NDArray[np.int8]
    min_col_ops: int
    max_parallel_steps: int


def check_correct_elimination(
    matrix: npt.NDArray[np.int8], final_matrix: npt.NDArray[np.int8], col_ops: list[tuple[int, int]]
) -> bool:
    """Check if the matrix is correctly eliminated."""
    matrix = matrix.copy()
    for op in col_ops:
        matrix[:, op[1]] = (matrix[:, op[0]] + matrix[:, op[1]]) % 2
    return np.array_equal(matrix, final_matrix)


def get_n_parallel_layers(ops: list[tuple[int, int]]) -> int:
    """Get the number of parallel layers in the elimination."""
    used_cols: set[int] = set()
    layer = 0
    for op in ops:
        if op[0] in used_cols or op[1] in used_cols:
            layer += 1
            used_cols = set()
        used_cols.add(op[0])
        used_cols.add(op[1])
    return layer


@pytest.fixture
def identity_matrix() -> MatrixTest:
    """Return a 4x4 identity matrix."""
    return MatrixTest(np.eye(4, dtype=np.int8), 0, 0)


@pytest.fixture
def full_matrix() -> MatrixTest:
    """Return a 4x4 matrix with all ones."""
    return MatrixTest(np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.int8), 3, 2)


class MeasurementTest(NamedTuple):
    """Class containing all information for a measurement test."""

    qc: QuantumCircuit
    stab: list[Qubit]
    ancilla: AncillaQubit
    measurement_bit: ClBit


def _make_measurement_test(n: int, stab: list[int]) -> MeasurementTest:
    q = QuantumRegister(n, "q")
    c = ClassicalRegister(1, "c")
    anc = AncillaRegister(1, "anc")
    qc = QuantumCircuit(q, c, anc)
    stab_qubits = [q[i] for i in stab]
    ancilla = anc[0]
    measurement_bit = c[0]
    return MeasurementTest(qc, stab_qubits, ancilla, measurement_bit)


@pytest.mark.parametrize("test_vals", ["identity_matrix", "full_matrix"])
def test_min_column_ops(test_vals: MatrixTest, request) -> None:  # type: ignore[no-untyped-def]
    """Check correct number of column operations are returned."""
    fixture = request.getfixturevalue(test_vals)
    matrix = fixture.matrix
    min_col_ops = fixture.min_col_ops
    rank = mod2.rank(matrix)
    res = gaussian_elimination_min_column_ops(
        matrix, lambda checks: final_matrix_constraint(checks, rank), max_eliminations=fixture.min_col_ops
    )
    assert res is not None
    reduced, ops = res
    assert len(ops) == min_col_ops
    assert check_correct_elimination(matrix, reduced, ops)


@pytest.mark.parametrize("test_vals", ["identity_matrix", "full_matrix"])
def test_min_parallel_eliminations(test_vals: MatrixTest, request) -> None:  # type: ignore[no-untyped-def]
    """Check correct number of parallel eliminations are returned."""
    fixture = request.getfixturevalue(test_vals)
    matrix = fixture.matrix
    rank = mod2.rank(matrix)
    max_parallel_steps = fixture.max_parallel_steps
    res = gaussian_elimination_min_parallel_eliminations(
        matrix, lambda checks: final_matrix_constraint(checks, rank), max_parallel_steps=fixture.max_parallel_steps
    )
    assert res is not None
    reduced, ops = res

    assert check_correct_elimination(matrix, reduced, ops)

    n_parallel_layers = get_n_parallel_layers(ops)
    assert n_parallel_layers <= max_parallel_steps


def correct_stabilizer_propagation(
    qc: QuantumCircuit, stab: list[Qubit], ancilla: AncillaQubit, z_measurement: bool
) -> bool:
    """Check that the stabilizer is propagated correctly."""
    circ = qiskit_to_stim_circuit(qc)
    pauli = "Z" if z_measurement else "X"
    anc_idx = qc.find_bit(ancilla).index
    initial_pauli = PauliString("_" * (anc_idx) + "Z" + "_" * (len(qc.qubits) - anc_idx - 1))
    final_pauli = PauliString(
        "".join([pauli if q in stab else "_" for q in qc.qubits]) + "_" * (len(qc.qubits) - anc_idx - 1)
    )
    f = Flow(input=initial_pauli, output=final_pauli, measurements=[qc.num_ancillas - 1])
    return bool(circ.has_flow(f))


@pytest.mark.parametrize("w", list(range(4, 12)))
@pytest.mark.parametrize("z_measurement", [True, False])
def test_one_flag_measurements(w: int, z_measurement: bool) -> None:
    """Test one-flag measurement circuits."""
    z_test = _make_measurement_test(w, list(range(w)))
    qc = z_test.qc
    stab = z_test.stab
    ancilla = z_test.ancilla
    measurement_bit = z_test.measurement_bit

    measure_flagged(qc, stab, ancilla, measurement_bit, t=1, z_measurement=z_measurement)
    assert qc.depth() == len(stab) + 3 + 2 * int(not z_measurement)  # 6 CNOTs + Measurement + 2 possible hadamards
    assert qc.count_ops().get("cx", 0) == len(stab) + 2  # CNOTs from measurement + 2 flagging CNOTs
    assert correct_stabilizer_propagation(qc, stab, ancilla, z_measurement)


@pytest.mark.parametrize("w", list(range(4, 20)))
@pytest.mark.parametrize("z_measurement", [True, False])
def test_two_flag_measurements(w: int, z_measurement: bool) -> None:
    """Test two-flag measurement circuits."""
    z_test = _make_measurement_test(w, list(range(w)))
    qc = z_test.qc
    stab = z_test.stab
    ancilla = z_test.ancilla
    measurement_bit = z_test.measurement_bit

    measure_flagged(qc, stab, ancilla, measurement_bit, t=2, z_measurement=z_measurement)
    assert correct_stabilizer_propagation(qc, stab, ancilla, z_measurement)


@pytest.mark.parametrize("w", [4, 5, 6, 7, 8, 11, 12])
@pytest.mark.parametrize("z_measurement", [True, False])
def test_three_flag_measurements(w: int, z_measurement: bool) -> None:
    """Test three-flag measurement circuits."""
    z_test = _make_measurement_test(w, list(range(w)))
    qc = z_test.qc
    stab = z_test.stab
    ancilla = z_test.ancilla
    measurement_bit = z_test.measurement_bit

    measure_flagged(qc, stab, ancilla, measurement_bit, t=3, z_measurement=z_measurement)
    assert correct_stabilizer_propagation(qc, stab, ancilla, z_measurement)


@pytest.mark.parametrize("w", list(range(4, 30)))
@pytest.mark.parametrize("z_measurement", [True, False])
def test_unflagged_measurements(w: int, z_measurement: bool) -> None:
    """Test unflagged measurement circuits."""
    z_test = _make_measurement_test(w, list(range(w)))
    qc = z_test.qc
    stab = z_test.stab
    ancilla = z_test.ancilla
    measurement_bit = z_test.measurement_bit

    measure_stab_unflagged(qc, stab, ancilla, measurement_bit, z_measurement)
    assert correct_stabilizer_propagation(qc, stab, ancilla, z_measurement)


@pytest.mark.parametrize("w", [5, 6])
@pytest.mark.parametrize("z_measurement", [True, False])
def test_w_flag(w: int, z_measurement: bool) -> None:
    """Test three-flag measurement circuits."""
    z_test = _make_measurement_test(w, list(range(w)))
    qc = z_test.qc
    stab = z_test.stab
    ancilla = z_test.ancilla
    measurement_bit = z_test.measurement_bit

    measure_flagged(qc, stab, ancilla, measurement_bit, t=w, z_measurement=z_measurement)
    assert correct_stabilizer_propagation(qc, stab, ancilla, z_measurement)


def test_compact_stim_circuit() -> None:
    """Test compaction method."""
    circ = stim.Circuit()
    circ.append("H", [0])
    circ.append("CX", [0, 1])
    circ.append("H", [2])
    circ.append("CX", [2, 3])

    assert len(circ) == 4
    compacted = compact_stim_circuit(circ)
    assert len(compacted) == 2
