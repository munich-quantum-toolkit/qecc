# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test utility functions for the circuit synthesis module."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import pytest
import stim
import z3
from qiskit import AncillaRegister, ClassicalRegister, QuantumCircuit, QuantumRegister
from stim import Flow, PauliString

from mqt.qecc.circuit_synthesis.circuit_utils import (
    collect_circuit_layers,
    compact_stim_circuit,
    compose_circuits,
    compose_compact_stim_circuits,
    measured_qubits,
    qiskit_to_stim_circuit,
    unmeasured_qubits,
)
from mqt.qecc.circuit_synthesis.state_prep import final_matrix_constraint
from mqt.qecc.circuit_synthesis.synthesis_utils import (
    gaussian_elimination_min_column_ops,
    gaussian_elimination_min_parallel_eliminations,
    heuristic_gaussian_elimination,
    measure_flagged,
    measure_stab_unflagged,
    odd_overlap,
    symbolic_scalar_mult,
    symbolic_vector_add,
    symbolic_vector_eq,
    vars_to_stab,
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
    matrix: npt.NDArray[np.int8],
    final_matrix: npt.NDArray[np.int8],
    col_ops: list[tuple[int, int]],
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
    return MatrixTest(
        np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.int8),
        3,
        2,
    )


@pytest.fixture
def single_row_matrix() -> MatrixTest:
    """Return a matrix with a single row."""
    return MatrixTest(np.array([[1, 0, 1, 0]], dtype=np.int8), 1, 1)


@pytest.fixture
def single_column_matrix() -> MatrixTest:
    """Return a matrix with a single ."""
    return MatrixTest(np.array([[1], [0], [1], [1]], dtype=np.int8), 0, 0)


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


@pytest.mark.parametrize(
    "test_vals",
    ["identity_matrix", "full_matrix", "single_row_matrix", "single_column_matrix"],
)
def test_min_column_ops(test_vals: MatrixTest, request) -> None:  # type: ignore[no-untyped-def]
    """Check correct number of column operations are returned."""
    fixture = request.getfixturevalue(test_vals)
    matrix = fixture.matrix
    min_col_ops = fixture.min_col_ops
    rank = mod2.rank(matrix)
    res = gaussian_elimination_min_column_ops(
        matrix,
        lambda checks: final_matrix_constraint(checks, rank),
        max_eliminations=fixture.min_col_ops,
    )
    assert res is not None
    reduced, ops = res
    assert len(ops) == min_col_ops
    assert check_correct_elimination(matrix, reduced, ops)


@pytest.mark.parametrize(
    "test_vals",
    ["identity_matrix", "full_matrix", "single_row_matrix", "single_column_matrix"],
)
def test_min_parallel_eliminations(test_vals: MatrixTest, request) -> None:  # type: ignore[no-untyped-def]
    """Check correct number of parallel eliminations are returned."""
    fixture = request.getfixturevalue(test_vals)
    matrix = fixture.matrix
    rank = mod2.rank(matrix)
    max_parallel_steps = fixture.max_parallel_steps
    res = gaussian_elimination_min_parallel_eliminations(
        matrix,
        lambda checks: final_matrix_constraint(checks, rank),
        max_parallel_steps=fixture.max_parallel_steps,
    )
    assert res is not None
    reduced, ops = res

    assert check_correct_elimination(matrix, reduced, ops)

    n_parallel_layers = get_n_parallel_layers(ops)
    assert n_parallel_layers <= max_parallel_steps


@pytest.mark.parametrize(
    "test_vals",
    ["identity_matrix", "full_matrix", "single_row_matrix", "single_column_matrix"],
)
def test_heuristic_gaussian_elimination(test_vals: MatrixTest, request) -> None:  # type: ignore[no-untyped-def]
    """Test heuristic Gaussian elimination method."""
    fixture = request.getfixturevalue(test_vals)
    matrix = fixture.matrix
    res_seq = heuristic_gaussian_elimination(matrix, parallel_elimination=False)
    res_parallel = heuristic_gaussian_elimination(matrix, parallel_elimination=True)

    assert res_seq is not None
    reduced_seq, ops_seq = res_seq

    assert res_parallel is not None
    reduced_parallel, ops_parallel = res_parallel

    assert check_correct_elimination(matrix, reduced_seq, ops_seq)
    assert check_correct_elimination(matrix, reduced_parallel, ops_parallel)

    n_parallel_layers_seq = get_n_parallel_layers(ops_seq)
    assert n_parallel_layers_seq <= fixture.max_parallel_steps

    n_parallel_layers_parallel = get_n_parallel_layers(ops_parallel)
    assert n_parallel_layers_parallel <= fixture.max_parallel_steps

    assert n_parallel_layers_parallel <= n_parallel_layers_seq


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
    compacted = compact_stim_circuit(circ, scheduling_method="asap")
    assert len(compacted) == 2
    compacted = compact_stim_circuit(circ, scheduling_method="alap")
    assert len(compacted) == 2


def test_compose_compact_stim_circuits() -> None:
    """Test compaction method."""
    circ1 = stim.Circuit()
    circ1.append("H", [0])
    circ1.append("CX", [0, 1])
    circ2 = stim.Circuit()
    circ2.append("CX", [2, 3])

    compacted1 = compose_compact_stim_circuits([circ1, circ2], align="start")
    assert len(compacted1) == 2
    assert compacted1 == stim.Circuit("H 0\nCX 2 3 0 1")
    compacted2 = compose_compact_stim_circuits([circ1, circ2], align="end")
    assert len(compacted2) == 2
    assert compacted2 == stim.Circuit("H 0\nCX 0 1 2 3")


def test_invalid_compose_option() -> None:
    """Test invalid compaction option raises an error."""
    circ1 = stim.Circuit(" ")
    with pytest.raises(ValueError, match="align must be 'start' or 'end'"):
        compose_compact_stim_circuits([circ1], align="foo")


class TestSymbolicVectorOperations:
    """Test class for symbolic vector operations, including ~odd_overlap~."""

    x = z3.Bool("x")
    y = z3.Bool("y")

    @pytest.mark.parametrize(
        ("lhs", "rhs", "expected_result"),
        [
            (np.array([True, False, True]), np.array([False, True, False]), z3.unsat),
            (np.array([True, False, True]), np.array([True, False, True]), z3.sat),
            (np.array([x, y, z3.Not(x)]), np.array([x, y, z3.Not(x)]), z3.sat),
            (np.array([x, y, z3.Not(x)]), np.array([y, x, y]), z3.unsat),
            (np.array([True, False, x]), np.array([True, False, x]), z3.sat),
            (np.array([True, False, x]), np.array([False, True, x]), z3.unsat),
            (np.array([], dtype=bool), np.array([], dtype=bool), z3.sat),
        ],
    )
    def test_symbolic_vector_eq(self, lhs, rhs, expected_result):  # noqa: PLR6301
        """Parameterized test for ~symbolic_vector_eq~."""
        # Test equal vectors
        solver = z3.Solver()
        solver.add(symbolic_vector_eq(lhs, rhs))
        assert solver.check() == expected_result

    def test_symbolic_vector_eq_different_lengths(self):
        """Test symbolic_vector_eq with vectors of different lengths."""
        lhs = np.array([True, False, self.x])
        rhs = np.array([True, False])
        with pytest.raises(ValueError, match=r"Vectors must have the same length for equality check."):
            symbolic_vector_eq(lhs, rhs)

    @pytest.mark.parametrize(
        ("v_sym", "v_con", "expected_result"),
        [
            # Empty constant vector
            (
                np.array([z3.Bool(f"x{i}") for i in range(5)]),
                np.array([0, 0, 0, 0, 0], dtype=np.int8),
                z3.unsat,
            ),
            (
                np.array([z3.Bool(f"x{i}") for i in range(5)]),
                np.array([1, 0, 1, 0, 0], dtype=np.int8),
                z3.sat,
            ),
            (
                np.array([z3.Bool(f"x{i}") for i in range(5)]),
                np.array([1, 0, 1, 0, 1], dtype=np.int8),
                z3.sat,
            ),
            # Mixed boolean and symbolic values
            (
                np.array([True, z3.Bool("x1"), False, z3.Bool("x3"), True]),
                np.array([1, 1, 0, 1, 0], dtype=np.int8),
                z3.sat,
            ),
            (np.array([True, x, x]), np.array([False, True, True]), z3.unsat),
        ],
    )
    def test_odd_overlap(self, v_sym, v_con, expected_result):  # noqa: PLR6301
        """Parameterized test for ~odd_overlap~."""
        solver = z3.Solver()
        solver.add(odd_overlap(v_sym, v_con))
        assert solver.check() == expected_result, f"Test failed for v_sym={v_sym}, v_con={v_con}"

    @pytest.mark.parametrize(
        ("v", "scalar", "expected_result"),
        [
            # Empty vector
            (np.array([], dtype=np.int8), True, np.array([])),
            # Scalar multiplication with True
            (np.array([1, 0, 1], dtype=np.int8), True, np.array([True, False, True])),
            # Scalar multiplication with False
            (
                np.array([1, 0, 1], dtype=np.int8),
                False,
                np.array([False, False, False]),
            ),
            # Scalar multiplication with a Z3 variable (x)
            (np.array([1, 0, 1], dtype=np.int8), x, np.array([x, False, x])),
            # Scalar multiplication with a Z3 variable (y)
            (np.array([1, 0, 0, 1], dtype=np.int8), y, np.array([y, False, False, y])),
        ],
    )
    def test_symbolic_scalar_mult(self, v, scalar, expected_result):  # noqa: PLR6301
        """Parameterized test for ~symbolic_scalar_mult~."""
        result = symbolic_scalar_mult(v, scalar)
        assert np.array_equal(result, expected_result), f"Test failed for v={v}, scalar={scalar}"

    @pytest.mark.parametrize(
        ("v1", "v2", "expected_result"),
        [
            # Empty vectors
            (np.array([]), np.array([]), np.array([])),
            # Addition of boolean vectors
            (
                np.array([True, False, True]),
                np.array([False, True, False]),
                np.array([True, True, True]),
            ),
            # Addition of symbolic vectors
            (
                np.array([x, y, z3.Not(x)]),
                np.array([x, y, z3.Not(x)]),
                np.array([z3.BoolVal(False), z3.BoolVal(False), z3.BoolVal(False)]),
            ),
            # Mixed boolean and symbolic vectors
            (
                np.array([True, False, x]),
                np.array([False, True, y]),
                np.array([True, True, z3.Xor(x, y)]),
            ),
            # Boolean and symbolic simplifications
            (
                np.array([True, x, False]),
                np.array([False, y, True]),
                np.array([True, z3.Xor(x, y), True]),
            ),
        ],
    )
    def test_symbolic_vector_add(self, v1, v2, expected_result):  # noqa: PLR6301
        """Parameterized test for ~symbolic_vector_add~."""
        result = symbolic_vector_add(v1, v2)
        assert np.array_equal(result, expected_result), f"Test failed for v1={v1}, v2={v2}"

    @pytest.mark.parametrize(
        ("measurement", "generators", "expected_result"),
        [
            # Single generator
            (
                [True],
                np.array([[1, 0, 1]], dtype=np.int8),
                np.array([True, False, True]),
            ),
            # Multiple generators with boolean measurements
            (
                [True, False],
                np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8),
                np.array([True, False, True]),
            ),
            # Multiple generators with symbolic measurements
            (
                [x, y],
                np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8),
                np.array([x, y, x]),
            ),
            # Mixed boolean and symbolic measurements
            (
                [True, y],
                np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8),
                np.array([True, y, True]),
            ),
        ],
    )
    def test_vars_to_stab_valid_inputs(self, measurement, generators, expected_result):  # noqa: PLR6301
        """Test ~vars_to_stab~ with valid inputs."""
        result = vars_to_stab(measurement, generators)
        assert np.array_equal(result, expected_result), (
            f"Test failed for measurement={measurement}, generators={generators}"
        )

    @pytest.mark.parametrize(
        ("measurement", "generators", "expected_exception", "expected_message"),
        [
            # Empty measurement
            (
                [],
                np.array([[1, 0, 1]], dtype=np.int8),
                ValueError,
                "Measurement must not be empty",
            ),
            # Mismatched lengths
            (
                [True],
                np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8),
                ValueError,
                "Generators and measurement must have the same length",
            ),
        ],
    )
    def test_vars_to_stab_exceptions(self, measurement, generators, expected_exception, expected_message):  # noqa: PLR6301
        """Test ~vars_to_stab~ with invalid inputs that raise exceptions."""
        with pytest.raises(expected_exception, match=expected_message):
            vars_to_stab(measurement, generators)


def test_compact_stim_circuit_empty() -> None:
    """Test compaction method on empty circuit."""
    circ = stim.Circuit()
    assert len(circ) == 0
    compacted = compact_stim_circuit(circ)
    assert len(compacted) == 0


def test_collect_circuit_layers_asap() -> None:
    """Test collecting circuit layers with ASAP scheduling."""
    circ = stim.Circuit()
    circ.append("RX", [0])
    circ.append("CX", [0, 1])
    circ.append("TICK")
    circ.append("H", [2])

    layers = collect_circuit_layers(circ, scheduling_method="asap")

    # Two layers: RX+H first (parallel), then CX
    assert len(layers) == 2
    # RX 0 and H 2 can be in either order in the same layer
    assert layers[0] == stim.Circuit("RX 0\nH 2") or layers[0] == stim.Circuit("H 2\nRX 0")
    assert layers[1] == stim.Circuit("CX 0 1")


def test_collect_circuit_layers_alap() -> None:
    """Test collecting circuit layers with ALAP scheduling."""
    circ = stim.Circuit()
    circ.append("RX", [0])
    circ.append("CX", [0, 1])
    circ.append("TICK")
    circ.append("H", [2])

    layers = collect_circuit_layers(circ, scheduling_method="alap")

    # Two layers: CX first, then RX+H
    assert len(layers) == 2
    assert layers[0] == stim.Circuit("RX 0")
    assert layers[1] == stim.Circuit("CX 0 1\nH 2") or layers[1] == stim.Circuit("H 2\nCX 0 1")


def test_invalid_scheduling_method() -> None:
    """Test invalid scheduling method raises an error."""
    circ = stim.Circuit("H 0")
    with pytest.raises(ValueError, match="scheduling_method must be 'asap' or 'alap'"):
        collect_circuit_layers(circ, scheduling_method="foo")


def test_collect_circuit_layers_empty_circuit() -> None:
    """Test collecting layers from an empty circuit."""
    circ = stim.Circuit()
    layers = collect_circuit_layers(circ)
    assert len(layers) == 0  # No layers in an empty circuit


def test_collect_circuit_layers_single_operation() -> None:
    """Test collecting layers from a circuit with a single operation."""
    circ = stim.Circuit("H 0")
    layers = collect_circuit_layers(circ)
    assert len(layers) == 1  # One layer with a single operation
    assert layers[0] == stim.Circuit("H 0")


@pytest.mark.parametrize(
    ("circuit_operations", "expected_unmeasured"),
    [
        ([], []),
        ([("H", [0]), ("CX", [0, 1])], [0, 1]),
        ([("H", [0]), ("CX", [0, 1]), ("MR", [1])], [0]),
        ([("H", [0]), ("CX", [0, 1]), ("MR", [0]), ("MR", [1])], []),
        (
            [("H", [i]) for i in range(10)] + [("MR", [2]), ("MR", [5]), ("MR", [7])],
            [0, 1, 3, 4, 6, 8, 9],
        ),
    ],
)
def test_unmeasured_qubits(circuit_operations, expected_unmeasured):
    """Parameterized test for unmeasured_qubits."""
    circ = stim.Circuit()
    for op, targets in circuit_operations:
        circ.append_operation(op, targets)
    assert sorted(unmeasured_qubits(circ)) == sorted(expected_unmeasured)


@pytest.mark.parametrize(
    ("circuit_operations", "expected_measured"),
    [
        # Test case 1: Empty circuit
        ([], []),
        # Test case 2: Circuit with no measurements
        ([("H", [0]), ("CX", [0, 1])], []),
        # Test case 3: Circuit with one measurement
        ([("H", [0]), ("CX", [0, 1]), ("MR", [1])], [1]),
        # Test case 4: Circuit with multiple measurements
        ([("H", [0]), ("CX", [0, 1]), ("MR", [1]), ("MR", [0])], [1, 0]),
        # Test case 5: Circuit with interleaved operations and measurements
        (
            [("H", [0]), ("MR", [1]), ("CX", [0, 1]), ("MR", [0]), ("MR", [2])],
            [1, 0, 2],
        ),
        # Test case 6: Large circuit with measurements
        (
            [("H", [i]) for i in range(10)] + [("MR", [2]), ("MR", [5]), ("MR", [7])],
            [2, 5, 7],
        ),
    ],
)
def test_measured_qubits(circuit_operations, expected_measured):
    """Parameterized test for measured_qubits."""
    circ = stim.Circuit()
    for op, targets in circuit_operations:
        circ.append_operation(op, targets)
    assert measured_qubits(circ) == expected_measured


@pytest.mark.parametrize(
    (
        "circ1_ops",
        "circ2_ops",
        "wiring",
        "expected_num_qubits",
        "expected_mapping1",
        "expected_mapping2",
    ),
    [
        # Test case 1: No wiring, circuits are vertically stacked
        (
            [("H", [0]), ("CX", [0, 1])],  # circ1 operations
            [("H", [0]), ("CX", [0, 1])],  # circ2 operations
            None,  # No wiring
            4,  # Total qubits in composed circuit
            {0: 0, 1: 1},  # Mapping for circ1
            {0: 2, 1: 3},  # Mapping for circ2
        ),
        # Test case 2: Wiring connects qubits between circuits
        (
            [("H", [0]), ("CX", [0, 1])],
            [("H", [0]), ("CX", [0, 1])],
            {1: 0},  # Wiring connects qubit 1 of circ1 to qubit 0 of circ2
            3,  # Total qubits in composed circuit
            {0: 0, 1: 2},  # Mapping for circ1
            {0: 2, 1: 1},  # Mapping for circ2
        ),
        # Test case 3: Empty circuits
        (
            [],  # circ1 operations
            [],  # circ2 operations
            None,  # No wiring
            0,  # Total qubits in composed circuit
            {},  # Mapping for circ1
            {},  # Mapping for circ2
        ),
        # Test case 4: Wiring connects all qubits
        (
            [("H", [0]), ("CX", [0, 1])],
            [("H", [0]), ("CX", [0, 1])],
            {0: 0, 1: 1},  # Wiring connects all qubits
            2,  # Total qubits in composed circuit
            {0: 0, 1: 1},  # Mapping for circ1
            {0: 0, 1: 1},  # Mapping for circ2
        ),
    ],
)
def test_compose_circuits(
    circ1_ops,
    circ2_ops,
    wiring,
    expected_num_qubits,
    expected_mapping1,
    expected_mapping2,
):
    """Parameterized test for compose_circuits."""
    # Create the first circuit
    circ1 = stim.Circuit()
    for op, targets in circ1_ops:
        circ1.append_operation(op, targets)

    # Create the second circuit
    circ2 = stim.Circuit()
    for op, targets in circ2_ops:
        circ2.append_operation(op, targets)

    # Compose the circuits
    composed, mapping1, mapping2 = compose_circuits(circ1, circ2, wiring)

    # Check the number of qubits in the composed circuit
    assert composed.num_qubits == expected_num_qubits

    # Check the mappings
    assert mapping1 == expected_mapping1
    assert mapping2 == expected_mapping2
