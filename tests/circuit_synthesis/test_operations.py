# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for tableau operations."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from mqt.qecc.circuit_synthesis.operations import (
    CNOT,
    PauliOperation,
    SingleQubitClifford,
    Swap,
    Transvection,
)
from mqt.qecc.codes.pauli import CheckMatrix, StabilizerTableau


@pytest.fixture
def small_tableau() -> StabilizerTableau:
    """Create a small 3-qubit stabilizer tableau for testing."""
    return StabilizerTableau.from_stim_tableau(stim.Tableau(3))


@pytest.fixture
def check_matrix_x() -> CheckMatrix:
    """Create a simple X-type check matrix."""
    matrix = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)
    return CheckMatrix(matrix, "X")


@pytest.fixture
def check_matrix_z() -> CheckMatrix:
    """Create a simple Z-type check matrix."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
    return CheckMatrix(matrix, "Z")


# =====================================================================================
# Transvection Tests
# =====================================================================================


def test_transvection_all_two_qubit_transvections_count_and_validity() -> None:
    """Test that all_two_qubit_transvections returns exactly 9 valid transvections."""
    transvections = Transvection.all_two_qubit_transvections()
    assert len(transvections) == 9
    assert len(set(transvections)) == 9
    for v in transvections:
        assert len(v) == 4
        xi, xj, zi, zj = v
        assert all(x in {0, 1} for x in v)
        assert not (xi == 0 and zi == 0)
        assert not (xj == 0 and zj == 0)


def test_transvection_qubits() -> None:
    """Test that qubits() returns the correct qubit indices."""
    op = Transvection((1, 1, 0, 0), 0, 2)
    assert op.qubits() == {0, 2}

    op2 = Transvection((1, 0, 1, 0), 3, 7)
    assert op2.qubits() == {3, 7}


def test_transvection_hash_equality() -> None:
    """Test that identical transvections have equal hashes."""
    op1 = Transvection((1, 1, 0, 0), 0, 1)
    op2 = Transvection((1, 1, 0, 0), 0, 1)
    op3 = Transvection((1, 0, 1, 0), 0, 1)
    op4 = Transvection((1, 1, 0, 0), 1, 0)

    assert hash(op1) == hash(op2)
    assert hash(op1) != hash(op3)
    assert hash(op1) != hash(op4)


def test_transvection_invalid_pauli_raises() -> None:
    """Test that transvection with trivial Pauli raises ValueError."""
    circuit = stim.Circuit()
    op1 = Transvection((0, 1, 0, 0), 0, 1)
    op2 = Transvection((1, 0, 0, 0), 0, 1)

    with pytest.raises(ValueError, match="Expected non-trivial Pauli"):
        op1.append_to_circuit(circuit)

    with pytest.raises(ValueError, match="Expected non-trivial Pauli"):
        op2.append_to_circuit(circuit)


def test_transvection_not_implemented_for_check_matrix(check_matrix_x: CheckMatrix) -> None:
    """Test that transvection raises NotImplementedError for CheckMatrix."""
    op = Transvection((1, 1, 0, 0), 0, 1)
    with pytest.raises(NotImplementedError, match="not implemented for CheckMatrix"):
        op.apply_check_matrix(check_matrix_x)


def test_transvection_repr() -> None:
    """Test string representation of transvection."""
    op = Transvection((1, 1, 0, 0), 0, 1)
    repr_str = repr(op)
    assert "Transvection" in repr_str
    assert "{0, 1}" in repr_str or "{1, 0}" in repr_str


# =====================================================================================
# SingleQubitClifford Tests
# =====================================================================================


def test_single_qubit_clifford_available_cliffords_complete() -> None:
    """Test that available_cliffords returns all 10 Cliffords."""
    cliffords = SingleQubitClifford.available_cliffords()
    assert len(cliffords) == 10
    assert set(cliffords) == {"H", "S", "SDAG", "HS", "SH", "HSH", "SDAGH", "HSDAG", "HSDAGH", "I"}


@pytest.mark.parametrize("clifford", ["H", "S", "SDAG", "HS", "SH", "HSH", "SDAGH", "HSDAG", "HSDAGH", "I"])
def test_single_qubit_clifford_inplace_vs_copy(small_tableau: StabilizerTableau, clifford: str) -> None:
    """Test that in-place and copy modes produce equivalent results."""
    original = small_tableau.copy()

    op = SingleQubitClifford(0, clifford)
    result_copy = op.apply_stabilizer_tableau(original, inplace=False)
    result_inplace = op.apply_stabilizer_tableau(small_tableau, inplace=True)

    assert result_inplace is small_tableau
    assert result_copy is not original
    assert result_copy == result_inplace


def test_single_qubit_clifford_qubits() -> None:
    """Test qubits() method."""
    op = SingleQubitClifford(2, "H")
    assert op.qubits() == {2}

    op2 = SingleQubitClifford(5, "S")
    assert op2.qubits() == {5}


def test_single_qubit_clifford_hash() -> None:
    """Test hashing."""
    op1 = SingleQubitClifford(0, "H")
    op2 = SingleQubitClifford(0, "H")
    op3 = SingleQubitClifford(0, "S")
    op4 = SingleQubitClifford(1, "H")

    assert hash(op1) == hash(op2)
    assert hash(op1) != hash(op3)
    assert hash(op1) != hash(op4)


def test_single_qubit_clifford_invalid_raises(small_tableau: StabilizerTableau) -> None:
    """Test that invalid Clifford name raises ValueError."""
    op = SingleQubitClifford(0, "INVALID")
    with pytest.raises(ValueError, match="Unsupported single-qubit Clifford"):
        op.apply_stabilizer_tableau(small_tableau)


def test_single_qubit_clifford_invalid_append_raises() -> None:
    """Test that invalid Clifford name raises ValueError on append."""
    circuit = stim.Circuit()
    op = SingleQubitClifford(0, "INVALID")
    with pytest.raises(ValueError, match="Unsupported single-qubit Clifford"):
        op.append_to_circuit(circuit)


def test_single_qubit_clifford_not_implemented_for_check_matrix(check_matrix_x: CheckMatrix) -> None:
    """Test that SingleQubitClifford raises NotImplementedError for CheckMatrix."""
    op = SingleQubitClifford(0, "H")
    with pytest.raises(NotImplementedError, match="not implemented for CheckMatrix"):
        op.apply_check_matrix(check_matrix_x)


@pytest.mark.parametrize(
    ("clifford", "inverse"),
    [
        ("H", "H"),
        ("S", "SDAG"),
        ("SDAG", "S"),
        ("HS", "SDAGH"),
        ("SH", "HSDAG"),
        ("HSH", "HSDAGH"),
        ("SDAGH", "HS"),
        ("HSDAG", "SH"),
        ("HSDAGH", "HSH"),
        ("I", "I"),
    ],
)
def test_single_qubit_clifford_inverse_correctness(
    small_tableau: StabilizerTableau, clifford: str, inverse: str
) -> None:
    """Test that inverse operations correctly undo the original operation."""
    op = SingleQubitClifford(0, clifford)
    inv_op = op.inverse()

    assert inv_op.clifford == inverse
    assert inv_op.qubit == op.qubit

    original = small_tableau.copy()
    result = op.apply_stabilizer_tableau(original, inplace=False)
    restored = inv_op.apply_stabilizer_tableau(result, inplace=False)

    assert restored == small_tableau


@pytest.mark.parametrize("clifford", ["H", "S", "SDAG", "HS", "SH", "HSH", "SDAGH", "HSDAG", "HSDAGH", "I"])
def test_single_qubit_clifford_apply_inverse(small_tableau: StabilizerTableau, clifford: str) -> None:
    """Test apply_inverse method produces correct results."""
    op = SingleQubitClifford(0, clifford)

    original = small_tableau.copy()
    result = op.apply_stabilizer_tableau(original, inplace=False)
    restored = op.apply_inverse(result, inplace=False)

    assert restored == small_tableau


def test_single_qubit_clifford_apply_inverse_wrong_type(check_matrix_x: CheckMatrix) -> None:
    """Test that apply_inverse raises TypeError for non-StabilizerTableau."""
    op = SingleQubitClifford(0, "H")
    with pytest.raises(TypeError, match="can only be applied to StabilizerTableau"):
        op.apply_inverse(check_matrix_x)


def test_single_qubit_clifford_inverse_invalid_raises() -> None:
    """Test that inverse() with invalid Clifford raises ValueError."""
    op = SingleQubitClifford(0, "INVALID")
    with pytest.raises(ValueError, match="Unsupported single-qubit Clifford"):
        op.inverse()


def test_single_qubit_clifford_apply_inverse_invalid_raises(small_tableau: StabilizerTableau) -> None:
    """Test that apply_inverse with invalid Clifford raises ValueError."""
    op = SingleQubitClifford(0, "INVALID")
    with pytest.raises(ValueError, match="Unsupported single-qubit Clifford"):
        op.apply_inverse(small_tableau)


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        (np.array([[0, 1], [1, 0]], dtype=np.int8), "H"),
        (np.array([[1, 1], [0, 1]], dtype=np.int8), "S"),
        (np.array([[1, 0], [0, 1]], dtype=np.int8), "I"),
        (np.array([[0, 1], [1, 1]], dtype=np.int8), "SH"),
        (np.array([[1, 1], [1, 0]], dtype=np.int8), "HS"),
    ],
)
def test_single_qubit_clifford_from_symplectic_block(block: np.ndarray, expected: str) -> None:
    """Test creation from symplectic block."""
    op = SingleQubitClifford.from_symplectic_block(block, 0)
    assert op.clifford == expected
    assert op.qubit == 0


def test_single_qubit_clifford_from_symplectic_block_invalid_raises() -> None:
    """Test that invalid symplectic block raises ValueError."""
    invalid_block = np.array([[0, 0], [1, 1]], dtype=np.int8)
    with pytest.raises(ValueError, match="Unsupported single-qubit Clifford symplectic block"):
        SingleQubitClifford.from_symplectic_block(invalid_block, 0)


def test_single_qubit_clifford_circuit_length() -> None:
    """Test that generated circuits have expected gate counts."""
    test_cases = {
        "I": 1,
        "H": 1,
        "S": 1,
        "SDAG": 1,
        "HS": 2,
        "SH": 2,
        "HSH": 3,
        "SDAGH": 2,
        "HSDAG": 2,
        "HSDAGH": 3,
    }

    for clifford, expected_len in test_cases.items():
        op = SingleQubitClifford(0, clifford)
        circuit = op.to_stim_circuit()
        assert len(circuit) == expected_len


@pytest.mark.parametrize(
    ("clifford1", "clifford2", "expected"),
    [
        ("S", "SDAG", "I"),
        ("SDAG", "S", "I"),
        ("HS", "SDAGH", "I"),
        ("SDAGH", "HS", "I"),
        ("SH", "HSDAG", "I"),
        ("HSDAG", "SH", "I"),
    ],
)
def test_single_qubit_clifford_inverse_composition(
    small_tableau: StabilizerTableau, clifford1: str, clifford2: str, expected: str
) -> None:
    """Test that applying a Clifford followed by its inverse yields identity."""
    op1 = SingleQubitClifford(0, clifford1)
    op2 = SingleQubitClifford(0, clifford2)

    result = small_tableau.copy()
    op1.apply_stabilizer_tableau(result, inplace=True)
    op2.apply_stabilizer_tableau(result, inplace=True)

    if expected == "I":
        assert result == small_tableau


# =====================================================================================
# PauliOperation Tests
# =====================================================================================


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_pauli_operation_inplace_vs_copy(small_tableau: StabilizerTableau, pauli: str) -> None:
    """Test that in-place and copy modes produce equivalent results."""
    original = small_tableau.copy()

    op = PauliOperation(0, pauli)
    result_copy = op.apply_stabilizer_tableau(original, inplace=False)
    result_inplace = op.apply_stabilizer_tableau(small_tableau, inplace=True)

    assert result_inplace is small_tableau
    assert result_copy is not original
    assert result_copy == result_inplace


def test_pauli_operation_qubits() -> None:
    """Test qubits() method."""
    op = PauliOperation(1, "X")
    assert op.qubits() == {1}

    op2 = PauliOperation(4, "Z")
    assert op2.qubits() == {4}


def test_pauli_operation_hash() -> None:
    """Test hashing."""
    op1 = PauliOperation(0, "X")
    op2 = PauliOperation(0, "X")
    op3 = PauliOperation(0, "Y")
    op4 = PauliOperation(1, "X")

    assert hash(op1) == hash(op2)
    assert hash(op1) != hash(op3)
    assert hash(op1) != hash(op4)


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_pauli_operation_to_circuit(pauli: str) -> None:
    """Test conversion to Stim circuit."""
    op = PauliOperation(0, pauli)
    circuit = op.to_stim_circuit()
    assert isinstance(circuit, stim.Circuit)
    assert len(circuit) == 1


def test_pauli_operation_invalid_raises(small_tableau: StabilizerTableau) -> None:
    """Test that invalid Pauli raises ValueError."""
    op = PauliOperation(0, "INVALID")
    with pytest.raises(ValueError, match="Unsupported Pauli operation"):
        op.apply_stabilizer_tableau(small_tableau)


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_pauli_operation_check_matrix_unchanged(check_matrix_x: CheckMatrix, pauli: str) -> None:
    """Test that Pauli operations don't change check matrix."""
    op = PauliOperation(0, pauli)
    result = op.apply_check_matrix(check_matrix_x, inplace=False)
    assert np.array_equal(result.matrix, check_matrix_x.matrix)


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_pauli_operation_idempotent(small_tableau: StabilizerTableau, pauli: str) -> None:
    """Test that applying a Pauli twice returns to original state."""
    op = PauliOperation(0, pauli)

    original = small_tableau.copy()
    result = op.apply_stabilizer_tableau(original, inplace=False)
    restored = op.apply_stabilizer_tableau(result, inplace=False)

    assert restored == small_tableau


# =====================================================================================
# CNOT Tests
# =====================================================================================


def test_cnot_inplace_vs_copy(small_tableau: StabilizerTableau) -> None:
    """Test that in-place and copy modes produce equivalent results."""
    original = small_tableau.copy()

    op = CNOT(0, 1)
    result_copy = op.apply_stabilizer_tableau(original, inplace=False)
    result_inplace = op.apply_stabilizer_tableau(small_tableau, inplace=True)

    assert result_inplace is small_tableau
    assert result_copy is not original
    assert result_copy == result_inplace


def test_cnot_qubits() -> None:
    """Test qubits() method."""
    op = CNOT(0, 2)
    assert op.qubits() == {0, 2}

    op2 = CNOT(3, 7)
    assert op2.qubits() == {3, 7}


def test_cnot_hash() -> None:
    """Test hashing."""
    op1 = CNOT(0, 1)
    op2 = CNOT(0, 1)
    op3 = CNOT(1, 0)

    assert hash(op1) == hash(op2)
    assert hash(op1) != hash(op3)


def test_cnot_to_circuit() -> None:
    """Test conversion to Stim circuit."""
    op = CNOT(0, 1)
    circuit = op.to_stim_circuit()
    assert isinstance(circuit, stim.Circuit)
    assert len(circuit) == 1


def test_cnot_apply_check_matrix_x_type(check_matrix_x: CheckMatrix) -> None:
    """Test CNOT application to X-type check matrix."""
    op = CNOT(0, 1)
    original = check_matrix_x.matrix.copy()
    result = op.apply_check_matrix(check_matrix_x, inplace=False)
    assert result is not check_matrix_x
    expected = original.copy()
    expected[:, 1] ^= expected[:, 0]
    assert np.array_equal(result.matrix, expected)


def test_cnot_apply_check_matrix_inplace(check_matrix_x: CheckMatrix) -> None:
    """Test CNOT application to check matrix in-place."""
    original_id = id(check_matrix_x)
    original_matrix = check_matrix_x.matrix.copy()

    op = CNOT(0, 1)
    result = op.apply_check_matrix(check_matrix_x, inplace=True)

    assert id(result) == original_id
    expected = original_matrix.copy()
    expected[:, 1] ^= expected[:, 0]
    assert np.array_equal(result.matrix, expected)


def test_cnot_apply_dispatches_correctly(small_tableau: StabilizerTableau, check_matrix_x: CheckMatrix) -> None:
    """Test that apply() dispatches to the correct method based on input type."""
    op = CNOT(0, 1)

    result_tableau = op.apply(small_tableau, inplace=False)
    assert isinstance(result_tableau, StabilizerTableau)

    result_check = op.apply(check_matrix_x, inplace=False)
    assert isinstance(result_check, CheckMatrix)


def test_cnot_idempotent(small_tableau: StabilizerTableau) -> None:
    """Test that applying CNOT twice returns to original state."""
    op = CNOT(0, 1)

    original = small_tableau.copy()
    result = op.apply_stabilizer_tableau(original, inplace=False)
    restored = op.apply_stabilizer_tableau(result, inplace=False)

    assert restored == small_tableau


# =====================================================================================
# Swap Tests
# =====================================================================================


def test_swap_inplace_vs_copy(small_tableau: StabilizerTableau) -> None:
    """Test that in-place and copy modes produce equivalent results."""
    original = small_tableau.copy()

    op = Swap(0, 1)
    result_copy = op.apply_stabilizer_tableau(original, inplace=False)
    result_inplace = op.apply_stabilizer_tableau(small_tableau, inplace=True)

    assert result_inplace is small_tableau
    assert result_copy is not original
    assert result_copy == result_inplace


def test_swap_qubits() -> None:
    """Test qubits() method."""
    op = Swap(0, 2)
    assert op.qubits() == {0, 2}

    op2 = Swap(3, 7)
    assert op2.qubits() == {3, 7}


def test_swap_hash_symmetric() -> None:
    """Test hashing - swap is symmetric so (i,j) and (j,i) should have same hash."""
    op1 = Swap(0, 1)
    op2 = Swap(1, 0)
    op3 = Swap(0, 2)

    assert hash(op1) == hash(op2)
    assert hash(op1) != hash(op3)


def test_swap_to_circuit() -> None:
    """Test conversion to Stim circuit."""
    op = Swap(0, 1)
    circuit = op.to_stim_circuit()
    assert isinstance(circuit, stim.Circuit)
    assert len(circuit) == 1


def test_swap_apply_check_matrix(check_matrix_x: CheckMatrix) -> None:
    """Test SWAP application to check matrix."""
    op = Swap(0, 1)
    original = check_matrix_x.matrix.copy()
    result = op.apply_check_matrix(check_matrix_x, inplace=False)
    assert result is not check_matrix_x
    expected = original.copy()
    expected[:, [0, 1]] = expected[:, [1, 0]]
    assert np.array_equal(result.matrix, expected)


def test_swap_apply_check_matrix_inplace(check_matrix_x: CheckMatrix) -> None:
    """Test SWAP application to check matrix in-place."""
    original_id = id(check_matrix_x)
    original_matrix = check_matrix_x.matrix.copy()

    op = Swap(0, 1)
    result = op.apply_check_matrix(check_matrix_x, inplace=True)

    assert id(result) == original_id
    expected = original_matrix.copy()
    expected[:, [0, 1]] = expected[:, [1, 0]]
    assert np.array_equal(result.matrix, expected)


def test_swap_apply_dispatches_correctly(small_tableau: StabilizerTableau, check_matrix_x: CheckMatrix) -> None:
    """Test that apply() dispatches to the correct method based on input type."""
    op = Swap(0, 1)

    result_tableau = op.apply(small_tableau, inplace=False)
    assert isinstance(result_tableau, StabilizerTableau)

    result_check = op.apply(check_matrix_x, inplace=False)
    assert isinstance(result_check, CheckMatrix)


def test_swap_idempotent(small_tableau: StabilizerTableau) -> None:
    """Test that applying SWAP twice returns to original state."""
    op = Swap(0, 1)

    original = small_tableau.copy()
    result = op.apply_stabilizer_tableau(original, inplace=False)
    restored = op.apply_stabilizer_tableau(result, inplace=False)

    assert restored == small_tableau


def test_swap_commutativity(small_tableau: StabilizerTableau) -> None:
    """Test that Swap(i,j) and Swap(j,i) produce the same result."""
    op1 = Swap(0, 1)
    op2 = Swap(1, 0)

    result1 = op1.apply_stabilizer_tableau(small_tableau.copy(), inplace=False)
    result2 = op2.apply_stabilizer_tableau(small_tableau.copy(), inplace=False)

    assert result1 == result2


# =====================================================================================
# Integration Tests
# =====================================================================================


def test_operations_compose(small_tableau: StabilizerTableau) -> None:
    """Test that operations can be composed sequentially."""
    op1 = SingleQubitClifford(0, "H")
    op2 = CNOT(0, 1)
    op3 = SingleQubitClifford(1, "S")

    result = op1.apply_stabilizer_tableau(small_tableau, inplace=False)
    result = op2.apply_stabilizer_tableau(result, inplace=True)
    result = op3.apply_stabilizer_tableau(result, inplace=True)

    assert isinstance(result, StabilizerTableau)


def test_operations_to_circuit_compose() -> None:
    """Test that operation circuits can be composed."""
    circuit = stim.Circuit()

    ops = [
        SingleQubitClifford(0, "H"),
        CNOT(0, 1),
        PauliOperation(1, "X"),
        Swap(0, 2),
    ]

    for op in ops:
        op.append_to_circuit(circuit)

    assert len(circuit) == 4


def test_repr_methods() -> None:
    """Test __repr__ methods exist and return non-empty strings."""
    ops = [
        Transvection((1, 1, 0, 0), 0, 1),
        SingleQubitClifford(0, "H"),
        PauliOperation(1, "X"),
        CNOT(0, 1),
        Swap(0, 2),
    ]

    for op in ops:
        repr_str = repr(op)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0
        assert op.__class__.__name__ in repr_str


def test_cnot_swap_equivalence(small_tableau: StabilizerTableau) -> None:
    """Test that SWAP can be decomposed into three CNOTs."""
    swap = Swap(0, 1)

    original = small_tableau.copy()
    result_swap = swap.apply_stabilizer_tableau(original, inplace=False)

    result_cnot = small_tableau.copy()
    CNOT(0, 1).apply_stabilizer_tableau(result_cnot, inplace=True)
    CNOT(1, 0).apply_stabilizer_tableau(result_cnot, inplace=True)
    CNOT(0, 1).apply_stabilizer_tableau(result_cnot, inplace=True)

    assert result_swap == result_cnot


def test_clifford_composition(small_tableau: StabilizerTableau) -> None:
    """Test composition of Clifford operations."""
    h_then_s = small_tableau.copy()
    SingleQubitClifford(0, "H").apply_stabilizer_tableau(h_then_s, inplace=True)
    SingleQubitClifford(0, "S").apply_stabilizer_tableau(h_then_s, inplace=True)

    hs = small_tableau.copy()
    SingleQubitClifford(0, "HS").apply_stabilizer_tableau(hs, inplace=True)

    assert h_then_s == hs


def test_pauli_commutation_with_clifford(small_tableau: StabilizerTableau) -> None:
    """Test that Clifford gates transform Paulis correctly."""
    original = small_tableau.copy()

    PauliOperation(0, "X").apply_stabilizer_tableau(original, inplace=True)
    SingleQubitClifford(0, "H").apply_stabilizer_tableau(original, inplace=True)

    expected = small_tableau.copy()
    SingleQubitClifford(0, "H").apply_stabilizer_tableau(expected, inplace=True)
    PauliOperation(0, "Z").apply_stabilizer_tableau(expected, inplace=True)

    assert original == expected


def test_s_sdag_composition(small_tableau: StabilizerTableau) -> None:
    """Test that S followed by S† returns to identity."""
    result = small_tableau.copy()
    SingleQubitClifford(0, "S").apply_stabilizer_tableau(result, inplace=True)
    SingleQubitClifford(0, "SDAG").apply_stabilizer_tableau(result, inplace=True)

    assert result == small_tableau


def test_all_new_cliffords_have_correct_inverses(small_tableau: StabilizerTableau) -> None:
    """Test that all new Clifford gates have correct inverses."""
    new_cliffords = ["SDAG", "SDAGH", "HSDAG", "HSDAGH"]

    for clifford in new_cliffords:
        op = SingleQubitClifford(0, clifford)
        inv_op = op.inverse()

        result = small_tableau.copy()
        op.apply_stabilizer_tableau(result, inplace=True)
        inv_op.apply_stabilizer_tableau(result, inplace=True)

        assert result == small_tableau
