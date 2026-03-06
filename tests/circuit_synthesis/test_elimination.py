# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for the elimination module in circuit synthesis."""

import numpy as np
import pytest
import stim

from mqt.qecc.circuit_synthesis.elimination import EliminationSequence
from mqt.qecc.circuit_synthesis.operations import CNOT, Swap, Transvection
from mqt.qecc.codes.pauli import CheckMatrix, StabilizerTableau


@pytest.fixture
def simple_check_matrix() -> CheckMatrix:
    """Fixture to create a simple CSS check matrix."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
    return CheckMatrix(matrix, pauli_type="X")


def test_check_matrix_fixture(simple_check_matrix: CheckMatrix) -> None:
    """Test that the check matrix fixture is created correctly."""
    assert simple_check_matrix.num_qubits() == 3
    assert simple_check_matrix.num_rows() == 2
    assert simple_check_matrix.type == "X"


def test_elimination_sequence_empty() -> None:
    """Test that an empty elimination sequence is created correctly."""
    seq = EliminationSequence([])
    assert len(seq.operations) == 0
    assert seq.num_two_qubit_gates() == 0
    assert seq.num_transvections() == 0
    assert seq.num_cnots() == 0
    assert seq.depth() == 0


def test_elimination_sequence_single_cnot() -> None:
    """Test elimination sequence with a single CNOT."""
    cnot = CNOT(0, 1)
    seq = EliminationSequence([cnot])

    assert len(seq.operations) == 1
    assert seq.num_two_qubit_gates() == 1
    assert seq.num_cnots() == 1
    assert seq.num_transvections() == 0
    assert seq.depth() == 1


def test_elimination_sequence_multiple_operations() -> None:
    """Test elimination sequence with multiple operations."""
    ops = [
        CNOT(0, 1),
        CNOT(1, 2),
        Swap(0, 2),
        CNOT(2, 1),
    ]
    seq = EliminationSequence(ops)

    assert len(seq.operations) == 4
    assert seq.num_two_qubit_gates() == 4
    assert seq.num_cnots() == 3
    assert seq.num_transvections() == 0


def test_elimination_sequence_with_transvections() -> None:
    """Test elimination sequence with transvections."""
    ops = [
        Transvection((0, 1, 0, 1), 0, 1),
        Transvection((1, 1, 0, 1), 1, 2),
        CNOT(0, 2),
    ]
    seq = EliminationSequence(ops)

    assert seq.num_two_qubit_gates() == 3
    assert seq.num_transvections() == 2
    assert seq.num_cnots() == 1


def test_elimination_sequence_to_circuit() -> None:
    """Test converting elimination sequence to Stim circuit."""
    ops = [
        CNOT(0, 1),
        CNOT(1, 2),
    ]
    seq = EliminationSequence(ops)

    circuit = seq.to_circuit()
    assert isinstance(circuit, stim.Circuit)
    assert circuit == stim.Circuit("CX 0 1\nCX 1 2\n")


def test_elimination_sequence_to_circuit_inverse() -> None:
    """Test converting elimination sequence to inverse Stim circuit."""
    ops = [
        CNOT(0, 1),
        CNOT(1, 2),
    ]
    seq = EliminationSequence(ops)

    circuit = seq.to_circuit()
    inverse_circuit = seq.to_circuit_inverse()

    assert isinstance(inverse_circuit, stim.Circuit)
    assert len(inverse_circuit) == len(circuit)


def test_elimination_sequence_add_operation() -> None:
    """Test adding operations to elimination sequence."""
    seq = EliminationSequence([])

    assert seq.depth() == 0

    seq.add_operation(CNOT(0, 1))
    assert len(seq.operations) == 1
    assert seq.depth() == 1

    seq.add_operation(CNOT(2, 3))
    assert len(seq.operations) == 2
    assert seq.depth() == 1

    seq.add_operation(CNOT(0, 2))
    assert len(seq.operations) == 3
    assert seq.depth() == 2


def test_elimination_sequence_depth_calculation() -> None:
    """Test that depth is correctly calculated with parallel operations."""
    seq = EliminationSequence([])

    seq.add_operation(CNOT(0, 1))
    seq.add_operation(CNOT(2, 3))
    assert seq.depth() == 1

    seq.add_operation(CNOT(1, 2))
    assert seq.depth() == 2

    seq.add_operation(CNOT(4, 5))
    assert seq.depth() == 2


def test_elimination_sequence_apply() -> None:
    """Test applying elimination sequence to a tableau."""
    tableau = StabilizerTableau.from_pauli_strings(["ZZ", "XX"])
    ops = [CNOT(0, 1)]
    seq = EliminationSequence(ops)

    result = seq.apply(tableau, inplace=False)

    assert isinstance(result, StabilizerTableau)
    assert result.n == 2
    assert tableau != result


def test_elimination_sequence_apply_inplace() -> None:
    """Test applying elimination sequence in-place."""
    tableau = StabilizerTableau.from_pauli_strings(["ZZ", "XX"])
    original_matrix = tableau.tableau.matrix.copy()

    ops = [CNOT(0, 1)]
    seq = EliminationSequence(ops)

    result = seq.apply(tableau, inplace=True)

    assert result is tableau
    assert isinstance(result, StabilizerTableau)
    assert not np.array_equal(result.tableau.matrix, original_matrix)


def test_elimination_sequence_extend() -> None:
    """Test extending elimination sequence with another sequence."""
    seq1 = EliminationSequence([CNOT(0, 1)])
    seq2 = EliminationSequence([CNOT(1, 2), CNOT(2, 3)])

    seq1.extend(seq2)

    assert len(seq1.operations) == 3
    assert seq1.num_cnots() == 3


def test_elimination_sequence_iter() -> None:
    """Test iterating over elimination sequence."""
    ops = [CNOT(0, 1), CNOT(1, 2), CNOT(2, 3)]
    seq = EliminationSequence(ops)

    collected = list(seq)
    assert len(collected) == 3
    assert collected[0] == ops[0]
    assert collected[1] == ops[1]
    assert collected[2] == ops[2]


def test_elimination_sequence_reversed() -> None:
    """Test reversed iteration over elimination sequence."""
    ops = [CNOT(0, 1), CNOT(1, 2), CNOT(2, 3)]
    seq = EliminationSequence(ops)

    collected = list(reversed(seq))
    assert len(collected) == 3
    assert collected[0] == ops[2]
    assert collected[1] == ops[1]
    assert collected[2] == ops[0]


def test_elimination_sequence_depth_with_swaps() -> None:
    """Test depth calculation with swap operations."""
    seq = EliminationSequence([])

    seq.add_operation(Swap(0, 1))
    assert seq.depth() == 1

    seq.add_operation(Swap(2, 3))
    assert seq.depth() == 1

    seq.add_operation(Swap(1, 2))
    assert seq.depth() == 2
