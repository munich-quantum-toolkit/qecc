# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test fault set functionality in the MQT QECC library."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.circuits import CNOTCircuit
from mqt.qecc.circuit_synthesis.faults import PureFaultSet


@pytest.fixture
def stabilizer_matrix():
    """Fixture for a sample stabilizer matrix."""
    return np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)


@pytest.fixture
def empty_stabilizer_matrix():
    """Fixture for an empty stabilizer matrix."""
    return np.zeros((0, 3), dtype=np.int8)


@pytest.fixture
def fault_set():
    """Fixture for a sample fault set."""
    fault_set = PureFaultSet(num_qubits=3)
    fault_set.add_fault(np.array([1, 0, 1], dtype=np.int8))
    fault_set.add_fault(np.array([0, 1, 1], dtype=np.int8))
    fault_set.add_fault(np.array([1, 1, 0], dtype=np.int8))
    return fault_set


def test_add_fault():
    """Test adding faults to the fault set."""
    fault_set = PureFaultSet(num_qubits=3)

    # Add a fault
    fault_set.add_fault(np.array([1, 0, 1], dtype=np.int8))
    assert np.array_equal(fault_set.to_array(), np.array([[1, 0, 1]], dtype=np.int8)), "Fault was not added correctly."

    # Add another fault
    fault_set.add_fault(np.array([0, 1, 0], dtype=np.int8))
    assert np.array_equal(fault_set.to_array(), np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)), (
        "Second fault was not added correctly."
    )


def test_add_fault_invalid_length():
    """Test adding a fault with an invalid length."""
    fault_set = PureFaultSet(num_qubits=3)

    # Attempt to add a fault with incorrect length
    with pytest.raises(ValueError, match=r"Fault must have length 3."):
        fault_set.add_fault(np.array([1, 0], dtype=np.int8))


def test_combine_fault_sets():
    """Test combining two fault sets."""
    fault_set_1 = PureFaultSet(num_qubits=3)
    fault_set_1.add_fault(np.array([1, 0, 1], dtype=np.int8))

    fault_set_2 = PureFaultSet(num_qubits=3)
    fault_set_2.add_fault(np.array([0, 1, 0], dtype=np.int8))

    # Combine the fault sets
    combined_fault_set = fault_set_1.combine(fault_set_2)
    expected = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)
    assert combined_fault_set.to_set() == set(map(tuple, expected)), "Fault sets were not combined correctly."


def test_combine_fault_sets_invalid():
    """Test combining fault sets with different numbers of qubits."""
    fault_set_1 = PureFaultSet(num_qubits=3)
    fault_set_2 = PureFaultSet(num_qubits=4)

    # Attempt to combine fault sets with different numbers of qubits
    with pytest.raises(ValueError, match=r"Fault sets must have the same number of qubits to combine."):
        fault_set_1.combine(fault_set_2)


def test_from_fault_array():
    """Test creating a PureFaultSet from a numpy array."""
    faults = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)
    fault_set = PureFaultSet.from_fault_array(faults)

    # Convert the fault set to an array
    result = fault_set.to_array()

    # Check that the rows in the result match the expected rows, regardless of order
    assert set(map(tuple, result)) == set(map(tuple, faults)), "Fault set was not created correctly from array."


@pytest.mark.parametrize(
    ("stabs_fixture", "initial_faults", "expected_faults"),
    [
        # Test case: Remove equivalent faults
        ("stabilizer_matrix", [[1, 0, 1], [0, 1, 1], [1, 1, 0]], []),
        # Test case: Fault reduced to coset representative
        ("stabilizer_matrix", [[1, 0, 0]], [[0, 0, 1]]),
        # Test case: Empty fault set
        ("stabilizer_matrix", [], []),
        # Test case: Empty stabilizer matrix
        ("empty_stabilizer_matrix", [[1, 0, 1], [0, 1, 0]], [[1, 0, 1], [0, 1, 0]]),
        # Test case: No reduction
        ("stabilizer_matrix", [[0, 0, 1]], [[0, 0, 1]]),
    ],
)
def test_remove_equivalent(request, stabs_fixture, initial_faults, expected_faults):
    """Test removing equivalent faults with respect to a stabilizer group."""
    # Use the fixture dynamically
    stabs = request.getfixturevalue(stabs_fixture)

    # Initialize the fault set
    fault_set = PureFaultSet(num_qubits=3)
    for fault in initial_faults:
        fault_set.add_fault(np.array(fault, dtype=np.int8))

    # Remove equivalent faults
    fault_set.remove_equivalent(stabs)

    # Check the result
    assert fault_set.to_set() == set(map(tuple, expected_faults)), (
        "Fault set was not reduced to unique coset representatives correctly."
    )


def test_from_cnot_circuit_x_faults():
    """Test generating X-type faults from a CNOT circuit."""
    # Create a CNOT circuit
    circ = CNOTCircuit()
    circ.add_cnot(0, 1)
    circ.add_cnot(1, 2)

    # Generate the fault set
    fault_set = PureFaultSet.from_cnot_circuit(circ, kind="X")

    # Expected faults
    expected_faults = np.array(
        [
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.int8,
    )

    # Check the result
    assert fault_set.to_set() == set(map(tuple, expected_faults)), (
        "X-type faults were not generated correctly from the CNOT circuit."
    )


def test_from_cnot_circuit_z_faults():
    """Test generating Z-type faults from a CNOT circuit."""
    # Create a CNOT circuit
    circ = CNOTCircuit()
    circ.add_cnot(0, 1)
    circ.add_cnot(1, 2)

    # Generate the fault set
    fault_set = PureFaultSet.from_cnot_circuit(circ, kind="Z")

    # Expected faults
    expected_faults = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.int8,
    )

    # Check the result
    assert fault_set.to_set() == set(map(tuple, expected_faults)), (
        "Z-type faults were not generated correctly from the CNOT circuit."
    )


def test_from_cnot_circuit_invalid_kind():
    """Test that an invalid fault kind raises an assertion error."""
    # Create a CNOT circuit
    circ = CNOTCircuit()
    circ.add_cnot(0, 1)

    # Attempt to generate faults with an invalid kind
    with pytest.raises(AssertionError, match=r"Kind must be either 'X' or 'Z'."):
        PureFaultSet.from_cnot_circuit(circ, kind="Y")
