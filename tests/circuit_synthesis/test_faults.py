# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
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
from mqt.qecc.circuit_synthesis.faults import PureFaultSet, XZFaultList, coset_leader, stabilizer_equivalent, t_distinct


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


def test_from_fault_array_invalid_dimension():
    """Test creating a PureFaultSet from an array with invalid dimensions."""
    faults = np.array([1, 0, 1], dtype=np.int8)  # 1D array instead of 2D

    with pytest.raises(ValueError, match=r"Input array must be 2-dimensional."):
        PureFaultSet.from_fault_array(faults)


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


def test_coset_leader_no_generators():
    """Test coset leader computation when no stabilizer generators are provided."""
    fault = np.array([1, 0, 1], dtype=np.int8)
    generators = np.zeros((0, 3), dtype=np.int8)  # No generators

    # Compute the coset leader
    leader = coset_leader(fault, generators)

    # Expected result: the fault itself
    expected = fault
    assert np.array_equal(leader, expected), "Coset leader should be the fault itself when no generators are provided."


def test_coset_leader_single_generator():
    """Test coset leader computation with a single stabilizer generator."""
    fault = np.array([1, 0, 1], dtype=np.int8)
    generators = np.array([[1, 0, 1]], dtype=np.int8)  # Single generator

    # Compute the coset leader
    leader = coset_leader(fault, generators)

    # Expected result: the zero vector (fault is in the stabilizer group)
    expected = np.array([0, 0, 0], dtype=np.int8)
    assert np.array_equal(leader, expected), (
        "Coset leader should be the zero vector when fault is in the stabilizer group."
    )


def test_coset_leader_multiple_generators():
    """Test coset leader computation with multiple stabilizer generators."""
    fault = np.array([1, 1, 0], dtype=np.int8)
    generators = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )  # Two generators

    # Compute the coset leader
    leader = coset_leader(fault, generators)

    # Expected result: the minimal weight representative
    expected = np.array([0, 0, 0], dtype=np.int8)  # Minimal weight representative
    assert np.array_equal(leader, expected), "Coset leader computation failed for multiple generators."


def test_coset_leader_fault_not_in_stabilizer():
    """Test coset leader computation when the fault is not in the stabilizer group."""
    fault = np.array([1, 1, 1], dtype=np.int8)
    generators = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=np.int8,
    )  # Two generators

    # Compute the coset leader
    leader = coset_leader(fault, generators)

    # Expected result: the minimal weight representative
    expected = np.array([0, 0, 1], dtype=np.int8)  # Minimal weight representative
    assert np.array_equal(leader, expected), "Coset leader computation failed for a fault not in the stabilizer group."


def test_filter_by_weight_basic():
    """Test filtering faults by weight with a simple stabilizer group."""
    stabs = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)  # Stabilizer group
    fault_set = PureFaultSet(num_qubits=3)
    fault_set.add_fault(np.array([1, 0, 1], dtype=np.int8))
    fault_set.add_fault(np.array([0, 1, 1], dtype=np.int8))
    fault_set.add_fault(np.array([1, 1, 0], dtype=np.int8))
    fault_set.add_fault(np.array([1, 1, 1], dtype=np.int8))

    # Filter faults with weight >= 2
    fault_set.filter_by_weight_at_least(2, stabs)

    # Expected faults after filtering
    expected_faults = PureFaultSet(3)

    assert fault_set == expected_faults, "Faults were not filtered correctly by weight."


def test_filter_by_weight_empty_stabilizer():
    """Test filtering with an empty stabilizer group."""
    stabs = np.zeros((0, 3), dtype=np.int8)  # Empty stabilizer group
    fault_set = PureFaultSet(num_qubits=3)
    fault_set.add_fault(np.array([1, 0, 1], dtype=np.int8))
    fault_set.add_fault(np.array([0, 1, 1], dtype=np.int8))

    # Filter faults with weight >= 2
    fault_set.filter_by_weight_at_least(2, stabs)

    # Expected faults after filtering
    expected_faults = PureFaultSet.from_fault_array(
        np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
            ],
            dtype=np.int8,
        )
    )

    assert fault_set == expected_faults, "Faults should remain unchanged when the stabilizer group is empty."


def test_filter_by_weight_complex():
    """Test filtering by weight with a complex stabilizer group."""
    hx = np.array([[1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 1, 1]], dtype=np.int8)

    fault_set = PureFaultSet(num_qubits=7)
    fault_set.add_fault(np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int8))
    fault_set.add_fault(np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.int8))
    fault_set.add_fault(np.array([0, 1, 1, 1, 0, 1, 1], dtype=np.int8))
    fault_set.add_fault(np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int8))

    fault_set.filter_by_weight_at_least(2, hx)

    expected_faults = PureFaultSet(num_qubits=7)
    expected_faults.add_fault(np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.int8))
    expected_faults.add_fault(np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.int8))
    expected_faults.add_fault(np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int8))

    assert stabilizer_equivalent(fault_set, expected_faults, hx), (
        "Faults were not filtered correctly by weight with a complex stabilizer group."
    )


def test_stabilizer_equivalent_identical_fault_sets():
    """Test equivalence of two identical fault sets."""
    stabs = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)  # Stabilizer group
    fault_set_1 = PureFaultSet(num_qubits=3)
    fault_set_1.add_fault(np.array([1, 0, 1], dtype=np.int8))
    fault_set_1.add_fault(np.array([0, 1, 1], dtype=np.int8))

    fault_set_2 = PureFaultSet(num_qubits=3)
    fault_set_2.add_fault(np.array([1, 0, 1], dtype=np.int8))
    fault_set_2.add_fault(np.array([0, 1, 1], dtype=np.int8))

    # Check equivalence
    assert stabilizer_equivalent(fault_set_1, fault_set_2, stabs), "Identical fault sets should be equivalent."


def test_stabilizer_equivalent_different_fault_sets():
    """Test non-equivalence of two different fault sets."""
    stabs = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)  # Stabilizer group
    fault_set_1 = PureFaultSet(num_qubits=3)
    fault_set_1.add_fault(np.array([1, 0, 0], dtype=np.int8))

    fault_set_2 = PureFaultSet(num_qubits=3)
    fault_set_2.add_fault(np.array([0, 1, 1], dtype=np.int8))

    # Check equivalence
    assert not stabilizer_equivalent(fault_set_1, fault_set_2, stabs), "Different fault sets should not be equivalent."


def test_stabilizer_equivalent_equivalent_fault_sets():
    """Test equivalence of two fault sets that are equivalent under the stabilizer group."""
    stabs = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)  # Stabilizer group
    fault_set_1 = PureFaultSet(num_qubits=3)
    fault_set_1.add_fault(np.array([1, 0, 1], dtype=np.int8))

    fault_set_2 = PureFaultSet(num_qubits=3)
    fault_set_2.add_fault(np.array([0, 0, 0], dtype=np.int8))  # Equivalent under stabilizer group

    # Check equivalence
    assert stabilizer_equivalent(fault_set_1, fault_set_2, stabs), (
        "Fault sets equivalent under the stabilizer group should be equivalent."
    )


def test_stabilizer_equivalent_different_num_qubits():
    """Test that fault sets with different numbers of qubits raise an error."""
    stabs = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)  # Stabilizer group
    fault_set_1 = PureFaultSet(num_qubits=3)
    fault_set_1.add_fault(np.array([1, 0, 1], dtype=np.int8))

    fault_set_2 = PureFaultSet(num_qubits=4)
    fault_set_2.add_fault(np.array([1, 0, 1, 0], dtype=np.int8))

    # Check for ValueError
    with pytest.raises(ValueError, match=r"Fault sets must have the same number of qubits to compare."):
        stabilizer_equivalent(fault_set_1, fault_set_2, stabs)


def test_all_faults_detected():
    """Test whether all faults are detected by the stabilizers."""
    stabs = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)  # Stabilizer matrix
    fault_set = PureFaultSet(num_qubits=3)
    fault_set.add_fault(np.array([1, 0, 1], dtype=np.int8))  # Detectable
    fault_set.add_fault(np.array([0, 1, 1], dtype=np.int8))  # Detectable

    # Check if all faults are detected
    assert fault_set.all_faults_detected(stabs), "All faults should be detected by the stabilizers."


def test_not_all_faults_detected():
    """Test when not all faults are detected by the stabilizers."""
    stabs = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)  # Stabilizer matrix
    fault_set = PureFaultSet(num_qubits=3)
    fault_set.add_fault(np.array([1, 0, 0], dtype=np.int8))  # Undetectable
    fault_set.add_fault(np.array([0, 1, 1], dtype=np.int8))  # Detectable

    # Check if all faults are detected
    assert not fault_set.all_faults_detected(stabs), "Not all faults should be detected by the stabilizers."


def test_get_undetectable_faults():
    """Test retrieving undetectable faults."""
    stabs = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)  # Stabilizer matrix
    fault_set = PureFaultSet(num_qubits=3)
    fault_set.add_fault(np.array([1, 0, 0], dtype=np.int8))  # Detectable
    fault_set.add_fault(np.array([1, 1, 1], dtype=np.int8))  # Undetectable

    # Get undetectable faults
    undetectable_faults = fault_set.get_undetectable_faults(stabs)
    expected_faults = np.array([[1, 1, 1]], dtype=np.int8)
    assert np.array_equal(undetectable_faults, expected_faults), "The undetectable faults were not retrieved correctly."


def test_remove_undetectable_faults():
    """Test removing undetectable faults."""
    stabs = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)  # Stabilizer matrix
    fault_set = PureFaultSet(num_qubits=3)
    fault_set.add_fault(np.array([1, 0, 0], dtype=np.int8))  # Detectable
    fault_set.add_fault(np.array([1, 1, 1], dtype=np.int8))  # Undetectable

    # Remove undetectable faults
    fault_set.remove_undetectable_faults(stabs)

    # Expected faults after removal
    expected_faults = np.array([[1, 0, 0]], dtype=np.int8)
    assert np.array_equal(fault_set.to_array(), expected_faults), "Undetectable faults were not removed correctly."


@pytest.mark.parametrize(
    (
        "faults",
        "expected_all_detected",
        "expected_undetectable_indices",
        "expected_undetectable_faults",
        "expected_remaining_faults",
    ),
    [
        # Case 1: All faults are detectable
        (
            [[1, 0, 1], [0, 1, 1]],  # Faults
            True,  # All faults detected
            [],  # No undetectable faults
            np.empty(shape=(0, 3), dtype=np.int8),  # No undetectable faults
            [[1, 0, 1], [0, 1, 1]],  # Remaining faults
        ),
        # Case 2: Not all faults are detectable
        (
            [[1, 0, 0], [1, 1, 1]],  # Faults
            False,  # Not all faults detected
            [1],  # Index of undetectable fault
            [[1, 1, 1]],  # Undetectable fault
            [[1, 0, 0]],  # Remaining faults
        ),
        # Case 3: Multiple undetectable faults
        (
            [[0, 0, 0], [1, 1, 1]],  # Faults
            False,  # All faults detected
            [0, 1],  # Indices of undetectable faults
            [[0, 0, 0], [1, 1, 1]],  # Undetectable faults
            np.empty(shape=(0, 3), dtype=np.int8),  # No remaining faults
        ),
    ],
)
def test_fault_detection_and_removal(
    stabilizer_matrix,
    faults,
    expected_all_detected,
    expected_undetectable_indices,
    expected_undetectable_faults,
    expected_remaining_faults,
):
    """Unified test for fault detection and removal methods."""
    # Initialize the fault set
    fault_set = PureFaultSet(num_qubits=3)
    for fault in faults:
        fault_set.add_fault(np.array(fault, dtype=np.int8))

    # Test all_faults_detected
    assert fault_set.all_faults_detected(stabilizer_matrix) == expected_all_detected, (
        "Fault detection result is incorrect."
    )

    # Test _get_undetectable_faults_idx
    undetectable_indices = fault_set.get_undetectable_faults_idx(stabilizer_matrix)
    assert np.array_equal(undetectable_indices, expected_undetectable_indices), (
        "Undetectable fault indices are incorrect."
    )

    # Test get_undetectable_faults
    undetectable_faults = fault_set.get_undetectable_faults(stabilizer_matrix)
    assert np.array_equal(undetectable_faults, np.array(expected_undetectable_faults, dtype=np.int8)), (
        "Undetectable faults are incorrect."
    )

    # Test remove_undetectable_faults
    fault_set.remove_undetectable_faults(stabilizer_matrix)
    assert np.array_equal(fault_set.to_array(), np.array(expected_remaining_faults, dtype=np.int8)), (
        "Remaining faults after removal are incorrect."
    )


def test_filter_faults_weight_threshold():
    """Test filtering faults based on a weight threshold."""
    # Create a fault set
    fault_set = PureFaultSet(num_qubits=3)
    fault_set.add_fault(np.array([1, 0, 1], dtype=np.int8))  # Weight = 2
    fault_set.add_fault(np.array([0, 1, 1], dtype=np.int8))  # Weight = 2
    fault_set.add_fault(np.array([1, 1, 0], dtype=np.int8))  # Weight = 2
    fault_set.add_fault(np.array([0, 0, 1], dtype=np.int8))  # Weight = 1

    # Define a predicate to filter faults with weight >= 2
    def weight_at_least_2(fault: np.ndarray) -> bool:
        return bool(np.sum(fault) >= 2)

    # Apply the filter
    fault_set.filter_faults(weight_at_least_2)

    # Expected faults after filtering
    expected_faults = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=np.int8,
    )

    # Check the result
    assert np.array_equal(fault_set.to_array(), expected_faults), (
        "Faults with weight < 2 were not filtered out correctly."
    )


def test_filter_faults_no_match():
    """Test filtering when no faults satisfy the predicate."""
    # Create a fault set
    fault_set = PureFaultSet(num_qubits=3)
    fault_set.add_fault(np.array([1, 0, 1], dtype=np.int8))  # Weight = 2
    fault_set.add_fault(np.array([0, 1, 1], dtype=np.int8))  # Weight = 2

    # Define a predicate that no fault satisfies
    def always_false(fault: np.ndarray) -> bool:  # noqa: ARG001
        return False

    # Apply the filter
    fault_set.filter_faults(always_false)

    # Check the result
    assert fault_set.to_array().size == 0, "Fault set should be empty when no faults satisfy the predicate."


def test_filter_faults_all_match():
    """Test filtering when all faults satisfy the predicate."""
    # Create a fault set
    fault_set = PureFaultSet(num_qubits=3)
    fault_set.add_fault(np.array([1, 0, 1], dtype=np.int8))  # Weight = 2
    fault_set.add_fault(np.array([0, 1, 1], dtype=np.int8))  # Weight = 2

    # Define a predicate that all faults satisfy
    def always_true(fault: np.ndarray) -> bool:  # noqa: ARG001
        return True

    # Apply the filter
    fault_set.filter_faults(always_true)

    # Expected faults after filtering
    expected_faults = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )

    # Check the result
    assert np.array_equal(fault_set.to_array(), expected_faults), (
        "All faults should remain when all satisfy the predicate."
    )


def test_t_distinct_basic():
    """Test t-distinctness of two fault sets."""
    fs1 = PureFaultSet.from_fault_array(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int8))
    fs2 = PureFaultSet.from_fault_array(np.array([[0, 0, 1], [1, 1, 0]], dtype=np.int8))
    t = 2

    assert t_distinct(fs1, fs2, t) is True, "fs1 and fs2 should be t-distinct"


def test_t_distinct_four_qubits():
    """Test t-distinctness of two fault sets with four qubits with respect to stabilizers."""
    fs1 = PureFaultSet.from_fault_array(np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8))
    fs2 = PureFaultSet.from_fault_array(np.array([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.int8))
    stabs = np.array([[1, 1, 1, 1]], dtype=np.int8)
    t = 4

    assert t_distinct(fs1, fs2, t, stabs) is True, "fs1 and fs2 should be 4-distinct"


def test_not_t_distinct_four_qubits():
    """Test that two fault sets are not t-distinct."""
    fs1 = PureFaultSet.from_fault_array(np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8))
    fs2 = PureFaultSet.from_fault_array(np.array([[1, 1, 1, 1]], dtype=np.int8))
    t = 4

    assert t_distinct(fs1, fs2, t) is False, "fs1 and fs2 should be 4-distinct"


def test_permute_qubits_basic():
    """Test basic permutation of faults."""
    faults = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
    fault_set = PureFaultSet.from_fault_array(faults)
    permutation = [2, 0, 1]

    permuted_fault_set = fault_set.permute_qubits(permutation, inplace=False)

    assert np.array_equal(permuted_fault_set.faults, faults[:, permutation]), "Faults were not permuted correctly"
    assert fault_set == PureFaultSet.from_fault_array(faults), "Original fault set should remain unchanged"


def test_permute_qubits_inplace():
    """Test inplace permutation of fault set."""
    faults = np.array([[1, 1, 0], [0, 0, 1]], dtype=np.int8)
    fault_set = PureFaultSet.from_fault_array(faults)
    permutation = [2, 0, 1]

    fault_set.permute_qubits(permutation, inplace=True)

    assert fault_set != PureFaultSet.from_fault_array(faults), "Faults were not permuted correctly in place"


def test_invalid_fault_kind():
    """Test that an invalid kind raises an assertion error."""
    with pytest.raises(AssertionError, match=r"Kind must be either 'X' or 'Z'."):
        pfs = PureFaultSet(5, kind="Y")

    with pytest.raises(AssertionError, match=r"Kind must be either 'X' or 'Z'."):
        pfs = PureFaultSet.from_fault_array(np.array([[1, 0, 1]], dtype=np.int8), kind="Y")

    pfs = PureFaultSet(5)
    with pytest.raises(AssertionError, match=r"Kind must be either 'X' or 'Z'."):
        pfs.kind = "Y"


def test_apply_cnot_x():
    """Test applying a CNOT gate to the fault set."""
    faults1 = np.array([[1, 0, 0]], dtype=np.int8)
    fault_set1 = PureFaultSet.from_fault_array(faults1, kind="X")

    # Apply CNOT with control=0 and target=1
    fault_set1.apply_cnot(control=0, target=1)

    expected_faults1 = np.array([[1, 1, 0]], dtype=np.int8)
    assert np.array_equal(fault_set1.to_array(), expected_faults1), (
        "CNOT gate was not applied correctly to the fault set"
    )

    faults2 = np.array([[0, 1, 0]], dtype=np.int8)
    fault_set2 = PureFaultSet.from_fault_array(faults2, kind="X")

    # Apply CNOT with control=0 and target=1
    fault_set2.apply_cnot(control=0, target=1)

    expected_faults2 = np.array([[0, 1, 0]], dtype=np.int8)
    assert np.array_equal(fault_set2.to_array(), expected_faults2), (
        "CNOT gate was not applied correctly to the fault set"
    )


def test_apply_cnot_z():
    """Test applying a CNOT gate to the fault set."""
    faults1 = np.array([[1, 0, 0]], dtype=np.int8)
    fault_set1 = PureFaultSet.from_fault_array(faults1, kind="Z")

    # Apply CNOT with control=0 and target=1
    fault_set1.apply_cnot(control=0, target=1)

    expected_faults1 = np.array([[1, 0, 0]], dtype=np.int8)
    assert np.array_equal(fault_set1.to_array(), expected_faults1), (
        "CNOT gate was not applied correctly to the fault set"
    )

    faults2 = np.array([[0, 1, 0]], dtype=np.int8)
    fault_set2 = PureFaultSet.from_fault_array(faults2, kind="Z")

    # Apply CNOT with control=0 and target=1
    fault_set2.apply_cnot(control=0, target=1)

    expected_faults2 = np.array([[1, 1, 0]], dtype=np.int8)
    assert np.array_equal(fault_set2.to_array(), expected_faults2), (
        "CNOT gate was not applied correctly to the fault set"
    )


def test_apply_cnot_invalid_qubits():
    """Test that applying a CNOT gate with invalid qubit indices raises an error."""
    faults = np.array([[1, 0, 0]], dtype=np.int8)
    fault_set = PureFaultSet.from_fault_array(faults)

    with pytest.raises(ValueError, match=r"Control and target qubits must be different."):
        fault_set.apply_cnot(control=0, target=0)

    with pytest.raises(ValueError, match=r"Control and target indices must be between 0 and 2."):
        fault_set.apply_cnot(control=3, target=1)


def test_apply_cnot_not_inplace():
    """Test that applying a CNOT gate does not modify the original fault set when inplace=False."""
    faults = np.array([[1, 0, 0]], dtype=np.int8)
    fault_set = PureFaultSet.from_fault_array(faults)

    # Apply CNOT with control=0 and target=1 without modifying the original fault set
    new_fault_set = fault_set.apply_cnot(control=0, target=1, inplace=False)

    expected_new_faults = np.array([[1, 1, 0]], dtype=np.int8)
    assert np.array_equal(new_fault_set.to_array(), expected_new_faults), (
        "CNOT gate was not applied correctly to the new fault set"
    )
    assert np.array_equal(fault_set.to_array(), faults), "Original fault set should remain unchanged"


"""XZFaultList Tests"""


@pytest.fixture
def fault_list() -> XZFaultList:
    faults = XZFaultList(num_qubits=3)
    faults.add_fault((np.array([1, 0, 1], dtype=np.int8), np.array([0, 1, 0], dtype=np.int8)))
    faults.add_fault((np.array([0, 1, 0], dtype=np.int8), np.array([1, 0, 1], dtype=np.int8)))
    return faults


def test_initialization_creates_empty_fault_arrays() -> None:
    faults = XZFaultList(num_qubits=4)

    assert faults.num_qubits == 4
    assert np.array_equal(faults.faults["X"], np.zeros((0, 4), dtype=np.int8))
    assert np.array_equal(faults.faults["Z"], np.zeros((0, 4), dtype=np.int8))


def test_add_fault_appends_x_and_z_rows() -> None:
    faults = XZFaultList(num_qubits=3)

    faults.add_fault((np.array([1, 0, 1], dtype=np.int8), np.array([0, 1, 0], dtype=np.int8)))

    assert np.array_equal(faults.faults["X"], np.array([[1, 0, 1]], dtype=np.int8))
    assert np.array_equal(faults.faults["Z"], np.array([[0, 1, 0]], dtype=np.int8))


def test_add_fault_rejects_wrong_length() -> None:
    faults = XZFaultList(num_qubits=3)

    with pytest.raises(ValueError, match=r"Faults must have length 3."):
        faults.add_fault((np.array([1, 0], dtype=np.int8), np.array([0, 1, 0], dtype=np.int8)))


def test_add_fault_replaces_none_with_zeros() -> None:
    faults = XZFaultList(num_qubits=3)

    faults.add_fault((None, np.array([0, 1, 0], dtype=np.int8)))
    faults.add_fault((np.array([1, 0, 1], dtype=np.int8), None))

    assert np.array_equal(faults.faults["X"], np.array([[0, 0, 0], [1, 0, 1]], dtype=np.int8))
    assert np.array_equal(faults.faults["Z"], np.array([[0, 1, 0], [0, 0, 0]], dtype=np.int8))


def test_add_fault_rejects_both_none() -> None:
    faults = XZFaultList(num_qubits=3)

    with pytest.raises(ValueError, match=r"At least one fault must be provided."):
        faults.add_fault((None, None))


def test_add_faults_appends_multiple_rows() -> None:
    faults = XZFaultList(num_qubits=3)

    faults.add_faults((
        np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int8),
        np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8),
    ))

    assert np.array_equal(faults.faults["X"], np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int8))
    assert np.array_equal(faults.faults["Z"], np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8))


def test_add_faults_rejects_wrong_column_count() -> None:
    faults = XZFaultList(num_qubits=3)

    with pytest.raises(ValueError, match=r"Faults must have 3 columns."):
        faults.add_faults((
            np.array([[1, 0]], dtype=np.int8),
            np.array([[0, 1]], dtype=np.int8),
        ))


def test_add_faults_replaces_none_with_zeros() -> None:
    faults = XZFaultList(num_qubits=3)

    faults.add_faults((
        None,
        np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8),
    ))
    faults.add_faults((
        np.array([[1, 0, 1]], dtype=np.int8),
        None,
    ))

    assert np.array_equal(
        faults.faults["X"],
        np.array([[0, 0, 0], [0, 0, 0], [1, 0, 1]], dtype=np.int8),
    )
    assert np.array_equal(
        faults.faults["Z"],
        np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=np.int8),
    )


def test_add_faults_rejects_both_none() -> None:
    faults = XZFaultList(num_qubits=3)

    with pytest.raises(ValueError, match=r"At least one fault array must be provided."):
        faults.add_faults((None, None))


def test_copy_returns_independent_fault_list(fault_list: XZFaultList) -> None:
    copied = fault_list.copy()

    copied.faults["X"][0, 0] = 0
    copied.faults["Z"][1, 2] = 0

    assert np.array_equal(fault_list.faults["X"], np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8))
    assert np.array_equal(fault_list.faults["Z"], np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8))


def test_iter_yields_fault_pairs_in_order(fault_list: XZFaultList) -> None:
    pairs = list(fault_list)

    assert len(pairs) == 2
    assert np.array_equal(pairs[0][0], np.array([1, 0, 1], dtype=np.int8))
    assert np.array_equal(pairs[0][1], np.array([0, 1, 0], dtype=np.int8))
    assert np.array_equal(pairs[1][0], np.array([0, 1, 0], dtype=np.int8))
    assert np.array_equal(pairs[1][1], np.array([1, 0, 1], dtype=np.int8))


def test_apply_cnot_updates_x_and_z_faults(fault_list: XZFaultList) -> None:
    updated = fault_list.apply_cnot(control=0, target=1, inplace=False)

    expected_x = np.array([[1, 1, 1], [0, 1, 0]], dtype=np.int8)
    expected_z = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.int8)

    assert np.array_equal(updated.faults["X"], expected_x)
    assert np.array_equal(updated.faults["Z"], expected_z)
    assert np.array_equal(fault_list.faults["X"], np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8))
    assert np.array_equal(fault_list.faults["Z"], np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8))


def test_apply_cnot_inplace_modifies_current_fault_list(fault_list: XZFaultList) -> None:
    result = fault_list.apply_cnot(control=1, target=2, inplace=True)

    expected_x = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)
    expected_z = np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int8)

    assert result is fault_list
    assert np.array_equal(fault_list.faults["X"], expected_x)
    assert np.array_equal(fault_list.faults["Z"], expected_z)


def test_apply_cnot_rejects_invalid_qubits(fault_list: XZFaultList) -> None:
    with pytest.raises(ValueError, match=r"All qubits must be different."):
        fault_list.apply_cnot(control=1, target=1)

    with pytest.raises(ValueError, match=r"Qubit indices must be between 0 and 2."):
        fault_list.apply_cnot(control=3, target=1)


def test_apply_hadamard_swaps_x_and_z_on_target_qubit(fault_list: XZFaultList) -> None:
    updated = fault_list.apply_hadamard(qubit=1, inplace=False)

    expected_x = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.int8)
    expected_z = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int8)

    assert np.array_equal(updated.faults["X"], expected_x)
    assert np.array_equal(updated.faults["Z"], expected_z)
    assert np.array_equal(fault_list.faults["X"], np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8))
    assert np.array_equal(fault_list.faults["Z"], np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8))


def test_apply_hadamard_inplace_modifies_current_fault_list(fault_list: XZFaultList) -> None:
    result = fault_list.apply_hadamard(qubit=0, inplace=True)

    expected_x = np.array([[0, 0, 1], [1, 1, 0]], dtype=np.int8)
    expected_z = np.array([[1, 1, 0], [0, 0, 1]], dtype=np.int8)

    assert result is fault_list
    assert np.array_equal(fault_list.faults["X"], expected_x)
    assert np.array_equal(fault_list.faults["Z"], expected_z)


def test_apply_hadamard_rejects_invalid_qubit(fault_list: XZFaultList) -> None:
    with pytest.raises(ValueError, match=r"Qubit index must be between 0 and 2."):
        fault_list.apply_hadamard(qubit=3)


def test_apply_reset_rejects_invalid_qubit(fault_list: XZFaultList) -> None:
    with pytest.raises(ValueError, match=r"Qubit index must be between 0 and 2."):
        fault_list.apply_reset(qubit=3)


def test_apply_ccz_updates_z_faults_non_inplace() -> None:
    faults = XZFaultList(num_qubits=3)
    faults.add_fault((np.array([1, 1, 1], dtype=np.int8), np.array([0, 0, 0], dtype=np.int8)))

    updated = faults.apply_ccz(control1=0, control2=1, control3=2, inplace=False)

    expected_x = np.array([[1, 1, 1]], dtype=np.int8)
    expected_z = np.array([[1, 1, 1]], dtype=np.int8)

    assert np.array_equal(updated.faults["X"], expected_x)
    assert np.array_equal(updated.faults["Z"], expected_z)
    assert np.array_equal(faults.faults["X"], np.array([[1, 1, 1]], dtype=np.int8))
    assert np.array_equal(faults.faults["Z"], np.array([[0, 0, 0]], dtype=np.int8))


def test_apply_ccz_inplace_modifies_current_fault_list() -> None:
    faults = XZFaultList(num_qubits=3)
    faults.add_fault((np.array([1, 1, 1], dtype=np.int8), np.array([0, 0, 0], dtype=np.int8)))

    result = faults.apply_ccz(control1=0, control2=1, control3=2, inplace=True)

    expected_x = np.array([[1, 1, 1]], dtype=np.int8)
    expected_z = np.array([[1, 1, 1]], dtype=np.int8)

    assert result is faults
    assert np.array_equal(faults.faults["X"], expected_x)
    assert np.array_equal(faults.faults["Z"], expected_z)


def test_apply_ccz_rejects_invalid_controls() -> None:
    faults = XZFaultList(num_qubits=3)

    with pytest.raises(ValueError, match=r"All qubits must be different."):
        faults.apply_ccz(control1=0, control2=0, control3=2)

    with pytest.raises(ValueError, match=r"Qubit indices must be between 0 and 2."):
        faults.apply_ccz(control1=0, control2=1, control3=3)


def test_apply_ccx_rejects_invalid_qubits() -> None:
    faults = XZFaultList(num_qubits=3)

    with pytest.raises(ValueError, match=r"All qubits must be different."):
        faults.apply_ccx(control1=0, control2=1, target=1)

    with pytest.raises(ValueError, match=r"Qubit indices must be between 0 and 2."):
        faults.apply_ccx(control1=0, control2=1, target=3)


def test_apply_ccz_unit_tests() -> None:
    # its always 0,1,2 for the controls
    # input x, output x, output z
    unit_tests = [
        [(0, 0, 0), (0, 0, 0), (0, 0, 0)],
        [(0, 0, 1), (0, 0, 1), (0, 0, 0)],
        [(0, 1, 0), (0, 1, 0), (0, 0, 0)],
        [(0, 1, 1), (0, 1, 1), (1, 0, 0)],
        [(1, 0, 0), (1, 0, 0), (0, 0, 0)],
        [(1, 0, 1), (1, 0, 1), (0, 1, 0)],
        [(1, 1, 0), (1, 1, 0), (0, 0, 1)],
        [(1, 1, 1), (1, 1, 1), (1, 1, 1)],
    ]

    for input_x, expected_x, expected_z in unit_tests:
        faults = XZFaultList(num_qubits=3)
        faults.add_fault((np.array(input_x, dtype=np.int8), np.array([0, 0, 0], dtype=np.int8)))

        updated = faults.apply_ccz(control1=0, control2=1, control3=2, inplace=False)

        assert np.array_equal(updated.faults["X"], np.array([expected_x], dtype=np.int8))
        assert np.array_equal(updated.faults["Z"], np.array([expected_z], dtype=np.int8))

def test_apply_ccx_unit_tests() -> None:
    # controls are always qubits 0 and 1, target is qubit 2
    # For CCX (Toffoli) with initial Z=0, the resulting X is (c1, c2, t + c1 & c2)
    unit_tests = [
        ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 1), (0, 0, 1)),
        ((0, 1, 0), (0, 1, 0)),
        ((0, 1, 1), (0, 1, 1)),
        ((1, 0, 0), (1, 0, 0)),
        ((1, 0, 1), (1, 0, 1)),
        ((1, 1, 0), (1, 1, 1)),
        ((1, 1, 1), (1, 1, 0)),
    ]

    for input_x, expected_x in unit_tests:
        faults = XZFaultList(num_qubits=3)
        faults.add_fault((np.array(input_x, dtype=np.int8), np.array([0, 0, 0], dtype=np.int8)))

        updated = faults.apply_ccx(control1=0, control2=1, target=2, inplace=False)

        assert np.array_equal(updated.faults["X"], np.array([expected_x], dtype=np.int8))
        assert np.array_equal(updated.faults["Z"], np.array([[0, 0, 0]], dtype=np.int8))


def test_apply_reset_clears_selected_qubit_errors(fault_list: XZFaultList) -> None:
    updated = fault_list.apply_reset(qubit=1, inplace=False)

    expected_x = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.int8)
    expected_z = np.array([[0, 0, 0], [1, 0, 1]], dtype=np.int8)

    assert np.array_equal(updated.faults["X"], expected_x)
    assert np.array_equal(updated.faults["Z"], expected_z)
    assert np.array_equal(fault_list.faults["X"], np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8))
    assert np.array_equal(fault_list.faults["Z"], np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8))


def test_apply_reset_clears_selected_qubit_errors_inplace(fault_list: XZFaultList) -> None:
    result = fault_list.apply_reset(qubit=1, inplace=True)

    expected_x = np.array([[1, 0, 1], [0, 0, 0]], dtype=np.int8)
    expected_z = np.array([[0, 0, 0], [1, 0, 1]], dtype=np.int8)

    assert result is fault_list
    assert np.array_equal(fault_list.faults["X"], expected_x)
    assert np.array_equal(fault_list.faults["Z"], expected_z)


# based on tests for mqt.qecc.circuit_synthesis.faults.coset_leader
def test_reduce_to_coset_leaders_no_generators() -> None:
    """Test coset leader reduction with no generators (should not modify faults)."""
    faults = XZFaultList(num_qubits=3)
    faults.add_fault((np.array([1, 0, 1], dtype=np.int8), np.array([0, 1, 0], dtype=np.int8)))

    # No generators - faults should remain unchanged
    reduced = faults.reduce_to_coset_leaders((None, None), inplace=False)

    assert np.array_equal(reduced.faults["X"], np.array([[1, 0, 1]], dtype=np.int8))
    assert np.array_equal(reduced.faults["Z"], np.array([[0, 1, 0]], dtype=np.int8))


def test_reduce_to_coset_leaders_x_generators() -> None:
    """Test coset leader reduction with X generators."""
    faults = XZFaultList(num_qubits=3)
    # Add X fault that is in the stabilizer group
    faults.add_fault((np.array([1, 0, 1], dtype=np.int8), np.array([0, 0, 0], dtype=np.int8)))

    # X generator that matches the X fault
    x_generators = np.array([[1, 0, 1]], dtype=np.int8)

    reduced = faults.reduce_to_coset_leaders((x_generators, None), inplace=False)

    # X fault should be reduced to zero (it's in the stabilizer group)
    assert np.array_equal(reduced.faults["X"], np.array([[0, 0, 0]], dtype=np.int8))
    # Z fault should remain unchanged
    assert np.array_equal(reduced.faults["Z"], np.array([[0, 0, 0]], dtype=np.int8))


def test_reduce_to_coset_leaders_z_generators() -> None:
    """Test coset leader reduction with Z generators."""
    faults = XZFaultList(num_qubits=3)
    # Add Z fault that is in the stabilizer group
    faults.add_fault((np.array([0, 0, 0], dtype=np.int8), np.array([0, 1, 1], dtype=np.int8)))

    # Z generator that matches the Z fault
    z_generators = np.array([[0, 1, 1]], dtype=np.int8)

    reduced = faults.reduce_to_coset_leaders((None, z_generators), inplace=False)

    # X fault should remain unchanged
    assert np.array_equal(reduced.faults["X"], np.array([[0, 0, 0]], dtype=np.int8))
    # Z fault should be reduced to zero
    assert np.array_equal(reduced.faults["Z"], np.array([[0, 0, 0]], dtype=np.int8))


def test_reduce_to_coset_leaders_both_generators() -> None:
    """Test coset leader reduction with both X and Z generators."""
    faults = XZFaultList(num_qubits=3)
    faults.add_fault((np.array([1, 0, 1], dtype=np.int8), np.array([0, 1, 1], dtype=np.int8)))
    faults.add_fault((np.array([0, 1, 0], dtype=np.int8), np.array([1, 0, 0], dtype=np.int8)))

    # Generators that match the faults
    x_generators = np.array([[1, 0, 1]], dtype=np.int8)
    z_generators = np.array([[0, 1, 1]], dtype=np.int8)

    reduced = faults.reduce_to_coset_leaders((x_generators, z_generators), inplace=False)

    # Both matching faults should be reduced to zero
    assert np.array_equal(reduced.faults["X"][0], np.array([0, 0, 0], dtype=np.int8))
    assert np.array_equal(reduced.faults["Z"][0], np.array([0, 0, 0], dtype=np.int8))
    # Non-matching faults should be reduced to their coset leaders
    assert reduced.faults["X"].shape[0] == 2
    assert reduced.faults["Z"].shape[0] == 2


def test_reduce_to_coset_leaders_inplace() -> None:
    """Test that inplace=True modifies the original fault list."""
    faults = XZFaultList(num_qubits=3)
    faults.add_fault((np.array([1, 0, 1], dtype=np.int8), np.array([0, 1, 0], dtype=np.int8)))

    x_generators = np.array([[1, 0, 1]], dtype=np.int8)

    result = faults.reduce_to_coset_leaders((x_generators, None), inplace=True)

    # Result should be the same object
    assert result is faults
    # X fault should be reduced
    assert np.array_equal(faults.faults["X"], np.array([[0, 0, 0]], dtype=np.int8))
    # Z fault should remain unchanged
    assert np.array_equal(faults.faults["Z"], np.array([[0, 1, 0]], dtype=np.int8))


def test_reduce_to_coset_leaders_not_inplace() -> None:
    """Test that inplace=False returns a new independent fault list."""
    original = XZFaultList(num_qubits=3)
    original.add_fault((np.array([1, 0, 1], dtype=np.int8), np.array([0, 1, 0], dtype=np.int8)))

    x_generators = np.array([[1, 0, 1]], dtype=np.int8)
    reduced = original.reduce_to_coset_leaders((x_generators, None), inplace=False)

    # Result should be a different object
    assert reduced is not original
    # Reduced fault list should be modified
    assert np.array_equal(reduced.faults["X"], np.array([[0, 0, 0]], dtype=np.int8))
    # Original should remain unchanged
    assert np.array_equal(original.faults["X"], np.array([[1, 0, 1]], dtype=np.int8))


def test_reduce_to_coset_leaders_multiple_faults() -> None:
    """Test coset leader reduction with multiple faults."""
    faults = XZFaultList(num_qubits=3)
    faults.add_fault((np.array([1, 0, 1], dtype=np.int8), np.array([0, 0, 0], dtype=np.int8)))
    faults.add_fault((np.array([0, 1, 0], dtype=np.int8), np.array([0, 0, 0], dtype=np.int8)))
    faults.add_fault((np.array([1, 1, 1], dtype=np.int8), np.array([0, 0, 0], dtype=np.int8)))

    # Two X generators
    x_generators = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)

    reduced = faults.reduce_to_coset_leaders((x_generators, None), inplace=False)

    # First two faults are in the stabilizer group, third should be reduced to coset leader
    assert np.array_equal(reduced.faults["X"][0], np.array([0, 0, 0], dtype=np.int8))
    assert np.array_equal(reduced.faults["X"][1], np.array([0, 0, 0], dtype=np.int8))


def test_reduce_to_coset_leaders_invalid_generator_shape() -> None:
    """Test that invalid generator shapes raise ValueError."""
    faults = XZFaultList(num_qubits=3)
    faults.add_fault((np.array([1, 0, 1], dtype=np.int8), np.array([0, 1, 0], dtype=np.int8)))

    # Wrong number of columns in generator
    x_generators = np.array([[1, 0]], dtype=np.int8)

    with pytest.raises(ValueError, match=r"Generators must be a 2D array with 3 columns."):
        faults.reduce_to_coset_leaders((x_generators, None), inplace=False)


def test_reduce_to_coset_leaders_invalid_generator_dimension() -> None:
    """Test that 1D generators raise ValueError."""
    faults = XZFaultList(num_qubits=3)
    faults.add_fault((np.array([1, 0, 1], dtype=np.int8), np.array([0, 1, 0], dtype=np.int8)))

    # 1D array instead of 2D
    x_generators = np.array([1, 0, 1], dtype=np.int8)

    with pytest.raises(ValueError, match=r"Generators must be a 2D array with 3 columns."):
        faults.reduce_to_coset_leaders((x_generators, None), inplace=False)


def test_reduce_to_coset_leaders_empty_fault_list() -> None:
    """Test coset leader reduction with an empty fault list."""
    faults = XZFaultList(num_qubits=3)

    x_generators = np.array([[1, 0, 1]], dtype=np.int8)

    reduced = faults.reduce_to_coset_leaders((x_generators, None), inplace=False)

    # Should remain empty
    assert reduced.faults["X"].shape == (0, 3)
    assert reduced.faults["Z"].shape == (0, 3)


def test_reduce_to_coset_leaders_empty_generators() -> None:
    """Test coset leader reduction with empty generators (no reduction applied)."""
    faults = XZFaultList(num_qubits=3)
    faults.add_fault((np.array([1, 0, 1], dtype=np.int8), np.array([0, 1, 0], dtype=np.int8)))

    # Empty generators
    x_generators = np.empty((0, 3), dtype=np.int8)

    reduced = faults.reduce_to_coset_leaders((x_generators, None), inplace=False)

    # Faults should remain unchanged
    assert np.array_equal(reduced.faults["X"], np.array([[1, 0, 1]], dtype=np.int8))
    assert np.array_equal(reduced.faults["Z"], np.array([[0, 1, 0]], dtype=np.int8))
