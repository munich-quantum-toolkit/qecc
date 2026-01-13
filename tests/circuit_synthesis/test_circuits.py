# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test circuit representation classes."""

from __future__ import annotations

import numpy as np
import pytest
import stim
from qiskit import QuantumCircuit

from mqt.qecc.circuit_synthesis.circuits import CNOTCircuit, compose_cnot_circuits


def test_add_cnot():
    """Test adding individual CNOT gates to the circuit."""
    circuit = CNOTCircuit()
    circuit.add_cnot(0, 1)
    circuit.add_cnot(2, 3)
    assert circuit.cnots == [(0, 1), (2, 3)], "CNOT gates were not added correctly."


def test_add_cnots():
    """Test adding multiple CNOT gates to the circuit."""
    circuit = CNOTCircuit()
    circuit.add_cnots([(0, 1), (2, 3), (4, 5)])
    assert circuit.cnots == [(0, 1), (2, 3), (4, 5)], "Multiple CNOT gates were not added correctly."


def test_add_cnot_invalid():
    """Test adding an invalid CNOT gate raises an error."""
    circuit = CNOTCircuit()
    with pytest.raises(ValueError, match=r"Control and target qubits must have non-negative indices."):
        circuit.add_cnot(-1, 1)

    with pytest.raises(ValueError, match=r"Control and target qubits cannot be the same."):
        circuit.add_cnot(0, 0)


def test_initialize_qubit():
    """Test initializing qubits in the circuit."""
    circuit = CNOTCircuit()
    circuit.initialize_qubit(0, "Z")
    circuit.initialize_qubit(1, "X")
    assert circuit.initializations == {0: "Z", 1: "X"}, "Qubits were not initialized correctly."


def test_initialize_invalid_basis():
    """Test initializing a qubit with an invalid basis."""
    circuit = CNOTCircuit()
    with pytest.raises(ValueError, match=r"Initialization basis must be 'Z' or 'X'."):
        circuit.initialize_qubit(0, "Y")


def test_initialize_invalid_qubit():
    """Test initializing a qubit with an invalid index."""
    circuit = CNOTCircuit()
    with pytest.raises(ValueError, match=r"Qubit index must be non-negative."):
        circuit.initialize_qubit(-1, "Z")  # Negative index


def test_to_stim_circuit():
    """Test conversion to a stim.Circuit."""
    circuit = CNOTCircuit()
    circuit.add_cnot(0, 1)
    circuit.add_cnot(2, 3)
    circuit.initialize_qubit(0, "Z")
    circuit.initialize_qubit(1, "X")
    stim_circuit = circuit.to_stim_circuit()

    expected_stim = stim.Circuit()
    expected_stim.append_operation("RZ", [0])
    expected_stim.append_operation("RX", [1])
    expected_stim.append_operation("CX", [0, 1, 2, 3])

    assert str(stim_circuit) == str(expected_stim), "Stim circuit conversion failed."


def test_to_qiskit_circuit():
    """Test conversion to a qiskit.QuantumCircuit."""
    circuit = CNOTCircuit()
    circuit.add_cnot(0, 1)
    circuit.add_cnot(2, 3)
    circuit.initialize_qubit(0, "Z")
    circuit.initialize_qubit(1, "X")
    qiskit_circuit = circuit.to_qiskit_circuit(remove_resets=False)

    expected_qiskit = QuantumCircuit(4)
    expected_qiskit.reset(0)
    expected_qiskit.reset(1)
    expected_qiskit.h(1)
    expected_qiskit.cx(0, 1)
    expected_qiskit.cx(2, 3)

    assert qiskit_circuit == expected_qiskit, "Qiskit circuit conversion failed."


def test_is_state():
    """Test the is_state method.

    This test ensures that the ~is_state~ method correctly determines whether
    all qubits involved in the circuit (i.e., those used in CNOT operations)
    are initialized.
    """
    circuit = CNOTCircuit()
    circuit.add_cnot(0, 1)
    circuit.add_cnot(2, 3)
    circuit.initialize_qubit(0, "Z")
    circuit.initialize_qubit(1, "X")
    circuit.initialize_qubit(2, "Z")
    assert not circuit.is_state(), "is_state should return False when not all qubits are initialized."

    circuit.initialize_qubit(3, "X")
    assert circuit.is_state(), "is_state should return True when all qubits are initialized."


def test_cnot_with_uninitialized_qubits():
    """Test a circuit with uninitialized qubits.

    This test ensures that the ~is_state~ method returns False when qubits
    involved in CNOT operations are not initialized.
    """
    circuit = CNOTCircuit()
    circuit.add_cnot(0, 1)
    assert not circuit.is_state(), "is_state should return False when qubits in CNOT are not initialized."


def test_get_code_simple():
    """Test generating a CSS code from a simple CNOT circuit."""
    # Create a CNOT circuit
    circ = CNOTCircuit()
    circ.initialize_qubit(0, "X")
    circ.initialize_qubit(1, "Z")
    circ.add_cnot(0, 1)

    # Generate the CSS code
    code = circ.get_code()

    # Expected stabilizer matrices
    expected_hz = np.array([[1, 1]], dtype=np.int8)  # Z-type stabilizers
    expected_hx = np.array([[1, 1]], dtype=np.int8)  # X-type stabilizers

    # Check the result
    assert np.array_equal(code.Hz, expected_hz), "Z-type stabilizers were not generated correctly."
    assert np.array_equal(code.Hx, expected_hx), "X-type stabilizers were not generated correctly."


def test_get_code_complex():
    """Test generating a CSS code from a more complex CNOT circuit."""
    # Create a CNOT circuit
    circ = CNOTCircuit()
    circ.initialize_qubit(0, "X")
    circ.initialize_qubit(1, "Z")
    circ.initialize_qubit(2, "X")
    circ.add_cnot(0, 1)
    circ.add_cnot(2, 3)
    circ.add_cnot(1, 2)

    # Generate the CSS code
    code = circ.get_code()

    # Expected stabilizer matrices
    expected_hz = np.array([[1, 1, 0, 0]], dtype=np.int8)
    expected_hx = np.array(
        [
            [1, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.int8,
    )

    # Check the result
    assert np.array_equal(code.Hz, expected_hz), "Z-type stabilizers were not generated correctly."
    assert np.array_equal(code.Hx, expected_hx), "X-type stabilizers were not generated correctly."
    assert code.n == 4, "The number of qubits in the code should be 4."
    assert code.k == 1, "The number of logical qubits in the code should be 1."


def test_from_qiskit_circuit_simple():
    """Test converting a simple Qiskit circuit with CNOT gates."""
    # Create a Qiskit circuit
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Convert to CNOTCircuit
    cnot_circuit = CNOTCircuit.from_qiskit_circuit(qc)

    # Expected CNOT gates
    expected_cnots = [(0, 1), (1, 2)]

    # Check the result
    assert cnot_circuit.cnots == expected_cnots, "CNOT gates were not extracted correctly."
    assert cnot_circuit.initializations == {}, "No qubits should be initialized."


def test_from_qiskit_circuit_with_initialization():
    """Test converting a Qiskit circuit with qubit initialization."""
    # Create a Qiskit circuit
    qc = QuantumCircuit(3)
    qc.h(0)  # Initialize qubit 0 in |+>
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Convert to CNOTCircuit
    cnot_circuit = CNOTCircuit.from_qiskit_circuit(qc, initialized_qubits=[0])

    # Expected CNOT gates and initializations
    expected_cnots = [(0, 1), (1, 2)]
    expected_initializations = {0: "X"}

    # Check the result
    assert cnot_circuit.cnots == expected_cnots, "CNOT gates were not extracted correctly."
    assert cnot_circuit.initializations == expected_initializations, "Qubit initialization was not handled correctly."


def test_from_qiskit_circuit_init_all():
    """Test converting a Qiskit circuit with init_all=True."""
    # Create a Qiskit circuit
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Convert to CNOTCircuit with init_all=True
    cnot_circuit = CNOTCircuit.from_qiskit_circuit(qc, init_all=True)

    # Expected CNOT gates and initializations
    expected_cnots = [(0, 1), (1, 2)]
    expected_initializations = {0: "Z", 1: "Z", 2: "Z"}

    # Check the result
    assert cnot_circuit.cnots == expected_cnots, "CNOT gates were not extracted correctly."
    assert cnot_circuit.initializations == expected_initializations, "All qubits should be initialized in the Z basis."


def test_from_qiskit_circuit_unsupported_gate():
    """Test that an unsupported gate raises a ValueError."""
    # Create a Qiskit circuit with an unsupported gate
    qc = QuantumCircuit(3)
    qc.rx(0.5, 0)  # Unsupported gate

    # Attempt to convert to CNOTCircuit
    with pytest.raises(ValueError, match=r"Unsupported gate rx in the circuit."):
        CNOTCircuit.from_qiskit_circuit(qc)


def test_from_qiskit_circuit_hadamard_on_uninitialized_qubit():
    """Test that a Hadamard gate on an uninitialized qubit raises a ValueError."""
    # Create a Qiskit circuit
    qc = QuantumCircuit(3)
    qc.h(0)  # Hadamard on qubit 0

    # Attempt to convert to CNOTCircuit without initializing qubit 0
    with pytest.raises(ValueError, match=r"Hadamard gate on uninitialized qubit 0."):
        CNOTCircuit.from_qiskit_circuit(qc)


def test_from_stim_circuit_simple():
    """Test converting a simple stim circuit with CNOT gates."""
    # Create a stim circuit
    stim_circ = stim.Circuit()
    stim_circ.append_operation("RZ", [0])
    stim_circ.append_operation("RZ", [1])
    stim_circ.append_operation("CX", [0, 1])
    stim_circ.append_operation("CX", [1, 2])

    # Convert to CNOTCircuit
    cnot_circuit = CNOTCircuit.from_stim_circuit(stim_circ)

    # Expected CNOT gates and initializations
    expected_cnots = [(0, 1), (1, 2)]
    expected_initializations = {0: "Z", 1: "Z"}

    # Check the result
    assert cnot_circuit.cnots == expected_cnots, "CNOT gates were not extracted correctly."
    assert cnot_circuit.initializations == expected_initializations, "Qubit initializations were not handled correctly."


def test_from_stim_circuit_with_x_initialization():
    """Test converting a stim circuit with X-basis initialization."""
    # Create a stim circuit
    stim_circ = stim.Circuit()
    stim_circ.append_operation("RX", [0])
    stim_circ.append_operation("CX", [0, 1])

    # Convert to CNOTCircuit
    cnot_circuit = CNOTCircuit.from_stim_circuit(stim_circ)

    # Expected CNOT gates and initializations
    expected_cnots = [(0, 1)]
    expected_initializations = {0: "X"}

    # Check the result
    assert cnot_circuit.cnots == expected_cnots, "CNOT gates were not extracted correctly."
    assert cnot_circuit.initializations == expected_initializations, "X-basis initialization was not handled correctly."


def test_from_stim_circuit_reset_error():
    """Test that resetting a qubit during the circuit raises a ValueError."""
    # Create a stim circuit
    stim_circ = stim.Circuit()
    stim_circ.append_operation("RZ", [0])
    stim_circ.append_operation("RZ", [0])  # Resetting qubit 0

    # Attempt to convert to CNOTCircuit
    with pytest.raises(ValueError, match=r"Qubit 0 reset during circuit."):
        CNOTCircuit.from_stim_circuit(stim_circ)


def test_from_stim_circuit_unsupported_gate():
    """Test that an unsupported gate raises a ValueError."""
    # Create a stim circuit
    stim_circ = stim.Circuit()
    stim_circ.append_operation("H", [0])  # Unsupported gate

    # Attempt to convert to CNOTCircuit
    with pytest.raises(ValueError, match=r"Unsupported gate H in the circuit."):
        CNOTCircuit.from_stim_circuit(stim_circ)


def test_depth_empty_circuit():
    """Test the depth of an empty circuit."""
    circ = CNOTCircuit()
    assert circ.depth() == 0, "The depth of an empty circuit should be 0."


def test_depth_single_cnot():
    """Test the depth of a circuit with a single CNOT gate."""
    circ = CNOTCircuit()
    circ.add_cnot(0, 1)
    assert circ.depth() == 1, "The depth of a single CNOT gate should be 1."


def test_depth_linear_circuit():
    """Test the depth of a linear circuit."""
    circ = CNOTCircuit()
    circ.add_cnot(0, 1)
    circ.add_cnot(1, 2)
    circ.add_cnot(2, 3)
    assert circ.depth() == 3, "The depth of a linear circuit should equal the number of gates."


def test_depth_parallel_circuit():
    """Test the depth of a circuit with parallel CNOT gates."""
    circ = CNOTCircuit()
    circ.add_cnot(0, 1)
    circ.add_cnot(2, 3)
    assert circ.depth() == 1, "Parallel CNOT gates should not increase the depth."


def test_depth_mixed_circuit():
    """Test the depth of a mixed circuit with both linear and parallel gates."""
    circ = CNOTCircuit()
    circ.add_cnot(0, 1)
    circ.add_cnot(1, 2)
    circ.add_cnot(2, 3)
    circ.add_cnot(0, 2)  # Parallel with the first gate
    assert circ.depth() == 4, "The depth of the mixed circuit should be calculated correctly."


def test_num_input_qubits_no_init():
    """Test the number of input qubits in a CNOT circuit with no initializations."""
    circ = CNOTCircuit()
    circ.add_cnot(0, 1)
    circ.add_cnot(2, 3)
    assert circ.num_inputs() == 4, "The number of input qubits should match the highest qubit index used."


def test_num_input_qubits_state():
    """Test the number of input qubits of a state."""
    circ = CNOTCircuit()
    circ.initialize_qubit(0, "Z")
    circ.initialize_qubit(1, "X")
    circ.add_cnot(0, 1)
    assert circ.num_inputs() == 0, "The number of input qubits of a state is 0."


def test_num_input_qubits_isometry():
    """Test the number of input qubits in a CNOT circuit with initializations."""
    circ = CNOTCircuit()
    circ.initialize_qubit(0, "Z")
    circ.initialize_qubit(1, "X")
    circ.add_cnot(0, 1)
    circ.add_cnot(2, 3)
    assert circ.num_inputs() == 2, "The number of input qubits should match the highest qubit index used."


def test_get_uninitialized_qubits():
    """Test the get_uninitialized method.

    This test ensures that the method correctly identifies qubits that have not been initialized.
    """
    circuit = CNOTCircuit()
    circuit.initialize_qubit(0, "Z")
    circuit.initialize_qubit(1, "X")
    circuit.add_cnot(0, 2)
    circuit.add_cnot(3, 4)

    expected_uninitialized = [2, 3, 4]

    assert circuit.get_uninitialized() == expected_uninitialized, "Uninitialized qubits were not identified correctly."


def test_get_logicals():
    """Test the get_logicals method.

    This test ensures that the method correctly identifies both logical X and Z operators for input qubits.
    """
    circuit = CNOTCircuit()
    circuit.add_cnot(0, 1)
    circuit.add_cnot(1, 2)

    expected_logical_x = {
        0: np.array([1, 1, 1], dtype=np.int8),
        1: np.array([0, 1, 1], dtype=np.int8),
        2: np.array([0, 0, 1], dtype=np.int8),
    }
    expected_logical_z = {
        0: np.array([1, 0, 0], dtype=np.int8),
        1: np.array([1, 1, 0], dtype=np.int8),
        2: np.array([0, 1, 1], dtype=np.int8),
    }

    # Check the result
    logicals = circuit.get_logicals()
    for qubit, operator in expected_logical_x.items():
        assert np.array_equal(logicals[qubit][0], operator), f"Logical X operator for qubit {qubit} is incorrect."
    for qubit, operator in expected_logical_z.items():
        assert np.array_equal(logicals[qubit][1], operator), f"Logical Z operator for qubit {qubit} is incorrect."

    x = circuit.get_logical_x()
    z = circuit.get_logical_z()

    for qubit, (x_op, z_op) in logicals.items():
        assert np.array_equal(x[qubit], x_op), f"Logical X operator for qubit {qubit} is incorrect."
        assert np.array_equal(z[qubit], z_op), f"Logical Z operator for qubit {qubit} is incorrect."


def test_logicals_state():
    """Test that states do not have logicals."""
    circuit = CNOTCircuit()
    circuit.initialize_qubit(0, "Z")
    circuit.initialize_qubit(1, "X")

    # Check that the circuit has no logicals
    logicals = circuit.get_logicals()
    assert not logicals, "States should not have logical operators."
    assert circuit.get_logical_x() == {}, "States should not have logical X operators."
    assert circuit.get_logical_z() == {}, "States should not have logical Z operators."


def test_copy_circuit():
    """Test the copy method.

    This test ensures that the copy method creates an independent copy of the circuit
    with the same CNOT gates and initializations.
    """
    # Create an original CNOT circuit
    original_circuit = CNOTCircuit()
    original_circuit.add_cnot(0, 1)
    original_circuit.add_cnot(2, 3)
    original_circuit.initialize_qubit(0, "Z")
    original_circuit.initialize_qubit(1, "X")

    # Create a copy of the circuit
    copied_circuit = original_circuit.copy()

    # Verify that the copied circuit has the same gates and initializations
    assert copied_circuit.cnots == original_circuit.cnots, "CNOT gates were not copied correctly."
    assert copied_circuit.initializations == original_circuit.initializations, (
        "Initializations were not copied correctly."
    )

    # Modify the original circuit and ensure the copy remains unchanged
    original_circuit.add_cnot(4, 5)
    original_circuit.initialize_qubit(2, "Z")

    assert copied_circuit.cnots != original_circuit.cnots, (
        "Copied circuit should not reflect changes to the original circuit."
    )
    assert copied_circuit.initializations != original_circuit.initializations, (
        "Copied circuit should not reflect changes to the original circuit."
    )


def test_relabel_qubits():
    """Test the relabel_qubits method.

    This test ensures that the method correctly updates the qubit indices
    in the circuit according to the provided mapping.
    """
    # Create a CNOT circuit
    circuit = CNOTCircuit()
    circuit.add_cnot(0, 1)
    circuit.add_cnot(2, 3)
    circuit.initialize_qubit(0, "Z")
    circuit.initialize_qubit(1, "X")

    # Define a mapping for relabeling qubits
    mapping = {0: 10, 1: 11, 2: 12, 3: 13}

    # Relabel the qubits
    circuit.relabel_qubits(mapping)

    # Expected CNOT gates and initializations after relabeling
    expected_cnots = [(10, 11), (12, 13)]
    expected_initializations = {10: "Z", 11: "X"}

    # Check the result
    assert circuit.cnots == expected_cnots, "CNOT gates were not relabeled correctly."
    assert circuit.initializations == expected_initializations, "Initializations were not relabeled correctly."


def test_relabel_qubits_invalid_mapping():
    """Test relabeling with an invalid mapping.

    This test ensures that an error is raised when the mapping contains invalid indices.
    """
    # Create a CNOT circuit
    circuit = CNOTCircuit()
    circuit.initialize_qubit(0, "Z")

    # Define an invalid mapping (negative index)
    invalid_mapping = {0: -1}

    # Attempt to relabel the qubits and expect a ValueError
    with pytest.raises(ValueError, match=r"Invalid initialization on negative qubit index: -1"):
        circuit.relabel_qubits(invalid_mapping)


def test_compose_cnot_circuits_no_wiring():
    """Test composing two CNOT circuits without wiring.

    This test ensures that the circuits are vertically stacked when no wiring is provided.
    """
    # Create the first CNOT circuit
    circ1 = CNOTCircuit()
    circ1.add_cnot(0, 1)
    circ1.add_cnot(2, 3)
    circ1.initialize_qubit(0, "Z")
    circ1.initialize_qubit(1, "X")

    # Create the second CNOT circuit
    circ2 = CNOTCircuit()
    circ2.add_cnot(0, 1)
    circ2.add_cnot(1, 2)
    circ2.initialize_qubit(0, "X")
    circ2.initialize_qubit(2, "Z")

    # Compose the circuits without wiring
    composed_circuit, m1, m2 = compose_cnot_circuits(circ1, circ2)

    # Expected CNOT gates and initializations
    expected_cnots = [(0, 1), (2, 3), (4, 5), (5, 6)]
    expected_initializations = {0: "Z", 1: "X", 4: "X", 6: "Z"}

    # Check the result
    assert composed_circuit.cnots == expected_cnots, "CNOT gates were not composed correctly."
    assert composed_circuit.initializations == expected_initializations, "Initializations were not composed correctly."
    assert m1 == {0: 0, 1: 1, 2: 2, 3: 3}, "Mapping m1 should be identity."
    assert m2 == {0: 4, 1: 5, 2: 6}, "Mapping m2 should map circ2 qubits to the end of circ1."


def test_compose_cnot_circuits_with_wiring():
    """Test composing two CNOT circuits with wiring.

    This test ensures that the circuits are composed along the specified wiring.
    """
    # Create the first CNOT circuit
    circ1 = CNOTCircuit()
    circ1.add_cnot(0, 1)
    circ1.add_cnot(2, 3)
    circ1.initialize_qubit(0, "Z")
    circ1.initialize_qubit(1, "X")

    # Create the second CNOT circuit
    circ2 = CNOTCircuit()
    circ2.add_cnot(0, 1)
    circ2.add_cnot(1, 2)
    circ2.initialize_qubit(1, "Z")

    # Define wiring between the circuits
    wiring = {1: 0, 3: 2}

    # Compose the circuits with wiring
    composed_circuit, m1, m2 = compose_cnot_circuits(circ1, circ2, wiring)

    # Expected CNOT gates and initializations
    expected_cnots = [(0, 3), (1, 4), (3, 2), (2, 4)]
    expected_initializations = {0: "Z", 3: "X", 2: "Z"}

    # Check the result
    assert composed_circuit.cnots == expected_cnots, "CNOT gates were not composed correctly with wiring."
    assert composed_circuit.initializations == expected_initializations, (
        "Initializations were not composed correctly with wiring."
    )
    assert m1 == {0: 0, 1: 3, 2: 1, 3: 4}, "Mapping m1 should map circ1 qubits to the composed circuit."
    assert m2 == {0: 3, 1: 2, 2: 4}, "Mapping m2 should map circ2 qubits to the composed circuit."


def test_compose_invalid_wiring():
    """Test composition with invalid wiring."""
    circ1 = CNOTCircuit()
    circ1.add_cnot(0, 1)

    circ2 = CNOTCircuit()
    circ2.initialize_qubit(0, "Z")

    wiring = {1: 0}  # Invalid wiring, circ1 qubit 1 cannot be connected to initialized qubit 0 of circ2

    with pytest.raises(
        ValueError, match=r"Cannot compose circuits with wiring that connects to initialized qubits in circ2."
    ):
        compose_cnot_circuits(circ1, circ2, wiring)
