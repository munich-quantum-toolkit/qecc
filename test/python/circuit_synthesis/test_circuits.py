# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
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

from mqt.qecc.circuit_synthesis import CNOTCircuit


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
    qiskit_circuit = circuit.to_qiskit_circuit()

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
    assert circ.depth() == 3, "The depth of the mixed circuit should be calculated correctly."
