# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""General circuit constructions."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit, QuantumRegister
from stim import Circuit


def reorder_qubits(circ: QuantumCircuit, qubit_mapping: dict[int, int]) -> QuantumCircuit:
    """Reorders the qubits in a QuantumCircuit based on the given mapping.

    Parameters:
        circuit: The original quantum circuit.
        qubit_mapping: A dictionary mapping original qubit indices to new qubit indices.

    Returns:
        A new quantum circuit with qubits reordered.
    """
    # Validate the qubit_mapping
    if sorted(qubit_mapping.keys()) != list(range(len(circ.qubits))) or sorted(qubit_mapping.values()) != list(
        range(len(circ.qubits))
    ):
        msg = "Invalid qubit_mapping: It must be a permutation of the original qubit indices."
        raise ValueError(msg)

    # Create a new quantum register
    num_qubits = len(circ.qubits)
    new_register = QuantumRegister(num_qubits, "q")
    new_circuit = QuantumCircuit(new_register)

    # Remap instructions based on the qubit_mapping
    for instruction, qubits, clbits in circ.data:
        new_qubits = [new_register[qubit_mapping[circ.find_bit(q)[0]]] for q in qubits]
        new_circuit.append(instruction, new_qubits, clbits)

    return new_circuit


def relabel_qubits(circ: Circuit, qubit_mapping: dict[int, int] | int) -> Circuit:
    """Relabels the qubits in a stim circuit based on the given mapping.

    Parameters:
        circ (Circuit): The original stim circuit.
        qubit_mapping (dict[int, int] | int): Either a dictionary mapping original qubit indices to new qubit indices or a constant offset to add to all qubit indices.

    Returns:
        Circuit: A new stim circuit with qubits relabeled.
    """
    new_circ = Circuit()
    for op in circ:
        if isinstance(qubit_mapping, dict):
            relabelled_qubits = [qubit_mapping[q.value] for q in op.targets_copy()]
        else:
            relabelled_qubits = [q.value + qubit_mapping for q in op.targets_copy()]
        new_circ.append(op.name, relabelled_qubits)
    return new_circ


def qiskit_to_stim_circuit(qc: QuantumCircuit) -> Circuit:
    """Convert a Qiskit circuit to a Stim circuit."""
    single_qubit_gate_map = {
        "h": "H",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "s": "S",
        "sdg": "S_DAG",
        "sx": "SQRT_X",
        "measure": "MR",
        "reset": "R",
    }
    stim_circuit = Circuit()
    for gate in qc:
        op = gate.operation.name
        qubit = qc.find_bit(gate.qubits[0])[0]
        if op in single_qubit_gate_map:
            stim_circuit.append_operation(single_qubit_gate_map[op], [qubit])
        elif op == "cx":
            target = qc.find_bit(gate.qubits[1])[0]
            stim_circuit.append_operation("CX", [qubit, target])
        else:
            msg = f"Unsupported gate: {op}"
            raise ValueError(msg)
    return stim_circuit


def compact_stim_circuit(circ: Circuit) -> Circuit:
    """Move circuit instructions to the front and ignore TICKS.

    Args:
         circ: stim circuit to compact
    Returns:
         A compacted stim circuit.
    """
    compact_circ = Circuit()
    for layer in collect_circuit_layers(circ):
        compact_circ += layer
    return compact_circ


def collect_circuit_layers(circ: Circuit) -> list[Circuit]:
    """Collect all layers that can be executed in parallel.

    Args:
        circ: Stim circuit to process.

    Returns:
        list of circuit layers. All instructions in one layer can be executed in parallel. It holds that circ=sum(collect_circuit_layers(circ)).
    """
    # Copy the circuit and separate all instructions by ticks
    circ_cpy = Circuit()
    for instr in circ:
        for grp in instr.target_groups():
            qubits = [q.qubit_value for q in grp]
            circ_cpy.append_operation(instr.name, qubits)
            circ_cpy.append_operation("TICK", [])

    # Now work with the copied circuit
    circ = circ_cpy
    n_qubits = circ.num_qubits
    layers = []

    while len(circ) > 0:
        layer = Circuit()
        qubit_layer_used = [False] * n_qubits  # Track used qubits in this layer
        instr_to_delete = []  # Track instructions to delete after adding them to the layer
        idx = 0

        while idx < len(circ):
            instr = circ[idx]

            # Skip TICK instructions
            while instr is not None and instr.name == "TICK" and idx < len(circ):
                circ.pop(idx)
                instr = circ[idx] if idx < len(circ) else None

            if instr is None:  # No more instructions to process
                break

            qubits = [q.qubit_value for q in instr.targets_copy()]

            # Check if any qubit from this instruction is already used in the layer
            if not any(qubit_layer_used[q] for q in qubits):
                layer.append_operation(instr.name, qubits)
                instr_to_delete.append(idx)  # Mark this instruction for removal

                # Mark the qubits used in this instruction
                for q in qubits:
                    qubit_layer_used[q] = True

            idx += 1

        # Add the layer to the list
        layers.append(layer)

        # Remove the instructions that were added to the layer
        for n_deleted, gate_idx in enumerate(instr_to_delete):
            circ.pop(gate_idx - n_deleted)

    return layers
