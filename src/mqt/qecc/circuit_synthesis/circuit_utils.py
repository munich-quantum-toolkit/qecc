# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility methods for circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stim import Circuit

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


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


def compose_circuits(
    circ1: Circuit, circ2: Circuit, wiring: dict[int, int] | None = None
) -> tuple[Circuit, dict[int, int], dict[int, int]]:
    """Compose two Stim circuits.

    The circuits are composed only along the qubits that are connected by the `wiring` dict.
    All other qubits are assumed to be unconnected.
    If wire is None, then the circuits are simply vertically stacked.

    Args:
        circ1: The first stim circuit.
        circ2: The second stim circuit.
        wiring: Optional dict mapping outputs of `circ1` to inputs of `circ2`.

    Returns:
        A tuple containing the composed stim circuit and two mappings:
        - mapping1: Maps qubits of circ1 to the composed circuit.
        - mapping2: Maps qubits of circ2 to the composed circuit.
    """
    if wiring is None:
        wiring = {}

    connected = wiring.keys()
    non_connected_circ1 = set(range(circ1.num_qubits)) - set(connected)
    non_connected_circ2 = set(range(circ2.num_qubits)) - set(wiring.values())
    # map non-connected of circ1 to the first n_connected qubits
    non_connected_mapping1 = {q: i for i, q in enumerate(non_connected_circ1)}
    # map non-connected of circ 2 to the qubits n_connected...n_connected + len(circ2)-1
    non_connected_mapping2 = {q: i + len(non_connected_circ1) for i, q in enumerate(non_connected_circ2)}
    # map connected qubits to the last n_connected qubits
    connected_mapping1 = {q: i + len(non_connected_circ1) + len(non_connected_circ2) for i, q in enumerate(connected)}
    connected_mapping2 = {
        wiring[q]: i + len(non_connected_circ1) + len(non_connected_circ2) for i, q in enumerate(connected)
    }

    mapping1 = {**non_connected_mapping1, **connected_mapping1}
    mapping2 = {**non_connected_mapping2, **connected_mapping2}

    composed = circ1.copy()
    composed = relabel_qubits(composed, mapping1)
    circ2_relabelled = relabel_qubits(circ2, mapping2)
    composed += circ2_relabelled
    return composed, mapping1, mapping2
