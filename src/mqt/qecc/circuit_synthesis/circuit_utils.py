# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility methods for circuits."""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from stim import Circuit


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
    # qiskit already does the job for us
    qiskit_circ = QuantumCircuit.from_qasm_str(circ.to_qasm(open_qasm_version=2))
    dag = circuit_to_dag(qiskit_circ)
    layers = dag.layers()
    new_circ = QuantumCircuit(qiskit_circ.num_qubits)
    for layer in layers:
        layer_circ = dag_to_circuit(layer["graph"])
        new_circ.compose(layer_circ, inplace=True)

    # Convert back to stim circuit
    return qiskit_to_stim_circuit(new_circ)
