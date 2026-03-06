# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utilities for quantum circuit compilation: random circuit generation, code-switch analysis, node parsing, and placeholder insertion."""

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode


def naive_switching(circuit: QuantumCircuit) -> tuple[int, list[str | None]]:
    """Count how many code switches are needed for a circuit assuming.

      - Code A supports H, CNOT
      - Code B supports CNOT, T
      - CNOT can only occur between qubits in the same code
      - Each qubit starts in the code of its first gate.

    Returns:
        int: total number of code switches
        dict: final code assignment per qubit
    """
    # Define transversal capabilities
    code_a = {"h", "cx"}
    code_b = {"t", "cx"}

    num_qubits = circuit.num_qubits
    current_code: list[str | None] = [None] * num_qubits
    switch_count: int = 0

    # Helper: which codes support a gate
    def compatible_codes(gate_name: str) -> set[str]:
        if gate_name in code_a and gate_name in code_b:
            return {"A", "B"}
        if gate_name in code_a:
            return {"A"}
        if gate_name in code_b:
            return {"B"}
        return set()

    for instr in circuit.data:
        gate = instr.operation.name
        qubits = [circuit.find_bit(q).index for q in instr.qubits]

        if gate == "id":
            continue

        compat = compatible_codes(gate)
        if not compat:
            msg = f"Gate {gate} not supported by any code"
            raise ValueError(msg)

        # Initialize codes for untouched qubits
        for q in qubits:
            if current_code[q] is None:
                current_code[q] = "A" if "A" in compat else "B"

        # If it's a multi-qubit gate, ensure code consistency
        involved_codes = {current_code[q] for q in qubits}

        if len(involved_codes) > 1:
            # Must synchronize codes before performing CNOT
            # For simplicity: switch all involved to one common valid code
            target_code = "A" if "A" in compat else "B"
            for q in qubits:
                if current_code[q] != target_code:
                    current_code[q] = target_code
                    switch_count += 1

        # Check if qubits are in a valid code for this gate
        for q in qubits:
            if current_code[q] not in compat:
                new_code = "A" if current_code[q] == "B" else "B"
                current_code[q] = new_code
                switch_count += 1

    return switch_count, current_code


def insert_switch_placeholders(
    circuit: QuantumCircuit,
    switch_positions: list[tuple[int, int]],
    placeholder_depth: int = 1,
) -> QuantumCircuit:
    """Return a new circuit with 'switch' placeholders inserted between global DAG layers.

    This function inserts placeholders *after* the entire global layer with index
    `layer_index` (i.e., between layer `layer_index` and the next). Placeholders are
    placed on the requested qubit regardless of whether that qubit was active in that layer.

    Parameters
    ----------
    circuit : QuantumCircuit
        The original circuit to augment.
    switch_positions : List[Tuple[int, int]]
        List of (qubit_index, layer_index). `layer_index` refers to the index
        from `list(circuit_to_dag(circuit).layers())`. A placeholder for
        `(q, k)` will be inserted after global layer `k`.
    placeholder_depth : int, optional
        Virtual depth (single-qubit layers) the placeholder should represent.
    expand_placeholder : bool, optional
        If True, expand each placeholder into `placeholder_depth` calls to
        `QuantumCircuit.id(qubit)` so that `QuantumCircuit.depth()` increases.
        If False, append a single `SwitchGate` marker (informational only).

    Returns:
    -------
    QuantumCircuit
        New circuit with placeholders inserted.
    """
    # Build DAG and layers
    dag = circuit_to_dag(circuit)
    layers = list(dag.layers())

    # Normalize and group requested placeholders by qubit
    placeholders_by_qubit: dict[int, list[int]] = {}
    for qidx, layer_idx in switch_positions:
        if layer_idx < 0:
            continue
        placeholders_by_qubit.setdefault(qidx, []).append(layer_idx)

    # Sort layer indices per qubit for deterministic behavior
    for depths in placeholders_by_qubit.values():
        depths.sort()

    # Prepare output circuit with same registers
    new_qc = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name + "_with_switches")

    # Track which placeholders we already inserted
    inserted_placeholders: dict[int, set[int]] = {q: set() for q in placeholders_by_qubit}

    def _append_placeholder_on_qubit(q_index: int, depth_equiv: int) -> None:
        """Append a placeholder (either expanded ids or a SwitchGate) on the qubit."""
        qubit = circuit.qubits[q_index]
        # append id several times so `.depth()` counts them
        for _ in range(max(1, int(depth_equiv))):
            new_qc.id(qubit)

    # Iterate layers in order; copy each op, then after the whole layer insert placeholders for that layer
    for depth_idx, layer in enumerate(layers):
        layer_graph = layer["graph"]
        for node in layer_graph.op_nodes():
            assert isinstance(node, DAGOpNode)
            # Map node.qargs (Qubit objects) to the corresponding qubit objects of the original circuit
            qbit_objs = list(node.qargs)
            cbit_objs = list(node.cargs) if hasattr(node, "cargs") else []

            # Append the operation to the new circuit on the same physical qubits/bits
            new_qc.append(node.op, qbit_objs, cbit_objs)

        # --- AFTER FINISHING THIS GLOBAL LAYER: insert placeholders targeted at this layer ---
        for q_index, depths in placeholders_by_qubit.items():
            for target_depth in depths:
                if target_depth == depth_idx and target_depth not in inserted_placeholders[q_index]:
                    _append_placeholder_on_qubit(q_index, placeholder_depth)
                    inserted_placeholders[q_index].add(target_depth)

    # Append any placeholders whose requested layer index was beyond the number of layers
    for q_index, depths in placeholders_by_qubit.items():
        for target_depth in depths:
            if target_depth not in inserted_placeholders[q_index]:
                # target was never inserted (layer out of range) -> append at end
                _append_placeholder_on_qubit(q_index, placeholder_depth)
                inserted_placeholders[q_index].add(target_depth)

    return new_qc
