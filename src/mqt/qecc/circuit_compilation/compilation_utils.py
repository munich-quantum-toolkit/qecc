# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility to generate random universal quantum circuits."""

import numpy as np
from qiskit import QuantumCircuit


def random_universal_circuit(
    num_qubits: int, depth: int, gate_probs: dict[str, float] | None = None, seed: int | None = None
) -> QuantumCircuit:
    """Generate a random universal quantum circuit using H, T, CNOT, and ID gates.

    Each depth layer assigns one operation per qubit (unless it's part of a CNOT).
    Avoids consecutive identical non-ID single-qubit gates (even across ID layers).

    Args:
        num_qubits (int): Number of qubits in the circuit.
        depth (int): Number of layers.
        gate_probs (dict): Probabilities for each gate, e.g. {"h": 0.3, "t": 0.3, "cx": 0.2, "id": 0.2}.
        seed (int, optional): RNG seed for reproducibility.

    Returns:
        QuantumCircuit: Randomly generated circuit.
    """
    if gate_probs is None:
        gate_probs = {"h": 0.15, "t": 0.15, "cx": 0.15, "id": 0.55}

    # Normalize probabilities
    total = sum(gate_probs.values())
    gate_probs = {k: v / total for k, v in gate_probs.items()}

    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(num_qubits)

    gates = list(gate_probs.keys())
    probs = list(gate_probs.values())

    # Track last gate and last non-id single-qubit gate
    last_gate = ["id"] * num_qubits
    last_non_id_gate = ["id"] * num_qubits

    for _ in range(depth):
        available_qubits = set(range(num_qubits))

        for q in available_qubits.copy():
            if q not in available_qubits:
                continue  # already consumed by a CNOT

            # Draw a gate with back-to-back restrictions
            while True:
                gate = rng.choice(gates, p=probs)

                # Always allow id
                if gate == "id":
                    break

                # For single-qubit gates, avoid repeating last non-id gate
                if gate in {"h", "t"} and gate != last_non_id_gate[q]:
                    break

                # For CX, handled separately
                if gate == "cx":
                    break

            # Two-qubit gate handling
            if gate == "cx":
                others = list(available_qubits - {q})
                if not others:
                    # Fallback to single-qubit gate if no partner available
                    gate = rng.choice(
                        ["h", "t", "id"],
                        p=[
                            gate_probs[g] / (gate_probs["h"] + gate_probs["t"] + gate_probs["id"])
                            for g in ["h", "t", "id"]
                        ],
                    )
                else:
                    target = rng.choice(others)
                    if rng.random() < 0.5:
                        circuit.cx(q, target)
                    else:
                        circuit.cx(target, q)
                    available_qubits.discard(q)
                    available_qubits.discard(target)
                    last_gate[q] = last_gate[target] = "cx"
                    # don't update last_non_id_gate (CX isn't a single-qubit gate)
                    continue

            # Apply single-qubit gate
            if gate == "h":
                circuit.h(q)
            elif gate == "t":
                circuit.t(q)
            elif gate == "id":
                pass  # do nothing

            available_qubits.discard(q)
            last_gate[q] = gate
            if gate != "id":
                last_non_id_gate[q] = gate

    return circuit


def count_code_switches(circuit: QuantumCircuit) -> tuple[int, list[str | None]]:
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

        # Skip ID gates, they don't constrain anything
        if gate == "id":
            continue

        compat = compatible_codes(gate)
        if not compat:
            msg = f"Gate {gate} not supported by any code"
            raise ValueError(msg)

        # Initialize codes for untouched qubits
        for q in qubits:
            if current_code[q] is None:
                # Assign to one of the compatible codes (deterministically)
                current_code[q] = next(iter(compat))

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
                # Need to switch this qubit's code
                new_code = ({"A", "B"} - {current_code[q]}).pop()
                current_code[q] = new_code
                switch_count += 1

    return switch_count, current_code
