# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit extraction from SAT models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import stim

from ..circuits import CliffordIsometry, CNOTCircuit

if TYPE_CHECKING:
    import z3


def extract_clifford_gate_count_circuit(
    model: z3.ModelRef,
    n: int,
    max_gates: int,
    h_vars: list[z3.BoolRef],
    s_vars: list[z3.BoolRef],
    c_vars: list[z3.BoolRef],
    alpha_vars: list[z3.BitVecRef],
    beta_vars: list[z3.BitVecRef],
    k: int,
) -> CliffordIsometry:
    """Extract Clifford circuit from gate-count SAT model.

    The synthesis builds the circuit in reverse (reducing target to identity),
    so we need to invert and reverse the gate sequence.

    Args:
        model: Z3 model from satisfiable formula.
        n: Number of qubits.
        max_gates: Maximum number of gates.
        h_vars: Hadamard gate selection variables.
        s_vars: S gate selection variables.
        c_vars: CNOT gate selection variables.
        alpha_vars: First qubit index variables.
        beta_vars: Second qubit index variables.
        k: Number of logical qubits.

    Returns:
        Extracted CliffordIsometry circuit.
    """
    gates: list[tuple[str, int, int]] = []
    for slot in range(max_gates):
        h = model.eval(h_vars[slot], model_completion=True)
        s = model.eval(s_vars[slot], model_completion=True)
        c = model.eval(c_vars[slot], model_completion=True)

        if not (h or s or c):
            continue

        alpha = model.eval(alpha_vars[slot], model_completion=True).as_long()

        if h:
            gates.append(("H", alpha, 0))
        elif s:
            gates.append(("S", alpha, 0))
        elif c:
            beta = model.eval(beta_vars[slot], model_completion=True).as_long()
            gates.append(("CX", alpha, beta))

    stim_circuit = stim.Circuit()

    m = n - k
    if m > 0:
        stim_circuit.append("R", list(range(k, n)))

    for gate in reversed(gates):
        if gate[0] == "H":
            stim_circuit.append("H", [gate[1]])
        elif gate[0] == "S":
            stim_circuit.append("S_DAG", [gate[1]])
        elif gate[0] == "CX":
            stim_circuit.append("CX", [gate[1], gate[2]])

    return CliffordIsometry.from_stim_circuit(stim_circuit)


def extract_clifford_depth_circuit(
    model: z3.ModelRef,
    n: int,
    max_depth: int,
    h_vars: list[list[z3.BoolRef]],
    s_vars: list[list[z3.BoolRef]],
    cx_vars: list[list[z3.BoolRef]],
    k: int,
) -> CliffordIsometry:
    """Extract Clifford circuit from depth SAT model.

    The synthesis builds the circuit in reverse (reducing target to identity),
    so we need to invert and reverse the layer sequence.

    Args:
        model: Z3 model from satisfiable formula.
        n: Number of qubits.
        max_depth: Maximum depth.
        h_vars: Hadamard gate variables [layer][qubit].
        s_vars: S gate variables [layer][qubit].
        cx_vars: CNOT gate variables [layer][cx_idx].
        k: Number of logical qubits.

    Returns:
        Extracted CliffordIsometry circuit.
    """
    layers: list[list[tuple[str, int, int]]] = []
    for layer in range(max_depth):
        layer_gates: list[tuple[str, int, int]] = []

        for q in range(n):
            h = model.eval(h_vars[layer][q], model_completion=True)
            s = model.eval(s_vars[layer][q], model_completion=True)

            if h:
                layer_gates.append(("H", q, 0))
            elif s:
                layer_gates.append(("S", q, 0))

        cx_idx = 0
        for ctrl in range(n):
            for tgt in range(n):
                if ctrl == tgt:
                    continue
                cx = model.eval(cx_vars[layer][cx_idx], model_completion=True)
                if cx:
                    layer_gates.append(("CX", ctrl, tgt))
                cx_idx += 1

        if layer_gates:
            layers.append(layer_gates)

    stim_circuit = stim.Circuit()

    m = n - k
    if m > 0:
        stim_circuit.append("R", list(range(k, n)))

    for layer_gates in reversed(layers):
        for gate in layer_gates:
            if gate[0] == "H":
                stim_circuit.append("H", [gate[1]])
            elif gate[0] == "S":
                stim_circuit.append("S_DAG", [gate[1]])
            elif gate[0] == "CX":
                stim_circuit.append("CX", [gate[1], gate[2]])

    return CliffordIsometry.from_stim_circuit(stim_circuit)


def extract_cnot_gate_count_circuit(
    model: z3.ModelRef,
    n: int,
    max_gates: int,
    alpha_vars: list[z3.BitVecRef],
    beta_vars: list[z3.BitVecRef],
    init_x: list[int],
    init_z: list[int],
) -> CNOTCircuit:
    """Extract CNOT circuit from gate-count SAT model.

    The synthesis builds the circuit in reverse (reducing target to identity),
    so we need to reverse the CNOT sequence.

    Args:
        model: Z3 model from satisfiable formula.
        n: Number of qubits.
        max_gates: Maximum number of gates.
        alpha_vars: Control qubit index variables.
        beta_vars: Target qubit index variables.
        init_x: Qubits initialized in |+⟩ state.
        init_z: Qubits initialized in |0⟩ state.

    Returns:
        Extracted CNOTCircuit.
    """
    cnots = []
    for slot in range(max_gates):
        alpha = model.eval(alpha_vars[slot], model_completion=True).as_long()
        beta = model.eval(beta_vars[slot], model_completion=True).as_long()
        cnots.append((alpha, beta))

    reversed_cnots = list(reversed(cnots))

    inputs = [i for i in range(n) if i not in init_x and i not in init_z]

    return CNOTCircuit.from_cnot_list(
        reversed_cnots,
        initialize_z=init_z,
        initialize_x=init_x,
        inputs=inputs,
    )


def extract_cnot_depth_circuit(
    model: z3.ModelRef,
    n: int,
    max_depth: int,
    cx_vars: list[list[z3.BoolRef]],
    init_x: list[int],
    init_z: list[int],
) -> CNOTCircuit:
    """Extract CNOT circuit from depth SAT model.

    The synthesis builds the circuit in reverse (reducing target to identity),
    so we need to reverse the layer sequence.

    Args:
        model: Z3 model from satisfiable formula.
        n: Number of qubits.
        max_depth: Maximum depth.
        cx_vars: CNOT gate variables [layer][cx_idx].
        init_x: Qubits initialized in |+⟩ state.
        init_z: Qubits initialized in |0⟩ state.

    Returns:
        Extracted CNOTCircuit.
    """
    cnots = []
    for layer in range(max_depth):
        cx_idx = 0
        for ctrl in range(n):
            for tgt in range(n):
                if ctrl == tgt:
                    continue
                cx = model.eval(cx_vars[layer][cx_idx], model_completion=True)
                if cx:
                    cnots.append((ctrl, tgt))
                cx_idx += 1

    reversed_cnots = list(reversed(cnots))

    inputs = [i for i in range(n) if i not in init_x and i not in init_z]

    return CNOTCircuit.from_cnot_list(
        reversed_cnots,
        initialize_z=init_z,
        initialize_x=init_x,
        inputs=inputs,
    )
