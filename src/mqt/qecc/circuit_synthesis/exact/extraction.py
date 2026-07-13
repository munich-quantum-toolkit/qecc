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

    from .gate_operations import SymbolicGateOperation
    from .vars import CliffordDepthVars, CliffordGateCountVars


def _decode_layer_gates(
    model: z3.ModelRef,
    gate_name: str,
    gate_cls: type[SymbolicGateOperation],
    layer_vars: list[z3.BoolRef],
    n: int,
) -> list[tuple[str, int, int]]:
    """Decode active gate operations from a layer's model values."""
    gates: list[tuple[str, int, int]] = []
    if not gate_cls.IS_TWO_QUBIT:
        for q, var in enumerate(layer_vars):
            if model.eval(var, model_completion=True):
                gates.append((gate_name, q, 0))
    elif not gate_cls.IS_SYMMETRIC:
        pair_idx = 0
        for ctrl in range(n):
            for tgt in range(n):
                if ctrl == tgt:
                    continue
                if model.eval(layer_vars[pair_idx], model_completion=True):
                    gates.append((gate_name, ctrl, tgt))
                pair_idx += 1
    else:
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if model.eval(layer_vars[pair_idx], model_completion=True):
                    gates.append((gate_name, i, j))
                pair_idx += 1
    return gates


def extract_clifford_gate_count_circuit(
    model: z3.ModelRef,
    n: int,
    max_gates: int,
    enc: CliffordGateCountVars,
    k: int,
    pivot_qubits: list[int] | None = None,
) -> CliffordIsometry:
    """Extract Clifford circuit from gate-count SAT model.

    The synthesis builds the circuit in reverse (reducing target to identity),
    so we invert and reverse the gate sequence using each gate class's
    :meth:`~SymbolicGateOperation.inverse_stim_gate`.

    Args:
        model: Z3 model from satisfiable formula.
        n: Number of qubits.
        max_gates: Maximum number of gates.
        enc: Variable container returned by :func:`encode_clifford_gate_count`.
        k: Number of logical qubits.
        pivot_qubits: Qubits to reset (stabilizer/ancilla qubits). Determined
            from the satisfying model; defaults to range(k, n) if not provided.

    Returns:
        Extracted CliffordIsometry circuit.
    """
    gates: list[tuple[str, int, int]] = []
    for slot in range(max_gates):
        for gate_name, sel in enc.gate_sel.items():
            if model.eval(sel[slot], model_completion=True):
                alpha = model.eval(enc.alpha[slot], model_completion=True).as_long()
                beta = model.eval(enc.beta[slot], model_completion=True).as_long()
                gates.append((gate_name, alpha, beta))
                break

    stim_circuit = stim.Circuit()

    init_qubits = pivot_qubits if pivot_qubits is not None else list(range(k, n))
    if init_qubits:
        stim_circuit.append("R", init_qubits)

    for gate_name, q1, q2 in reversed(gates):
        gate_inst = enc.gate_set[gate_name].from_qubits(q1, q2)
        inv_name, inv_qubits = gate_inst.inverse_stim_gate()
        stim_circuit.append(inv_name, inv_qubits)

    return CliffordIsometry.from_stim_circuit(stim_circuit)


def extract_clifford_depth_circuit(
    model: z3.ModelRef,
    n: int,
    max_depth: int,
    enc: CliffordDepthVars,
    k: int,
    pivot_qubits: list[int] | None = None,
) -> CliffordIsometry:
    """Extract Clifford circuit from depth SAT model.

    The synthesis builds the circuit in reverse (reducing target to identity),
    so we invert and reverse the layer sequence.

    The variable index structure recorded in ``enc.gate_vars`` is decoded by
    inspecting the number of booleans per layer:

    - ``n`` entries → single-qubit gate; the index equals the qubit.
    - ``n*(n-1)`` entries → ordered two-qubit gate (CX-like); the pair
      ``(ctrl, tgt)`` is decoded from
      ``ctrl*(n-1) + (tgt if tgt < ctrl else tgt-1)``.
    - ``n*(n-1)//2`` entries → symmetric two-qubit gate (CZ-like); the pair
      ``(i, j)`` with ``i < j`` is decoded from
      ``i*(2n-i-1)//2 + (j-i-1)``.

    Args:
        model: Z3 model from satisfiable formula.
        n: Number of qubits.
        max_depth: Maximum depth.
        enc: Variable container returned by :func:`encode_clifford_depth`.
        k: Number of logical qubits.
        pivot_qubits: Qubits to reset (stabilizer/ancilla qubits). Determined
            from the satisfying model; defaults to range(k, n) if not provided.

    Returns:
        Extracted CliffordIsometry circuit.
    """
    layers: list[list[tuple[str, int, int]]] = []
    for layer in range(max_depth):
        layer_gates: list[tuple[str, int, int]] = []
        for gate_name, all_layer_vars in enc.gate_vars.items():
            if gate_name == "ID":
                continue
            gate_cls = enc.gate_set[gate_name]
            layer_gates.extend(_decode_layer_gates(model, gate_name, gate_cls, all_layer_vars[layer], n))
        if layer_gates:
            layers.append(layer_gates)

    stim_circuit = stim.Circuit()

    init_qubits = pivot_qubits if pivot_qubits is not None else list(range(k, n))
    if init_qubits:
        stim_circuit.append("R", init_qubits)

    for layer_gates in reversed(layers):
        for gate_name, q1, q2 in layer_gates:
            gate_inst = enc.gate_set[gate_name].from_qubits(q1, q2)
            inv_name, inv_qubits = gate_inst.inverse_stim_gate()
            stim_circuit.append(inv_name, inv_qubits)

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
