# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit extraction from SAT models."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import z3

    from ..circuits import CliffordIsometry, CNOTCircuit


def extract_clifford_depth_circuit(
    model: z3.ModelRef,
    n: int,
    max_depth: int,
    h_vars: list[list[z3.BoolRef]],
    s_vars: list[list[z3.BoolRef]],
    cx_vars: list[list[z3.BoolRef]],
) -> CliffordIsometry:
    """Extract Clifford circuit from depth-bounded model.

    Args:
        model: Satisfying Z3 model.
        n: Number of qubits.
        max_depth: Depth bound.
        h_vars: H gate variables.
        s_vars: S gate variables.
        cx_vars: CNOT gate variables.

    Returns:
        Extracted CliffordIsometry.
    """
    import stim

    from ..circuits import CliffordIsometry

    circuit = stim.Circuit()

    for depth in range(max_depth):
        for i in range(n):
            if model.eval(h_vars[depth][i], model_completion=True):
                circuit.append("H", [i])
            elif model.eval(s_vars[depth][i], model_completion=True):
                circuit.append("S", [i])

        for cx_idx, cx_var in enumerate(cx_vars[depth]):
            if model.eval(cx_var, model_completion=True):
                control = cx_idx // n
                target = cx_idx % n
                circuit.append("CX", [control, target])

    return CliffordIsometry.from_stim_circuit(circuit)


def extract_cnot_depth_circuit(
    model: z3.ModelRef,
    n: int,
    max_depth: int,
    cx_vars: list[list[z3.BoolRef]],
    init_x: list[int],
    init_z: list[int],
) -> CNOTCircuit:
    """Extract CNOT circuit from depth-bounded model.

    Args:
        model: Satisfying Z3 model.
        n: Number of qubits.
        max_depth: Depth bound.
        cx_vars: CNOT gate variables.
        init_x: Qubits to initialize in X basis (|+>).
        init_z: Qubits to initialize in Z basis (|0>).

    Returns:
        Extracted CNOTCircuit.
    """
    from ..circuits import CNOTCircuit

    cnots: list[tuple[int, int]] = []

    for depth in range(max_depth):
        for cx_idx, cx_var in enumerate(cx_vars[depth]):
            if model.eval(cx_var, model_completion=True):
                control = cx_idx // n
                target = cx_idx % n
                cnots.append((control, target))

    return CNOTCircuit.from_cnot_list(cnots, initialize_z=init_z, initialize_x=init_x)


def extract_clifford_gate_count_circuit(
    model: z3.ModelRef,
    n: int,
    max_gates: int,
    h_vars: list[z3.BoolRef],
    s_vars: list[z3.BoolRef],
    c_vars: list[z3.BoolRef],
    alpha_vars: list[z3.BitVecRef],
    beta_vars: list[z3.BitVecRef],
) -> CliffordIsometry:
    """Extract Clifford circuit from gate-count-bounded model.

    Args:
        model: Satisfying Z3 model.
        n: Number of qubits.
        max_gates: Gate count bound.
        h_vars: H gate selection variables.
        s_vars: S gate selection variables.
        c_vars: CNOT gate selection variables.
        alpha_vars: Index variables for single-qubit gates / CNOT control.
        beta_vars: Index variables for CNOT target.

    Returns:
        Extracted CliffordIsometry.
    """
    import stim

    from ..circuits import CliffordIsometry

    circuit = stim.Circuit()

    for slot in range(max_gates):
        if model.eval(h_vars[slot], model_completion=True):
            qubit = model.eval(alpha_vars[slot], model_completion=True).as_long()
            circuit.append("H", [qubit])
        elif model.eval(s_vars[slot], model_completion=True):
            qubit = model.eval(alpha_vars[slot], model_completion=True).as_long()
            circuit.append("S", [qubit])
        elif model.eval(c_vars[slot], model_completion=True):
            control = model.eval(alpha_vars[slot], model_completion=True).as_long()
            target = model.eval(beta_vars[slot], model_completion=True).as_long()
            circuit.append("CX", [control, target])

    return CliffordIsometry.from_stim_circuit(circuit)


def extract_cnot_gate_count_circuit(
    model: z3.ModelRef,
    n: int,
    max_gates: int,
    alpha_vars: list[z3.BitVecRef],
    beta_vars: list[z3.BitVecRef],
    init_x: list[int],
    init_z: list[int],
) -> CNOTCircuit:
    """Extract CNOT circuit from gate-count-bounded model.

    Args:
        model: Satisfying Z3 model.
        n: Number of qubits.
        max_gates: Gate count bound.
        alpha_vars: CNOT control index variables.
        beta_vars: CNOT target index variables.
        init_x: Qubits to initialize in X basis.
        init_z: Qubits to initialize in Z basis.

    Returns:
        Extracted CNOTCircuit.
    """
    from ..circuits import CNOTCircuit

    cnots: list[tuple[int, int]] = []

    for slot in range(max_gates):
        control = model.eval(alpha_vars[slot], model_completion=True).as_long()
        target = model.eval(beta_vars[slot], model_completion=True).as_long()
        cnots.append((control, target))

    return CNOTCircuit.from_cnot_list(cnots, initialize_z=init_z, initialize_x=init_x)
