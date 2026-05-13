# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for circuit extraction from SAT models."""

from __future__ import annotations

import pytest
import z3

from mqt.qecc.circuit_synthesis.exact.extraction import (
    extract_clifford_depth_circuit,
    extract_clifford_gate_count_circuit,
    extract_cnot_depth_circuit,
    extract_cnot_gate_count_circuit,
)


@pytest.fixture
def simple_gate_count_model() -> tuple[z3.ModelRef, list, list, list, list, list]:
    """Create a simple SAT model for gate-count extraction."""
    solver = z3.Solver()

    h_vars = [z3.Bool(f"h_{i}") for i in range(2)]
    s_vars = [z3.Bool(f"s_{i}") for i in range(2)]
    c_vars = [z3.Bool(f"c_{i}") for i in range(2)]
    alpha_vars = [z3.BitVec(f"alpha_{i}", 2) for i in range(2)]
    beta_vars = [z3.BitVec(f"beta_{i}", 2) for i in range(2)]

    solver.add(h_vars[0])
    solver.add(z3.Not(s_vars[0]))
    solver.add(z3.Not(c_vars[0]))
    solver.add(alpha_vars[0] == 0)

    solver.add(z3.Not(h_vars[1]))
    solver.add(z3.Not(s_vars[1]))
    solver.add(c_vars[1])
    solver.add(alpha_vars[1] == 0)
    solver.add(beta_vars[1] == 1)

    assert solver.check() == z3.sat
    model = solver.model()

    return model, h_vars, s_vars, c_vars, alpha_vars, beta_vars


def test_extract_single_h_gate() -> None:
    """Test extraction of single H gate."""
    solver = z3.Solver()

    h_vars = [z3.Bool("h_0")]
    s_vars = [z3.Bool("s_0")]
    c_vars = [z3.Bool("c_0")]
    alpha_vars = [z3.BitVec("alpha_0", 1)]
    beta_vars = [z3.BitVec("beta_0", 1)]

    solver.add(h_vars[0])
    solver.add(z3.Not(s_vars[0]))
    solver.add(z3.Not(c_vars[0]))
    solver.add(alpha_vars[0] == 0)

    assert solver.check() == z3.sat
    model = solver.model()

    circuit = extract_clifford_gate_count_circuit(model, 1, 1, h_vars, s_vars, c_vars, alpha_vars, beta_vars, k=0)

    assert circuit is not None
    stim_circuit = circuit.to_stim_circuit(with_resets=False)
    assert stim_circuit.num_qubits == 1

    gate_found = False
    for instruction in stim_circuit:
        if instruction.name == "H":
            gate_found = True
    assert gate_found


def test_extract_h_then_cnot(simple_gate_count_model: tuple) -> None:
    """Test extraction of H followed by CNOT."""
    model, h_vars, s_vars, c_vars, alpha_vars, beta_vars = simple_gate_count_model

    circuit = extract_clifford_gate_count_circuit(model, 2, 2, h_vars, s_vars, c_vars, alpha_vars, beta_vars, k=0)

    assert circuit is not None
    stim_circuit = circuit.to_stim_circuit(with_resets=False)
    assert stim_circuit.num_qubits == 2


def test_extract_empty_circuit() -> None:
    """Test extraction with no gates."""
    solver = z3.Solver()

    h_vars = []
    s_vars = []
    c_vars = []
    alpha_vars = []
    beta_vars = []

    assert solver.check() == z3.sat
    model = solver.model()

    circuit = extract_clifford_gate_count_circuit(model, 1, 0, h_vars, s_vars, c_vars, alpha_vars, beta_vars, k=0)

    assert circuit is not None
    stim_circuit = circuit.to_stim_circuit(with_resets=False)
    assert stim_circuit.num_qubits == 1


def test_extract_parallel_h_gates() -> None:
    """Test extraction of parallel H gates in single layer."""
    solver = z3.Solver()
    n = 2
    depth = 1

    h_vars = [[z3.Bool(f"h_{layer}_{q}") for q in range(n)] for layer in range(depth)]
    s_vars = [[z3.Bool(f"s_{layer}_{q}") for q in range(n)] for layer in range(depth)]
    cx_vars = [[z3.Bool(f"cx_{layer}_{i}_{j}") for i in range(n) for j in range(n) if i != j] for layer in range(depth)]

    solver.add(h_vars[0][0])
    solver.add(h_vars[0][1])
    solver.add(z3.Not(s_vars[0][0]))
    solver.add(z3.Not(s_vars[0][1]))
    for cx in cx_vars[0]:
        solver.add(z3.Not(cx))

    assert solver.check() == z3.sat
    model = solver.model()

    circuit = extract_clifford_depth_circuit(model, n, depth, h_vars, s_vars, cx_vars, k=0)

    assert circuit is not None
    stim_circuit = circuit.to_stim_circuit(with_resets=False)
    assert stim_circuit.num_qubits == n


def test_extract_single_cnot() -> None:
    """Test extraction of single CNOT gate."""
    solver = z3.Solver()

    alpha_vars = [z3.BitVec("alpha_0", 2)]
    beta_vars = [z3.BitVec("beta_0", 2)]

    solver.add(alpha_vars[0] == 0)
    solver.add(beta_vars[0] == 1)

    assert solver.check() == z3.sat
    model = solver.model()

    init_x = [0]
    init_z = [0, 1]

    circuit = extract_cnot_gate_count_circuit(model, 2, 1, alpha_vars, beta_vars, init_x, init_z)

    assert circuit is not None
    assert circuit.num_qubits() == 2


def test_extract_cnot_depth_circuit() -> None:
    """Test extraction of CNOT depth circuit."""
    solver = z3.Solver()
    n = 3
    depth = 2

    cx_vars = [[z3.Bool(f"cx_{layer}_{i}_{j}") for i in range(n) for j in range(n) if i != j] for layer in range(depth)]

    solver.add(cx_vars[0][0])
    for i in range(1, len(cx_vars[0])):
        solver.add(z3.Not(cx_vars[0][i]))

    solver.add(cx_vars[1][2])
    for i in range(len(cx_vars[1])):
        if i != 2:
            solver.add(z3.Not(cx_vars[1][i]))

    assert solver.check() == z3.sat
    model = solver.model()

    init_x = []
    init_z = [0, 1, 2]

    circuit = extract_cnot_depth_circuit(model, n, depth, cx_vars, init_x, init_z)

    assert circuit is not None
    assert circuit.num_qubits() == n
