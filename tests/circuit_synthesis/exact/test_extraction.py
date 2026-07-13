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
from mqt.qecc.circuit_synthesis.exact.gate_operations import CNOTGate, HGate, SGate
from mqt.qecc.circuit_synthesis.exact.vars import CliffordDepthVars, CliffordGateCountVars


@pytest.fixture
def simple_gate_count_enc() -> tuple[z3.ModelRef, CliffordGateCountVars]:
    """Create a simple SAT model and CliffordGateCountVars for gate-count extraction.

    Encodes: slot 0 = H(0), slot 1 = CX(0→1).
    """
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

    enc = CliffordGateCountVars(
        solver=solver,
        gate_sel={"H": h_vars, "S": s_vars, "CX": c_vars},
        alpha=alpha_vars,
        beta=beta_vars,
        gate_set={"H": HGate, "S": SGate, "CX": CNOTGate},
    )
    return model, enc


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

    enc = CliffordGateCountVars(
        solver=solver,
        gate_sel={"H": h_vars, "S": s_vars, "CX": c_vars},
        alpha=alpha_vars,
        beta=beta_vars,
        gate_set={"H": HGate, "S": SGate, "CX": CNOTGate},
    )
    circuit = extract_clifford_gate_count_circuit(model, 1, 1, enc, k=0)

    assert circuit is not None
    stim_circuit = circuit.to_stim_circuit(with_resets=False)
    assert stim_circuit.num_qubits == 1

    gate_found = False
    for instruction in stim_circuit:
        if instruction.name == "H":
            gate_found = True
    assert gate_found


def test_extract_h_then_cnot(simple_gate_count_enc: tuple[z3.ModelRef, CliffordGateCountVars]) -> None:
    """Test extraction of H followed by CNOT."""
    model, enc = simple_gate_count_enc
    circuit = extract_clifford_gate_count_circuit(model, 2, 2, enc, k=0)

    assert circuit is not None
    stim_circuit = circuit.to_stim_circuit(with_resets=False)
    assert stim_circuit.num_qubits == 2


def test_extract_empty_circuit() -> None:
    """Test extraction with no gates."""
    solver = z3.Solver()
    assert solver.check() == z3.sat
    model = solver.model()

    enc = CliffordGateCountVars(
        solver=solver,
        gate_sel={},
        alpha=[],
        beta=[],
        gate_set={},
    )
    circuit = extract_clifford_gate_count_circuit(model, 1, 0, enc, k=0)

    assert circuit is not None
    stim_circuit = circuit.to_stim_circuit(with_resets=True)
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

    enc = CliffordDepthVars(
        solver=solver,
        gate_vars={"H": h_vars, "S": s_vars, "CX": cx_vars},
        n=n,
        gate_set={"H": HGate, "S": SGate, "CX": CNOTGate},
    )
    circuit = extract_clifford_depth_circuit(model, n, depth, enc, k=0)

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

    init_x: list[int] = []
    init_z = [0, 1, 2]

    circuit = extract_cnot_depth_circuit(model, n, depth, cx_vars, init_x, init_z)

    assert circuit is not None
    assert circuit.num_qubits() == n
