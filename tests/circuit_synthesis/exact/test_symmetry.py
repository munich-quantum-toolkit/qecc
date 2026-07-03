# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for symmetry-breaking constraints."""

from __future__ import annotations

import pytest
import z3

from mqt.qecc.circuit_synthesis.exact.gate_operations import CNOTGate, HGate, SGate
from mqt.qecc.circuit_synthesis.exact.symmetry import add_clifford_gate_count_symmetry_breaking
from mqt.qecc.circuit_synthesis.exact.vars import CliffordGateCountVars


def _make_enc(
    solver: z3.Solver,
    max_gates: int,
    n_bits: int,
    prefix: str = "",
) -> CliffordGateCountVars:
    """Build a minimal CliffordGateCountVars for symmetry-breaking tests."""
    h_vars = [z3.Bool(f"{prefix}h_{i}") for i in range(max_gates)]
    s_vars = [z3.Bool(f"{prefix}s_{i}") for i in range(max_gates)]
    c_vars = [z3.Bool(f"{prefix}c_{i}") for i in range(max_gates)]
    alpha_vars = [z3.BitVec(f"{prefix}alpha_{i}", n_bits) for i in range(max_gates)]
    beta_vars = [z3.BitVec(f"{prefix}beta_{i}", n_bits) for i in range(max_gates)]
    return CliffordGateCountVars(
        solver=solver,
        gate_sel={"H": h_vars, "S": s_vars, "CX": c_vars},
        alpha=alpha_vars,
        beta=beta_vars,
        gate_set={"H": HGate, "S": SGate, "CX": CNOTGate},
    )


@pytest.fixture
def solver() -> z3.Solver:
    """Create a fresh Z3 solver."""
    return z3.Solver()


def test_adjacent_h_cancellation(solver: z3.Solver) -> None:
    """Test that adjacent H gates on same qubit are forbidden."""
    max_gates = 2
    enc = _make_enc(solver, max_gates, n_bits=2)
    add_clifford_gate_count_symmetry_breaking(solver, max_gates, enc)

    solver.add(enc.gate_sel["H"][0])
    solver.add(enc.gate_sel["H"][1])
    solver.add(enc.alpha[0] == 0)
    solver.add(enc.alpha[1] == 0)

    assert solver.check() == z3.unsat


def test_adjacent_cnot_cancellation(solver: z3.Solver) -> None:
    """Test that adjacent identical CNOTs are forbidden."""
    max_gates = 2
    enc = _make_enc(solver, max_gates, n_bits=2)
    add_clifford_gate_count_symmetry_breaking(solver, max_gates, enc)

    solver.add(enc.gate_sel["CX"][0])
    solver.add(enc.gate_sel["CX"][1])
    solver.add(enc.alpha[0] == 0)
    solver.add(enc.alpha[1] == 0)
    solver.add(enc.beta[0] == 1)
    solver.add(enc.beta[1] == 1)

    assert solver.check() == z3.unsat


def test_different_gates_allowed(solver: z3.Solver) -> None:
    """Test that different gates or different qubits are allowed."""
    max_gates = 2
    enc = _make_enc(solver, max_gates, n_bits=2)
    add_clifford_gate_count_symmetry_breaking(solver, max_gates, enc)

    solver.add(enc.gate_sel["H"][0])
    solver.add(enc.gate_sel["H"][1])
    solver.add(enc.alpha[0] == 0)
    solver.add(enc.alpha[1] == 1)

    assert solver.check() == z3.sat
