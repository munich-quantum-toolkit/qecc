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

from mqt.qecc.circuit_synthesis.exact.symmetry import add_clifford_gate_count_symmetry_breaking


@pytest.fixture
def solver() -> z3.Solver:
    """Create a fresh Z3 solver."""
    return z3.Solver()


def test_adjacent_h_cancellation(solver: z3.Solver) -> None:
    """Test that adjacent H gates on same qubit are forbidden."""
    max_gates = 2
    n_bits = 2

    h_vars = [z3.Bool(f"h_{i}") for i in range(max_gates)]
    s_vars = [z3.Bool(f"s_{i}") for i in range(max_gates)]
    c_vars = [z3.Bool(f"c_{i}") for i in range(max_gates)]
    alpha_vars = [z3.BitVec(f"alpha_{i}", n_bits) for i in range(max_gates)]
    beta_vars = [z3.BitVec(f"beta_{i}", n_bits) for i in range(max_gates)]

    add_clifford_gate_count_symmetry_breaking(solver, max_gates, h_vars, s_vars, c_vars, alpha_vars, beta_vars)

    solver.add(h_vars[0])
    solver.add(h_vars[1])
    solver.add(alpha_vars[0] == 0)
    solver.add(alpha_vars[1] == 0)

    assert solver.check() == z3.unsat


def test_adjacent_cnot_cancellation(solver: z3.Solver) -> None:
    """Test that adjacent identical CNOTs are forbidden."""
    max_gates = 2
    n_bits = 2

    h_vars = [z3.Bool(f"h_{i}") for i in range(max_gates)]
    s_vars = [z3.Bool(f"s_{i}") for i in range(max_gates)]
    c_vars = [z3.Bool(f"c_{i}") for i in range(max_gates)]
    alpha_vars = [z3.BitVec(f"alpha_{i}", n_bits) for i in range(max_gates)]
    beta_vars = [z3.BitVec(f"beta_{i}", n_bits) for i in range(max_gates)]

    add_clifford_gate_count_symmetry_breaking(solver, max_gates, h_vars, s_vars, c_vars, alpha_vars, beta_vars)

    solver.add(c_vars[0])
    solver.add(c_vars[1])
    solver.add(alpha_vars[0] == 0)
    solver.add(alpha_vars[1] == 0)
    solver.add(beta_vars[0] == 1)
    solver.add(beta_vars[1] == 1)

    assert solver.check() == z3.unsat


def test_different_gates_allowed(solver: z3.Solver) -> None:
    """Test that different gates or different qubits are allowed."""
    max_gates = 2
    n_bits = 2

    h_vars = [z3.Bool(f"h_{i}") for i in range(max_gates)]
    s_vars = [z3.Bool(f"s_{i}") for i in range(max_gates)]
    c_vars = [z3.Bool(f"c_{i}") for i in range(max_gates)]
    alpha_vars = [z3.BitVec(f"alpha_{i}", n_bits) for i in range(max_gates)]
    beta_vars = [z3.BitVec(f"beta_{i}", n_bits) for i in range(max_gates)]

    add_clifford_gate_count_symmetry_breaking(solver, max_gates, h_vars, s_vars, c_vars, alpha_vars, beta_vars)

    solver.add(h_vars[0])
    solver.add(h_vars[1])
    solver.add(alpha_vars[0] == 0)
    solver.add(alpha_vars[1] == 1)

    assert solver.check() == z3.sat
