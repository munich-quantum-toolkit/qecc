# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for terminal constraint encoding."""

from __future__ import annotations

import numpy as np
import pytest
import z3

from mqt.qecc.circuit_synthesis.exact.terminal import (
    add_clifford_isometry_terminal,
    add_css_isometry_terminal,
)


@pytest.fixture
def solver() -> z3.Solver:
    """Create a fresh Z3 solver."""
    return z3.Solver()


def test_unitary_terminal_identity_satisfiable(solver: z3.Solver) -> None:
    """Test that identity tableau satisfies unitary terminal constraints."""
    n = 2
    k = 2

    tableau_x = np.array([[z3.Bool(f"x_{r}_{q}") for q in range(n)] for r in range(2 * k)], dtype=object)
    tableau_z = np.array([[z3.Bool(f"z_{r}_{q}") for q in range(n)] for r in range(2 * k)], dtype=object)

    for r in range(k):
        for q in range(n):
            solver.add(tableau_x[r, q] == (r == q))
            solver.add(tableau_z[r, q] == False)

    for r in range(k, 2 * k):
        for q in range(n):
            solver.add(tableau_x[r, q] == False)
            solver.add(tableau_z[r, q] == (r - k == q))

    add_clifford_isometry_terminal(solver, n, k, tableau_x, tableau_z, allow_qubit_permutation=True)

    assert solver.check() == z3.sat


def test_state_terminal_zero_logical_qubits(solver: z3.Solver) -> None:
    """Test terminal constraints for state preparation (k=0)."""
    n = 2
    k = 0
    m = n

    tableau_x = np.array([[z3.Bool(f"x_{r}_{q}") for q in range(n)] for r in range(m)], dtype=object)
    tableau_z = np.array([[z3.Bool(f"z_{r}_{q}") for q in range(n)] for r in range(m)], dtype=object)

    for r in range(m):
        for q in range(n):
            solver.add(tableau_x[r, q] == False)
            solver.add(tableau_z[r, q] == (r == q))

    add_clifford_isometry_terminal(solver, n, k, tableau_x, tableau_z, allow_qubit_permutation=True)

    assert solver.check() == z3.sat


def test_isometry_terminal_with_stabilizers(solver: z3.Solver) -> None:
    """Test terminal constraints for k=1, n=2 isometry."""
    n = 2
    k = 1

    tableau_x = np.array([[z3.Bool(f"x_{r}_{q}") for q in range(n)] for r in range(2 * k + (n - k))], dtype=object)
    tableau_z = np.array([[z3.Bool(f"z_{r}_{q}") for q in range(n)] for r in range(2 * k + (n - k))], dtype=object)

    solver.add(tableau_x[0, 0])
    solver.add(z3.Not(tableau_x[0, 1]))
    solver.add(z3.Not(tableau_z[0, 0]))
    solver.add(z3.Not(tableau_z[0, 1]))

    solver.add(z3.Not(tableau_x[1, 0]))
    solver.add(z3.Not(tableau_x[1, 1]))
    solver.add(tableau_z[1, 0])
    solver.add(z3.Not(tableau_z[1, 1]))

    solver.add(z3.Not(tableau_x[2, 0]))
    solver.add(z3.Not(tableau_x[2, 1]))
    solver.add(z3.Not(tableau_z[2, 0]))
    solver.add(tableau_z[2, 1])

    add_clifford_isometry_terminal(solver, n, k, tableau_x, tableau_z, allow_qubit_permutation=True)

    assert solver.check() == z3.sat


def test_invalid_stabilizer_x_part_unsatisfiable(solver: z3.Solver) -> None:
    """Test that nonzero stabilizer X-part is rejected."""
    n = 2
    k = 1

    tableau_x = np.array([[z3.Bool(f"x_{r}_{q}") for q in range(n)] for r in range(2 * k + (n - k))], dtype=object)
    tableau_z = np.array([[z3.Bool(f"z_{r}_{q}") for q in range(n)] for r in range(2 * k + (n - k))], dtype=object)

    for r in range(2 * k):
        for q in range(n):
            solver.add(tableau_x[r, q] == (r == q and q == 0))
            solver.add(tableau_z[r, q] == False)

    solver.add(tableau_x[2, 0])
    for q in range(n):
        if q != 0:
            solver.add(z3.Not(tableau_x[2, q]))
    solver.add(z3.Not(tableau_z[2, 0]))
    solver.add(tableau_z[2, 1])

    add_clifford_isometry_terminal(solver, n, k, tableau_x, tableau_z, allow_qubit_permutation=True)

    assert solver.check() == z3.unsat


def test_permutation_disabled_requires_canonical_order(solver: z3.Solver) -> None:
    """Test that disabling permutation enforces canonical qubit order."""
    n = 2
    k = 2

    tableau_x = np.array([[z3.Bool(f"x_{r}_{q}") for q in range(n)] for r in range(2 * k)], dtype=object)
    tableau_z = np.array([[z3.Bool(f"z_{r}_{q}") for q in range(n)] for r in range(2 * k)], dtype=object)

    solver.add(z3.Not(tableau_x[0, 0]))
    solver.add(tableau_x[0, 1])
    solver.add(z3.Not(tableau_z[0, 0]))
    solver.add(z3.Not(tableau_z[0, 1]))

    solver.add(tableau_x[1, 0])
    solver.add(z3.Not(tableau_x[1, 1]))
    solver.add(z3.Not(tableau_z[1, 0]))
    solver.add(z3.Not(tableau_z[1, 1]))

    solver.add(z3.Not(tableau_x[2, 0]))
    solver.add(tableau_z[2, 1])
    solver.add(z3.Not(tableau_z[2, 0]))
    solver.add(z3.Not(tableau_x[2, 1]))

    solver.add(tableau_z[3, 0])
    solver.add(z3.Not(tableau_z[3, 1]))
    solver.add(z3.Not(tableau_x[3, 0]))
    solver.add(z3.Not(tableau_x[3, 1]))

    add_clifford_isometry_terminal(solver, n, k, tableau_x, tableau_z, allow_qubit_permutation=False)

    assert solver.check() == z3.unsat


def test_css_state_terminal_satisfiable(solver: z3.Solver) -> None:
    """Test CSS state terminal (k=0) is satisfiable with valid matrix."""
    n = 3
    k = 0
    m_x = 2

    matrix = np.array([[z3.Bool(f"m_{r}_{q}") for q in range(n)] for r in range(m_x)], dtype=object)

    solver.add(z3.Not(matrix[0, 0]))
    solver.add(z3.Not(matrix[0, 1]))
    solver.add(matrix[0, 2])

    solver.add(z3.Not(matrix[1, 0]))
    solver.add(matrix[1, 1])
    solver.add(z3.Not(matrix[1, 2]))

    add_css_isometry_terminal(solver, n, k, m_x, matrix)

    assert solver.check() == z3.sat


def test_css_isometry_terminal_with_logicals(solver: z3.Solver) -> None:
    """Test CSS isometry terminal with k=1 logical qubit."""
    n = 3
    k = 1
    m_x = 1

    matrix = np.array([[z3.Bool(f"m_{r}_{q}") for q in range(n)] for r in range(k + m_x)], dtype=object)

    solver.add(matrix[0, 0])
    solver.add(z3.Not(matrix[0, 1]))
    solver.add(z3.Not(matrix[0, 2]))

    solver.add(z3.Not(matrix[1, 0]))
    solver.add(matrix[1, 1])
    solver.add(matrix[1, 2])

    add_css_isometry_terminal(solver, n, k, m_x, matrix)

    assert solver.check() == z3.sat


def test_css_terminal_rejects_dependent_rows(solver: z3.Solver) -> None:
    """Test that CSS terminal rejects linearly dependent stabilizer rows."""
    n = 3
    k = 0
    m_x = 2

    matrix = np.array([[z3.Bool(f"m_{r}_{q}") for q in range(n)] for r in range(m_x)], dtype=object)

    for q in range(n):
        solver.add(matrix[0, q] == matrix[1, q])

    add_css_isometry_terminal(solver, n, k, m_x, matrix)

    assert solver.check() == z3.unsat


def test_css_terminal_logical_overlap_forbidden(solver: z3.Solver) -> None:
    """Test that logical qubits cannot share non-pivot columns."""
    n = 4
    k = 2
    m_x = 1

    matrix = np.array([[z3.Bool(f"m_{r}_{q}") for q in range(n)] for r in range(k + m_x)], dtype=object)

    solver.add(matrix[0, 0])
    solver.add(z3.Not(matrix[0, 1]))
    solver.add(z3.Not(matrix[0, 2]))
    solver.add(z3.Not(matrix[0, 3]))

    solver.add(matrix[1, 0])
    solver.add(z3.Not(matrix[1, 1]))
    solver.add(z3.Not(matrix[1, 2]))
    solver.add(z3.Not(matrix[1, 3]))

    solver.add(z3.Not(matrix[2, 0]))
    solver.add(matrix[2, 1])
    solver.add(matrix[2, 2])
    solver.add(matrix[2, 3])

    add_css_isometry_terminal(solver, n, k, m_x, matrix)

    assert solver.check() == z3.unsat
