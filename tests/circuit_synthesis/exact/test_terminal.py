# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for terminal constraints."""

from __future__ import annotations

import numpy as np
import pytest
import z3

from mqt.qecc.circuit_synthesis.exact.terminal import (
    add_clifford_unitary_terminal,
    add_css_isometry_terminal,
    add_css_state_terminal,
    add_stabilizer_state_terminal,
)


class TestCliffordUnitaryTerminal:
    """Tests for Clifford unitary terminal constraints."""

    @pytest.fixture
    def solver(self) -> z3.Solver:
        """Create a fresh Z3 solver."""
        return z3.Solver()

    def test_identity_tableau_satisfies_terminal(self, solver: z3.Solver) -> None:
        """Test that identity tableau satisfies unitary terminal constraints."""
        n = 2

        # Create symbolic variables for identity tableau
        tableau_x = np.array([[z3.Bool(f"x_{r}_{q}") for q in range(n)] for r in range(2 * n)], dtype=object)
        tableau_z = np.array([[z3.Bool(f"z_{r}_{q}") for q in range(n)] for r in range(2 * n)], dtype=object)

        # Set to identity tableau
        for r in range(2 * n):
            for q in range(n):
                if r == q:
                    solver.add(tableau_x[r, q])
                    solver.add(z3.Not(tableau_z[r, q]))
                elif r == q + n:
                    solver.add(z3.Not(tableau_x[r, q]))
                    solver.add(tableau_z[r, q])
                else:
                    solver.add(z3.Not(tableau_x[r, q]))
                    solver.add(z3.Not(tableau_z[r, q]))

        # Add terminal constraints
        add_clifford_unitary_terminal(solver, n, tableau_x, tableau_z, allow_permutation=False)

        # Should be satisfiable
        assert solver.check() == z3.sat

    def test_permuted_identity_satisfies_with_permutation(self, solver: z3.Solver) -> None:
        """Test that permuted identity satisfies terminal with permutation allowed."""
        n = 2

        tableau_x = np.array([[z3.Bool(f"x_{r}_{q}") for q in range(n)] for r in range(2 * n)], dtype=object)
        tableau_z = np.array([[z3.Bool(f"z_{r}_{q}") for q in range(n)] for r in range(2 * n)], dtype=object)

        # Set to permuted identity: qubit 0 and 1 swapped
        for r in range(2 * n):
            for q in range(n):
                target_q = 1 - q  # Swap qubits
                if r == target_q:
                    solver.add(tableau_x[r, q])
                    solver.add(z3.Not(tableau_z[r, q]))
                elif r == target_q + n:
                    solver.add(z3.Not(tableau_x[r, q]))
                    solver.add(tableau_z[r, q])
                else:
                    solver.add(z3.Not(tableau_x[r, q]))
                    solver.add(z3.Not(tableau_z[r, q]))

        # Add terminal constraints with permutation allowed
        add_clifford_unitary_terminal(solver, n, tableau_x, tableau_z, allow_permutation=True)

        # Should be satisfiable
        assert solver.check() == z3.sat

    def test_permuted_identity_unsat_without_permutation(self, solver: z3.Solver) -> None:
        """Test that permuted identity is UNSAT without permutation allowed."""
        n = 2

        tableau_x = np.array([[z3.Bool(f"x_{r}_{q}") for q in range(n)] for r in range(2 * n)], dtype=object)
        tableau_z = np.array([[z3.Bool(f"z_{r}_{q}") for q in range(n)] for r in range(2 * n)], dtype=object)

        # Set to permuted identity: qubit 0 and 1 swapped
        for r in range(2 * n):
            for q in range(n):
                target_q = 1 - q
                if r == target_q:
                    solver.add(tableau_x[r, q])
                    solver.add(z3.Not(tableau_z[r, q]))
                elif r == target_q + n:
                    solver.add(z3.Not(tableau_x[r, q]))
                    solver.add(tableau_z[r, q])
                else:
                    solver.add(z3.Not(tableau_x[r, q]))
                    solver.add(z3.Not(tableau_z[r, q]))

        # Add terminal constraints without permutation
        add_clifford_unitary_terminal(solver, n, tableau_x, tableau_z, allow_permutation=False)

        # Should be unsatisfiable
        assert solver.check() == z3.unsat


class TestStabilizerStateTerminal:
    """Tests for stabilizer state terminal constraints."""

    @pytest.fixture
    def solver(self) -> z3.Solver:
        """Create a fresh Z3 solver."""
        return z3.Solver()

    def test_zero_x_part_satisfies_terminal(self, solver: z3.Solver) -> None:
        """Test that tableau with zero X-part satisfies state terminal."""
        n = 2

        tableau_x = np.array([[z3.Bool(f"x_{r}_{q}") for q in range(n)] for r in range(n)], dtype=object)
        tableau_z = np.array([[z3.Bool(f"z_{r}_{q}") for q in range(n)] for r in range(n)], dtype=object)

        # Set X-part to zero
        for r in range(n):
            for q in range(n):
                solver.add(z3.Not(tableau_x[r, q]))

        # Add terminal constraints
        add_stabilizer_state_terminal(solver, n, tableau_x, tableau_z)

        # Should be satisfiable
        assert solver.check() == z3.sat

    def test_nonzero_x_part_violates_terminal(self, solver: z3.Solver) -> None:
        """Test that tableau with nonzero X-part violates state terminal."""
        n = 2

        tableau_x = np.array([[z3.Bool(f"x_{r}_{q}") for q in range(n)] for r in range(n)], dtype=object)
        tableau_z = np.array([[z3.Bool(f"z_{r}_{q}") for q in range(n)] for r in range(n)], dtype=object)

        # Set one X-part entry to true
        solver.add(tableau_x[0, 0])

        # Add terminal constraints
        add_stabilizer_state_terminal(solver, n, tableau_x, tableau_z)

        # Should be unsatisfiable
        assert solver.check() == z3.unsat


class TestCSSTerminal:
    """Tests for CSS terminal constraints."""

    @pytest.fixture
    def solver(self) -> z3.Solver:
        """Create a fresh Z3 solver."""
        return z3.Solver()

    def test_css_state_terminal_with_pivots(self, solver: z3.Solver) -> None:
        """Test CSS state terminal with correct number of pivots."""
        n = 3
        m_x = 2

        matrix_final = np.array([[z3.Bool(f"m_{r}_{q}") for q in range(n)] for r in range(m_x)], dtype=object)

        # Set up a valid terminal matrix with 2 pivot columns
        # Row 0: [1, 0, 0]
        # Row 1: [0, 1, 0]
        solver.add(matrix_final[0, 0])
        solver.add(z3.Not(matrix_final[0, 1]))
        solver.add(z3.Not(matrix_final[0, 2]))
        solver.add(z3.Not(matrix_final[1, 0]))
        solver.add(matrix_final[1, 1])
        solver.add(z3.Not(matrix_final[1, 2]))

        # Add terminal constraints
        add_css_state_terminal(solver, n, m_x, matrix_final)

        # Should be satisfiable
        assert solver.check() == z3.sat

    def test_css_isometry_terminal_basic(self, solver: z3.Solver) -> None:
        """Test CSS isometry terminal with basic setup."""
        n = 3
        k = 1
        m_x = 1

        matrix_final = np.array([[z3.Bool(f"m_{r}_{q}") for q in range(n)] for r in range(k + m_x)], dtype=object)

        # Set up a valid terminal matrix
        # Logical row: [1, 0, 0]
        # Stabilizer row: [0, 1, 0]
        solver.add(matrix_final[0, 0])
        solver.add(z3.Not(matrix_final[0, 1]))
        solver.add(z3.Not(matrix_final[0, 2]))
        solver.add(z3.Not(matrix_final[1, 0]))
        solver.add(matrix_final[1, 1])
        solver.add(z3.Not(matrix_final[1, 2]))

        # Add terminal constraints
        add_css_isometry_terminal(solver, n, k, m_x, matrix_final)

        # Should be satisfiable
        assert solver.check() == z3.sat
