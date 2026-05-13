# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for tableau transition constraints."""

from __future__ import annotations

import numpy as np
import pytest
import z3

from mqt.qecc.circuit_synthesis.exact.transitions import (
    add_clifford_cx_transition,
    add_clifford_h_transition,
    add_clifford_identity_transition,
    add_clifford_s_transition,
    add_css_cnot_transition,
    add_css_identity_transition,
)
from mqt.qecc.codes.pauli import StabilizerTableau


@pytest.fixture
def solver() -> z3.Solver:
    """Create a fresh Z3 solver."""
    return z3.Solver()


@pytest.fixture
def tableau_vars_2q() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create symbolic tableau variables for 2 qubits, 2 rows."""
    n = 2
    num_rows = 2

    curr_x = np.array([[z3.Bool(f"curr_x_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    curr_z = np.array([[z3.Bool(f"curr_z_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    next_x = np.array([[z3.Bool(f"next_x_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    next_z = np.array([[z3.Bool(f"next_z_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)

    return curr_x, curr_z, next_x, next_z


@pytest.fixture
def css_matrix_vars_3q() -> tuple[np.ndarray, np.ndarray]:
    """Create symbolic CSS matrix variables for 3 qubits, 2 rows."""
    n = 3
    num_rows = 2

    curr = np.array([[z3.Bool(f"curr_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    next_m = np.array([[z3.Bool(f"next_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)

    return curr, next_m


def test_h_transition_swaps_x_and_z(
    solver: z3.Solver, tableau_vars_2q: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Test that H gate transition swaps X and Z columns."""
    curr_x, curr_z, next_x, next_z = tableau_vars_2q

    solver.add(curr_x[0, 0])
    solver.add(z3.Not(curr_z[0, 0]))

    add_clifford_h_transition(solver, 0, curr_x, curr_z, next_x, next_z)

    assert solver.check() == z3.sat
    model = solver.model()

    assert not model.eval(next_x[0, 0], model_completion=True)
    assert model.eval(next_z[0, 0], model_completion=True)


def test_s_transition_updates_z_column(
    solver: z3.Solver, tableau_vars_2q: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Test that S gate transition updates Z column correctly."""
    curr_x, curr_z, next_x, next_z = tableau_vars_2q

    solver.add(curr_x[0, 0])
    solver.add(z3.Not(curr_z[0, 0]))

    add_clifford_s_transition(solver, 0, curr_x, curr_z, next_x, next_z)

    assert solver.check() == z3.sat
    model = solver.model()

    assert model.eval(next_x[0, 0], model_completion=True)
    assert model.eval(next_z[0, 0], model_completion=True)


def test_cx_transition_updates_both_qubits(
    solver: z3.Solver, tableau_vars_2q: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Test that CX gate transition updates control and target correctly."""
    curr_x, curr_z, next_x, next_z = tableau_vars_2q

    solver.add(curr_x[0, 0])
    solver.add(z3.Not(curr_x[0, 1]))
    solver.add(z3.Not(curr_z[0, 0]))
    solver.add(z3.Not(curr_z[0, 1]))

    add_clifford_cx_transition(solver, 0, 1, curr_x, curr_z, next_x, next_z)

    assert solver.check() == z3.sat
    model = solver.model()

    assert model.eval(next_x[0, 0], model_completion=True)
    assert model.eval(next_x[0, 1], model_completion=True)
    assert not model.eval(next_z[0, 0], model_completion=True)
    assert not model.eval(next_z[0, 1], model_completion=True)


def test_identity_transition_preserves_tableau(
    solver: z3.Solver, tableau_vars_2q: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> None:
    """Test that identity transition preserves the tableau."""
    curr_x, curr_z, next_x, next_z = tableau_vars_2q

    solver.add(curr_x[0, 0])
    solver.add(z3.Not(curr_z[0, 0]))

    add_clifford_identity_transition(solver, 0, curr_x, curr_z, next_x, next_z)

    assert solver.check() == z3.sat
    model = solver.model()

    assert model.eval(next_x[0, 0], model_completion=True)
    assert not model.eval(next_z[0, 0], model_completion=True)


def test_transitions_match_stim_simulation() -> None:
    """Test that symbolic transitions match Stim tableau simulation."""
    init_tableau = StabilizerTableau.identity(2)

    stim_tableau = init_tableau.copy()
    stim_tableau.apply_h(0)

    solver = z3.Solver()
    n = 2
    num_rows = 4

    curr_x = np.array([[z3.Bool(f"curr_x_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    curr_z = np.array([[z3.Bool(f"curr_z_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    next_x = np.array([[z3.Bool(f"next_x_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    next_z = np.array([[z3.Bool(f"next_z_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)

    for r in range(num_rows):
        for q in range(n):
            solver.add(curr_x[r, q] == bool(init_tableau.tableau.matrix[r, q]))
            solver.add(curr_z[r, q] == bool(init_tableau.tableau.matrix[r, q + n]))

    add_clifford_h_transition(solver, 0, curr_x, curr_z, next_x, next_z)

    for r in range(num_rows):
        solver.add(next_x[r, 1] == curr_x[r, 1])
        solver.add(next_z[r, 1] == curr_z[r, 1])

    assert solver.check() == z3.sat
    model = solver.model()

    for r in range(num_rows):
        for q in range(n):
            symbolic_x = model.eval(next_x[r, q], model_completion=True)
            symbolic_z = model.eval(next_z[r, q], model_completion=True)
            stim_x = bool(stim_tableau.tableau.matrix[r, q])
            stim_z = bool(stim_tableau.tableau.matrix[r, q + n])

            assert z3.is_true(symbolic_x) == stim_x
            assert z3.is_true(symbolic_z) == stim_z


def test_css_cnot_updates_target_column(solver: z3.Solver, css_matrix_vars_3q: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that CSS CNOT updates target column correctly."""
    curr, next_m = css_matrix_vars_3q

    solver.add(curr[0, 0])
    solver.add(z3.Not(curr[1, 0]))
    solver.add(z3.Not(curr[0, 1]))
    solver.add(curr[1, 1])

    add_css_cnot_transition(solver, 0, 1, curr, next_m)

    assert solver.check() == z3.sat
    model = solver.model()

    assert model.eval(next_m[0, 0], model_completion=True)
    assert not model.eval(next_m[1, 0], model_completion=True)
    assert model.eval(next_m[0, 1], model_completion=True)
    assert model.eval(next_m[1, 1], model_completion=True)


def test_css_identity_preserves_matrix(solver: z3.Solver, css_matrix_vars_3q: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that CSS identity preserves the matrix."""
    curr, next_m = css_matrix_vars_3q

    solver.add(curr[0, 0])
    solver.add(z3.Not(curr[1, 0]))

    add_css_identity_transition(solver, 0, curr, next_m)

    assert solver.check() == z3.sat
    model = solver.model()

    assert model.eval(next_m[0, 0], model_completion=True)
    assert not model.eval(next_m[1, 0], model_completion=True)
