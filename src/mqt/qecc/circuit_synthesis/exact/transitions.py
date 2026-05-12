# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tableau transition constraint builders for exact synthesis."""

from __future__ import annotations

import z3


def add_clifford_h_transition(
    solver: z3.Solver,
    qubit: int,
    n: int,
    tableau_x_curr: list[list[z3.BoolRef]],
    tableau_z_curr: list[list[z3.BoolRef]],
    tableau_x_next: list[list[z3.BoolRef]],
    tableau_z_next: list[list[z3.BoolRef]],
) -> None:
    """Add H gate transition constraints.

    H gate swaps X and Z columns: X_i <-> Z_i.

    Args:
        solver: Z3 solver instance.
        qubit: Qubit index i to apply H gate to.
        n: Total number of qubits.
        tableau_x_curr: Current X part of tableau.
        tableau_z_curr: Current Z part of tableau.
        tableau_x_next: Next X part of tableau.
        tableau_z_next: Next Z part of tableau.
    """
    num_rows = len(tableau_x_curr)

    for row in range(num_rows):
        # X[:, i] <- Z[:, i]
        solver.add(tableau_x_next[row][qubit] == tableau_z_curr[row][qubit])
        # Z[:, i] <- X[:, i]
        solver.add(tableau_z_next[row][qubit] == tableau_x_curr[row][qubit])

    # All other columns unchanged
    for q in range(n):
        if q == qubit:
            continue
        for row in range(num_rows):
            solver.add(tableau_x_next[row][q] == tableau_x_curr[row][q])
            solver.add(tableau_z_next[row][q] == tableau_z_curr[row][q])


def add_clifford_s_transition(
    solver: z3.Solver,
    qubit: int,
    n: int,
    tableau_x_curr: list[list[z3.BoolRef]],
    tableau_z_curr: list[list[z3.BoolRef]],
    tableau_x_next: list[list[z3.BoolRef]],
    tableau_z_next: list[list[z3.BoolRef]],
) -> None:
    """Add S gate transition constraints.

    S gate: Z[:, i] <- Z[:, i] XOR X[:, i].

    Args:
        solver: Z3 solver instance.
        qubit: Qubit index i to apply S gate to.
        n: Total number of qubits.
        tableau_x_curr: Current X part of tableau.
        tableau_z_curr: Current Z part of tableau.
        tableau_x_next: Next X part of tableau.
        tableau_z_next: Next Z part of tableau.
    """
    num_rows = len(tableau_x_curr)

    for row in range(num_rows):
        # X[:, i] unchanged
        solver.add(tableau_x_next[row][qubit] == tableau_x_curr[row][qubit])
        # Z[:, i] <- Z[:, i] XOR X[:, i]
        solver.add(tableau_z_next[row][qubit] == z3.Xor(tableau_z_curr[row][qubit], tableau_x_curr[row][qubit]))

    # All other columns unchanged
    for q in range(n):
        if q == qubit:
            continue
        for row in range(num_rows):
            solver.add(tableau_x_next[row][q] == tableau_x_curr[row][q])
            solver.add(tableau_z_next[row][q] == tableau_z_curr[row][q])


def add_clifford_cx_transition(
    solver: z3.Solver,
    control: int,
    target: int,
    n: int,
    tableau_x_curr: list[list[z3.BoolRef]],
    tableau_z_curr: list[list[z3.BoolRef]],
    tableau_x_next: list[list[z3.BoolRef]],
    tableau_z_next: list[list[z3.BoolRef]],
) -> None:
    """Add CX gate transition constraints.

    CX gate with control i and target j:
    - X[:, j] <- X[:, j] XOR X[:, i]
    - Z[:, i] <- Z[:, i] XOR Z[:, j]

    Args:
        solver: Z3 solver instance.
        control: Control qubit index i.
        target: Target qubit index j.
        n: Total number of qubits.
        tableau_x_curr: Current X part of tableau.
        tableau_z_curr: Current Z part of tableau.
        tableau_x_next: Next X part of tableau.
        tableau_z_next: Next Z part of tableau.
    """
    num_rows = len(tableau_x_curr)

    for row in range(num_rows):
        # X[:, i] unchanged
        solver.add(tableau_x_next[row][control] == tableau_x_curr[row][control])
        # X[:, j] <- X[:, j] XOR X[:, i]
        solver.add(tableau_x_next[row][target] == z3.Xor(tableau_x_curr[row][target], tableau_x_curr[row][control]))
        # Z[:, i] <- Z[:, i] XOR Z[:, j]
        solver.add(tableau_z_next[row][control] == z3.Xor(tableau_z_curr[row][control], tableau_z_curr[row][target]))
        # Z[:, j] unchanged
        solver.add(tableau_z_next[row][target] == tableau_z_curr[row][target])

    # All other columns unchanged
    for q in range(n):
        if q in {control, target}:
            continue
        for row in range(num_rows):
            solver.add(tableau_x_next[row][q] == tableau_x_curr[row][q])
            solver.add(tableau_z_next[row][q] == tableau_z_curr[row][q])


def add_clifford_identity_transition(
    solver: z3.Solver,
    qubit: int,
    n: int,
    tableau_x_curr: list[list[z3.BoolRef]],
    tableau_z_curr: list[list[z3.BoolRef]],
    tableau_x_next: list[list[z3.BoolRef]],
    tableau_z_next: list[list[z3.BoolRef]],
) -> None:
    """Add identity transition constraints.

    Identity: all columns unchanged.

    Args:
        solver: Z3 solver instance.
        qubit: Qubit index (for consistency, though identity affects all).
        n: Total number of qubits.
        tableau_x_curr: Current X part of tableau.
        tableau_z_curr: Current Z part of tableau.
        tableau_x_next: Next X part of tableau.
        tableau_z_next: Next Z part of tableau.
    """
    num_rows = len(tableau_x_curr)

    for q in range(n):
        for row in range(num_rows):
            solver.add(tableau_x_next[row][q] == tableau_x_curr[row][q])
            solver.add(tableau_z_next[row][q] == tableau_z_curr[row][q])


def add_css_cnot_transition(
    solver: z3.Solver,
    control: int,
    target: int,
    n: int,
    matrix_curr: list[list[z3.BoolRef]],
    matrix_next: list[list[z3.BoolRef]],
) -> None:
    """Add CSS CNOT transition constraints.

    CSS CNOT: M[:, j] <- M[:, j] XOR M[:, i].

    Args:
        solver: Z3 solver instance.
        control: Control qubit index i.
        target: Target qubit index j.
        n: Total number of qubits.
        matrix_curr: Current CSS matrix.
        matrix_next: Next CSS matrix.
    """
    num_rows = len(matrix_curr)

    for row in range(num_rows):
        # M[:, i] unchanged
        solver.add(matrix_next[row][control] == matrix_curr[row][control])
        # M[:, j] <- M[:, j] XOR M[:, i]
        solver.add(matrix_next[row][target] == z3.Xor(matrix_curr[row][target], matrix_curr[row][control]))

    # All other columns unchanged
    for q in range(n):
        if q in {control, target}:
            continue
        for row in range(num_rows):
            solver.add(matrix_next[row][q] == matrix_curr[row][q])


def add_css_identity_transition(
    solver: z3.Solver,
    qubit: int,
    n: int,
    matrix_curr: list[list[z3.BoolRef]],
    matrix_next: list[list[z3.BoolRef]],
) -> None:
    """Add CSS identity transition constraints.

    CSS identity: all columns unchanged.

    Args:
        solver: Z3 solver instance.
        qubit: Qubit index (for consistency).
        n: Total number of qubits.
        matrix_curr: Current CSS matrix.
        matrix_next: Next CSS matrix.
    """
    num_rows = len(matrix_curr)

    for q in range(n):
        for row in range(num_rows):
            solver.add(matrix_next[row][q] == matrix_curr[row][q])
