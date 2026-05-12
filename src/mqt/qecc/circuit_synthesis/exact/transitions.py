# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tableau transition constraint builders for exact synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import z3

if TYPE_CHECKING:
    import numpy.typing as npt


def add_clifford_h_transition(
    solver: z3.Solver,
    qubit: int,
    tableau_x_curr: npt.NDArray,
    tableau_z_curr: npt.NDArray,
    tableau_x_next: npt.NDArray,
    tableau_z_next: npt.NDArray,
) -> None:
    """Add H gate transition constraints for a single qubit.

    H gate swaps X and Z columns: X_i <-> Z_i.
    This function only constrains the qubit being acted upon.
    Identity constraints on other qubits must be handled by the caller.

    Args:
        solver: Z3 solver instance.
        qubit: Qubit index i to apply H gate to.
        tableau_x_curr: Current X part of tableau (num_rows x n array of z3.BoolRef).
        tableau_z_curr: Current Z part of tableau (num_rows x n array of z3.BoolRef).
        tableau_x_next: Next X part of tableau (num_rows x n array of z3.BoolRef).
        tableau_z_next: Next Z part of tableau (num_rows x n array of z3.BoolRef).
    """
    num_rows = tableau_x_curr.shape[0]

    for row in range(num_rows):
        # X[:, i] <- Z[:, i]
        solver.add(tableau_x_next[row, qubit] == tableau_z_curr[row, qubit])
        # Z[:, i] <- X[:, i]
        solver.add(tableau_z_next[row, qubit] == tableau_x_curr[row, qubit])


def add_clifford_s_transition(
    solver: z3.Solver,
    qubit: int,
    tableau_x_curr: npt.NDArray,
    tableau_z_curr: npt.NDArray,
    tableau_x_next: npt.NDArray,
    tableau_z_next: npt.NDArray,
) -> None:
    """Add S gate transition constraints for a single qubit.

    S gate: Z[:, i] <- Z[:, i] XOR X[:, i].
    This function only constrains the qubit being acted upon.
    Identity constraints on other qubits must be handled by the caller.

    Args:
        solver: Z3 solver instance.
        qubit: Qubit index i to apply S gate to.
        tableau_x_curr: Current X part of tableau (num_rows x n array of z3.BoolRef).
        tableau_z_curr: Current Z part of tableau (num_rows x n array of z3.BoolRef).
        tableau_x_next: Next X part of tableau (num_rows x n array of z3.BoolRef).
        tableau_z_next: Next Z part of tableau (num_rows x n array of z3.BoolRef).
    """
    num_rows = tableau_x_curr.shape[0]

    for row in range(num_rows):
        # X[:, i] unchanged
        solver.add(tableau_x_next[row, qubit] == tableau_x_curr[row, qubit])
        # Z[:, i] <- Z[:, i] XOR X[:, i]
        solver.add(tableau_z_next[row, qubit] == z3.Xor(tableau_z_curr[row, qubit], tableau_x_curr[row, qubit]))


def add_clifford_cx_transition(
    solver: z3.Solver,
    control: int,
    target: int,
    tableau_x_curr: npt.NDArray,
    tableau_z_curr: npt.NDArray,
    tableau_x_next: npt.NDArray,
    tableau_z_next: npt.NDArray,
) -> None:
    """Add CX gate transition constraints for control and target qubits.

    CX gate with control i and target j:
    - X[:, j] <- X[:, j] XOR X[:, i]
    - Z[:, i] <- Z[:, i] XOR Z[:, j]

    This function only constrains the qubits being acted upon.
    Identity constraints on other qubits must be handled by the caller.

    Args:
        solver: Z3 solver instance.
        control: Control qubit index i.
        target: Target qubit index j.
        tableau_x_curr: Current X part of tableau (num_rows x n array of z3.BoolRef).
        tableau_z_curr: Current Z part of tableau (num_rows x n array of z3.BoolRef).
        tableau_x_next: Next X part of tableau (num_rows x n array of z3.BoolRef).
        tableau_z_next: Next Z part of tableau (num_rows x n array of z3.BoolRef).
    """
    num_rows = tableau_x_curr.shape[0]

    for row in range(num_rows):
        # X[:, i] unchanged
        solver.add(tableau_x_next[row, control] == tableau_x_curr[row, control])
        # X[:, j] <- X[:, j] XOR X[:, i]
        solver.add(tableau_x_next[row, target] == z3.Xor(tableau_x_curr[row, target], tableau_x_curr[row, control]))
        # Z[:, i] <- Z[:, i] XOR Z[:, j]
        solver.add(tableau_z_next[row, control] == z3.Xor(tableau_z_curr[row, control], tableau_z_curr[row, target]))
        # Z[:, j] unchanged
        solver.add(tableau_z_next[row, target] == tableau_z_curr[row, target])


def add_clifford_identity_transition(
    solver: z3.Solver,
    qubit: int,
    tableau_x_curr: npt.NDArray,
    tableau_z_curr: npt.NDArray,
    tableau_x_next: npt.NDArray,
    tableau_z_next: npt.NDArray,
) -> None:
    """Add identity transition constraints for a single qubit.

    Identity: the specified qubit column remains unchanged.

    Args:
        solver: Z3 solver instance.
        qubit: Qubit index to assert identity on.
        tableau_x_curr: Current X part of tableau (num_rows x n array of z3.BoolRef).
        tableau_z_curr: Current Z part of tableau (num_rows x n array of z3.BoolRef).
        tableau_x_next: Next X part of tableau (num_rows x n array of z3.BoolRef).
        tableau_z_next: Next Z part of tableau (num_rows x n array of z3.BoolRef).
    """
    num_rows = tableau_x_curr.shape[0]

    for row in range(num_rows):
        solver.add(tableau_x_next[row, qubit] == tableau_x_curr[row, qubit])
        solver.add(tableau_z_next[row, qubit] == tableau_z_curr[row, qubit])


def add_full_tableau_identity(
    solver: z3.Solver,
    n: int,
    tableau_x_curr: npt.NDArray,
    tableau_z_curr: npt.NDArray,
    tableau_x_next: npt.NDArray,
    tableau_z_next: npt.NDArray,
) -> None:
    """Add identity transition constraints for all qubits.

    This asserts that the entire tableau remains unchanged.
    Useful for gate-count encoding where unaffected qubits must stay the same.

    Args:
        solver: Z3 solver instance.
        n: Total number of qubits.
        tableau_x_curr: Current X part of tableau (num_rows x n array of z3.BoolRef).
        tableau_z_curr: Current Z part of tableau (num_rows x n array of z3.BoolRef).
        tableau_x_next: Next X part of tableau (num_rows x n array of z3.BoolRef).
        tableau_z_next: Next Z part of tableau (num_rows x n array of z3.BoolRef).
    """
    num_rows = tableau_x_curr.shape[0]

    for q in range(n):
        for row in range(num_rows):
            solver.add(tableau_x_next[row, q] == tableau_x_curr[row, q])
            solver.add(tableau_z_next[row, q] == tableau_z_curr[row, q])


def add_css_cnot_transition(
    solver: z3.Solver,
    control: int,
    target: int,
    matrix_curr: npt.NDArray,
    matrix_next: npt.NDArray,
) -> None:
    """Add CSS CNOT transition constraints for control and target qubits.

    CSS CNOT: M[:, j] <- M[:, j] XOR M[:, i].

    This function only constrains the qubits being acted upon.
    Identity constraints on other qubits must be handled by the caller.

    Args:
        solver: Z3 solver instance.
        control: Control qubit index i.
        target: Target qubit index j.
        matrix_curr: Current CSS matrix (num_rows x n array of z3.BoolRef).
        matrix_next: Next CSS matrix (num_rows x n array of z3.BoolRef).
    """
    num_rows = matrix_curr.shape[0]

    for row in range(num_rows):
        # M[:, i] unchanged
        solver.add(matrix_next[row, control] == matrix_curr[row, control])
        # M[:, j] <- M[:, j] XOR M[:, i]
        solver.add(matrix_next[row, target] == z3.Xor(matrix_curr[row, target], matrix_curr[row, control]))


def add_css_identity_transition(
    solver: z3.Solver,
    qubit: int,
    matrix_curr: npt.NDArray,
    matrix_next: npt.NDArray,
) -> None:
    """Add CSS identity transition constraints for a single qubit.

    CSS identity: the specified qubit column remains unchanged.

    Args:
        solver: Z3 solver instance.
        qubit: Qubit index to assert identity on.
        matrix_curr: Current CSS matrix (num_rows x n array of z3.BoolRef).
        matrix_next: Next CSS matrix (num_rows x n array of z3.BoolRef).
    """
    num_rows = matrix_curr.shape[0]

    for row in range(num_rows):
        solver.add(matrix_next[row, qubit] == matrix_curr[row, qubit])


def add_css_full_identity(
    solver: z3.Solver,
    n: int,
    matrix_curr: npt.NDArray,
    matrix_next: npt.NDArray,
) -> None:
    """Add CSS identity transition constraints for all qubits.

    This asserts that the entire CSS matrix remains unchanged.
    Useful for gate-count encoding where unaffected qubits must stay the same.

    Args:
        solver: Z3 solver instance.
        n: Total number of qubits.
        matrix_curr: Current CSS matrix (num_rows x n array of z3.BoolRef).
        matrix_next: Next CSS matrix (num_rows x n array of z3.BoolRef).
    """
    num_rows = matrix_curr.shape[0]

    for q in range(n):
        for row in range(num_rows):
            solver.add(matrix_next[row, q] == matrix_curr[row, q])
