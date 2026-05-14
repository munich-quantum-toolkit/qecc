# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Terminal constraint builders for exact synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import z3

if TYPE_CHECKING:
    import numpy.typing as npt


def add_clifford_unitary_terminal(
    solver: z3.Solver,
    n: int,
    tableau_x_final: npt.NDArray[np.object_],
    tableau_z_final: npt.NDArray[np.object_],
    allow_permutation: bool = True,
) -> None:
    """Add terminal constraints for Clifford unitary synthesis.

    For k=n (no stabilizer rows), the final tableau should be identity
    up to qubit permutation if allowed.

    Args:
        solver: Z3 solver instance.
        n: Number of qubits.
        tableau_x_final: Final X part of tableau (2n x n array of z3.BoolRef).
        tableau_z_final: Final Z part of tableau (2n x n array of z3.BoolRef).
        allow_permutation: Allow final qubit permutation.
    """
    if not allow_permutation:
        # Require exact identity tableau
        for q in range(n):
            for row in range(2 * n):
                if row == q:
                    # X-part: row q has X on qubit q
                    solver.add(tableau_x_final[row, q])
                    solver.add(z3.Not(tableau_z_final[row, q]))
                elif row == q + n:
                    # Z-part: row q+n has Z on qubit q
                    solver.add(z3.Not(tableau_x_final[row, q]))
                    solver.add(tableau_z_final[row, q])
                else:
                    # All other entries are identity
                    solver.add(z3.Not(tableau_x_final[row, q]))
                    solver.add(z3.Not(tableau_z_final[row, q]))
    else:
        # Allow qubit permutation using selector variables
        # For each logical qubit i, select which physical qubit q carries it
        selector = np.array([[z3.Bool(f"unitary_selector_{i}_{q}") for q in range(n)] for i in range(n)], dtype=object)

        # Each logical qubit selects exactly one physical qubit
        for i in range(n):
            solver.add(z3.PbEq([(selector[i, q], 1) for q in range(n)], 1))

        # Each physical qubit is selected by at most one logical qubit
        for q in range(n):
            solver.add(z3.PbLe([(selector[i, q], 1) for i in range(n)], 1))

        # If logical qubit i selects physical qubit q, require canonical X/Z pair
        for i in range(n):
            for q in range(n):
                x_row = i
                z_row = i + n

                # selector[i,q] => X part has X_q for row i, Z part has Z_q for row i+n
                solver.add(
                    z3.Implies(
                        selector[i, q],
                        z3.And(
                            tableau_x_final[x_row, q],
                            z3.Not(tableau_z_final[x_row, q]),
                            z3.Not(tableau_x_final[z_row, q]),
                            tableau_z_final[z_row, q],
                        ),
                    )
                )

                # If not selected, those entries must be identity
                solver.add(
                    z3.Implies(
                        z3.Not(selector[i, q]),
                        z3.And(
                            z3.Not(tableau_x_final[x_row, q]),
                            z3.Not(tableau_z_final[x_row, q]),
                            z3.Not(tableau_x_final[z_row, q]),
                            z3.Not(tableau_z_final[z_row, q]),
                        ),
                    )
                )


def add_clifford_isometry_terminal(
    solver: z3.Solver,
    n: int,
    k: int,
    tableau_x_final: npt.NDArray[np.object_],
    tableau_z_final: npt.NDArray[np.object_],
    allow_permutation: bool = True,
) -> None:
    """Add terminal constraints for general Clifford isometry synthesis.

    For k < n, require:
    - Stabilizer X-part is zero
    - Exactly n-k pivot columns in stabilizer Z-part
    - Each logical qubit selects one non-pivot column with canonical X/Z pair
    - Non-pivot non-selected entries vanish
    - Pivot columns have no logical X-support

    Args:
        solver: Z3 solver instance.
        n: Number of physical qubits.
        k: Number of logical qubits.
        tableau_x_final: Final X part of tableau (num_rows x n array of z3.BoolRef).
        tableau_z_final: Final Z part of tableau (num_rows x n array of z3.BoolRef).
        allow_permutation: Allow logical qubit permutation.
    """
    if k == n:
        # This is actually a unitary
        add_clifford_unitary_terminal(solver, n, tableau_x_final, tableau_z_final, allow_permutation)
        return

    if k == 0:
        # Stabilizer state preparation
        add_stabilizer_state_terminal(solver, n, tableau_x_final, tableau_z_final)
        return

    # General isometry case: k < n
    num_stab = n - k

    # Stabilizer rows are rows 2k to 2k + num_stab - 1
    stab_x = tableau_x_final[2 * k : 2 * k + num_stab, :]
    stab_z = tableau_z_final[2 * k : 2 * k + num_stab, :]

    # Logical X rows: 0 to k-1
    logical_x_x = tableau_x_final[:k, :]
    logical_x_z = tableau_z_final[:k, :]

    # Logical Z rows: k to 2k-1
    logical_z_x = tableau_x_final[k : 2 * k, :]
    logical_z_z = tableau_z_final[k : 2 * k, :]

    # 1. Stabilizer X-part must be zero
    for row in range(num_stab):
        for q in range(n):
            solver.add(z3.Not(stab_x[row, q]))

    # 2. Introduce pivot variables for stabilizer Z columns
    pivot = np.array([z3.Bool(f"pivot_{q}") for q in range(n)], dtype=object)

    for q in range(n):
        # pivot_q <=> OR over stabilizer rows of stab_z[row,q]
        has_support = z3.Or([stab_z[row, q] for row in range(num_stab)])
        solver.add(pivot[q] == has_support)

    # Exactly n-k pivots
    solver.add(z3.PbEq([(pivot[q], 1) for q in range(n)], num_stab))

    # 3. Logical selector variables
    if allow_permutation:
        # Each logical qubit selects one non-pivot column
        selector = np.array([[z3.Bool(f"logical_selector_{i}_{q}") for q in range(n)] for i in range(k)], dtype=object)

        for i in range(k):
            # Each logical selects exactly one column
            solver.add(z3.PbEq([(selector[i, q], 1) for q in range(n)], 1))

            for q in range(n):
                # Selected column must be non-pivot
                solver.add(z3.Implies(selector[i, q], z3.Not(pivot[q])))

        # Each non-pivot column selected by at most one logical qubit
        for q in range(n):
            solver.add(z3.PbLe([(selector[i, q], 1) for i in range(k)], 1))

        # If logical i selects column q, require canonical X/Z pair
        for i in range(k):
            for q in range(n):
                # Build canonical X and Z vectors for logical i
                # Logical X should have X on qubit q only
                x_canonical = [q == qp for qp in range(n)]
                z_canonical_x = [False for _ in range(n)]

                # Logical Z should have Z on qubit q only
                z_canonical_x_part = [False for _ in range(n)]
                z_canonical_z = [q == qp for qp in range(n)]

                conditions: list[z3.BoolRef] = []
                for qp in range(n):
                    conditions.extend((
                        logical_x_x[i, qp] == x_canonical[qp],
                        logical_x_z[i, qp] == z_canonical_x[qp],
                        logical_z_x[i, qp] == z_canonical_x_part[qp],
                        logical_z_z[i, qp] == z_canonical_z[qp],
                    ))

                solver.add(z3.Implies(selector[i, q], z3.And(conditions)))

                # If not selected and not pivot, entries must vanish
                solver.add(
                    z3.Implies(
                        z3.And(z3.Not(pivot[q]), z3.Not(selector[i, q])),
                        z3.And(
                            z3.Not(logical_x_x[i, q]),
                            z3.Not(logical_x_z[i, q]),
                            z3.Not(logical_z_x[i, q]),
                            z3.Not(logical_z_z[i, q]),
                        ),
                    )
                )

        # 4. Pivot columns: no logical X-support
        for q in range(n):
            for i in range(k):
                solver.add(z3.Implies(pivot[q], z3.Not(logical_x_x[i, q])))
                solver.add(z3.Implies(pivot[q], z3.Not(logical_z_x[i, q])))
    else:
        # Fixed qubit order: logical i must be on qubit i
        for i in range(k):
            # Qubit i must not be a pivot
            solver.add(z3.Not(pivot[i]))

            # Logical X row i should have X on qubit i
            for q in range(n):
                if q == i:
                    solver.add(logical_x_x[i, q])
                    solver.add(z3.Not(logical_x_z[i, q]))
                else:
                    solver.add(z3.Not(logical_x_x[i, q]))
                    solver.add(z3.Not(logical_x_z[i, q]))

            # Logical Z row i should have Z on qubit i
            for q in range(n):
                if q == i:
                    solver.add(z3.Not(logical_z_x[i, q]))
                    solver.add(logical_z_z[i, q])
                else:
                    solver.add(z3.Not(logical_z_x[i, q]))
                    solver.add(z3.Not(logical_z_z[i, q]))


def add_stabilizer_state_terminal(
    solver: z3.Solver,
    n: int,
    tableau_x_final: npt.NDArray[np.object_],
    _tableau_z_final: npt.NDArray[np.object_],
) -> None:
    """Add terminal constraints for stabilizer-state preparation.

    For k=0, require stabilizer X-part to be zero.

    Args:
        solver: Z3 solver instance.
        n: Number of qubits.
        tableau_x_final: Final X part of tableau (n x n array of z3.BoolRef).
        _tableau_z_final: Final Z part of tableau (unused for stabilizer state terminal).
    """
    # All stabilizer X entries must be zero
    for row in range(n):
        for q in range(n):
            solver.add(z3.Not(tableau_x_final[row, q]))


def add_css_isometry_terminal(
    solver: z3.Solver,
    n: int,
    k: int,
    m_x: int,
    matrix_final: npt.NDArray[np.object_],
) -> None:
    """Add terminal constraints for CSS CNOT isometry synthesis.

    Args:
        solver: Z3 solver instance.
        n: Number of physical qubits.
        k: Number of logical qubits.
        m_x: Number of independent X-stabilizer generators (rank of H_X).
        matrix_final: Final CSS matrix [L; H] (num_rows x n array of z3.BoolRef).
    """
    if k == 0:
        # CSS state preparation
        add_css_state_terminal(solver, n, m_x, matrix_final)
        return

    # General CSS isometry
    # Logical rows: 0 to k-1
    # Stabilizer rows: k to k + m_x - 1

    # 1. Introduce pivot variables for stabilizer columns
    pivot = np.array([z3.Bool(f"css_pivot_{q}") for q in range(n)], dtype=object)

    for q in range(n):
        # pivot_q <=> OR over stabilizer rows
        has_support = z3.Or([matrix_final[row, q] for row in range(k, k + m_x)])
        solver.add(pivot[q] == has_support)

    # Exactly m_x pivots
    solver.add(z3.PbEq([(pivot[q], 1) for q in range(n)], m_x))

    # 2. Each logical row must have exactly one nonzero entry on a non-pivot column
    for i in range(k):
        nonzero_nonpivot = [z3.And(z3.Not(pivot[q]), matrix_final[i, q]) for q in range(n)]
        solver.add(z3.PbEq([(var, 1) for var in nonzero_nonpivot], 1))

    # 3. No non-pivot column may be used by two different logical rows
    for q in range(n):
        logical_support = [matrix_final[i, q] for i in range(k)]
        solver.add(z3.Implies(z3.Not(pivot[q]), z3.PbLe([(var, 1) for var in logical_support], 1)))

    # No restriction on pivot columns for logical rows


def add_css_state_terminal(
    solver: z3.Solver,
    n: int,
    m_x: int,
    matrix_final: npt.NDArray[np.object_],
) -> None:
    """Add terminal constraints for CSS state preparation.

    For k=0, the matrix contains only stabilizer rows.
    Require exactly m_x pivot columns.

    Args:
        solver: Z3 solver instance.
        n: Number of qubits.
        m_x: Number of independent X-stabilizers.
        matrix_final: Final CSS matrix (m_x x n array of z3.BoolRef).
    """
    # Introduce pivot variables
    pivot = np.array([z3.Bool(f"css_state_pivot_{q}") for q in range(n)], dtype=object)

    for q in range(n):
        has_support = z3.Or([matrix_final[row, q] for row in range(m_x)])
        solver.add(pivot[q] == has_support)

    # Exactly m_x pivots
    solver.add(z3.PbEq([(pivot[q], 1) for q in range(n)], m_x))
