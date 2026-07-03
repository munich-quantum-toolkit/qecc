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
        for q in range(n):
            for row in range(2 * n):
                if row == q:
                    solver.add(tableau_x_final[row, q])
                    solver.add(z3.Not(tableau_z_final[row, q]))
                elif row == q + n:
                    solver.add(z3.Not(tableau_x_final[row, q]))
                    solver.add(tableau_z_final[row, q])
                else:
                    solver.add(z3.Not(tableau_x_final[row, q]))
                    solver.add(z3.Not(tableau_z_final[row, q]))
    else:
        selector = np.array([[z3.Bool(f"unitary_selector_{i}_{q}") for q in range(n)] for i in range(n)], dtype=object)

        for i in range(n):
            solver.add(z3.PbEq([(selector[i, q], 1) for q in range(n)], 1))

        for q in range(n):
            solver.add(z3.PbLe([(selector[i, q], 1) for i in range(n)], 1))

        for i in range(n):
            for q in range(n):
                x_row = i
                z_row = i + n

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

    The pivot columns (stabilizer qubits) are determined at extraction time
    by reading the satisfying model, so they are free to be any n-k columns.
    This gives the solver maximum freedom and avoids forced SWAP overhead.

    Args:
        solver: Z3 solver instance.
        n: Number of physical qubits.
        k: Number of logical qubits.
        tableau_x_final: Final X part of tableau (num_rows x n array of z3.BoolRef).
        tableau_z_final: Final Z part of tableau (num_rows x n array of z3.BoolRef).
        allow_permutation: Allow logical qubit permutation.
    """
    if k == n:
        add_clifford_unitary_terminal(solver, n, tableau_x_final, tableau_z_final, allow_permutation)
        return

    if k == 0:
        add_stabilizer_state_terminal(solver, n, tableau_x_final, tableau_z_final)
        return

    num_stab = n - k

    stab_x = tableau_x_final[2 * k : 2 * k + num_stab, :]
    stab_z = tableau_z_final[2 * k : 2 * k + num_stab, :]
    logical_x_x = tableau_x_final[:k, :]
    logical_x_z = tableau_z_final[:k, :]
    logical_z_x = tableau_x_final[k : 2 * k, :]
    logical_z_z = tableau_z_final[k : 2 * k, :]

    # 1. Stabilizer X-part must be zero
    for row in range(num_stab):
        for q in range(n):
            solver.add(z3.Not(stab_x[row, q]))

    # 2. Pivot variables: column q is a pivot iff any stabilizer row has Z on q
    pivot = np.array([z3.Bool(f"pivot_{q}") for q in range(n)], dtype=object)

    for q in range(n):
        has_support = z3.Or([stab_z[row, q] for row in range(num_stab)])
        solver.add(pivot[q] == has_support)

    # Exactly n-k pivots
    solver.add(z3.PbEq([(pivot[q], 1) for q in range(n)], num_stab))

    # 3. Logical selector variables
    if allow_permutation:
        selector = np.array([[z3.Bool(f"logical_selector_{i}_{q}") for q in range(n)] for i in range(k)], dtype=object)

        for i in range(k):
            solver.add(z3.PbEq([(selector[i, q], 1) for q in range(n)], 1))

            for q in range(n):
                solver.add(z3.Implies(selector[i, q], z3.Not(pivot[q])))

        for q in range(n):
            solver.add(z3.PbLe([(selector[i, q], 1) for i in range(k)], 1))

        for i in range(k):
            for q in range(n):
                conditions: list[z3.BoolRef] = []
                for qp in range(n):
                    conditions.extend((
                        logical_x_x[i, qp] == (q == qp),
                        logical_x_z[i, qp] == False,
                        logical_z_x[i, qp] == False,
                        logical_z_z[i, qp] == (q == qp),
                    ))

                solver.add(z3.Implies(selector[i, q], z3.And(conditions)))

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
        for i in range(k):
            solver.add(z3.Not(pivot[i]))

            for q in range(n):
                if q == i:
                    solver.add(logical_x_x[i, q])
                    solver.add(z3.Not(logical_x_z[i, q]))
                else:
                    solver.add(z3.Not(logical_x_x[i, q]))
                    solver.add(z3.Not(logical_x_z[i, q]))

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
        add_css_state_terminal(solver, n, m_x, matrix_final)
        return

    pivot = np.array([z3.Bool(f"css_pivot_{q}") for q in range(n)], dtype=object)

    for q in range(n):
        has_support = z3.Or([matrix_final[row, q] for row in range(k, k + m_x)])
        solver.add(pivot[q] == has_support)

    solver.add(z3.PbEq([(pivot[q], 1) for q in range(n)], m_x))

    for i in range(k):
        nonzero_nonpivot = [z3.And(z3.Not(pivot[q]), matrix_final[i, q]) for q in range(n)]
        solver.add(z3.PbEq([(var, 1) for var in nonzero_nonpivot], 1))

    for q in range(n):
        logical_support = [matrix_final[i, q] for i in range(k)]
        solver.add(z3.Implies(z3.Not(pivot[q]), z3.PbLe([(var, 1) for var in logical_support], 1)))


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
    pivot = np.array([z3.Bool(f"css_state_pivot_{q}") for q in range(n)], dtype=object)

    for q in range(n):
        has_support = z3.Or([matrix_final[row, q] for row in range(m_x)])
        solver.add(pivot[q] == has_support)

    solver.add(z3.PbEq([(pivot[q], 1) for q in range(n)], m_x))
