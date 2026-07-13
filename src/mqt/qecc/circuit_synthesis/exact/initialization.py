# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Initial-state constraints shared by the gate-count and depth encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import z3

    from ...codes.pauli import CheckMatrix, StabilizerTableau


def constrain_initial_clifford_tableau(
    solver: z3.Solver,
    target: StabilizerTableau,
    tableau_x: npt.NDArray[np.object_],
    tableau_z: npt.NDArray[np.object_],
    n: int,
    num_rows: int,
) -> None:
    """Constrain the step-0 tableau variables to equal the target stabilizer tableau."""
    for row in range(num_rows):
        for q in range(n):
            solver.add(tableau_x[0, row, q] == bool(target.tableau.matrix[row, q]))
            solver.add(tableau_z[0, row, q] == bool(target.tableau.matrix[row, q + n]))


def constrain_initial_css_matrix(
    solver: z3.Solver,
    target: CheckMatrix,
    matrix: npt.NDArray[np.object_],
    n: int,
    num_rows: int,
) -> None:
    """Constrain the step-0 check-matrix variables to equal the target check matrix."""
    for row in range(num_rows):
        for q in range(n):
            solver.add(matrix[0, row, q] == bool(target.matrix[row, q]))
