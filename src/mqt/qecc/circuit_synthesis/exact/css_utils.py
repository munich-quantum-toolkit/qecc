# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for CSS circuit synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import z3


def row_echelon_pivot_cols(matrix: np.ndarray) -> list[int]:
    """Compute row echelon form and return pivot column indices.

    Args:
        matrix: Binary matrix (m x n) with dtype np.int8.

    Returns:
        List of column indices that contain pivots in row echelon form.
    """
    mat = matrix.copy()
    m, n = mat.shape
    pivot_cols = []
    current_row = 0

    for col in range(n):
        pivot_found = False
        for row in range(current_row, m):
            if mat[row, col] == 1:
                if row != current_row:
                    mat[[current_row, row]] = mat[[row, current_row]]
                pivot_found = True
                break

        if not pivot_found:
            continue

        pivot_cols.append(col)

        for row in range(m):
            if row != current_row and mat[row, col] == 1:
                mat[row] ^= mat[current_row]

        current_row += 1
        if current_row >= m:
            break

    return pivot_cols


def determine_css_initializations(
    model: z3.ModelRef,
    n: int,
    num_rows: int,
    k: int,
    matrix_vars: np.ndarray,
    is_x_type: bool,
) -> tuple[list[int], list[int]]:
    """Determine which qubits to initialize based on terminal tableau.

    Args:
        model: Z3 model from satisfiable formula.
        n: Number of qubits.
        num_rows: Number of rows in check matrix.
        k: Number of logical qubits.
        matrix_vars: Boolean matrix variables from encoding.
        is_x_type: Whether target is X-type check matrix.

    Returns:
        Tuple of (init_x, init_z) lists.
    """
    final_matrix = np.array(
        [[bool(model.eval(matrix_vars[row, q], model_completion=True)) for q in range(n)] for row in range(num_rows)],
        dtype=np.int8,
    )

    m = num_rows - k

    if m == 0:
        if is_x_type:
            return list(range(k, n)), []
        return [], list(range(k, n))

    logical_part = final_matrix[:k]
    stabilizer_part = final_matrix[k:]

    stabilizer_pivot_cols = row_echelon_pivot_cols(stabilizer_part)

    input_qubits = []
    for col in range(n):
        if col in stabilizer_pivot_cols:
            continue
        for row in range(k):
            if logical_part[row, col] == 1:
                input_qubits.append(col)
                break

    ancilla_qubits = [q for q in range(n) if q not in input_qubits]

    init_x: list[int] = []
    init_z: list[int] = []

    if is_x_type:
        init_x = [q for q in stabilizer_pivot_cols if q in ancilla_qubits]
        init_z = [q for q in ancilla_qubits if q not in init_x]
    else:
        init_z = [q for q in stabilizer_pivot_cols if q in ancilla_qubits]
        init_x = [q for q in ancilla_qubits if q not in init_z]

    return init_x, init_z
