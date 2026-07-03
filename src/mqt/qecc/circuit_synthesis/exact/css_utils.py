# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for CSS circuit synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2
import numpy as np

if TYPE_CHECKING:
    import z3


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

    stabilizer_pivot_cols = mod2.row_echelon(stabilizer_part, full=True)[3]

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
