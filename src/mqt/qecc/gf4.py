# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utilities for additive linear algebra over GF(4).

GF(4) is the field with four elements, which has characteristic two: addition is
bitwise XOR, not integer arithmetic modulo four. The functions here mirror those
of :mod:`mqt.qecc.mod2` for matrices whose entries encode GF(4) elements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def row_echelon(
    matrix: npt.NDArray[np.integer],
    *,
    full: bool = False,
) -> tuple[npt.NDArray[np.integer], int]:
    """Convert a GF(4) matrix to row echelon form over GF(2).

    The input is assumed to be a dense integer matrix whose entries encode
    GF(4) elements as two-bit integers from zero to three. Since the considered
    codes are GF(2)-additive, row addition is bitwise XOR and pivots are selected
    independently for both binary components. The input is copied internally so
    the caller's array is never modified.

    Args:
        matrix: GF(4) matrix to reduce.
        full: If ``True``, eliminate entries above and below each pivot
            (reduced row echelon form). Otherwise only eliminate below.

    Returns:
        A tuple containing

        - the row echelon form,
        - the binary rank.

        The order matches :func:`mqt.qecc.mod2.row_echelon`, which additionally
        returns a transformation matrix and the pivot columns.
    """
    num_rows, num_cols = matrix.shape
    if num_rows == 0:
        return np.copy(matrix), 0

    the_matrix = np.copy(matrix)
    pivot_row = 0

    for bit_col in range(2 * num_cols):
        col = bit_col % num_cols
        bit = bit_col // num_cols

        pivot = next((row for row in range(pivot_row, num_rows) if (the_matrix[row, col] >> bit) & 1), None)
        if pivot is None:
            continue

        if pivot != pivot_row:
            the_matrix[[pivot, pivot_row]] = the_matrix[[pivot_row, pivot]]

        elimination_range = range(num_rows) if full else range(pivot_row + 1, num_rows)
        for row in elimination_range:
            if full and row == pivot_row:
                continue
            if (the_matrix[row, col] >> bit) & 1:
                the_matrix[row] ^= the_matrix[pivot_row]

        pivot_row += 1
        if pivot_row >= num_rows:
            break

    return the_matrix, pivot_row


def rank(matrix: npt.NDArray[np.integer]) -> int:
    """Compute the binary rank of a GF(2)-additive GF(4) matrix.

    Args:
        matrix: GF(4) matrix.

    Returns:
        The binary rank of the matrix.
    """
    return row_echelon(matrix)[1]


def row_basis(matrix: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
    """Compute a basis for the additive row space of a GF(4) matrix.

    Args:
        matrix: GF(4) matrix.

    Returns:
        A matrix whose rows form a basis of the additive row space.
    """
    reduced, matrix_rank = row_echelon(matrix)
    return reduced[:matrix_rank, :]


def matmul_gf2(lhs: npt.NDArray[np.integer], rhs: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
    """Multiply a GF(2) matrix by a GF(4) matrix.

    Args:
        lhs: Binary left-hand matrix.
        rhs: Right-hand matrix whose entries encode elements of GF(4).

    Returns:
        The matrix product using XOR for addition.

    Raises:
        ValueError: If the inner matrix dimensions differ.
    """
    num_rows, lhs_cols = lhs.shape
    rhs_rows, num_cols = rhs.shape
    if lhs_cols != rhs_rows:
        msg = "Incompatible shapes for matrix multiplication."
        raise ValueError(msg)

    summands = rhs.astype(np.uint8)
    product = np.zeros((num_rows, num_cols), dtype=np.uint8)
    for row in range(num_rows):
        for selected_row in np.flatnonzero(lhs[row]):
            product[row] ^= summands[selected_row]

    return product
