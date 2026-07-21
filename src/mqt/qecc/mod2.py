# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utilities for binary linear algebra (mod 2) to replace dependency on ldpc.mod2."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def row_echelon(
    matrix: npt.NDArray[np.integer],
    *,
    full: bool = False,
) -> tuple[npt.NDArray[np.integer], int, npt.NDArray[np.integer], list[int]]:
    """Convert a binary matrix to row echelon form over GF(2).

    The input is assumed to be a dense binary integer matrix (entries 0 or 1).
    It is copied internally so the caller's array is never modified.

    Args:
        matrix: Binary matrix to reduce.
        full: If ``True``, eliminate entries above and below each pivot
            (reduced row echelon form). Otherwise only eliminate below.

    Returns:
        A tuple containing

        - the row echelon form,
        - the matrix rank,
        - the transformation matrix ``T`` such that ``(T @ matrix) % 2`` equals
          the row echelon form,
        - the pivot column indices.
    """
    num_rows, num_cols = matrix.shape
    if num_rows == 0:
        return np.copy(matrix), 0, np.eye(0, dtype=int), []

    the_matrix = np.copy(matrix)
    transform_matrix = np.eye(num_rows, dtype=int)

    pivot_row = 0
    pivot_cols: list[int] = []

    for col in range(num_cols):
        if the_matrix[pivot_row, col] != 1:
            # Binary columns contain only 0/1, so argmax locates the first pivot.
            swap_idx = pivot_row + int(np.argmax(the_matrix[pivot_row:num_rows, col]))
            if the_matrix[swap_idx, col] == 1:
                the_matrix[[swap_idx, pivot_row]] = the_matrix[[pivot_row, swap_idx]]
                transform_matrix[[swap_idx, pivot_row]] = transform_matrix[[pivot_row, swap_idx]]

        if the_matrix[pivot_row, col]:
            elim_range = range(num_rows) if full else range(pivot_row + 1, num_rows)
            for j in elim_range:
                if full and j == pivot_row:
                    continue
                if the_matrix[j, col]:
                    the_matrix[j] = (the_matrix[j] + the_matrix[pivot_row]) % 2
                    transform_matrix[j] = (transform_matrix[j] + transform_matrix[pivot_row]) % 2

            pivot_row += 1
            pivot_cols.append(col)

        if pivot_row >= num_rows:
            break

    return the_matrix, pivot_row, transform_matrix, pivot_cols


def rank(matrix: npt.NDArray[np.integer]) -> int:
    """Compute the rank of a binary matrix over GF(2).

    Args:
        matrix: Binary matrix.

    Returns:
        The rank of the matrix.
    """
    return row_echelon(matrix)[1]


def nullspace(matrix: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
    """Compute a basis for the nullspace of a binary matrix over GF(2).

    Args:
        matrix: Binary matrix.

    Returns:
        A matrix whose rows form a basis of the nullspace, i.e. every row ``v``
        satisfies ``matrix @ v % 2 == 0``.
    """
    transpose = matrix.T
    num_rows, _ = transpose.shape
    _, matrix_rank, transform, _ = row_echelon(transpose)
    return transform[matrix_rank:num_rows]


def row_basis(matrix: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
    """Compute a basis for the row space of a binary matrix over GF(2).

    Args:
        matrix: Binary matrix.

    Returns:
        A matrix whose rows are a linearly independent subset of the rows of
        ``matrix`` that spans the same row space.
    """
    return matrix[row_echelon(matrix.T)[3]]


def is_in_row_space(vector: npt.NDArray[np.integer], basis: npt.NDArray[np.integer]) -> bool:
    """Check whether a binary vector lies in the binary row space of a basis.

    Args:
        vector: The binary vector to test.
        basis: A binary matrix whose rows span the space.

    Returns:
        True if the vector is a binary linear combination of the basis rows.
    """
    if basis.shape[0] == 0:
        return not np.any(vector % 2)
    return bool(rank(np.vstack((basis, vector)) % 2) == rank(basis % 2))


def are_in_same_coset(
    lhs: npt.NDArray[np.integer],
    rhs: npt.NDArray[np.integer],
    basis: npt.NDArray[np.integer],
) -> bool:
    """Check whether two binary vectors lie in the same coset of a binary row space of a basis.

    Equivalently, can rhs be obtained from lhs by adding a linear combination of the rows of basis.

    Args:
        lhs: The first binary vector.
        rhs: The second binary vector.
        basis: A binary matrix whose rows span the space.

    Returns:
        True if the difference of the vectors is in the row space of the basis.
    """
    return is_in_row_space((lhs + rhs) % 2, basis)
