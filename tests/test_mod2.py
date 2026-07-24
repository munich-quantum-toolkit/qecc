# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the mod2 binary linear algebra utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc.mod2 import are_in_same_coset, is_in_row_space, nullspace, rank, row_basis, row_echelon


def test_row_echelon_empty_matrix() -> None:
    """A matrix with zero rows is returned unchanged with rank 0 and no pivots."""
    matrix = np.empty((0, 4), dtype=int)
    reduced, matrix_rank, transform, pivots = row_echelon(matrix)

    assert matrix_rank == 0
    assert pivots == []
    assert np.array_equal(reduced, matrix)
    assert np.array_equal(transform, np.eye(0, dtype=int))


def test_row_echelon_does_not_mutate_input() -> None:
    """The input matrix is copied internally and left untouched."""
    matrix = np.array([[1, 1], [1, 0]])
    original = matrix.copy()
    row_echelon(matrix)
    assert np.array_equal(matrix, original)


def test_row_echelon_transform_relation() -> None:
    """The transform ``T`` satisfies ``(T @ matrix) % 2 == reduced`` in both modes."""
    matrix = np.array([[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
    for full in (False, True):
        reduced, _, transform, _ = row_echelon(matrix, full=full)
        assert np.array_equal((transform @ matrix) % 2, reduced)


def test_row_echelon_pivots_and_rank() -> None:
    """A matrix with a dependent row has the expected rank and pivot columns."""
    matrix = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])  # third row = row1 + row2 (mod 2)
    reduced, matrix_rank, _, pivots = row_echelon(matrix)

    assert matrix_rank == 2
    assert pivots == [0, 1]
    assert np.array_equal(reduced[matrix_rank:], np.zeros((1, 3), dtype=int))


def test_row_echelon_full_is_reduced() -> None:
    """In full mode each pivot column is a unit column (reduced row echelon form)."""
    matrix = np.array([[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0]])
    reduced, matrix_rank, _, pivots = row_echelon(matrix, full=True)

    for pivot_row, col in enumerate(pivots):
        expected = np.zeros(matrix_rank, dtype=int)
        expected[pivot_row] = 1
        assert np.array_equal(reduced[:matrix_rank, col], expected)


def test_row_echelon_full_is_keyword_only() -> None:
    """``full`` must be passed as a keyword argument."""
    with pytest.raises(TypeError):
        row_echelon(np.eye(2, dtype=int), True)  # ty: ignore[too-many-positional-arguments]


def test_rank() -> None:
    """Rank matches known values, including for a full-rank and a rank-deficient matrix."""
    assert rank(np.eye(3, dtype=int)) == 3
    assert rank(np.zeros((3, 3), dtype=int)) == 0
    # Rank over GF(2): [[1,1],[1,1]] has rank 1.
    assert rank(np.array([[1, 1], [1, 1]])) == 1


def test_nullspace() -> None:
    """Every nullspace row ``v`` satisfies ``matrix @ v % 2 == 0`` and the count is ``cols - rank``."""
    matrix = np.array([[1, 0, 1], [0, 1, 1]])  # rank 2, 3 columns -> 1-dimensional nullspace
    ns = nullspace(matrix)

    assert ns.shape[0] == matrix.shape[1] - rank(matrix)
    for row in ns:
        assert np.array_equal((matrix @ row) % 2, np.zeros(matrix.shape[0], dtype=int))
    # The unique non-trivial kernel vector of this matrix is [1, 1, 1].
    assert np.array_equal(ns, np.array([[1, 1, 1]]))


def test_nullspace_full_rank_is_empty() -> None:
    """A full-column-rank matrix has a trivial (empty) nullspace basis."""
    ns = nullspace(np.eye(3, dtype=int))
    assert ns.shape[0] == 0


def test_row_basis() -> None:
    """The row basis is a linearly independent subset spanning the original row space."""
    matrix = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])  # third row is the sum of the first two
    basis = row_basis(matrix)

    assert basis.shape[0] == rank(matrix) == 2
    # Every basis row is one of the original rows.
    for row in basis:
        assert any(np.array_equal(row, original) for original in matrix)
    # The basis spans the same row space as the full matrix.
    assert rank(np.vstack((matrix, basis))) == rank(matrix)


def test_row_space_helpers() -> None:
    """Test is_in_row_space and are_in_same_coset, including empty bases."""
    basis = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
    assert is_in_row_space(np.array([1, 0, 1], dtype=np.int8), basis)
    assert not is_in_row_space(np.array([1, 0, 0], dtype=np.int8), basis)
    assert is_in_row_space(np.zeros(3, dtype=np.int8), basis)

    empty = np.zeros((0, 3), dtype=np.int8)
    assert is_in_row_space(np.zeros(3, dtype=np.int8), empty)
    assert not is_in_row_space(np.array([1, 0, 0], dtype=np.int8), empty)

    e1 = np.array([1, 0, 0], dtype=np.int8)
    e2 = np.array([0, 1, 1], dtype=np.int8)
    assert are_in_same_coset(e1, (e1 + basis[0]) % 2, basis)
    assert not are_in_same_coset(e1, e2, empty)
    assert are_in_same_coset(e1, e1, empty)
