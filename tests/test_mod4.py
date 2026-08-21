# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the additive GF(4) linear algebra utilities."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc.mod4 import matmul_gf2_gf4, rank, row_basis, row_echelon


def test_row_echelon_empty_matrix() -> None:
    """A matrix with zero rows is returned unchanged with rank zero."""
    matrix = np.empty((0, 4), dtype=np.uint8)
    matrix_rank, reduced = row_echelon(matrix)

    assert matrix_rank == 0
    assert np.array_equal(reduced, matrix)


def test_row_echelon_does_not_mutate_input() -> None:
    """Reduction preserves the input and finds pivots in both binary components."""
    matrix = np.array([[1, 2], [3, 1], [2, 3]], dtype=np.uint8)
    original = matrix.copy()

    matrix_rank, reduced = row_echelon(matrix, full=True)

    assert np.array_equal(matrix, original)
    assert matrix_rank == 2
    assert np.array_equal(reduced, np.array([[1, 2], [2, 3], [0, 0]], dtype=np.uint8))


def test_row_echelon_full_is_reduced() -> None:
    """Full reduction eliminates entries both above and below each pivot."""
    matrix = np.array([[1, 1], [0, 1]], dtype=np.uint8)

    echelon, reduced = row_echelon(matrix)[1], row_echelon(matrix, full=True)[1]

    assert np.array_equal(echelon, matrix)
    assert np.array_equal(reduced, np.eye(2, dtype=np.uint8))


def test_row_echelon_full_is_keyword_only() -> None:
    """``full`` must be passed as a keyword argument."""
    with pytest.raises(TypeError):
        row_echelon(np.eye(2, dtype=np.uint8), True)  # ty: ignore[too-many-positional-arguments]


def test_row_basis_spans_additive_row_space() -> None:
    """The returned rows form a basis after removing an XOR-dependent row."""
    matrix = np.array([[1, 2], [2, 1], [3, 3]], dtype=np.uint8)

    basis = row_basis(matrix)

    assert basis.shape == (2, 2)
    assert any(np.array_equal(basis[0] ^ basis[1], row) for row in matrix)


def test_rank() -> None:
    """Binary rank matches known full-rank and rank-deficient matrices."""
    assert rank(np.array([[1, 0], [0, 2]], dtype=np.uint8)) == 2
    assert rank(np.array([[1, 2], [1, 2]], dtype=np.uint8)) == 1
    assert rank(np.zeros((3, 3), dtype=np.uint8)) == 0


def test_matmul_gf2_gf4() -> None:
    """Binary coefficients select and XOR rows of the GF(4) matrix."""
    lhs = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    rhs = np.array([[1, 2], [2, 3], [3, 1]], dtype=np.uint8)

    assert np.array_equal(matmul_gf2_gf4(lhs, rhs), np.array([[2, 3], [1, 2]], dtype=np.uint8))


def test_matmul_gf2_gf4_rejects_incompatible_shapes() -> None:
    """Matrix multiplication rejects mismatched inner dimensions."""
    with pytest.raises(ValueError, match="Incompatible shapes"):
        matmul_gf2_gf4(np.zeros((2, 3), dtype=np.uint8), np.zeros((2, 4), dtype=np.uint8))
