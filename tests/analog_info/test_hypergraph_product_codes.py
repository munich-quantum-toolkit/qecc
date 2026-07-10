# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test product code constructions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mqt.qecc.codes.constructions.hypergraph_product_code import (
    generate_3d_product_code,
    generate_4d_product_code,
    generate_sparse_3d_product_code,
    generate_sparse_4d_product_code,
)


@pytest.fixture
def boundary_maps() -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """Fixture for boundary maps."""
    a_1 = np.array([[1, 1]], dtype=np.int32)
    a_2 = np.array([[1, 1], [1, 1]], dtype=np.int32)
    p = np.array([[1, 1]], dtype=np.int32)
    return a_1, a_2, p


def test_generate_3d_product_code(
    boundary_maps: tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]],
) -> None:
    """Test 3D HGP code construction."""
    a_1, a_2, p = boundary_maps
    d_1, d_2, d_3 = generate_3d_product_code(a_1, a_2, p)

    assert np.array_equal(d_1, np.array([[1, 1, 1, 1]], dtype=np.int32))
    assert d_2.shape == (4, 6)
    assert d_3.shape == (6, 4)
    assert not np.any(d_1 @ d_2 % 2)
    assert not np.any(d_2 @ d_3 % 2)


def test_generate_sparse_3d_product_code_matches_dense(
    boundary_maps: tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]],
) -> None:
    """Test sparse 3D HGP code construction against the dense construction."""
    a_1, a_2, p = boundary_maps
    expected = generate_3d_product_code(a_1, a_2, p)

    result = generate_sparse_3d_product_code(csr_matrix(a_1), csr_matrix(a_2), csr_matrix(p))

    for sparse_mat, dense_mat in zip(result, expected, strict=True):
        assert np.array_equal(sparse_mat.toarray(), dense_mat)


def test_generate_4d_product_code(
    boundary_maps: tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]],
) -> None:
    """Test 4D HGP code construction."""
    a_1, a_2, p = boundary_maps
    a_3 = np.array([[1], [1]], dtype=np.int32)

    d_1, d_2, d_3, d_4 = generate_4d_product_code(a_1, a_2, a_3, p)

    assert d_1.shape == (1, 4)
    assert d_2.shape == (4, 6)
    assert d_3.shape == (6, 5)
    assert d_4.shape == (5, 2)
    assert not np.any(d_1 @ d_2 % 2)
    assert not np.any(d_2 @ d_3 % 2)
    assert not np.any(d_3 @ d_4 % 2)


def test_generate_sparse_4d_product_code_matches_dense(
    boundary_maps: tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]],
) -> None:
    """Test sparse 4D HGP code construction against the dense construction."""
    a_1, a_2, p = boundary_maps
    a_3 = np.array([[1], [1]], dtype=np.int32)
    expected = generate_4d_product_code(a_1, a_2, a_3, p)

    result = generate_sparse_4d_product_code(csr_matrix(a_1), csr_matrix(a_2), csr_matrix(a_3), csr_matrix(p))

    for sparse_mat, dense_mat in zip(result, expected, strict=True):
        assert np.array_equal(sparse_mat.toarray(), dense_mat)


def test_generate_3d_product_code_rejects_invalid_boundary_maps() -> None:
    """Test that invalid boundaries are rejected."""
    a_1 = np.array([[1, 0]], dtype=np.int32)
    a_2 = np.array([[1], [0]], dtype=np.int32)
    p = np.array([[1]], dtype=np.int32)

    with pytest.raises(RuntimeError, match="boundary maps do not square to zero"):
        generate_3d_product_code(a_1, a_2, p)


def test_generate_4d_product_code_rejects_invalid_boundary_maps() -> None:
    """Test that invalid boundaries are rejected."""
    a_1 = np.array([[1, 0]], dtype=np.int32)
    a_2 = np.array([[1], [0]], dtype=np.int32)
    a_3 = np.array([[1], [0]], dtype=np.int32)
    p = np.array([[1]], dtype=np.int32)

    with pytest.raises(RuntimeError, match="boundary maps do not square to zero"):
        generate_4d_product_code(a_1, a_2, a_3, p)
