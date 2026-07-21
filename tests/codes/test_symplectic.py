# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the SymplecticVector and SymplecticMatrix classes."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mqt.qecc.codes.core.symplectic import (
    SymplecticMatrix,
    SymplecticVector,
    symplectic_product,
)


def test_symplectic() -> None:
    """Test the SymplecticMatrix and SymplecticVector classes."""
    ones = SymplecticVector.ones(3)
    zeros = SymplecticVector.zeros(3)
    assert ones - ones == zeros
    assert ones + ones == zeros

    v = SymplecticVector(np.array([1, 0, 0, 0, 0, 1]))
    w = SymplecticVector(np.array([0, 1, 0, 0, 0, 1]))
    assert w + v == v + w
    assert w - v == -v + w

    obj = "abc"
    assert v != obj

    assert v @ w == 0
    u = SymplecticVector(np.array([0, 0, 1, 0, 0, 0]))
    assert v @ u == 1

    eye = SymplecticMatrix.symplectic_identity(3)
    zero_mat = SymplecticMatrix.zeros(6, 3)
    assert eye + eye == zero_mat
    assert eye - eye == zero_mat

    vs = [v.vector, w.vector, u.vector, ones.vector, zeros.vector, v.vector]
    m = SymplecticMatrix(np.array(vs))
    assert eye @ m.transpose() == m
    assert m @ eye == m

    for i, row in enumerate(m):
        assert np.array_equal(row, vs[i])

    assert m != obj
    assert len(m) == 6
    assert m.shape == (6, 6)
    assert m.n == 3


def test_symplectic_product_shapes() -> None:
    """Test symplectic_product for vector and matrix operands."""
    v = np.array([1, 0, 0, 0, 0, 1], dtype=np.int8)  # XIZ
    u = np.array([0, 0, 1, 0, 0, 0], dtype=np.int8)  # IIX
    assert symplectic_product(v, v) == 0
    assert symplectic_product(v, u) == 1

    m = np.array([v, u], dtype=np.int8)
    assert_array_equal(symplectic_product(m, v), np.array([0, 1]))
    assert_array_equal(symplectic_product(v, m), np.array([0, 1]))
    assert_array_equal(symplectic_product(m, m), np.array([[0, 1], [1, 0]]))

    empty = np.zeros((0, 6), dtype=np.int8)
    assert symplectic_product(empty, m).shape == (0, 2)


def test_symplectic_product_validation() -> None:
    """Test that symplectic_product rejects invalid inputs."""
    v = np.array([1, 0, 0, 0, 0, 1], dtype=np.int8)
    with pytest.raises(ValueError, match="even"):
        symplectic_product(v, np.array([1, 0, 0], dtype=np.int8))
    with pytest.raises(ValueError, match="same number of qubits"):
        symplectic_product(v, np.array([1, 0, 0, 0], dtype=np.int8))
    with pytest.raises(ValueError, match="dimensions"):
        symplectic_product(v, np.zeros((1, 2, 6), dtype=np.int8))
    with pytest.raises(ValueError, match="nonzero"):
        symplectic_product(np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.int8))
