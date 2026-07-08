# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the mod2 binary linear algebra utilities."""

from __future__ import annotations

import numpy as np

from mqt.qecc.mod2 import row_echelon


def test_row_echelon_empty_matrix() -> None:
    """A matrix with zero rows is returned unchanged with rank 0 and no pivots."""
    matrix = np.empty((0, 4), dtype=int)
    reduced, matrix_rank, transform, pivots = row_echelon(matrix)

    assert matrix_rank == 0
    assert pivots == []
    assert np.array_equal(reduced, matrix)
    assert np.array_equal(transform, np.eye(0, dtype=int))
