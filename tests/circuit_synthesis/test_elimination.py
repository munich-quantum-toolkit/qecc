# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for the elimination module in circuit synthesis."""

import numpy as np
import pytest

from mqt.qecc.codes.pauli import CheckMatrix


@pytest.fixture
def simple_check_matrix() -> CheckMatrix:
    """Fixture to create a simple CSS check matrix."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
    return CheckMatrix(matrix, type="X")


def test_check_matrix_fixture(simple_check_matrix: CheckMatrix) -> None:
    """Test that the check matrix fixture is created correctly."""
    assert simple_check_matrix.num_qubits() == 3
    assert simple_check_matrix.num_rows() == 2
    assert simple_check_matrix.type == "X"
