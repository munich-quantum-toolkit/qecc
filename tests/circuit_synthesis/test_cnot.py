# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for CNOT-based elimination in circuit synthesis."""

import time

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.cnot import eliminate_cnot
from mqt.qecc.codes.pauli import CheckMatrix


@pytest.fixture
def identity_matrix() -> CheckMatrix:
    """Fixture to create an identity matrix."""
    return CheckMatrix(np.array([[1, 0], [0, 1]], dtype=np.int8), type="X")


@pytest.fixture
def cnot_matrix() -> CheckMatrix:
    """Fixture to create a CNOT check matrix."""
    matrix = np.array([[1, 1], [0, 1]], dtype=np.int8)
    return CheckMatrix(matrix, type="X")


@pytest.mark.parametrize(
    "check_matrix",
    [
        "identity_matrix",
        "cnot_matrix",
    ],
)
def test_eliminate_cnot_exact(check_matrix: CheckMatrix, request) -> None:
    """Test the eliminate_cnot function with exact elimination."""
    target_matrix = request.getfixturevalue(check_matrix)
    operations, result_matrix = eliminate_cnot(target_matrix, exact=True)
    assert result_matrix.is_identity()
    assert operations.apply(target_matrix) == result_matrix


@pytest.mark.parametrize(
    ("check_matrix", "num_cnots"),
    [("identity_matrix", 0), ("cnot_matrix", 1)],
)
def test_eliminate_cnot_up_to_row_ops(check_matrix: CheckMatrix, num_cnots: int, request) -> None:
    """Test the eliminate_cnot function up to row operations."""
    target_matrix = request.getfixturevalue(check_matrix)
    operations, result_matrix = eliminate_cnot(target_matrix, exact=True)
    assert operations.num_two_qubit_gates() == num_cnots
    assert operations.apply(target_matrix) == result_matrix


def test_eliminate_cnot_performance():
    """Performance test for CNOT elimination on a 20x20 check matrix."""
    np.random.seed(42)
    n = 20
    density = 0.3
    matrix_data = (np.random.random((n, n)) < density).astype(np.int8)
    check_matrix = CheckMatrix(matrix_data, type="X")

    start_time = time.perf_counter()
    operations, _result_matrix = eliminate_cnot(check_matrix, exact=False, lookahead=0)
    elapsed_time = time.perf_counter() - start_time

    print(f"\nCNOT elimination completed in {elapsed_time:.4f} seconds")
    print(f"Number of CNOTs: {operations.num_two_qubit_gates()}")
    print(f"Circuit depth: {operations.depth()}")

    assert elapsed_time < 5.0
