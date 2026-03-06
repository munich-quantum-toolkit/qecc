# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for CNOT-based elimination in circuit synthesis."""

import time

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.synthesis import CnotSynthesisConfig, synthesize_cnot
from mqt.qecc.codes.pauli import CheckMatrix


@pytest.fixture
def identity_matrix() -> CheckMatrix:
    """Fixture to create an identity matrix."""
    return CheckMatrix(np.array([[1, 0], [0, 1]], dtype=np.int8), pauli_type="X")


@pytest.fixture
def cnot_matrix() -> CheckMatrix:
    """Fixture to create a CNOT check matrix."""
    matrix = np.array([[1, 1], [0, 1]], dtype=np.int8)
    return CheckMatrix(matrix, pauli_type="X")


@pytest.fixture
def cnot_synthesis_config() -> CnotSynthesisConfig:
    """Fixture to create a CNOT synthesis configuration."""
    return CnotSynthesisConfig(
        optimization_criterion="gates",
        exact=True,
        lookahead=0,
        num_lookahead_candidates=10,
        enable_early_termination=False,
    )


@pytest.mark.parametrize(
    "check_matrix",
    [
        "identity_matrix",
        "cnot_matrix",
    ],
)
def test_eliminate_cnot_exact(
    check_matrix: str, cnot_synthesis_config: CnotSynthesisConfig, request: pytest.FixtureRequest
) -> None:
    """Test the eliminate_cnot function with exact elimination."""
    target_matrix = request.getfixturevalue(check_matrix)
    cnot_synthesis_config.exact = True
    operations, result_matrix = synthesize_cnot(target_matrix)
    assert result_matrix.is_identity()
    assert operations.apply(target_matrix) == result_matrix


@pytest.mark.parametrize(
    ("check_matrix", "num_cnots"),
    [("identity_matrix", 0), ("cnot_matrix", 1)],
)
def test_eliminate_cnot_up_to_row_ops(
    check_matrix: str,
    num_cnots: int,
    cnot_synthesis_config: CnotSynthesisConfig,
    request: pytest.FixtureRequest,
) -> None:
    """Test the eliminate_cnot function up to row operations."""
    target_matrix = request.getfixturevalue(check_matrix)
    cnot_synthesis_config.exact = True
    operations, result_matrix = synthesize_cnot(target_matrix, cnot_synthesis_config)
    assert operations.num_two_qubit_gates() == num_cnots
    assert operations.apply(target_matrix) == result_matrix


def test_eliminate_cnot_performance(cnot_synthesis_config: CnotSynthesisConfig) -> None:
    """Performance test for CNOT elimination on a 20x20 check matrix."""
    rng = np.random.default_rng(42)
    n = 8
    density = 0.3
    while True:
        matrix_data = rng.random((n, n)) < density
        matrix_data = matrix_data.astype(np.int8)
        if mod2.rank(matrix_data) == n:
            break

    check_matrix = CheckMatrix(matrix_data.astype(np.int8), pauli_type="X")
    start_time = time.perf_counter()
    operations, _result_matrix = synthesize_cnot(check_matrix, cnot_synthesis_config)
    elapsed_time = time.perf_counter() - start_time

    print(f"\nCNOT elimination completed in {elapsed_time:.4f} seconds")
    print(f"Number of CNOTs: {operations.num_two_qubit_gates()}")
    print(f"Circuit depth: {operations.depth()}")

    assert elapsed_time < 5.0
