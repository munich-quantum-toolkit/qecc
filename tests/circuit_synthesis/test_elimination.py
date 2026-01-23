# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for the elimination module in circuit synthesis."""

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.elimination import (
    eliminate_non_css,
    eliminate_non_css_with_lookahead,
    eliminate_css,
    CheckMatrix
)
from mqt.qecc.codes.pauli import StabilizerTableau, SymplecticMatrix


@pytest.fixture
def identity_tableau() -> StabilizerTableau:
    """Fixture to create an identity stabilizer tableau."""
    tableau_matrix = np.array([[1, 0], [0, 1]], dtype=np.int8)
    return StabilizerTableau(SymplecticMatrix(tableau_matrix))


@pytest.fixture
def cnot_tableau() -> StabilizerTableau:
    """Fixture to create a CNOT stabilizer tableau."""
    tableau_matrix = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]], dtype=np.int8)
    return StabilizerTableau(SymplecticMatrix(tableau_matrix))


@pytest.mark.parametrize(
    "tableau_matrix",
    [
        "identity_tableau",
        "cnot_tableau",
    ],
)
def test_eliminate_non_css(tableau_matrix: StabilizerTableau, request) -> None:
    """Test the eliminate function."""
    target_tableau = request.getfixturevalue(tableau_matrix)
    _operations, result_tableau = eliminate_non_css(
        target_tableau,
    )
    assert result_tableau.is_identity()


@pytest.mark.parametrize(
    "tableau_matrix",
    [
        "identity_tableau",
        "cnot_tableau",
    ],
)
def test_eliminate_non_css_with_lookahead(tableau_matrix: StabilizerTableau, request) -> None:
    """Test the eliminate function with lookahead."""
    target_tableau = request.getfixturevalue(tableau_matrix)
    _operations, result_tableau = eliminate_non_css_with_lookahead(
        target_tableau, lookahead=3
    )
    assert result_tableau.is_identity()


@pytest.fixture
def identity_matrix() -> np.ndarray:    
    """Fixture to create an identity matrix."""    
    return np.array([[1, 0], [0, 1]], dtype=np.int8)

@pytest.fixture
def cnot_matrix() -> CheckMatrix:    
    """Fixture to create a CNOT check matrix."""    
    matrix = np.array([[1,1], [0,1]], dtype=np.int8)    
    return CheckMatrix(matrix, type="X")

@pytest.mark.parametrize(
    "check_matrix",
    [
        "identity_matrix",
        "cnot_matrix",
    ],
)
def test_eliminate_css(check_matrix: CheckMatrix, request) -> None:
    """Test the eliminate_css function."""
    target_matrix = request.getfixturevalue(check_matrix)
    _operations, result_matrix = eliminate_css(target_matrix)
    assert result_matrix.is_identity()
