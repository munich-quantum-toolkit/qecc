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
    CheckMatrix,
    eliminate_cnot,
    eliminate_non_css,
    eliminate_non_css_with_lookahead,
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
        # "identity_tableau",
        "cnot_tableau",
    ],
)
def test_eliminate_non_css(tableau_matrix: StabilizerTableau, request) -> None:
    """Test the eliminate function."""
    target_tableau = request.getfixturevalue(tableau_matrix)
    operations, result_tableau = eliminate_non_css(
        target_tableau,
    )
    assert result_tableau.is_identity()
    print(result_tableau.to_pauli_list())
    print(operations.apply(target_tableau).to_pauli_list())
    assert operations.apply(target_tableau) == result_tableau
    

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
    operations, result_tableau = eliminate_non_css_with_lookahead(
        target_tableau, lookahead=3
    )
    assert result_tableau.is_identity()
    assert operations.apply(target_tableau) == result_tableau


@pytest.fixture
def identity_matrix() -> np.ndarray:    
    """Fixture to create an identity matrix."""    
    return CheckMatrix(np.array([[1, 0], [0, 1]], dtype=np.int8), type="X")

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
def test_eliminate_cnot_exact(check_matrix: CheckMatrix, request) -> None:
    """Test the eliminate_css function."""
    target_matrix = request.getfixturevalue(check_matrix)
    operations, result_matrix = eliminate_cnot(target_matrix, exact=True)
    assert result_matrix.is_identity()
    assert operations.apply(target_matrix) == result_matrix


@pytest.mark.parametrize(
    ("check_matrix", "num_cnots"),
    [
        ("identity_matrix", 0),
        ("cnot_matrix", 1)
    ],
)
def test_eliminate_cnot_up_to_row_ops(check_matrix: CheckMatrix, num_cnots: int, request) -> None:
    """Test the eliminate_css function."""
    target_matrix = request.getfixturevalue(check_matrix)
    operations, result_matrix = eliminate_cnot(target_matrix, exact=True)
    assert operations.num_two_qubit_gates() == num_cnots
    assert operations.apply(target_matrix) == result_matrix
    
    
