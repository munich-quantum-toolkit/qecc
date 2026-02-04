# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for the elimination module in circuit synthesis."""

import time

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.elimination import (
    CheckMatrix,
    Transvection,
    eliminate_cnot,
    eliminate_non_css,
    eliminate_non_css_state,
    eliminate_non_css_with_lookahead,
    score_stateprep,
)
from mqt.qecc.circuit_synthesis.encoding import gottesman_encoding_circuit
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
    operations, result_tableau = eliminate_non_css_with_lookahead(target_tableau, lookahead=3)
    assert result_tableau.is_identity()
    assert operations.apply(target_tableau) == result_tableau


@pytest.fixture
def identity_matrix() -> np.ndarray:
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
    """Test the eliminate_css function."""
    target_matrix = request.getfixturevalue(check_matrix)
    operations, result_matrix = eliminate_cnot(target_matrix, exact=True)
    assert result_matrix.is_identity()
    assert operations.apply(target_matrix) == result_matrix


@pytest.mark.parametrize(
    ("check_matrix", "num_cnots"),
    [("identity_matrix", 0), ("cnot_matrix", 1)],
)
def test_eliminate_cnot_up_to_row_ops(check_matrix: CheckMatrix, num_cnots: int, request) -> None:
    """Test the eliminate_css function."""
    target_matrix = request.getfixturevalue(check_matrix)
    operations, result_matrix = eliminate_cnot(target_matrix, exact=True)
    assert operations.num_two_qubit_gates() == num_cnots
    assert operations.apply(target_matrix) == result_matrix


def test_transvection_circuit_consistency() -> None:
    """Test that all transvections produce circuits consistent with their tableau action."""
    all_transvections = Transvection.all_two_qubit_transvections()

    for v in all_transvections:
        i, j = 0, 1
        transvection = Transvection(v, i, j)

        circuit = transvection.to_stim_circuit()
        identity = StabilizerTableau.identity(2)
        circuit_tableau = StabilizerTableau.from_stim_circuit(circuit)
        result_tableau = transvection.apply(identity)
        print(circuit_tableau.tableau)
        print(result_tableau.tableau)
        assert result_tableau == circuit_tableau


def test_score_stateprep():
    tab = StabilizerTableau.from_pauli_strings(["XI", "IZ"])
    score = score_stateprep(tab)
    assert score == 0

    tab = StabilizerTableau.from_pauli_strings(["XX", "IZ"])
    score = score_stateprep(tab)
    assert score == 1

    tab = StabilizerTableau.from_pauli_strings(["IXZXI", "IIXZX", "-ZXZIZ", "ZXIXZ", "-IXZIZ"])
    score = score_stateprep(tab)
    assert score == 21


def test_eliminate_non_css_performance():
    """Performance test for non-CSS elimination on a 12-qubit encoding isometry."""
    iso = gottesman_encoding_circuit(["ZZXYIXZXYZIX",
                            "IZYXYYZYIIIX",
                            "IIIYZYYXYZIX",
                            "IZYXZXIYXZZX",
                            "IIIZIIIYYYZY",
                            "IIIXIIIZZZXZ",
                            "XZZZIIIXIIIZ",
                            "ZYYYIIIZIIIY"])
    tab = StabilizerTableau.from_stim_circuit(iso.to_stim_circuit())

    
    start_time = time.perf_counter()
    operations, result_tableau = eliminate_non_css(tab, optimization_criterion="gates")
    elapsed_time = time.perf_counter() - start_time
    
    print(f"\nNon-CSS elimination completed in {elapsed_time:.4f} seconds")
    print(f"Number of operations: {len(operations.operations)}")
    print(f"Number of two-qubit gates: {operations.num_transvections()}")
    
    assert result_tableau.is_identity()
    assert elapsed_time < 10.0
    assert False


def test_eliminate_non_css_with_lookahead_performance():
    """Performance test for non-CSS elimination with lookahead on a 10-qubit encoding isometry."""
    iso = gottesman_encoding_circuit(["ZZXYIXZXYZIX",
                            "IZYXYYZYIIIX",
                            "IIIYZYYXYZIX",
                            "IZYXZXIYXZZX",
                            "IIIZIIIYYYZY",
                            "IIIXIIIZZZXZ",
                            "XZZZIIIXIIIZ",
                            "ZYYYIIIZIIIY"])
    tab = StabilizerTableau.from_stim_circuit(iso.to_stim_circuit())
    
    start_time = time.perf_counter()
    operations, result_tableau = eliminate_non_css_with_lookahead(
        tab, 
        optimization_criterion="gates",
        lookahead=1,
        num_lookahead_candidates=5
    )
    elapsed_time = time.perf_counter() - start_time
    
    print(f"\nNon-CSS elimination with lookahead completed in {elapsed_time:.4f} seconds")
    print(f"Number of operations: {len(operations.operations)}")
    print(f"Number of two-qubit gates: {operations.num_transvections()}")
    
    assert result_tableau.is_identity()
    assert elapsed_time < 30.0
    assert False


def test_eliminate_cnot_performance():
    """Performance test for CNOT elimination on a 20x20 check matrix."""
    np.random.seed(42)
    n = 20
    density = 0.3
    matrix_data = (np.random.random((n, n)) < density).astype(np.int8)
    check_matrix = CheckMatrix(matrix_data, type="X")
    
    start_time = time.perf_counter()
    operations, result_matrix = eliminate_cnot(check_matrix, exact=False, lookahead=0)
    elapsed_time = time.perf_counter() - start_time
    
    print(f"\nCNOT elimination completed in {elapsed_time:.4f} seconds")
    print(f"Number of CNOTs: {operations.num_two_qubit_gates()}")
    print(f"Circuit depth: {operations.depth()}")
    
    assert elapsed_time < 5.0
