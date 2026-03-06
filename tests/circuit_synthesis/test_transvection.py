# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for transvection-based elimination in circuit synthesis."""

import time

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.encoding import gottesman_encoding_circuit
from mqt.qecc.circuit_synthesis.operations import Transvection
from mqt.qecc.circuit_synthesis.synthesis import SynthesisConfig, synthesize_non_css
from mqt.qecc.circuit_synthesis.transvection import (
    reduce_with_swaps,
)
from mqt.qecc.codes.pauli import StabilizerTableau
from mqt.qecc.codes.symplectic import SymplecticMatrix


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


@pytest.fixture
def clifford_synthesis_config() -> SynthesisConfig:
    """Fixture to create a Clifford synthesis configuration."""
    return SynthesisConfig(
        optimization_criterion="gates",
        lookahead=0,
        num_lookahead_candidates=10,
        enable_early_termination=False,
    )


@pytest.mark.parametrize(
    "tableau_matrix",
    [
        "identity_tableau",
        "cnot_tableau",
    ],
)
def test_synthesize_non_css(
    tableau_matrix: str, clifford_synthesis_config: SynthesisConfig, request: pytest.FixtureRequest
) -> None:
    """Test the synthesize_non_css function."""
    target_tableau = request.getfixturevalue(tableau_matrix)
    operations, result_tableau = synthesize_non_css(target_tableau, config=clifford_synthesis_config)
    assert result_tableau.is_identity()
    assert operations.apply(target_tableau) == result_tableau


@pytest.mark.parametrize(
    "tableau_matrix",
    [
        "identity_tableau",
        "cnot_tableau",
    ],
)
def test_synthesize_non_css_with_lookahead(
    tableau_matrix: str, clifford_synthesis_config: SynthesisConfig, request: pytest.FixtureRequest
) -> None:
    """Test the synthesize_non_css_with_lookahead function."""
    target_tableau = request.getfixturevalue(tableau_matrix)
    clifford_synthesis_config.lookahead = 2
    clifford_synthesis_config.num_lookahead_candidates = 5
    operations, result_tableau = synthesize_non_css(target_tableau, config=clifford_synthesis_config)
    assert result_tableau.is_identity()
    assert operations.apply(target_tableau) == result_tableau


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
        assert result_tableau == circuit_tableau


def test_synthesize_non_css_performance(clifford_synthesis_config: SynthesisConfig) -> None:
    """Performance test for non-CSS elimination on a 12-qubit encoding isometry."""
    iso = gottesman_encoding_circuit([
        "ZZXYIXZXYZIX",
        "IZYXYYZYIIIX",
        "IIIYZYYXYZIX",
        "IZYXZXIYXZZX",
        "IIIZIIIYYYZY",
        "IIIXIIIZZZXZ",
        "XZZZIIIXIIIZ",
        "ZYYYIIIZIIIY",
    ])
    tab = StabilizerTableau.from_stim_circuit(iso.to_stim_circuit())

    start_time = time.perf_counter()
    operations, result_tableau = synthesize_non_css(tab, config=clifford_synthesis_config)
    elapsed_time = time.perf_counter() - start_time

    print(f"\nNon-CSS elimination completed in {elapsed_time:.4f} seconds")
    print(f"Number of operations: {len(operations.operations)}")
    print(f"Number of two-qubit gates: {operations.num_transvections()}")

    assert result_tableau.is_identity()
    assert elapsed_time < 10.0


def test_four_qubit(clifford_synthesis_config: SynthesisConfig) -> None:
    """Test elimination on the 4-qubit error detection code."""
    stabs = StabilizerTableau.from_pauli_strings(["XXXI", "IIIZ", "XXII", "IXXI", "ZZZZ", "XXXX", "IZZI", "ZZII"])

    operations, result_tableau = synthesize_non_css(stabs, config=clifford_synthesis_config)
    assert result_tableau.is_identity()
    assert operations.apply(stabs) == result_tableau
    StabilizerTableau.from_stim_circuit(operations.to_circuit_inverse())
    assert StabilizerTableau.from_stim_circuit(operations.to_circuit().inverse()) == stabs


def test_reduce_with_swaps() -> None:
    """Test the reduce_with_swaps function."""
    stabs = [
        "IYIIIIII",
        "IIYIIIII",
        "IIIIZIII",
        "-IIIIIIIZ",
        "-XIIIIIII",
        "IIIIIIXI",
        "-IIIZIIII",
        "IIIIIXII",
        "-IZIIIIII",
        "IIXIIIII",
        "IIIIXIII",
        "-IIIIIIIY",
        "ZIIIIIII",
        "-IIIIIIYI",
        "IIIYIIII",
        "-IIIIIYII",
    ]
    tab = StabilizerTableau.from_pauli_strings(stabs)
    _swaps, reduced_tab = reduce_with_swaps(tab)

    for i in range(reduced_tab.tableau.shape[0] // 2):
        has_diagonal = False
        destab = reduced_tab.tableau[i]
        stab = reduced_tab.tableau[i + reduced_tab.n]
        for j, val in enumerate(stab):
            if (i == j or j == i + reduced_tab.n) and val == 1:
                has_diagonal = True
            else:
                assert val == 0
        for j, val in enumerate(destab):
            if i == j or j == i + reduced_tab.n:
                has_diagonal = True
            else:
                assert val == 0
        assert has_diagonal
