# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for transvection-based elimination in circuit synthesis."""

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.synthesis import SynthesisConfig, synthesize_non_css
from mqt.qecc.circuit_synthesis.transvection import (
    reduce_with_swaps,
)
from mqt.qecc.codes.core.pauli import StabilizerTableau
from mqt.qecc.codes.core.symplectic import SymplecticMatrix


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
        rollout=0,
        num_rollout_candidates=10,
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
def test_synthesize_non_css_with_rollout(
    tableau_matrix: str, clifford_synthesis_config: SynthesisConfig, request: pytest.FixtureRequest
) -> None:
    """Test the synthesize_non_css_with_rollout function."""
    target_tableau = request.getfixturevalue(tableau_matrix)
    clifford_synthesis_config.rollout = 2
    clifford_synthesis_config.num_rollout_candidates = 5
    operations, result_tableau = synthesize_non_css(target_tableau, config=clifford_synthesis_config)
    assert result_tableau.is_identity()
    assert operations.apply(target_tableau) == result_tableau


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
            if (i == j or j == i + reduced_tab.n) and val == 1:
                has_diagonal = True
            else:
                assert val == 0
        assert has_diagonal
