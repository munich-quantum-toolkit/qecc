# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for exact synthesis main API."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.exact import (
    ExactSynthesisOptions,
    GateFamily,
    Objective,
    SynthesisStatus,
    TargetKind,
    synthesize_exact,
)
from mqt.qecc.codes.pauli import CheckMatrix, StabilizerTableau


def test_synthesize_exact_api() -> None:
    """Test that synthesize_exact API works with basic inputs."""
    # Create a simple stabilizer state target
    target = StabilizerTableau.from_pauli_strings(["XX", "ZZ"])

    result = synthesize_exact(
        target,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=5),
    )

    # Currently returns UNSAT placeholder
    assert result.status == SynthesisStatus.UNSAT
    assert result.optimal is False
    assert result.circuit is None


def test_validate_target_css_with_clifford_gates() -> None:
    """Test that CSS targets require CNOT gate family."""
    check_matrix = CheckMatrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8), pauli_type="X")

    with pytest.raises(ValueError, match="GateFamily.CLIFFORD cannot be used with"):
        synthesize_exact(
            check_matrix,
            TargetKind.CSS_STATE_PREP,
            GateFamily.CLIFFORD,  # Wrong gate family
            Objective.GATE_COUNT,
            ExactSynthesisOptions(max_bound=5),
        )


def test_validate_target_cnot_requires_css() -> None:
    """Test that CNOT gate family requires CSS target kind."""
    target = StabilizerTableau.from_pauli_strings(["XX", "ZZ"])

    with pytest.raises(ValueError, match="GateFamily.CNOT requires target_kind"):
        synthesize_exact(
            target,
            TargetKind.STABILIZER_STATE,  # Not CSS
            GateFamily.CNOT,
            Objective.GATE_COUNT,
            ExactSynthesisOptions(max_bound=5),
        )


def test_validate_clifford_unitary_tableau_size() -> None:
    """Test that Clifford unitary requires full 2n x 2n tableau."""
    # Create incomplete tableau
    target = StabilizerTableau.from_pauli_strings(["XX", "ZZ"])

    with pytest.raises(ValueError, match="Clifford unitary requires a full 2n x 2n tableau"):
        synthesize_exact(
            target,
            TargetKind.CLIFFORD_UNITARY,
            GateFamily.CLIFFORD,
            Objective.GATE_COUNT,
            ExactSynthesisOptions(max_bound=5),
        )


def test_validate_stabilizer_state_tableau_size() -> None:
    """Test that stabilizer state requires n x 2n tableau."""
    # Create full tableau (not state)
    target = StabilizerTableau.identity(2)

    with pytest.raises(ValueError, match="Stabilizer state preparation requires n x 2n tableau"):
        synthesize_exact(
            target,
            TargetKind.STABILIZER_STATE,
            GateFamily.CLIFFORD,
            Objective.GATE_COUNT,
            ExactSynthesisOptions(max_bound=5),
        )
