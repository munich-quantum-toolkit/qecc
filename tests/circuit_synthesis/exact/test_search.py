# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for exact synthesis search API."""

from __future__ import annotations

import pytest

from mqt.qecc.circuit_synthesis.exact import synthesize_exact
from mqt.qecc.circuit_synthesis.exact.types import GateFamily, Objective, SynthesisStatus, TargetKind
from mqt.qecc.circuit_synthesis.exact.verification import (
    verify_clifford_isometry,
    verify_clifford_unitary,
    verify_css_isometry,
    verify_css_state,
    verify_stabilizer_state,
)
from mqt.qecc.codes.pauli import CheckMatrix, StabilizerTableau


def test_bell_state_preparation(bell_state_tableau: StabilizerTableau) -> None:
    """Test synthesis of Bell state preparation."""
    result = synthesize_exact(
        target=bell_state_tableau,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=1,
        upper_bound=5,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count >= 1
    assert result.verified is True
    assert verify_stabilizer_state(result.circuit, bell_state_tableau)


def test_bell_state_unsat(bell_state_tableau: StabilizerTableau) -> None:
    """Test that Bell state cannot be prepared with 0 gates."""
    result = synthesize_exact(
        target=bell_state_tableau,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=0,
    )

    assert result.status == SynthesisStatus.UNSAT
    assert result.circuit is None


def test_ghz_state_preparation(ghz_state_tableau: StabilizerTableau) -> None:
    """Test synthesis of GHZ state preparation."""
    result = synthesize_exact(
        target=ghz_state_tableau,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=1,
        upper_bound=5,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count >= 2
    assert result.verified is True
    assert verify_stabilizer_state(result.circuit, ghz_state_tableau)


def test_plus_state_preparation(plus_state_tableau: StabilizerTableau) -> None:
    """Test synthesis of |+⟩ state preparation."""
    result = synthesize_exact(
        target=plus_state_tableau,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=3,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == 1
    assert result.verified is True
    assert verify_stabilizer_state(result.circuit, plus_state_tableau)


def test_hadamard_unitary(hadamard_tableau: StabilizerTableau) -> None:
    """Test synthesis of Hadamard gate."""
    result = synthesize_exact(
        target=hadamard_tableau,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=1,
        upper_bound=3,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == 1
    assert result.verified is True
    assert verify_clifford_unitary(result.circuit, hadamard_tableau)


def test_cnot_unitary(cnot_tableau: StabilizerTableau) -> None:
    """Test synthesis of CNOT gate."""
    result = synthesize_exact(
        target=cnot_tableau,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=1,
        upper_bound=3,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == 1
    assert result.verified is True
    assert verify_clifford_unitary(result.circuit, cnot_tableau)


def test_verification_enabled(plus_state_tableau: StabilizerTableau) -> None:
    """Test that verification works correctly."""
    result = synthesize_exact(
        target=plus_state_tableau,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=1,
        upper_bound=3,
        verify=True,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.verified is True
    assert result.circuit is not None
    assert verify_stabilizer_state(result.circuit, plus_state_tableau)


def test_verification_disabled(plus_state_tableau: StabilizerTableau) -> None:
    """Test that verification can be disabled."""
    result = synthesize_exact(
        target=plus_state_tableau,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=1,
        upper_bound=3,
        verify=False,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.verified is False
    assert result.circuit is not None


def test_css_state_synthesis(repetition_code_check_matrix: CheckMatrix) -> None:
    """Test CSS state synthesis."""
    result = synthesize_exact(
        target=repetition_code_check_matrix,
        target_kind=TargetKind.CSS_STATE,
        gate_family=GateFamily.CSS_CNOT,
        objective=Objective.GATE_COUNT,
        lower_bound=1,
        upper_bound=5,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count >= 1
    assert result.verified is True
    assert verify_css_state(result.circuit, repetition_code_check_matrix)


def test_clifford_isometry_synthesis() -> None:
    """Test Clifford isometry synthesis with logical qubits."""
    target = StabilizerTableau.from_pauli_strings([
        "XX",
        "ZI",
        "ZZ",
    ])

    result = synthesize_exact(
        target=target,
        target_kind=TargetKind.CLIFFORD_ISOMETRY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        k=1,
        lower_bound=1,
        upper_bound=5,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count >= 1
    assert result.verified is True
    assert verify_clifford_isometry(result.circuit, target, k=1)


def test_css_isometry_synthesis() -> None:
    """Test CSS isometry synthesis."""
    target = CheckMatrix.from_numpy_array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        "X",
    )

    result = synthesize_exact(
        target=target,
        target_kind=TargetKind.CSS_ISOMETRY,
        gate_family=GateFamily.CSS_CNOT,
        objective=Objective.GATE_COUNT,
        k=1,
        lower_bound=1,
        upper_bound=5,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count >= 1
    assert result.verified is True
    assert verify_css_isometry(result.circuit, target, k=1)


def test_invalid_target_kind() -> None:
    """Test that invalid target kind raises ValueError."""
    target = StabilizerTableau.from_pauli_strings(["X"])

    with pytest.raises(ValueError, match="k must be provided for isometry synthesis"):
        synthesize_exact(
            target=target,
            target_kind=TargetKind.CLIFFORD_ISOMETRY,
            gate_family=GateFamily.CLIFFORD,
            objective=Objective.GATE_COUNT,
            lower_bound=1,
            upper_bound=3,
        )


def test_depth_optimization_not_implemented(plus_state_tableau: StabilizerTableau) -> None:
    """Test that depth optimization raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Depth optimization not yet implemented"):
        synthesize_exact(
            target=plus_state_tableau,
            target_kind=TargetKind.STABILIZER_STATE,
            gate_family=GateFamily.CLIFFORD,
            objective=Objective.DEPTH,
            lower_bound=1,
            upper_bound=3,
        )


def test_qubit_permutation_disabled(hadamard_tableau: StabilizerTableau) -> None:
    """Test synthesis with qubit permutation disabled."""
    result = synthesize_exact(
        target=hadamard_tableau,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=1,
        upper_bound=3,
        allow_qubit_permutation=False,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert verify_clifford_unitary(result.circuit, hadamard_tableau)


def test_identity_state() -> None:
    """Test synthesis of |0⟩ state (identity circuit)."""
    target = StabilizerTableau.from_pauli_strings(["Z"])

    result = synthesize_exact(
        target=target,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=2,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == 0
    assert result.verified is True
    assert verify_stabilizer_state(result.circuit, target)


def test_identity_unitary(identity_2q_tableau: StabilizerTableau) -> None:
    """Test synthesis of identity gate."""
    result = synthesize_exact(
        target=identity_2q_tableau,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=2,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == 0
    assert result.verified is True
    assert verify_clifford_unitary(result.circuit, identity_2q_tableau)


def test_two_qubit_state() -> None:
    """Test synthesis of |00⟩ state."""
    target = StabilizerTableau.from_pauli_strings(["ZI", "IZ"])

    result = synthesize_exact(
        target=target,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=3,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == 0
    assert result.verified is True
    assert verify_stabilizer_state(result.circuit, target)


def test_s_gate_unitary(s_gate_tableau: StabilizerTableau) -> None:
    """Test synthesis of S gate."""
    result = synthesize_exact(
        target=s_gate_tableau,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=1,
        upper_bound=3,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == 1
    assert result.verified is True
    assert verify_clifford_unitary(result.circuit, s_gate_tableau)


def test_swap_unitary() -> None:
    """Test synthesis of SWAP gate."""
    target = StabilizerTableau.from_pauli_strings(["IX", "ZI", "XI", "IZ"])

    result = synthesize_exact(
        target=target,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=1,
        upper_bound=5,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count >= 3
    assert result.verified is True
    assert verify_clifford_unitary(result.circuit, target)
