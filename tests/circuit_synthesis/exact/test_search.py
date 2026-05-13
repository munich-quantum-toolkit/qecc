# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for exact synthesis search API."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.exact import synthesize_exact
from mqt.qecc.circuit_synthesis.exact.types import GateFamily, Objective, SynthesisStatus, TargetKind
from mqt.qecc.circuit_synthesis.exact.verification import (
    verify_css_isometry,
    verify_css_state,
    verify_stabilizer_state,
)
from mqt.qecc.codes.pauli import CheckMatrix, StabilizerTableau


def test_bell_state_preparation(bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test synthesis of Bell state preparation."""
    stabilizers, _x_logicals, _z_logicals = bell_state

    result = synthesize_exact(
        target=stabilizers,
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
    assert verify_stabilizer_state(result.circuit, stabilizers)


def test_bell_state_unsat(bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test that Bell state cannot be prepared with 0 gates."""
    stabilizers, _x_logicals, _z_logicals = bell_state

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=0,
    )

    assert result.status == SynthesisStatus.UNSAT
    assert result.circuit is None


def test_ghz_state_preparation(ghz_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test synthesis of GHZ state preparation."""
    stabilizers, _x_logicals, _z_logicals = ghz_state

    result = synthesize_exact(
        target=stabilizers,
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
    assert verify_stabilizer_state(result.circuit, stabilizers)


def test_plus_state_preparation(plus_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test synthesis of |+⟩ state preparation."""
    stabilizers, _x_logicals, _z_logicals = plus_state

    result = synthesize_exact(
        target=stabilizers,
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
    assert verify_stabilizer_state(result.circuit, stabilizers)


def test_hadamard_unitary(hadamard_unitary: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test synthesis of Hadamard gate."""
    stabilizers, x_logicals, z_logicals = hadamard_unitary

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        k=1,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
        lower_bound=1,
        upper_bound=3,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == 1
    assert result.verified is True


def test_cnot_unitary(cnot_unitary: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test synthesis of CNOT gate."""
    stabilizers, x_logicals, z_logicals = cnot_unitary

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        k=2,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
        lower_bound=1,
        upper_bound=3,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == 1
    assert result.verified is True


def test_s_gate_unitary(s_gate_unitary: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test synthesis of S gate."""
    stabilizers, x_logicals, z_logicals = s_gate_unitary

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        k=1,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
        lower_bound=1,
        upper_bound=3,
    )

    print(result.circuit.to_stim_circuit(with_resets=False))
    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == 1
    assert result.verified is True


def test_swap_unitary(swap_unitary: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test synthesis of SWAP gate."""
    stabilizers, x_logicals, z_logicals = swap_unitary

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        k=2,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
        lower_bound=0,
        upper_bound=5,
        allow_qubit_permutation=False,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == 3
    assert result.verified is True


def test_verification_enabled(plus_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test that verification works correctly."""
    stabilizers, _x_logicals, _z_logicals = plus_state

    result = synthesize_exact(
        target=stabilizers,
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
    assert verify_stabilizer_state(result.circuit, stabilizers)


def test_verification_disabled(plus_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test that verification can be disabled."""
    stabilizers, _x_logicals, _z_logicals = plus_state

    result = synthesize_exact(
        target=stabilizers,
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


def test_css_isometry_synthesis() -> None:
    """Test CSS isometry synthesis with X-type checks."""
    checks = CheckMatrix(
        np.array([[1, 1, 1, 1]], dtype=np.int8),
        "X",
    )
    x_logicals = CheckMatrix(
        np.array([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=np.int8),
        "X",
    )
    z_logicals = CheckMatrix(
        np.array([[1, 0, 1, 0], [1, 1, 0, 0]], dtype=np.int8),
        "Z",
    )

    result = synthesize_exact(
        target=checks,
        target_kind=TargetKind.CSS_ISOMETRY,
        gate_family=GateFamily.CSS_CNOT,
        objective=Objective.GATE_COUNT,
        k=2,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
        lower_bound=1,
        upper_bound=5,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count >= 1
    assert result.verified is True
    assert verify_css_isometry(result.circuit, checks, x_logicals, k=2)


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


def test_depth_optimization_not_implemented(
    plus_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Test that depth optimization raises NotImplementedError."""
    stabilizers, _x_logicals, _z_logicals = plus_state

    with pytest.raises(NotImplementedError, match="Depth optimization not yet implemented"):
        synthesize_exact(
            target=stabilizers,
            target_kind=TargetKind.STABILIZER_STATE,
            gate_family=GateFamily.CLIFFORD,
            objective=Objective.DEPTH,
            lower_bound=1,
            upper_bound=3,
        )


def test_qubit_permutation_disabled(
    hadamard_unitary: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Test synthesis with qubit permutation disabled."""
    stabilizers, x_logicals, z_logicals = hadamard_unitary

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        k=1,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
        lower_bound=1,
        upper_bound=3,
        allow_qubit_permutation=False,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None


def test_identity_state(zero_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test synthesis of |0⟩ state (identity circuit)."""
    stabilizers, _x_logicals, _z_logicals = zero_state

    result = synthesize_exact(
        target=stabilizers,
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
    assert verify_stabilizer_state(result.circuit, stabilizers)


def test_identity_unitary(identity_unitary: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test synthesis of 2-qubit identity gate."""
    stabilizers, x_logicals, z_logicals = identity_unitary

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        k=2,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
        lower_bound=0,
        upper_bound=2,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == 0
    assert result.verified is True


def test_two_qubit_state(two_qubit_zero_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test synthesis of |00⟩ state."""
    stabilizers, _x_logicals, _z_logicals = two_qubit_zero_state

    result = synthesize_exact(
        target=stabilizers,
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
    assert verify_stabilizer_state(result.circuit, stabilizers)
