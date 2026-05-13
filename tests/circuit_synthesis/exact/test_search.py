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

from mqt.qecc.circuit_synthesis.exact.search import synthesize_exact
from mqt.qecc.circuit_synthesis.exact.types import GateFamily, Objective, SynthesisStatus, TargetKind
from mqt.qecc.circuit_synthesis.exact.verification import (
    verify_css_isometry,
    verify_css_state,
    verify_stabilizer_state,
)
from mqt.qecc.codes.pauli import CheckMatrix, StabilizerTableau


@pytest.mark.parametrize(
    ("fixture_name", "expected_gates"),
    [
        ("zero_state", 0),
        ("plus_state", 1),
        ("bell_state", 2),
        ("ghz_state", 3),
    ],
)
def test_stabilizer_state_gate_count(fixture_name: str, expected_gates: int, request: pytest.FixtureRequest) -> None:
    """Test stabilizer state preparation with optimal gate count."""
    stabilizers, _x_logicals, _z_logicals = request.getfixturevalue(fixture_name)

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=10,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == expected_gates
    assert result.verified is True
    assert verify_stabilizer_state(result.circuit, stabilizers)


@pytest.mark.parametrize(
    ("fixture_name", "k", "expected_gates"),
    [
        ("identity_unitary", 2, 0),
        ("hadamard_unitary", 1, 1),
        ("s_gate_unitary", 1, 1),
        ("cnot_unitary", 2, 1),
    ],
)
def test_clifford_unitary_gate_count(
    fixture_name: str, k: int, expected_gates: int, request: pytest.FixtureRequest
) -> None:
    """Test Clifford unitary synthesis with optimal gate count."""
    stabilizers, x_logicals, z_logicals = request.getfixturevalue(fixture_name)

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        k=k,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
        lower_bound=0,
        upper_bound=5,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.gate_count == expected_gates
    assert result.verified is True


@pytest.mark.parametrize(
    ("fixture_name", "expected_depth"),
    [
        ("zero_state", 0),
        ("plus_state", 1),
        ("bell_state", 2),
        ("ghz_state", 3),
    ],
)
def test_stabilizer_state_depth(fixture_name: str, expected_depth: int, request: pytest.FixtureRequest) -> None:
    """Test stabilizer state preparation with optimal depth."""
    stabilizers, _x_logicals, _z_logicals = request.getfixturevalue(fixture_name)

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.DEPTH,
        lower_bound=0,
        upper_bound=10,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.depth == expected_depth
    assert result.verified is True
    assert verify_stabilizer_state(result.circuit, stabilizers)


@pytest.mark.parametrize(
    ("fixture_name", "k", "expected_depth"),
    [
        ("hadamard_unitary", 1, 1),
        ("s_gate_unitary", 1, 1),
        ("cnot_unitary", 2, 1),
        ("swap_unitary", 2, 3),
    ],
)
def test_clifford_unitary_depth(fixture_name: str, k: int, expected_depth: int, request: pytest.FixtureRequest) -> None:
    """Test Clifford unitary synthesis with optimal depth."""
    stabilizers, x_logicals, z_logicals = request.getfixturevalue(fixture_name)

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.DEPTH,
        k=k,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
        lower_bound=0,
        upper_bound=10,
        allow_qubit_permutation=False,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.depth == expected_depth
    assert result.verified is True


def test_css_state_preparation(repetition_code_check_matrix: CheckMatrix) -> None:
    """Test CSS state preparation."""
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


def test_css_isometry_synthesis(css_isometry_4q: tuple[CheckMatrix, CheckMatrix, CheckMatrix]) -> None:
    """Test CSS isometry synthesis."""
    checks, x_logicals, z_logicals = css_isometry_4q

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


def test_css_state_depth(repetition_code_check_matrix: CheckMatrix) -> None:
    """Test CSS state preparation with depth optimization."""
    result = synthesize_exact(
        target=repetition_code_check_matrix,
        target_kind=TargetKind.CSS_STATE,
        gate_family=GateFamily.CSS_CNOT,
        objective=Objective.DEPTH,
        lower_bound=1,
        upper_bound=5,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.depth == 2
    assert result.verified is True
    assert verify_css_state(result.circuit, repetition_code_check_matrix)


def test_css_isometry_depth(css_isometry_4q: tuple[CheckMatrix, CheckMatrix, CheckMatrix]) -> None:
    """Test CSS isometry synthesis with depth optimization."""
    checks, x_logicals, z_logicals = css_isometry_4q

    result = synthesize_exact(
        target=checks,
        target_kind=TargetKind.CSS_ISOMETRY,
        gate_family=GateFamily.CSS_CNOT,
        objective=Objective.DEPTH,
        k=2,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
        lower_bound=1,
        upper_bound=5,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.depth == 2
    assert result.verified is True
    assert verify_css_isometry(result.circuit, checks, x_logicals, k=2)


def test_unsat_with_insufficient_bound(
    bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Test that synthesis fails when bound is too low."""
    stabilizers, _x_logicals, _z_logicals = bell_state

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=1,
    )

    assert result.status == SynthesisStatus.UNSAT
    assert result.circuit is None


def test_unsat_depth_zero(bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test that Bell state cannot be prepared with depth 0."""
    stabilizers, _x_logicals, _z_logicals = bell_state

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.DEPTH,
        lower_bound=0,
        upper_bound=0,
    )

    assert result.status == SynthesisStatus.UNSAT
    assert result.circuit is None


@pytest.mark.parametrize("verify_flag", [True, False])
def test_verification_flag(
    plus_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau], verify_flag: bool
) -> None:
    """Test that verification flag is respected."""
    stabilizers, _x_logicals, _z_logicals = plus_state

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=5,
        verify=verify_flag,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.verified is verify_flag


def test_lower_bound_respected(bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test that search respects lower bound."""
    stabilizers, _x_logicals, _z_logicals = bell_state

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=2,
        upper_bound=10,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.gate_count == 2


def test_exact_bound_match(bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test when optimal solution exactly matches bound."""
    stabilizers, _x_logicals, _z_logicals = bell_state

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=2,
        upper_bound=2,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.gate_count == 2


def test_qubit_permutation_disabled(
    swap_unitary: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Test synthesis with qubit permutation disabled."""
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


def test_zero_bound_with_identity_state(
    two_qubit_zero_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Test that identity state can be prepared with bound 0."""
    stabilizers, _x_logicals, _z_logicals = two_qubit_zero_state

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=0,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.gate_count == 0


def test_zero_bound_with_nontrivial_state(
    plus_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Test that non-trivial state cannot be prepared with bound 0."""
    stabilizers, _x_logicals, _z_logicals = plus_state

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=0,
    )

    assert result.status == SynthesisStatus.UNSAT


def test_state_with_minus_sign() -> None:
    """Test synthesis handles phases correctly for |-⟩ state."""
    target = StabilizerTableau.from_pauli_strings(["-X"])

    result = synthesize_exact(
        target,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=5,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.verified is True


def test_state_with_imaginary_phase() -> None:
    """Test synthesis handles imaginary phases for |i⟩ state."""
    target = StabilizerTableau.from_pauli_strings(["iY"])

    result = synthesize_exact(
        target,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=5,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.verified is True


def test_css_single_check() -> None:
    """Test CSS state with single check."""
    check_matrix = CheckMatrix(np.array([[1, 1]], dtype=np.int8), pauli_type="X")

    result = synthesize_exact(
        check_matrix,
        TargetKind.CSS_STATE,
        GateFamily.CSS_CNOT,
        Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=5,
    )

    assert result.status == SynthesisStatus.SUCCESS


def test_css_fully_connected_check() -> None:
    """Test CSS state with all-ones check (GHZ-like state)."""
    check_matrix = CheckMatrix(np.array([[1, 1, 1, 1]], dtype=np.int8), pauli_type="X")

    result = synthesize_exact(
        check_matrix,
        TargetKind.CSS_STATE,
        GateFamily.CSS_CNOT,
        Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=10,
    )

    assert result.status == SynthesisStatus.SUCCESS


def test_four_qubit_ghz_state() -> None:
    """Test synthesis on 4-qubit system."""
    target = StabilizerTableau.from_pauli_strings(["XXXX", "ZZII", "IZZI", "IIZZ"])

    result = synthesize_exact(
        target,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=10,
    )

    assert result.status == SynthesisStatus.SUCCESS


def test_invalid_bounds(plus_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test that invalid bounds are rejected."""
    stabilizers, _x_logicals, _z_logicals = plus_state

    with pytest.raises(ValueError, match="Invalid bounds"):
        synthesize_exact(
            target=stabilizers,
            target_kind=TargetKind.STABILIZER_STATE,
            gate_family=GateFamily.CLIFFORD,
            objective=Objective.GATE_COUNT,
            lower_bound=10,
            upper_bound=5,
        )


def test_negative_k_rejected(bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test that negative k is rejected."""
    stabilizers, _x_logicals, _z_logicals = bell_state

    with pytest.raises(ValueError, match="k must be non-negative"):
        synthesize_exact(
            target=stabilizers,
            target_kind=TargetKind.CLIFFORD_ISOMETRY,
            gate_family=GateFamily.CLIFFORD,
            objective=Objective.GATE_COUNT,
            k=-1,
            lower_bound=0,
            upper_bound=5,
        )


def test_clifford_isometry_missing_k(
    bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Test that isometry synthesis requires k parameter."""
    stabilizers, _x_logicals, _z_logicals = bell_state

    with pytest.raises(ValueError, match="k must be provided for isometry synthesis"):
        synthesize_exact(
            target=stabilizers,
            target_kind=TargetKind.CLIFFORD_ISOMETRY,
            gate_family=GateFamily.CLIFFORD,
            objective=Objective.GATE_COUNT,
            lower_bound=0,
            upper_bound=5,
        )


def test_clifford_isometry_missing_logicals(
    bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Test that Clifford isometry requires logical operators."""
    stabilizers, _x_logicals, _z_logicals = bell_state

    with pytest.raises(ValueError, match="x_logicals and z_logicals must be provided"):
        synthesize_exact(
            target=stabilizers,
            target_kind=TargetKind.CLIFFORD_ISOMETRY,
            gate_family=GateFamily.CLIFFORD,
            objective=Objective.GATE_COUNT,
            k=1,
            lower_bound=0,
            upper_bound=5,
        )


def test_mismatched_logical_count(bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test that mismatched logical operator counts are rejected."""
    stabilizers, _x_logicals, _z_logicals = bell_state
    x_logicals = StabilizerTableau.from_pauli_strings(["XI", "IX"])
    z_logicals = StabilizerTableau.from_pauli_strings(["ZI"])

    with pytest.raises(ValueError):
        synthesize_exact(
            target=stabilizers,
            target_kind=TargetKind.CLIFFORD_ISOMETRY,
            gate_family=GateFamily.CLIFFORD,
            objective=Objective.GATE_COUNT,
            k=2,
            x_logicals=x_logicals,
            z_logicals=z_logicals,
            lower_bound=0,
            upper_bound=5,
        )


def test_css_with_clifford_gate_family(repetition_code_check_matrix: CheckMatrix) -> None:
    """Test that CSS targets require CSS_CNOT gate family."""
    with pytest.raises(ValueError, match="CLIFFORD gate family requires StabilizerTableau"):
        synthesize_exact(
            target=repetition_code_check_matrix,
            target_kind=TargetKind.CSS_STATE,
            gate_family=GateFamily.CLIFFORD,
            objective=Objective.GATE_COUNT,
            lower_bound=0,
            upper_bound=5,
        )


def test_clifford_with_css_gate_family(
    plus_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Test that Clifford targets require CLIFFORD gate family."""
    stabilizers, _x_logicals, _z_logicals = plus_state

    with pytest.raises(ValueError, match="CSS_CNOT gate family requires CheckMatrix"):
        synthesize_exact(
            target=stabilizers,
            target_kind=TargetKind.STABILIZER_STATE,
            gate_family=GateFamily.CSS_CNOT,
            objective=Objective.GATE_COUNT,
            lower_bound=0,
            upper_bound=5,
        )


def test_result_contains_metadata(plus_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]) -> None:
    """Test that result contains all expected metadata."""
    stabilizers, _x_logicals, _z_logicals = plus_state

    result = synthesize_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        gate_family=GateFamily.CLIFFORD,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=5,
    )

    assert result.status is not None
    assert result.gate_count is not None
    assert result.verified is not None
    assert result.message is not None
    assert result.gate_set is not None
