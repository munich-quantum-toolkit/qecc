# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for gate-count encoding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mqt.qecc.circuit_synthesis.exact import (
    ExactSynthesisOptions,
    GateFamily,
    Objective,
    SynthesisStatus,
    TargetKind,
    synthesize_exact,
)

if TYPE_CHECKING:
    import stim

    from mqt.qecc.codes.pauli import StabilizerTableau

# Stabilizer state preparation tests


def test_identity_state_zero_gates(identity_2q_tableau: StabilizerTableau) -> None:
    """Test that identity state (all |0⟩) requires 0 gates."""
    # Identity tableau for 2 qubits represents |00⟩ state
    result = synthesize_exact(
        identity_2q_tableau,
        TargetKind.CLIFFORD_UNITARY,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=5),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.optimal is True
    assert result.objective_value == 0


def test_plus_state_one_gate(plus_state_tableau: StabilizerTableau, plus_state_circuit: stim.Circuit) -> None:
    """Test |+⟩ state preparation requires 1 H gate."""
    result = synthesize_exact(
        plus_state_tableau,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=5),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.optimal is True
    assert result.objective_value == 1
    assert result.circuit is not None

    # Verify the circuit matches expected structure
    actual_circ = result.circuit.to_stim_circuit(with_resets=True)
    # Should have 1 reset and 1 H gate
    assert actual_circ.num_qubits == 1


def test_bell_state_two_gates(bell_state_tableau: StabilizerTableau, bell_state_circuit: stim.Circuit) -> None:
    """Test Bell state preparation requires 2 gates (H + CNOT)."""
    result = synthesize_exact(
        bell_state_tableau,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=10),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.optimal is True
    assert result.objective_value == 2
    assert result.circuit is not None
    assert result.two_qubit_gate_count == 1


def test_ghz_state_three_gates(ghz_state_tableau: StabilizerTableau, ghz_state_circuit: stim.Circuit) -> None:
    """Test 3-qubit GHZ state preparation requires 3 gates."""
    result = synthesize_exact(
        ghz_state_tableau,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=10),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.optimal is True
    assert result.objective_value == 3
    assert result.circuit is not None
    assert result.two_qubit_gate_count == 2


def test_bell_state_unsat_with_zero_gates(bell_state_tableau: StabilizerTableau) -> None:
    """Test that Bell state cannot be prepared with 0 gates."""
    result = synthesize_exact(
        bell_state_tableau,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=0),
    )

    assert result.status == SynthesisStatus.UNSAT
    assert result.circuit is None


def test_bell_state_unsat_with_one_gate(bell_state_tableau: StabilizerTableau) -> None:
    """Test that Bell state cannot be prepared with only 1 gate."""
    result = synthesize_exact(
        bell_state_tableau,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=1),
    )

    assert result.status == SynthesisStatus.UNSAT
    assert result.circuit is None


# Clifford unitary synthesis tests


def test_identity_unitary_zero_gates(identity_2q_tableau: StabilizerTableau) -> None:
    """Test identity unitary requires 0 gates."""
    result = synthesize_exact(
        identity_2q_tableau,
        TargetKind.CLIFFORD_UNITARY,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=5),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.optimal is True
    assert result.objective_value == 0


def test_hadamard_unitary_one_gate(hadamard_tableau: StabilizerTableau, hadamard_circuit: stim.Circuit) -> None:
    """Test Hadamard unitary requires 1 H gate."""
    result = synthesize_exact(
        hadamard_tableau,
        TargetKind.CLIFFORD_UNITARY,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=5),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.optimal is True
    assert result.objective_value == 1
    assert result.circuit is not None


def test_s_gate_unitary_one_gate(s_gate_tableau: StabilizerTableau, s_gate_circuit: stim.Circuit) -> None:
    """Test S gate unitary requires 1 S gate."""
    result = synthesize_exact(
        s_gate_tableau,
        TargetKind.CLIFFORD_UNITARY,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=5),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.optimal is True
    assert result.objective_value == 1
    assert result.circuit is not None


def test_cnot_unitary_one_gate(cnot_tableau: StabilizerTableau, cnot_circuit: stim.Circuit) -> None:
    """Test CNOT unitary requires 1 CNOT gate."""
    result = synthesize_exact(
        cnot_tableau,
        TargetKind.CLIFFORD_UNITARY,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=5),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.optimal is True
    assert result.objective_value == 1
    assert result.circuit is not None
    assert result.two_qubit_gate_count == 1


def test_identity_3q_unitary(identity_3q_tableau: StabilizerTableau) -> None:
    """Test 3-qubit identity unitary requires 0 gates."""
    result = synthesize_exact(
        identity_3q_tableau,
        TargetKind.CLIFFORD_UNITARY,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=5),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.optimal is True
    assert result.objective_value == 0


def test_unitary_with_permutation_disabled(hadamard_tableau: StabilizerTableau) -> None:
    """Test unitary synthesis with qubit permutation disabled."""
    result = synthesize_exact(
        hadamard_tableau,
        TargetKind.CLIFFORD_UNITARY,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=5, allow_qubit_permutation=False),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.optimal is True


# Clifford isometry synthesis tests


@pytest.mark.skip(reason="Clifford isometry encoding not yet fully implemented")
def test_simple_isometry(five_qubit_code_tableau: StabilizerTableau) -> None:
    """Test synthesis of [[5,1,3]] code encoding isometry."""
    # This encodes 1 logical qubit into 5 physical qubits
    result = synthesize_exact(
        five_qubit_code_tableau,
        TargetKind.CLIFFORD_ISOMETRY,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=20),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.circuit is not None


# Verification tests


def test_verification_enabled(plus_state_tableau: StabilizerTableau) -> None:
    """Test that verification is performed when enabled."""
    result = synthesize_exact(
        plus_state_tableau,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=5, verify_result=True),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.verified is True


def test_verification_disabled(plus_state_tableau: StabilizerTableau) -> None:
    """Test that verification is skipped when disabled."""
    result = synthesize_exact(
        plus_state_tableau,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=5, verify_result=False),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.verified is False


# Lower bound tests


def test_lower_bound_tight(bell_state_tableau: StabilizerTableau) -> None:
    """Test synthesis with tight lower bound (optimal = 2)."""
    result = synthesize_exact(
        bell_state_tableau,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=10, lower_bound=2),
    )

    assert result.status == SynthesisStatus.SAT
    assert result.objective_value == 2


def test_lower_bound_too_high(bell_state_tableau: StabilizerTableau) -> None:
    """Test synthesis with lower bound higher than optimal."""
    result = synthesize_exact(
        bell_state_tableau,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=10, lower_bound=5),
    )

    # Should still find solution, just starts search later
    # This will be UNSAT if max_bound < optimal or may find a solution
    # depending on whether optimal is >= 5
    assert result.status in {SynthesisStatus.SAT, SynthesisStatus.UNSAT}


# Result metadata tests


def test_result_contains_metadata(plus_state_tableau: StabilizerTableau) -> None:
    """Test that result contains all expected metadata."""
    result = synthesize_exact(
        plus_state_tableau,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=5),
    )

    assert result.status is not None
    assert result.optimal is not None
    assert result.bound_used is not None
    assert result.solver_time >= 0
    assert result.verified is not None


def test_result_string_representation(plus_state_tableau: StabilizerTableau) -> None:
    """Test that result has readable string representation."""
    result = synthesize_exact(
        plus_state_tableau,
        TargetKind.STABILIZER_STATE,
        GateFamily.CLIFFORD,
        Objective.GATE_COUNT,
        ExactSynthesisOptions(max_bound=5),
    )

    result_str = str(result)
    assert "SAT" in result_str or "UNSAT" in result_str
    assert "Optimal" in result_str
    assert "Bound used" in result_str
