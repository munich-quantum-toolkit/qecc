# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared fixtures for exact synthesis tests."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from mqt.qecc.codes.pauli import CheckMatrix, StabilizerTableau


@pytest.fixture
def identity_2q_tableau() -> StabilizerTableau:
    """2-qubit identity tableau."""
    return StabilizerTableau.identity(2)


@pytest.fixture
def identity_3q_tableau() -> StabilizerTableau:
    """3-qubit identity tableau."""
    return StabilizerTableau.identity(3)


@pytest.fixture
def bell_state_tableau() -> StabilizerTableau:
    """Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 with stabilizers XX and ZZ."""
    return StabilizerTableau.from_pauli_strings(["XX", "ZZ"])


@pytest.fixture
def bell_state_circuit() -> stim.Circuit:
    """Known optimal circuit for Bell state: H on qubit 0, then CNOT(0,1)."""
    circuit = stim.Circuit()
    circuit.append("H", [0])
    circuit.append("CX", [0, 1])
    return circuit


@pytest.fixture
def plus_state_tableau() -> StabilizerTableau:
    """|+⟩ state with stabilizer X."""
    return StabilizerTableau.from_pauli_strings(["X"])


@pytest.fixture
def plus_state_circuit() -> stim.Circuit:
    """Known optimal circuit for |+⟩ state: H on qubit 0."""
    circuit = stim.Circuit()
    circuit.append("H", [0])
    return circuit


@pytest.fixture
def ghz_state_tableau() -> StabilizerTableau:
    """3-qubit GHZ state (|000⟩ + |111⟩)/√2 with stabilizers XXX, ZZI, IZZ."""
    return StabilizerTableau.from_pauli_strings(["XXX", "ZZI", "IZZ"])


@pytest.fixture
def ghz_state_circuit() -> stim.Circuit:
    """Known optimal circuit for GHZ state: H on 0, CNOT(0,1), CNOT(1,2)."""
    circuit = stim.Circuit()
    circuit.append("H", [0])
    circuit.append("CX", [0, 1])
    circuit.append("CX", [1, 2])
    return circuit


@pytest.fixture
def hadamard_tableau() -> StabilizerTableau:
    """Single-qubit Hadamard unitary."""
    # H swaps X and Z
    return StabilizerTableau.from_pauli_strings(["Z", "X"])


@pytest.fixture
def hadamard_circuit() -> stim.Circuit:
    """Known circuit for Hadamard: H gate."""
    circuit = stim.Circuit()
    circuit.append("H", [0])
    return circuit


@pytest.fixture
def s_gate_tableau() -> StabilizerTableau:
    """Single-qubit S gate unitary."""
    # S: X -> Y, Z -> Z
    return StabilizerTableau.from_pauli_strings(["Y", "Z"])


@pytest.fixture
def s_gate_circuit() -> stim.Circuit:
    """Known circuit for S gate: S gate."""
    circuit = stim.Circuit()
    circuit.append("S", [0])
    return circuit


@pytest.fixture
def cnot_tableau() -> StabilizerTableau:
    """2-qubit CNOT unitary.

    CNOT(0,1): X0 -> X0X1, Z0 -> Z0, X1 -> X1, Z1 -> Z0Z1
    Tableau rows: [logical X0, logical X1, logical Z0, logical Z1]
    """
    return StabilizerTableau.from_pauli_strings(["XX", "IX", "ZI", "ZZ"])


@pytest.fixture
def cnot_circuit() -> stim.Circuit:
    """Known circuit for CNOT: CNOT(0,1)."""
    circuit = stim.Circuit()
    circuit.append("CX", [0, 1])
    return circuit


@pytest.fixture
def repetition_code_check_matrix() -> CheckMatrix:
    """3-qubit repetition code check matrix (2 X-type checks)."""
    hx = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
    return CheckMatrix(hx, pauli_type="X")


@pytest.fixture
def simple_css_encoder_circuit() -> stim.Circuit:
    """Simple CSS encoder circuit for testing.

    This is a placeholder - actual optimal circuit depends on the code.
    """
    # TODO: Determine exact optimal circuit
    return stim.Circuit()


@pytest.fixture
def five_qubit_code_tableau() -> StabilizerTableau:
    """[[5,1,3]] five-qubit perfect code stabilizers."""
    return StabilizerTableau.from_pauli_strings([
        "XZZXI",
        "IXZZX",
        "XIXZZ",
        "ZXIXZ",
    ])
