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

from mqt.qecc.codes.pauli import CheckMatrix, StabilizerTableau


@pytest.fixture
def bell_state() -> tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]:
    """Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2 with stabilizers XX and ZZ."""
    stabilizers = StabilizerTableau.from_pauli_strings(["XX", "ZZ"])
    x_logicals = StabilizerTableau.empty(2)
    z_logicals = StabilizerTableau.empty(2)
    return stabilizers, x_logicals, z_logicals


@pytest.fixture
def plus_state() -> tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]:
    """|+⟩ state with stabilizer X."""
    stabilizers = StabilizerTableau.from_pauli_strings(["X"])
    x_logicals = StabilizerTableau.empty(1)
    z_logicals = StabilizerTableau.empty(1)
    return stabilizers, x_logicals, z_logicals


@pytest.fixture
def zero_state() -> tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]:
    """|0⟩ state with stabilizer Z."""
    stabilizers = StabilizerTableau.from_pauli_strings(["Z"])
    x_logicals = StabilizerTableau.empty(1)
    z_logicals = StabilizerTableau.empty(1)
    return stabilizers, x_logicals, z_logicals


@pytest.fixture
def two_qubit_zero_state() -> tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]:
    """|00⟩ state with stabilizers ZI and IZ."""
    stabilizers = StabilizerTableau.from_pauli_strings(["ZI", "IZ"])
    x_logicals = StabilizerTableau.empty(2)
    z_logicals = StabilizerTableau.empty(2)
    return stabilizers, x_logicals, z_logicals


@pytest.fixture
def ghz_state() -> tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]:
    """3-qubit GHZ state (|000⟩ + |111⟩)/√2 with stabilizers XXX, ZZI, IZZ."""
    stabilizers = StabilizerTableau.from_pauli_strings(["XXX", "ZZI", "IZZ"])
    x_logicals = StabilizerTableau.empty(3)
    z_logicals = StabilizerTableau.empty(3)
    return stabilizers, x_logicals, z_logicals


@pytest.fixture
def hadamard_unitary() -> tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]:
    """Single-qubit Hadamard unitary (swaps X and Z)."""
    stabilizers = StabilizerTableau.empty(1)
    x_logicals = StabilizerTableau.from_pauli_strings(["Z"])
    z_logicals = StabilizerTableau.from_pauli_strings(["X"])
    return stabilizers, x_logicals, z_logicals


@pytest.fixture
def s_gate_unitary() -> tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]:
    """Single-qubit S gate unitary (X -> Y, Z -> Z)."""
    stabilizers = StabilizerTableau.empty(1)
    x_logicals = StabilizerTableau.from_pauli_strings(["Y"])
    z_logicals = StabilizerTableau.from_pauli_strings(["Z"])
    return stabilizers, x_logicals, z_logicals


@pytest.fixture
def cnot_unitary() -> tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]:
    """2-qubit CNOT unitary."""
    stabilizers = StabilizerTableau.empty(2)
    x_logicals = StabilizerTableau.from_pauli_strings(["XI", "XX"])
    z_logicals = StabilizerTableau.from_pauli_strings(["ZZ", "IZ"])
    return stabilizers, x_logicals, z_logicals


@pytest.fixture
def swap_unitary() -> tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]:
    """2-qubit SWAP unitary."""
    stabilizers = StabilizerTableau.empty(2)
    x_logicals = StabilizerTableau.from_pauli_strings(["IX", "XI"])
    z_logicals = StabilizerTableau.from_pauli_strings(["IZ", "ZI"])
    return stabilizers, x_logicals, z_logicals


@pytest.fixture
def identity_unitary() -> tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]:
    """2-qubit identity unitary."""
    stabilizers = StabilizerTableau.empty(2)
    x_logicals = StabilizerTableau.from_pauli_strings(["XI", "IX"])
    z_logicals = StabilizerTableau.from_pauli_strings(["ZI", "IZ"])
    return stabilizers, x_logicals, z_logicals


@pytest.fixture
def repetition_code_check_matrix() -> CheckMatrix:
    """3-qubit repetition code check matrix (2 X-type checks)."""
    hx = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8)
    return CheckMatrix(hx, pauli_type="X")


@pytest.fixture
def five_qubit_code() -> tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau]:
    """[[5,1,3]] five-qubit perfect code encoder."""
    stabilizers = StabilizerTableau.from_pauli_strings([
        "XZZXI",
        "IXZZX",
        "XIXZZ",
        "ZXIXZ",
    ])
    x_logicals = StabilizerTableau.from_pauli_strings(["XXXXX"])
    z_logicals = StabilizerTableau.from_pauli_strings(["ZZZZZ"])
    return stabilizers, x_logicals, z_logicals
