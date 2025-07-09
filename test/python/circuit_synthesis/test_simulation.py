# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the simulation of fault-tolerant state preparation circuits."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import (
    LutDecoder,
    NoisyNDFTStatePrepSimulator,
    gate_optimal_verification_circuit,
    heuristic_prep_circuit,
    naive_verification_circuit,
)

if TYPE_CHECKING:  # pragma: no cover
    from qiskit import QuantumCircuit


@pytest.fixture
def steane_code() -> CSSCode:
    """Return the Steane code."""
    return CSSCode.from_code_name("steane")


@pytest.fixture
def non_ft_steane_zero(steane_code: CSSCode) -> QuantumCircuit:
    """Return a non fault-tolerant Steane code state preparation circuit."""
    return heuristic_prep_circuit(steane_code).circ


@pytest.fixture
def non_ft_steane_plus(steane_code: CSSCode) -> QuantumCircuit:
    """Return a non fault-tolerant Steane code state preparation circuit."""
    return heuristic_prep_circuit(steane_code, zero_state=False).circ


@pytest.fixture
def ft_steane_zero(steane_code: CSSCode) -> QuantumCircuit:
    """Return a fault-tolerant Steane code state preparation circuit."""
    circ = heuristic_prep_circuit(steane_code)
    return gate_optimal_verification_circuit(circ, max_timeout=2)


@pytest.fixture
def ft_steane_zero_naive(steane_code: CSSCode) -> QuantumCircuit:
    """Return a fault-tolerant Steane code state preparation circuit measuring all stabilizers."""
    circ = heuristic_prep_circuit(steane_code)
    return naive_verification_circuit(circ, flag_first_layer=True)


@pytest.fixture
def ft_steane_plus(steane_code: CSSCode) -> QuantumCircuit:
    """Return a fault-tolerant Steane code state preparation circuit."""
    circ = heuristic_prep_circuit(steane_code, zero_state=False)
    return gate_optimal_verification_circuit(circ, max_timeout=2)


@pytest.fixture
def steane_lut(steane_code: CSSCode) -> LutDecoder:
    """Return a LutDecoder for the Steane code."""
    return LutDecoder(steane_code, init_luts=True)


def test_lut(steane_code: CSSCode) -> None:
    """Test the LutDecoder class."""
    assert steane_code.Hx is not None, "Steane code does not have X stabilizers."
    assert steane_code.Hz is not None, "Steane code does not have Z stabilizers."

    lut = LutDecoder(steane_code, init_luts=False)

    assert len(lut.x_lut) == 0
    assert len(lut.z_lut) == 0

    lut.generate_x_lut()
    lut.generate_z_lut()

    assert len(lut.x_lut) != 0
    assert lut.x_lut is lut.z_lut  # Code is self dual so luts should be the same

    error_1 = np.zeros(steane_code.n, dtype=np.int8)
    error_1[0] = 1

    error_w1 = (steane_code.Hx[0] + error_1) % 2
    syndrome_1 = steane_code.get_x_syndrome(error_w1)
    estimate_1 = lut.decode_x(syndrome_1.astype(np.int8))
    assert steane_code.stabilizer_eq_x_error(estimate_1, error_1)
    assert steane_code.stabilizer_eq_z_error(estimate_1, error_1)

    error_2 = np.zeros(steane_code.n, dtype=np.int8)
    error_2[0] = 1
    error_2[1] = 1
    error_w2 = (steane_code.Hx[0] + error_2) % 2
    syndrome_2 = steane_code.get_x_syndrome(error_w2)
    estimate_2 = lut.decode_x(syndrome_2.astype(np.int8))

    # Weight 2 error should have be estimated to be weight 1
    assert not steane_code.stabilizer_eq_x_error(estimate_2, error_2)
    assert np.sum(estimate_2) == 1

    error_3 = np.ones((steane_code.n), dtype=np.int8)
    error_w3 = (steane_code.Hx[0] + error_3) % 2
    syndrome_3 = steane_code.get_x_syndrome(error_w3)
    estimate_3 = lut.decode_x(syndrome_3.astype(np.int8))
    # Weight 3 error should have be estimated to be weight 0
    assert not steane_code.stabilizer_eq_x_error(estimate_3, error_3)
    assert steane_code.stabilizer_eq_x_error(estimate_3, np.zeros(steane_code.n, dtype=np.int8))
    assert np.sum(estimate_3) == 0


def test_non_ft_sim_zero(steane_code: CSSCode, non_ft_steane_zero: QuantumCircuit) -> None:
    """Test the simulation of a non fault-tolerant state preparation circuit for the Steane |0>."""
    tol = 5e-4
    p = 1e-3
    lower = 1e-4
    simulator = NoisyNDFTStatePrepSimulator(
        non_ft_steane_zero,
        steane_code,
        p=p,
        p_idle=p * 0.01,
    )
    p_l, _, _, _ = simulator.logical_error_rate(min_errors=10)

    assert p_l - tol > lower


@pytest.mark.skipif(os.getenv("CI") is not None and sys.platform == "win32", reason="Too slow for CI on Windows")
def test_ft_sim_zero(steane_code: CSSCode, ft_steane_zero: QuantumCircuit) -> None:
    """Test the simulation of a fault-tolerant state preparation circuit for the Steane |0>."""
    tol = 5e-4
    p = 1e-3
    lower = 1e-4
    simulator = NoisyNDFTStatePrepSimulator(
        ft_steane_zero,
        steane_code,
        p=p,
        p_idle=p * 0.01,
    )
    p_l, _, _, _ = simulator.logical_error_rate(min_errors=10)

    assert p_l - tol < lower


def test_non_ft_sim_plus(steane_code: CSSCode, non_ft_steane_plus: QuantumCircuit, steane_lut: LutDecoder) -> None:
    """Test the simulation of a non fault-tolerant state preparation circuit for the Steane |0>."""
    tol = 5e-4
    p = 1e-3
    lower = 1e-4
    simulator = NoisyNDFTStatePrepSimulator(
        non_ft_steane_plus, steane_code, p=p, p_idle=p * 0.01, zero_state=False, decoder=steane_lut
    )
    p_l, _, _, _ = simulator.logical_error_rate(min_errors=10)

    assert p_l - tol > lower


@pytest.mark.skipif(os.getenv("CI") is not None and sys.platform == "win32", reason="Too slow for CI on Windows")
def test_ft_sim_plus(steane_code: CSSCode, ft_steane_plus: QuantumCircuit, steane_lut: LutDecoder) -> None:
    """Test the simulation of a fault-tolerant state preparation circuit for the Steane |0>."""
    tol = 5e-4
    p = 1e-3
    lower = 1e-4
    simulator = NoisyNDFTStatePrepSimulator(
        ft_steane_plus, steane_code, p=p, p_idle=p * 0.01, zero_state=False, decoder=steane_lut
    )
    p_l, _, _, _ = simulator.logical_error_rate(min_errors=10)

    assert p_l - tol < lower


def test_naive_verification_circuit_with_flags(
    steane_code: CSSCode, ft_steane_zero_naive: QuantumCircuit, steane_lut: LutDecoder
) -> None:
    """Test that naive verification is correct."""
    tol = 5e-4
    p = 1e-3
    lower = 1e-4
    simulator = NoisyNDFTStatePrepSimulator(
        ft_steane_zero_naive, steane_code, p=p, p_idle=p * 0.01, zero_state=True, decoder=steane_lut
    )
    p_l, _, _, _ = simulator.logical_error_rate(min_errors=10)

    assert p_l - tol < lower
