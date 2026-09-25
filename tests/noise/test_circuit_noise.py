# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test noisy circuit construction."""

from __future__ import annotations

import pytest
from stim import Circuit

from mqt.qecc.noise import (
    BitFlipChannel,
    CircuitNoiseModel,
    DepolarizingChannel,
    IdentityChannel,
    ParallelSchedule,
    SequentialSchedule,
    StimCircuitNoiseAdapter,
)


def uniform_model(p_idle: float = 0.0, ideal_qubits: frozenset[int] = frozenset()) -> CircuitNoiseModel:
    """The depolarizing configuration these tests pin, at the rates used throughout."""
    return CircuitNoiseModel(
        single_qubit_gate=DepolarizingChannel(0.02),
        two_qubit_gate=DepolarizingChannel(0.01),
        reset=DepolarizingChannel(0.04),
        measurement=BitFlipChannel(0.03),
        idle=DepolarizingChannel(p_idle) if p_idle else IdentityChannel(),
        ideal_qubits=ideal_qubits,
    )


@pytest.mark.parametrize(
    ("noise_free", "expected_noisy"),
    [
        (
            Circuit("RX 0\nR 1\n CX 0 1\n MR 0\n H 1\n MRX 1"),
            Circuit(
                "RX 0\nDEPOLARIZE1(0.04) 0\nR 1\nDEPOLARIZE1(0.04) 1\nCX 0 1\nDEPOLARIZE2(0.01) 0 1\nMR(0.03) 0\nH 1\nDEPOLARIZE1(0.02) 1\nMRX(0.03) 1\n"
            ),
        ),
        (Circuit(), Circuit()),
        (
            Circuit(
                "RX 0\nDEPOLARIZE1(0.04) 0\nR 1\nDEPOLARIZE1(0.04) 1\nCX 0 1\nDEPOLARIZE2(0.01) 0 1\nMR(0.03) 0\nH 1\nDEPOLARIZE1(0.02) 1\nMRX(0.03) 1\n"
            ),
            Circuit(
                "RX 0\nDEPOLARIZE1(0.04) 0\nDEPOLARIZE1(0.04) 0\nR 1\nDEPOLARIZE1(0.04) 1\nDEPOLARIZE1(0.04) 1\nCX 0 1\nDEPOLARIZE2(0.01) 0 1\nDEPOLARIZE2(0.01) 0 1\nMR(0.03) 0\nH 1\nDEPOLARIZE1(0.02) 1\nDEPOLARIZE1(0.02) 1\nMRX(0.03) 1\n"
            ),
        ),
    ],
)
def test_circuit_level_noise(noise_free, expected_noisy):
    """Test the circuit-level noise model."""
    noise_model = StimCircuitNoiseAdapter(uniform_model())
    noisy = noise_model.apply(noise_free)

    # Check that the noisy circuit has the expected operations
    assert noisy == expected_noisy, f"Expected: {expected_noisy}, Got: {noisy}"


@pytest.mark.parametrize(
    ("noise_free", "expected_noisy"),
    [
        (
            Circuit("RX 0 1 2\nCX 0 1\nCX 1 2"),
            Circuit(
                "RX 0\nDEPOLARIZE1(0.04) 0\nRX 1\n\nDEPOLARIZE1(0.04) 1\nRX 2\n\nDEPOLARIZE1(0.04) 2\nCX 0 1\nDEPOLARIZE2(0.01) 0 1\nCX 1 2\nDEPOLARIZE2(0.01) 1 2\nDEPOLARIZE1(0.5) 0\n"
            ),
        ),
        (Circuit(), Circuit()),
        (
            Circuit("RX 0 1\nH 0\nH 0\nH 0\nCX 0 1"),
            Circuit(
                "RX 0\nDEPOLARIZE1(0.04) 0\nRX 1\nDEPOLARIZE1(0.04) 1\nH 0\nDEPOLARIZE1(0.02) 0\nH 0\nDEPOLARIZE1(0.02) 0\nH 0\nDEPOLARIZE1(0.02) 0\nCX 0 1\nDEPOLARIZE2(0.01) 0 1"
            ),
        ),
    ],
)
def test_circuit_level_noise_idling_parallel_alap(noise_free, expected_noisy):
    """Test the circuit-level noise model."""
    noise_model = StimCircuitNoiseAdapter(uniform_model(p_idle=0.5), ParallelSchedule("alap"))
    noisy = noise_model.apply(noise_free)

    # Check that the noisy circuit has the expected operations
    assert noisy == expected_noisy, f"Expected: {expected_noisy}, Got: {noisy}"


@pytest.mark.parametrize(
    ("noise_free", "expected_noisy"),
    [
        (
            Circuit("RX 0 1 2\nCX 0 1\nCX 1 2"),
            Circuit(
                "RX 0\nDEPOLARIZE1(0.04) 0\nRX 1\n\nDEPOLARIZE1(0.04) 1\nRX 2\nDEPOLARIZE1(0.04) 2\nCX 0 1\nDEPOLARIZE2(0.01) 0 1\nDEPOLARIZE1(0.5) 2\nCX 1 2\nDEPOLARIZE2(0.01) 1 2\nDEPOLARIZE1(0.5) 0\n"
            ),
        ),
        (Circuit(), Circuit()),
        (
            Circuit("RX 0 1\nH 0\nH 0\nH 0\nCX 0 1"),
            Circuit(
                "RX 0\nDEPOLARIZE1(0.04) 0\nRX 1\nDEPOLARIZE1(0.04) 1\nH 0\nDEPOLARIZE1(0.02) 0\nDEPOLARIZE1(0.5) 1\nH 0\nDEPOLARIZE1(0.02) 0\nDEPOLARIZE1(0.5) 1\nH 0\nDEPOLARIZE1(0.02) 0\nDEPOLARIZE1(0.5) 1\nCX 0 1\nDEPOLARIZE2(0.01) 0 1"
            ),
        ),
    ],
)
def test_circuit_level_noise_idling_parallel_asap(noise_free, expected_noisy):
    """Test the circuit-level noise model."""
    noise_model = StimCircuitNoiseAdapter(uniform_model(p_idle=0.5), ParallelSchedule("asap"))
    noisy = noise_model.apply(noise_free)

    # Check that the noisy circuit has the expected operations
    assert noisy == expected_noisy, f"Expected: {expected_noisy}, Got: {noisy}"


@pytest.mark.parametrize(
    ("noise_free", "expected_noisy"),
    [
        (
            Circuit("RX 0 1 2\nCX 0 1\nCX 1 2"),
            Circuit(
                "RX 0\nDEPOLARIZE1(0.04) 0\nRX 1\nDEPOLARIZE1(0.04) 1\nRX 2\nDEPOLARIZE1(0.04) 2\nCX 0 1\nDEPOLARIZE2(0.01) 0 1\nCX 1 2\nDEPOLARIZE2(0.01) 1 2\nDEPOLARIZE1(0.5) 0\n"
            ),
        ),
        (Circuit(), Circuit()),
        (
            Circuit("RX 0 1\nH 0\nH 0\nH 0\nCX 0 1"),
            Circuit(
                "RX 0\nDEPOLARIZE1(0.04) 0\nRX 1\nDEPOLARIZE1(0.04) 1\nH 0\nDEPOLARIZE1(0.02) 0\nH 0\nDEPOLARIZE1(0.02) 0\nH 0\nDEPOLARIZE1(0.02) 0\nCX 0 1\nDEPOLARIZE2(0.01) 0 1"
            ),
        ),
    ],
)
def test_circuit_level_noise_idling_sequential_alap(noise_free, expected_noisy):
    """Test the circuit-level noise model."""
    noise_model = StimCircuitNoiseAdapter(uniform_model(p_idle=0.5), SequentialSchedule("alap"))
    noisy = noise_model.apply(noise_free)

    # Check that the noisy circuit has the expected operations
    assert noisy == expected_noisy, f"Expected: {expected_noisy}, Got: {noisy}"


@pytest.mark.parametrize(
    ("noise_free", "expected_noisy"),
    [
        (
            Circuit("RX 0 1 2\nCX 0 1\nCX 1 2"),
            Circuit(
                "RX 0\nDEPOLARIZE1(0.04) 0\nRX 1\nDEPOLARIZE1(0.04) 1\nDEPOLARIZE1(0.5) 0\nRX 2\nDEPOLARIZE1(0.04) 2\nDEPOLARIZE1(0.5) 0 1\nCX 0 1\nDEPOLARIZE2(0.01) 0 1\nDEPOLARIZE1(0.5) 2\nCX 1 2\nDEPOLARIZE2(0.01) 1 2\nDEPOLARIZE1(0.5) 0\n"
            ),
        ),
        (Circuit(), Circuit()),
    ],
)
def test_circuit_level_noise_idling_sequential_asap(noise_free, expected_noisy):
    """Test the circuit-level noise model."""
    noise_model = StimCircuitNoiseAdapter(uniform_model(p_idle=0.5), SequentialSchedule("asap"))
    noisy = noise_model.apply(noise_free)

    # Check that the noisy circuit has the expected operations
    assert noisy == expected_noisy, f"Expected: {expected_noisy}, Got: {noisy}"


@pytest.mark.parametrize(
    ("noise_free", "ideal_qubits", "expected_noisy"),
    [
        (
            Circuit("RX 0 1 2\nCX 0 1\nCX 1 2"),
            {0},
            Circuit(
                "RX 0\nRX 1\nDEPOLARIZE1(0.04) 1\nRX 2\nDEPOLARIZE1(0.04) 2\nCX 0 1\nCX 1 2\nDEPOLARIZE2(0.01) 1 2\n"
            ),
        ),
        (Circuit(), {0, 1, 2}, Circuit()),
    ],
)
def test_ideal_qubits(noise_free, ideal_qubits, expected_noisy):
    """Test no noise on ideal qubits."""
    noise_model = StimCircuitNoiseAdapter(uniform_model(ideal_qubits=frozenset(ideal_qubits)))
    noisy = noise_model.apply(noise_free)

    # Check that the noisy circuit has the expected operations
    assert noisy == expected_noisy, f"Expected: {expected_noisy}, Got: {noisy}"
