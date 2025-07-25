# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test noisy circuit construction."""

from __future__ import annotations

import pytest
from stim import Circuit

from mqt.qecc.circuit_synthesis.noise import (
    CircuitLevelNoise,
    CircuitLevelNoiseIdlingParallel,
    CircuitLevelNoiseIdlingSequential,
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
    ],
)
def test_circuit_level_noise(noise_free, expected_noisy):
    """Test the circuit-level noise model."""
    noise_model = CircuitLevelNoise(p_tqg=0.01, p_sqg=0.02, p_meas=0.03, p_init=0.04)
    noisy = noise_model.apply(noise_free)

    # Check that the noisy circuit has the expected operations
    assert noisy == expected_noisy, f"Expected: {expected_noisy}, Got: {noisy}"


@pytest.mark.parametrize(
    ("noise_free", "expected_noisy"),
    [
        (
            Circuit("RX 0 1 2\nCX 0 1\nCX 1 2"),
            Circuit(
                "RX 0 1 2\nDEPOLARIZE1(0.04) 0 1 2\nCX 0 1\nDEPOLARIZE2(0.01) 0 1\nCX 1 2\nDEPOLARIZE2(0.01) 1 2\nDEPOLARIZE1(0.5) 0\n"
            ),
        ),
        (Circuit(), Circuit()),
        (
            Circuit("RX 0 1\nH 0\nH 0\nH 0\nCX 0 1"),
            Circuit(
                "RX 0 1\nDEPOLARIZE1(0.04) 0 1\nH 0\nDEPOLARIZE1(0.02) 0\nH 0\nDEPOLARIZE1(0.02) 0\nH 0\nDEPOLARIZE1(0.02) 0\nCX 0 1\nDEPOLARIZE2(0.01) 0 1"
            ),
        ),
    ],
)
def test_circuit_level_noise_idling_parallel_alap(noise_free, expected_noisy):
    """Test the circuit-level noise model."""
    noise_model = CircuitLevelNoiseIdlingParallel(
        p_tqg=0.01, p_sqg=0.02, p_meas=0.03, p_init=0.04, p_idle=0.5, resets_alap=True
    )
    noisy = noise_model.apply(noise_free)

    # Check that the noisy circuit has the expected operations
    assert noisy == expected_noisy, f"Expected: {expected_noisy}, Got: {noisy}"


@pytest.mark.parametrize(
    ("noise_free", "expected_noisy"),
    [
        (
            Circuit("RX 0 1 2\nCX 0 1\nCX 1 2"),
            Circuit(
                "RX 0 1 2\nDEPOLARIZE1(0.04) 0 1 2\nCX 0 1\nDEPOLARIZE2(0.01) 0 1\nDEPOLARIZE1(0.5) 2\nCX 1 2\nDEPOLARIZE2(0.01) 1 2\nDEPOLARIZE1(0.5) 0\n"
            ),
        ),
        (Circuit(), Circuit()),
        (
            Circuit("RX 0 1\nH 0\nH 0\nH 0\nCX 0 1"),
            Circuit(
                "RX 0 1\nDEPOLARIZE1(0.04) 0 1\nH 0\nDEPOLARIZE1(0.02) 0\nDEPOLARIZE1(0.5) 1\nH 0\nDEPOLARIZE1(0.02) 0\nDEPOLARIZE1(0.5) 1\nH 0\nDEPOLARIZE1(0.02) 0\nDEPOLARIZE1(0.5) 1\nCX 0 1\nDEPOLARIZE2(0.01) 0 1"
            ),
        ),
    ],
)
def test_circuit_level_noise_idling_parallel_asap(noise_free, expected_noisy):
    """Test the circuit-level noise model."""
    noise_model = CircuitLevelNoiseIdlingParallel(
        p_tqg=0.01, p_sqg=0.02, p_meas=0.03, p_init=0.04, p_idle=0.5, resets_alap=False
    )
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
    noise_model = CircuitLevelNoiseIdlingSequential(
        p_tqg=0.01, p_sqg=0.02, p_meas=0.03, p_init=0.04, p_idle=0.5, resets_alap=True
    )
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
        # (
        #     Circuit("RX 0 1\nH 0\nH 0\nH 0\nCX 0 1"),
        #     Circuit(
        #         "RX 0\nDEPOLARIZE1(0.04) 0\nRX 1\nDEPOLARIZE1(0.04) 1\nRX 2\nDEPOLARIZE1(0.04) 2\nCX 0 1\nDEPOLARIZE2(0.01) 0 1\nCX 1 2\nDEPOLARIZE2(0.01) 1 2\nDEPOLARIZE1(0.5) 0\n"
        #     ),
        # ),
    ],
)
def test_circuit_level_noise_idling_sequential_asap(noise_free, expected_noisy):
    """Test the circuit-level noise model."""
    noise_model = CircuitLevelNoiseIdlingSequential(
        p_tqg=0.01, p_sqg=0.02, p_meas=0.03, p_init=0.04, p_idle=0.5, resets_alap=False
    )
    noisy = noise_model.apply(noise_free)

    # Check that the noisy circuit has the expected operations
    assert noisy == expected_noisy, f"Expected: {expected_noisy}, Got: {noisy}"
