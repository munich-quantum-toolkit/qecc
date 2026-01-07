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

from mqt.qecc.circuit_synthesis.noise import (
    CircuitLevelNoise,
    CircuitLevelNoiseIdlingParallel,
    CircuitLevelNoiseIdlingSequential,
    ComposedNoiseModel,
    NoiseModel,
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
    noise_model = CircuitLevelNoise(p_tqg=0.01, p_sqg=0.02, p_meas=0.03, p_init=0.04, ideal_qubits=ideal_qubits)
    noisy = noise_model.apply(noise_free)

    # Check that the noisy circuit has the expected operations
    assert noisy == expected_noisy, f"Expected: {expected_noisy}, Got: {noisy}"


class MockNoiseModel(NoiseModel):
    """Mock noise model for testing purposes."""

    def __init__(self, operation: str, probability: float):
        """Initialize the mock noise model."""
        self.operation = operation
        self.probability = probability

    def apply(self, circ: Circuit) -> Circuit:
        """Apply the mock noise model."""
        noisy_circ = circ.copy()
        for i in range(circ.num_qubits):
            noisy_circ.append_operation(self.operation, [i], self.probability)
        return noisy_circ


@pytest.mark.parametrize(
    ("circ_string", "noise_models", "expected_circ_string"),
    [
        # Test case 1: Single noise model
        (
            "H 0\nCX 0 1",
            [MockNoiseModel("DEPOLARIZE1", 0.01)],
            "H 0\nCX 0 1\nDEPOLARIZE1(0.01) 0 1",
        ),
        # Test case 2: Multiple noise models
        (
            "H 0\nCX 0 1",
            [
                MockNoiseModel("Z_ERROR", 0.01),
                MockNoiseModel("X_ERROR", 0.02),
            ],
            "H 0\nCX 0 1\nZ_ERROR(0.01) 0 1\nX_ERROR(0.02) 0 1",
        ),
        # Test case 3: No noise models
        (
            "H 0\nCX 0 1",
            [],
            "H 0\nCX 0 1",
        ),
    ],
)
def test_composed_noise_model(circ_string, noise_models, expected_circ_string):
    """Parameterized test for ComposedNoiseModel."""
    # Create the circuit from the string
    circ = Circuit(circ_string)

    # Create the composed noise model
    composed_model = ComposedNoiseModel(noise_models)

    # Apply the composed noise model
    noisy_circ = composed_model.apply(circ)

    # Convert the noisy circuit to a string
    actual_circ_string = str(noisy_circ)

    # Check that the resulting circuit matches the expected circuit
    assert actual_circ_string.strip() == expected_circ_string.strip()
