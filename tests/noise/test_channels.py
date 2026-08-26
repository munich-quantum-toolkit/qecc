# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for backend-independent noise channels and models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from mqt.qecc.noise import (
    BitFlipChannel,
    CircuitNoiseModel,
    DepolarizingChannel,
    GaussianReadoutChannel,
    PauliChannel,
    PhenomenologicalNoiseModel,
)


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("inf"), float("nan")])
def test_probability_validation(probability: float) -> None:
    """Reject values that are not probabilities."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        BitFlipChannel(probability)


def test_pauli_channel_from_bias() -> None:
    """Normalize finite bias and support a single infinite component."""
    channel = PauliChannel.from_total_probability(0.6, bias=(1.0, 2.0, 3.0))
    assert (channel.p_x, channel.p_y, channel.p_z) == pytest.approx((0.1, 0.2, 0.3))
    assert PauliChannel.from_total_probability(0.2, bias=(float("inf"), 0.0, 0.0)) == PauliChannel(0.2, 0.0, 0.0)


@pytest.mark.parametrize(
    ("bias", "match"),
    [
        ((0.0, 0.0, 0.0), "positive finite sum"),
        ((-1.0, 1.0, 1.0), "nonnegative"),
        ((float("inf"), float("inf"), 0.0), "At most one"),
    ],
)
def test_invalid_pauli_bias(bias: tuple[float, float, float], match: str) -> None:
    """Reject ambiguous or invalid Pauli bias."""
    with pytest.raises(ValueError, match=match):
        PauliChannel.from_total_probability(0.1, bias=bias)


def test_pauli_total_probability_validation() -> None:
    """Reject Pauli components whose sum exceeds one."""
    with pytest.raises(ValueError, match="sum to at most 1"):
        PauliChannel(0.4, 0.4, 0.4)


def test_phenomenological_model_owns_per_qubit_assignments() -> None:
    """Keep spatial channel assignments immutable and separate from channels."""
    overrides = {1: PauliChannel(0.2, 0.0, 0.0)}
    model = PhenomenologicalNoiseModel(data=PauliChannel(0.1, 0.0, 0.0), data_by_qubit=overrides)
    overrides[1] = PauliChannel(0.9, 0.0, 0.0)
    assert model.data_channel(0) == PauliChannel(0.1, 0.0, 0.0)
    assert model.data_channel(1) == PauliChannel(0.2, 0.0, 0.0)
    with pytest.raises(TypeError):
        model.data_by_qubit[1] = PauliChannel(0.3, 0.0, 0.0)  # ty: ignore[invalid-assignment]


def test_gaussian_conversion_round_trip() -> None:
    """Convert consistently between analog width and hard-decision error rate."""
    channel = GaussianReadoutChannel.from_bit_error_probability(0.1)
    assert channel.bit_error_probability == pytest.approx(0.1)


@pytest.mark.parametrize("sigma", [-0.1, float("inf"), float("nan")])
def test_gaussian_width_validation(sigma: float) -> None:
    """Reject invalid Gaussian standard deviations."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        GaussianReadoutChannel(sigma)


def test_zero_width_gaussian_conversion() -> None:
    """Handle the noiseless endpoint without dividing by zero."""
    channel = GaussianReadoutChannel.from_bit_error_probability(0.0)
    assert channel.sigma == pytest.approx(0.0)
    assert channel.bit_error_probability == pytest.approx(0.0)


@pytest.mark.parametrize("probability", [0.5, 1.0])
def test_gaussian_conversion_rejects_ambiguous_error_rates(probability: float) -> None:
    """Reject hard-decision error rates that cannot define a finite width."""
    with pytest.raises(ValueError, match=r"below 0\.5"):
        GaussianReadoutChannel.from_bit_error_probability(probability)


def test_phenomenological_model_validation() -> None:
    """Reject invalid per-qubit assignments and probability-array shapes."""
    with pytest.raises(ValueError, match="non-negative integers"):
        PhenomenologicalNoiseModel(data=PauliChannel(0.0, 0.0, 0.0), data_by_qubit={-1: PauliChannel(0.0, 0.0, 0.0)})
    with pytest.raises(ValueError, match="non-negative"):
        PhenomenologicalNoiseModel(data=PauliChannel(0.0, 0.0, 0.0)).data_channel(-1)
    with pytest.raises(ValueError, match="one-dimensional"):
        PhenomenologicalNoiseModel.from_pauli_probabilities(np.zeros((1, 2)), np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="identical shapes"):
        PhenomenologicalNoiseModel.from_pauli_probabilities(np.zeros(1), np.zeros(2), np.zeros(1))


def test_channels_and_models_are_immutable() -> None:
    """Keep configuration safe to share between simulation backends."""
    channel = DepolarizingChannel(0.1)
    with pytest.raises(FrozenInstanceError):
        channel.probability = 0.2  # ty: ignore[invalid-assignment]
    model = CircuitNoiseModel(ideal_qubits=frozenset({1}))
    with pytest.raises(FrozenInstanceError):
        model.ideal_qubits = frozenset()  # ty: ignore[invalid-assignment]


def test_ideal_qubit_validation() -> None:
    """Reject negative ideal-qubit indices."""
    with pytest.raises(ValueError, match="non-negative"):
        CircuitNoiseModel(ideal_qubits=frozenset({-1}))
