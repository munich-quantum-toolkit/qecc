# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for phenomenological noise sampling."""

from __future__ import annotations

import numpy as np

from mqt.qecc.noise import (
    BitFlipChannel,
    GaussianReadoutChannel,
    PauliChannel,
    PhenomenologicalNoiseModel,
    PhenomenologicalNoiseSampler,
)


def test_seeded_data_sampling_is_reproducible() -> None:
    """Produce the same sequence from generators initialized with the same seed."""
    model = PhenomenologicalNoiseModel(data=PauliChannel(0.2, 0.1, 0.3))
    first = PhenomenologicalNoiseSampler(model, np.random.default_rng(42)).sample_data(100)
    second = PhenomenologicalNoiseSampler(model, np.random.default_rng(42)).sample_data(100)
    assert np.array_equal(first, second)


def test_data_sampling_does_not_mutate_residual() -> None:
    """Return new arrays when composing sampled faults with a residual error."""
    residual = (np.ones(4, dtype=np.int32), np.zeros(4, dtype=np.int32))
    model = PhenomenologicalNoiseModel(data=PauliChannel(1.0, 0.0, 0.0))
    sampled = PhenomenologicalNoiseSampler(model, np.random.default_rng(1)).sample_data(4, residual)
    assert np.array_equal(residual[0], np.ones(4, dtype=np.int32))
    assert np.array_equal(sampled[0], np.zeros(4, dtype=np.int32))


def test_per_qubit_data_channel_assignments() -> None:
    """Resolve model-level channel assignments while sampling data errors."""
    model = PhenomenologicalNoiseModel(
        data=PauliChannel(0.0, 0.0, 0.0),
        data_by_qubit={0: PauliChannel(1.0, 0.0, 0.0)},
    )
    sampled = PhenomenologicalNoiseSampler(model, np.random.default_rng(1)).sample_data(2)
    assert np.array_equal(sampled[0], np.array([1, 0]))
    assert np.array_equal(sampled[1], np.array([0, 0]))


def test_model_factory_compresses_probability_arrays_to_overrides() -> None:
    """Construct a default channel plus assignments from legacy arrays."""
    model = PhenomenologicalNoiseModel.from_pauli_probabilities(np.array([0.1, 0.2]), np.zeros(2), np.zeros(2))
    assert model.data == PauliChannel(0.1, 0.0, 0.0)
    assert model.data_by_qubit == {1: PauliChannel(0.2, 0.0, 0.0)}


def test_bit_flip_syndrome_channel() -> None:
    """Apply certain classical flips to a binary syndrome."""
    model = PhenomenologicalNoiseModel(data=PauliChannel(0.0, 0.0, 0.0), x_syndrome=BitFlipChannel(1.0))
    sampled = PhenomenologicalNoiseSampler(model, np.random.default_rng(1)).sample_x_syndrome(np.array([0, 1]))
    assert np.array_equal(sampled, np.array([1, 0]))


def test_gaussian_syndrome_channel_without_noise() -> None:
    """Map binary syndromes to signed observations at zero width."""
    model = PhenomenologicalNoiseModel(data=PauliChannel(0.0, 0.0, 0.0), z_syndrome=GaussianReadoutChannel(0.0))
    sampled = PhenomenologicalNoiseSampler(model, np.random.default_rng(1)).sample_z_syndrome(np.array([0, 1]))
    assert np.array_equal(sampled, np.array([1.0, -1.0]))
