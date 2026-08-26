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


def test_bit_flip_syndrome_channel() -> None:
    """Apply certain classical flips to a binary syndrome."""
    model = PhenomenologicalNoiseModel(data=PauliChannel(0.0, 0.0, 0.0), syndrome=BitFlipChannel(1.0))
    sampled = PhenomenologicalNoiseSampler(model, np.random.default_rng(1)).sample_syndrome(np.array([0, 1]))
    assert np.array_equal(sampled, np.array([1, 0]))


def test_gaussian_syndrome_channel_without_noise() -> None:
    """Map binary syndromes to signed observations at zero width."""
    model = PhenomenologicalNoiseModel(data=PauliChannel(0.0, 0.0, 0.0), syndrome=GaussianReadoutChannel(0.0))
    sampled = PhenomenologicalNoiseSampler(model, np.random.default_rng(1)).sample_syndrome(np.array([0, 1]))
    assert np.array_equal(sampled, np.array([1.0, -1.0]))
