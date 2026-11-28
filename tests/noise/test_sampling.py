# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for phenomenological noise sampling."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc.noise import (
    BitFlipChannel,
    GaussianReadoutChannel,
    IdentityChannel,
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


def test_model_expands_default_and_overrides_to_arrays() -> None:
    """Expand the default channel and per-qubit overrides into probability arrays."""
    model = PhenomenologicalNoiseModel(data=PauliChannel(0.1, 0.0, 0.0), data_by_qubit={1: PauliChannel(0.2, 0.0, 0.0)})
    p_x, p_y, p_z = model.pauli_probabilities(2)
    assert np.array_equal(p_x, np.array([0.1, 0.2]))
    assert np.array_equal(p_y, np.zeros(2))
    assert np.array_equal(p_z, np.zeros(2))
    assert np.array_equal(model.x_marginals(2), np.array([0.1, 0.2]))
    assert np.array_equal(model.z_marginals(2), np.zeros(2))


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


def test_data_sampling_validates_dimensions() -> None:
    """Reject invalid sample sizes, residuals, and out-of-range assignments."""
    sampler = PhenomenologicalNoiseSampler(PhenomenologicalNoiseModel(data=PauliChannel(0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="nonnegative"):
        sampler.sample_data(-1)
    with pytest.raises(ValueError, match="shape"):
        sampler.sample_data(2, (np.zeros(1, dtype=np.int32), np.zeros(2, dtype=np.int32)))
    assigned = PhenomenologicalNoiseSampler(
        PhenomenologicalNoiseModel(data=PauliChannel(0.0, 0.0, 0.0), data_by_qubit={2: PauliChannel(0.0, 0.0, 0.0)})
    )
    with pytest.raises(ValueError, match="below n_qubits"):
        assigned.sample_data(2)


def test_syndrome_sampling_edge_cases() -> None:
    """Validate syndrome input and preserve identity-channel semantics."""
    sampler = PhenomenologicalNoiseSampler(PhenomenologicalNoiseModel(data=PauliChannel(0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="binary"):
        sampler.sample_syndrome(np.array([0, 2]), IdentityChannel())
    syndrome = np.array([0, 1], dtype=np.int32)
    sampled = sampler.sample_syndrome(syndrome, IdentityChannel())
    assert np.array_equal(sampled, syndrome)
    assert sampled is not syndrome
    with pytest.raises(TypeError, match="Unsupported readout channel"):
        sampler.sample_syndrome(syndrome, object())  # ty: ignore[invalid-argument-type]


def test_channel_rejects_probabilities_that_exceed_one() -> None:
    """The channel enforces what the removed array sampler used to re-check."""
    with pytest.raises(ValueError, match="sum to at most 1"):
        PauliChannel(0.6, 0.5, 0.0)


def test_seeded_sampling_reproduces_a_full_sequence() -> None:
    """Two samplers seeded alike agree over a long run, not just one draw."""
    model = PhenomenologicalNoiseModel(data=PauliChannel(0.2, 0.1, 0.3))
    first = PhenomenologicalNoiseSampler(model, np.random.default_rng(42))
    second = PhenomenologicalNoiseSampler(model, np.random.default_rng(42))
    for _ in range(10):
        assert np.array_equal(first.sample_data(100), second.sample_data(100))
