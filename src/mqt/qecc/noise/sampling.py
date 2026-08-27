# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sampling implementations for noise models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .channels import (
    BitFlipChannel,
    GaussianReadoutChannel,
    IdentityChannel,
    ReadoutChannel,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .models import PhenomenologicalNoiseModel


def _apply_pauli_samples(
    p_x: NDArray[np.float64],
    p_y: NDArray[np.float64],
    p_z: NDArray[np.float64],
    residual: tuple[NDArray[np.int32], NDArray[np.int32]],
    rng: np.random.Generator,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Draw one Pauli per location and XOR it onto a copy of the residual error."""
    samples = rng.random(p_x.shape)
    z_mask = samples < p_z
    x_mask = (p_z <= samples) & (samples < p_z + p_x)
    y_mask = (p_z + p_x <= samples) & (samples < p_z + p_x + p_y)
    error_x = np.array(residual[0], dtype=np.int32, copy=True) ^ np.asarray(x_mask | y_mask, dtype=np.int32)
    error_z = np.array(residual[1], dtype=np.int32, copy=True) ^ np.asarray(z_mask | y_mask, dtype=np.int32)
    return error_x, error_z


# ----------------------------------------------------------------------------------------------------
#   Phenomenological sampling
# ----------------------------------------------------------------------------------------------------


class PhenomenologicalNoiseSampler:
    """Sample data and syndrome noise using an explicitly owned generator."""

    def __init__(self, model: PhenomenologicalNoiseModel, rng: np.random.Generator | None = None) -> None:
        """Initialize the sampler.

        Args:
            model: Phenomenological noise model to sample.
            rng: Random generator. A fresh generator is used when omitted.
        """
        self.model = model
        self.rng = np.random.default_rng() if rng is None else rng

    def sample_data(
        self, n_qubits: int, residual: tuple[NDArray[np.int32], NDArray[np.int32]] | None = None
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Sample the X and Z components of Pauli errors."""
        probabilities = self.model.pauli_probabilities(n_qubits)
        if residual is None:
            residual = (np.zeros(n_qubits, dtype=np.int32), np.zeros(n_qubits, dtype=np.int32))
        elif residual[0].shape != (n_qubits,) or residual[1].shape != (n_qubits,):
            msg = "Residual X and Z arrays must both have shape (n_qubits,)."
            raise ValueError(msg)

        return _apply_pauli_samples(*probabilities, residual, self.rng)

    def sample_x_syndrome(self, perfect_syndrome: NDArray[np.int32]) -> NDArray[np.int32] | NDArray[np.float64]:
        """Apply the configured X-syndrome channel to a perfect syndrome."""
        return self.sample_syndrome(perfect_syndrome, self.model.x_syndrome)

    def sample_z_syndrome(self, perfect_syndrome: NDArray[np.int32]) -> NDArray[np.int32] | NDArray[np.float64]:
        """Apply the configured Z-syndrome channel to a perfect syndrome."""
        return self.sample_syndrome(perfect_syndrome, self.model.z_syndrome)

    def sample_syndrome(
        self, perfect_syndrome: NDArray[np.int32], channel: ReadoutChannel
    ) -> NDArray[np.int32] | NDArray[np.float64]:
        """Apply a syndrome channel to a perfect binary syndrome."""
        syndrome = np.asarray(perfect_syndrome, dtype=np.int32)
        if not np.all((syndrome == 0) | (syndrome == 1)):
            msg = "A perfect syndrome must contain only binary values."
            raise ValueError(msg)
        if isinstance(channel, IdentityChannel):
            return syndrome.copy()
        if isinstance(channel, BitFlipChannel):
            flips = self.rng.random(syndrome.shape) < channel.probability
            return syndrome ^ np.asarray(flips, dtype=np.int32)
        if isinstance(channel, GaussianReadoutChannel):
            signed = np.where(syndrome == 0, 1.0, -1.0)
            return np.asarray(self.rng.normal(signed, channel.sigma), dtype=np.float64)
        msg = f"Unsupported readout channel: {type(channel).__name__}."
        raise TypeError(msg)


def sample_inhomogeneous_pauli(
    channel_probabilities: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    residual: tuple[NDArray[np.int32], NDArray[np.int32]],
    rng: np.random.Generator,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Sample an inhomogeneous Pauli channel used by legacy simulators."""
    p_x, p_y, p_z = (np.asarray(probabilities, dtype=np.float64) for probabilities in channel_probabilities)
    if (
        p_x.shape != p_y.shape
        or p_x.shape != p_z.shape
        or residual[0].shape != p_x.shape
        or residual[1].shape != p_x.shape
    ):
        msg = "Channel probabilities and residual errors must have identical shapes."
        raise ValueError(msg)
    if np.any(~np.isfinite(p_x + p_y + p_z)) or np.any(p_x < 0) or np.any(p_y < 0) or np.any(p_z < 0):
        msg = "Pauli probabilities must be finite and non-negative."
        raise ValueError(msg)
    if np.any(p_x + p_y + p_z > 1.0):
        msg = "Pauli probabilities must sum to at most 1 at every location."
        raise ValueError(msg)
    return _apply_pauli_samples(p_x, p_y, p_z, residual, rng)
