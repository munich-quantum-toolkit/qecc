# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""NumPy backend for sampling phenomenological noise models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .channels import BitFlipChannel, GaussianReadoutChannel, IdentityChannel

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .models import PhenomenologicalNoiseModel


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
        if n_qubits < 0:
            msg = f"n_qubits must be nonnegative, got {n_qubits}."
            raise ValueError(msg)
        if residual is None:
            error_x = np.zeros(n_qubits, dtype=np.int32)
            error_z = np.zeros(n_qubits, dtype=np.int32)
        else:
            if residual[0].shape != (n_qubits,) or residual[1].shape != (n_qubits,):
                msg = "Residual X and Z arrays must both have shape (n_qubits,)."
                raise ValueError(msg)
            error_x = np.array(residual[0], dtype=np.int32, copy=True)
            error_z = np.array(residual[1], dtype=np.int32, copy=True)

        channel = self.model.data
        samples = self.rng.random(n_qubits)
        z_mask = samples < channel.p_z
        x_mask = (channel.p_z <= samples) & (samples < channel.p_z + channel.p_x)
        y_mask = (channel.p_z + channel.p_x <= samples) & (samples < channel.probability)
        error_x ^= np.asarray(x_mask | y_mask, dtype=np.int32)
        error_z ^= np.asarray(z_mask | y_mask, dtype=np.int32)
        return error_x, error_z

    def sample_syndrome(self, perfect_syndrome: NDArray[np.int32]) -> NDArray[np.int32] | NDArray[np.float64]:
        """Apply the configured syndrome channel to a perfect binary syndrome."""
        syndrome = np.asarray(perfect_syndrome, dtype=np.int32)
        if not np.all((syndrome == 0) | (syndrome == 1)):
            msg = "A perfect syndrome must contain only binary values."
            raise ValueError(msg)
        channel = self.model.syndrome
        if isinstance(channel, IdentityChannel):
            return syndrome.copy()
        if isinstance(channel, BitFlipChannel):
            flips = self.rng.random(syndrome.shape) < channel.probability
            return syndrome ^ np.asarray(flips, dtype=np.int32)
        if isinstance(channel, GaussianReadoutChannel):
            signed = np.where(syndrome == 0, 1.0, -1.0)
            return np.asarray(self.rng.normal(signed, channel.sigma), dtype=np.float64)
        raise TypeError(type(channel))


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
        msg = "Pauli probabilities must be finite and nonnegative."
        raise ValueError(msg)
    if np.any(p_x + p_y + p_z > 1.0):
        msg = "Pauli probabilities must sum to at most 1 at every location."
        raise ValueError(msg)
    samples = rng.random(p_x.shape)
    z_mask = samples < p_z
    x_mask = (p_z <= samples) & (samples < p_z + p_x)
    y_mask = (p_z + p_x <= samples) & (samples < p_z + p_x + p_y)
    error_x = np.array(residual[0], dtype=np.int32, copy=True) ^ np.asarray(x_mask | y_mask, dtype=np.int32)
    error_z = np.array(residual[1], dtype=np.int32, copy=True) ^ np.asarray(z_mask | y_mask, dtype=np.int32)
    return error_x, error_z
