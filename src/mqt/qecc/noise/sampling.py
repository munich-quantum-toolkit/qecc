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
        p_x, p_y, p_z = self.model.pauli_probabilities(n_qubits)
        if residual is None:
            residual = (np.zeros(n_qubits, dtype=np.int32), np.zeros(n_qubits, dtype=np.int32))
        elif residual[0].shape != (n_qubits,) or residual[1].shape != (n_qubits,):
            msg = "Residual X and Z arrays must both have shape (n_qubits,)."
            raise ValueError(msg)

        # Draw one Pauli per qubit by inverse transform sampling, then XOR it onto the residual.
        # Adapted from
        # https://github.com/quantumgizmos/bp_osd/blob/a179e6e86237f4b9cc2c952103fce919da2777c8/src/bposd/css_decode_sim.py#L430
        # and
        # https://github.com/MikeVasmer/single_shot_3D_HGP/blob/bdfb437b2abcfa514997f26be97a711b878448cb/sim_scripts/single_shot_hgp3d.cpp#L207
        samples = self.rng.random(n_qubits)
        z_mask = samples < p_z
        x_mask = (p_z <= samples) & (samples < p_z + p_x)
        y_mask = (p_z + p_x <= samples) & (samples < p_z + p_x + p_y)
        error_x = np.array(residual[0], dtype=np.int32, copy=True) ^ np.asarray(x_mask | y_mask, dtype=np.int32)
        error_z = np.array(residual[1], dtype=np.int32, copy=True) ^ np.asarray(z_mask | y_mask, dtype=np.int32)
        return error_x, error_z

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
