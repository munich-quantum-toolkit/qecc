# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Backend-independent quantum and readout noise channels."""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.special import erfc, erfcinv


def _validate_probability(value: float, name: str = "probability") -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        msg = f"{name} must be finite and between 0 and 1, got {value}."
        raise ValueError(msg)


# ----------------------------------------------------------------------------------------------------
#   Quantum channels
# ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityChannel:
    """Channel that leaves its input unchanged."""


@dataclass(frozen=True)
class BitFlipChannel:
    """Classical or quantum bit-flip channel."""

    probability: float

    def __post_init__(self) -> None:
        """Validate the flip probability."""
        _validate_probability(self.probability)


@dataclass(frozen=True)
class DepolarizingChannel:
    """Uniform non-identity Pauli channel."""

    probability: float

    def __post_init__(self) -> None:
        """Validate the depolarizing probability."""
        _validate_probability(self.probability)


@dataclass(frozen=True)
class PauliChannel:
    """Single-qubit Pauli channel with explicit X, Y, and Z probabilities."""

    p_x: float
    p_y: float
    p_z: float

    def __post_init__(self) -> None:
        """Validate the component and total probabilities."""
        _validate_probability(self.p_x, "p_x")
        _validate_probability(self.p_y, "p_y")
        _validate_probability(self.p_z, "p_z")
        total = self.probability
        if total > 1.0:
            msg = f"Pauli probabilities must sum to at most 1, got {total}."
            raise ValueError(msg)

    @property
    def probability(self) -> float:
        """Total probability of a non-identity error."""
        return self.p_x + self.p_y + self.p_z

    @classmethod
    def from_total_probability(
        cls, probability: float, *, bias: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> PauliChannel:
        """Construct a Pauli channel from a total rate and relative X/Y/Z bias.

        Infinite bias selects the corresponding Pauli error exclusively. At most
        one bias component may be infinite.

        Args:
            probability: Total probability of a non-identity error.
            bias: Nonnegative relative weights for X, Y, and Z errors.

        Returns:
            The corresponding normalized Pauli channel.
        """
        _validate_probability(probability)
        if any(math.isnan(value) or value < 0.0 for value in bias):
            msg = f"Pauli bias must contain nonnegative numbers, got {bias}."
            raise ValueError(msg)
        infinite = [index for index, value in enumerate(bias) if math.isinf(value)]
        if len(infinite) > 1:
            msg = "At most one Pauli bias component may be infinite."
            raise ValueError(msg)
        if infinite:
            probabilities = [0.0, 0.0, 0.0]
            probabilities[infinite[0]] = probability
            return cls(*probabilities)
        total_bias = sum(bias)
        if total_bias <= 0.0 or not math.isfinite(total_bias):
            msg = f"Pauli bias must have a positive finite sum, got {bias}."
            raise ValueError(msg)
        return cls(*(probability * value / total_bias for value in bias))


# ----------------------------------------------------------------------------------------------------
#   Readout channels
# ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GaussianReadoutChannel:
    """Additive Gaussian channel for signed analog measurement outcomes."""

    sigma: float

    def __post_init__(self) -> None:
        """Validate the standard deviation."""
        if not math.isfinite(self.sigma) or self.sigma < 0.0:
            msg = f"sigma must be finite and nonnegative, got {self.sigma}."
            raise ValueError(msg)

    @classmethod
    def from_bit_error_probability(cls, probability: float) -> GaussianReadoutChannel:
        """Construct a Gaussian channel with the given hard-decision error rate."""
        _validate_probability(probability)
        if probability >= 0.5:
            msg = f"Gaussian hard-decision error probability must be below 0.5, got {probability}."
            raise ValueError(msg)
        if math.isclose(probability, 0.0):
            return cls(0.0)
        return cls(float(1.0 / (math.sqrt(2.0) * erfcinv(2.0 * probability))))

    @property
    def bit_error_probability(self) -> float:
        """Hard-decision error probability of the channel."""
        if math.isclose(self.sigma, 0.0):
            return 0.0
        return float(0.5 * erfc(1.0 / math.sqrt(2.0 * self.sigma**2)))


# Channel families (qubit site or classical readout site) used to constrain noise models and backend adapters
QuantumChannel = IdentityChannel | BitFlipChannel | DepolarizingChannel | PauliChannel
ReadoutChannel = IdentityChannel | BitFlipChannel | GaussianReadoutChannel
