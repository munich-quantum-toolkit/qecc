# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit-level and phenomenological noise-model configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import starmap
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from .channels import IdentityChannel, PauliChannel, QuantumChannel, ReadoutChannel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike

# ----------------------------------------------------------------------------------------------------
#   Circuit-level models
# ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitNoiseModel:
    """Noise channels associated with circuit operation locations."""

    single_qubit_gate: QuantumChannel = field(default_factory=IdentityChannel)
    two_qubit_gate: QuantumChannel = field(default_factory=IdentityChannel)
    reset: QuantumChannel = field(default_factory=IdentityChannel)
    measurement: QuantumChannel = field(default_factory=IdentityChannel)
    idle: QuantumChannel = field(default_factory=IdentityChannel)
    ideal_qubits: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Validate ideal-qubit indices."""
        if any(qubit < 0 for qubit in self.ideal_qubits):
            msg = f"Ideal-qubit indices must be non-negative, got {sorted(self.ideal_qubits)}."
            raise ValueError(msg)


# ----------------------------------------------------------------------------------------------------
#   Phenomenological models
# ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PhenomenologicalNoiseModel:
    """Assign data-Pauli and syndrome channels to phenomenological locations."""

    data: PauliChannel
    data_by_qubit: Mapping[int, PauliChannel] = field(default_factory=dict)
    x_syndrome: ReadoutChannel = field(default_factory=IdentityChannel)
    z_syndrome: ReadoutChannel = field(default_factory=IdentityChannel)

    def __post_init__(self) -> None:
        """Validate and own the per-qubit channel assignments."""
        if any(not isinstance(qubit, int) or qubit < 0 for qubit in self.data_by_qubit):
            msg = f"Data-qubit indices must be non-negative integers, got {list(self.data_by_qubit)}."
            raise ValueError(msg)
        object.__setattr__(self, "data_by_qubit", MappingProxyType(dict(self.data_by_qubit)))

    def data_channel(self, qubit: int) -> PauliChannel:
        """Return a qubit override when present, otherwise the default data channel."""
        if qubit < 0:
            msg = f"Data-qubit indices must be non-negative, got {qubit}."
            raise ValueError(msg)
        return self.data_by_qubit.get(qubit, self.data)

    @classmethod
    def from_pauli_probabilities(
        cls,
        p_x: ArrayLike,
        p_y: ArrayLike,
        p_z: ArrayLike,
        *,
        x_syndrome: ReadoutChannel | None = None,
        z_syndrome: ReadoutChannel | None = None,
    ) -> PhenomenologicalNoiseModel:
        """Construct a model from per-qubit Pauli probability arrays.

        The first qubit's channel becomes the default; only locations with a
        different channel are retained as explicit assignments.
        """
        arrays = tuple(np.asarray(values, dtype=np.float64) for values in (p_x, p_y, p_z))
        if any(values.ndim != 1 for values in arrays):
            msg = "Per-qubit Pauli probabilities must be one-dimensional."
            raise ValueError(msg)
        if arrays[0].shape != arrays[1].shape or arrays[0].shape != arrays[2].shape:
            msg = "Per-qubit Pauli probability arrays must have identical shapes."
            raise ValueError(msg)
        channels = list(starmap(PauliChannel, zip(*arrays, strict=True)))
        default = channels[0] if channels else PauliChannel(0.0, 0.0, 0.0)
        overrides = {qubit: channel for qubit, channel in enumerate(channels) if channel != default}
        return cls(
            data=default,
            data_by_qubit=overrides,
            x_syndrome=IdentityChannel() if x_syndrome is None else x_syndrome,
            z_syndrome=IdentityChannel() if z_syndrome is None else z_syndrome,
        )
