# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit-level and phenomenological noise-model configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from .channels import IdentityChannel, PauliChannel, QuantumChannel, ReadoutChannel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

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

    def pauli_probabilities(
        self, n_qubits: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Expand the data channels into per-qubit X, Y, and Z probability arrays."""
        self._check_extent(n_qubits)
        p_x = np.full(n_qubits, self.data.p_x, dtype=np.float64)
        p_y = np.full(n_qubits, self.data.p_y, dtype=np.float64)
        p_z = np.full(n_qubits, self.data.p_z, dtype=np.float64)
        for qubit, channel in self.data_by_qubit.items():
            p_x[qubit], p_y[qubit], p_z[qubit] = channel.p_x, channel.p_y, channel.p_z
        return p_x, p_y, p_z

    def x_marginals(self, n_qubits: int) -> NDArray[np.float64]:
        """Per-qubit probability of an X-type error, as used for decoder priors."""
        return self._marginals(n_qubits, "x_marginal")

    def z_marginals(self, n_qubits: int) -> NDArray[np.float64]:
        """Per-qubit probability of a Z-type error, as used for decoder priors."""
        return self._marginals(n_qubits, "z_marginal")

    def _marginals(self, n_qubits: int, attribute: str) -> NDArray[np.float64]:
        self._check_extent(n_qubits)
        marginals = np.full(n_qubits, getattr(self.data, attribute), dtype=np.float64)
        for qubit, channel in self.data_by_qubit.items():
            marginals[qubit] = getattr(channel, attribute)
        return marginals

    def _check_extent(self, n_qubits: int) -> None:
        if n_qubits < 0:
            msg = f"n_qubits must be nonnegative, got {n_qubits}."
            raise ValueError(msg)
        if any(qubit >= n_qubits for qubit in self.data_by_qubit):
            msg = f"Per-qubit data assignments must be below n_qubits={n_qubits}."
            raise ValueError(msg)
