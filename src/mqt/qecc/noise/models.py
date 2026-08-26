# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit-level and phenomenological noise-model configurations."""

from __future__ import annotations

from dataclasses import dataclass, field

from .channels import DiscreteChannel, IdentityChannel, PauliChannel, SyndromeChannel

# ----------------------------------------------------------------------------------------------------
#   Circuit-level models
# ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitNoiseModel:
    """Noise channels associated with circuit operation locations."""

    single_qubit_gate: DiscreteChannel = field(default_factory=IdentityChannel)
    two_qubit_gate: DiscreteChannel = field(default_factory=IdentityChannel)
    reset: DiscreteChannel = field(default_factory=IdentityChannel)
    measurement: DiscreteChannel = field(default_factory=IdentityChannel)
    idle: DiscreteChannel = field(default_factory=IdentityChannel)
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
    """Data-Pauli and syndrome channels for phenomenological simulation."""

    data: PauliChannel
    syndrome: SyndromeChannel = field(default_factory=IdentityChannel)
