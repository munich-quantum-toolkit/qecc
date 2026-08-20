# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Classes and functions for constructing noisy circuits."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mqt.qecc.noise import (
    BitFlipChannel,
    CircuitNoiseModel,
    DepolarizingChannel,
    IdentityChannel,
    ParallelSchedule,
    SequentialSchedule,
    StimCircuitNoiseAdapter,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    from stim import Circuit


class NoiseModel:
    """Class representing a noise model for a quantum circuit."""

    def __init__(self, ideal_qubits: set[int] | None = None) -> None:
        """Initialize the noise model.

        Args:
           ideal_qubits: Set of qubit indices that are ideal (not subject to noise).
        """
        self.ideal_qubits = ideal_qubits or set()

    def _apply_noise(self, circ: Circuit, op: str, targets: list[int], p: float) -> None:
        """Apply noise to the circuit only if the targets are not ideal qubits.

        If any of the targets are in the set of ideal qubits, the noise operation is not applied to those targets.

        Args:
           circ: The circuit to which the noise is applied.
           op: The noise operation (e.g., "DEPOLARIZE1").
           targets: List of qubit indices to apply the noise to.
           p: Probability of the noise operation.
        """
        assert targets, "Targets cannot be empty."

        any_ideal = any(t in self.ideal_qubits for t in targets)
        if not any_ideal:
            circ.append_operation(op, targets, p)

    def apply(self, circ: Circuit) -> Circuit:
        """Apply the noise model to a quantum circuit."""
        raise NotImplementedError


class ComposedNoiseModel(NoiseModel):
    """Noise model composed of multiple other noise models."""

    def __init__(self, models: Iterable[NoiseModel], ideal_qubits: set[int] | None = None) -> None:
        """Initialize the noise model.

        Args:
           models: The noise models to compose.
           ideal_qubits: Set of qubit indices that are ideal (not subject to noise).
        """
        super().__init__(ideal_qubits)
        self.models = list(models)

    def add_model(self, model: NoiseModel) -> None:
        """Add noise model to models."""
        self.models.append(model)

    def apply(self, circ: Circuit) -> Circuit:
        """Apply the noise model to a quantum circuit."""
        noisy_circ = circ.copy()
        for model in self.models:
            noisy_circ = model.apply(noisy_circ)
        return noisy_circ


class CircuitLevelNoise(NoiseModel):
    """Class representing circuit-level noise.

    The following noise model is applied to the circuit:
        - Qubit initialization flips with probability p_init (depolaring noise after initialization).
        - Measurements flip with probability p_meas (depolarizing noise before measuring).
        - Single-qubit gates are subject to depolarizing noise of strength p_sqg.
        - Two-qubit gates are subject to depolarizing noise of strength p_tqg.
    """

    def __init__(
        self, p_tqg: float, p_sqg: float, p_meas: float, p_init: float, ideal_qubits: set[int] | None = None
    ) -> None:
        """Initialize the circuit-level noise model.

        Args:
            p_tqg: Probability of depolarizing noise for two-qubit gates.
            p_sqg: Probability of depolarizing noise for single-qubit gates.
            p_meas: Probability of depolarizing noise for measurements.
            p_init: Probability of depolarizing noise after initialization.
            ideal_qubits: Set of qubit indices that are ideal (not subject to noise).
        """
        super().__init__(ideal_qubits)
        self.p_tqg = p_tqg
        self.p_sqg = p_sqg
        self.p_meas = p_meas
        self.p_init = p_init

    def apply(self, circ: Circuit) -> Circuit:
        """Apply the noise model to a stim circuit."""
        return StimCircuitNoiseAdapter(self._as_location_model()).apply(circ)

    def _as_location_model(self, p_idle: float = 0.0) -> CircuitNoiseModel:
        """Translate legacy probabilities to the location-based model."""
        return CircuitNoiseModel(
            single_qubit_gate=DepolarizingChannel(self.p_sqg),
            two_qubit_gate=DepolarizingChannel(self.p_tqg),
            reset=DepolarizingChannel(self.p_init),
            measurement=BitFlipChannel(self.p_meas),
            idle=IdentityChannel() if math.isclose(p_idle, 0.0) else DepolarizingChannel(p_idle),
            ideal_qubits=frozenset(self.ideal_qubits),
        )


class CircuitLevelNoiseIdlingParallel(CircuitLevelNoise):
    """Class representing circuit-level noise with idling qubits and parallel gates.

    A qubit is considered idle if it is not involved in any gate operation at a given time step.

    The following noise model is applied to the circuit:
        - Qubit initialization flips with probability p_init (depolaring noise after initialization).
        - Measurements flip with probability p_meas (depolarizing noise before measuring).
        - Single-qubit gates are subject to depolarizing noise of strength p_sqg.
        - Two-qubit gates are subject to depolarizing noise of strength p_tqg.
        - Idling qubits are subject to depolarizing noise of strength p_idle.
    """

    def __init__(
        self,
        p_tqg: float,
        p_sqg: float,
        p_meas: float,
        p_init: float,
        p_idle: float,
        resets_alap: bool = False,
        ideal_qubits: set[int] | None = None,
    ) -> None:
        """Initialize the circuit-level noise model.

        Args:
            p_tqg: Probability of depolarizing noise for two-qubit gates.
            p_sqg: Probability of depolarizing noise for single-qubit gates.
            p_meas: Probability of depolarizing noise for measurements.
            p_init: Probability of depolarizing noise after initialization.
            p_idle: Probability of depolarizing noise for idling qubits.
            resets_alap: If True, resets are applied as late as possible, i.e. just before the first gate where the qubit is used (ALAP).
            ideal_qubits: Set of qubit indices that are ideal (not subject to noise).
        """
        super().__init__(p_tqg, p_sqg, p_meas, p_init, ideal_qubits)
        self.resets_alap = resets_alap
        self.p_idle = p_idle

    def apply(self, circ: Circuit) -> Circuit:
        """Apply the noise model to a stim circuit."""
        timing = "alap" if self.resets_alap else "asap"
        return StimCircuitNoiseAdapter(self._as_location_model(self.p_idle), ParallelSchedule(timing)).apply(circ)


class CircuitLevelNoiseIdlingSequential(CircuitLevelNoise):
    """Class representing circuit-level noise with idling qubits and sequential gates.

    A qubit is considered idle if it is not involved in any gate operation at a given time step.
    Since gates are executed sequentially, most qubits are subject to idle noise when a gate is executed.

    The following noise model is applied to the circuit:
        - Qubit initialization flips with probability p_init (depolaring noise after initialization).
        - Measurements flip with probability p_meas (depolarizing noise before measuring).
        - Single-qubit gates are subject to depolarizing noise of strength p_sqg.
        - Two-qubit gates are subject to depolarizing noise of strength p_tqg.
        - Idling qubits are subject to depolarizing noise of strength p_idle.
    """

    def __init__(
        self,
        p_tqg: float,
        p_sqg: float,
        p_meas: float,
        p_init: float,
        p_idle: float,
        resets_alap: bool = False,
        ideal_qubits: set[int] | None = None,
    ) -> None:
        """Initialize the circuit-level noise model.

        Args:
            p_tqg: Probability of depolarizing noise for two-qubit gates.
            p_sqg: Probability of depolarizing noise for single-qubit gates.
            p_meas: Probability of depolarizing noise for measurements.
            p_init: Probability of depolarizing noise after initialization.
            p_idle: Probability of depolarizing noise for idling qubits.
            resets_alap: If True, resets are applied as late as possible, i.e. just before the first gate where the qubit is used (ALAP).
            ideal_qubits: Set of qubit indices that are ideal (not subject to noise).
        """
        super().__init__(p_tqg, p_sqg, p_meas, p_init, ideal_qubits)
        self.resets_alap = resets_alap
        self.p_idle = p_idle

    def apply(self, circ: Circuit) -> Circuit:
        """Apply the noise model to a stim circuit."""
        timing = "alap" if self.resets_alap else "asap"
        return StimCircuitNoiseAdapter(self._as_location_model(self.p_idle), SequentialSchedule(timing)).apply(circ)
