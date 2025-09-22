# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Classes and functions for constructing noisy circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

from stim import Circuit

from .circuit_utils import collect_circuit_layers
from .definitions import STIM_MEASUREMENTS, STIM_RESETS, STIM_SQGS, STIM_TQGS

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable


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
        noisy_circ = Circuit()

        for op in circ:
            name = op.name
            if name in STIM_SQGS:
                for targets in op.target_groups():
                    noisy_circ.append_operation(op.name, targets)
                    self._apply_noise(noisy_circ, "DEPOLARIZE1", [trgt.qubit_value for trgt in targets], self.p_sqg)

            elif name in STIM_RESETS:
                for targets in op.target_groups():
                    noisy_circ.append_operation(op.name, targets)
                    self._apply_noise(noisy_circ, "DEPOLARIZE1", [trgt.qubit_value for trgt in targets], self.p_init)

            elif name in STIM_TQGS:
                for targets in (
                    op.target_groups()
                ):  # errors might propagate so we have to apply noise to every target group individually
                    noisy_circ.append_operation(op.name, targets)
                    self._apply_noise(noisy_circ, "DEPOLARIZE2", [trgt.qubit_value for trgt in targets], self.p_tqg)

            elif name in STIM_MEASUREMENTS:
                for targets in op.target_groups():
                    if not any(t in self.ideal_qubits for t in [trgt.qubit_value for trgt in targets]):
                        noisy_circ.append_operation(op.name, targets, self.p_meas)
                    else:
                        noisy_circ.append_operation(op.name, targets)
            else:
                noisy_circ.append_operation(op)
        return noisy_circ


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
        layers = collect_circuit_layers(circ)

        if self.resets_alap:
            return _add_idling_noise_to_layers_alap(layers, self, self.p_idle, circ.num_qubits)
        return _add_idling_noise_to_layers_asap(layers, self, self.p_idle, circ.num_qubits)


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
        layers = []

        for op in circ:
            for grp in op.target_groups():
                layer_circ = Circuit()
                layer_circ.append(op.name, grp)
                layers.append(layer_circ)

        if self.resets_alap:
            return _add_idling_noise_to_layers_alap(layers, self, self.p_idle, circ.num_qubits)
        return _add_idling_noise_to_layers_asap(layers, self, self.p_idle, circ.num_qubits)


def _add_idling_noise_to_layers_alap(
    layers: list[Circuit], noise: CircuitLevelNoise, p_idle: float, n_qubits: int
) -> Circuit:
    noisy_circ = Circuit()

    initialized_qubits: set[int] = set()
    uninitialized_qubits = set(range(n_qubits))

    for layer in layers:
        idling = _get_idle_qubits_layer(layer, n_qubits) - uninitialized_qubits
        non_idling = _get_non_idle_qubits_layer(layer)
        resets = _get_reset_qubits_layer(layer)

        non_idling_non_resets = non_idling - resets
        noisy_layer = CircuitLevelNoise.apply(noise, layer)  # apply regular noise

        uninitialized_qubits -= non_idling_non_resets
        initialized_qubits = initialized_qubits.union(non_idling_non_resets)

        for q in idling:
            noisy_layer.append_operation("DEPOLARIZE1", q, p_idle)

        noisy_circ += noisy_layer
    return noisy_circ


def _add_idling_noise_to_layers_asap(
    layers: list[Circuit], noise: CircuitLevelNoise, p_idle: float, n_qubits: int
) -> Circuit:
    noisy_circ = Circuit()

    uninitialized_qubits = set(range(n_qubits))

    for layer in layers:
        idling = _get_idle_qubits_layer(layer, n_qubits) - uninitialized_qubits
        non_idling = _get_non_idle_qubits_layer(layer)

        noisy_layer = CircuitLevelNoise.apply(noise, layer)  # apply regular noise

        uninitialized_qubits -= non_idling

        for q in idling:
            noisy_layer.append_operation("DEPOLARIZE1", q, p_idle)

        noisy_circ += noisy_layer
    return noisy_circ


def _get_reset_qubits_layer(circ: Circuit) -> set[int]:
    """Get the list of reset qubits in the current layer of the circuit."""
    resets = set()
    for instr in circ:
        if instr.name in STIM_RESETS:
            resets.update([q.qubit_value for q in instr.targets_copy()])
    return resets


def _get_non_idle_qubits_layer(circ: Circuit) -> set[int]:
    qubits = set()
    for instr in circ:
        qubits.update([q.qubit_value for q in instr.targets_copy()])
    return qubits


def _get_idle_qubits_layer(circ: Circuit, n_qubits: int) -> set[int]:
    """Get the list of idle qubits in the current layer of the circuit."""
    non_idle = _get_non_idle_qubits_layer(circ)
    return set(range(n_qubits)) - non_idle
