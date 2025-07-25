# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Classes and functions for constructing noisy circuits."""

from __future__ import annotations

from stim import Circuit

from .circuit_utils import collect_circuit_layers

STIM_SQGS = {
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "S_DAG",
    "SQRT_X",
    "C_XYZ",
    "C_ZYX",
    "H_XY",
    "H_XZ",
    "H_YZ",
    "SQRT_X_DAG",
    "SQRT_Y",
    "SQRT_Y_DAG",
    "SQRT_Z",
    "SQRT_Z_DAG",
}
STIM_TQGS = {
    "CNOT",
    "CX",
    "CXSWAP",
    "CY",
    "CZ",
    "CZSWAP",
    "ISWAP",
    "ISWAP_DAG",
    "SQRT_XX",
    "SQRT_XX_DAG",
    "SQRT_YY",
    "SQRT_YY_DAG",
    "SQRT_ZZ",
    "SQRT_ZZ_DAG",
    "SWAP",
    "SWAPCX",
    "SWAPCZ",
    "XCX",
    "XCY",
    "XCZ",
    "YCX",
    "YCY",
    "YCZ",
    "ZCX",
    "ZCY",
    "ZCZ",
}
STIM_MEASUREMENTS = {"MR", "MRX", "MRY", "MRZ"}
STIM_RESETS = {"R", "RX", "RY", "RZ"}


class NoiseModel:
    """Class representing a noise model for a quantum circuit."""

    def apply(self, circ: Circuit) -> Circuit:
        """Apply the noise model to a quantum circuit."""
        raise NotImplementedError


class CircuitLevelNoise(NoiseModel):
    """Class representing circuit-level noise.

    The following noise model is applied to the circuit:
        - Qubit initialization flips with probability p_init (depolaring noise after initialization).
        - Measurements flip with probability p_meas (depolarizing noise before measuring).
        - Single-qubit gates are subject to depolarizing noise of strength p_sqg.
        - Two-qubit gates are subject to depolarizing noise of strength p_tqg.
    """

    def __init__(self, p_tqg: float, p_sqg: float, p_meas: float, p_init: float) -> None:
        """Initialize the circuit-level noise model.

        Args:
            p_tqg: Probability of depolarizing noise for two-qubit gates.
            p_sqg: Probability of depolarizing noise for single-qubit gates.
            p_meas: Probability of depolarizing noise for measurements.
            p_init: Probability of depolarizing noise after initialization.
        """
        self.set_noise_parameters(p_tqg, p_sqg, p_meas, p_init)

    def set_noise_parameters(self, p_tqg: float, p_sqg: float, p_meas: float, p_init: float) -> None:
        """Set the noise parameters for the noise model.

        Args:
            p_tqg: Probability of depolarizing noise for two-qubit gates.
            p_sqg: Probability of depolarizing noise for single-qubit gates.
            p_meas: Probability of depolarizing noise for measurements.
            p_init: Probability of depolarizing noise after initialization.
        """
        self.p_tqg = p_tqg
        self.p_sqg = p_sqg
        self.p_meas = p_meas
        self.p_init = p_init

    def apply(self, circ: Circuit) -> Circuit:
        """Apply the noise model to a stim circuit."""
        noisy_circ = Circuit()

        for op in circ:
            name = op.name
            targets = op.targets_copy()
            if name in STIM_SQGS:
                noisy_circ.append_operation(op.name, targets)
                noisy_circ.append_operation("DEPOLARIZE1", targets, self.p_sqg)

            elif name in STIM_RESETS:
                noisy_circ.append_operation(op.name, targets)
                noisy_circ.append_operation("DEPOLARIZE1", targets, self.p_init)

            elif name in STIM_TQGS:
                for grp in (
                    op.target_groups()
                ):  # errors might propagate so we have to apply noise to every target group individually
                    noisy_circ.append_operation(op.name, grp)
                    noisy_circ.append_operation("DEPOLARIZE2", grp, self.p_tqg)

            elif name in STIM_MEASUREMENTS:
                noisy_circ.append_operation(op.name, targets, self.p_meas)

        return noisy_circ


class CircuitLevelNoiseIdlingParallel(NoiseModel):
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
        self, p_tqg: float, p_sqg: float, p_meas: float, p_init: float, p_idle: float, resets_alap: bool = False
    ) -> None:
        """Initialize the circuit-level noise model.

        Args:
            p_tqg: Probability of depolarizing noise for two-qubit gates.
            p_sqg: Probability of depolarizing noise for single-qubit gates.
            p_meas: Probability of depolarizing noise for measurements.
            p_init: Probability of depolarizing noise after initialization.
            p_idle: Probability of depolarizing noise for idling qubits.
            resets_alap: If True, resets are applied as late as possible, i.e. just before the first gate where the qubit is used (ALAP).
        """
        self.standard_noise = CircuitLevelNoise(p_tqg, p_sqg, p_meas, p_init)
        self.resets_alap = resets_alap
        self.p_idle = p_idle

    def set_noise_parameters(self, p_tqg: float, p_sqg: float, p_meas: float, p_init: float, p_idle: float) -> None:
        """Set the noise parameters for the noise model.

        Args:
            p_tqg: Probability of depolarizing noise for two-qubit gates.
            p_sqg: Probability of depolarizing noise for single-qubit gates.
            p_meas: Probability of depolarizing noise for measurements.
            p_init: Probability of depolarizing noise after initialization.
            p_idle: Probability of depolarizing noise for idling qubits.
        """
        self.standard_noise.set_noise_parameters(p_tqg, p_sqg, p_meas, p_init)
        self.p_idle = p_idle

    def apply(self, circ: Circuit) -> Circuit:
        """Apply the noise model to a stim circuit."""
        layers = collect_circuit_layers(circ)

        if self.resets_alap:
            return _add_idling_noise_to_layers_alap(layers, self.standard_noise, self.p_idle, circ.num_qubits)
        return _add_idling_noise_to_layers_asap(layers, self.standard_noise, self.p_idle, circ.num_qubits)


class CircuitLevelNoiseIdlingSequential(NoiseModel):
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
        self, p_tqg: float, p_sqg: float, p_meas: float, p_init: float, p_idle: float, resets_alap: bool = False
    ) -> None:
        """Initialize the circuit-level noise model.

        Args:
            p_tqg: Probability of depolarizing noise for two-qubit gates.
            p_sqg: Probability of depolarizing noise for single-qubit gates.
            p_meas: Probability of depolarizing noise for measurements.
            p_init: Probability of depolarizing noise after initialization.
            p_idle: Probability of depolarizing noise for idling qubits.
            resets_alap: If True, resets are applied as late as possible, i.e. just before the first gate where the qubit is used (ALAP).
        """
        self.standard_noise = CircuitLevelNoise(p_tqg, p_sqg, p_meas, p_init)
        self.resets_alap = resets_alap
        self.p_idle = p_idle

    def set_noise_parameters(self, p_tqg: float, p_sqg: float, p_meas: float, p_init: float, p_idle: float) -> None:
        """Set the noise parameters for the noise model.

        Args:
            p_tqg: Probability of depolarizing noise for two-qubit gates.
            p_sqg: Probability of depolarizing noise for single-qubit gates.
            p_meas: Probability of depolarizing noise for measurements.
            p_init: Probability of depolarizing noise after initialization.
            p_idle: Probability of depolarizing noise for idling qubits.
        """
        self.standard_noise.set_noise_parameters(p_tqg, p_sqg, p_meas, p_init)
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
            return _add_idling_noise_to_layers_alap(layers, self.standard_noise, self.p_idle, circ.num_qubits)
        return _add_idling_noise_to_layers_asap(layers, self.standard_noise, self.p_idle, circ.num_qubits)


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
        noisy_layer = noise.apply(layer)  # apply regular noise

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

        noisy_layer = noise.apply(layer)  # apply regular noise

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
