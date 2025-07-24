# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Classes and functions for constructing noisy circuits."""

from __future__ import annotations

from stim import Circuit

single_qubit_gates = {
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
two_qubit_gates = {
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
measurements = {"MR", "MRX", "MRY", "MRZ"}
resets = {"R", "RX", "RY", "RZ"}


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
            if name in single_qubit_gates:
                for target in op.targets_copy():
                    noisy_circ.append_operation(op.name, target)
                    noisy_circ.append_operation("DEPOLARIZE1", target, self.p_sqg)

            elif name in resets:
                for target in op.targets_copy():
                    noisy_circ.append_operation(op.name, target)
                    noisy_circ.append_operation("DEPOLARIZE1", target, self.p_init)
                
            elif name in two_qubit_gates:
                for ctrl, trgt in op.target_groups():
                    noisy_circ.append_operation(op.name, [ctrl, trgt])
                    noisy_circ.append_operation("DEPOLARIZE2", [ctrl, trgt], self.p_tqg)

            elif name in measurements:
                for target in op.targets_copy():
                    noisy_circ.append_operation("DEPOLARIZE1", target, self.p_meas)
                    noisy_circ.append_operation(op.name, target)

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

    def __init__(self, p_tqg: float, p_sqg: float, p_meas: float, p_init: float, p_idle: float) -> None:
        """Initialize the circuit-level noise model.

        Args:
            p_tqg: Probability of depolarizing noise for two-qubit gates.
            p_sqg: Probability of depolarizing noise for single-qubit gates.
            p_meas: Probability of depolarizing noise for measurements.
            p_init: Probability of depolarizing noise after initialization.
            p_idle: Probability of depolarizing noise for idling qubits.
        """
        self.set_noise_parameters(p_tqg, p_sqg, p_meas, p_init, p_idle)

    def set_noise_parameters(self, p_tqg: float, p_sqg: float, p_meas: float, p_init: float, p_idle: float) -> None:
        """Set the noise parameters for the noise model.

        Args:
            p_tqg: Probability of depolarizing noise for two-qubit gates.
            p_sqg: Probability of depolarizing noise for single-qubit gates.
            p_meas: Probability of depolarizing noise for measurements.
            p_init: Probability of depolarizing noise after initialization.
            p_idle: Probability of depolarizing noise for idling qubits.
        """
        super().set_noise_parameters(p_tqg, p_sqg, p_meas, p_init)
        self.p_idle = p_idle


    def _add_dummy_id_gates(circ: Circuit) -> Circuit:
        """Add dummy identity gates to the circuit to represent idling qubits."""
        new_circ = Circuit()

        # traverse the circuit layer by layer
        layers = []
        done = False
        
        while not done:
            layer = []
            
        
