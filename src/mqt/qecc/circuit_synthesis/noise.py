# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Classes and functions for constructing noisy circuits."""

from __future__ import annotations

from stim import Circuit


class NoiseModel:
    """Class representing a noise model for a quantum circuit."""

    def apply(self, circ: Circuit) -> Circuit:
        """Apply the noise model to a quantum circuit."""
        raise NotImplementedError


class CircuitLevelNoiseNoIdling(NoiseModel):
    """Class representing circuit-level noise.

    The following noise model is applied to the circuit:
        - Qubit initialization flips with probability 2/3p (depolaring noise after initialization).
        - Measurements flip with probability 2/3p (depolarizing noise before measuring).
        - Single- and two-qubit gates are subject to depolarizing noise of strength p.
    """

    def __init__(self, p: float) -> None:
        """Initialize the circuit-level noise model.

        Args:
            p: The noise parameter for the noise model.
        """
        self.p = p

    def set_noise_parameters(self, p: float) -> None:
        """Set the noise parameters for the noise model.

        Args:
            p: The noise parameter for the noise model.
        """
        self.p = p

    def apply(self, circ: Circuit) -> Circuit:
        """Apply the noise model to a stim circuit."""
        noisy_circ = Circuit()

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

        for op in circ:
            name = op.name
            if name in single_qubit_gates or name in resets:
                for target in op.targets_copy():
                    noisy_circ.append_operation(op.name, target)
                    noisy_circ.append_operation("DEPOLARIZE1", target, self.p)

            elif name in two_qubit_gates:
                for ctrl, trgt in op.target_groups():
                    noisy_circ.append_operation(op.name, [ctrl, trgt])
                    noisy_circ.append_operation("DEPOLARIZE2", [ctrl, trgt], self.p)

            elif name in measurements:
                for target in op.targets_copy():
                    noisy_circ.append_operation("DEPOLARIZE1", target, self.p)
                    noisy_circ.append_operation(op.name, target)

        return noisy_circ
