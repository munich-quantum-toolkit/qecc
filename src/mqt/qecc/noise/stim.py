# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Stim backend adapter for circuit-level noise models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import stim

from .channels import BitFlipChannel, DepolarizingChannel, IdentityChannel, PauliChannel

if TYPE_CHECKING:
    from .models import CircuitNoiseModel

_SINGLE_QUBIT_GATES = {
    "C_XYZ",
    "C_ZYX",
    "H",
    "H_XY",
    "H_XZ",
    "H_YZ",
    "S",
    "S_DAG",
    "SQRT_X",
    "SQRT_X_DAG",
    "SQRT_Y",
    "SQRT_Y_DAG",
    "SQRT_Z",
    "SQRT_Z_DAG",
    "X",
    "Y",
    "Z",
}
_TWO_QUBIT_GATES = {
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
_MEASUREMENTS = {"M", "MR", "MRX", "MRY", "MRZ", "MX", "MY", "MZ"}
_RESETS = {"R", "RX", "RY", "RZ"}


@dataclass(frozen=True)
class ParallelSchedule:
    """Schedule nonconflicting operations in the same time step."""

    reset_timing: Literal["asap", "alap"] = "asap"


@dataclass(frozen=True)
class SequentialSchedule:
    """Schedule every operation target group in a separate time step."""

    reset_timing: Literal["asap", "alap"] = "asap"


Schedule = ParallelSchedule | SequentialSchedule


class StimCircuitNoiseAdapter:
    """Compile a backend-independent circuit noise model into Stim operations."""

    def __init__(self, model: CircuitNoiseModel, schedule: Schedule | None = None) -> None:
        """Initialize the adapter.

        Args:
            model: Location-based circuit noise configuration.
            schedule: Scheduling policy used for idle noise. If omitted, idle
                noise must be the identity channel and the source ordering is kept.
        """
        if schedule is None and not isinstance(model.idle, IdentityChannel):
            msg = "A schedule is required when idle noise is configured."
            raise ValueError(msg)
        self.model = model
        self.schedule = schedule

    def apply(self, circuit: stim.Circuit) -> stim.Circuit:
        """Return a noisy copy of a Stim circuit."""
        if self.schedule is None:
            return self._apply_locations(circuit)
        layers = self._collect_layers(circuit)
        return self._apply_scheduled(layers, circuit.num_qubits)

    def _apply_locations(self, circuit: stim.Circuit) -> stim.Circuit:
        noisy = stim.Circuit()
        for operation in circuit:
            if not isinstance(operation, stim.CircuitInstruction):
                msg = "Stim repeat blocks are not supported by the circuit noise adapter; flatten the circuit first."
                raise TypeError(msg)
            name = operation.name
            if name not in _SINGLE_QUBIT_GATES | _TWO_QUBIT_GATES | _MEASUREMENTS | _RESETS:
                noisy.append(operation)
                continue
            for targets in operation.target_groups():
                target_list = list(targets)
                qubits = [target.qubit_value for target in target_list]
                if any(qubit is None for qubit in qubits):
                    msg = f"Operation {name} contains a non-qubit target."
                    raise ValueError(msg)
                integer_qubits = cast("list[int]", qubits)
                ideal = any(qubit in self.model.ideal_qubits for qubit in integer_qubits)
                if name in _MEASUREMENTS and isinstance(self.model.measurement, BitFlipChannel) and not ideal:
                    noisy.append(name, target_list, self.model.measurement.probability)
                    continue
                noisy.append(name, target_list, operation.gate_args_copy())
                if ideal:
                    continue
                if name in _SINGLE_QUBIT_GATES:
                    self._append_channel(noisy, self.model.single_qubit_gate, integer_qubits, arity=1)
                elif name in _TWO_QUBIT_GATES:
                    self._append_channel(noisy, self.model.two_qubit_gate, integer_qubits, arity=2)
                elif name in _RESETS:
                    self._append_channel(noisy, self.model.reset, integer_qubits, arity=1)
                else:
                    self._append_channel(noisy, self.model.measurement, integer_qubits, arity=1)
        return noisy

    def _apply_scheduled(self, layers: list[stim.Circuit], n_qubits: int) -> stim.Circuit:
        noisy = stim.Circuit()
        uninitialized = set(range(n_qubits))
        reset_alap = self.schedule is not None and self.schedule.reset_timing == "alap"
        for layer in layers:
            active = _qubits_in(layer)
            resets = _reset_qubits(layer)
            idle = set(range(n_qubits)) - active - uninitialized
            noisy_layer = self._apply_locations(layer)
            uninitialized -= active - resets if reset_alap else active
            for qubit in sorted(idle):
                if qubit not in self.model.ideal_qubits:
                    self._append_channel(noisy_layer, self.model.idle, [qubit], arity=1)
            noisy += noisy_layer
        return noisy

    def _collect_layers(self, circuit: stim.Circuit) -> list[stim.Circuit]:
        groups: list[stim.Circuit] = []
        for operation in circuit:
            if not isinstance(operation, stim.CircuitInstruction):
                msg = "Stim repeat blocks are not supported by scheduled noise; flatten the circuit first."
                raise TypeError(msg)
            for targets in operation.target_groups():
                group = stim.Circuit()
                group.append(operation.name, list(targets), operation.gate_args_copy())
                groups.append(group)
        if isinstance(self.schedule, SequentialSchedule):
            return groups
        layers: list[stim.Circuit] = []
        remaining = groups
        while remaining:
            layer = stim.Circuit()
            used: set[int] = set()
            deferred: list[stim.Circuit] = []
            blocked: set[int] = set()
            for group in remaining:
                qubits = _qubits_in(group)
                if not used.isdisjoint(qubits) or not blocked.isdisjoint(qubits):
                    deferred.append(group)
                    blocked.update(qubits)
                else:
                    layer += group
                    used.update(qubits)
            layers.append(layer)
            remaining = deferred
        return layers

    @staticmethod
    def _append_channel(
        circuit: stim.Circuit,
        channel: IdentityChannel | BitFlipChannel | DepolarizingChannel | PauliChannel,
        targets: list[int],
        *,
        arity: int,
    ) -> None:
        if isinstance(channel, IdentityChannel):
            return
        if math.isclose(channel.probability, 0.0):
            return
        if isinstance(channel, DepolarizingChannel):
            circuit.append(f"DEPOLARIZE{arity}", targets, channel.probability)
        elif isinstance(channel, BitFlipChannel):
            circuit.append("X_ERROR", targets, channel.probability)
        elif isinstance(channel, PauliChannel):
            if arity != 1:
                msg = "PauliChannel describes one-qubit noise and cannot be applied to a two-qubit location."
                raise ValueError(msg)
            circuit.append("PAULI_CHANNEL_1", targets, [channel.p_x, channel.p_y, channel.p_z])
        else:  # pragma: no cover
            raise TypeError(type(channel))


def _qubits_in(circuit: stim.Circuit) -> set[int]:
    qubits: set[int] = set()
    for operation in circuit:
        if isinstance(operation, stim.CircuitInstruction):
            for target in operation.targets_copy():
                if target.qubit_value is not None:
                    qubits.add(target.qubit_value)
    return qubits


def _reset_qubits(circuit: stim.Circuit) -> set[int]:
    qubits: set[int] = set()
    for operation in circuit:
        if isinstance(operation, stim.CircuitInstruction) and operation.name in _RESETS:
            qubits.update(target.qubit_value for target in operation.targets_copy() if target.qubit_value is not None)
    return qubits
