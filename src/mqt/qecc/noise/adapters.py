# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Adapters between QECC noise models and simulation backends."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import stim

from .channels import (
    BitFlipChannel,
    DepolarizingChannel,
    GaussianReadoutChannel,
    IdentityChannel,
    PauliChannel,
    QuantumChannel,
    ReadoutChannel,
)
from .scheduling import Schedule, qubits_in_stim_circuit, schedule_stim_circuit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .models import CircuitNoiseModel, PhenomenologicalNoiseModel


def _append_quantum_channel(
    circuit: stim.Circuit,
    channel: QuantumChannel,
    targets: Sequence[int],
    *,
    arity: int = 1,
) -> None:
    """Append the Stim instruction that realizes a quantum channel."""
    if isinstance(channel, IdentityChannel):
        return
    if math.isclose(channel.probability, 0.0):
        return
    if isinstance(channel, DepolarizingChannel):
        circuit.append(f"DEPOLARIZE{arity}", list(targets), channel.probability)
    elif isinstance(channel, BitFlipChannel):
        circuit.append("X_ERROR", list(targets), channel.probability)
    elif isinstance(channel, PauliChannel):
        if arity != 1:
            msg = "PauliChannel describes one-qubit noise and cannot be applied to a two-qubit location."
            raise ValueError(msg)
        # A channel supported on a single axis is exactly the corresponding Pauli error.
        axes = [(channel.p_x, "X_ERROR"), (channel.p_y, "Y_ERROR"), (channel.p_z, "Z_ERROR")]
        supported = [(probability, name) for probability, name in axes if probability > 0.0]
        if len(supported) == 1:
            circuit.append(supported[0][1], list(targets), supported[0][0])
        else:
            circuit.append("PAULI_CHANNEL_1", list(targets), [channel.p_x, channel.p_y, channel.p_z])
    else:
        msg = f"Unsupported quantum channel: {type(channel).__name__}."
        raise TypeError(msg)


def _readout_flip_probability(channel: ReadoutChannel) -> float:
    """Return the probability that a readout channel flips a measurement outcome.

    A Gaussian channel has no Stim counterpart, so it contributes its
    hard-decision error probability.
    """
    if isinstance(channel, IdentityChannel):
        return 0.0
    if isinstance(channel, BitFlipChannel):
        return channel.probability
    if isinstance(channel, GaussianReadoutChannel):
        return channel.bit_error_probability
    msg = f"Unsupported readout channel: {type(channel).__name__}."
    raise TypeError(msg)


# ----------------------------------------------------------------------------------------------------
#   Stim adapters
# ----------------------------------------------------------------------------------------------------


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
        layers = schedule_stim_circuit(circuit, self.schedule)
        return self._apply_scheduled(layers, circuit.num_qubits)

    def _apply_locations(self, circuit: stim.Circuit) -> stim.Circuit:
        noisy = stim.Circuit()
        for operation in circuit:
            if not isinstance(operation, stim.CircuitInstruction):
                msg = "Stim repeat blocks are not supported by the circuit noise adapter; flatten the circuit first."
                raise TypeError(msg)
            name = operation.name
            gate = stim.gate_data(name)
            is_measurement = gate.produces_measurements
            is_reset = gate.is_reset and not is_measurement
            is_single_qubit_gate = gate.is_unitary and gate.is_single_qubit_gate and name != "I"
            is_two_qubit_gate = gate.is_unitary and gate.is_two_qubit_gate
            if not (is_single_qubit_gate or is_two_qubit_gate or is_measurement or is_reset):
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
                if is_measurement and isinstance(self.model.measurement, BitFlipChannel) and not ideal:
                    noisy.append(name, target_list, self.model.measurement.probability)
                    continue
                noisy.append(name, target_list, operation.gate_args_copy())
                if ideal:
                    continue
                if is_single_qubit_gate:
                    _append_quantum_channel(noisy, self.model.single_qubit_gate, integer_qubits, arity=1)
                elif is_two_qubit_gate:
                    _append_quantum_channel(noisy, self.model.two_qubit_gate, integer_qubits, arity=2)
                elif is_reset:
                    _append_quantum_channel(noisy, self.model.reset, integer_qubits, arity=1)
                else:
                    _append_quantum_channel(noisy, self.model.measurement, integer_qubits, arity=1)
        return noisy

    def _apply_scheduled(self, layers: list[stim.Circuit], n_qubits: int) -> stim.Circuit:
        noisy = stim.Circuit()
        uninitialized = set(range(n_qubits))
        reset_alap = self.schedule is not None and self.schedule.reset_timing == "alap"
        for layer in layers:
            active = qubits_in_stim_circuit(layer)
            resets = _reset_qubits(layer)
            idle = set(range(n_qubits)) - active - uninitialized
            noisy_layer = self._apply_locations(layer)
            uninitialized -= active - resets if reset_alap else active
            for qubit in sorted(idle):
                if qubit not in self.model.ideal_qubits:
                    _append_quantum_channel(noisy_layer, self.model.idle, [qubit], arity=1)
            noisy += noisy_layer
        return noisy


def _reset_qubits(circuit: stim.Circuit) -> set[int]:
    qubits: set[int] = set()
    for operation in circuit:
        if isinstance(operation, stim.CircuitInstruction) and stim.gate_data(operation.name).is_reset:
            qubits.update(target.qubit_value for target in operation.targets_copy() if target.qubit_value is not None)
    return qubits


class PhenomenologicalStimAdapter:
    """Emit phenomenological noise into a Stim circuit under construction.

    Phenomenological noise cannot be annotated onto a finished circuit the way
    circuit-level noise is: data noise happens between measurement rounds, and
    readout noise is an argument of the measurement instruction itself. The
    adapter therefore exposes exactly those two pieces to a round builder.
    """

    def __init__(self, model: PhenomenologicalNoiseModel) -> None:
        """Initialize the adapter.

        Args:
            model: Phenomenological noise configuration.
        """
        self.model = model

    def append_data_noise(self, circuit: stim.Circuit, qubits: Sequence[int]) -> None:
        """Append one round of data-qubit noise."""
        if any(qubit < 0 for qubit in qubits):
            msg = f"Qubit indices must be non-negative, got {sorted(qubits)}."
            raise ValueError(msg)
        _append_quantum_channel(circuit, self.model.data, [q for q in qubits if q not in self.model.data_by_qubit])
        for qubit in qubits:
            override = self.model.data_by_qubit.get(qubit)
            if override is not None:
                _append_quantum_channel(circuit, override, [qubit])

    @property
    def x_readout_probability(self) -> float:
        """Flip probability for an X-check measurement."""
        return _readout_flip_probability(self.model.x_syndrome)

    @property
    def z_readout_probability(self) -> float:
        """Flip probability for a Z-check measurement."""
        return _readout_flip_probability(self.model.z_syndrome)
