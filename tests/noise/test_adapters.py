# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for noise-model backend adapters."""

from __future__ import annotations

import pytest
import stim

from mqt.qecc.noise import (
    BitFlipChannel,
    CircuitNoiseModel,
    DepolarizingChannel,
    GaussianReadoutChannel,
    ParallelSchedule,
    PauliChannel,
    PhenomenologicalNoiseModel,
    PhenomenologicalStimAdapter,
    SequentialSchedule,
    StimCircuitNoiseAdapter,
)
from mqt.qecc.noise.scheduling import schedule_stim_circuit


def test_location_channels_compile_to_stim() -> None:
    """Compile gate, reset, and measurement channels at their locations."""
    model = CircuitNoiseModel(
        single_qubit_gate=PauliChannel(0.1, 0.2, 0.3),
        two_qubit_gate=DepolarizingChannel(0.4),
        reset=BitFlipChannel(0.5),
        measurement=BitFlipChannel(0.6),
    )
    circuit = stim.Circuit("R 0\nH 0\nCX 0 1\nM 0 1")
    expected = stim.Circuit(
        "R 0\nX_ERROR(0.5) 0\nH 0\nPAULI_CHANNEL_1(0.1,0.2,0.3) 0\nCX 0 1\nDEPOLARIZE2(0.4) 0 1\nM(0.6) 0 1"
    )
    assert StimCircuitNoiseAdapter(model).apply(circuit) == expected


def test_plain_measurements_are_recognized() -> None:
    """Handle the full family of non-reset measurement operations."""
    model = CircuitNoiseModel(measurement=BitFlipChannel(0.1))
    assert StimCircuitNoiseAdapter(model).apply(stim.Circuit("M 0\nMX 1")) == stim.Circuit("M(0.1) 0\nMX(0.1) 1")


def test_stim_metadata_classifies_gates_not_listed_by_qecc() -> None:
    """Use Stim metadata as the source of truth for supported unitary gates."""
    model = CircuitNoiseModel(single_qubit_gate=DepolarizingChannel(0.1))
    assert StimCircuitNoiseAdapter(model).apply(stim.Circuit("SQRT_X 0")) == stim.Circuit(
        "SQRT_X 0\nDEPOLARIZE1(0.1) 0"
    )


def test_identity_gate_is_not_a_gate_location() -> None:
    """Treat the identity as a timing marker, so only idle noise applies to it."""
    model = CircuitNoiseModel(single_qubit_gate=DepolarizingChannel(0.1))
    assert StimCircuitNoiseAdapter(model).apply(stim.Circuit("I 0")) == stim.Circuit("I 0")


def test_idle_noise_requires_schedule() -> None:
    """Require an explicit interpretation of circuit time for idle noise."""
    model = CircuitNoiseModel(idle=DepolarizingChannel(0.1))
    with pytest.raises(ValueError, match="schedule is required"):
        StimCircuitNoiseAdapter(model)


def test_parallel_idle_noise() -> None:
    """Apply idle noise only after qubits have been initialized."""
    model = CircuitNoiseModel(idle=DepolarizingChannel(0.1))
    circuit = stim.Circuit("R 0 1\nH 0\nH 0")
    expected = stim.Circuit("R 0 1\nH 0\nDEPOLARIZE1(0.1) 1\nH 0\nDEPOLARIZE1(0.1) 1")
    assert StimCircuitNoiseAdapter(model, ParallelSchedule()).apply(circuit) == expected


def test_single_qubit_pauli_rejected_at_two_qubit_location() -> None:
    """Fail explicitly instead of silently changing a channel's semantics."""
    model = CircuitNoiseModel(two_qubit_gate=PauliChannel(0.1, 0.0, 0.0))
    with pytest.raises(ValueError, match="one-qubit noise"):
        StimCircuitNoiseAdapter(model).apply(stim.Circuit("CX 0 1"))


def test_adapter_preserves_unrecognized_operations_and_ideal_qubits() -> None:
    """Pass metadata through and omit noise from ideal qubits."""
    model = CircuitNoiseModel(single_qubit_gate=DepolarizingChannel(0.1), ideal_qubits=frozenset({0}))
    circuit = stim.Circuit("TICK\nH 0")
    assert StimCircuitNoiseAdapter(model).apply(circuit) == circuit


def test_non_bit_flip_measurement_channel_is_appended() -> None:
    """Append quantum measurement noise when it is not a readout flip."""
    model = CircuitNoiseModel(measurement=DepolarizingChannel(0.1))
    assert StimCircuitNoiseAdapter(model).apply(stim.Circuit("M 0")) == stim.Circuit("M 0\nDEPOLARIZE1(0.1) 0")


def test_zero_probability_channel_is_omitted() -> None:
    """Avoid emitting redundant zero-probability Stim operations."""
    circuit = stim.Circuit("H 0")
    model = CircuitNoiseModel(single_qubit_gate=PauliChannel(0.0, 0.0, 0.0))
    assert StimCircuitNoiseAdapter(model).apply(circuit) == circuit


def test_repeat_blocks_require_flattening() -> None:
    """Reject repeat blocks in both unscheduled and scheduled paths."""
    circuit = stim.Circuit("REPEAT 2 {\nH 0\n}")
    with pytest.raises(TypeError, match="flatten"):
        StimCircuitNoiseAdapter(CircuitNoiseModel()).apply(circuit)
    with pytest.raises(TypeError, match="flatten"):
        schedule_stim_circuit(circuit, SequentialSchedule())


def test_sequential_schedule_separates_target_groups() -> None:
    """Place every target group in its own layer."""
    assert schedule_stim_circuit(stim.Circuit("H 0 1"), SequentialSchedule()) == [
        stim.Circuit("H 0"),
        stim.Circuit("H 1"),
    ]


def test_rec_controlled_gate_is_rejected_as_quantum_gate_noise_location() -> None:
    """Reject recognized gates containing classical record targets."""
    with pytest.raises(ValueError, match="non-qubit target"):
        StimCircuitNoiseAdapter(CircuitNoiseModel()).apply(stim.Circuit("CX rec[-1] 0"))


def test_phenomenological_adapter_matches_scalar_noise() -> None:
    """Emit data noise and expose the readout flip probability for a round."""
    model = PhenomenologicalNoiseModel(data=PauliChannel(0.02, 0.0, 0.0), z_syndrome=BitFlipChannel(0.05))
    adapter = PhenomenologicalStimAdapter(model)
    circuit = stim.Circuit()
    adapter.append_data_noise(circuit, [0, 1, 2])
    assert circuit == stim.Circuit("X_ERROR(0.02) 0 1 2")
    assert adapter.z_readout_probability == pytest.approx(0.05)
    assert adapter.x_readout_probability == pytest.approx(0.0)  # identity by default


def test_phenomenological_adapter_honors_per_qubit_overrides() -> None:
    """Give overridden qubits their own instruction."""
    model = PhenomenologicalNoiseModel(
        data=PauliChannel(0.02, 0.0, 0.0), data_by_qubit={1: PauliChannel(0.0, 0.0, 0.3)}
    )
    circuit = stim.Circuit()
    PhenomenologicalStimAdapter(model).append_data_noise(circuit, [0, 1, 2])
    assert circuit == stim.Circuit("X_ERROR(0.02) 0 2\nZ_ERROR(0.3) 1")


def test_gaussian_readout_uses_hard_decision_probability() -> None:
    """Stim has no analog readout, so the hard-decision rate is used."""
    channel = GaussianReadoutChannel.from_bit_error_probability(0.1)
    model = PhenomenologicalNoiseModel(data=PauliChannel(0.0, 0.0, 0.0), z_syndrome=channel)
    assert PhenomenologicalStimAdapter(model).z_readout_probability == pytest.approx(0.1)
