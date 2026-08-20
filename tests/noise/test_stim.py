# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Stim circuit-noise adapter."""

from __future__ import annotations

import pytest
import stim

from mqt.qecc.noise import (
    BitFlipChannel,
    CircuitNoiseModel,
    DepolarizingChannel,
    ParallelSchedule,
    PauliChannel,
    StimCircuitNoiseAdapter,
)


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
