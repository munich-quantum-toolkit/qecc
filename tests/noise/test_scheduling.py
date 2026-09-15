# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for scheduling policies."""

from __future__ import annotations

import pytest
import stim

from mqt.qecc.noise import ParallelSchedule
from mqt.qecc.noise.scheduling import schedule_stim_circuit


@pytest.mark.parametrize("annotation", ["DETECTOR rec[-1]", "OBSERVABLE_INCLUDE(0) rec[-1]", "QUBIT_COORDS(0, 0) 0"])
def test_positional_annotations_are_rejected(annotation: str) -> None:
    """Reject annotations rather than silently invalidating their references."""
    circuit = stim.Circuit(f"M 0\n{annotation}\n")
    with pytest.raises(ValueError, match="cannot be scheduled"):
        schedule_stim_circuit(circuit, ParallelSchedule())


def test_source_ticks_are_dropped() -> None:
    """Derive time steps from the operations, ignoring source ticks."""
    layers = schedule_stim_circuit(stim.Circuit("H 0\nTICK\nH 1\n"), ParallelSchedule())
    assert layers == [stim.Circuit("H 0\nH 1")]
