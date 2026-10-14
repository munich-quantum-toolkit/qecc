# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Scheduling policies used when applying time-dependent noise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import stim


@dataclass(frozen=True)
class ParallelSchedule:
    """Schedule nonconflicting operations in the same time step."""

    reset_timing: Literal["asap", "alap"] = "asap"


@dataclass(frozen=True)
class SequentialSchedule:
    """Schedule every operation target group in a separate time step."""

    reset_timing: Literal["asap", "alap"] = "asap"


Schedule = ParallelSchedule | SequentialSchedule
POSITIONAL_ANNOTATIONS = frozenset({"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"})


def schedule_stim_circuit(circuit: stim.Circuit, policy: Schedule) -> list[stim.Circuit]:
    """Split a Stim circuit into layers according to a scheduling policy.

    Time steps are derived from the operations themselves, so ``TICK`` markers in
    the source circuit carry no information and are dropped.
    """
    groups: list[stim.Circuit] = []
    for operation in circuit:
        if not isinstance(operation, stim.CircuitInstruction):
            msg = "Stim repeat blocks are not supported by scheduled noise; flatten the circuit first."
            raise TypeError(msg)
        if operation.name == "TICK":
            continue
        if operation.name in POSITIONAL_ANNOTATIONS:
            msg = (
                f"Annotation {operation.name} cannot be scheduled because layering reorders operations, "
                "which would invalidate its references. Remove annotations before applying scheduled noise."
            )
            raise ValueError(msg)
        for targets in operation.target_groups():
            group = stim.Circuit()
            group.append(operation.name, list(targets), operation.gate_args_copy())
            groups.append(group)
    if isinstance(policy, SequentialSchedule):
        return groups

    layers: list[stim.Circuit] = []
    remaining = groups
    while remaining:
        layer = stim.Circuit()
        used: set[int] = set()
        deferred: list[stim.Circuit] = []
        blocked: set[int] = set()
        for group in remaining:
            qubits = qubits_in_stim_circuit(group)
            if not used.isdisjoint(qubits) or not blocked.isdisjoint(qubits):
                deferred.append(group)
                blocked.update(qubits)
            else:
                layer += group
                used.update(qubits)
        layers.append(layer)
        remaining = deferred
    return layers


def qubits_in_stim_circuit(circuit: stim.Circuit) -> set[int]:
    """Return the qubits targeted by a Stim circuit."""
    qubits: set[int] = set()
    for operation in circuit:
        if isinstance(operation, stim.CircuitInstruction):
            for target in operation.targets_copy():
                if target.qubit_value is not None:
                    qubits.add(target.qubit_value)
    return qubits
