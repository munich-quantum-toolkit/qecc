# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Noisy simulation of found circuits with the simulator and noise models of MQT QECC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import stim

from ..noise import CircuitLevelNoiseIdlingParallel, NoiseModel
from ..simulation import VerificationNDFTStatePrepSimulator

if TYPE_CHECKING:
    from ...codes.core.css_code import CSSCode
    from .api import FrontCircuit


def logical_error_rate(
    circ: FrontCircuit | stim.Circuit,
    code: CSSCode,
    p: float,
    *,
    zero_state: bool = True,
    noise: NoiseModel | None = None,
    shots: int = 100000,
    min_errors: int = 500,
    max_shots: int | None = 10**7,
) -> tuple[float, float, int, int]:
    """Estimate the logical error rate of a found circuit by Monte Carlo sampling.

    A shot is accepted if all detectors of the circuit are 0. The simulation is
    :class:`~mqt.qecc.circuit_synthesis.simulation.VerificationNDFTStatePrepSimulator`.

    Args:
        circ: The circuit: a front entry, or a stim circuit whose unmeasured qubits are the code's qubits in the order
            of the code.
        code: The code whose logical zero (``zero_state``) or plus state the circuit prepares.
        p: Physical error rate of the default noise model: depolarizing noise of strength ``p`` after every gate and
            initialization and before every measurement, and of strength ``p / 100`` on idling qubits. Resets are
            scheduled as late as possible, so a qubit idles only after its initialization.
        zero_state: Whether the circuit prepares the logical zero (else the plus) state.
        noise: Another noise model instead of the default.
        shots: Shots to sample at least.
        min_errors: Sampling continues until this many logical errors are seen.
        max_shots: Sampling stops at the first batch (up to 100000 shots) that reaches this many shots, even with
            fewer than ``min_errors`` logical errors. None: no limit, which never ends if no logical error occurs.

    Returns:
        Logical error rate per logical qubit, acceptance rate, number of logical errors, number of shots.
    """
    model = CircuitLevelNoiseIdlingParallel(p, p, p, p, p / 100, resets_alap=True) if noise is None else noise
    sim = VerificationNDFTStatePrepSimulator(circ if isinstance(circ, stim.Circuit) else circ.circuit, code, zero_state)
    return sim.logical_error_rate(model, shots=shots, min_errors=min_errors, max_shots=max_shots)
