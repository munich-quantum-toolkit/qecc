# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Type definitions for exact synthesis framework."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..circuits import CliffordIsometry, CNOTCircuit


class TargetKind(Enum):
    """Type of synthesis target."""

    CLIFFORD_UNITARY = "clifford_unitary"
    STABILIZER_STATE = "stabilizer_state"
    CLIFFORD_ISOMETRY = "clifford_isometry"
    CSS_STATE = "css_state"
    CSS_ISOMETRY = "css_isometry"


class GateFamily(Enum):
    """Gate family to use for synthesis."""

    CLIFFORD = "clifford"
    CSS_CNOT = "css_cnot"


class Objective(Enum):
    """Optimization objective."""

    GATE_COUNT = "gate_count"
    DEPTH = "depth"


class SynthesisStatus(Enum):
    """Status of synthesis attempt."""

    SUCCESS = "success"
    UNSAT = "unsat"
    TIMEOUT = "timeout"
    ERROR = "error"


class SynthesisResult:
    """Result of exact synthesis attempt."""

    def __init__(
        self,
        status: SynthesisStatus,
        circuit: CliffordIsometry | CNOTCircuit | None = None,
        gate_count: int | None = None,
        depth: int | None = None,
        verified: bool = False,
        message: str = "",
    ) -> None:
        """Initialize synthesis result.

        Args:
            status: Status of synthesis attempt.
            circuit: Synthesized circuit (None if failed).
            gate_count: Number of gates in circuit.
            depth: Circuit depth.
            verified: Whether circuit was verified.
            message: Additional information or error message.
        """
        self.status = status
        self.circuit = circuit
        self.gate_count = gate_count
        self.depth = depth
        self.verified = verified
        self.message = message
