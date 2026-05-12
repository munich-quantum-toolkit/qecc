# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Type definitions for exact synthesis framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..circuits import CliffordIsometry, CNOTCircuit


class TargetKind(Enum):
    """Kind of synthesis target."""

    CLIFFORD_UNITARY = "clifford_unitary"
    STABILIZER_STATE = "stabilizer_state"
    CLIFFORD_ISOMETRY = "clifford_isometry"
    CSS_STATE_PREP = "css_state_prep"
    CSS_ISOMETRY = "css_isometry"


class GateFamily(Enum):
    """Gate set for synthesis."""

    CLIFFORD = "clifford"  # {H, S, CX}
    CNOT = "cnot"  # CNOT-only for CSS problems


class Objective(Enum):
    """Optimization objective."""

    GATE_COUNT = "gate_count"
    DEPTH = "depth"
    DEPTH_THEN_TWO_QUBIT_COUNT = "depth_then_two_qubit_count"


class SearchStrategy(Enum):
    """Strategy for searching over resource bounds."""

    LINEAR = "linear"  # Linear search from lower bound upward
    BINARY = "binary"  # Binary search (requires known upper bound)
    GEOMETRIC = "geometric"  # Geometric growth to find upper bound, then binary search


class SynthesisStatus(Enum):
    """Status of synthesis result."""

    SAT = "sat"  # Solution found
    UNSAT = "unsat"  # Proven no solution exists within bounds
    TIMEOUT = "timeout"  # Solver timed out
    UNKNOWN = "unknown"  # Solver returned unknown status


@dataclass
class ExactSynthesisOptions:
    """Options for exact synthesis.

    Attributes:
        max_bound: Maximum resource bound to search up to (required).
        lower_bound: Known lower bound on optimal resource (default: 0).
        upper_bound: Known upper bound on optimal resource (default: None).
        search_strategy: Strategy for searching over bounds (default: LINEAR).
        enable_symmetry_breaking: Enable symmetry-breaking constraints (default: False).
        timeout_per_bound: Timeout in seconds per bound attempt (default: None).
        solver_params: Additional parameters to pass to Z3 solver (default: {}).
        allow_qubit_permutation: Allow final qubit permutation in unitaries (default: True).
        verify_result: Independently verify synthesized circuit (default: True).
    """

    max_bound: int
    lower_bound: int = 0
    upper_bound: int | None = None
    search_strategy: SearchStrategy = SearchStrategy.LINEAR
    enable_symmetry_breaking: bool = False
    timeout_per_bound: int | None = None
    solver_params: dict = field(default_factory=dict)
    allow_qubit_permutation: bool = True
    verify_result: bool = True

    def __post_init__(self) -> None:
        """Validate options."""
        if self.max_bound < 0:
            msg = "max_bound must be non-negative"
            raise ValueError(msg)
        if self.lower_bound < 0:
            msg = "lower_bound must be non-negative"
            raise ValueError(msg)
        if self.lower_bound > self.max_bound:
            msg = "lower_bound cannot exceed max_bound"
            raise ValueError(msg)
        if self.upper_bound is not None:
            if self.upper_bound < self.lower_bound:
                msg = "upper_bound cannot be less than lower_bound"
                raise ValueError(msg)
            if self.upper_bound > self.max_bound:
                msg = "upper_bound cannot exceed max_bound"
                raise ValueError(msg)
        if self.search_strategy == SearchStrategy.BINARY and self.upper_bound is None:
            msg = "Binary search requires an upper_bound"
            raise ValueError(msg)


@dataclass
class ExactSynthesisResult:
    """Result of exact synthesis.

    Attributes:
        status: Synthesis status (SAT, UNSAT, TIMEOUT, UNKNOWN).
        optimal: Whether the solution is proven optimal.
        objective_value: Value of the objective (gate count or depth), or None if no solution.
        circuit: Synthesized circuit, or None if no solution.
        bound_used: The resource bound at which synthesis succeeded or failed.
        solver_time: Total time spent in solver (seconds).
        verified: Whether the circuit was independently verified.
        error_message: Error or status message if applicable.
        two_qubit_gate_count: Number of two-qubit gates (for lexicographic objectives).
    """

    status: SynthesisStatus
    optimal: bool
    objective_value: int | None
    circuit: CliffordIsometry | CNOTCircuit | None
    bound_used: int
    solver_time: float
    verified: bool
    error_message: str | None = None
    two_qubit_gate_count: int | None = None

    def __str__(self) -> str:
        """Return human-readable summary."""
        lines = [f"Exact Synthesis Result: {self.status.value.upper()}"]
        if self.optimal:
            lines.append("  Optimal: Yes")
        if self.objective_value is not None:
            lines.append(f"  Objective value: {self.objective_value}")
        if self.two_qubit_gate_count is not None:
            lines.append(f"  Two-qubit gates: {self.two_qubit_gate_count}")
        lines.extend((
            f"  Bound used: {self.bound_used}",
            f"  Solver time: {self.solver_time:.3f}s",
            f"  Verified: {self.verified}",
        ))
        if self.error_message:
            lines.append(f"  Message: {self.error_message}")
        return "\n".join(lines)
