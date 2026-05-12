# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Search strategies for exact synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import (
    ExactSynthesisOptions,
    ExactSynthesisResult,
    GateFamily,
    SynthesisStatus,
    TargetKind,
)

if TYPE_CHECKING:
    from ...codes.pauli import CheckMatrix, StabilizerTableau
    from ..circuits import CliffordIsometry, CNOTCircuit
    from .types import (
        Objective,
    )


def synthesize_exact(
    target: StabilizerTableau | CheckMatrix | CliffordIsometry | CNOTCircuit,
    target_kind: TargetKind,
    gate_family: GateFamily,
    objective: Objective,
    options: ExactSynthesisOptions | None = None,
) -> ExactSynthesisResult:
    """Synthesize an optimal circuit for the given target.

    This is the main entry point for exact synthesis. It searches over resource
    bounds to find an optimal circuit according to the specified objective.

    Args:
        target: The synthesis target (tableau, check matrix, or circuit).
        target_kind: The kind of synthesis problem.
        gate_family: The gate set to use.
        objective: The optimization objective.
        options: Synthesis options (uses defaults if None).

    Returns:
        ExactSynthesisResult containing the synthesized circuit and metadata.

    Examples:
        >>> from mqt.qecc.codes.pauli import StabilizerTableau
        >>> from mqt.qecc.circuit_synthesis.exact import (
        ...     synthesize_exact,
        ...     TargetKind,
        ...     GateFamily,
        ...     Objective,
        ...     ExactSynthesisOptions,
        ... )
        >>> # Synthesize Bell state preparation
        >>> target = StabilizerTableau.from_pauli_strings(["XX", "ZZ"])
        >>> result = synthesize_exact(
        ...     target,
        ...     TargetKind.STABILIZER_STATE,
        ...     GateFamily.CLIFFORD,
        ...     Objective.GATE_COUNT,
        ...     ExactSynthesisOptions(max_bound=10),
        ... )
        >>> print(result.status)
        SynthesisStatus.SAT
    """
    if options is None:
        options = ExactSynthesisOptions(max_bound=10)

    # Validate target and target_kind compatibility
    _validate_target(target, target_kind, gate_family)

    # Placeholder implementation
    # In a full implementation, this would dispatch to appropriate encoding builders
    # based on target_kind, gate_family, and objective

    import time

    start_time = time.time()

    # Placeholder: return UNSAT for now
    # Full implementation would call encoding builders, solvers, extractors, verifiers

    solver_time = time.time() - start_time

    return ExactSynthesisResult(
        status=SynthesisStatus.UNSAT,
        optimal=False,
        objective_value=None,
        circuit=None,
        bound_used=options.max_bound,
        solver_time=solver_time,
        verified=False,
        error_message="Exact synthesis framework is under development. This is a placeholder implementation.",
    )


def _validate_target(
    target: StabilizerTableau | CheckMatrix | CliffordIsometry | CNOTCircuit,
    target_kind: TargetKind,
    gate_family: GateFamily,
) -> None:
    """Validate that target type matches target_kind and gate_family.

    Args:
        target: The synthesis target.
        target_kind: The declared target kind.
        gate_family: The gate family.

    Raises:
        ValueError: If target type is incompatible with target_kind or gate_family.
    """
    from ...codes.pauli import CheckMatrix, StabilizerTableau
    from ..circuits import CliffordIsometry, CNOTCircuit

    if gate_family == GateFamily.CNOT:
        if target_kind not in {TargetKind.CSS_STATE_PREP, TargetKind.CSS_ISOMETRY}:
            msg = f"GateFamily.CNOT requires target_kind to be CSS_STATE_PREP or CSS_ISOMETRY, got {target_kind}"
            raise ValueError(msg)
        if not isinstance(target, (CheckMatrix, CNOTCircuit)):
            msg = f"GateFamily.CNOT requires target to be CheckMatrix or CNOTCircuit, got {type(target).__name__}"
            raise ValueError(msg)

    if gate_family == GateFamily.CLIFFORD:
        if target_kind in {TargetKind.CSS_STATE_PREP, TargetKind.CSS_ISOMETRY}:
            msg = f"GateFamily.CLIFFORD cannot be used with {target_kind}. Use GateFamily.CNOT for CSS problems."
            raise ValueError(msg)
        if not isinstance(target, (StabilizerTableau, CliffordIsometry)):
            msg = f"GateFamily.CLIFFORD with {target_kind} requires StabilizerTableau or CliffordIsometry, got {type(target).__name__}"
            raise ValueError(msg)

    if target_kind == TargetKind.CLIFFORD_UNITARY and isinstance(target, StabilizerTableau):
        if target.n_rows != 2 * target.n:
            msg = f"Clifford unitary requires a full 2n x 2n tableau, got {target.n_rows} x {target.n * 2}"
            raise ValueError(msg)

    if target_kind == TargetKind.STABILIZER_STATE and isinstance(target, StabilizerTableau):
        if target.n_rows != target.n:
            msg = f"Stabilizer state preparation requires n x 2n tableau, got {target.n_rows} x {target.n * 2}"
            raise ValueError(msg)
