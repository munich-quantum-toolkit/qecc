# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Search strategies for exact synthesis."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2

from .encoding_gate_count import encode_clifford_gate_count, encode_css_gate_count
from .extraction import extract_clifford_gate_count_circuit, extract_cnot_gate_count_circuit
from .types import (
    ExactSynthesisOptions,
    ExactSynthesisResult,
    GateFamily,
    Objective,
    SynthesisStatus,
    TargetKind,
)
from .verification import (
    verify_clifford_isometry,
    verify_clifford_unitary,
    verify_css_isometry,
    verify_stabilizer_state,
)

if TYPE_CHECKING:
    from ...codes.pauli import CheckMatrix, StabilizerTableau
    from ..circuits import CliffordIsometry, CNOTCircuit


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

    # Convert target to appropriate representation
    tableau, check_matrix, k, m_x = _prepare_target(target, target_kind, gate_family)

    start_time = time.time()

    # Dispatch to appropriate encoding
    if objective == Objective.GATE_COUNT:
        result = _synthesize_gate_count(
            tableau,
            check_matrix,
            target_kind,
            gate_family,
            k,
            m_x,
            options,
        )
    elif objective == Objective.DEPTH:
        # Placeholder for depth optimization
        solver_time = time.time() - start_time
        return ExactSynthesisResult(
            status=SynthesisStatus.UNSAT,
            optimal=False,
            objective_value=None,
            circuit=None,
            bound_used=options.max_bound,
            solver_time=solver_time,
            verified=False,
            error_message="Depth optimization not yet implemented.",
        )
    elif objective == Objective.DEPTH_THEN_TWO_QUBIT_COUNT:
        # Placeholder for lexicographic optimization
        solver_time = time.time() - start_time
        return ExactSynthesisResult(
            status=SynthesisStatus.UNSAT,
            optimal=False,
            objective_value=None,
            circuit=None,
            bound_used=options.max_bound,
            solver_time=solver_time,
            verified=False,
            error_message="Lexicographic depth-then-gate-count optimization not yet implemented.",
        )
    else:
        msg = f"Unsupported objective: {objective}"
        raise ValueError(msg)

    result.solver_time = time.time() - start_time
    return result


def _synthesize_gate_count(
    tableau: StabilizerTableau | None,
    check_matrix: CheckMatrix | None,
    target_kind: TargetKind,
    gate_family: GateFamily,
    k: int,
    m_x: int,
    options: ExactSynthesisOptions,
) -> ExactSynthesisResult:
    """Synthesize with gate-count objective."""
    if gate_family == GateFamily.CLIFFORD:
        assert tableau is not None
        return _synthesize_clifford_gate_count(tableau, k, options, target_kind)
    assert check_matrix is not None
    return _synthesize_css_gate_count(check_matrix, k, m_x, options, target_kind)


def _synthesize_clifford_gate_count(
    tableau: StabilizerTableau,
    k: int,
    options: ExactSynthesisOptions,
    target_kind: TargetKind,
) -> ExactSynthesisResult:
    """Synthesize Clifford circuit with gate-count objective."""
    import z3

    # Linear search over gate counts
    for bound in range(options.lower_bound, options.max_bound + 1):
        solver, h_vars, s_vars, c_vars, alpha_vars, beta_vars = encode_clifford_gate_count(
            tableau,
            k,
            bound,
            options.allow_qubit_permutation,
        )

        # Set timeout if specified
        if options.timeout_per_bound is not None:
            solver.set("timeout", options.timeout_per_bound * 1000)  # Z3 uses milliseconds

        # Apply additional solver parameters
        for param, value in options.solver_params.items():
            solver.set(param, value)

        check_result = solver.check()

        if check_result == z3.sat:
            # Extract circuit
            model = solver.model()
            n = tableau.n

            circuit = extract_clifford_gate_count_circuit(
                model,
                n,
                bound,
                h_vars,
                s_vars,
                c_vars,
                alpha_vars,
                beta_vars,
            )

            # Verify if requested
            verified = False
            if options.verify_result:
                if target_kind == TargetKind.CLIFFORD_UNITARY:
                    verified = verify_clifford_unitary(circuit, tableau)
                elif target_kind == TargetKind.STABILIZER_STATE:
                    verified = verify_stabilizer_state(circuit, tableau)
                else:  # CLIFFORD_ISOMETRY
                    verified = verify_clifford_isometry(circuit, tableau, k)

            # Count two-qubit gates
            two_qubit_count = sum(1 for slot in range(bound) if model.eval(c_vars[slot], model_completion=True))

            return ExactSynthesisResult(
                status=SynthesisStatus.SAT,
                optimal=True,  # Linear search from lower bound guarantees optimality
                objective_value=bound,
                circuit=circuit,
                bound_used=bound,
                solver_time=0.0,  # Will be filled by caller
                verified=verified,
                two_qubit_gate_count=two_qubit_count,
            )
        if check_result == z3.unknown:
            return ExactSynthesisResult(
                status=SynthesisStatus.UNKNOWN,
                optimal=False,
                objective_value=None,
                circuit=None,
                bound_used=bound,
                solver_time=0.0,
                verified=False,
                error_message=f"Solver returned unknown at bound {bound}: {solver.reason_unknown()}",
            )

    # All bounds exhausted without finding solution
    return ExactSynthesisResult(
        status=SynthesisStatus.UNSAT,
        optimal=True,
        objective_value=None,
        circuit=None,
        bound_used=options.max_bound,
        solver_time=0.0,
        verified=False,
        error_message=f"No solution found within max_bound={options.max_bound}",
    )


def _synthesize_css_gate_count(
    check_matrix: CheckMatrix,
    k: int,
    m_x: int,
    options: ExactSynthesisOptions,
    target_kind: TargetKind,
) -> ExactSynthesisResult:
    """Synthesize CSS CNOT circuit with gate-count objective."""
    import z3

    n = check_matrix.num_qubits()

    # Determine which qubits are initialized
    # For CSS state prep: k=0, all qubits initialized
    # For CSS isometry: k>0, need to determine from final matrix

    # Linear search over gate counts
    for bound in range(options.lower_bound, options.max_bound + 1):
        solver, alpha_vars, beta_vars = encode_css_gate_count(
            check_matrix,
            k,
            m_x,
            bound,
        )

        # Set timeout if specified
        if options.timeout_per_bound is not None:
            solver.set("timeout", options.timeout_per_bound * 1000)

        # Apply additional solver parameters
        for param, value in options.solver_params.items():
            solver.set(param, value)

        check_result = solver.check()

        if check_result == z3.sat:
            # Extract circuit
            model = solver.model()

            # Determine initialized qubits from final matrix structure
            # This is a simplification; full implementation would extract from model
            init_x: list[int] = []
            init_z: list[int] = []
            if target_kind == TargetKind.CSS_STATE_PREP:
                # For state prep, determine from check matrix type
                if check_matrix.is_x_type():
                    init_x = list(range(n))
                else:
                    init_z = list(range(n))
            else:
                # For isometry, would need to extract from terminal condition
                # Placeholder: initialize based on check matrix structure
                pass

            circuit = extract_cnot_gate_count_circuit(
                model,
                n,
                bound,
                alpha_vars,
                beta_vars,
                init_x,
                init_z,
            )

            # Verify if requested
            verified = False
            if options.verify_result:
                if target_kind == TargetKind.CSS_STATE_PREP:
                    verified = verify_css_isometry(circuit, check_matrix, k)
                else:  # CSS_ISOMETRY
                    verified = verify_css_isometry(circuit, check_matrix, k)

            return ExactSynthesisResult(
                status=SynthesisStatus.SAT,
                optimal=True,
                objective_value=bound,
                circuit=circuit,
                bound_used=bound,
                solver_time=0.0,
                verified=verified,
                two_qubit_gate_count=bound,  # All gates are CNOTs
            )
        if check_result == z3.unknown:
            return ExactSynthesisResult(
                status=SynthesisStatus.UNKNOWN,
                optimal=False,
                objective_value=None,
                circuit=None,
                bound_used=bound,
                solver_time=0.0,
                verified=False,
                error_message=f"Solver returned unknown at bound {bound}: {solver.reason_unknown()}",
            )

    # All bounds exhausted
    return ExactSynthesisResult(
        status=SynthesisStatus.UNSAT,
        optimal=True,
        objective_value=None,
        circuit=None,
        bound_used=options.max_bound,
        solver_time=0.0,
        verified=False,
        error_message=f"No solution found within max_bound={options.max_bound}",
    )


def _prepare_target(
    target: StabilizerTableau | CheckMatrix | CliffordIsometry | CNOTCircuit,
    target_kind: TargetKind,
    gate_family: GateFamily,
) -> tuple[StabilizerTableau | None, CheckMatrix | None, int, int]:
    """Prepare target in appropriate representation.

    Returns:
        Tuple of (tableau, check_matrix, k, m_x) where:
        - tableau: For Clifford synthesis, None for CSS
        - check_matrix: For CSS synthesis, None for Clifford
        - k: Number of logical qubits
        - m_x: Number of X-stabilizers (CSS only)
    """
    from ...codes.pauli import CheckMatrix, StabilizerTableau
    from ..circuits import CliffordIsometry, CNOTCircuit

    if gate_family == GateFamily.CLIFFORD:
        # Convert to StabilizerTableau
        if isinstance(target, StabilizerTableau):
            tableau = target
        elif isinstance(target, CliffordIsometry):
            tableau = StabilizerTableau.from_stim_circuit(target.to_stim_circuit(with_resets=False))
        else:
            msg = f"Invalid target type for Clifford synthesis: {type(target)}"
            raise TypeError(msg)

        # Determine k
        if target_kind == TargetKind.CLIFFORD_UNITARY:
            k = tableau.n
        elif target_kind == TargetKind.STABILIZER_STATE:
            k = 0
        else:  # CLIFFORD_ISOMETRY
            # k = (total_rows - n_stabilizers) / 2
            k = (tableau.n_rows - tableau.n) // 2

        return tableau, None, k, 0

    # GateFamily.CNOT
    # Convert to CheckMatrix
    if isinstance(target, CheckMatrix):
        check_matrix = target
    elif isinstance(target, CNOTCircuit):
        code = target.get_code()
        # Use the check matrix with fewer rows
        if code.Hx.shape[0] <= code.Hz.shape[0]:
            check_matrix = CheckMatrix(code.Hx, pauli_type="X")
        else:
            check_matrix = CheckMatrix(code.Hz, pauli_type="Z")
    else:
        msg = f"Invalid target type for CSS synthesis: {type(target)}"
        raise TypeError(msg)

    # Determine k and m_x
    if target_kind == TargetKind.CSS_STATE_PREP:
        k = 0
        m_x = mod2.rank(check_matrix.matrix)
    else:  # CSS_ISOMETRY
        # Would need to extract k from target
        # Placeholder: assume it's encoded in the matrix structure
        m_x = mod2.rank(check_matrix.matrix)
        k = check_matrix.num_qubits() - m_x  # Simplified

    return None, check_matrix, k, m_x


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
