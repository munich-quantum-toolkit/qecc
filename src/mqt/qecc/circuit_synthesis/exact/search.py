# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Search strategies for exact synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .encoding_gate_count import encode_clifford_gate_count, encode_css_gate_count
from .extraction import extract_clifford_gate_count_circuit, extract_cnot_gate_count_circuit
from .types import GateFamily, Objective, SynthesisResult, SynthesisStatus, TargetKind
from .verification import (
    verify_clifford_isometry,
    verify_clifford_unitary,
    verify_css_isometry,
    verify_css_state,
    verify_stabilizer_state,
)

if TYPE_CHECKING:
    import z3

    from ...codes.pauli import CheckMatrix, StabilizerTableau


def synthesize_exact(
    target: StabilizerTableau | CheckMatrix,
    target_kind: TargetKind,
    gate_family: GateFamily,
    objective: Objective,
    lower_bound: int = 0,
    upper_bound: int = 10,
    k: int | None = None,
    verify: bool = True,
    allow_qubit_permutation: bool = True,
) -> SynthesisResult:
    """Synthesize optimal circuit for given target using exact methods.

    Args:
        target: Target tableau or check matrix.
        target_kind: Kind of synthesis problem.
        gate_family: Gate family to use.
        objective: Optimization objective.
        lower_bound: Lower bound on resource count.
        upper_bound: Upper bound on resource count.
        k: Number of logical qubits (required for isometry).
        verify: Whether to verify synthesized circuit.
        allow_qubit_permutation: Allow qubit permutation in unitaries.

    Returns:
        SynthesisResult with circuit and metadata.

    Raises:
        ValueError: If parameters are invalid.
        NotImplementedError: If objective not yet implemented.
    """
    from ...codes.pauli import CheckMatrix, StabilizerTableau

    if lower_bound < 0 or upper_bound < lower_bound:
        msg = f"Invalid bounds: lower_bound={lower_bound}, upper_bound={upper_bound}"
        raise ValueError(msg)

    if target_kind in {TargetKind.CLIFFORD_ISOMETRY, TargetKind.CSS_ISOMETRY}:
        if k is None:
            msg = "k must be provided for isometry synthesis"
            raise ValueError(msg)
        if k < 0:
            msg = f"k must be non-negative, got {k}"
            raise ValueError(msg)

    if objective == Objective.DEPTH:
        msg = "Depth optimization not yet implemented"
        raise NotImplementedError(msg)

    if gate_family == GateFamily.CLIFFORD:
        if not isinstance(target, StabilizerTableau):
            msg = f"CLIFFORD gate family requires StabilizerTableau, got {type(target).__name__}"
            raise ValueError(msg)
        if target_kind in {TargetKind.CSS_STATE, TargetKind.CSS_ISOMETRY}:
            msg = f"CLIFFORD gate family incompatible with {target_kind.value}"
            raise ValueError(msg)
    elif gate_family == GateFamily.CSS_CNOT:
        if not isinstance(target, CheckMatrix):
            msg = f"CSS_CNOT gate family requires CheckMatrix, got {type(target).__name__}"
            raise ValueError(msg)
        if target_kind not in {TargetKind.CSS_STATE, TargetKind.CSS_ISOMETRY}:
            msg = f"CSS_CNOT gate family requires CSS_STATE or CSS_ISOMETRY, got {target_kind.value}"
            raise ValueError(msg)

    if gate_family == GateFamily.CLIFFORD:
        return _synthesize_clifford(
            target,
            target_kind,
            objective,
            lower_bound,
            upper_bound,
            k,
            verify,
            allow_qubit_permutation,
        )
    return _synthesize_css(
        target,
        target_kind,
        objective,
        lower_bound,
        upper_bound,
        k,
        verify,
    )


def _synthesize_clifford(
    target: StabilizerTableau,
    target_kind: TargetKind,
    objective: Objective,
    lower_bound: int,
    upper_bound: int,
    k: int | None,
    verify: bool,
    allow_qubit_permutation: bool,
) -> SynthesisResult:
    """Synthesize Clifford circuit."""
    import z3

    if k is None:
        if target_kind == TargetKind.CLIFFORD_UNITARY:
            k = target.n
        elif target_kind == TargetKind.STABILIZER_STATE:
            k = 0
        else:
            k = (target.n_rows - target.n) // 2

    for bound in range(lower_bound, upper_bound + 1):
        solver, h_vars, s_vars, c_vars, alpha_vars, beta_vars = encode_clifford_gate_count(
            target,
            k,
            bound,
            allow_qubit_permutation,
        )

        result = solver.check()

        if result == z3.sat:
            model = solver.model()
            n = target.n

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

            actual_count = sum(
                1
                for slot in range(bound)
                if model.eval(z3.Or(h_vars[slot], s_vars[slot], c_vars[slot]), model_completion=True)
            )

            verified = False
            if verify:
                if target_kind == TargetKind.CLIFFORD_UNITARY:
                    verified = verify_clifford_unitary(circuit, target)
                elif target_kind == TargetKind.STABILIZER_STATE:
                    verified = verify_stabilizer_state(circuit, target)
                else:
                    verified = verify_clifford_isometry(circuit, target, k)

            return SynthesisResult(
                status=SynthesisStatus.SUCCESS,
                circuit=circuit,
                gate_count=actual_count,
                verified=verified,
                message=f"Found solution with {actual_count} gates",
            )

        if result == z3.unknown:
            return SynthesisResult(
                status=SynthesisStatus.ERROR,
                message=f"Solver returned unknown at bound {bound}: {solver.reason_unknown()}",
            )

    return SynthesisResult(
        status=SynthesisStatus.UNSAT,
        message=f"No solution found within bounds [{lower_bound}, {upper_bound}]",
    )


def _row_echelon_pivot_cols(matrix: np.ndarray) -> list[int]:
    """Compute row echelon form and return pivot column indices.

    Args:
        matrix: Binary matrix (m x n) with dtype np.int8.

    Returns:
        List of column indices that contain pivots in row echelon form.
    """
    mat = matrix.copy()
    m, n = mat.shape
    pivot_cols = []
    current_row = 0

    for col in range(n):
        pivot_found = False
        for row in range(current_row, m):
            if mat[row, col] == 1:
                if row != current_row:
                    mat[[current_row, row]] = mat[[row, current_row]]
                pivot_found = True
                break

        if not pivot_found:
            continue

        pivot_cols.append(col)

        for row in range(m):
            if row != current_row and mat[row, col] == 1:
                mat[row] ^= mat[current_row]

        current_row += 1
        if current_row >= m:
            break

    return pivot_cols


def _determine_css_initializations(
    model: z3.ModelRef,
    n: int,
    num_rows: int,
    k: int,
    matrix_vars: np.ndarray,
    is_x_type: bool,
) -> tuple[list[int], list[int]]:
    """Determine which qubits to initialize based on terminal tableau.

    Args:
        model: Z3 model from satisfiable formula.
        n: Number of qubits.
        num_rows: Number of rows in check matrix.
        k: Number of logical qubits.
        matrix_vars: Boolean matrix variables from encoding.
        is_x_type: Whether target is X-type check matrix.

    Returns:
        Tuple of (init_x, init_z) lists.
    """
    final_matrix = np.array(
        [[bool(model.eval(matrix_vars[row, q], model_completion=True)) for q in range(n)] for row in range(num_rows)],
        dtype=np.int8,
    )

    m = num_rows - k

    if m == 0:
        if is_x_type:
            return list(range(k, n)), []
        return [], list(range(k, n))

    stabilizer_part = final_matrix[k:]
    pivot_cols = _row_echelon_pivot_cols(stabilizer_part)

    init_x: list[int] = []
    init_z: list[int] = []

    if is_x_type:
        init_x = pivot_cols
        init_z = [q for q in range(n) if q not in pivot_cols and q >= k]
    else:
        init_z = pivot_cols
        init_x = [q for q in range(n) if q not in pivot_cols and q >= k]

    return init_x, init_z


def _synthesize_css(
    target: CheckMatrix,
    target_kind: TargetKind,
    objective: Objective,
    lower_bound: int,
    upper_bound: int,
    k: int | None,
    verify: bool,
) -> SynthesisResult:
    """Synthesize CSS CNOT circuit."""
    import z3

    n = target.num_qubits()

    if k is None:
        if target_kind == TargetKind.CSS_STATE:
            k = 0
        else:
            msg = "k must be provided for CSS isometry synthesis"
            raise ValueError(msg)

    m_x = target.num_rows() - k

    for bound in range(lower_bound, upper_bound + 1):
        solver, alpha_vars, beta_vars = encode_css_gate_count(
            target,
            k,
            m_x,
            bound,
        )

        result = solver.check()

        if result == z3.sat:
            model = solver.model()

            num_rows = target.num_rows()
            matrix_vars_final = np.array(
                [[z3.Bool(f"m_{bound}_{row}_{q}") for q in range(n)] for row in range(num_rows)], dtype=object
            )

            init_x, init_z = _determine_css_initializations(
                model,
                n,
                num_rows,
                k,
                matrix_vars_final,
                target.is_x_type(),
            )

            circuit = extract_cnot_gate_count_circuit(
                model,
                n,
                bound,
                alpha_vars,
                beta_vars,
                init_x,
                init_z,
            )

            actual_count = bound

            verified = False
            if verify:
                if target_kind == TargetKind.CSS_STATE:
                    verified = verify_css_state(circuit, target)
                else:
                    verified = verify_css_isometry(circuit, target, k)

            return SynthesisResult(
                status=SynthesisStatus.SUCCESS,
                circuit=circuit,
                gate_count=actual_count,
                verified=verified,
                message=f"Found solution with {actual_count} CNOTs",
            )

        if result == z3.unknown:
            return SynthesisResult(
                status=SynthesisStatus.ERROR,
                message=f"Solver returned unknown at bound {bound}: {solver.reason_unknown()}",
            )

    return SynthesisResult(
        status=SynthesisStatus.UNSAT,
        message=f"No solution found within bounds [{lower_bound}, {upper_bound}]",
    )
