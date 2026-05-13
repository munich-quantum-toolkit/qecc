# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Search strategies for exact synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import stim

from ...codes.pauli import CheckMatrix, StabilizerTableau
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


def synthesize_exact(
    target: StabilizerTableau | CheckMatrix,
    target_kind: TargetKind,
    gate_family: GateFamily,
    objective: Objective,
    lower_bound: int = 0,
    upper_bound: int = 10,
    k: int | None = None,
    x_logicals: StabilizerTableau | CheckMatrix | None = None,
    z_logicals: StabilizerTableau | CheckMatrix | None = None,
    verify: bool = True,
    allow_qubit_permutation: bool = True,
) -> SynthesisResult:
    """Synthesize optimal circuit for given target using exact methods.

    Args:
        target: Target stabilizer generators (for states/isometries) or check matrix (for CSS).
        target_kind: Kind of synthesis problem.
        gate_family: Gate family to use.
        objective: Optimization objective.
        lower_bound: Lower bound on resource count.
        upper_bound: Upper bound on resource count.
        k: Number of logical qubits (required for isometry).
        x_logicals: Logical X operators (StabilizerTableau for Clifford, CheckMatrix or StabilizerTableau for CSS).
        z_logicals: Logical Z operators (StabilizerTableau for Clifford, CheckMatrix or StabilizerTableau for CSS).
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

    if target_kind == TargetKind.CLIFFORD_ISOMETRY and (x_logicals is None or z_logicals is None):
        msg = "x_logicals and z_logicals must be provided for Clifford isometry synthesis"
        raise ValueError(msg)

    if target_kind == TargetKind.CSS_ISOMETRY:
        if not isinstance(target, CheckMatrix):
            msg = f"CSS_ISOMETRY requires CheckMatrix, got {type(target).__name__}"
            raise ValueError(msg)
        if target.is_x_type() and x_logicals is None:
            msg = "x_logicals must be provided for CSS isometry with X-type checks"
            raise ValueError(msg)
        if target.is_z_type() and z_logicals is None:
            msg = "z_logicals must be provided for CSS isometry with Z-type checks"
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
            x_logicals,
            z_logicals,
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
        x_logicals,
        z_logicals,
        verify,
    )


def _combine_stabilizers_and_logicals(
    stabilizers: StabilizerTableau,
    k: int,
    x_logicals: StabilizerTableau | None = None,
    z_logicals: StabilizerTableau | None = None,
) -> StabilizerTableau:
    """Combine stabilizers and logicals into a single tableau for synthesis.

    Args:
        stabilizers: Stabilizer generators.
        k: Number of logical qubits.
        x_logicals: Logical X operators.
        z_logicals: Logical Z operators.

    Returns:
        Combined tableau with rows ordered as [X_logicals, Z_logicals, stabilizers].
    """
    if k == 0:
        return stabilizers

    if x_logicals is None or z_logicals is None:
        msg = "x_logicals and z_logicals must be provided when k > 0"
        raise ValueError(msg)

    if x_logicals.num_rows() != k or z_logicals.num_rows() != k:
        msg = f"Expected {k} logical X and Z operators, got {x_logicals.num_rows()} X and {z_logicals.num_rows()} Z"
        raise ValueError(msg)

    combined_matrix = np.vstack([
        x_logicals.tableau.matrix,
        z_logicals.tableau.matrix,
        stabilizers.tableau.matrix,
    ])

    combined_phase = np.concatenate([
        x_logicals.phase,
        z_logicals.phase,
        stabilizers.phase,
    ])

    return StabilizerTableau(combined_matrix, combined_phase)


def _ensure_all_qubits_present(circuit: stim.Circuit, n: int) -> stim.Circuit:
    """Ensure all qubits from 0 to n-1 are present in the circuit.

    Stim silently removes unused qubits, which can cause issues with verification.
    This function adds identity operations on any missing qubits to ensure they exist.

    Args:
        circuit: The stim circuit.
        n: The number of qubits that should be present.

    Returns:
        A circuit with all qubits from 0 to n-1 present.
    """
    if n == 0:
        return circuit

    used_qubits = set()
    for instruction in circuit:
        for target_group in instruction.target_groups():
            used_qubits.update(target.qubit_value for target in target_group)

    missing_qubits = [q for q in range(n) if q not in used_qubits]

    if not missing_qubits:
        return circuit

    result = stim.Circuit()
    result.append("I", missing_qubits)
    result += circuit

    return result


def _apply_pauli_sign_correction(circuit: stim.Circuit, n: int) -> stim.Circuit:
    """Apply Pauli sign correction to a circuit to match target phases.

    Args:
        circuit: The synthesized circuit (may have incorrect signs).
        n: Number of qubits.

    Returns:
        Circuit with Pauli correction prepended if needed.
    """
    circuit = _ensure_all_qubits_present(circuit, n)

    stim_tableau = circuit.to_tableau().to_numpy()

    x_part = np.vstack((stim_tableau[0].astype(np.int8), stim_tableau[2].astype(np.int8)))
    z_part = np.vstack((stim_tableau[1].astype(np.int8), stim_tableau[3].astype(np.int8)))
    signs = np.concatenate((stim_tableau[-2].astype(int), stim_tableau[-1].astype(int)))

    signed_tableau = np.hstack((x_part, z_part, np.array([signs]).T))

    if np.all(signed_tableau[:, -1] == 0):
        return circuit

    kernel = mod2.nullspace(signed_tableau)

    if kernel.size == 0:
        return circuit

    correction_symplectic = kernel[-1]

    if correction_symplectic[-1] != 1:
        return circuit

    z_correction = correction_symplectic[:n]
    x_correction = correction_symplectic[n:-1]

    corrected_circuit = stim.Circuit()

    for q, (xv, zv) in enumerate(zip(x_correction, z_correction, strict=False)):
        if xv == 1 and zv == 1:
            corrected_circuit.append("Y", [q])
        elif xv == 1:
            corrected_circuit.append("X", [q])
        elif zv == 1:
            corrected_circuit.append("Z", [q])

    corrected_circuit += circuit

    return corrected_circuit


def _synthesize_clifford(
    stabilizers: StabilizerTableau,
    target_kind: TargetKind,
    objective: Objective,
    lower_bound: int,
    upper_bound: int,
    k: int | None,
    x_logicals: StabilizerTableau | CheckMatrix | None,
    z_logicals: StabilizerTableau | CheckMatrix | None,
    verify: bool,
    allow_qubit_permutation: bool,
) -> SynthesisResult:
    """Synthesize Clifford circuit."""
    import z3

    if k is None:
        if target_kind == TargetKind.CLIFFORD_UNITARY:
            k = stabilizers.n
            x_logicals = StabilizerTableau.from_pauli_strings([
                "I" * i + "X" + "I" * (stabilizers.n - i - 1) for i in range(stabilizers.n)
            ])
            z_logicals = StabilizerTableau.from_pauli_strings([
                "I" * i + "Z" + "I" * (stabilizers.n - i - 1) for i in range(stabilizers.n)
            ])
        elif target_kind == TargetKind.STABILIZER_STATE:
            k = 0
        else:
            msg = "k must be provided for isometry synthesis"
            raise ValueError(msg)

    if k > 0 and (not isinstance(x_logicals, StabilizerTableau) or not isinstance(z_logicals, StabilizerTableau)):
        msg = "x_logicals and z_logicals must be StabilizerTableau for Clifford synthesis"
        raise ValueError(msg)

    target = _combine_stabilizers_and_logicals(stabilizers, k, x_logicals, z_logicals)

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
                k,
            )

            actual_count = sum(
                1
                for slot in range(bound)
                if model.eval(z3.Or(h_vars[slot], s_vars[slot], c_vars[slot]), model_completion=True)
            )

            stim_circuit = circuit.to_stim_circuit(with_resets=False)
            corrected_stim_circuit = _apply_pauli_sign_correction(stim_circuit, n)

            from ..circuits import CliffordIsometry

            corrected_circuit = CliffordIsometry.from_stim_circuit(corrected_stim_circuit)

            if target_kind == TargetKind.STABILIZER_STATE:
                for q in circuit.get_zero_initialized():
                    corrected_circuit.initialize_qubit(q, basis="Z")

            verified = False
            if verify:
                if target_kind == TargetKind.CLIFFORD_UNITARY:
                    verified = verify_clifford_unitary(corrected_circuit, target)
                elif target_kind == TargetKind.STABILIZER_STATE:
                    verified = verify_stabilizer_state(corrected_circuit, stabilizers)
                else:
                    verified = verify_clifford_isometry(corrected_circuit, target, k)

            return SynthesisResult(
                status=SynthesisStatus.SUCCESS,
                circuit=corrected_circuit,
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

    logical_part = final_matrix[:k]
    stabilizer_part = final_matrix[k:]

    stabilizer_pivot_cols = _row_echelon_pivot_cols(stabilizer_part)

    input_qubits = []
    for col in range(n):
        if col in stabilizer_pivot_cols:
            continue
        for row in range(k):
            if logical_part[row, col] == 1:
                input_qubits.append(col)
                break

    ancilla_qubits = [q for q in range(n) if q not in input_qubits]

    init_x: list[int] = []
    init_z: list[int] = []

    if is_x_type:
        init_x = [q for q in stabilizer_pivot_cols if q in ancilla_qubits]
        init_z = [q for q in ancilla_qubits if q not in init_x]
    else:
        init_z = [q for q in stabilizer_pivot_cols if q in ancilla_qubits]
        init_x = [q for q in ancilla_qubits if q not in init_z]

    return init_x, init_z


def _synthesize_css(
    checks: CheckMatrix,
    target_kind: TargetKind,
    objective: Objective,
    lower_bound: int,
    upper_bound: int,
    k: int | None,
    x_logicals: StabilizerTableau | CheckMatrix | None,
    z_logicals: StabilizerTableau | CheckMatrix | None,
    verify: bool,
) -> SynthesisResult:
    """Synthesize CSS CNOT circuit."""
    import z3

    n = checks.num_qubits()

    if k is None:
        if target_kind == TargetKind.CSS_STATE:
            k = 0
        else:
            msg = "k must be provided for CSS isometry synthesis"
            raise ValueError(msg)

    if k > 0:
        logicals = x_logicals if checks.is_x_type() else z_logicals
        if logicals is None:
            check_type = "X" if checks.is_x_type() else "Z"
            msg = f"{check_type.lower()}_logicals must be provided for CSS isometry with {check_type}-type checks"
            raise ValueError(msg)

        if isinstance(logicals, CheckMatrix):
            logical_matrix = logicals.matrix
        else:
            logical_matrix = logicals.get_x_part() if checks.is_x_type() else logicals.get_z_part()

        target = CheckMatrix(
            np.vstack([logical_matrix, checks.matrix]),
            pauli_type=checks.type,
        )
    else:
        target = checks

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
                    verified = verify_css_state(circuit, checks)
                else:
                    verified = verify_css_isometry(circuit, checks, x_logicals if checks.type == "X" else z_logicals, k)

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
