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
import z3

from ...codes.pauli import CheckMatrix, StabilizerTableau
from ..circuits import CliffordIsometry
from .encoding_depth import encode_clifford_depth, encode_css_depth
from .encoding_gate_count import encode_clifford_gate_count, encode_css_gate_count
from .extraction import (
    extract_clifford_depth_circuit,
    extract_clifford_gate_count_circuit,
    extract_cnot_depth_circuit,
    extract_cnot_gate_count_circuit,
)
from .types import GateFamily, Objective, SynthesisResult, SynthesisStatus, TargetKind
from .verification import (
    verify_clifford_isometry,
    verify_clifford_unitary,
    verify_css_isometry,
    verify_css_state,
    verify_stabilizer_state,
)

if TYPE_CHECKING:
    from ..circuits import CNOTCircuit


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
    """
    _validate_synthesis_parameters(
        target,
        target_kind,
        gate_family,
        objective,
        lower_bound,
        upper_bound,
        k,
        x_logicals,
        z_logicals,
    )

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


def _validate_synthesis_parameters(
    target: StabilizerTableau | CheckMatrix,
    target_kind: TargetKind,
    gate_family: GateFamily,
    objective: Objective,
    lower_bound: int,
    upper_bound: int,
    k: int | None,
    x_logicals: StabilizerTableau | CheckMatrix | None,
    z_logicals: StabilizerTableau | CheckMatrix | None,
) -> None:
    """Validate synthesis parameters.

    Raises:
        ValueError: If parameters are invalid.
    """
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


def _prepare_clifford_target(
    stabilizers: StabilizerTableau,
    target_kind: TargetKind,
    k: int | None,
    x_logicals: StabilizerTableau | CheckMatrix | None,
    z_logicals: StabilizerTableau | CheckMatrix | None,
) -> tuple[StabilizerTableau, int]:
    """Prepare combined target tableau for Clifford synthesis.

    Args:
        stabilizers: Stabilizer generators.
        target_kind: Kind of synthesis problem.
        k: Number of logical qubits.
        x_logicals: Logical X operators.
        z_logicals: Logical Z operators.

    Returns:
        Tuple of (combined_target, k).
    """
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
    return target, k


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


def _apply_pauli_correction_to_clifford(
    circuit: CliffordIsometry,
    n: int,
    target_kind: TargetKind,
) -> CliffordIsometry:
    """Apply Pauli sign correction and initialize ancillas.

    Args:
        circuit: Extracted circuit from SAT model.
        n: Number of qubits.
        target_kind: Kind of synthesis problem.

    Returns:
        Corrected circuit with proper initialization.
    """
    stim_circuit = circuit.to_stim_circuit(with_resets=False)
    corrected_stim_circuit = _apply_pauli_sign_correction(stim_circuit, n)
    corrected_circuit = CliffordIsometry.from_stim_circuit(corrected_stim_circuit)

    if target_kind == TargetKind.STABILIZER_STATE:
        for q in circuit.get_zero_initialized():
            corrected_circuit.initialize_qubit(q, basis="Z")

    return corrected_circuit


def _ensure_all_qubits_present(circuit: stim.Circuit, n: int) -> stim.Circuit:
    """Ensure all qubits from 0 to n-1 are present in the circuit.

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


def _verify_clifford_result(
    circuit: CliffordIsometry,
    target: StabilizerTableau,
    target_kind: TargetKind,
    k: int,
    stabilizers: StabilizerTableau,
) -> bool:
    """Verify Clifford synthesis result.

    Args:
        circuit: Synthesized circuit.
        target: Combined target tableau.
        target_kind: Kind of synthesis problem.
        k: Number of logical qubits.
        stabilizers: Original stabilizer generators.

    Returns:
        True if verification succeeds.
    """
    if target_kind == TargetKind.CLIFFORD_UNITARY:
        return verify_clifford_unitary(circuit, target)
    if target_kind == TargetKind.STABILIZER_STATE:
        return verify_stabilizer_state(circuit, stabilizers)
    return verify_clifford_isometry(circuit, target, k)


def _compute_actual_gate_count(
    model: z3.ModelRef,
    bound: int,
    h_vars: list[z3.BoolRef],
    s_vars: list[z3.BoolRef],
    c_vars: list[z3.BoolRef],
) -> int:
    """Compute actual gate count from SAT model.

    Args:
        model: Z3 model.
        bound: Maximum number of gates.
        h_vars: Hadamard variables.
        s_vars: S gate variables.
        c_vars: CNOT variables.

    Returns:
        Actual number of gates used.
    """
    return sum(
        1 for slot in range(bound) if model.eval(z3.Or(h_vars[slot], s_vars[slot], c_vars[slot]), model_completion=True)
    )


def _compute_actual_depth_clifford(
    model: z3.ModelRef,
    bound: int,
    n: int,
    h_vars: list[list[z3.BoolRef]],
    s_vars: list[list[z3.BoolRef]],
    cx_vars: list[list[z3.BoolRef]],
) -> int:
    """Compute actual circuit depth from SAT model for Clifford circuits.

    Args:
        model: Z3 model.
        bound: Maximum depth.
        n: Number of qubits.
        h_vars: Hadamard variables [layer][qubit].
        s_vars: S gate variables [layer][qubit].
        cx_vars: CNOT variables [layer][cx_idx].

    Returns:
        Actual circuit depth.
    """
    actual_depth = 0
    for layer in range(bound):
        layer_has_gate = False
        for q in range(n):
            if model.eval(h_vars[layer][q], model_completion=True) or model.eval(
                s_vars[layer][q], model_completion=True
            ):
                layer_has_gate = True
                break
        if not layer_has_gate:
            for cx_idx in range(len(cx_vars[layer])):
                if model.eval(cx_vars[layer][cx_idx], model_completion=True):
                    layer_has_gate = True
                    break
        if layer_has_gate:
            actual_depth += 1
    return actual_depth


def _compute_actual_depth_css(
    model: z3.ModelRef,
    bound: int,
    cx_vars: list[list[z3.BoolRef]],
) -> int:
    """Compute actual circuit depth from SAT model for CSS circuits.

    Args:
        model: Z3 model.
        bound: Maximum depth.
        cx_vars: CNOT variables [layer][cx_idx].

    Returns:
        Actual circuit depth.
    """
    actual_depth = 0
    for layer in range(bound):
        layer_has_gate = False
        for cx_idx in range(len(cx_vars[layer])):
            if model.eval(cx_vars[layer][cx_idx], model_completion=True):
                layer_has_gate = True
                break
        if layer_has_gate:
            actual_depth += 1
    return actual_depth


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
    target, k = _prepare_clifford_target(stabilizers, target_kind, k, x_logicals, z_logicals)

    if objective == Objective.GATE_COUNT:
        return _synthesize_clifford_gate_count(
            target,
            target_kind,
            lower_bound,
            upper_bound,
            k,
            stabilizers,
            verify,
            allow_qubit_permutation,
        )
    return _synthesize_clifford_depth(
        target,
        target_kind,
        lower_bound,
        upper_bound,
        k,
        stabilizers,
        verify,
        allow_qubit_permutation,
    )


def _synthesize_clifford_gate_count(
    target: StabilizerTableau,
    target_kind: TargetKind,
    lower_bound: int,
    upper_bound: int,
    k: int,
    stabilizers: StabilizerTableau,
    verify: bool,
    allow_qubit_permutation: bool,
) -> SynthesisResult:
    """Synthesize Clifford circuit with gate-count optimization."""
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

            actual_count = _compute_actual_gate_count(model, bound, h_vars, s_vars, c_vars)

            corrected_circuit = _apply_pauli_correction_to_clifford(circuit, n, target_kind)

            verified = False
            if verify:
                verified = _verify_clifford_result(corrected_circuit, target, target_kind, k, stabilizers)

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


def _synthesize_clifford_depth(
    target: StabilizerTableau,
    target_kind: TargetKind,
    lower_bound: int,
    upper_bound: int,
    k: int,
    stabilizers: StabilizerTableau,
    verify: bool,
    allow_qubit_permutation: bool,
) -> SynthesisResult:
    """Synthesize Clifford circuit with depth optimization."""
    for bound in range(lower_bound, upper_bound + 1):
        solver, h_vars, s_vars, cx_vars, _id_vars = encode_clifford_depth(
            target,
            k,
            bound,
            allow_qubit_permutation,
        )

        result = solver.check()

        if result == z3.sat:
            model = solver.model()
            n = target.n

            circuit = extract_clifford_depth_circuit(
                model,
                n,
                bound,
                h_vars,
                s_vars,
                cx_vars,
                k,
            )

            actual_depth = _compute_actual_depth_clifford(model, bound, n, h_vars, s_vars, cx_vars)

            corrected_circuit = _apply_pauli_correction_to_clifford(circuit, n, target_kind)

            verified = False
            if verify:
                verified = _verify_clifford_result(corrected_circuit, target, target_kind, k, stabilizers)

            return SynthesisResult(
                status=SynthesisStatus.SUCCESS,
                circuit=corrected_circuit,
                depth=actual_depth,
                verified=verified,
                message=f"Found solution with depth {actual_depth}",
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


def _prepare_css_target(
    checks: CheckMatrix,
    k: int,
    x_logicals: StabilizerTableau | CheckMatrix | None,
    z_logicals: StabilizerTableau | CheckMatrix | None,
) -> tuple[CheckMatrix, int]:
    """Prepare combined CSS target matrix for synthesis.

    Args:
        checks: CSS check matrix.
        k: Number of logical qubits.
        x_logicals: Logical X operators.
        z_logicals: Logical Z operators.

    Returns:
        Tuple of (combined_target, m_x) where m_x is number of stabilizers.
    """
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
    return target, m_x


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


def _verify_css_result(
    circuit: CNOTCircuit,
    target_kind: TargetKind,
    checks: CheckMatrix,
    x_logicals: StabilizerTableau | CheckMatrix | None,
    z_logicals: StabilizerTableau | CheckMatrix | None,
    k: int,
) -> bool:
    """Verify CSS synthesis result.

    Args:
        circuit: Synthesized circuit.
        target_kind: Kind of synthesis problem.
        checks: CSS check matrix.
        x_logicals: Logical X operators.
        z_logicals: Logical Z operators.
        k: Number of logical qubits.

    Returns:
        True if verification succeeds.
    """
    if target_kind == TargetKind.CSS_STATE:
        return verify_css_state(circuit, checks)
    return verify_css_isometry(circuit, checks, x_logicals if checks.type == "X" else z_logicals, k)


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
    n = checks.num_qubits()

    if k is None:
        if target_kind == TargetKind.CSS_STATE:
            k = 0
        else:
            msg = "k must be provided for CSS isometry synthesis"
            raise ValueError(msg)

    target, m_x = _prepare_css_target(checks, k, x_logicals, z_logicals)

    if objective == Objective.GATE_COUNT:
        return _synthesize_css_gate_count(
            target,
            target_kind,
            lower_bound,
            upper_bound,
            k,
            m_x,
            n,
            checks,
            x_logicals,
            z_logicals,
            verify,
        )
    return _synthesize_css_depth(
        target,
        target_kind,
        lower_bound,
        upper_bound,
        k,
        m_x,
        n,
        checks,
        x_logicals,
        z_logicals,
        verify,
    )


def _synthesize_css_gate_count(
    target: CheckMatrix,
    target_kind: TargetKind,
    lower_bound: int,
    upper_bound: int,
    k: int,
    m_x: int,
    n: int,
    checks: CheckMatrix,
    x_logicals: StabilizerTableau | CheckMatrix | None,
    z_logicals: StabilizerTableau | CheckMatrix | None,
    verify: bool,
) -> SynthesisResult:
    """Synthesize CSS CNOT circuit with gate-count optimization."""
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
                verified = _verify_css_result(circuit, target_kind, checks, x_logicals, z_logicals, k)

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


def _synthesize_css_depth(
    target: CheckMatrix,
    target_kind: TargetKind,
    lower_bound: int,
    upper_bound: int,
    k: int,
    m_x: int,
    n: int,
    checks: CheckMatrix,
    x_logicals: StabilizerTableau | CheckMatrix | None,
    z_logicals: StabilizerTableau | CheckMatrix | None,
    verify: bool,
) -> SynthesisResult:
    """Synthesize CSS CNOT circuit with depth optimization."""
    for bound in range(lower_bound, upper_bound + 1):
        solver, cx_vars, _id_vars = encode_css_depth(
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

            circuit = extract_cnot_depth_circuit(
                model,
                n,
                bound,
                cx_vars,
                init_x,
                init_z,
            )

            actual_depth = _compute_actual_depth_css(model, bound, cx_vars)

            verified = False
            if verify:
                verified = _verify_css_result(circuit, target_kind, checks, x_logicals, z_logicals, k)

            return SynthesisResult(
                status=SynthesisStatus.SUCCESS,
                circuit=circuit,
                depth=actual_depth,
                verified=verified,
                message=f"Found solution with depth {actual_depth}",
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
