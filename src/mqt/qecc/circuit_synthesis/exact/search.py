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
from ..circuits import CliffordIsometry, CNOTCircuit
from .encoding_interface import (
    CliffordDepthEncoding,
    CliffordGateCountEncoding,
    CSSDepthEncoding,
    CSSGateCountEncoding,
)
from .gate_operations import get_standard_clifford_gate_set, get_standard_css_gate_set
from .types import (
    Objective,
    SynthesisResult,
    SynthesisStatus,
    TargetKind,
)
from .verification import (
    verify_clifford_isometry,
    verify_clifford_unitary,
    verify_css_isometry,
    verify_css_state,
    verify_stabilizer_state,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .encoding_interface import (
        SynthesisEncoding,
    )
    from .gate_operations import SymbolicGateOperation

_CLIFFORD_KINDS = {TargetKind.CLIFFORD_UNITARY, TargetKind.CLIFFORD_ISOMETRY, TargetKind.STABILIZER_STATE}
_CSS_KINDS = {TargetKind.CSS_STATE, TargetKind.CSS_ISOMETRY}


def synthesize_exact(
    target: StabilizerTableau | CheckMatrix,
    target_kind: TargetKind,
    objective: Objective,
    lower_bound: int = 0,
    upper_bound: int = 10,
    x_logicals: StabilizerTableau | CheckMatrix | None = None,
    z_logicals: StabilizerTableau | CheckMatrix | None = None,
    verify: bool = True,
    allow_qubit_permutation: bool = True,
    gate_set: dict[str, type[SymbolicGateOperation]] | None = None,
    use_symmetry_breaking: bool = False,
    timeout: int | None = None,
) -> SynthesisResult:
    """Synthesize optimal circuit for given target using exact methods.

    The gate family (Clifford vs CSS-CNOT) is inferred from ``target_kind``.
    The number of logical qubits ``k`` is derived from the provided logicals.

    Args:
        target: Target stabilizer generators (StabilizerTableau) or check matrix (CheckMatrix).
        target_kind: Kind of synthesis problem.
        objective: Optimization objective (gate count or depth).
        lower_bound: Lower bound on resource count.
        upper_bound: Upper bound on resource count.
        x_logicals: Logical X operators. Required for CLIFFORD_UNITARY, CLIFFORD_ISOMETRY,
            and CSS_ISOMETRY with X-type checks. Must be a StabilizerTableau for Clifford synthesis.
        z_logicals: Logical Z operators. Required for CLIFFORD_UNITARY, CLIFFORD_ISOMETRY,
            and CSS_ISOMETRY with Z-type checks. Must be a StabilizerTableau for Clifford synthesis.
        verify: Whether to verify synthesized circuit.
        allow_qubit_permutation: Allow qubit permutation in the terminal constraint.
        gate_set: Custom gate set. If None, uses the standard gate set for the inferred family.
        use_symmetry_breaking: Add symmetry-breaking constraints to prune the SAT search space.
        timeout: Per-bound solver timeout in seconds. If the solver exceeds the timeout
            at any bound, synthesis stops and returns SynthesisStatus.TIMEOUT.

    Returns:
        SynthesisResult with circuit and metadata.

    Raises:
        ValueError: If parameters are invalid.
    """
    if gate_set is None:
        gate_set = get_standard_clifford_gate_set() if target_kind in _CLIFFORD_KINDS else get_standard_css_gate_set()

    _validate_synthesis_parameters(
        target,
        target_kind,
        lower_bound,
        upper_bound,
        x_logicals,
        z_logicals,
    )

    if target_kind in _CLIFFORD_KINDS:
        assert isinstance(target, StabilizerTableau)
        return _synthesize_clifford(
            target,
            target_kind,
            objective,
            lower_bound,
            upper_bound,
            x_logicals,
            z_logicals,
            verify,
            allow_qubit_permutation,
            gate_set,
            use_symmetry_breaking,
            timeout,
        )
    assert isinstance(target, CheckMatrix)
    return _synthesize_css(
        target,
        target_kind,
        objective,
        lower_bound,
        upper_bound,
        x_logicals,
        z_logicals,
        verify,
        gate_set,
        use_symmetry_breaking,
        timeout,
    )


def _validate_synthesis_parameters(
    target: StabilizerTableau | CheckMatrix,
    target_kind: TargetKind,
    lower_bound: int,
    upper_bound: int,
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

    if target_kind in {TargetKind.CLIFFORD_UNITARY, TargetKind.CLIFFORD_ISOMETRY} and (
        x_logicals is None or z_logicals is None
    ):
        msg = f"x_logicals and z_logicals must be provided for {target_kind.value} synthesis"
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

    if target_kind in _CLIFFORD_KINDS and not isinstance(target, StabilizerTableau):
        msg = f"{target_kind.value} requires StabilizerTableau, got {type(target).__name__}"
        raise ValueError(msg)
    if target_kind in _CSS_KINDS and not isinstance(target, CheckMatrix):
        msg = f"{target_kind.value} requires CheckMatrix, got {type(target).__name__}"
        raise ValueError(msg)


def _search_with_encoding(
    encoding: SynthesisEncoding,
    target: StabilizerTableau | CheckMatrix,
    lower_bound: int,
    upper_bound: int,
    k: int,
    verify_fn: Callable[[CliffordIsometry | CNOTCircuit], bool],
    is_depth: bool,
    verify: bool,
    postprocess: Callable[[CliffordIsometry | CNOTCircuit], CliffordIsometry | CNOTCircuit] | None = None,
    timeout: int | None = None,
) -> SynthesisResult:
    """Generic search loop using an encoding.

    Args:
        encoding: Encoding strategy to use.
        target: Combined target (tableau or check matrix).
        lower_bound: Lower bound on resources.
        upper_bound: Upper bound on resources.
        k: Number of logical qubits.
        verify_fn: Verification function called on the (post-processed) circuit.
        is_depth: Whether optimizing depth (vs gate count).
        verify: Whether to verify the synthesized circuit.
        postprocess: Optional transform applied to the extracted circuit before verification.
        timeout: Per-bound solver timeout in seconds. If the solver times out at any
            bound, returns TIMEOUT immediately.

    Returns:
        SynthesisResult.
    """
    gate_set = encoding.gate_set

    for bound in range(lower_bound, upper_bound + 1):
        solver = encoding.encode(target, k, bound)

        if timeout is not None:
            solver.set("timeout", timeout * 1000)

        result = solver.check()

        if result == z3.sat:
            model = solver.model()

            circuit: CliffordIsometry | CNOTCircuit = encoding.extract_circuit(model)

            if postprocess is not None:
                circuit = postprocess(circuit)

            actual_resources = encoding.compute_actual_resources(model)

            verified = verify and verify_fn(circuit)

            resource_key = "depth" if is_depth else "gate_count"
            resource_name = "depth" if is_depth else "gates"

            return SynthesisResult(
                status=SynthesisStatus.SUCCESS,
                circuit=circuit,
                **{resource_key: actual_resources},
                verified=verified,
                message=f"Found solution with {actual_resources} {resource_name}",
                gate_set=gate_set,
            )

        if result == z3.unknown:
            reason = solver.reason_unknown()
            if "timeout" in reason:
                return SynthesisResult(
                    status=SynthesisStatus.TIMEOUT,
                    message=f"Solver timed out at bound {bound}",
                    gate_set=gate_set,
                )
            return SynthesisResult(
                status=SynthesisStatus.ERROR,
                message=f"Solver returned unknown at bound {bound}: {reason}",
                gate_set=gate_set,
            )

    return SynthesisResult(
        status=SynthesisStatus.UNSAT,
        message=f"No solution found within bounds [{lower_bound}, {upper_bound}]",
        gate_set=gate_set,
    )


def _prepare_clifford_target(
    stabilizers: StabilizerTableau,
    target_kind: TargetKind,
    x_logicals: StabilizerTableau | CheckMatrix | None,
    z_logicals: StabilizerTableau | CheckMatrix | None,
) -> tuple[StabilizerTableau, int]:
    """Prepare combined target tableau for Clifford synthesis.

    Args:
        stabilizers: Stabilizer generators.
        target_kind: Kind of synthesis problem.
        x_logicals: Logical X operators.
        z_logicals: Logical Z operators.

    Returns:
        Tuple of (combined_target, k).
    """
    if target_kind == TargetKind.STABILIZER_STATE:
        return stabilizers, 0

    if not isinstance(x_logicals, StabilizerTableau) or not isinstance(z_logicals, StabilizerTableau):
        msg = "x_logicals and z_logicals must be StabilizerTableau for Clifford synthesis"
        raise TypeError(msg)

    k = x_logicals.num_rows()

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
    target_tableau: StabilizerTableau,
) -> CliffordIsometry:
    """Apply Pauli sign correction and initialize ancillas.

    Args:
        circuit: Extracted circuit from SAT model.
        n: Number of qubits.
        target_tableau: Target tableau with correct phases.

    Returns:
        Corrected circuit with proper initialization.
    """
    pivot_qubits = circuit.get_zero_initialized()
    stim_circuit = circuit.to_stim_circuit(with_resets=False)
    corrected_stim_circuit = _apply_pauli_sign_correction(stim_circuit, n, target_tableau, pivot_qubits)
    corrected_circuit = CliffordIsometry.from_stim_circuit(corrected_stim_circuit)

    for q in pivot_qubits:
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

    used_qubits: set[int] = set()
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


def _apply_pauli_sign_correction(
    circuit: stim.Circuit,
    n: int,
    target_tableau: StabilizerTableau,
    pivot_qubits: list[int] | None = None,
) -> stim.Circuit:
    """Apply Pauli sign correction to a circuit to match target phases.

    Args:
        circuit: The synthesized circuit (may have incorrect signs).
        n: Number of qubits.
        target_tableau: Target tableau with correct phases.
        pivot_qubits: Z-reset (ancilla) qubits. For these qubits get_code() reads
            zs_sign (the Z-image sign) rather than xs_sign (the X-image sign) when
            extracting stabilizer phases, so the comparison must use the same image.

    Returns:
        Circuit with Pauli correction prepended if needed.
    """
    circuit = _ensure_all_qubits_present(circuit, n)

    stim_tableau_data = circuit.to_tableau().to_numpy()

    num_target_rows = target_tableau.num_rows()

    synth_x = np.vstack((stim_tableau_data[0].astype(np.int8), stim_tableau_data[2].astype(np.int8)))
    synth_z = np.vstack((stim_tableau_data[1].astype(np.int8), stim_tableau_data[3].astype(np.int8)))

    xs_sign = stim_tableau_data[-2].astype(np.int8)
    zs_sign = stim_tableau_data[-1].astype(np.int8)

    # Default: xs_sign for positions 0..n-1, zs_sign for positions n..2n-1.
    # For Z-reset pivot qubits, get_code() uses z_output (zs_sign), not x_output
    # (xs_sign), so replace xs_sign[p] with zs_sign[p] at each pivot position p.
    synth_signs = np.concatenate((xs_sign.copy(), zs_sign))
    if pivot_qubits:
        for p in pivot_qubits:
            if p < n:
                synth_signs[p] = zs_sign[p]

    target_x = synth_x[:num_target_rows, :]
    target_z = synth_z[:num_target_rows, :]
    target_synth_signs = synth_signs[:num_target_rows]

    target_signs = target_tableau.phase.astype(np.int8)

    sign_difference = target_synth_signs ^ target_signs

    if np.all(sign_difference == 0):
        return circuit

    signed_tableau = np.hstack((target_x, target_z, np.array([sign_difference]).T))

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
    x_logicals: StabilizerTableau | CheckMatrix | None,
    z_logicals: StabilizerTableau | CheckMatrix | None,
    verify: bool,
    allow_qubit_permutation: bool,
    gate_set: dict[str, type[SymbolicGateOperation]],
    use_symmetry_breaking: bool = False,
    timeout: int | None = None,
) -> SynthesisResult:
    """Synthesize Clifford circuit."""
    target, k = _prepare_clifford_target(stabilizers, target_kind, x_logicals, z_logicals)
    n = target.n

    if objective == Objective.GATE_COUNT:
        encoding: SynthesisEncoding = CliffordGateCountEncoding(
            gate_set, allow_qubit_permutation, use_symmetry_breaking
        )
        is_depth = False
    else:
        encoding = CliffordDepthEncoding(gate_set, allow_qubit_permutation, use_symmetry_breaking)
        is_depth = True

    def postprocess(circuit: CliffordIsometry | CNOTCircuit) -> CliffordIsometry | CNOTCircuit:
        assert isinstance(circuit, CliffordIsometry)
        return _apply_pauli_correction_to_clifford(circuit, n, target)

    def verify_fn(circuit: CliffordIsometry | CNOTCircuit) -> bool:
        if target_kind == TargetKind.CLIFFORD_UNITARY:
            return verify_clifford_unitary(circuit, target)
        if target_kind == TargetKind.STABILIZER_STATE:
            return verify_stabilizer_state(circuit, stabilizers)
        return verify_clifford_isometry(circuit, target, k)

    return _search_with_encoding(
        encoding,
        target,
        lower_bound,
        upper_bound,
        k,
        verify_fn,
        is_depth,
        verify,
        postprocess=postprocess,
        timeout=timeout,
    )


def _prepare_css_target(
    checks: CheckMatrix,
    target_kind: TargetKind,
    x_logicals: StabilizerTableau | CheckMatrix | None,
    z_logicals: StabilizerTableau | CheckMatrix | None,
) -> tuple[CheckMatrix, int, int]:
    """Prepare combined CSS target matrix for synthesis.

    Args:
        checks: CSS check matrix.
        target_kind: Kind of synthesis problem.
        x_logicals: Logical X operators.
        z_logicals: Logical Z operators.

    Returns:
        Tuple of (combined_target, k, m_x) where k is the number of logical qubits
        and m_x is the number of stabilizers.
    """
    if target_kind == TargetKind.CSS_STATE:
        m_x = checks.num_rows()
        return checks, 0, m_x

    logicals = x_logicals if checks.is_x_type() else z_logicals
    if logicals is None:
        check_type = "X" if checks.is_x_type() else "Z"
        msg = f"{check_type.lower()}_logicals must be provided for CSS isometry with {check_type}-type checks"
        raise ValueError(msg)

    if isinstance(logicals, CheckMatrix):
        logical_matrix = logicals.matrix
    else:
        logical_matrix = logicals.get_x_part() if checks.is_x_type() else logicals.get_z_part()

    k = logical_matrix.shape[0]
    target = CheckMatrix(
        np.vstack([logical_matrix, checks.matrix]),
        pauli_type=checks.type,
    )

    m_x = target.num_rows() - k
    return target, k, m_x


def _synthesize_css(
    checks: CheckMatrix,
    target_kind: TargetKind,
    objective: Objective,
    lower_bound: int,
    upper_bound: int,
    x_logicals: StabilizerTableau | CheckMatrix | None,
    z_logicals: StabilizerTableau | CheckMatrix | None,
    verify: bool,
    gate_set: dict[str, type[SymbolicGateOperation]],
    use_symmetry_breaking: bool = False,
    timeout: int | None = None,
) -> SynthesisResult:
    """Synthesize CSS CNOT circuit."""
    target, k, m_x = _prepare_css_target(checks, target_kind, x_logicals, z_logicals)

    if objective == Objective.GATE_COUNT:
        encoding: SynthesisEncoding = CSSGateCountEncoding(gate_set, m_x, use_symmetry_breaking)
        is_depth = False
    else:
        encoding = CSSDepthEncoding(gate_set, m_x, use_symmetry_breaking)
        is_depth = True

    def verify_fn(circuit: CliffordIsometry | CNOTCircuit) -> bool:
        if not isinstance(circuit, CNOTCircuit):
            return False
        if target_kind == TargetKind.CSS_STATE:
            return verify_css_state(circuit, checks)
        return verify_css_isometry(circuit, checks, x_logicals if checks.type == "X" else z_logicals, k)

    return _search_with_encoding(
        encoding,
        target,
        lower_bound,
        upper_bound,
        k,
        verify_fn,
        is_depth,
        verify,
        timeout=timeout,
    )
