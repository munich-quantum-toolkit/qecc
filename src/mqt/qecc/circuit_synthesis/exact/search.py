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

    import numpy.typing as npt

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
    max_two_qubit_gates: int | None = None,
    timeout: int | None = None,
    use_exponential_backoff: bool = False,
    min_timeout: int = 1,
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
        max_two_qubit_gates: Maximum total number of two-qubit (CX/CNOT) gates allowed in the
            circuit. Only meaningful for depth-optimal synthesis, where it constrains the CNOT
            count while still minimizing depth. Ignored for gate-count synthesis (where the
            gate count already bounds the total).
        timeout: Per-bound solver timeout in seconds. In the default fixed-timeout strategy
            the first timeout stops the search immediately. When ``use_exponential_backoff``
            is enabled this becomes the *maximum* per-bound budget.
        use_exponential_backoff: Use exponential-backoff search instead of the default
            fixed-timeout linear scan. The search starts at ``min_timeout`` seconds per
            bound and doubles the budget after each full pass over remaining (timed-out)
            bounds, up to ``timeout``. Bounds proven UNSAT are dropped permanently. Once
            a SAT solution is found, a descending phase tries lower bounds at the maximum
            budget. Not guaranteed to find the optimum within the time budget, but often
            finds good solutions faster in practice.
        min_timeout: Starting per-bound timeout in seconds for the exponential-backoff
            strategy. Only used when ``use_exponential_backoff`` is True.

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
            max_two_qubit_gates,
            timeout,
            use_exponential_backoff,
            min_timeout,
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
        max_two_qubit_gates,
        timeout,
        use_exponential_backoff,
        min_timeout,
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
            return _make_success_result(
                encoding, solver.model(), is_depth, verify, verify_fn, postprocess, proven_optimal=True
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


def _make_success_result(
    encoding: SynthesisEncoding,
    model: z3.ModelRef,
    is_depth: bool,
    verify: bool,
    verify_fn: Callable[[CliffordIsometry | CNOTCircuit], bool],
    postprocess: Callable[[CliffordIsometry | CNOTCircuit], CliffordIsometry | CNOTCircuit] | None,
    proven_optimal: bool = False,
) -> SynthesisResult:
    """Extract a SAT model into a SynthesisResult."""
    circuit: CliffordIsometry | CNOTCircuit = encoding.extract_circuit(model)
    if postprocess is not None:
        circuit = postprocess(circuit)
    actual_resources = encoding.compute_actual_resources(model)
    verified = verify and verify_fn(circuit)
    if is_depth:
        opt_depth = actual_resources
        opt_gate_count = _circuit_gate_count(circuit)
    else:
        opt_gate_count = actual_resources
        opt_depth = _circuit_depth(circuit)
    resource_name = "depth" if is_depth else "gates"
    return SynthesisResult(
        status=SynthesisStatus.SUCCESS,
        circuit=circuit,
        gate_count=opt_gate_count,
        depth=opt_depth,
        verified=verified,
        message=f"Found solution with {actual_resources} {resource_name}",
        gate_set=encoding.gate_set,
        proven_optimal=proven_optimal,
    )


def _search_with_exponential_backoff(
    encoding: SynthesisEncoding,
    target: StabilizerTableau | CheckMatrix,
    lower_bound: int,
    upper_bound: int,
    k: int,
    verify_fn: Callable[[CliffordIsometry | CNOTCircuit], bool],
    is_depth: bool,
    verify: bool,
    postprocess: Callable[[CliffordIsometry | CNOTCircuit], CliffordIsometry | CNOTCircuit] | None = None,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> SynthesisResult:
    """Search with exponential timeout backoff.

    Phase A — ascending linear scan: starting with ``min_timeout`` seconds per
    bound, scan from ``lower_bound`` to ``upper_bound``.  Bounds proven UNSAT
    are dropped permanently; timed-out bounds are retried with a doubled budget.
    This continues until a SAT result is found or all pending bounds time out at
    ``max_timeout``.

    Phase B — descending refinement: once a SAT solution is found at bound
    ``b``, descend from ``b - 1`` downward using ``max_timeout`` per bound.
    Stops when UNSAT (optimal proven) or TIMEOUT (best-known solution returned).

    Args:
        encoding: Encoding strategy to use.
        target: Combined target (tableau or check matrix).
        lower_bound: Lower bound on resources.
        upper_bound: Upper bound on resources.
        k: Number of logical qubits.
        verify_fn: Verification function called on the (post-processed) circuit.
        is_depth: Whether optimizing depth (vs gate count).
        verify: Whether to verify the synthesized circuit.
        postprocess: Optional transform applied to the extracted circuit.
        min_timeout: Starting per-bound timeout in seconds.
        max_timeout: Maximum per-bound timeout in seconds.

    Returns:
        SynthesisResult.
    """
    gate_set = encoding.gate_set
    pending = list(range(lower_bound, upper_bound + 1))
    curr_timeout = min_timeout

    while pending:
        still_pending: list[int] = []
        for bound in pending:
            solver = encoding.encode(target, k, bound)
            solver.set("timeout", curr_timeout * 1000)
            result = solver.check()

            if result == z3.sat:
                best = _make_success_result(encoding, solver.model(), is_depth, verify, verify_fn, postprocess)
                # Phase B: descend with max_timeout to tighten the solution
                proven = False
                for b in range(bound - 1, lower_bound - 1, -1):
                    s = encoding.encode(target, k, b)
                    s.set("timeout", max_timeout * 1000)
                    r = s.check()
                    if r == z3.sat:
                        best = _make_success_result(encoding, s.model(), is_depth, verify, verify_fn, postprocess)
                    elif r == z3.unsat:
                        proven = True
                        break
                    else:  # timeout: best known is all we have
                        break
                else:
                    # Exhausted all smaller bounds — all SAT, so lower_bound is optimal.
                    proven = True
                best.proven_optimal = proven
                return best

            if result == z3.unknown:
                reason = solver.reason_unknown()
                if "timeout" in reason:
                    still_pending.append(bound)
                else:
                    return SynthesisResult(
                        status=SynthesisStatus.ERROR,
                        message=f"Solver returned unknown at bound {bound}: {reason}",
                        gate_set=gate_set,
                    )
            # z3.unsat: proven infeasible, not retried

        pending = still_pending
        if not pending or curr_timeout >= max_timeout:
            break
        curr_timeout = min(curr_timeout * 2, max_timeout)

    if pending:
        return SynthesisResult(
            status=SynthesisStatus.TIMEOUT,
            message=f"Bounds {pending[0]}..{pending[-1]} unresolved at max timeout ({max_timeout}s)",
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
    corrected_stim_circuit = _apply_pauli_sign_correction(stim_circuit, n, target_tableau)
    corrected_circuit = CliffordIsometry.from_stim_circuit(corrected_stim_circuit)

    for q in pivot_qubits:
        corrected_circuit.initialize_qubit(q, basis="Z")

    return corrected_circuit


_SINGLE_QUBIT_CLIFFORD = {"H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG"}
_TWO_QUBIT_CLIFFORD = {"CX", "CZ"}


def _circuit_gate_count(circuit: CliffordIsometry | CNOTCircuit) -> int:
    """Count the total number of non-identity non-Pauli gates in a synthesized circuit."""
    if isinstance(circuit, CNOTCircuit):
        return circuit.num_cnots()
    count = 0
    for inst in circuit.to_stim_circuit():
        if inst.name in _SINGLE_QUBIT_CLIFFORD:
            count += len(inst.targets_copy())
        elif inst.name in _TWO_QUBIT_CLIFFORD:
            count += len(inst.targets_copy()) // 2
    return count


def _circuit_depth(circuit: CliffordIsometry | CNOTCircuit) -> int:
    """Return the two-qubit-gate depth of a synthesized circuit."""
    return circuit.depth()


def _gf2_rref_track(mat: npt.NDArray[np.int8]) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """GF(2) row-reduce mat to RREF, tracking the transformation.

    Args:
        mat: Binary matrix to row-reduce.

    Returns:
        (E, R) where E = RREF(mat) and R @ mat = E mod 2.
    """
    n_rows, n_cols = mat.shape
    work = mat.copy().astype(np.int8)
    r_mat = np.eye(n_rows, dtype=np.int8)
    current_row = 0
    for col in range(n_cols):
        pivot = next((r for r in range(current_row, n_rows) if work[r, col]), None)
        if pivot is None:
            continue
        work[[current_row, pivot]] = work[[pivot, current_row]]
        r_mat[[current_row, pivot]] = r_mat[[pivot, current_row]]
        for r in range(n_rows):
            if r != current_row and work[r, col]:
                work[r] ^= work[current_row]
                r_mat[r] ^= r_mat[current_row]
        current_row += 1
    return work, r_mat


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
) -> stim.Circuit:
    """Apply Pauli sign correction to a circuit to match target phases.

    The circuit U maps source generators to target generators:
    - X-logical row i  → U(X_{selector_i}) → X_logical_i
    - Z-logical row i  → U(Z_{selector_i}) → Z_logical_i
    - Stabilizer rows  → U(Z_{pivot_q})    → some element of target stabilizer group

    To flip the sign of U(X_q): prepend Z_q (anticommutes with X_q) → z_correction[q].
    To flip the sign of U(Z_q): prepend X_q (anticommutes with Z_q) → x_correction[q].

    For stabilizer rows the circuit may use a different GF(2) basis than the target, so
    individual row matching does not work. We work at the group level: prepending X on
    each pivot qubit q flips the sign of that generator in the circuit's stabilizer group.
    The correct x_correction[q] for pivot qubits is determined by solving the linear system
    that makes the circuit's signed stabilizer group equal to the target's.

    Args:
        circuit: The synthesized circuit (without reset gates, may have incorrect signs).
        n: Number of qubits.
        target_tableau: Target tableau with correct phases.

    Returns:
        Circuit with Pauli correction prepended if needed.
    """
    circuit = _ensure_all_qubits_present(circuit, n)

    stim_tableau_data = circuit.to_tableau().to_numpy()

    num_target_rows = target_tableau.num_rows()
    # num_target_rows = 2k + (n-k) = n+k  →  k = num_target_rows - n
    k = num_target_rows - n

    xs_sign = stim_tableau_data[-2].astype(np.int8)
    zs_sign = stim_tableau_data[-1].astype(np.int8)
    x2x = stim_tableau_data[0].astype(np.int8)
    x2z = stim_tableau_data[1].astype(np.int8)
    z2x = stim_tableau_data[2].astype(np.int8)
    z2z = stim_tableau_data[3].astype(np.int8)

    target_x = target_tableau.tableau.matrix[:num_target_rows, :n].astype(np.int8)
    target_z = target_tableau.tableau.matrix[:num_target_rows, n:].astype(np.int8)
    target_signs = target_tableau.phase[:num_target_rows].astype(np.int8)

    x_correction = np.zeros(n, dtype=np.int8)
    z_correction = np.zeros(n, dtype=np.int8)
    selector_qubits: set[int] = set()

    # X-logical rows (0..k-1): U(X_q) exactly matches each target row.
    # Fix sign by prepending Z_q (anticommutes with X_q) → z_correction[q].
    for row_idx in range(k):
        tx, tz = target_x[row_idx], target_z[row_idx]
        for q in range(n):
            if np.array_equal(x2x[q], tx) and np.array_equal(x2z[q], tz):
                selector_qubits.add(q)
                if xs_sign[q] ^ target_signs[row_idx]:
                    z_correction[q] ^= 1
                break

    # Z-logical rows (k..2k-1): U(Z_q) exactly matches each target row for selector q.
    # Fix sign by prepending X_q (anticommutes with Z_q) → x_correction[q].
    for row_idx in range(k, 2 * k):
        tx, tz = target_x[row_idx], target_z[row_idx]
        for q in range(n):
            if np.array_equal(z2x[q], tx) and np.array_equal(z2z[q], tz):
                if zs_sign[q] ^ target_signs[row_idx]:
                    x_correction[q] ^= 1
                break

    # Stabilizer rows (2k..n+k-1): pivot qubits.
    # The circuit's stabilizer generators and the target's may span the same group
    # but use different GF(2) bases. Row-reduce both to the same canonical form;
    # the canonical sign vectors must agree, and the correction is found by solving
    # R_A * x_corr = (s_A_can XOR s_B_can) where R_A is the circuit's reduction matrix.

    pivot_qubits = [q for q in range(n) if q not in selector_qubits]
    num_stab = n - k

    if num_stab > 0:
        target_stab_x = target_x[2 * k :]
        target_stab_z = target_z[2 * k :]
        target_stab_signs = target_signs[2 * k :]

        circ_stab_symp = np.hstack([z2x[pivot_qubits], z2z[pivot_qubits]])  # (num_stab x 2n)
        circ_stab_sign = zs_sign[np.array(pivot_qubits)]
        targ_stab_symp = np.hstack([target_stab_x, target_stab_z])  # (num_stab x 2n)

        _, r_circ = _gf2_rref_track(circ_stab_symp)
        _, r_targ = _gf2_rref_track(targ_stab_symp)

        s_circ_can = r_circ @ circ_stab_sign % 2
        s_targ_can = r_targ @ target_stab_signs % 2

        phase_diff = (s_circ_can ^ s_targ_can).astype(np.int8)
        if np.any(phase_diff):
            aug = np.hstack([r_circ, phase_diff.reshape(-1, 1)])
            ns = mod2.nullspace(aug)
            for vec in ns:
                if vec[-1] == 1:
                    for idx, q in enumerate(pivot_qubits):
                        x_correction[q] ^= int(vec[idx])
                    break

    if np.all(x_correction == 0) and np.all(z_correction == 0):
        return circuit

    corrected_circuit = stim.Circuit()

    for q in range(n):
        xv = x_correction[q]
        zv = z_correction[q]
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
    max_two_qubit_gates: int | None = None,
    timeout: int | None = None,
    use_exponential_backoff: bool = False,
    min_timeout: int = 1,
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
        encoding = CliffordDepthEncoding(gate_set, allow_qubit_permutation, use_symmetry_breaking, max_two_qubit_gates)
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

    if use_exponential_backoff:
        return _search_with_exponential_backoff(
            encoding,
            target,
            lower_bound,
            upper_bound,
            k,
            verify_fn,
            is_depth,
            verify,
            postprocess=postprocess,
            min_timeout=min_timeout,
            max_timeout=timeout if timeout is not None else 3600,
        )
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
    max_two_qubit_gates: int | None = None,
    timeout: int | None = None,
    use_exponential_backoff: bool = False,
    min_timeout: int = 1,
) -> SynthesisResult:
    """Synthesize CSS CNOT circuit."""
    target, k, m_x = _prepare_css_target(checks, target_kind, x_logicals, z_logicals)

    if objective == Objective.GATE_COUNT:
        encoding: SynthesisEncoding = CSSGateCountEncoding(gate_set, m_x, use_symmetry_breaking)
        is_depth = False
    else:
        encoding = CSSDepthEncoding(gate_set, m_x, use_symmetry_breaking, max_two_qubit_gates)
        is_depth = True

    def verify_fn(circuit: CliffordIsometry | CNOTCircuit) -> bool:
        if not isinstance(circuit, CNOTCircuit):
            return False
        if target_kind == TargetKind.CSS_STATE:
            return verify_css_state(circuit, checks)
        return verify_css_isometry(circuit, checks, x_logicals if checks.type == "X" else z_logicals, k)

    if use_exponential_backoff:
        return _search_with_exponential_backoff(
            encoding,
            target,
            lower_bound,
            upper_bound,
            k,
            verify_fn,
            is_depth,
            verify,
            min_timeout=min_timeout,
            max_timeout=timeout if timeout is not None else 3600,
        )
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
