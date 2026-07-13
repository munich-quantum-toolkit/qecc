# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods for synthesizing encoding circuits for CSS codes."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import stim

from ..codes import CSSCode
from ..codes.pauli import CheckMatrix, StabilizerTableau, complete_stabilizer_tableau_with_destabilizers
from .circuits import CliffordIsometry, CNOTCircuit
from .exact import (
    Objective,
    SynthesisStatus,
    TargetKind,
    get_clifford_extended_gate_set,
    synthesize_isometry_exact,
)
from .operations import CNOT
from .synthesis import SynthesisConfig, synthesize_cnot, synthesize_non_css
from .transvection import lexicographical_compare_np, score_symplectic

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from ..codes import StabilizerCode


logger = logging.getLogger(__name__)


def depth_optimal_encoding_circuit_non_css(
    code: StabilizerCode,
    max_depth: int,
    min_depth: int = 1,
    max_two_qubit_gates: int | None = None,
) -> CliffordIsometry | str:
    """Synthesize a depth-optimal encoding circuit for a (possibly non-CSS) stabilizer code.

    Thin wrapper around :func:`~mqt.qecc.circuit_synthesis.exact.synthesize_isometry_exact`
    (``CLIFFORD_ISOMETRY`` / ``DEPTH`` over the extended gate set); see the Exact Circuit
    Synthesis guide for the full, general interface.

    Args:
        code: The stabilizer code to synthesize an encoding circuit for.
        max_depth: Maximum circuit depth to search up to.
        min_depth: Minimum circuit depth to start the search from. Raising this skips the
            (often expensive) proofs that no shallower circuit exists.
        max_two_qubit_gates: Optional upper bound on the total number of two-qubit gates.

    Returns:
        The encoding circuit as a :class:`CliffordIsometry`, or ``"UNSAT"`` if no circuit exists
        in ``[min_depth, max_depth]`` (or a diagnostic string if the search is inconclusive).
    """
    assert code.x_logicals is not None
    assert code.z_logicals is not None
    result = synthesize_isometry_exact(
        target=code.generators,
        target_kind=TargetKind.CLIFFORD_ISOMETRY,
        objective=Objective.DEPTH,
        lower_bound=min_depth,
        upper_bound=max_depth,
        x_logicals=code.x_logicals,
        z_logicals=code.z_logicals,
        gate_set=get_clifford_extended_gate_set(),
        use_symmetry_breaking=True,
        max_two_qubit_gates=max_two_qubit_gates,
    )
    if result.status == SynthesisStatus.SUCCESS:
        assert isinstance(result.circuit, CliffordIsometry)
        return result.circuit
    if result.status == SynthesisStatus.UNSAT:
        return "UNSAT"
    return f"UNKNOWN: {result.message}"


def _optimal_css_encoding_circuit(
    code: CSSCode,
    objective: Objective,
    lower_bound: int,
    upper_bound: int,
    min_timeout: int,
    max_timeout: int,
) -> CNOTCircuit | None:
    """Synthesize an optimal CSS encoding circuit via exact synthesis.

    Shared implementation for the gate- and depth-optimal CSS encoders; delegates to
    :func:`~mqt.qecc.circuit_synthesis.exact.synthesize_isometry_exact` (see the Exact Circuit
    Synthesis guide for the full, general interface). Uses the check matrix with the fewest
    rows for efficiency and returns ``None`` when no circuit exists within the bounds/timeout.
    """
    checks, logicals = _get_matrix_with_fewest_checks(code)
    result = synthesize_isometry_exact(
        target=checks,
        target_kind=TargetKind.CSS_ISOMETRY,
        objective=objective,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        x_logicals=logicals if checks.type == "X" else None,
        z_logicals=logicals if checks.type == "Z" else None,
        use_exponential_backoff=True,
        min_timeout=min_timeout,
        timeout=max_timeout,
    )
    if result.status != SynthesisStatus.SUCCESS:
        return None
    assert isinstance(result.circuit, CNOTCircuit)
    return result.circuit


def gate_optimal_encoding_circuit(
    code: CSSCode,
    min_gates: int = 1,
    max_gates: int = 10,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> CNOTCircuit | None:
    """Synthesize an encoding circuit for the given CSS code using the minimal number of gates.

    Args:
        code: The CSS code to synthesize the encoding circuit for.
        min_gates: The minimum number of gates to use in the circuit.
        max_gates: The maximum number of gates to use in the circuit.
        min_timeout: The minimum time to spend on the synthesis.
        max_timeout: The maximum time to spend on the synthesis.

    Returns:
        The synthesized encoding circuit and the qubits that are used to encode the logical qubits.
    """
    return _optimal_css_encoding_circuit(code, Objective.GATE_COUNT, min_gates, max_gates, min_timeout, max_timeout)


def depth_optimal_encoding_circuit(
    code: CSSCode,
    min_depth: int = 1,
    max_depth: int = 10,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> CNOTCircuit | None:
    """Synthesize an encoding circuit for the given CSS code using minimal depth.

    Args:
        code: The CSS code to synthesize the encoding circuit for.
        min_depth: The minimum number of gates to use in the circuit.
        max_depth: The maximum number of gates to use in the circuit.
        min_timeout: The minimum time to spend on the synthesis.
        max_timeout: The maximum time to spend on the synthesis.

    Returns:
        The synthesized encoding circuit and the qubits that are used to encode the logical qubits.
    """
    return _optimal_css_encoding_circuit(code, Objective.DEPTH, min_depth, max_depth, min_timeout, max_timeout)


def _get_matrix_with_fewest_checks(code: CSSCode) -> tuple[CheckMatrix, CheckMatrix]:
    """Return the stabilizer matrix with the fewest checks, the corresponding logicals and a bool indicating whether X- or Z-checks have been returned."""
    use_x_checks = code.Hx.shape[0] < code.Hz.shape[0]
    checks = code.Hx if use_x_checks else code.Hz
    logicals = code.Lx if use_x_checks else code.Lz
    type_ = "X" if use_x_checks else "Z"
    return CheckMatrix(checks, type_), CheckMatrix(logicals, type_)


def gottesman_encoding_circuit(tableau: StabilizerTableau | Sequence[str]) -> CliffordIsometry:
    """Synthesize encoding circuit for a stabilizer code as described in chapter 6.4 of Gottesman's book.

    Assumes all signs of the stabilizers are +1.

    Args:
        tableau: The stabilizer tableau of the code to synthesize the encoding circuit for.

    Returns:
        stim circuit implementing the encoding and a list of qubits that are used to encode the logical qubits.
    """
    if isinstance(tableau, Sequence):
        tableau = StabilizerTableau.from_pauli_strings(tableau)  # ty: ignore[invalid-argument-type]

    nq = tableau.n
    mat = tableau.tableau.matrix.copy()
    x_part = mat[:, :nq]
    z_part = mat[:, nq:]

    circ = stim.Circuit()
    n_rows = mat.shape[0]

    initialized = []
    for row in range(n_rows):
        # find row with either x_part[row][i] = 1 or z_part[row][i] = 1
        pivot = row
        column = row

        while column < nq and x_part[pivot][column] != 1 and z_part[pivot][column] != 1:
            found_pivot = False
            for p in range(row, n_rows):
                if x_part[p][column] == 1 or z_part[p][column] == 1:
                    pivot = p
                    found_pivot = True
                    break
            if not found_pivot:
                column += 1
                pivot = row
        if column >= nq:
            # No valid pivot found, invalid tableau
            msg = "Invalid tableau: could not find a valid pivot."
            raise ValueError(msg)
        initialized.append(column)
        # swap to row i
        t = x_part[pivot].copy()
        x_part[pivot] = x_part[row]
        x_part[row] = t

        t = z_part[pivot].copy()
        z_part[pivot] = z_part[row]
        z_part[row] = t

        if x_part[row][column] == 0:
            circ.append("H", [column])
            t = x_part[:, column].copy()
            x_part[:, column] = z_part[:, column]
            z_part[:, column] = t

        # reduce column
        for q in np.where(x_part[row])[0]:
            if q == column:
                continue
            circ.append("CX", [column, q])
            x_part[:, q] ^= x_part[:, column]
            z_part[:, column] ^= z_part[:, q]

        if z_part[row][column] == 1:
            circ.append("S", [column])
            z_part[:, column] ^= x_part[:, column]

        for q in np.where(z_part[row])[0]:
            if q == column:
                continue
            circ.append("CZ", [column, q])
            z_part[:, q] ^= x_part[:, column]
            z_part[:, column] ^= x_part[:, q]

        # reduce stabilizers below row
        x_part[:, column] = 0
        x_part[row, column] = 1

    circ.append("H", initialized)
    circ = circ.inverse()

    signs = [s.sign for s in circ.to_tableau().to_stabilizers()]
    for row, sign in enumerate(signs):
        if sign == -1:
            circ.insert(0, stim.CircuitInstruction("X", [row]))
    iso = CliffordIsometry.from_stim_circuit(circ)
    for q in initialized:
        iso.initialize_qubit(q, basis="Z")
    return iso


def synthesize_clifford(
    tableau: StabilizerTableau,
    use_cnots_if_css: bool = True,
    config: SynthesisConfig | None = None,
) -> CliffordIsometry:
    """Synthesize a stim circuit implementing a Clifford operation to minimize two-qubit gate count.

    Args:
        tableau: The stabilizer tableau representing the Clifford operation to synthesize.
        use_cnots_if_css: Whether to use CNOT-only synthesis if the tableau is CSS.
        config: Configuration options for the synthesis process.

    Returns:
        A CliffordIsometry representing the synthesized Clifford operation that implements
        the same operation as the input tableau. The synthesis aims to minimize the two-qubit
        gate count. If the tableau is CSS and use_cnots_if_css is True, the circuit uses only
        CNOT gates; otherwise, a general Clifford synthesis is performed.
    """
    if tableau.is_css() and use_cnots_if_css:
        x_checks, z_checks = tableau.to_css()
        assert isinstance(config, SynthesisConfig) or config is None, (
            "CNOTSynthesisConfig must be provided when use_cnots_if_css is True."
        )
        logicals = x_checks if x_checks.num_rows() <= z_checks.num_rows() else z_checks
        return cnot_encoding_circuit(
            CheckMatrix(np.empty((0, tableau.n), dtype=np.int8), pauli_type=logicals.type),
            logicals,
            config=config,
        )

    assert isinstance(config, SynthesisConfig) or config is None, (
        "CliffordSynthesisConfig must be provided when use_cnots_if_css is False."
    )
    ops, _ = synthesize_non_css(
        tableau,
        config=config,
    )
    return CliffordIsometry.from_stim_circuit(ops.to_circuit_inverse())


def synthesize_encoding_circuit(
    code: StabilizerCode,
    config: SynthesisConfig | None = None,
    use_cnots_if_css: bool = True,
    fixed_logical_qubits: dict[int, str] | None = None,
) -> CliffordIsometry:
    """Synthesize an encoding circuit for the given stabilizer code.

    Args:
        code: The stabilizer code to synthesize the encoding circuit for.
        config: Configuration options for the synthesis process.
        use_cnots_if_css: Whether to use CNOT-only synthesis if the code is CSS.
        fixed_logical_qubits: Dictionary mapping logical qubit indices to their desired states ('0' or '+') in the circuit.

    Returns:
        A CliffordIsometry that implements the encoding circuit for the given stabilizer code.
    """
    if fixed_logical_qubits:
        if not all(0 <= q < code.k and z in {"0", "+"} for q, z in fixed_logical_qubits.items()):
            msg = "Fixed logical qubit indices must be in the range [0, k-1] and states must be '0' or '+'."
            raise ValueError(msg)

        additional_x_checks = sorted(q for q, z in fixed_logical_qubits.items() if z == "+")
        additional_z_checks = sorted(q for q, z in fixed_logical_qubits.items() if z == "0")
        additional_checks = additional_x_checks + additional_z_checks
    else:
        additional_x_checks = []
        additional_z_checks = []
        additional_checks = []

    if use_cnots_if_css and isinstance(code, CSSCode):
        x_checks = CheckMatrix(np.vstack((code.Hx, code.Lx[additional_x_checks])), pauli_type="X")
        z_checks = CheckMatrix(np.vstack((code.Hz, code.Lz[additional_z_checks])), pauli_type="Z")
        x_logicals = CheckMatrix(np.delete(code.Lx, additional_checks, axis=0), pauli_type="X")
        z_logicals = CheckMatrix(np.delete(code.Lz, additional_checks, axis=0), pauli_type="Z")

        checks, logicals = (
            (x_checks, x_logicals) if x_checks.num_rows() <= z_checks.num_rows() else (z_checks, z_logicals)
        )

        assert isinstance(config, SynthesisConfig) or config is None, (
            "CNOTSynthesisConfig must be provided when use_cnots_if_css is True."
        )
        return cnot_encoding_circuit(checks, logicals, config=config)

    assert isinstance(config, SynthesisConfig) or config is None, (
        "CliffordSynthesisConfig must be provided when use_cnots_if_css is False."
    )

    gens_mat: npt.NDArray[np.int8] = np.vstack((
        code.symplectic,
        code.x_logicals.tableau.matrix[additional_x_checks],
        code.z_logicals.tableau.matrix[additional_z_checks],
    ))
    gens_phase: npt.NDArray[np.int8] = np.hstack((
        code.generators.phase,
        np.zeros(len(additional_x_checks), dtype=np.int8),
        np.zeros(len(additional_z_checks), dtype=np.int8),
    ))
    log_mat: npt.NDArray[np.int8] = np.vstack((
        np.delete(code.x_logicals.tableau.matrix, additional_checks, axis=0),
        np.delete(code.z_logicals.tableau.matrix, additional_checks, axis=0),
    ))
    log_phase: npt.NDArray[np.int8] = np.hstack((
        np.delete(code.x_logicals.phase, additional_checks, axis=0),
        np.delete(code.z_logicals.phase, additional_checks, axis=0),
    ))

    return encoder_from_stabilizers_and_logicals(
        StabilizerTableau(gens_mat, gens_phase), StabilizerTableau(log_mat, log_phase), config=config
    )


def resynthesize_stim_circuit(
    circ: stim.Circuit,
    use_cnots_if_css: bool = True,
    config: SynthesisConfig | None = None,
) -> stim.Circuit:
    """Resynthesize a stim circuit implementing a Clifford operation to minimize two-qubit gate count.

    Args:
        circ: The stim.Circuit to resynthesize.
        use_cnots_if_css: Whether to use CNOT-only synthesis if the circuit is CSS.
        config: Configuration options for the synthesis process.

    Returns:
        A stim.Circuit that implements the same operation as the input circuit but with potentially fewer two
    """
    tableau = StabilizerTableau.from_stim_circuit(circ)
    return synthesize_clifford(tableau, use_cnots_if_css=use_cnots_if_css, config=config).to_stim_circuit()


def encoder_from_stabilizers_and_logicals(
    stabilizers: StabilizerTableau,
    logicals: StabilizerTableau,
    optimize_tableau_before_synthesis: bool = True,
    config: SynthesisConfig | None = None,
) -> CliffordIsometry:
    """Synthesize an encoding circuit for a stabilizer code given its stabilizers and logicals as tableaux.

    Args:
        stabilizers: A tableau representing the stabilizers of the code.
        logicals: A tableau representing the logical operators of the code.
        optimize_tableau_before_synthesis: Whether to perform row operations on the combined tableau to optimize it for synthesis before synthesizing the circuit.
        config: Configuration options for the synthesis process.

    Returns:
        A CliffordIsometry that implements the encoding circuit for the given stabilizer code.
    """
    if stabilizers.n != logicals.n:
        msg = "Stabilizers and logicals must have the same number of qubits."
        raise ValueError(msg)
    if stabilizers.num_rows() + logicals.num_rows() > stabilizers.n * 2:
        msg = "The total number of stabilizers and logicals must be less than or equal to 2n."
        raise ValueError(msg)

    full_tableau = combine_stabilizer_and_logical_tableau(stabilizers, logicals)
    stab_indices = list(range(logicals.num_rows() // 2, logicals.num_rows() // 2 + stabilizers.num_rows()))
    if optimize_tableau_before_synthesis:
        optimized_tableau = optimize_tableau(full_tableau, stab_rows=stab_indices)
    else:
        optimized_tableau = full_tableau

    iso = synthesize_clifford(
        optimized_tableau,
        use_cnots_if_css=False,
        config=config,
    )
    iso.initialize_qubits(stab_indices, basis="Z")
    return iso


def optimize_tableau(tableau: StabilizerTableau, stab_rows: list[int]) -> StabilizerTableau:
    """Optimize a stabilizer tableau by performing row operations to reduce the cost of the initial tableau for synthesis."""
    tab = tableau.copy()

    best = (tab, score_symplectic(tab)[0])
    improved = True
    half = tableau.num_rows() // 2
    x_logical_rows = [i for i in range(half) if i not in stab_rows]
    z_logical_rows = [i + half for i in range(half) if i not in stab_rows]
    logical_rows = x_logical_rows + z_logical_rows
    k = len(logical_rows) // 2
    while improved:
        improved = False
        for i in range(len(stab_rows)):
            for j in range(len(stab_rows)):
                if i == j:
                    continue
                tab = tableau.copy()
                mat = tab.tableau.matrix
                destabs = mat[:half][stab_rows]
                stabs = mat[half:][stab_rows]
                stabs[i] ^= stabs[j]
                destabs[j] ^= destabs[i]
                mat[:half][stab_rows] = destabs
                mat[half:][stab_rows] = stabs
                new_score, _ = score_symplectic(StabilizerTableau(mat, tableau.phase.copy()))
                if lexicographical_compare_np(new_score, best[1]):
                    best = (tab, new_score)
                    improved = True
            for j in range(len(logical_rows)):
                tab = tableau.copy()
                mat = tab.tableau.matrix
                destabs = mat[:half][stab_rows]
                stabs = mat[half:][stab_rows]

                other_log = mat[logical_rows[(j + k) % (2 * k)]]

                destabs[i] ^= other_log
                logj = mat[logical_rows[j]]

                logj ^= stabs[i]
                mat[:half][stab_rows] = destabs
                mat[logical_rows[j]] = logj
                new_score, _ = score_symplectic(StabilizerTableau(mat, tableau.phase.copy()))
                if lexicographical_compare_np(new_score, best[1]):
                    best = (tab, new_score)
                    improved = True
        tableau = best[0]

    return best[0]


def combine_stabilizer_and_logical_tableau(
    stabilizers: StabilizerTableau, logicals: StabilizerTableau
) -> StabilizerTableau:
    """Combine a stabilizer tableau and a logical tableau, then complete with destabilizers.

    Args:
        stabilizers: A tableau representing the stabilizers of the code (without destabilizers).
        logicals: A tableau containing logical operators.

    Returns:
        A combined tableau with destabilizers added, suitable for circuit synthesis.
    """
    if stabilizers.n != logicals.n:
        msg = "Stabilizers and logicals must act on the same number of qubits."
        raise ValueError(msg)

    m = stabilizers.num_rows()

    # Combine stabilizers and logicals into a single tableau
    x_logicals = logicals.tableau.matrix[: logicals.num_rows() // 2]
    z_logicals = logicals.tableau.matrix[logicals.num_rows() // 2 :]
    x_logicals_phase = logicals.phase[: logicals.num_rows() // 2]
    z_logicals_phase = logicals.phase[logicals.num_rows() // 2 :]
    combined_matrix = np.vstack([x_logicals, z_logicals, stabilizers.tableau.matrix])

    combined_phase = np.hstack([x_logicals_phase, z_logicals_phase, stabilizers.phase])
    combined_tableau = StabilizerTableau(combined_matrix, combined_phase)

    # Complete with destabilizers for the stabilizers only
    # The stabilizer rows are at indices 0 to m-1
    stab_rows = list(range(logicals.num_rows(), logicals.num_rows() + m))
    return complete_stabilizer_tableau_with_destabilizers(combined_tableau, stab_rows)


def _remove_redundant_stabilizers(checks: CheckMatrix) -> CheckMatrix:
    """Remove redundant stabilizers from the check matrix without impacting stabilizer weight."""
    rnk = mod2.rank(checks.matrix)
    if rnk == checks.num_rows():
        return checks
    independent_checks = np.array([checks.matrix[0]])
    prev_rnk = 1
    for row in checks.matrix[1:]:
        stacked = np.vstack((independent_checks, row))
        new_rnk = mod2.rank(stacked)
        if new_rnk > prev_rnk:
            independent_checks = stacked
            prev_rnk = new_rnk
        if prev_rnk == rnk:
            break
    return CheckMatrix(independent_checks, pauli_type=checks.type)


def cnot_encoding_circuit(
    checks: CheckMatrix, logicals: CheckMatrix, config: SynthesisConfig | None = None
) -> CNOTCircuit:
    """Synthesize an encoding circuit for the given CSS code using a heuristic greedy search.

    Args:
        checks: The stabilizer check matrix of the CSS code.
        logicals: The logical operator matrix of the CSS code.
        config: The configuration for the CNOT synthesis process.

    Returns:
        The synthesized encoding circuit.
    """
    logger.info("Starting encoding circuit synthesis.")

    if config is None:
        config = SynthesisConfig()

    checks = _remove_redundant_stabilizers(checks)
    n_stab = checks.num_rows()

    if checks.type != logicals.type:
        msg = f"Check matrix and logical matrix must have the same Pauli type. Got checks.type={checks.type}, logicals.type={logicals.type}"
        raise ValueError(msg)

    if checks.num_qubits() != logicals.num_qubits():
        msg = f"Check matrix and logical matrix must have the same number of qubits. Got checks: {checks.num_qubits()} qubits, logicals: {logicals.num_qubits()} qubits"
        raise ValueError(msg)

    mat = CheckMatrix(np.vstack((checks.matrix, logicals.matrix)), pauli_type=checks.type)

    ops, reduced_checks = synthesize_cnot(mat, config=config, n_stabs=n_stab)
    assert isinstance(reduced_checks, CheckMatrix)

    cnots = [(c.control, c.target) for c in reversed(ops) if isinstance(c, CNOT)]

    x_qubits = set()
    for row in range(n_stab):
        for col in range(reduced_checks.num_qubits()):
            if reduced_checks.matrix[row, col] == 1:
                x_qubits.add(col)

    logical_qubits = set()
    for row in range(n_stab, reduced_checks.num_rows()):
        for col in range(reduced_checks.num_qubits()):
            if reduced_checks.matrix[row, col] == 1:
                logical_qubits.add(col)

    z_qubits = set(range(reduced_checks.num_qubits())) - x_qubits - logical_qubits
    if checks.type == "Z":
        z_qubits, x_qubits = x_qubits, z_qubits

    return CNOTCircuit.from_cnot_list(cnots, z_qubits, x_qubits)
