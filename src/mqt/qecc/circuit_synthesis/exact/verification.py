# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent verification of synthesized circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...codes import StabilizerCode
from ...codes.pauli import CheckMatrix, Pauli, StabilizerTableau

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..circuits import CliffordIsometry, CNOTCircuit


def verify_clifford_unitary(circuit: CliffordIsometry, target: StabilizerTableau) -> bool:
    """Verify that circuit implements the target Clifford unitary up to input-qubit permutation.

    The SAT encoding with allow_qubit_permutation=True finds circuits that implement
    the target Clifford composed with some input-qubit permutation.  A permuted
    implementation is still valid (relabelling input wires costs no gates), so we
    check whether there exists a consistent permutation that maps actual rows to
    target rows with matching phases.

    Args:
        circuit: Synthesized circuit.
        target: Target Clifford unitary tableau (2n rows: n X-images then n Z-images).

    Returns:
        True if circuit implements the target Clifford up to input-qubit permutation.
    """
    actual = StabilizerTableau.from_stim_circuit(circuit.to_stim_circuit(with_resets=False))

    if actual.n != target.n or actual.n_rows != target.n_rows:
        return False

    n = actual.n
    if actual.n_rows != 2 * n:
        return actual == target

    act_x: npt.NDArray[np.int8] = actual.tableau.matrix[:n]
    act_z: npt.NDArray[np.int8] = actual.tableau.matrix[n:]
    act_xp: npt.NDArray[np.int8] = actual.phase[:n]
    act_zp: npt.NDArray[np.int8] = actual.phase[n:]

    tgt_x: npt.NDArray[np.int8] = target.tableau.matrix[:n]
    tgt_z: npt.NDArray[np.int8] = target.tableau.matrix[n:]
    tgt_xp: npt.NDArray[np.int8] = target.phase[:n]
    tgt_zp: npt.NDArray[np.int8] = target.phase[n:]

    # Find permutation π such that actual.X[q] == target.X[π(q)] for each q.
    perm: dict[int, int] = {}
    for q in range(n):
        for r in range(n):
            if np.array_equal(act_x[q], tgt_x[r]) and act_xp[q] == tgt_xp[r]:
                perm[q] = r
                break
        else:
            return False

    if len(set(perm.values())) != n:
        return False

    # Verify Z-images obey the same permutation.
    return all(np.array_equal(act_z[q], tgt_z[perm[q]]) and act_zp[q] == tgt_zp[perm[q]] for q in range(n))


def verify_stabilizer_state(circuit: CliffordIsometry, stabilizers: StabilizerTableau) -> bool:
    """Verify that circuit prepares the target stabilizer state.

    Args:
        circuit: Synthesized circuit.
        stabilizers: Target stabilizer generators.

    Returns:
        True if circuit prepares target state.
    """
    if not circuit.is_state():
        return False

    circuit_code = circuit.get_code()
    target_code = StabilizerCode(stabilizers)

    return circuit_code.equal_stabilizer_group(target_code)


def verify_clifford_isometry(
    circuit: CliffordIsometry,
    target: StabilizerTableau,
    k: int,
) -> bool:
    """Verify that circuit implements the target Clifford isometry.

    Args:
        circuit: Synthesized circuit.
        target: Combined target tableau with logicals and stabilizers.
        k: Number of logical qubits.

    Returns:
        True if circuit implements target isometry.
    """
    if circuit.num_inputs() != k:
        return False

    num_rows = target.n_rows
    expected_rows = 2 * k + (target.n - k)

    if num_rows != expected_rows:
        return False

    x_logicals = StabilizerTableau(target.tableau.matrix[:k, :], target.phase[:k])
    z_logicals = StabilizerTableau(target.tableau.matrix[k : 2 * k, :], target.phase[k : 2 * k])
    stabilizers = StabilizerTableau(target.tableau.matrix[2 * k :, :], target.phase[2 * k :])

    circuit_code = circuit.get_code()
    target_code = StabilizerCode(stabilizers, x_logicals=x_logicals, z_logicals=z_logicals)

    return circuit_code.is_equivalent(target_code)


def verify_css_state(circuit: CNOTCircuit, checks: CheckMatrix) -> bool:
    """Verify that CNOT circuit prepares the target CSS state.

    Args:
        circuit: Synthesized CNOT circuit.
        checks: Target CSS check matrix.

    Returns:
        True if circuit prepares target CSS state.
    """
    if not circuit.is_state():
        return False

    circuit_code = circuit.get_code()
    h_circ = circuit_code.Hx if checks.is_x_type() else circuit_code.Hz

    return checks.equ_span(h_circ)


def verify_css_isometry(
    circuit: CNOTCircuit,
    checks: CheckMatrix,
    logicals: CheckMatrix | StabilizerTableau | None,
    k: int,
) -> bool:
    """Verify that CNOT circuit implements the target CSS isometry.

    Args:
        circuit: Synthesized CNOT circuit.
        checks: Target CSS check matrix.
        logicals: Target logical operators (CheckMatrix matching check type, or StabilizerTableau).
        k: Number of logical qubits.

    Returns:
        True if circuit implements target CSS isometry.
    """
    if circuit.num_inputs() != k:
        return False

    if logicals is None:
        msg = "logicals must be provided for CSS isometry verification"
        raise ValueError(msg)

    circuit_code = circuit.get_code()
    h_circ = circuit_code.Hx if checks.is_x_type() else circuit_code.Hz

    if not checks.equ_span(h_circ):
        return False

    if isinstance(logicals, CheckMatrix):
        logical_matrix = logicals.matrix
    else:
        logical_matrix = logicals.get_x_part() if checks.is_x_type() else logicals.get_z_part()

    if logical_matrix.shape[0] != k:
        return False

    for i in range(k):
        logical_row = logical_matrix[i]
        pauli_str = "".join("I" if val == 0 else ("X" if checks.is_x_type() else "Z") for val in logical_row)
        pauli = Pauli.from_pauli_string(pauli_str)

        if checks.is_x_type():
            if not circuit_code.is_x_logical(pauli):
                return False
        elif not circuit_code.is_z_logical(pauli):
            return False

    return True
