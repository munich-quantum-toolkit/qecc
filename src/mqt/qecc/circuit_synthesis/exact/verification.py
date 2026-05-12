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

if TYPE_CHECKING:
    from ...codes.pauli import CheckMatrix, StabilizerTableau
    from ..circuits import CliffordIsometry, CNOTCircuit


def verify_clifford_unitary(circuit: CliffordIsometry, target: StabilizerTableau) -> bool:
    """Verify that circuit implements the target Clifford unitary.

    Args:
        circuit: Synthesized circuit.
        target: Target Clifford unitary tableau.

    Returns:
        True if circuit matches target.
    """
    from ...codes.pauli import StabilizerTableau

    actual = StabilizerTableau.from_stim_circuit(circuit.to_stim_circuit(with_resets=False))

    if actual.n != target.n or actual.n_rows != target.n_rows:
        return False

    return actual == target


def verify_stabilizer_state(circuit: CliffordIsometry, target: StabilizerTableau) -> bool:
    """Verify that circuit prepares the target stabilizer state.

    Args:
        circuit: Synthesized circuit.
        target: Target stabilizer generators (n x 2n).

    Returns:
        True if circuit prepares target state.
    """
    from ...codes.pauli import StabilizerTableau

    actual = StabilizerTableau.from_stim_circuit(circuit.to_stim_circuit(with_resets=True))

    if actual.n != target.n:
        return False

    actual_stabs = actual.tableau.matrix[actual.n :]
    target_stabs = target.tableau.matrix

    if actual_stabs.shape[0] != target_stabs.shape[0]:
        return False

    return _check_stabilizer_equivalence(actual_stabs, target_stabs)


def verify_clifford_isometry(circuit: CliffordIsometry, target: StabilizerTableau, k: int) -> bool:
    """Verify that circuit implements the target Clifford isometry.

    Args:
        circuit: Synthesized circuit.
        target: Target tableau with k logical qubits.
        k: Number of logical qubits.

    Returns:
        True if circuit implements target isometry.
    """
    from ...codes.pauli import StabilizerTableau

    actual = StabilizerTableau.from_stim_circuit(circuit.to_stim_circuit(with_resets=True))

    if actual.n != target.n:
        return False

    n = actual.n
    m = target.n_rows - 2 * k

    actual_logicals_x = actual.tableau.matrix[:k]
    actual_logicals_z = actual.tableau.matrix[k : 2 * k]
    actual_stabs = actual.tableau.matrix[2 * k :]

    target_logicals_x = target.tableau.matrix[:k]
    target_logicals_z = target.tableau.matrix[k : 2 * k]
    target_stabs = target.tableau.matrix[2 * k :]

    if actual_stabs.shape[0] != m or target_stabs.shape[0] != m:
        return False

    if not _check_stabilizer_equivalence(actual_stabs, target_stabs):
        return False

    return _check_logical_equivalence(actual_logicals_x, actual_logicals_z, target_logicals_x, target_logicals_z, n)


def verify_css_state(circuit: CNOTCircuit, target: CheckMatrix) -> bool:
    """Verify that CNOT circuit prepares the target CSS state.

    Args:
        circuit: Synthesized CNOT circuit.
        target: Target CSS check matrix.

    Returns:
        True if circuit prepares target CSS state.
    """
    code = circuit.get_code()

    import ldpc.mod2.mod2_numpy as mod2

    actual = code.Hx if target.is_x_type() else code.Hz

    return bool(mod2.rank(target.matrix) == mod2.rank(actual) == mod2.rank(np.vstack([target.matrix, actual])))


def verify_css_isometry(circuit: CNOTCircuit, target: CheckMatrix, k: int) -> bool:
    """Verify that CNOT circuit implements the target CSS isometry.

    Args:
        circuit: Synthesized CNOT circuit.
        target: Target CSS check matrix.
        k: Number of logical qubits.

    Returns:
        True if circuit implements target CSS isometry.
    """
    code = circuit.get_code()

    import ldpc.mod2.mod2_numpy as mod2

    actual = code.Hx if target.is_x_type() else code.Hz

    n = target.num_qubits()
    m = target.num_rows() - k

    if actual.shape[0] != m:
        return False

    return bool(mod2.rank(target.matrix[k:]) == mod2.rank(actual) == mod2.rank(np.vstack([target.matrix[k:], actual])))


def _check_stabilizer_equivalence(actual: np.ndarray, target: np.ndarray) -> bool:
    """Check if two sets of stabilizers generate the same stabilizer group."""
    import ldpc.mod2.mod2_numpy as mod2

    if actual.shape != target.shape:
        return False

    rank_actual = mod2.rank(actual)
    rank_target = mod2.rank(target)

    if rank_actual != rank_target:
        return False

    combined = np.vstack([actual, target])
    rank_combined = mod2.rank(combined)

    return rank_combined == rank_actual


def _check_logical_equivalence(
    actual_x: np.ndarray, actual_z: np.ndarray, target_x: np.ndarray, target_z: np.ndarray, n: int
) -> bool:
    """Check if logical operators are equivalent up to qubit permutation and stabilizers."""
    import ldpc.mod2.mod2_numpy as mod2

    k = actual_x.shape[0]

    if target_x.shape[0] != k or actual_z.shape[0] != k or target_z.shape[0] != k:
        return False

    actual_logical = np.hstack([actual_x, actual_z])
    target_logical = np.hstack([target_x, target_z])

    if mod2.rank(actual_logical) != k or mod2.rank(target_logical) != k:
        return False

    combined = np.vstack([actual_logical, target_logical])
    return mod2.rank(combined) == k
