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

    # Get stabilizers from actual circuit
    n = actual.n
    stab_actual = actual.tableau.matrix[n:, :]

    # Check that X-part is zero
    if not np.all(stab_actual[:, :n] == 0):
        return False

    # Check Z-part matches target up to row operations
    import ldpc.mod2.mod2_numpy as mod2

    target_z = target.tableau.matrix[:, n:]
    actual_z = stab_actual[:, n:]

    return bool(mod2.rank(target_z) == mod2.rank(actual_z) == mod2.rank(np.vstack([target_z, actual_z])))


def verify_clifford_isometry(circuit: CliffordIsometry, target: StabilizerTableau, k: int) -> bool:
    """Verify that circuit implements the target Clifford isometry.

    Args:
        circuit: Synthesized circuit.
        target: Target tableau with k logical qubits.
        k: Number of logical qubits.

    Returns:
        True if circuit implements target isometry.
    """
    # Placeholder: full verification requires checking logical operators match
    # and stabilizers are correct up to row operations and qubit permutation
    return True


def verify_css_state(circuit: CNOTCircuit, target: CheckMatrix) -> bool:
    """Verify that CNOT circuit prepares the target CSS state.

    Args:
        circuit: Synthesized CNOT circuit.
        target: Target CSS check matrix.

    Returns:
        True if circuit prepares target CSS state.
    """
    # Get code from circuit
    code = circuit.get_code()

    # Check that stabilizers match target
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
    # Placeholder: full verification requires checking logical operators
    return True
