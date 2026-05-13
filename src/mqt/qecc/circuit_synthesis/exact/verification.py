# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent verification of synthesized circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def verify_stabilizer_state(circuit: CliffordIsometry, stabilizers: StabilizerTableau) -> bool:
    """Verify that circuit prepares the target stabilizer state.

    Args:
        circuit: Synthesized circuit.
        stabilizers: Target stabilizer generators.

    Returns:
        True if circuit prepares target state.
    """
    from ...codes import StabilizerCode

    if not circuit.is_state():
        return False

    circuit_code = circuit.get_code()
    target_code = StabilizerCode(stabilizers)

    return circuit_code.equal_stabilizer_group(target_code)


def verify_clifford_isometry(
    circuit: CliffordIsometry,
    stabilizers: StabilizerTableau,
    x_logicals: StabilizerTableau | None,
    z_logicals: StabilizerTableau | None,
    k: int,
) -> bool:
    """Verify that circuit implements the target Clifford isometry.

    Args:
        circuit: Synthesized circuit.
        stabilizers: Target stabilizer generators.
        x_logicals: Target X logical operators.
        z_logicals: Target Z logical operators.
        k: Number of logical qubits.

    Returns:
        True if circuit implements target isometry.
    """
    from ...codes import StabilizerCode

    if circuit.num_inputs() != k:
        return False

    if x_logicals is None or z_logicals is None:
        msg = "x_logicals and z_logicals must be provided for isometry verification"
        raise ValueError(msg)

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
    from ...codes import CSSCode

    if not circuit.is_state():
        return False

    circuit_code = circuit.get_code()

    if not isinstance(circuit_code, CSSCode):
        return False

    h_circ = circuit_code.Hx if checks.is_x_type() else circuit_code.Hz

    if h_circ is None:
        return False

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
    from ...codes import CSSCode
    from ...codes.pauli import CheckMatrix, Pauli

    if circuit.num_inputs() != k:
        return False

    if logicals is None:
        msg = "logicals must be provided for CSS isometry verification"
        raise ValueError(msg)

    circuit_code = circuit.get_code()

    if not isinstance(circuit_code, CSSCode):
        return False

    h_circ = circuit_code.Hx if checks.is_x_type() else circuit_code.Hz

    if h_circ is None:
        return False

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
