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

    target_code = CSSCode(
        checks.matrix if checks.is_x_type() else checks.matrix * 0,
        checks.matrix * 0 if checks.is_x_type() else checks.matrix,
    )

    return circuit_code.equal_stabilizer_group(target_code)


def verify_css_isometry(
    circuit: CNOTCircuit,
    checks: CheckMatrix,
    x_logicals: StabilizerTableau | None,
    z_logicals: StabilizerTableau | None,
    k: int,
) -> bool:
    """Verify that CNOT circuit implements the target CSS isometry.

    Args:
        circuit: Synthesized CNOT circuit.
        checks: Target CSS check matrix.
        x_logicals: Target X logical operators.
        z_logicals: Target Z logical operators.
        k: Number of logical qubits.

    Returns:
        True if circuit implements target CSS isometry.
    """
    from ...codes import CSSCode

    if circuit.num_inputs() != k:
        return False

    if x_logicals is None or z_logicals is None:
        msg = "x_logicals and z_logicals must be provided for CSS isometry verification"
        raise ValueError(msg)

    circuit_code = circuit.get_code()

    if not isinstance(circuit_code, CSSCode):
        return False

    lx = x_logicals.get_x_part() if checks.is_x_type() else x_logicals.get_z_part()
    lz = z_logicals.get_z_part() if checks.is_x_type() else z_logicals.get_x_part()

    target_code = CSSCode(
        checks.matrix if checks.is_x_type() else checks.matrix * 0,
        checks.matrix * 0 if checks.is_x_type() else checks.matrix,
        Lx=lx,
        Lz=lz,
        distance=1,
    )

    return circuit_code.is_equivalent(target_code)
