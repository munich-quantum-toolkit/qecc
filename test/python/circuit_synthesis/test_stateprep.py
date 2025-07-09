# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test synthesis of state preparation and verification circuits for FT state preparation."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import (
    depth_optimal_prep_circuit,
    gate_optimal_prep_circuit,
    gate_optimal_verification_circuit,
    gate_optimal_verification_stabilizers,
    heuristic_prep_circuit,
    heuristic_verification_circuit,
    heuristic_verification_stabilizers,
)
from mqt.qecc.codes import SquareOctagonColorCode

from .utils import eq_span, get_stabs_css, in_span

if TYPE_CHECKING:  # pragma: no cover
    from mqt.qecc.circuit_synthesis import StatePrepCircuit


@pytest.fixture(scope="session")
def steane_code() -> CSSCode:
    """Return the Steane code."""
    return CSSCode.from_code_name("Steane")


@pytest.fixture(scope="session")
def css_4_2_2_code() -> CSSCode:
    """Return the 4,2,2  code."""
    return CSSCode(np.array([[1] * 4]), np.array([[1] * 4]), 2)


@pytest.fixture(scope="session")
def css_6_2_2_code() -> CSSCode:
    """Return the 4,2,2  code."""
    return CSSCode(np.array([[1] * 6]), np.array([[1] * 6]), 2)


@pytest.fixture(scope="session")
def surface_code() -> CSSCode:
    """Return the distance 3 rotated Surface Code."""
    return CSSCode.from_code_name("surface", 3)


@pytest.fixture(scope="session")
def tetrahedral_code() -> CSSCode:
    """Return the tetrahedral code."""
    return CSSCode.from_code_name("tetrahedral")


@pytest.fixture(scope="session")
def cc_4_8_8_code() -> CSSCode:
    """Return the d=5 4,8,8 color code."""
    return SquareOctagonColorCode(5)


@pytest.fixture(scope="session")
def steane_code_sp(steane_code: CSSCode) -> StatePrepCircuit:
    """Return a non-ft state preparation circuit for the Steane code."""
    sp_circ = heuristic_prep_circuit(steane_code)
    sp_circ.compute_fault_sets()
    return sp_circ


@pytest.fixture(scope="session")
def tetrahedral_code_sp(tetrahedral_code: CSSCode) -> StatePrepCircuit:
    """Return a non-ft state preparation circuit for the tetrahedral code."""
    sp_circ = heuristic_prep_circuit(tetrahedral_code)
    sp_circ.compute_fault_sets()
    return sp_circ


@pytest.fixture(scope="session")
def color_code_d5_sp(cc_4_8_8_code: CSSCode) -> StatePrepCircuit:
    """Return a non-ft state preparation circuit for the d=5 4,8,8 color code."""
    sp_circ = heuristic_prep_circuit(cc_4_8_8_code)
    sp_circ.compute_fault_sets()
    return sp_circ


def test_heuristic_overcomplete_stabilizers() -> None:
    """Check that synthesis also works for overcomplete stabilizers."""
    code = CSSCode(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), 2)
    sp_circ = heuristic_prep_circuit(code)
    circ = sp_circ.circ
    x, z = get_stabs_css(circ)
    assert eq_span(code.Hx, x)
    assert eq_span(np.vstack((code.Hz, code.Lz)), z)


@pytest.mark.parametrize(
    "code", ["steane_code", "css_4_2_2_code", "css_6_2_2_code", "tetrahedral_code", "surface_code"]
)
def test_heuristic_prep_consistent(code: CSSCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check that heuristic_prep_circuit returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code)

    sp_circ = heuristic_prep_circuit(code)
    circ = sp_circ.circ
    max_cnots = np.sum(code.Hx) + np.sum(code.Hz)  # type: ignore[operator]

    assert circ.num_qubits == code.n
    assert circ.num_nonlocal_gates() <= max_cnots

    x, z = get_stabs_css(circ)
    assert eq_span(code.Hx, x)
    assert eq_span(np.vstack((code.Hz, code.Lz)), z)


@pytest.mark.skipif(os.getenv("CI") is not None and sys.platform == "win32", reason="Too slow for CI on Windows")
@pytest.mark.parametrize("code", ["css_4_2_2_code", "css_6_2_2_code"])
def test_gate_optimal_prep_consistent(code: CSSCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check that gate_optimal_prep_circuit returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code)
    sp_circ = gate_optimal_prep_circuit(code, max_timeout=3)
    assert sp_circ is not None
    assert sp_circ.zero_state

    circ = sp_circ.circ
    max_cnots = np.sum(code.Hx) + np.sum(code.Hz)  # type: ignore[operator]

    assert circ.num_qubits == code.n
    assert circ.num_nonlocal_gates() <= max_cnots

    x, z = get_stabs_css(circ)
    assert eq_span(code.Hx, x)
    assert eq_span(np.vstack((code.Hz, code.Lz)), z)


@pytest.mark.skipif(os.getenv("CI") is not None and sys.platform == "win32", reason="Too slow for CI on Windows")
@pytest.mark.parametrize("code", ["css_4_2_2_code", "css_6_2_2_code"])
def test_depth_optimal_prep_consistent(code: CSSCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check that depth_optimal_prep_circuit returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code)

    sp_circ = depth_optimal_prep_circuit(code, max_timeout=3)
    assert sp_circ is not None
    circ = sp_circ.circ
    max_cnots = np.sum(code.Hx) + np.sum(code.Hz)  # type: ignore[operator]

    assert circ.num_qubits == code.n
    assert circ.num_nonlocal_gates() <= max_cnots

    x, z = get_stabs_css(circ)
    assert eq_span(code.Hx, x)
    assert eq_span(np.vstack((code.Hz, code.Lz)), z)


@pytest.mark.skipif(os.getenv("CI") is not None and sys.platform == "win32", reason="Too slow for CI on Windows")
@pytest.mark.parametrize("code", ["css_4_2_2_code", "css_6_2_2_code"])
def test_plus_state_gate_optimal(code: CSSCode, request) -> None:  # type: ignore[no-untyped-def]
    """Test synthesis of the plus state."""
    code = request.getfixturevalue(code)
    sp_circ_plus = gate_optimal_prep_circuit(code, max_timeout=3, zero_state=False)

    assert sp_circ_plus is not None
    assert not sp_circ_plus.zero_state

    circ_plus = sp_circ_plus.circ
    max_cnots = np.sum(code.Hx) + np.sum(code.Hz)  # type: ignore[operator]

    assert circ_plus.num_qubits == code.n
    assert circ_plus.num_nonlocal_gates() <= max_cnots

    x, z = get_stabs_css(circ_plus)
    assert eq_span(code.Hz, z)
    assert eq_span(np.vstack((code.Hx, code.Lx)), x)

    sp_circ_zero = gate_optimal_prep_circuit(code, max_timeout=5, zero_state=True)

    assert sp_circ_zero is not None

    circ_zero = sp_circ_zero.circ
    x_zero, z_zero = get_stabs_css(circ_zero)

    if code.is_self_dual():
        assert np.array_equal(x, z_zero)
        assert np.array_equal(z, x_zero)
    else:
        assert not np.array_equal(x, z_zero)
        assert not np.array_equal(z, x_zero)


@pytest.mark.parametrize(
    "code", ["steane_code", "css_4_2_2_code", "css_6_2_2_code", "surface_code", "tetrahedral_code"]
)
def test_plus_state_heuristic(code: CSSCode, request) -> None:  # type: ignore[no-untyped-def]
    """Test synthesis of the plus state."""
    code = request.getfixturevalue(code)
    sp_circ_plus = heuristic_prep_circuit(code, zero_state=False)

    assert sp_circ_plus is not None
    assert not sp_circ_plus.zero_state

    circ_plus = sp_circ_plus.circ
    max_cnots = np.sum(code.Hx) + np.sum(code.Hz)  # type: ignore[operator]

    assert circ_plus.num_qubits == code.n
    assert circ_plus.num_nonlocal_gates() <= max_cnots

    x, z = get_stabs_css(circ_plus)
    assert eq_span(code.Hz, z)
    assert eq_span(np.vstack((code.Hx, code.Lx)), x)

    sp_circ_zero = heuristic_prep_circuit(code, zero_state=True)
    circ_zero = sp_circ_zero.circ
    x_zero, z_zero = get_stabs_css(circ_zero)

    if code.is_self_dual():
        assert np.array_equal(x, z_zero)
        assert np.array_equal(z, x_zero)
    else:
        assert not np.array_equal(x, z_zero)
        assert not np.array_equal(z, x_zero)


@pytest.mark.skipif(os.getenv("CI") is not None and sys.platform == "win32", reason="Too slow for CI on Windows")
def test_optimal_steane_verification_circuit(steane_code_sp: StatePrepCircuit) -> None:
    """Test that the optimal verification circuit for the Steane code is correct."""
    circ = steane_code_sp
    ver_stabs_layers = gate_optimal_verification_stabilizers(circ, x_errors=True, max_timeout=5)

    assert len(ver_stabs_layers) == 1  # 1 Ancilla measurement

    ver_stabs = ver_stabs_layers[0]

    assert np.sum(ver_stabs) == 3  # 3 CNOTs
    z_gens = circ.z_checks

    for stab in ver_stabs:
        assert in_span(z_gens, stab)

    errors = circ.compute_fault_set(1)
    non_detected = np.where(np.all(ver_stabs @ errors.T % 2 == 0, axis=1))[0]
    assert len(non_detected) == 0

    # Check that circuit is correct
    circ_ver = gate_optimal_verification_circuit(circ)
    assert circ_ver.num_qubits == circ.num_qubits + 1
    assert circ_ver.num_nonlocal_gates() == np.sum(ver_stabs) + circ.circ.num_nonlocal_gates()
    assert circ_ver.depth() == np.sum(ver_stabs) + circ.circ.depth() + 1  # 1 for the measurement


def test_heuristic_steane_verification_circuit(steane_code_sp: StatePrepCircuit) -> None:
    """Test that the optimal verification circuit for the Steane code is correct."""
    circ = steane_code_sp

    ver_stabs_layers = heuristic_verification_stabilizers(circ, x_errors=True)

    assert len(ver_stabs_layers) == 1  # 1 layer of verification measurements

    ver_stabs = ver_stabs_layers[0]
    assert len(ver_stabs) == 1  # 1 Ancilla measurement
    assert np.sum(ver_stabs[0]) == 3  # 3 CNOTs
    z_gens = circ.z_checks

    for stab in ver_stabs:
        assert in_span(z_gens, stab)

    errors = circ.compute_fault_set(1)
    non_detected = np.where(np.all(ver_stabs @ errors.T % 2 == 0, axis=1))[0]
    assert len(non_detected) == 0

    # Check that circuit is correct
    circ_ver = heuristic_verification_circuit(circ)
    assert circ_ver.num_qubits == circ.num_qubits + 1
    assert circ_ver.num_nonlocal_gates() == np.sum(ver_stabs) + circ.circ.num_nonlocal_gates()
    assert circ_ver.depth() == np.sum(ver_stabs) + circ.circ.depth() + 1  # 1 for the measurement


@pytest.mark.skipif(os.getenv("CI") is not None and sys.platform == "win32", reason="Too slow for CI on Windows")
def test_not_full_ft_opt_cc5(color_code_d5_sp: StatePrepCircuit) -> None:
    """Test that the optimal verification is also correct for higher distance.

    Ignore Z errors.
    Due to time constraints, we set the timeout for each search to 4 seconds.
    """
    circ = color_code_d5_sp

    ver_stabs_layers = gate_optimal_verification_stabilizers(circ, x_errors=True, max_ancillas=3, max_timeout=4)

    assert len(ver_stabs_layers) == 2  # 2 layers of verification measurements

    ver_stabs_1 = ver_stabs_layers[0]
    assert len(ver_stabs_1) == 2  # 2 Ancilla measurements
    assert np.sum(ver_stabs_1) == 9  # 9 CNOTs

    ver_stabs_2 = ver_stabs_layers[1]
    assert len(ver_stabs_2) == 3  # 2 Ancilla measurements
    assert np.sum(ver_stabs_2) <= 14  # less than 14 CNOTs (sometimes 13, sometimes 14 depending on how fast the CPU is)

    z_gens = circ.z_checks

    for stab in np.vstack((ver_stabs_1, ver_stabs_2)):
        assert in_span(z_gens, stab)

    errors_1 = circ.compute_fault_set(1)
    non_detected = np.where(np.all(ver_stabs_1 @ errors_1.T % 2 == 0, axis=1))[0]
    assert len(non_detected) == 0

    errors_2 = circ.compute_fault_set(2)
    non_detected = np.where(np.all(ver_stabs_2 @ errors_2.T % 2 == 0, axis=1))[0]
    assert len(non_detected) == 0


def test_not_full_ft_heuristic_cc5(color_code_d5_sp: StatePrepCircuit) -> None:
    """Test that the optimal verification circuit for the Steane code is correct.

    Ignore Z errors.
    """
    circ = color_code_d5_sp
    ver_stabs_layers = heuristic_verification_stabilizers(circ, x_errors=True, max_covering_sets=1000)

    assert len(ver_stabs_layers) == 2  # 2 layers of verification measurements

    ver_stabs_1 = ver_stabs_layers[0]
    ver_stabs_2 = ver_stabs_layers[1]

    z_gens = circ.z_checks

    for stab in np.vstack((ver_stabs_1, ver_stabs_2)):
        assert in_span(z_gens, stab)

    errors_1 = circ.compute_fault_set(1)
    non_detected = np.where(np.all(ver_stabs_1 @ errors_1.T % 2 == 0, axis=1))[0]
    assert len(non_detected) == 0

    errors_2 = circ.compute_fault_set(2)
    non_detected = np.where(np.all(ver_stabs_2 @ errors_2.T % 2 == 0, axis=1))[0]
    assert len(non_detected) == 0

    # Check that circuit is correct
    circ_ver = heuristic_verification_circuit(circ, full_fault_tolerance=False)
    n_cnots = np.sum(ver_stabs_1) + np.sum(ver_stabs_2)  # type: ignore[operator]
    assert circ_ver.num_qubits == circ.num_qubits + len(ver_stabs_1) + len(ver_stabs_2)
    assert circ_ver.num_nonlocal_gates() == n_cnots + circ.circ.num_nonlocal_gates()


@pytest.mark.skipif(os.getenv("CI") is not None and sys.platform == "win32", reason="Too slow for CI on Windows")
def test_error_detection_code() -> None:
    """Test that different circuits are obtained when using an error detection code."""
    code = CSSCode.from_code_name("carbon")
    circ = heuristic_prep_circuit(code)

    circ_ver_correction = gate_optimal_verification_circuit(
        circ, max_ancillas=3, max_timeout=5, full_fault_tolerance=False
    )

    circ.set_error_detection(True)
    circ_ver_detection = gate_optimal_verification_circuit(
        circ, max_ancillas=3, max_timeout=5, full_fault_tolerance=False
    )

    assert circ_ver_detection.num_qubits > circ_ver_correction.num_qubits
    assert circ_ver_detection.num_nonlocal_gates() > circ_ver_correction.num_nonlocal_gates()
