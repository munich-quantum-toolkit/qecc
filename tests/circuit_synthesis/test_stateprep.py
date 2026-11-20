# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test synthesis of state preparation and verification circuits for FT state preparation."""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import (
    VerificationStabilizersNotFoundError,
    depth_optimal_prep_circuit,
    gate_optimal_prep_circuit,
    gate_optimal_verification_circuit,
    gate_optimal_verification_stabilizers,
    heuristic_prep_circuit,
    heuristic_verification_circuit,
    heuristic_verification_stabilizers,
    state_prep,
)
from mqt.qecc.circuit_synthesis.faults import PureFaultSet
from mqt.qecc.circuit_synthesis.synthesis_utils import SearchResult, iterative_search_with_timeout
from mqt.qecc.codes import RotatedSurfaceCode, SquareOctagonColorCode

from .utils import eq_span, in_span

if TYPE_CHECKING:  # pragma: no cover
    from mqt.qecc.circuit_synthesis import FaultyStatePrepCircuit


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
    return RotatedSurfaceCode(3)


@pytest.fixture(scope="session")
def tetrahedral_code() -> CSSCode:
    """Return the tetrahedral code."""
    return CSSCode.from_code_name("tetrahedral")


@pytest.fixture(scope="session")
def cc_4_8_8_code() -> CSSCode:
    """Return the d=5 4,8,8 color code."""
    return SquareOctagonColorCode(5)


@pytest.fixture(scope="session")
def steane_code_sp(steane_code: CSSCode) -> FaultyStatePrepCircuit:
    """Return a non-ft state preparation circuit for the Steane code."""
    sp_circ = heuristic_prep_circuit(steane_code)
    sp_circ.compute_fault_sets()
    return sp_circ


@pytest.fixture(scope="session")
def tetrahedral_code_sp(tetrahedral_code: CSSCode) -> FaultyStatePrepCircuit:
    """Return a non-ft state preparation circuit for the tetrahedral code."""
    sp_circ = heuristic_prep_circuit(tetrahedral_code)
    sp_circ.compute_fault_sets()
    return sp_circ


@pytest.fixture(scope="session")
def color_code_d5_sp(cc_4_8_8_code: CSSCode) -> FaultyStatePrepCircuit:
    """Return a non-ft state preparation circuit for the d=5 4,8,8 color code."""
    sp_circ = heuristic_prep_circuit(cc_4_8_8_code)
    sp_circ.compute_fault_sets()
    return sp_circ


def test_heuristic_overcomplete_stabilizers() -> None:
    """Check that synthesis also works for overcomplete stabilizers."""
    code = CSSCode(np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), np.array([[1, 1, 1, 1], [1, 1, 1, 1]]), 2)
    sp_circ = heuristic_prep_circuit(code)
    assert eq_span(code.Hx, sp_circ.x_checks)
    assert eq_span(np.vstack((code.Hz, code.Lz)), sp_circ.z_checks)


@pytest.mark.parametrize(
    "code", ["steane_code", "css_4_2_2_code", "css_6_2_2_code", "tetrahedral_code", "surface_code"]
)
def test_heuristic_prep_consistent(code: str, request: pytest.FixtureRequest) -> None:
    """Check that heuristic_prep_circuit returns a valid circuit with the correct stabilizers."""
    code_ = request.getfixturevalue(code)

    sp_circ = heuristic_prep_circuit(code_)
    circ = sp_circ.circ
    max_cnots = np.sum(code_.Hx) + np.sum(code_.Hz)

    assert circ.num_qubits() == code_.n
    assert circ.num_cnots() <= max_cnots

    assert eq_span(code_.Hx, sp_circ.x_checks)
    assert eq_span(np.vstack((code_.Hz, code_.Lz)), sp_circ.z_checks)


@pytest.mark.skipif(
    os.getenv("CI") is not None and (sys.platform == "win32" or sys.platform == "darwin"),
    reason="Too slow for CI on Windows or MacOS",
)
@pytest.mark.parametrize("code", ["css_4_2_2_code", "css_6_2_2_code"])
def test_gate_optimal_prep_consistent(code: str, request: pytest.FixtureRequest) -> None:
    """Check that gate_optimal_prep_circuit returns a valid circuit with the correct stabilizers."""
    code_ = request.getfixturevalue(code)
    sp_circ = gate_optimal_prep_circuit(code_, max_timeout=3)
    assert sp_circ is not None

    circ = sp_circ.circ
    max_cnots = np.sum(code_.Hx) + np.sum(code_.Hz)

    assert circ.num_qubits() == code_.n
    assert circ.num_cnots() <= max_cnots

    assert eq_span(code_.Hx, sp_circ.x_checks)
    assert eq_span(np.vstack((code_.Hz, code_.Lz)), sp_circ.z_checks)


@pytest.mark.skipif(
    os.getenv("CI") is not None and (sys.platform == "win32" or sys.platform == "darwin"),
    reason="Too slow for CI on Windows or MacOS",
)
@pytest.mark.parametrize("code", ["css_4_2_2_code", "css_6_2_2_code"])
def test_depth_optimal_prep_consistent(code: str, request: pytest.FixtureRequest) -> None:
    """Check that depth_optimal_prep_circuit returns a valid circuit with the correct stabilizers."""
    code_ = request.getfixturevalue(code)

    sp_circ = depth_optimal_prep_circuit(code_, max_timeout=3)
    assert sp_circ is not None
    circ = sp_circ.circ
    max_cnots = np.sum(code_.Hx) + np.sum(code_.Hz)

    assert circ.num_qubits() == code_.n
    assert circ.num_cnots() <= max_cnots

    assert eq_span(code_.Hx, sp_circ.x_checks)
    assert eq_span(np.vstack((code_.Hz, code_.Lz)), sp_circ.z_checks)


@pytest.mark.skipif(os.getenv("CI") is not None and sys.platform == "win32", reason="Too slow for CI on Windows")
@pytest.mark.parametrize("code", ["css_4_2_2_code", "css_6_2_2_code"])
def test_plus_state_gate_optimal(code: str, request: pytest.FixtureRequest) -> None:
    """Test synthesis of the plus state."""
    code_ = request.getfixturevalue(code)
    sp_circ_plus = gate_optimal_prep_circuit(code_, max_timeout=3, zero_state=False)

    assert sp_circ_plus is not None

    circ_plus = sp_circ_plus.circ
    max_cnots = np.sum(code_.Hx) + np.sum(code_.Hz)

    assert circ_plus.num_qubits() == code_.n
    assert circ_plus.num_cnots() <= max_cnots

    assert eq_span(code_.Hz, sp_circ_plus.z_checks)
    assert eq_span(np.vstack((code_.Hx, code_.Lx)), sp_circ_plus.x_checks)

    sp_circ_zero = gate_optimal_prep_circuit(code_, max_timeout=5, zero_state=True)

    assert sp_circ_zero is not None

    # The plus- and zero-state preps are independently optimal circuits, so for a self-dual
    # code their check matrices are dual up to span (not necessarily identical matrices).
    if code_.is_self_dual():
        assert eq_span(sp_circ_plus.x_checks, sp_circ_zero.z_checks)
        assert eq_span(sp_circ_plus.z_checks, sp_circ_zero.x_checks)
    else:
        assert not eq_span(sp_circ_plus.x_checks, sp_circ_zero.z_checks)
        assert not eq_span(sp_circ_plus.z_checks, sp_circ_zero.x_checks)


@pytest.mark.parametrize(
    "code", ["steane_code", "css_4_2_2_code", "css_6_2_2_code", "surface_code", "tetrahedral_code"]
)
def test_plus_state_heuristic(code: str, request: pytest.FixtureRequest) -> None:
    """Test synthesis of the plus state."""
    code_ = request.getfixturevalue(code)
    sp_circ_plus = heuristic_prep_circuit(code_, state="all_plus")

    assert sp_circ_plus is not None

    circ_plus = sp_circ_plus.circ
    print(circ_plus.to_stim_circuit().to_crumble_url())
    max_cnots = np.sum(code_.Hx) + np.sum(code_.Hz)

    assert circ_plus.num_qubits() == code_.n
    assert circ_plus.num_cnots() <= max_cnots

    assert eq_span(code_.Hz, sp_circ_plus.z_checks)
    assert eq_span(np.vstack((code_.Hx, code_.Lx)), sp_circ_plus.x_checks)

    sp_circ_zero = heuristic_prep_circuit(code_, state="all_zero")

    if code_.is_self_dual():
        assert eq_span(sp_circ_plus.x_checks, sp_circ_zero.z_checks)
        assert eq_span(sp_circ_plus.z_checks, sp_circ_zero.x_checks)
    else:
        assert not eq_span(sp_circ_plus.x_checks, sp_circ_zero.z_checks)
        assert not eq_span(sp_circ_plus.z_checks, sp_circ_zero.x_checks)


@pytest.mark.skipif(os.getenv("CI") is not None and sys.platform == "win32", reason="Too slow for CI on Windows")
def test_optimal_steane_verification_circuit(steane_code_sp: FaultyStatePrepCircuit) -> None:
    """Test that the optimal verification circuit for the Steane code is correct."""
    circ = steane_code_sp
    ver_stabs_layers = gate_optimal_verification_stabilizers(circ.x_fault_sets, circ.z_checks, max_timeout=5)

    assert len(ver_stabs_layers) == 1  # 1 Ancilla measurement

    ver_stabs = ver_stabs_layers[0]

    assert np.sum(ver_stabs) == 3  # 3 CNOTs
    z_gens = circ.z_checks

    for stab in ver_stabs:
        assert in_span(z_gens, stab)

    assert circ.x_fault_sets[0].all_faults_detected(np.asarray(ver_stabs, dtype=np.int8))

    # Check that circuit is correct
    circ_ver = gate_optimal_verification_circuit(circ)

    assert circ_ver.num_qubits == circ.num_qubits + 1
    assert circ_ver.num_nonlocal_gates() == np.sum(ver_stabs) + circ.circ.num_cnots()
    assert circ_ver.depth() <= np.sum(ver_stabs) + circ.circ.depth() + 2


def test_heuristic_steane_verification_circuit(steane_code_sp: FaultyStatePrepCircuit) -> None:
    """Test that the optimal verification circuit for the Steane code is correct."""
    circ = steane_code_sp

    ver_stabs_layers = heuristic_verification_stabilizers(circ.x_fault_sets, circ.z_checks, max_covering_sets=10000)

    assert len(ver_stabs_layers) == 1  # 1 layer of verification measurements

    ver_stabs = ver_stabs_layers[0]
    assert len(ver_stabs) == 1  # 1 Ancilla measurement
    assert np.sum(ver_stabs[0]) == 3  # 3 CNOTs
    z_gens = circ.z_checks

    for stab in ver_stabs:
        assert in_span(z_gens, stab)

    assert circ.x_fault_sets[0].all_faults_detected(np.asarray(ver_stabs, dtype=np.int8))

    # Check that circuit is correct
    circ_ver = heuristic_verification_circuit(circ)
    assert circ_ver.num_qubits == circ.num_qubits + 1
    assert circ_ver.num_nonlocal_gates() == np.sum(ver_stabs) + circ.circ.num_cnots()
    print(circ_ver)
    assert circ_ver.depth() <= np.sum(ver_stabs) + circ.circ.depth() + 2  # 1 for the measurement, 1 for the hadamards


def test_cc5() -> None:
    """Test that synthesis works for the d=5 4,8,8 color code.

    Regression test: Heuristic previously found circuits with 23 cnots.
    """
    code = SquareOctagonColorCode(5)
    sp_circ = heuristic_prep_circuit(code)
    circ = sp_circ.circ
    assert circ.num_cnots() <= 23


@pytest.mark.skipif(
    os.getenv("CI") is not None and (sys.platform == "win32" or sys.platform == "darwin"),
    reason="Too slow for CI on Windows or MacOS",
)
def test_not_full_ft_opt_cc5(color_code_d5_sp: FaultyStatePrepCircuit) -> None:
    """Test that the optimal verification is also correct for higher distance.

    Ignore Z errors.
    Due to time constraints, we limit the timeout for each search.
    """
    circ = color_code_d5_sp

    ver_stabs_layers = gate_optimal_verification_stabilizers(
        circ.x_fault_sets, circ.z_checks, max_ancillas=3, max_timeout=10
    )
    assert len(ver_stabs_layers) == 2  # 2 layers of verification measurements

    ver_stabs_1 = ver_stabs_layers[0]

    assert len(ver_stabs_1) <= 2  # 2 Ancilla measurements
    assert np.sum(ver_stabs_1) <= 10  # 9 CNOTs

    ver_stabs_2 = ver_stabs_layers[1]
    assert len(ver_stabs_2) == 3  # 2 Ancilla measurements
    assert np.sum(ver_stabs_2) <= 14  # less than 14 CNOTs (sometimes 13, sometimes 14 depending on how fast the CPU is)

    z_gens = circ.z_checks

    for stab in np.vstack((ver_stabs_1, ver_stabs_2)):
        assert in_span(z_gens, stab)

    assert circ.x_fault_sets[0].all_faults_detected(np.asarray(ver_stabs_1, dtype=np.int8))
    assert circ.x_fault_sets[1].all_faults_detected(np.asarray(ver_stabs_2, dtype=np.int8))


def test_full_ft_heuristic_cc5(color_code_d5_sp: FaultyStatePrepCircuit) -> None:
    """Test that the optimal verification circuit for the Steane code is correct.

    Ignore Z errors.
    """
    circ = color_code_d5_sp
    ver_stabs_layers = heuristic_verification_stabilizers(circ.x_fault_sets, circ.z_checks, max_covering_sets=1000)

    assert len(ver_stabs_layers) == 2  # 2 layers of verification measurements

    ver_stabs_1 = ver_stabs_layers[0]
    ver_stabs_2 = ver_stabs_layers[1]

    z_gens = circ.z_checks

    for stab in np.vstack((ver_stabs_1, ver_stabs_2)):
        assert in_span(z_gens, stab)

    assert circ.x_fault_sets[0].all_faults_detected(np.asarray(ver_stabs_1, dtype=np.int8))
    assert circ.x_fault_sets[1].all_faults_detected(np.asarray(ver_stabs_2, dtype=np.int8))

    # Check that circuit is correct
    circ_ver = heuristic_verification_circuit(circ, only_first_layer=True)
    n_cnots = np.sum(ver_stabs_1) + np.sum(ver_stabs_2)
    assert circ_ver.num_qubits == circ.num_qubits + len(ver_stabs_1) + len(ver_stabs_2)
    assert circ_ver.num_nonlocal_gates() == n_cnots + circ.circ.num_cnots()


@pytest.mark.parametrize(("state", "verify_x_first"), [("all_plus", True), ("all_plus", False), ("all_zero", False)])
def test_only_first_layer_without_measurements(state: str, verify_x_first: bool) -> None:
    """Test that only verifying the first layer works if that layer requires no measurement."""
    sp_circ = heuristic_prep_circuit(CSSCode.from_code_name("Shor"), state=state)
    circ_ver = heuristic_verification_circuit(sp_circ, only_first_layer=True, verify_x_first=verify_x_first)

    # no verification stabilizers are required, so the circuit is left unchanged
    assert circ_ver.num_qubits == sp_circ.num_qubits
    assert circ_ver.num_nonlocal_gates() == sp_circ.circ.num_cnots()


def _undetectable_fault_set() -> tuple[list[PureFaultSet], np.ndarray]:
    """Return a fault set whose only fault commutes with the only available check."""
    faults = PureFaultSet(4)
    faults.add_fault(np.array([1, 1, 1, 1], dtype=np.int8))
    checks = np.array([[1, 1, 0, 0]], dtype=np.int8)
    return [faults], checks


def test_empty_fault_set_needs_no_verification() -> None:
    """An empty fault set yields an empty layer and must not be reported as a failed search."""
    _, checks = _undetectable_fault_set()
    empty = [PureFaultSet(4)]

    assert heuristic_verification_stabilizers(empty, checks) == [[]]
    assert gate_optimal_verification_stabilizers(empty, checks, max_timeout=5) == [[]]


def test_undetectable_faults_are_reported_heuristic() -> None:
    """Faults that commute with every check cannot be verified and must be reported as such."""
    fault_sets, checks = _undetectable_fault_set()

    with pytest.raises(VerificationStabilizersNotFoundError) as exc_info:
        heuristic_verification_stabilizers(fault_sets, checks)

    assert exc_info.value.reason == "undetectable"
    assert exc_info.value.num_errors == 1


def test_undetectable_faults_are_reported_gate_optimal() -> None:
    """The gate optimal search must report undetectable faults instead of returning no stabilizers."""
    fault_sets, checks = _undetectable_fault_set()

    with pytest.raises(VerificationStabilizersNotFoundError) as exc_info:
        gate_optimal_verification_stabilizers(fault_sets, checks, max_timeout=5)

    assert exc_info.value.reason == "undetectable"


def test_faults_without_any_checks_are_undetectable() -> None:
    """Without any checks to measure, every fault is undetectable."""
    fault_sets, _ = _undetectable_fault_set()
    no_checks = np.zeros((0, 4), dtype=np.int8)

    with pytest.raises(VerificationStabilizersNotFoundError) as exc_info:
        heuristic_verification_stabilizers(fault_sets, no_checks)

    assert exc_info.value.reason == "undetectable"


def _detectable_fault_set() -> tuple[list[PureFaultSet], np.ndarray]:
    """Return a fault set whose only fault is detected by the only available check."""
    faults = PureFaultSet(4)
    faults.add_fault(np.array([1, 0, 0, 0], dtype=np.int8))
    checks = np.array([[1, 1, 0, 0]], dtype=np.int8)
    return [faults], checks


@pytest.mark.parametrize(("timed_out", "reason"), [(True, "timeout"), (False, "budget")])
def test_unsuccessful_gate_optimal_search_reports_reason(
    timed_out: bool, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gate optimal search without result reports whether it timed out or exhausted its budget."""
    fault_sets, checks = _detectable_fault_set()
    monkeypatch.setattr(
        state_prep,
        "iterative_search_with_timeout",
        lambda _fun, _min_param, max_param, *_args, **_kwargs: SearchResult(None, max_param, timed_out=timed_out),
    )

    with pytest.raises(VerificationStabilizersNotFoundError) as exc_info:
        gate_optimal_verification_stabilizers(fault_sets, checks, max_timeout=5)

    assert exc_info.value.reason == reason
    assert exc_info.value.num_errors == 1


def test_all_solutions_falls_back_to_the_solution_already_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """If enumerating all solutions yields nothing, the solution found by the search is kept."""
    fault_sets, checks = _detectable_fault_set()
    original = state_prep.all_verification_stabilizers

    def no_further_solutions(
        fault_set: PureFaultSet,
        stabs: np.ndarray,
        num_anc: int,
        num_cnots: int,
        return_all_solutions: bool = False,
    ) -> list[list[np.ndarray]] | None:
        if return_all_solutions:
            return None
        return original(fault_set, stabs, num_anc, num_cnots, return_all_solutions=False)

    monkeypatch.setattr(state_prep, "all_verification_stabilizers", no_further_solutions)

    layers = state_prep.all_gate_optimal_verification_stabilizers(
        fault_sets, checks, max_timeout=5, return_all_solutions=True
    )

    assert len(layers[0]) == 1
    assert fault_sets[0].all_faults_detected(np.asarray(layers[0][0], dtype=np.int8))


def _always_times_out(*_args: object, **_kwargs: object) -> list[list[np.ndarray]]:
    """Stand in for a stabilizer search that does not reach a conclusion."""
    raise VerificationStabilizersNotFoundError(reason="timeout", num_errors=1)


def test_strict_raises_on_unsuccessful_search(
    steane_code_sp: FaultyStatePrepCircuit, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With strict=True an unsuccessful search is raised to the caller."""
    monkeypatch.setattr("mqt.qecc.circuit_synthesis.state_prep.heuristic_verification_stabilizers", _always_times_out)

    with pytest.raises(VerificationStabilizersNotFoundError):
        heuristic_verification_circuit(steane_code_sp, strict=True)


def test_non_strict_warns_on_unsuccessful_search(
    steane_code_sp: FaultyStatePrepCircuit, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With strict=False an unsuccessful search is downgraded to a warning."""
    monkeypatch.setattr("mqt.qecc.circuit_synthesis.state_prep.heuristic_verification_stabilizers", _always_times_out)

    with caplog.at_level(logging.WARNING, logger="mqt.qecc.circuit_synthesis.state_prep"):
        circ = heuristic_verification_circuit(steane_code_sp, strict=False)

    assert circ.num_qubits == steane_code_sp.num_qubits
    assert "exceeded its time limit" in caplog.text


def test_only_first_layer_warns_about_skipped_error_type(caplog: pytest.LogCaptureFixture) -> None:
    """Skipping an error type that the circuit actually propagates must be reported."""
    sp_circ = heuristic_prep_circuit(CSSCode.from_code_name("Shor"), state="all_zero")

    with caplog.at_level(logging.WARNING, logger="mqt.qecc.circuit_synthesis.state_prep"):
        heuristic_verification_circuit(sp_circ, only_first_layer=True, verify_x_first=False)

    assert "not fault tolerant with respect to X errors" in caplog.text


def test_only_first_layer_is_quiet_when_nothing_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    """Skipping an error type with an empty fault set is sound and must not warn."""
    sp_circ = heuristic_prep_circuit(CSSCode.from_code_name("Shor"), state="all_zero")

    with caplog.at_level(logging.WARNING, logger="mqt.qecc.circuit_synthesis.state_prep"):
        heuristic_verification_circuit(sp_circ, only_first_layer=True, verify_x_first=True)

    assert "not fault tolerant" not in caplog.text


def _never_succeeds(_param: int) -> None:
    return None


def _too_slow(_param: int) -> None:
    time.sleep(5)


def test_search_result_distinguishes_exhausted_budget_from_timeout() -> None:
    """A search that returns no result must say whether it was inconclusive."""
    # Each attempt runs in a new process, and its timeout also covers starting that process. On Windows, starting
    # one and importing the test module can take longer than a second, so give the attempt that returns
    # immediately a generous timeout to keep start-up time from being reported as a timeout.
    exhausted = iterative_search_with_timeout(_never_succeeds, 1, 1, 60, 60)
    assert exhausted.result is None
    assert not exhausted.timed_out

    timed_out = iterative_search_with_timeout(_too_slow, 1, 1, 1, 1)
    assert timed_out.result is None
    assert timed_out.timed_out


@pytest.mark.parametrize("logical_state", ["00", "0+", "+0", "++"])
def test_heuristic_custom_logical_state(logical_state: str, css_4_2_2_code: CSSCode) -> None:
    """Test preparation of custom logical states."""
    sp_circ = heuristic_prep_circuit(css_4_2_2_code, state=logical_state)

    plus = [i for i, state in enumerate(logical_state) if state == "+"]
    zero = [i for i, state in enumerate(logical_state) if state == "0"]

    assert sp_circ.num_qubits == css_4_2_2_code.n
    assert eq_span(np.vstack((css_4_2_2_code.Hx, css_4_2_2_code.Lx[plus])), sp_circ.x_checks)
    assert eq_span(np.vstack((css_4_2_2_code.Hz, css_4_2_2_code.Lz[zero])), sp_circ.z_checks)


@pytest.mark.parametrize(("logical_state", "equivalent_string"), [("00", "all_zero"), ("++", "all_plus")])
def test_heuristic_explicit_compatibility(logical_state: str, equivalent_string: str, css_4_2_2_code: CSSCode) -> None:
    """Test preparation of custom logical states is compatible with the explicit string interface."""
    bool_circ = heuristic_prep_circuit(css_4_2_2_code, state=logical_state)
    str_circ = heuristic_prep_circuit(css_4_2_2_code, state=equivalent_string)

    assert eq_span(bool_circ.x_checks, str_circ.x_checks)
    assert eq_span(bool_circ.z_checks, str_circ.z_checks)


# @pytest.mark.skipif(os.getenv("CI") is not None and sys.platform == "win32", reason="Too slow for CI on Windows")
# def test_error_detection_code() -> None:
#     """Test that different circuits are obtained when using an error detection code."""
#     code = CSSCode.from_code_name("carbon")
#     circ = heuristic_prep_circuit(code)

#     circ.set_max_errors(1, 1)
#     circ_ver_correction = gate_optimal_verification_circuit(circ, max_ancillas=3, max_timeout=5, only_first_layer=True)

#     circ.set_max_errors(2, 2)
#     circ_ver_detection = gate_optimal_verification_circuit(circ, max_ancillas=3, max_timeout=5, only_first_layer=True)

#     assert circ_ver_detection.num_qubits > circ_ver_correction.num_qubits
#     assert circ_ver_detection.num_nonlocal_gates() > circ_ver_correction.num_nonlocal_gates()


# def test_combine_faults() -> None:
#     """Test `combine_faults` method of `FaultyStatePrepCircuit` class."""
#     code = CSSCode(
#         np.array([[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 1]]),
#         x_distance=1,
#         z_distance=6,
#     )  # d=5 rep code
#     circ = heuristic_prep_circuit(code)
#     # circuit has single-qubit z faults [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 0, 0]
#     circ.compute_fault_sets()
#     new_faults = PureFaultSet.from_fault_array(np.array([[1, 0, 1, 0, 0, 0]], dtype=np.int8))

#     combined_faults = circ.combine_faults(new_faults, x_errors=False, reduce=True)

#     print(combined_faults[1])
#     print(circ.z_fault_sets[1])

#     combined_1 = new_faults.combine(circ.z_fault_sets[0])

#     combined_2 = circ.z_fault_sets[1].copy()
#     combined_2.add_faults(np.array([[1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 0, 1]]))
#     combined_2.normalize(circ.z_checks)

#     assert combined_faults[0] == combined_1
#     assert combined_faults[1] == combined_2
