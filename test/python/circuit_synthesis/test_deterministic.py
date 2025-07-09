# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test synthesis and simulation of deterministic FT state preparation circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ldpc import mod2

try:
    from qsample import noise

    from mqt.qecc.circuit_synthesis.simulation_det import NoisyDFTStatePrepSimulator

    HAS_QSAMPLE = True
except ImportError:
    HAS_QSAMPLE = False


from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import (
    DeterministicVerificationHelper,
    heuristic_prep_circuit,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from mqt.qecc.circuit_synthesis import DeterministicVerification, StatePrepCircuit

# Simulation parameters

if HAS_QSAMPLE:
    err_params = {"q": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]}
    err_model = noise.E1_1
    shots_dss = 4000
    p_max = {"q": 0.01}
    L = 3


@pytest.fixture(scope="module")
def steane_code_sp_plus() -> StatePrepCircuit:
    """Return a non-ft state preparation circuit for the Steane code."""
    steane_code = CSSCode.from_code_name("Steane")
    sp_circ = heuristic_prep_circuit(steane_code, zero_state=False)
    sp_circ.compute_fault_sets()
    return sp_circ


@pytest.fixture(scope="module")
def verified_steane_data(
    steane_code_sp_plus: StatePrepCircuit,
) -> tuple[DeterministicVerification, DeterministicVerification, DeterministicVerification, DeterministicVerification]:
    """Prepare the solutions once, but make no assertions here."""
    verify_helper = DeterministicVerificationHelper(steane_code_sp_plus)
    verify_z_opt, verify_x_opt = verify_helper.get_solution()
    verify_z_global, verify_x_global = verify_helper.get_global_solution()
    return verify_z_opt, verify_x_opt, verify_z_global, verify_x_global


@pytest.fixture(scope="module")
def css_11_1_3_code_sp() -> StatePrepCircuit:
    """Return a non-ft state preparation circuit for the 11_1_3 code."""
    check_matrix = np.array([
        [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
    ])
    code = CSSCode(distance=3, Hx=check_matrix, Hz=check_matrix)
    sp_circ = heuristic_prep_circuit(code)
    sp_circ.compute_fault_sets()
    return sp_circ


@pytest.fixture(scope="module")
def verified_11_1_3_data(
    css_11_1_3_code_sp: StatePrepCircuit,
) -> tuple[DeterministicVerification, DeterministicVerification]:
    """Run deterministic verification once, return X/Z verification circuits."""
    verify_helper = DeterministicVerificationHelper(css_11_1_3_code_sp)
    verify_x, verify_z = verify_helper.get_solution()
    return verify_x, verify_z


def in_span(m: npt.NDArray[np.int_], v: npt.NDArray[np.int_]) -> bool:
    """Check if a vector is in the row space of a matrix."""
    return bool(mod2.rank(np.vstack((m, v))) == mod2.rank(m))


def assert_statistics(
    verify: DeterministicVerification,
    num_ancillas_verification: int,
    num_cnots_verification: int,
    num_ancillas_correction: int,
    num_cnots_correction: int,
    num_ancillas_hooks: int = 0,
    num_cnots_hooks: int = 0,
    num_ancillas_hook_corrections: int = 0,
    num_cnots_hook_corrections: int = 0,
) -> None:
    """Assert that the statistics of a deterministic verification are correct."""
    assert verify.num_ancillas_verification() == num_ancillas_verification
    assert verify.num_cnots_verification() == num_cnots_verification
    assert verify.num_ancillas_correction() <= num_ancillas_correction
    assert verify.num_cnots_correction() <= num_cnots_correction
    assert verify.num_ancillas_hooks() == num_ancillas_hooks
    assert verify.num_cnots_hooks() == num_cnots_hooks
    assert verify.num_ancillas_hook_corrections() == num_ancillas_hook_corrections
    assert verify.num_cnots_hook_corrections() == num_cnots_hook_corrections


def assert_stabs(verify: DeterministicVerification, code: CSSCode, z_stabs: bool) -> None:
    """Assert that the measurement stabs of a deterministic verification are correct."""
    checks = np.vstack((code.Hz, code.Lz))
    checks_other = np.vstack((code.Hx, code.Lx))
    if not z_stabs:
        checks, checks_other = checks_other, checks

    for stab in verify.stabs:
        assert in_span(checks, stab)
    for correction in verify.det_correction.values():
        stabs, _ = correction
        for stab in stabs:
            assert in_span(checks, stab)
    for hook in verify.hook_corrections:
        if not hook:
            continue
        for correction in hook.values():
            stabs, _ = correction
            for stab in stabs:
                assert in_span(checks_other, stab)


def assert_scaling(simulation_results: list[npt.NDArray[np.float64]]) -> None:
    """Assert that the logical error rates scales approximately quadratically."""
    dss_upper_bound = simulation_results[-2]
    x = np.log10(err_params["q"])
    y = np.log10(dss_upper_bound)
    m = np.diff(y) / np.diff(x)
    assert np.average(m[:3]) > 1.3


def test_11_1_3_det_verification_correctness(
    verified_11_1_3_data: tuple[DeterministicVerification, DeterministicVerification],
    css_11_1_3_code_sp: StatePrepCircuit,
) -> None:
    """Test correctness of deterministic verification circuit for 11_1_3 code."""
    verify_x, verify_z = verified_11_1_3_data

    # Check X-verification
    assert_statistics(verify_x, 2, 8, 4, 14, 0, 0)
    assert_stabs(verify_x, css_11_1_3_code_sp.code, z_stabs=True)

    # Check Z-verification
    assert_statistics(verify_z, 1, 4, 1, 4, 1, 2, 1, 3)
    assert_stabs(verify_z, css_11_1_3_code_sp.code, z_stabs=False)


@pytest.mark.skipif(not HAS_QSAMPLE, reason="Requires 'qsample' to be installed.")
def test_11_1_3_det_simulation(
    verified_11_1_3_data: tuple[DeterministicVerification, DeterministicVerification],
    css_11_1_3_code_sp: StatePrepCircuit,
) -> None:
    """Test simulated logical error rate for deterministic 11_1_3 state preparation."""
    verify_x, verify_z = verified_11_1_3_data

    simulator = NoisyDFTStatePrepSimulator(
        css_11_1_3_code_sp.circ, (verify_x, verify_z), css_11_1_3_code_sp.code, err_model
    )
    simulation_results = simulator.dss_logical_error_rates(err_params, p_max, L, shots_dss)
    assert_scaling(simulation_results)


def test_steane_det_verification(
    verified_steane_data: tuple[
        DeterministicVerification, DeterministicVerification, DeterministicVerification, DeterministicVerification
    ],
    steane_code_sp_plus: StatePrepCircuit,
) -> None:
    """Test correctness of deterministic verification circuit for the Steane code."""
    verify_z_opt, verify_x_opt, verify_z_global, verify_x_global = verified_steane_data

    for verify_x, verify_z in zip((verify_x_opt, verify_x_global), (verify_z_opt, verify_z_global)):
        assert_statistics(verify_z, 1, 3, 1, 3, 0, 0)
        assert_stabs(verify_z, steane_code_sp_plus.code, z_stabs=False)
        assert verify_x.num_ancillas_total() == 0
        assert verify_x.num_cnots_total() == 0


@pytest.mark.skipif(not HAS_QSAMPLE, reason="Requires 'qsample' to be installed.")
def test_steane_det_simulation(
    verified_steane_data: tuple[
        DeterministicVerification, DeterministicVerification, DeterministicVerification, DeterministicVerification
    ],
    steane_code_sp_plus: StatePrepCircuit,
) -> None:
    """Test simulated logical error rate for deterministic Steane state preparation."""
    verify_z_opt, verify_x_opt, verify_z_global, verify_x_global = verified_steane_data

    for verify_x, verify_z in zip((verify_x_opt, verify_x_global), (verify_z_opt, verify_z_global)):
        simulator = NoisyDFTStatePrepSimulator(
            steane_code_sp_plus.circ, (verify_z, verify_x), steane_code_sp_plus.code, err_model, False
        )
        simulation_results = simulator.dss_logical_error_rates(err_params, p_max, L, shots_dss)
        assert_scaling(simulation_results)
