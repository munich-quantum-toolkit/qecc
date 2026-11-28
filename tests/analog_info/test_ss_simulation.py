# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test single shot simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pytest

from mqt.qecc import CSSCode
from mqt.qecc.analog_information_decoding.simulators import simulation
from mqt.qecc.analog_information_decoding.simulators.simulation import SingleShotSimulator
from mqt.qecc.analog_information_decoding.utils.simulation_utils import (
    build_single_stage_pcm,
    error_channel_setup,
)
from mqt.qecc.noise import GaussianReadoutChannel, PhenomenologicalNoiseModel, PhenomenologicalNoiseSampler

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

from mqt.qecc.analog_information_decoding.utils.data_utils import BpParams


@pytest.fixture
def code_params() -> dict[str, int]:
    """Fixture for code params."""
    return {"n": 7, "k": 1, "d": 3}


@pytest.fixture
def mcm() -> NDArray[np.int32]:
    """Fixture for meta check matrix."""
    return np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]).astype(np.int32)


@pytest.fixture
def pcm() -> NDArray[np.int32]:
    """Fixture for parity check matrix."""
    return np.array([[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1, 1]]).astype(np.int32)


@pytest.fixture
def bias() -> NDArray[np.float64]:
    """Fixture for x,y,z bias vector."""
    return np.array([1.0, 1.0, 1.0])


@pytest.fixture
def error_rate() -> float:
    """Fixture for error rate."""
    return 0.1


@pytest.fixture
def single_stage_analog_simulator(
    error_rate: float,
    pcm: NDArray[np.int32],
    mcm: NDArray[np.int32],
    code_params: dict[str, int],
    bias: NDArray[np.float64],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SingleShotSimulator:
    """Fixture for single stage analog single shot simulator.

    The simulator writes its results below the current directory, so the fixture changes to a temporary directory.
    """
    monkeypatch.chdir(tmp_path)
    return SingleShotSimulator(
        codename="test",
        per=error_rate,
        ser=error_rate,
        single_stage=True,
        seed=666,
        bias=bias,
        x_meta=True,
        z_meta=False,
        sus_th_depth=1,
        bp_params=BpParams(osd_order=0, osd_method="osd0"),
        analog_info=False,
        analog_tg=True,
        hx=pcm,
        hz=pcm,
        mx=mcm,
        mz=None,
        lx=None,
        lz=None,
        code_params=code_params,
    )


@pytest.fixture
def two_stage_simulator(
    error_rate: float,
    pcm: NDArray[np.int32],
    mcm: NDArray[np.int32],
    code_params: dict[str, int],
    bias: NDArray[np.float64],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SingleShotSimulator:
    """Fixture for hard syndrome two stage simulator.

    The simulator writes its results below the current directory, so the fixture changes to a temporary directory.
    """
    monkeypatch.chdir(tmp_path)
    return SingleShotSimulator(
        codename="test",
        per=error_rate,
        ser=error_rate,
        single_stage=False,
        seed=666,
        bias=bias,
        x_meta=True,
        z_meta=False,
        sus_th_depth=1,
        bp_params=BpParams(osd_order=0, osd_method="osd0"),
        analog_info=False,
        analog_tg=False,
        hx=pcm,
        hz=pcm,
        mx=mcm,
        mz=None,
        lx=None,
        lz=None,
        code_params=code_params,
    )


def test_analog_ss_simulator_setup(
    error_rate: float,
    bias: NDArray[np.float64],
    pcm: NDArray[np.int32],
    mcm: NDArray[np.int32],
    single_stage_analog_simulator: SingleShotSimulator,
) -> None:
    """Test the initialization of error channels the simulator."""
    chnl = error_channel_setup(error_rate, bias)
    expected_x_syndr_chnl = np.full(pcm.shape[0], chnl.x_marginal)
    expected_z_syndr_chnl = np.full(pcm.shape[0], chnl.z_marginal)
    assert np.allclose(expected_x_syndr_chnl, single_stage_analog_simulator.x_syndr_error_channel)
    assert np.allclose(expected_z_syndr_chnl, single_stage_analog_simulator.z_syndr_error_channel)
    assert (
        single_stage_analog_simulator.sigma_x
        == GaussianReadoutChannel.from_bit_error_probability(expected_x_syndr_chnl[0]).sigma
    )
    assert (
        single_stage_analog_simulator.sigma_z
        == GaussianReadoutChannel.from_bit_error_probability(expected_z_syndr_chnl[0]).sigma
    )
    assert np.array_equal(
        single_stage_analog_simulator.x_apcm, np.hstack([pcm, np.identity(pcm.shape[0], dtype=np.int32)])
    )
    assert single_stage_analog_simulator.x_meta is True
    assert single_stage_analog_simulator.ss_x_pcm is not None
    assert np.array_equal(single_stage_analog_simulator.ss_x_pcm, build_single_stage_pcm(pcm, mcm))


def test_single_stage_initialization(single_stage_analog_simulator: SingleShotSimulator) -> None:
    """Test single stage single shot simulation initialization."""
    assert isinstance(single_stage_analog_simulator.noise_model, PhenomenologicalNoiseModel)
    assert isinstance(single_stage_analog_simulator.noise_sampler, PhenomenologicalNoiseSampler)
    res = single_stage_analog_simulator._single_sample()  # ruff:ignore[private-member-access]
    assert res[0] is not None
    assert res[1] is not None


def test_two_state_initialization(two_stage_simulator: SingleShotSimulator) -> None:
    """Test two stage single shot simultation initialization."""
    res = two_stage_simulator._two_stage_decoding(  # ruff:ignore[private-member-access]
        z_syndrome_w_err=np.array([0, 0, 0]), x_syndrome_w_err=np.array([0, 0, 0])
    )
    assert np.array_equal(res[0], np.array([0, 0, 0, 0, 0, 0, 0]))
    assert np.array_equal(res[1], np.array([0, 0, 0, 0, 0, 0, 0]))


def test_single_sample(single_stage_analog_simulator: SingleShotSimulator) -> None:
    """Test single sample overall."""
    with mock.patch("pathlib.Path.open"), mock.patch("json.dump"):
        res = single_stage_analog_simulator.run(1)
    assert res is not None
    assert res["pers"] == pytest.approx(0.1)
    assert res["code_N"] == 7
    assert res["x_ler"] is not None
    assert res["x_wer"] is not None
    assert res["z_wer"] is not None
    assert res["z_ler"] is not None
    assert res["x_success_cnt"] is not None
    assert res["z_success_cnt"] is not None


@pytest.fixture
def toric_code_checks() -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Check matrices of the distance-3 toric code, a hypergraph product of two cyclic repetition codes.

    Hx differs from Hz, so a check that swaps them gives wrong results.
    """
    h = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int32)
    eye = np.eye(3, dtype=np.int32)
    hx = np.hstack([np.kron(h, eye), np.kron(eye, h.T)]) % 2
    hz = np.hstack([np.kron(eye, h), np.kron(h.T, eye)]) % 2
    return hx, hz


def _two_stage_simulator_without_logicals(
    hx: NDArray[np.int32], hz: NDArray[np.int32], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SingleShotSimulator:
    monkeypatch.chdir(tmp_path)
    return SingleShotSimulator(
        codename="toric3",
        per=0.05,
        ser=0.0,
        single_stage=False,
        seed=1,
        bias=np.array([1.0, 1.0, 1.0]),
        x_meta=False,
        z_meta=False,
        sus_th_depth=0,
        bp_params=BpParams(osd_order=0, osd_method="osd0"),
        hx=hx,
        hz=hz,
        mx=None,
        mz=None,
        lx=None,
        lz=None,
        code_params={"n": hx.shape[1], "k": 2},
    )


def _fixed_errors(monkeypatch: pytest.MonkeyPatch, x_err: NDArray[np.int32], z_err: NDArray[np.int32]) -> None:
    """Make every sampled data error equal to X_ERR and Z_ERR."""
    monkeypatch.setattr(
        simulation,
        "generate_err",
        lambda nr_qubits, channel_probs, residual_err: (x_err.copy(), z_err.copy()),  # ruff: ignore[unused-lambda-argument]
    )


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_corrected_error_is_not_logical_without_logicals(
    toric_code_checks: tuple[NDArray[np.int32], NDArray[np.int32]],
    pauli: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without logical operators, a single-qubit error that the decoder corrects is not a logical error."""
    hx, hz = toric_code_checks
    sim = _two_stage_simulator_without_logicals(hx, hz, tmp_path, monkeypatch)
    single = np.zeros(hx.shape[1], dtype=np.int32)
    single[0] = 1
    none = np.zeros(hx.shape[1], dtype=np.int32)
    _fixed_errors(monkeypatch, *((single, none) if pauli == "X" else (none, single)))

    assert sim._single_sample() == (False, False)  # ruff: ignore[private-member-access]


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_logical_error_is_detected_without_logicals(
    toric_code_checks: tuple[NDArray[np.int32], NDArray[np.int32]],
    pauli: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without logical operators, an error equal to a logical operator counts as a logical error of its type."""
    hx, hz = toric_code_checks
    sim = _two_stage_simulator_without_logicals(hx, hz, tmp_path, monkeypatch)
    code = CSSCode(hx.astype(np.int8), hz.astype(np.int8))
    none = np.zeros(hx.shape[1], dtype=np.int32)
    if pauli == "X":
        _fixed_errors(monkeypatch, code.Lx[0].astype(np.int32), none)
        expected = (True, False)
    else:
        _fixed_errors(monkeypatch, none, code.Lz[0].astype(np.int32))
        expected = (False, True)

    assert sim._single_sample() == expected  # ruff: ignore[private-member-access]


@pytest.mark.parametrize("pauli", ["X", "Z"])
def test_stabilizer_error_is_not_logical_without_logicals(
    toric_code_checks: tuple[NDArray[np.int32], NDArray[np.int32]],
    pauli: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without logical operators, an error equal to a stabilizer of its own type is not a logical error.

    An X stabilizer lies in the row space of Hx but not of Hz, so this test fails if the check uses the wrong matrix.
    """
    hx, hz = toric_code_checks
    sim = _two_stage_simulator_without_logicals(hx, hz, tmp_path, monkeypatch)
    none = np.zeros(hx.shape[1], dtype=np.int32)
    _fixed_errors(monkeypatch, *((hx[0].copy(), none) if pauli == "X" else (none, hz[0].copy())))

    assert sim._single_sample() == (False, False)  # ruff: ignore[private-member-access]
