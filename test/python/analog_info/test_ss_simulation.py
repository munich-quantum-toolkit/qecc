# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
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

from mqt.qecc.analog_information_decoding.simulators.simulation import SingleShotSimulator
from mqt.qecc.analog_information_decoding.utils.simulation_utils import (
    build_single_stage_pcm,
    error_channel_setup,
    get_sigma_from_syndr_er,
)

if TYPE_CHECKING:
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
) -> SingleShotSimulator:
    """Fixture for single stage analog single shot simulator."""
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
        bp_params=BpParams(osd_method="osd0"),
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
) -> SingleShotSimulator:
    """Fixture for hard syndrome two stage simulator."""
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
        bp_params=BpParams(osd_method="osd0"),
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
    chnl = error_channel_setup(error_rate, bias, pcm.shape[0])
    expected_x_syndr_chnl = chnl[0] + chnl[1]
    expected_z_syndr_chnl = chnl[2] + chnl[1]
    assert np.allclose(expected_x_syndr_chnl, single_stage_analog_simulator.x_syndr_error_channel)
    assert np.allclose(expected_z_syndr_chnl, single_stage_analog_simulator.z_syndr_error_channel)
    assert single_stage_analog_simulator.sigma_x == get_sigma_from_syndr_er(expected_x_syndr_chnl[0])
    assert single_stage_analog_simulator.sigma_z == get_sigma_from_syndr_er(expected_z_syndr_chnl[0])
    assert np.array_equal(
        single_stage_analog_simulator.x_apcm, np.hstack([pcm, np.identity(pcm.shape[0], dtype=np.int32)])
    )
    assert single_stage_analog_simulator.x_meta is True
    assert np.array_equal(single_stage_analog_simulator.ss_x_pcm, build_single_stage_pcm(pcm, mcm))


def test_single_stage_initialization(single_stage_analog_simulator: SingleShotSimulator) -> None:
    """Test single stage single shot simulation initialization."""
    res = single_stage_analog_simulator._single_sample()  # noqa: SLF001
    assert res[0] is not None
    assert res[1] is not None


def test_two_state_initialization(two_stage_simulator: SingleShotSimulator) -> None:
    """Test two stage single shot simultation initialization."""
    res = two_stage_simulator._two_stage_decoding(  # noqa: SLF001
        z_syndrome_w_err=np.array([0, 0, 0]), x_syndrome_w_err=np.array([0, 0, 0])
    )
    assert np.array_equal(res[0], np.array([0, 0, 0, 0, 0, 0, 0]))
    assert np.array_equal(res[1], np.array([0, 0, 0, 0, 0, 0, 0]))


def test_single_sample(single_stage_analog_simulator: SingleShotSimulator) -> None:
    """Test single sample overall."""
    with mock.patch("pathlib.Path.open"), mock.patch("json.dump"):
        res = single_stage_analog_simulator.run(1)
    assert res is not None
    assert res["pers"] == 0.1
    assert res["code_N"] == 7
    assert res["x_ler"] is not None
    assert res["x_wer"] is not None
    assert res["z_wer"] is not None
    assert res["z_ler"] is not None
    assert res["x_success_cnt"] is not None
    assert res["z_success_cnt"] is not None
