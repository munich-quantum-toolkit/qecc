# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for analog Tannergraph decoder and simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pytest

from mqt.qecc.analog_information_decoding.simulators.analog_tannergraph_decoding import (
    AnalogTannergraphDecoder,
    AtdSimulator,
)
from mqt.qecc.analog_information_decoding.utils import simulation_utils
from mqt.qecc.analog_information_decoding.utils.data_utils import BpParams

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture
def code_params() -> dict[str, int]:
    """Return code parameters."""
    return {"n": 7, "k": 3, "d": 2}


@pytest.fixture
def error_channel() -> NDArray[np.int32]:
    """Return error channel."""
    return np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).astype(np.int32)


@pytest.fixture
def pcm() -> NDArray[np.int32]:
    """Return parity check matrix."""
    return np.array([[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 1, 1], [0, 0, 1, 0, 1, 0, 1]]).astype(np.int32)


@pytest.fixture
def atd(error_channel: NDArray[np.float64], pcm: NDArray[np.int32]) -> AnalogTannergraphDecoder:
    """Return distance of the hexagonal color code."""
    return AnalogTannergraphDecoder(
        pcm=pcm,
        bp_params=BpParams(osd_method="osd0"),
        error_channel=error_channel,
        sigma=0.1,
        ser=None,
    )


@pytest.fixture
def atd_simulator_sigma(pcm: NDArray[np.int32], code_params: dict[str, int]) -> AtdSimulator:
    """Return AtdSimulator using sigma to initialize syndrome channel."""
    return AtdSimulator(
        hx=pcm,
        hz=pcm,
        lx=np.array([1, 1, 1, 1, 1, 1, 1]),
        lz=np.array([1, 1, 1, 1, 1, 1, 1]),
        codename="test",
        data_err_rate=0.1,
        sigma=0.1,
        seed=666,
        bp_params=BpParams(osd_method="osd0"),
        decoding_method="atd",
        code_params=code_params,
    )


@pytest.fixture
def atd_simulator_ser(pcm: NDArray[np.int32], code_params: dict[str, int]) -> AtdSimulator:
    """Return AtdSimulator using error rate to initialize syndrome channel."""
    per = 0.1
    ser = 0.1
    return AtdSimulator(
        hx=pcm,
        hz=pcm,
        lx=np.array([1, 1, 1, 1, 1, 1, 1]),
        lz=np.array([1, 1, 1, 1, 1, 1, 1]),
        codename="test",
        data_err_rate=per,
        syndr_err_rate=ser,
        seed=666,
        bp_params=BpParams(osd_method="osd0"),
        decoding_method="atd",
        output_path="./res",
        code_params=code_params,
    )


def test_set_analog_syndrome(atd: AnalogTannergraphDecoder, error_channel: NDArray[np.float64]) -> None:
    """Test initialization of decoder channel probs with analog syndrome."""
    analog_syndr = np.array([0.1, 0.1, 0.1])
    expected_virtual_inits = np.array([2.0611536181902108e-09, 2.0611536181902108e-09, 2.0611536181902108e-09])
    expected = np.hstack([error_channel, expected_virtual_inits])

    res = atd.decode(analog_syndr)
    assert np.array_equal(atd.bposd_decoder.channel_probs, expected)
    assert res is not None
    assert len(res) == 10


def test_atd_simulator_data_error_channels_setup(pcm: NDArray[np.int32], code_params: dict[str, int]) -> None:
    """Test simulator data error channel computation and initialization."""
    per = 0.1
    sigma = 0.1
    sim = AtdSimulator(
        hx=pcm,
        hz=pcm,
        lx=np.array([]),
        lz=np.array([]),
        codename="test",
        data_err_rate=per,
        sigma=sigma,
        seed=666,
        bp_params=BpParams(osd_method="osd0"),
        decoding_method="atd",
        code_params=code_params,
    )
    expected_err_chnl = simulation_utils.error_channel_setup(
        error_rate=per,
        xyz_error_bias=np.array([0.1, 0.1, 0.1]).astype(np.float64),
        nr_qubits=pcm.shape[1],
    )
    assert np.array_equal(sim.x_decoder.error_channel, expected_err_chnl[0] + expected_err_chnl[1])
    assert np.array_equal(sim.z_decoder.error_channel, expected_err_chnl[2] + expected_err_chnl[1])


def test_atd_simulator_syndrome_error_channels_setup(atd_simulator_sigma: AtdSimulator) -> None:
    """Test AtdSimulator syndrome channel computation and initialization."""
    sigma = 0.1
    ser = simulation_utils.get_error_rate_from_sigma(sigma)
    expec_chnl = simulation_utils.error_channel_setup(
        error_rate=ser,
        xyz_error_bias=np.array([0.1, 0.1, 0.1]).astype(np.float64),
        nr_qubits=1,
    )
    assert atd_simulator_sigma.syndr_err_rate == simulation_utils.get_error_rate_from_sigma(sigma=sigma)
    assert atd_simulator_sigma.x_sigma == simulation_utils.get_sigma_from_syndr_er(expec_chnl[0][0] + expec_chnl[1][0])
    assert atd_simulator_sigma.z_sigma == simulation_utils.get_sigma_from_syndr_er(expec_chnl[2][0] + expec_chnl[1][0])


def test_atd_simulator_syndrome_error_channels_setup_ser(atd_simulator_ser: AtdSimulator) -> None:
    """Test AtdSimulator syndrome error computattion and initialization using error rate."""
    ser = 0.1
    expec_chnl = simulation_utils.error_channel_setup(
        error_rate=ser,
        xyz_error_bias=np.array([0.1, 0.1, 0.1]).astype(np.float64),
        nr_qubits=1,
    )
    assert atd_simulator_ser.x_sigma == simulation_utils.get_sigma_from_syndr_er(expec_chnl[0][0] + expec_chnl[1][0])
    assert atd_simulator_ser.z_sigma == simulation_utils.get_sigma_from_syndr_er(expec_chnl[2][0] + expec_chnl[1][0])


def test_single_sample(atd_simulator_ser: AtdSimulator) -> None:
    """Test single sample overall."""
    assert atd_simulator_ser.single_sample() is not None
    assert atd_simulator_ser.x_bp_iterations is not None
    assert atd_simulator_ser.z_bp_iterations is not None


def test_safe_results(atd_simulator_ser: AtdSimulator) -> None:
    """Test result saving."""
    with mock.patch("json.dump", return_value=True):
        res = atd_simulator_ser.save_results(1, 1, 1)
    assert res is not None
    assert res["code_K"] == 3
    assert res["code_N"] == 7


def test_run(atd_simulator_ser: AtdSimulator) -> None:
    """Test run method."""
    with mock.patch("json.dump", return_value=True):
        res = atd_simulator_ser.run(1)
    assert res is not None
    assert res["code_K"] == 3
    assert res["code_N"] == 7
