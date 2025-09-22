# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test quasi singleshot simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import numpy as np
import pytest

from mqt.qecc import QssSimulator
from mqt.qecc.analog_information_decoding.utils import simulation_utils
from mqt.qecc.analog_information_decoding.utils.data_utils import BpParams

if TYPE_CHECKING:
    from numpy.typing import NDArray


@pytest.fixture
def pcm() -> NDArray[np.int32]:
    """Return parity check matrix."""
    return np.array([[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 1, 1], [0, 0, 1, 0, 1, 0, 1]]).astype(np.int32)


@pytest.fixture
def code_params() -> dict[str, int]:
    """Return code parameters."""
    return {"n": 7, "k": 3, "d": 2}


@pytest.fixture
def qss_simulator(pcm: NDArray[np.int32], code_params: dict[str, int]) -> QssSimulator:
    """Return QSS simulator object."""
    return QssSimulator(
        pcm=pcm,
        per=0.1,
        ser=0.1,
        logicals=np.array([1, 1, 1, 1, 1, 1, 1]),
        bias=np.array([1.0, 1.0, 1.0]),
        codename="test",
        bp_params=BpParams(osd_method="osd0"),
        outpath="./results",
        repetitions=2,
        code_params=code_params,
        rounds=1,
    )


def test_err_channel_setup(qss_simulator: QssSimulator) -> None:
    """Test computation of error channels."""
    data_chnl = simulation_utils.error_channel_setup(
        error_rate=0.1, xyz_error_bias=np.array([1.0, 1.0, 1.0]), nr_qubits=7
    )
    syndr_chnl = simulation_utils.error_channel_setup(
        error_rate=0.1, xyz_error_bias=np.array([1.0, 1.0, 1.0]), nr_qubits=3
    )
    expected_syndr_chnl: NDArray[np.float64] = np.array(1.0 * (syndr_chnl[2] + syndr_chnl[1])).astype(np.float64)
    expected_data_chnl: NDArray[np.float64] = np.array(1.0 * (data_chnl[2] + data_chnl[1])).astype(np.float64)
    assert qss_simulator.err_idx == 1
    assert np.allclose(qss_simulator.data_err_channel, expected_data_chnl)
    assert np.allclose(qss_simulator.syndr_err_channel, expected_syndr_chnl)


def test_decoder_error_channel_setup(qss_simulator: QssSimulator) -> None:
    """Test simulator decoder error channel initialization."""
    expected = np.full((1, 20), 0.06666667)
    assert np.allclose(qss_simulator.decoder.channel_probs, expected)


def test_single_sample(qss_simulator: QssSimulator) -> None:
    """Test single sample overall."""
    with mock.patch("pathlib.Path.open"), mock.patch("json.dump"):
        res = qss_simulator.run(1)
    assert res is not None
    assert res["pers"] == 0.1
    assert res["code_N"] == 7
