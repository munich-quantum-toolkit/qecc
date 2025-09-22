# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the memory experiment simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ldpc.osd import bposd_decoder

from mqt.qecc.analog_information_decoding.simulators.memory_experiment_v2 import (
    build_multiround_pcm,
    decode_multiround,
    move_syndrome,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.sparse import csr_matrix


@pytest.fixture
def pcm() -> NDArray[np.int32]:
    """Fixture for parity check matrix of a rep code."""
    return np.array([[1, 1, 0], [0, 1, 1]])


@pytest.fixture
def repetitions() -> int:
    """Fixture for number of repetitions for multiround decoding."""
    return 4


@pytest.fixture
def h3d(pcm: NDArray[np.int32], repetitions: int) -> csr_matrix:
    """Fixture for multiround parity check matrix."""
    return build_multiround_pcm(pcm, repetitions - 1)


@pytest.fixture
def channel_probs(repetitions: int, pcm: NDArray[np.int32]) -> NDArray[np.float64]:
    """Fixture for error channel."""
    return np.full(shape=repetitions * pcm.shape[1] + repetitions * pcm.shape[0], fill_value=0.1)


@pytest.fixture
def decoder(channel_probs: NDArray[np.float64], h3d: NDArray[np.int32]) -> bposd_decoder:
    """Fixture for decoder."""
    return bposd_decoder(
        parity_check_matrix=h3d,
        channel_probs=channel_probs,
        max_iter=15,
        bp_method="msl",
        osd_order=0,
        osd_method="osd0",
        ms_scaling_factor=0.5,
    )


def test_build_mr_pcm() -> None:
    """Test build_multiround_pcm function."""
    pcm: NDArray[np.int32] = np.array([[1, 1, 0], [0, 1, 1]]).astype(np.int32)
    mr_pcm = build_multiround_pcm(pcm, 1)
    np.zeros((2, 3))
    np.identity(2)
    r1 = np.hstack([pcm, np.zeros(pcm.shape), np.identity(2), np.zeros((2, 2))])
    r2 = np.hstack([np.zeros(pcm.shape), pcm, np.identity(2), np.identity(2)])
    expected = np.vstack((r1, r2))

    assert np.array_equal(mr_pcm.toarray(), expected)


def test_move_syndrome() -> None:
    """Test move_syndrome function."""
    # three bit syndrome over 4 rounds
    syndr = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    res = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    assert np.array_equal(res, move_syndrome(syndr))

    syndr = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
    res = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])
    assert np.array_equal(res, move_syndrome(syndr))

    syndr = np.array([[0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0]])
    res = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]])
    assert np.array_equal(res, move_syndrome(syndr))


def test_decode_multiround_syndr_err(
    pcm: NDArray[np.int32], channel_probs: NDArray[np.float64], repetitions: int, decoder: bposd_decoder
) -> None:
    """Test decoding of multiround syndrome for three bit repetition code."""
    check_block_size = pcm.shape[1] * repetitions

    analog_syndr = np.array([[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    sigma = 0.3
    decoding_method = "bposd"
    syndrome = np.array([[0, 1, 0, 0], [0, 0, 0, 0]])
    res = decode_multiround(
        pcm=pcm,
        channel_probs=channel_probs,
        analog_syndr=analog_syndr,
        decoder=decoder,
        syndrome=syndrome,
        repetitions=repetitions,
        last_round=True,
        check_block_size=check_block_size,
        sigma=sigma,
        decoding_method=decoding_method,
    )
    assert np.array_equal(res[0], np.array([0, 0, 0]))  # estimate is all zeros
    assert np.array_equal(res[1], syndrome)
    assert np.array_equal(res[2], analog_syndr)


def test_decode_multiround_data_err(
    pcm: NDArray[np.int32],
    channel_probs: NDArray[np.float64],
    repetitions: int,
    h3d: csr_matrix,
) -> None:
    """Test decoding of multiround syndrome for three bit repetition code."""
    assert h3d is not None
    check_block_size = pcm.shape[1] * repetitions
    analog_syndr = np.array([[-1.0, -1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0]])
    sigma = 0.3
    decoding_method = "matching"
    syndrome = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    res = decode_multiround(
        pcm=pcm,
        channel_probs=channel_probs,
        analog_syndr=analog_syndr,
        decoder=decoder,
        syndrome=syndrome,
        repetitions=repetitions,
        last_round=False,
        check_block_size=check_block_size,
        sigma=sigma,
        decoding_method=decoding_method,
        h3d=h3d,
    )
    assert np.array_equal(res[0], np.array([1, 0, 0]))  # data error on qubit 1
    assert np.array_equal(res[1], syndrome)
    assert np.array_equal(res[2], analog_syndr)
