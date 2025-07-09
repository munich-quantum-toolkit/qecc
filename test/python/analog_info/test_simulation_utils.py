# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the simulation_utils module."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from mqt.qecc.analog_information_decoding.utils.simulation_utils import (
    build_single_stage_pcm,
    check_logical_err_h,
    error_channel_setup,
    generate_err,
    generate_syndr_err,
    get_analog_llr,
    get_binary_from_analog,
    get_error_rate_from_sigma,
    get_noisy_analog_syndrome,
    get_sigma_from_syndr_er,
    get_signed_from_binary,
    get_virtual_check_init_vals,
    is_logical_err,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_check_logical_err_h() -> None:
    """Test check_logical_err_h function."""
    h = np.array([[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1, 1]])
    # check with logical
    estimate = np.array([1, 0, 0, 0, 0, 0, 1])
    assert check_logical_err_h(h, np.array([0, 0, 0, 0, 1, 1, 0]), estimate) is True
    #
    # check with stabilizer
    estimate2 = np.array([0, 0, 0, 0, 0, 0, 1])
    assert check_logical_err_h(h, np.array([1, 1, 1, 0, 0, 0, 0]), estimate2) is False

    # check with all zeros
    estimate3 = np.array([0, 0, 0, 0, 0, 0, 0])
    assert check_logical_err_h(h, np.array([0, 0, 0, 0, 0, 0, 0]), estimate3) is False


def test_is_logical_err() -> None:
    """Test is_logical_err function."""
    # check with logical
    l_sc = np.array([
        [
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    ])
    residual = np.array([
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ])

    assert is_logical_err(l_sc, residual) is True

    # check with stabilizer
    residual2 = np.array([
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ])
    assert is_logical_err(l_sc, residual2) is False

    # check with all zeros
    residual2 = np.array([
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ])
    assert is_logical_err(l_sc, residual2) is False

    # check with non-min weight logical
    residual3 = np.array([
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ])
    assert is_logical_err(l_sc, residual3) is True


def test_get_analog_llr() -> None:
    """Test get_analog_llr function."""
    analog_syndr = np.array([0.5, 0, 0, -1, 0, 1])
    sigma = 0.8

    assert np.allclose(get_analog_llr(analog_syndr, sigma), np.array([1.5625, 0.0, 0.0, -3.125, 0.0, 3.125]))

    sigma = 0.1
    assert np.allclose(get_analog_llr(analog_syndr, sigma), np.array([100, 0.0, 0.0, -200.0, 0.0, 200.0]))


def test_generate_err() -> None:
    """Test generate_err function."""
    # no errors
    p = 0.0
    n = 10
    ch = np.ones(n) * p
    channel = np.copy(ch), np.copy(ch), np.copy(ch)
    residual: list[NDArray[np.int32]] = [np.zeros(n).astype(np.int32), np.zeros(n).astype(np.int32)]

    expected = np.array([np.zeros(n).astype(np.float64), np.zeros(n).astype(np.float64)])
    assert np.array_equal(generate_err(n, channel, residual), expected)

    residual[0][0] = 1
    residual[1][0] = 1

    expected = np.array([np.copy(residual[0]), np.copy(residual[1])])
    res = generate_err(n, channel, residual)
    assert np.array_equal(res[0], expected[0])
    assert np.array_equal(res[1], expected[1])


def test_get_sigma_from_syndr_er() -> None:
    """Test get_sigma_from_syndr_er function."""
    ser = 0.1

    assert math.ceil(get_sigma_from_syndr_er(ser)) == math.ceil(0.780304146072379)
    ser = 1.0
    assert math.ceil(get_sigma_from_syndr_er(ser)) == -0.0
    ser = 0.0
    assert math.ceil(get_sigma_from_syndr_er(ser)) == 0.0


def test_get_error_rate_from_sigma() -> None:
    """Test get_error_rate_from_sigma function."""
    sigma = 0.3
    assert np.isclose([get_error_rate_from_sigma(sigma)], [0.00042906])
    sigma = 0.5
    assert np.isclose([get_error_rate_from_sigma(sigma)], [0.02275])
    sigma = 0.0
    assert get_error_rate_from_sigma(sigma) == 0.0


def test_get_virtual_check_init_vals() -> None:
    """Test get_virtual_check_init_vals function."""
    noisy_syndr = np.array([0.5, 0, 0, -1, 0, 10])
    sigma = 0.8

    assert np.allclose(
        get_virtual_check_init_vals(noisy_syndr, sigma),
        np.array([
            1.73288206e-001,
            5.00000000e-001,
            5.00000000e-001,
            4.20877279e-002,
            5.00000000e-001,
            1.91855567e-136,
        ]),
    )

    sigma = 0.2
    assert np.allclose(
        get_virtual_check_init_vals(noisy_syndr, sigma),
        np.array([3.72007598e-44, 5.00000000e-01, 5.00000000e-01, 1.38389653e-87, 5.00000000e-01, 0.00000000e00]),
    )
    sigma = 0.0
    res = get_virtual_check_init_vals(noisy_syndr, sigma)

    assert res[0] == 0.0


def test_generate_syndr_err() -> None:
    """Test generate_syndr_err function."""
    channel = np.array([0.0, 1.0])

    assert np.array_equal(generate_syndr_err(channel), np.array([0.0, 1.0]))

    channel = np.array([0.0, 0.0, 0.0])
    assert np.array_equal(generate_syndr_err(channel), np.zeros_like(channel).astype(float))

    channel = np.array([1.0, 1.0, 1.0])
    assert np.array_equal(generate_syndr_err(channel), np.ones_like(channel).astype(float))


def test_get_noisy_analog_syndr() -> None:
    """Test get_noisy_analog_syndr function."""
    perfect_s = np.array([1, 0])
    sigma = 0.0

    assert np.array_equal(get_noisy_analog_syndrome(perfect_s, sigma), np.array([-1.0, 1.0]))


def test_err_chnl_setup() -> None:
    """Test error_channel_setup function."""
    p = 0.1
    bias = np.array([1.0, 1.0, 1.0])
    n = 10
    ar = np.ones(n) * p / 3
    exp = np.array([np.copy(ar), np.copy(ar), np.copy(ar)])
    res = error_channel_setup(p, bias, n)

    assert np.array_equal(res, exp)

    bias = np.array([1.0, 0.0, 0.0])
    ar = np.ones(n) * p
    exp = np.array([np.copy(ar), np.zeros(n), np.zeros(n)])
    res = error_channel_setup(p, bias, n)

    assert np.array_equal(res, exp)

    bias = np.array([1.0, 1.0, 0.0])
    ar = np.ones(n) * p / 2
    exp = np.array([np.copy(ar), np.copy(ar), np.zeros(n)])
    res = error_channel_setup(p, bias, n)

    assert np.array_equal(res, exp)

    bias = np.array([np.inf, 0.0, 0.0])
    ar = np.ones(n) * p
    exp = np.array([np.copy(ar), np.zeros(n), np.zeros(n)])
    res = error_channel_setup(p, bias, n)

    assert np.array_equal(res, exp)

    bias = np.array([0.0, np.inf, 0.0])
    ar = np.ones(n) * p
    exp = np.array([np.zeros(n), np.copy(ar), np.zeros(n)])
    res = error_channel_setup(p, bias, n)

    assert np.array_equal(res, exp)


def test_build_ss_pcm() -> None:
    """Test build_single_stage_pcm function."""
    h = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])
    m = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])
    id_r = np.identity(m.shape[1])
    zeros = np.zeros((m.shape[0], h.shape[1]))
    exp = np.block([[h, id_r], [zeros, m]])
    assert np.array_equal(build_single_stage_pcm(h, m), exp)


def test_get_signed_from_binary() -> None:
    """Test get_signed_from_binary function."""
    binary = np.array([1, 0, 0, 1, 0, 1])
    exp = np.array([-1, 1, 1, -1, 1, -1])

    assert np.array_equal(get_signed_from_binary(binary), exp)


def test_get_binary_from_analog() -> None:
    """Test get_binary_from_analog function."""
    exp = np.array([1, 0, 0, 1, 0, 1])
    analog = np.array([-1.0, 3.0, 1.0, -1.0, 1.0, -2])

    assert np.array_equal(get_binary_from_analog(analog), exp)
