# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
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
import pytest

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
from mqt.qecc.noise import PauliChannel

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


def test_noise_helper_input_validation() -> None:
    """Reject inconsistent legacy noise-helper inputs."""
    channel = (np.zeros(2), np.zeros(2), np.zeros(2))
    residual = [np.zeros(2, dtype=np.int32), np.zeros(2, dtype=np.int32)]
    with pytest.raises(ValueError, match="does not match channel size"):
        generate_err(1, channel, residual)
    with pytest.raises(ValueError, match="between 0 and 1"):
        generate_syndr_err(np.array([np.nan]))
    with pytest.raises(ValueError, match="binary"):
        get_noisy_analog_syndrome(np.array([2]), 0.1)
    with pytest.raises(ValueError, match="exactly three"):
        error_channel_setup(0.1, np.ones(2))


def test_explicit_generator_controls_noise_helpers() -> None:
    """Reproduce complete helper sequences with explicitly owned generators."""
    n = 100
    channel = (np.full(n, 0.2), np.full(n, 0.1), np.full(n, 0.3))
    residual = [np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)]
    first = generate_err(n, channel, residual, rng=np.random.default_rng(42))
    second = generate_err(n, channel, residual, rng=np.random.default_rng(42))
    assert np.array_equal(first, second)


def test_get_sigma_from_syndr_er() -> None:
    """Test get_sigma_from_syndr_er function."""
    ser = 0.1

    assert math.ceil(get_sigma_from_syndr_er(ser)) == math.ceil(0.780304146072379)
    ser = 1.0
    assert math.ceil(get_sigma_from_syndr_er(ser)) == pytest.approx(0.0, abs=1e-8)
    ser = 0.0
    assert math.ceil(get_sigma_from_syndr_er(ser)) == pytest.approx(0.0, abs=1e-8)


def test_get_error_rate_from_sigma() -> None:
    """Test get_error_rate_from_sigma function."""
    sigma = 0.3
    assert np.isclose([get_error_rate_from_sigma(sigma)], [0.00042906])
    sigma = 0.5
    assert np.isclose([get_error_rate_from_sigma(sigma)], [0.02275])
    sigma = 0.0
    assert get_error_rate_from_sigma(sigma) == pytest.approx(0.0, abs=1e-8)


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

    assert res[0] == pytest.approx(0.0, abs=1e-8)


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

    channel = error_channel_setup(p, np.array([1.0, 1.0, 1.0]))
    assert channel == PauliChannel(p / 3, p / 3, p / 3)

    assert error_channel_setup(p, np.array([1.0, 0.0, 0.0])) == PauliChannel(p, 0.0, 0.0)
    assert error_channel_setup(p, np.array([1.0, 1.0, 0.0])) == PauliChannel(p / 2, p / 2, 0.0)
    assert error_channel_setup(p, np.array([np.inf, 0.0, 0.0])) == PauliChannel(p, 0.0, 0.0)
    assert error_channel_setup(p, np.array([0.0, np.inf, 0.0])) == PauliChannel(0.0, p, 0.0)


def test_err_chnl_marginals() -> None:
    """The marginals are what decoders consume as priors."""
    channel = error_channel_setup(0.3, np.array([1.0, 1.0, 1.0]))
    assert channel.x_marginal == pytest.approx(0.2)  # p_x + p_y
    assert channel.z_marginal == pytest.approx(0.2)  # p_z + p_y


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
