# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the concatenated code classes."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc import CSSCode, StabilizerCode
from mqt.qecc.codes import ConcatenatedCode, ConcatenatedCSSCode, InvalidStabilizerCodeError


@pytest.fixture
def steane_code() -> CSSCode:
    """Return the Steane code."""
    hx = np.array([[1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 1, 0]])
    hz = hx
    return CSSCode(distance=3, Hx=hx, Hz=hz)


def test_trivial_css_concatenation(steane_code: CSSCode) -> None:
    """Test that the trivial concatenation of a CSS code is the code itself."""
    inner_code = CSSCode.get_trivial_code(1)
    concatenated = ConcatenatedCSSCode(steane_code, inner_code)

    assert concatenated.n == 7
    assert concatenated.k == 1
    assert concatenated.distance == 3
    assert concatenated == steane_code


@pytest.fixture
def five_qubit_code() -> StabilizerCode:
    """Return the [[5, 1, 3]] code."""
    return StabilizerCode(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"], 3, z_logicals=["ZZZZZ"], x_logicals=["XXXXX"])


@pytest.fixture
def bit_flip_code() -> StabilizerCode:
    """Return the [[3, 1, 1]] bit-flip repetition code."""
    return StabilizerCode(["ZZI", "IZZ"], 1, z_logicals=["ZII"], x_logicals=["XXX"])


def test_css_concatenation_keeps_inner_stabilizers(steane_code: CSSCode) -> None:
    """Concatenating the Steane code with itself gives a [[49, 1]] code that contains every inner check."""
    concatenated = ConcatenatedCSSCode(steane_code, steane_code)

    assert concatenated.n == 49
    assert concatenated.k == 1
    assert concatenated.distance == 9
    for block in range(7):
        for checks, all_checks in ((steane_code.Hx, concatenated.Hx), (steane_code.Hz, concatenated.Hz)):
            for check in checks:
                physical = np.zeros(49, dtype=np.int8)
                physical[7 * block : 7 * block + 7] = check
                assert any(np.array_equal(physical, row) for row in all_checks)


def test_css_concatenation_maps_outer_logicals(steane_code: CSSCode) -> None:
    """The logical operators of a concatenated CSS code are the outer logicals mapped through the inner logicals."""
    concatenated = ConcatenatedCSSCode(steane_code, steane_code)

    expected_lx = np.zeros(49, dtype=np.int8)
    expected_lz = np.zeros(49, dtype=np.int8)
    for block in np.flatnonzero(steane_code.Lx[0]):
        expected_lx[7 * block : 7 * block + 7] = steane_code.Lx[0]
    for block in np.flatnonzero(steane_code.Lz[0]):
        expected_lz[7 * block : 7 * block + 7] = steane_code.Lz[0]
    np.testing.assert_array_equal(concatenated.Lx[0], expected_lx)
    np.testing.assert_array_equal(concatenated.Lz[0], expected_lz)


def test_stabilizer_concatenation_keeps_inner_stabilizers(
    five_qubit_code: StabilizerCode, bit_flip_code: StabilizerCode
) -> None:
    """Concatenating the [[5, 1, 3]] code with a [[3, 1, 1]] code gives a code with one logical qubit."""
    concatenated = ConcatenatedCode(five_qubit_code, bit_flip_code)

    assert concatenated.n == 15
    assert concatenated.k == 1
    assert concatenated.generators.n_rows == 5 * 2 + 4


def test_concatenation_keeps_logical_types(five_qubit_code: StabilizerCode) -> None:
    """Concatenation keeps logical X operators as logical X operators and logical Z operators as logical Z operators."""
    concatenated = ConcatenatedCode(five_qubit_code, StabilizerCode.get_trivial_code(1))

    assert concatenated.x_logicals[0] == five_qubit_code.x_logicals[0]
    assert concatenated.z_logicals[0] == five_qubit_code.z_logicals[0]


@pytest.mark.parametrize("num_inner_codes", [4, 6])
def test_concatenation_rejects_wrong_number_of_inner_codes(
    five_qubit_code: StabilizerCode, num_inner_codes: int
) -> None:
    """A list of inner codes must contain one code for each qubit of the outer code."""
    inner_codes = [StabilizerCode.get_trivial_code(1)] * num_inner_codes
    with pytest.raises(InvalidStabilizerCodeError, match="number of inner codes"):
        ConcatenatedCode(five_qubit_code, inner_codes)


def test_css_concatenation_rejects_wrong_number_of_inner_codes(steane_code: CSSCode) -> None:
    """A list of inner CSS codes must contain one code for each qubit of the outer code."""
    with pytest.raises(InvalidStabilizerCodeError, match="number of inner codes"):
        ConcatenatedCSSCode(steane_code, [CSSCode.get_trivial_code(1)] * 6)
