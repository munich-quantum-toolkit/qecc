# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the CSSCode class."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc.codes import construct_bb_code


@pytest.mark.parametrize(("n", "k", "d"), [(72, 12, 6), (90, 8, 10), (108, 8, 10), (144, 12, 12), (288, 12, 18)])
def test_bb_codes(n: int, k: int, d: int) -> None:
    """Test that BB codes are constructed as valid CSS codes with the expected parameters."""
    code = construct_bb_code(n)
    assert code.n == n
    assert code.k == k
    assert code.distance == d
    assert code.Hx is not None
    assert code.Hz is not None
    assert np.all(code.Hx @ code.Hz.T % 2 == 0)


@pytest.mark.parametrize("n", [72, 90, 108, 144, 288])
def test_bb_codes_have_no_low_weight_logicals(n: int) -> None:
    """Test that BB codes admit no logical operator of weight at most two.

    A weight-one or weight-two operator can only commute with every check if the
    corresponding check matrix has a zero column or two identical columns. Ruling
    that out certifies a distance of at least three, which catches a swap of the
    two cyclic group orders l and m.
    """
    code = construct_bb_code(n)
    for h in (code.Hx, code.Hz):
        assert np.all(h.any(axis=0)), "check matrix has a zero column"
        assert np.unique(h, axis=1).shape[1] == h.shape[1], "check matrix has two identical columns"
