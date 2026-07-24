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


@pytest.mark.parametrize("n", [72, 90, 108, 144, 288])
def test_bb_codes(n: int) -> None:
    """Test that BB codes are constructed as valid CSS codes."""
    code = construct_bb_code(n)
    assert code.n == n
    assert code.Hx is not None
    assert code.Hz is not None
    assert np.all(code.Hx @ code.Hz.T % 2 == 0)
