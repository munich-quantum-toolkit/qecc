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

from mqt.qecc import CSSCode
from mqt.qecc.codes import ConcatenatedCSSCode


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
