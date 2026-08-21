# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test shared equivalence-checking helpers."""

from __future__ import annotations

import numpy as np
import pytest
import z3

from mqt.qecc.equivalence_checking.utils import elementwise_map


def test_elementwise_map_rejects_length_mismatch() -> None:
    """A SAT vector equality must not silently omit unmatched entries."""
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter"):
        elementwise_map(np.array([0, 1], dtype=np.uint8), [z3.Bool("x")])
