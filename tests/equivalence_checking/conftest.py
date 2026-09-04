# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared helpers for equivalence-checking tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.qecc.mod2 import is_in_row_space, rank

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


def assert_same_row_space(
    transformed: npt.NDArray[np.integer],
    target: npt.NDArray[np.integer],
) -> None:
    """Assert that two matrices span the same row space."""
    assert rank(transformed) == rank(target)
    assert all(is_in_row_space(row, target) for row in transformed)
