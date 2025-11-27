# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utilities for the circuit synthesis unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ldpc.mod2.mod2_numpy import rank

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt


def eq_span(a: npt.NDArray[np.int8], b: npt.NDArray[np.int8]) -> bool:
    """Check if two matrices have the same row space."""
    return (a.shape[1] == b.shape[1]) and (int(rank(np.vstack((a, b)))) == int(rank(a)) == int(rank(b)))


def in_span(m: npt.NDArray[np.int8], v: npt.NDArray[np.int8]) -> bool:
    """Check if a vector is in the row space of a matrix over GF(2)."""
    return bool(rank(np.vstack((m, v))) == rank(m))
