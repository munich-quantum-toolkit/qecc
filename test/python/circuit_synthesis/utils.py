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
from ldpc import mod2

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt


def eq_span(a: npt.NDArray[np.int_], b: npt.NDArray[np.int_]) -> bool:
    """Check if two matrices have the same row space."""
    return (a.shape[1] == b.shape[1]) and (int(mod2.rank(np.vstack((a, b)))) == int(mod2.rank(a)) == int(mod2.rank(b)))


def in_span(m: npt.NDArray[np.int_], v: npt.NDArray[np.int_]) -> bool:
    """Check if a vector is in the row space of a matrix over GF(2)."""
    return bool(mod2.rank(np.vstack((m, v))) == mod2.rank(m))
