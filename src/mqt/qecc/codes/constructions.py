# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Constructions of various known stabilizer codes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .css_code import CSSCode

if TYPE_CHECKING:
    import numpy.typing as npt


def construct_quantum_hamming_code(r: int) -> CSSCode:
    """Return the [[2^r, 2^r-r-1, 3]] quantum Hamming code."""
    h = _hamming_code_checks(r)
    return CSSCode(h, h, 3)


def construct_iceberg_code(m: int) -> CSSCode:
    """Return the [[2m, 2m-2, 2]] Iceberg code.

    The Iceberg code is a CSS code with stabilizer generators X^2m and Z^2m.
    https://errorcorrectionzoo.org/c/iceberg
    """
    n = 2 * m
    h = np.array([[1] * n], dtype=np.int8)
    return CSSCode(h, h, 2)


def construct_many_hypercube_code(level: int) -> CSSCode:
    """Return the [[6^l, 4^l, 2^l]] level l many-hypercube code (https://arxiv.org/abs/2403.16054).

    This code is obtained by (l-1)-fold concatenation of the [[6,4,2]] iceberg code with itself.
    """
    code = construct_iceberg_code(3)

    for _ in range(1, level):
        sx = np.hstack([code.Lx] * 6, dtype=np.int8)
        sx_rem = np.kron(np.eye(6, dtype=np.int8), code.Hx)
        sx = np.vstack((sx, sx_rem), dtype=np.int8)
        sz = sx
        code = CSSCode(sx, sz, code.distance * 2)
    return code


def _hamming_code_checks(r: int) -> npt.NDArray[np.int8]:
    """Return the check matrix for the [2^r-1, 2^r-r-1, 3] Hamming code."""
    n = 2**r - 1
    h = np.zeros((r, n), dtype=int)
    # columns are all binary strings up to 2^r
    for i in range(1, n + 1):
        h[:, i - 1] = np.array([int(x) for x in f"{i:b}".zfill(r)])

    return h
