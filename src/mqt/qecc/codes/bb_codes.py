# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Construction of Bivariate Bicycle LDPC codes from https://arxiv.org/abs/2308.07915."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .css_code import CSSCode, InvalidCSSCodeError

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt


def construct_bb_code(n: int) -> CSSCode:
    """Construct the check matrices for a bivariate bicycle LDPC code.

    Args:
        n: The number of qubits. Currently supported values are 72, 90, 108, 144, and 288.
    """
    if n == 72:
        x, z = _cyclic_code_check_matrix(6, 6, ([3], [1, 2]), ([1, 2], [3]))
        d = 6
    elif n == 90:
        x, z = _cyclic_code_check_matrix(15, 3, ([9], [1, 2]), ([2, 7], [0]))
        d = 10
    elif n == 108:
        x, z = _cyclic_code_check_matrix(9, 6, ([3], [1, 2]), ([1, 2], [3]))
        d = 10
    elif n == 144:
        x, z = _cyclic_code_check_matrix(12, 6, ([3], [1, 2]), ([1, 2], [3]))
        d = 12
    elif n == 288:
        x, z = _cyclic_code_check_matrix(12, 12, ([3], [2, 7]), ([1, 2], [3]))
        d = 18
    else:
        msg = f"No bb code with n = {n}."
        raise InvalidCSSCodeError(msg)
    return CSSCode(x, z, d)


def _shift_matrix(l_: int) -> npt.NDArray[np.int8]:
    s = np.zeros((l_, l_), dtype=np.int8)  # type: npt.NDArray[np.int8]
    for i in range(l_):
        s[i, (i + 1) % l_] = 1
    return s


def _x_matrix(l_: int, m: int) -> npt.NDArray[np.int8]:
    return np.kron(_shift_matrix(l_), np.eye(m)).astype(np.int8)


def _y_matrix(l_: int, m: int) -> npt.NDArray[np.int8]:
    return np.kron(np.eye(l_), _shift_matrix(m)).astype(np.int8)


def _cyclic_matrix_sum(x_terms: list[int], y_terms: list[int], l_: int, m: int) -> npt.NDArray[np.int8]:
    x = _x_matrix(l_, m)
    y = _y_matrix(l_, m)
    a = np.zeros(x.shape)
    for power in x_terms:
        a += np.linalg.matrix_power(x, power)
    for power in y_terms:
        a += np.linalg.matrix_power(y, power)

    return a.astype(np.int8)


def _cyclic_code_check_matrix(
    m: int, l_: int, a_terms: tuple[list[int], list[int]], b_terms: tuple[list[int], list[int]]
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    a = _cyclic_matrix_sum(a_terms[0], a_terms[1], l_, m) % 2
    b = _cyclic_matrix_sum(b_terms[0], b_terms[1], l_, m) % 2
    return np.hstack((a, b)), np.hstack((b.T, a.T))
