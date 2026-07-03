# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Functions for deciding permutation equivalence."""

from __future__ import annotations

from collections import Counter
from itertools import permutations
from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2
import numpy as np

from ..codes.css_code import CSSCode

if TYPE_CHECKING:  # pragma: no cover
    from ..codes.stabilizer_code import StabilizerCode

BRUTEFORCE_THRESHOLD_STB = 5
BRUTEFORCE_THRESHOLD_CSS = 7


def are_permutation_equivalent(code1: StabilizerCode | CSSCode, code2: StabilizerCode | CSSCode) -> bool:
    """Check if two stabilizer codes are permutation equivalent."""
    cheap_invariants = (
        preserved_n,
        preserved_k,
        preserved_d,
        preserved_number_zero_columns,
        preserved_number_duplicate_columns,
    )

    if not all(invariant(code1, code2) for invariant in cheap_invariants):
        return False

    if code1.n < 1:
        return True

    if isinstance(code1, CSSCode) and isinstance(code2, CSSCode):
        return css_p_eq(code1, code2)
    return stb_p_eq(code1, code2)


# ----------------------------------------------------------------------------------------------------
#   Algorithms
# ----------------------------------------------------------------------------------------------------


def css_p_eq(code1: CSSCode, code2: CSSCode) -> bool:
    """Check if two CSS codes are permutation equivalent."""
    reduced_hx1 = _row_basis(code1.Hx)
    reduced_hz1 = _row_basis(code1.Hz)

    reduced_hx2 = _row_basis(code2.Hx)
    reduced_hz2 = _row_basis(code2.Hz)

    if reduced_hx1.shape[0] == 0 and reduced_hz1.shape[0] == 0:
        return True

    if reduced_hx2.shape[0] != reduced_hz2.shape[0] or reduced_hx1.shape[0] != reduced_hz1.shape[0]:
        return False

    if code1.n <= BRUTEFORCE_THRESHOLD_CSS:
        return _bruteforce_css(reduced_hx1, reduced_hz1, reduced_hx2, reduced_hz2)

    symplectic1 = np.hstack([
        np.vstack([reduced_hx1, np.zeros_like(reduced_hz1)]),
        np.vstack([np.zeros_like(reduced_hx1), reduced_hz1]),
    ])
    symplectic2 = np.hstack([
        np.vstack([reduced_hx2, np.zeros_like(reduced_hz2)]),
        np.vstack([np.zeros_like(reduced_hx2), reduced_hz2]),
    ])

    if not preserved_linear_dependencies(symplectic1, symplectic2):
        return False

    raise NotImplementedError


def stb_p_eq(code1: StabilizerCode, code2: StabilizerCode) -> bool:
    """Check if two stabilizer codes are permutation equivalent."""
    reduced_symplectic_1 = _row_basis(code1.symplectic)
    reduced_symplectic_2 = _row_basis(code2.symplectic)

    if reduced_symplectic_1.shape[0] == 0 and reduced_symplectic_2.shape[0] == 0:
        return True

    if reduced_symplectic_1.shape[0] != reduced_symplectic_2.shape[0]:
        return False

    if code1.n <= BRUTEFORCE_THRESHOLD_STB:
        return _bruteforce_stb(reduced_symplectic_1, reduced_symplectic_2)

    if not preserved_linear_dependencies(reduced_symplectic_1, reduced_symplectic_2):
        return False

    raise NotImplementedError


# ----------------------------------------------------------------------------------------------------
#   Prototypes
# ----------------------------------------------------------------------------------------------------


def _bruteforce_css(hx1: np.ndarray, hz1: np.ndarray, hx2: np.ndarray, hz2: np.ndarray) -> bool:
    """Brute force check for permutation equivalence of two CSS codes."""
    n = hx1.shape[1]

    hx_rank = hx1.shape[0]
    hz_rank = hz1.shape[0]

    for perm in permutations(range(n)):
        if hx_rank and hx_rank != mod2.rank(np.vstack([hx1, hx2[:, perm]])):
            continue
        if hz_rank and hz_rank != mod2.rank(np.vstack([hz1, hz2[:, perm]])):
            continue
        return True

    return False


def _bruteforce_stb(c1: np.ndarray, c2: np.ndarray) -> bool:
    """Brute force check for permutation equivalence of two stabilizer codes."""
    c1_rank = c1.shape[0]
    n = c1.shape[1] // 2

    for perm in permutations(range(n)):
        perm_symplectic = perm + tuple(q + n for q in perm)

        if c1_rank == mod2.rank(np.vstack([c1, c2[:, perm_symplectic]])):
            return True

    return False


# ----------------------------------------------------------------------------------------------------
#   Invariants
# ----------------------------------------------------------------------------------------------------


def preserved_n(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check whether the number of qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.n == c2.n


def preserved_k(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check whether the number of logical qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.k == c2.k


def preserved_d(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check whether the distance is preserved, which is a necessary condition for P-equivalence."""
    if isinstance(c1, CSSCode) and isinstance(c2, CSSCode):
        return c1.x_distance == c2.x_distance and c1.z_distance == c2.z_distance
    return c1.distance == c2.distance


def preserved_number_zero_columns(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check whether the number of zero columns is preserved, which is a necessary condition for P-equivalence."""
    return int(np.count_nonzero(np.all(c1.symplectic == 0, axis=0))) == int(
        np.count_nonzero(np.all(c2.symplectic == 0, axis=0))
    )


def preserved_number_duplicate_columns(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check whether the number of duplicate columns is preserved, which is a necessary condition for P-equivalence."""

    def _duplicate_column(m: np.ndarray) -> list[int]:
        columns = [tuple(m[:, j].tolist()) for j in range(m.shape[1])]
        counts = Counter(columns)
        return sorted(counts.values())

    return _duplicate_column(c1.symplectic) == _duplicate_column(c2.symplectic)


def preserved_linear_dependencies(c1: np.ndarray, c2: np.ndarray) -> bool:
    """Check whether the linear dependencies between columns are preserved, which is a necessary condition for P-equivalence."""

    def _linear_dependencies(m: np.ndarray) -> tuple[list[int], list[int], list[int]]:
        n = m.shape[1] // 2

        one_columns = [_rank(np.column_stack([m[:, q], m[:, q + n]])) for q in range(n)]

        two_columns = [
            _rank(np.column_stack([m[:, i], m[:, i + n], m[:, j], m[:, j + n]]))
            for i in range(n)
            for j in range(i + 1, n)
        ]
        three_columns = [
            _rank(np.column_stack([m[:, i], m[:, i + n], m[:, j], m[:, j + n], m[:, k], m[:, k + n]]))
            for i in range(n)
            for j in range(i + 1, n)
            for k in range(j + 1, n)
        ]

        return (sorted(one_columns), sorted(two_columns), sorted(three_columns))

    return _linear_dependencies(c1) == _linear_dependencies(c2)


# ----------------------------------------------------------------------------------------------------
#   Helpers
# ----------------------------------------------------------------------------------------------------


def _rank(matrix: np.ndarray) -> int:
    if matrix.shape[0] == 0:
        return 0
    return mod2.rank(matrix)


def _kernel_basis(a: np.ndarray) -> np.ndarray:
    a = (np.asarray(a) & 1).astype(np.uint8)
    k = mod2.nullspace(a)
    if hasattr(k, "toarray"):
        k = k.toarray()
    k = (np.asarray(k) & 1).astype(np.uint8)
    if k.size == 0:
        return np.zeros((0, a.shape[1]), dtype=np.uint8)
    if k.ndim == 1:
        k = k.reshape(1, -1)
    if k.shape[1] != a.shape[1]:
        msg = "Kernel basis must have the same number of columns as the input matrix."
        raise ValueError(msg)
    return k


def _row_basis(m: np.ndarray) -> np.ndarray:
    m = (np.asarray(m) & 1).astype(np.uint8)
    if m.size == 0:
        return np.zeros((0, m.shape[1]), dtype=np.uint8)
    b = mod2.row_basis(m)
    if hasattr(b, "toarray"):
        b = b.toarray()
    b = (np.asarray(b) & 1).astype(np.uint8)
    if b.size == 0:
        return np.zeros((0, m.shape[1]), dtype=np.uint8)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return b
