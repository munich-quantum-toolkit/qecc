# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Functions for deciding permutation equivalence."""

from __future__ import annotations

import hashlib
import operator
from collections import Counter, defaultdict
from itertools import permutations
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import z3

from ..codes.core.css_code import CSSCode
from ..mod2 import nullspace, rank, row_basis

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy.typing as npt

    from ..codes.core.stabilizer_code import StabilizerCode


BRUTEFORCE_THRESHOLD_STB = 5
BRUTEFORCE_THRESHOLD_CSS = 5


def are_permutation_equivalent(code1: StabilizerCode | CSSCode, code2: StabilizerCode | CSSCode) -> list[int] | None:
    """Check if two stabilizer codes define the same code space up to a permutation of output qubits, not considering phase information.

    Args:
        code1: First stabilizer code.
        code2: Second stabilizer code.

    Returns:
        A list of integers representing the permutation of qubits that maps code1 to code2, or None if no such permutation exists.
    """
    cheap_invariants = (
        _preserved_n,
        _preserved_k,
        _preserved_d,
        _preserved_number_zero_columns,
        _preserved_number_duplicate_columns,
    )

    if not all(invariant(code1, code2) for invariant in cheap_invariants):
        return None

    if isinstance(code1, CSSCode) and isinstance(code2, CSSCode):
        return _permutation_eq_css_codes(code1, code2)
    return _permutation_eq_stabilizer_codes(code1, code2)


# ----------------------------------------------------------------------------------------------------
#   Algorithms
# ----------------------------------------------------------------------------------------------------


def _permutation_eq_css_codes(code1: CSSCode, code2: CSSCode) -> list[int] | None:
    """Check if two CSS codes are permutation equivalent."""
    reduced_hx1 = row_basis(code1.Hx)
    reduced_hz1 = row_basis(code1.Hz)

    reduced_hx2 = row_basis(code2.Hx)
    reduced_hz2 = row_basis(code2.Hz)

    if reduced_hx1.shape[0] == 0 and reduced_hz1.shape[0] == 0:
        return list(range(code1.n))

    if reduced_hx1.shape[0] != reduced_hx2.shape[0] or reduced_hz1.shape[0] != reduced_hz2.shape[0]:
        return None

    if code1.n <= BRUTEFORCE_THRESHOLD_CSS:
        return _bruteforce_css(reduced_hx1, reduced_hz1, reduced_hx2, reduced_hz2)

    if code1.n >= 20:
        symplectic1 = np.hstack([
            np.vstack([reduced_hx1, np.zeros_like(reduced_hz1)]),
            np.vstack([np.zeros_like(reduced_hx1), reduced_hz1]),
        ])
        symplectic2 = np.hstack([
            np.vstack([reduced_hx2, np.zeros_like(reduced_hz2)]),
            np.vstack([np.zeros_like(reduced_hx2), reduced_hz2]),
        ])
        if not _preserved_linear_dependencies(symplectic1, symplectic2):
            return None

    result, partition1, partition2 = _preserved_punctured_hull_weight_enumerator_css_code(
        reduced_hx1,
        reduced_hz1,
        reduced_hx2,
        reduced_hz2,
    )
    if not result:
        return None

    assert partition1 is not None
    assert partition2 is not None

    if code1.n <= 17:
        return _matroid_css_code(reduced_hx1, reduced_hz1, partition1, reduced_hx2, reduced_hz2, partition2)
    if code1.n < 30:
        r = reduced_hx1.shape[0] + reduced_hz1.shape[0]
        return (
            _matroid_css_code(reduced_hx1, reduced_hz1, partition1, reduced_hx2, reduced_hz2, partition2)
            if r < 10
            else _sat_css_code(reduced_hx1, reduced_hz1, partition1, reduced_hx2, reduced_hz2, partition2)
        )
    return _sat_css_code(reduced_hx1, reduced_hz1, partition1, reduced_hx2, reduced_hz2, partition2)


def _permutation_eq_stabilizer_codes(code1: StabilizerCode, code2: StabilizerCode) -> list[int] | None:
    """Check if two stabilizer codes are permutation equivalent."""
    reduced_symplectic_1 = row_basis(code1.symplectic)
    reduced_symplectic_2 = row_basis(code2.symplectic)

    if reduced_symplectic_1.shape[0] == 0 and reduced_symplectic_2.shape[0] == 0:
        return list(range(code1.n))

    if reduced_symplectic_1.shape[0] != reduced_symplectic_2.shape[0]:
        return None

    if code1.n <= BRUTEFORCE_THRESHOLD_STB:
        return _bruteforce_stb(reduced_symplectic_1, reduced_symplectic_2)

    if not _preserved_linear_dependencies(reduced_symplectic_1, reduced_symplectic_2):
        return None

    partition1 = {(): list(range(code1.n))}
    partition2 = {(): list(range(code2.n))}
    if code1.n <= 20:
        result, refined_partition1, refined_partition2 = _preserved_punctured_hull_weight_enumerator_stabilizer_code(
            reduced_symplectic_1, reduced_symplectic_2
        )

        if not result:
            return None

        partition1 = refined_partition1
        partition2 = refined_partition2

    assert partition1 is not None
    assert partition2 is not None
    return _sat_stabilizer_code(reduced_symplectic_1, partition1, reduced_symplectic_2, partition2)


# ----------------------------------------------------------------------------------------------------
#   Brute force algorithms
# ----------------------------------------------------------------------------------------------------


def _bruteforce_css(hx1: np.ndarray, hz1: np.ndarray, hx2: np.ndarray, hz2: np.ndarray) -> list[int] | None:
    """Brute force check for permutation equivalence of two CSS codes."""
    n = hx1.shape[1]

    hx_rank = hx1.shape[0]
    hz_rank = hz1.shape[0]

    for perm in permutations(range(n)):
        if hx_rank and hx_rank != rank(np.vstack([hx1, hx2[:, perm]])):
            continue
        if hz_rank and hz_rank != rank(np.vstack([hz1, hz2[:, perm]])):
            continue
        return list(perm)

    return None


def _bruteforce_stb(c1: np.ndarray, c2: np.ndarray) -> list[int] | None:
    """Brute force check for permutation equivalence of two stabilizer codes."""
    c1_rank = c1.shape[0]
    n = c1.shape[1] // 2

    for perm in permutations(range(n)):
        perm_symplectic = perm + tuple(q + n for q in perm)

        if c1_rank == rank(np.vstack([c1, c2[:, perm_symplectic]])):
            return list(perm)

    return None


# ----------------------------------------------------------------------------------------------------
#   Invariants
# ----------------------------------------------------------------------------------------------------


def _preserved_n(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check whether the number of qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.n == c2.n


def _preserved_k(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check whether the number of logical qubits is preserved, which is a necessary condition for P-equivalence."""
    return c1.k == c2.k


def _preserved_d(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check whether the distance is preserved, which is a necessary condition for P-equivalence."""
    if isinstance(c1, CSSCode) and isinstance(c2, CSSCode):
        return c1.x_distance == c2.x_distance and c1.z_distance == c2.z_distance
    return c1.distance == c2.distance


def _preserved_number_zero_columns(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check whether the number of zero columns is preserved, which is a necessary condition for P-equivalence."""
    return int(np.count_nonzero(np.all(c1.symplectic == 0, axis=0))) == int(
        np.count_nonzero(np.all(c2.symplectic == 0, axis=0))
    )


def _preserved_number_duplicate_columns(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check whether the number of duplicate columns is preserved, which is a necessary condition for P-equivalence."""

    def _duplicate_column(m: np.ndarray) -> list[int]:
        columns = [tuple(m[:, j].tolist()) for j in range(m.shape[1])]
        counts = Counter(columns)
        return sorted(counts.values())

    return _duplicate_column(c1.symplectic) == _duplicate_column(c2.symplectic)


def _preserved_linear_dependencies(c1: np.ndarray, c2: np.ndarray) -> bool:
    """Check whether the linear dependencies between columns are preserved, which is a necessary condition for P-equivalence."""

    def _linear_dependencies(m: np.ndarray) -> tuple[list[int], list[int], list[int]]:
        n = m.shape[1] // 2

        one_columns = [rank(np.column_stack([m[:, q], m[:, q + n]])) for q in range(n)]

        two_columns = [
            rank(np.column_stack([m[:, i], m[:, i + n], m[:, j], m[:, j + n]]))
            for i in range(n)
            for j in range(i + 1, n)
        ]
        three_columns = [
            rank(np.column_stack([m[:, i], m[:, i + n], m[:, j], m[:, j + n], m[:, k], m[:, k + n]]))
            for i in range(n)
            for j in range(i + 1, n)
            for k in range(j + 1, n)
        ]

        return (sorted(one_columns), sorted(two_columns), sorted(three_columns))

    return _linear_dependencies(c1) == _linear_dependencies(c2)


def _preserved_punctured_hull_weight_enumerator_css_code(
    hx1: np.ndarray, hz1: np.ndarray, hx2: np.ndarray, hz2: np.ndarray
) -> tuple[bool, dict[int, list[int]] | None, dict[int, list[int]] | None]:
    """Compute the partition of the qubits of a CSS code based on the combined Sendrier's invariant of the weight enumerator of the hull of the punctured code."""

    def _generator_matrix_from_parity_check(h: np.ndarray, n: int) -> np.ndarray:
        if h.size == 0 or h.shape[0] == 0:
            return np.eye(n, dtype=np.uint8)
        return nullspace(h)

    def _compute_signatures(g1: np.ndarray, g2: np.ndarray) -> list[int]:
        def _weight_enumerator_of_hull_punctured(g: np.ndarray, col_idx: int) -> list[int]:
            gp = np.delete(g, col_idx, axis=1)
            g_p = gp.shape[1]

            gram = (gp @ gp.T) & 1

            if gram.size == 0:
                hull_basis = np.zeros((0, g_p), dtype=np.uint8)
            elif not gram.any():
                hull_basis = row_basis(gp).astype(np.uint8)
            else:
                coeff_basis = nullspace(gram)

                if coeff_basis.shape[0] == 0:
                    hull_basis = np.zeros((0, g_p), dtype=np.uint8)
                else:
                    hull_basis = row_basis((coeff_basis @ gp) & 1).astype(np.uint8)

            h = hull_basis.shape[0]
            enumerator = [1] + [0] * g_p

            word = np.zeros(g_p, dtype=np.uint8)
            previous_gray = 0

            for t in range(1, 1 << h):
                gray = t ^ (t >> 1)
                changed = gray ^ previous_gray
                row_idx = changed.bit_length() - 1

                word ^= hull_basis[row_idx]
                enumerator[int(word.sum())] += 1

                previous_gray = gray

            return enumerator

        def _combine_invariants(inv_hx: list[int], inv_hz: list[int]) -> int:
            payload = (",".join(map(str, inv_hx)) + "|" + ",".join(map(str, inv_hz))).encode("ascii")
            return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")

        invariants = []

        for col_idx in range(g1.shape[1]):
            inv1 = _weight_enumerator_of_hull_punctured(g1, col_idx)
            inv2 = _weight_enumerator_of_hull_punctured(g2, col_idx)

            invariants.append(_combine_invariants(inv1, inv2))

        return invariants

    def _partition_columns_by_invariants(invariants: list[int]) -> dict[int, list[int]]:
        partition = defaultdict(list)
        for idx, inv in enumerate(invariants):
            partition[inv].append(idx)
        return dict(sorted(partition.items()))

    n = hx1.shape[1]

    gx1 = _generator_matrix_from_parity_check(hx1, n)
    gz1 = _generator_matrix_from_parity_check(hz1, n)
    gx2 = _generator_matrix_from_parity_check(hx2, n)
    gz2 = _generator_matrix_from_parity_check(hz2, n)

    signatures_c1 = _compute_signatures(gx1, gz1)
    signatures_c2 = _compute_signatures(gx2, gz2)

    partition_c1 = _partition_columns_by_invariants(signatures_c1)
    partition_c2 = _partition_columns_by_invariants(signatures_c2)

    if partition_c1.keys() != partition_c2.keys():
        return False, None, None
    if any(len(partition_c1[k]) != len(partition_c2[k]) for k in partition_c1):
        return False, None, None

    return True, partition_c1, partition_c2


def _preserved_punctured_hull_weight_enumerator_stabilizer_code(
    c1: np.ndarray, c2: np.ndarray
) -> tuple[bool, dict[tuple[int, ...], list[int]] | None, dict[tuple[int, ...], list[int]] | None]:
    """Compute the partition of the qubits of a stabilizer code based on Sendrier's invariant of the weight enumerator of the hull of the punctured code."""

    def _symplectic_to_gf4(symplectic: np.ndarray) -> np.ndarray:
        n = symplectic.shape[1] // 2
        return symplectic[:, :n] + 2 * symplectic[:, n:]

    def _compute_signatures(matrix: np.ndarray) -> list[tuple[int, ...]]:
        """Compute the combined Sendriers invariant of the weight enumerator of the hull of the punctured code of each column of the code."""

        def _gf4_column_gram_contributions(m: np.ndarray) -> np.ndarray:
            k, n = m.shape
            contributions = np.zeros((n, k, k), dtype=np.uint8)

            for col in range(n):
                x = m[:, col] & 1
                z = m[:, col] >> 1
                contributions[col, :, :] = (x[:, None] & z[None, :]) ^ (z[:, None] & x[None, :])

            return contributions

        def _gf4_rref(matrix: np.ndarray) -> tuple[int, np.ndarray]:
            matrix = matrix.copy()
            m, n = matrix.shape
            rank = 0
            row = 0

            for bit_col in range(2 * n):
                col = bit_col % n
                bit = bit_col // n

                pivot = None
                for r in range(row, m):
                    if (matrix[r, col] >> bit) & 1:
                        pivot = r
                        break

                if pivot is None:
                    continue

                if pivot != row:
                    matrix[[row, pivot]] = matrix[[pivot, row]]

                for r in range(m):
                    if r != row and ((matrix[r, col] >> bit) & 1):
                        matrix[r, :] ^= matrix[row, :]

                rank += 1
                row += 1
                if row == m:
                    break

            return rank, matrix

        def _gf4_row_basis(m: np.ndarray) -> np.ndarray:
            rank, rref = _gf4_rref(m)
            return rref[:rank, :]

        def _gf2_gf4_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            m, ra = a.shape
            rb, n = b.shape
            if ra != rb:
                msg = "Incompatible shapes for matrix multiplication."
                raise ValueError(msg)

            c = np.zeros((m, n), dtype=np.uint8)
            for i in range(m):
                rows = np.flatnonzero(a[i])
                if rows.size:
                    c[i, :] = np.bitwise_xor.reduce(b[rows, :], axis=0)

            return c

        def _weight_enumerator_of_hull_punctured(col_idx: int) -> list[int]:
            mp = np.delete(matrix, col_idx, axis=1)
            m_p = mp.shape[1]

            if mp.shape[0] == 0:
                return [1] + [0] * m_p

            # gram is in GF(2) due to the trace inner product that simulates the symplectic product (aka commutation/anti-commutation)
            gram = full_gram ^ column_gram_contributions[col_idx]

            # c @ gram = gram.T @ c.T = 0 -> x = c @ mp with <x, mp[i]> = 0 for all rows j -> x orthogonal to all rows of mp -> x in mp perp
            coeff_basis = nullspace(gram.T)
            if coeff_basis.shape[0] == 0:
                hull_basis = np.zeros((0, m_p), dtype=np.uint8)
            else:
                # c @ mp = x -> words in mp that are orthogonal to all rows of mp -> hull
                hull_basis = _gf4_row_basis(_gf2_gf4_matmul(coeff_basis, mp))

            hull_h, hull_n = hull_basis.shape
            enumerator = [1] + [0] * m_p

            word = np.zeros(hull_n, dtype=np.uint8)
            previous_gray = 0

            for t in range(1, 1 << hull_h):
                gray = t ^ (t >> 1)
                changed = gray ^ previous_gray
                row_idx = changed.bit_length() - 1

                # GF(2)-additive
                word ^= hull_basis[row_idx]

                wt = int(np.count_nonzero(word))
                enumerator[wt] += 1

                previous_gray = gray

            return enumerator

        column_gram_contributions = _gf4_column_gram_contributions(matrix)
        full_gram = np.bitwise_xor.reduce(column_gram_contributions, axis=0, initial=0)

        invariants = []

        for col_idx in range(matrix.shape[1]):
            inv = tuple(_weight_enumerator_of_hull_punctured(col_idx))
            invariants.append(inv)

        return invariants

    def _partition_columns_by_invariants(invariants: list[tuple[int, ...]]) -> dict[tuple[int, ...], list[int]]:
        partition = defaultdict(list)
        for idx, inv in enumerate(invariants):
            partition[inv].append(idx)
        return {k: sorted(v) for k, v in sorted(partition.items(), key=operator.itemgetter(0))}

    gf4_tableau_c1 = _symplectic_to_gf4(c1)
    gf4_tableau_c2 = _symplectic_to_gf4(c2)

    signatures_c1 = _compute_signatures(gf4_tableau_c1)
    signatures_c2 = _compute_signatures(gf4_tableau_c2)

    partition_c1 = _partition_columns_by_invariants(signatures_c1)
    partition_c2 = _partition_columns_by_invariants(signatures_c2)

    if partition_c1.keys() != partition_c2.keys():
        return False, None, None
    if any(len(partition_c1[k]) != len(partition_c2[k]) for k in partition_c1):
        return False, None, None

    for key1, key2 in zip(partition_c1.keys(), partition_c2.keys(), strict=False):
        if key1 != key2:
            return False, None, None
        if len(partition_c1[key1]) != len(partition_c2[key2]):
            return False, None, None

    return True, partition_c1, partition_c2


# ----------------------------------------------------------------------------------------------------
#   Decision procedures
# ----------------------------------------------------------------------------------------------------


def _sat_stabilizer_code(
    c1: np.ndarray,
    partition1: dict[tuple[int, ...], list[int]],
    c2: np.ndarray,
    partition2: dict[tuple[int, ...], list[int]],
) -> list[int] | None:
    """Map the permutation equivalence problem of two stabilizer codes to a SAT problem and solve it using Z3."""
    solver = z3.Solver()

    r, n = c1.shape[0], c1.shape[1] // 2

    # permutations
    aux_tableau = [z3.Bool(f"aux_{row}_{col}") for row in range(r) for col in range(2 * n)]

    permutation_variables = {
        (i, j): z3.Bool(f"p_{i}_{j}") for sig, col1 in partition1.items() for i in col1 for j in partition2[sig]
    }

    for i in range(n):
        solver.add(_exactly_one([var for (src, _), var in permutation_variables.items() if src == i]))
    for j in range(n):
        solver.add(_exactly_one([var for (_, tgt), var in permutation_variables.items() if tgt == j]))

    for (i, j), permutation_variable in permutation_variables.items():
        x_column_original = c1[:, i]
        z_column_original = c1[:, i + n]

        x_column_permuted = [aux_tableau[row * (2 * n) + j] for row in range(r)]
        z_column_permuted = [aux_tableau[row * (2 * n) + j + n] for row in range(r)]

        solver.add(
            z3.Implies(
                permutation_variable,
                z3.And(
                    _elementwise_map(x_column_original, x_column_permuted),
                    _elementwise_map(z_column_original, z_column_permuted),
                ),
            )
        )

    # row operations
    row_operation_coefficients = [z3.Bool(f"r_{i}_{j}") for i in range(r) for j in range(r)]

    for row in range(r):
        for q in range(2 * n):
            row_contributions = [
                row_operation_coefficients[row * r + contribution]
                for contribution in range(r)
                if c2[contribution, q] == 1
            ]

            solver.add(aux_tableau[row * (2 * n) + q] == _xor_list(row_contributions))

    if solver.check() != z3.sat:
        return None

    perm = [-1] * n
    model = solver.model()
    for i in range(n):
        perm[i] = next(
            j
            for (src, j), var in permutation_variables.items()
            if src == i and z3.is_true(model.eval(var, model_completion=True))
        )
    return perm


def _sat_css_code(
    hx1: np.ndarray,
    hz1: np.ndarray,
    partition1: dict[int, list[int]],
    hx2: np.ndarray,
    hz2: np.ndarray,
    partition2: dict[int, list[int]],
) -> list[int] | None:
    """Map the permutation equivalence problem of two CSS codes to a SAT problem and solve it using Z3."""
    solver = z3.Solver()

    n = hx1.shape[1]
    rx = hx1.shape[0]
    rz = hz1.shape[0]

    # permutations
    aux_tableau_x = [z3.Bool(f"aux_x_{row}_{col}") for row in range(rx) for col in range(n)]
    aux_tableau_z = [z3.Bool(f"aux_z_{row}_{col}") for row in range(rz) for col in range(n)]

    permutation_variables = {
        (i, j): z3.Bool(f"p_{i}_{j}") for sig, col1 in partition1.items() for i in col1 for j in partition2[sig]
    }

    for i in range(n):
        solver.add(_exactly_one([var for (src, _), var in permutation_variables.items() if src == i]))
    for j in range(n):
        solver.add(_exactly_one([var for (_, tgt), var in permutation_variables.items() if tgt == j]))

    for (i, j), permutation_variable in permutation_variables.items():
        x_column_original = hx1[:, i]
        z_column_original = hz1[:, i]

        x_column_permuted = [aux_tableau_x[row * n + j] for row in range(rx)]
        z_column_permuted = [aux_tableau_z[row * n + j] for row in range(rz)]

        solver.add(
            z3.Implies(
                permutation_variable,
                z3.And(
                    _elementwise_map(x_column_original, x_column_permuted),
                    _elementwise_map(z_column_original, z_column_permuted),
                ),
            )
        )

    # row operations
    row_operation_coefficients_x = [z3.Bool(f"r_x_{i}_{j}") for i in range(rx) for j in range(rx)]
    row_operation_coefficients_z = [z3.Bool(f"r_z_{i}_{j}") for i in range(rz) for j in range(rz)]

    for row in range(rx):
        for q in range(n):
            row_contributions = [
                row_operation_coefficients_x[row * rx + contribution]
                for contribution in range(rx)
                if hx2[contribution, q] == 1
            ]

            solver.add(aux_tableau_x[row * n + q] == _xor_list(row_contributions))

    for row in range(rz):
        for q in range(n):
            row_contributions = []
            for contribution in range(rz):
                if hz2[contribution, q] == 1:
                    row_contributions.append(row_operation_coefficients_z[row * rz + contribution])

            solver.add(aux_tableau_z[row * n + q] == _xor_list(row_contributions))

    if solver.check() != z3.sat:
        return None

    perm = [-1] * n
    model = solver.model()
    for i in range(n):
        perm[i] = next(
            j
            for (src, j), var in permutation_variables.items()
            if src == i and z3.is_true(model.eval(var, model_completion=True))
        )
    return perm


def _matroid_css_code(
    hx1: np.ndarray,
    hz1: np.ndarray,
    partition1: dict[int, list[int]],
    hx2: np.ndarray,
    hz2: np.ndarray,
    partition2: dict[int, list[int]],
) -> list[int] | None:
    """Map the permutation equivalence problem of two CSS codes to a matroid and graph isomorphism problem and solve it using the nauty algorithm."""

    def _circuits_binary_matroid(a: npt.NDArray[np.int8]) -> list[int]:
        def _row_support_as_mask(row: npt.NDArray[np.uint8]) -> int:
            support = 0
            for col in np.flatnonzero(row):
                support |= 1 << int(col)
            return support

        k = nullspace(a)
        k_m, _ = k.shape
        row_supports = [_row_support_as_mask(row) for row in k]
        circuits_by_size: list[list[int]] = [[] for _ in range(a.shape[1] + 1)]

        support = 0
        previous_gray = 0
        for mask in range(1, 1 << k_m):
            gray = mask ^ (mask >> 1)
            changed = gray ^ previous_gray
            support ^= row_supports[changed.bit_length() - 1]
            previous_gray = gray

            if not support:
                continue

            support_size = support.bit_count()

            if any(
                (circuit & support) == circuit
                for size in range(1, support_size + 1)
                for circuit in circuits_by_size[size]
            ):
                continue

            for size in range(support_size + 1, len(circuits_by_size)):
                if not circuits_by_size[size]:
                    continue
                circuits_by_size[size] = [
                    circuit for circuit in circuits_by_size[size] if (support & circuit) != support
                ]

            circuits_by_size[support_size].append(support)

        return [circuit for circuits in circuits_by_size for circuit in sorted(circuits)]

    def _graph_from_circuits_and_invariants(
        n: int, circuits_hx: list[int], circuits_hz: list[int], partition: dict[int, list[int]]
    ) -> nx.Graph:
        def _iter_mask_bits(mask: int) -> Iterator[int]:
            while mask:
                bit = mask & -mask
                yield bit.bit_length() - 1
                mask ^= bit

        n_hx = len(circuits_hx)
        n_hz = len(circuits_hz)

        hx_offset = n
        hz_offset = n + n_hx

        graph = nx.Graph()
        graph.add_nodes_from(range(n + n_hx + n_hz))

        qubit_color: dict[int, int] = {}
        for color, (_, columns) in enumerate(sorted(partition.items())):
            for column in columns:
                qubit_color[column] = color

        for q in range(n):
            graph.nodes[q]["color"] = ("qubit", qubit_color[q])

        def _add_edges_from_circuits(circuits: list[int], offset: int, kind: str) -> None:
            for i, circuit in enumerate(circuits):
                circuit_vertex = offset + i
                graph.nodes[circuit_vertex]["color"] = (kind,)
                for q in _iter_mask_bits(circuit):
                    graph.add_edge(q, circuit_vertex)

        _add_edges_from_circuits(circuits_hx, hx_offset, "hx")
        _add_edges_from_circuits(circuits_hz, hz_offset, "hz")

        return graph

    n = hx1.shape[1]

    circuits_c1_hx = _circuits_binary_matroid(hx1)
    circuits_c1_hz = _circuits_binary_matroid(hz1)

    graph_c1 = _graph_from_circuits_and_invariants(n, circuits_c1_hx, circuits_c1_hz, partition1)

    len_circuits_c1_hx = len(circuits_c1_hx)
    len_circuits_c1_hz = len(circuits_c1_hz)

    del circuits_c1_hx
    del circuits_c1_hz

    circuits_c2_hx = _circuits_binary_matroid(hx2)
    if len_circuits_c1_hx != len(circuits_c2_hx):
        return None

    circuits_c2_hz = _circuits_binary_matroid(hz2)
    if len_circuits_c1_hz != len(circuits_c2_hz):
        return None

    graph_c2 = _graph_from_circuits_and_invariants(n, circuits_c2_hx, circuits_c2_hz, partition2)

    del circuits_c2_hx
    del circuits_c2_hz

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph_c1, graph_c2, node_match=lambda a, b: a["color"] == b["color"]
    )

    if not matcher.is_isomorphic():
        return None

    mapping = matcher.mapping
    return [mapping[q] for q in range(n)]


# ----------------------------------------------------------------------------------------------------
#   Helper functions
# ----------------------------------------------------------------------------------------------------


def _elementwise_map(normal_bool: npt.NDArray[np.uint8], variables: Sequence[z3.BoolRef]) -> z3.BoolRef:
    return z3.And([v if bit == 1 else z3.Not(v) for bit, v in zip(normal_bool, variables, strict=False)])


def _exactly_one(variables: Sequence[z3.BoolRef]) -> z3.BoolRef:
    return z3.PbEq([(v, 1) for v in variables], 1)


def _xor_list(variables: Sequence[z3.BoolRef]) -> z3.BoolRef:
    acc = z3.BoolVal(False)
    for v in variables:
        acc = z3.Xor(acc, v)
    return acc
