# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Transvection-based candidate generation for non-CSS stabilizer codes."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from mqt.qecc import mod2

from ..codes.core.pauli import PauliTableau
from .elimination import CandidateGenerator, EliminationSequence, get_n
from .operations import PauliOperation, SingleQubitClifford, Swap, Transvection

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from .elimination import OperationFilter
    from .operations import TableauOperation
    from .types import BinaryMatrix


class GreedyTransvectionGenerator(CandidateGenerator):
    """Generates transvection candidates using greedy heuristic."""

    def __init__(self, filters: Sequence[OperationFilter] | None = None) -> None:
        """Initialize the greedy transvection generator."""
        self.operation_history: list[TableauOperation] = []
        self.filters = list(filters) if filters else []

    def get_candidates(self, tableau: BinaryMatrix) -> Sequence[tuple[TableauOperation, int | tuple[int, ...]]]:
        """Generate transvection candidates sorted by heuristic score."""
        if not isinstance(tableau, PauliTableau):
            return []

        unscored_candidates = _generate_transvection_operations(tableau)
        filtered_candidates = self._apply_filters(unscored_candidates)
        scored = _score_transvections(filtered_candidates, tableau)
        if scored:
            return scored

        self._reset_filters()
        filtered_candidates = self._apply_filters(unscored_candidates)
        return _score_transvections(filtered_candidates, tableau)

    def _apply_filters(self, candidates: Sequence[Transvection]) -> list[Transvection]:
        """Apply all filters to candidate list."""
        if not self.filters:
            return list(candidates)

        filtered = [op for op in candidates if all(f.should_include(op) for f in self.filters)]

        if not filtered:
            self._reset_filters()
            return list(candidates)

        return filtered

    def _reset_filters(self) -> None:
        """Reset all filters."""
        for f in self.filters:
            f.reset()

    def update(self, op: TableauOperation, tableau: BinaryMatrix) -> None:  # noqa: ARG002
        """Update operation history and filters after applying an operation."""
        self.operation_history.append(op)
        for f in self.filters:
            f.update(op)

    def reset(self) -> None:
        """Reset the operation history."""
        self.operation_history.clear()


def _bin2set(row: npt.NDArray[np.int8]) -> list[int]:
    """Convert a binary row to a list of column indices where the value is 1."""
    return [int(i) for i in np.flatnonzero(row)]


def _sp_gate_options(symplectic: npt.NDArray[np.int8]) -> list[tuple[int, int]]:
    """Return a reduced set of candidate pairs (i,j) to consider, based on R2/R1 structure."""
    r1, r2 = r1_r2(symplectic)
    n = symplectic.shape[0] // 2
    pairs: set[tuple[int, int]] = set()

    for row in range(n):
        r2_cols = _bin2set(r2[row])
        r1_cols = _bin2set(r1[row])

        for a in range(len(r2_cols) - 1):
            for b in range(a + 1, len(r2_cols)):
                i, j = int(r2_cols[a]), int(r2_cols[b])
                if i != j:
                    pairs.add((min(i, j), max(i, j)))

        for i0 in r2_cols:
            for j0 in r1_cols:
                i, j = int(i0), int(j0)
                if i != j:
                    pairs.add((min(i, j), max(i, j)))

    return sorted(pairs)


def _generate_transvection_operations(tableau: PauliTableau) -> list[Transvection]:
    """Generate all transvection operations without scoring."""
    n = get_n(tableau)
    symplectic = tableau.tableau.data
    pairs = _sp_gate_options(symplectic)

    if not pairs:
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    transvections = Transvection.all_two_qubit_transvections()
    operations: list[Transvection] = []
    for i, j in pairs:
        operations.extend(Transvection(v, i, j) for v in transvections)

    return operations


def _score_transvections(
    operations: Sequence[Transvection], tableau: PauliTableau
) -> list[tuple[TableauOperation, int | tuple[int, ...]]]:
    """Score transvection operations and return sorted list."""
    base_score, _ = score_symplectic(tableau)
    scored: list[tuple[TableauOperation, int | tuple[int, ...]]] = []

    original_state = tableau.tableau.data.copy()
    original_phase = tableau.phase.copy()
    for op in operations:
        op.apply_stabilizer_tableau_inplace(tableau)
        h_vec, _ = score_symplectic(tableau)

        if lexicographical_compare_np(h_vec, base_score):
            score_value = tuple(int(v) for v in h_vec.tolist())
            scored.append((op, score_value))

        tableau.tableau.data[:] = original_state
        tableau.phase[:] = original_phase
    scored.sort(key=operator.itemgetter(1))
    return scored


@nb.jit(nopython=True, cache=True)  # type: ignore[untyped-decorator]
def lexicographical_compare_np(arr1: np.ndarray, arr2: np.ndarray) -> bool:
    """Perform lexicographical comparison of two NumPy arrays using Numba."""
    for i in range(len(arr1)):
        if arr1[i] < arr2[i]:
            return True
        if arr1[i] > arr2[i]:
            return False
    return False


def _compute_r2_matrix(symplectic: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    n = symplectic.shape[0] // 2
    a_xx = symplectic[:n, :n]
    a_xz = symplectic[:n, n:]
    a_zx = symplectic[n:, :n]
    a_zz = symplectic[n:, n:]
    det = (a_xx & a_zz) ^ (a_xz & a_zx)
    return det.astype(np.int8)


@nb.jit(nopython=True, cache=True)  # type: ignore[untyped-decorator]
def r1_r2(symplectic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute R1 and R2 matrices from a symplectic matrix using Numba."""
    n = symplectic.shape[0] // 2

    # Extract quadrants of the symplectic matrix
    a_xx = symplectic[:n, :n]
    a_xz = symplectic[:n, n:]
    a_zx = symplectic[n:, :n]
    a_zz = symplectic[n:, n:]

    # Precompute intermediate results
    and_xx_zz = np.bitwise_and(a_xx, a_zz)
    and_xz_zx = np.bitwise_and(a_xz, a_zx)

    # Compute R2
    r2 = np.bitwise_xor(and_xx_zz, and_xz_zx)

    # Compute R0 (zero matrix) using explicit element-wise OR
    combined = np.zeros_like(a_xx, dtype=np.int8)
    for i in range(combined.shape[0]):
        for j in range(combined.shape[1]):
            combined[i, j] = a_xx[i, j] | a_xz[i, j] | a_zx[i, j] | a_zz[i, j]
    r0 = np.logical_not(combined).astype(np.int8)

    # Compute R1
    r1 = np.logical_not(np.bitwise_or(r2, r0)).astype(np.int8)

    return r1, r2


def is_terminal_transvection(tableau: PauliTableau) -> bool:
    """Check if the given stabilizer tableau is in terminal form for transvection elimination."""
    r1, r2 = r1_r2(tableau.tableau.data)
    if np.any(r1):
        return False
    if not np.all(r2.sum(axis=0) == 1):
        return False
    return bool(np.all(r2.sum(axis=1) == 1))


@nb.jit(nopython=True, cache=True)  # type: ignore[untyped-decorator]
def score_symplectic_numba(symplectic: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Score the given symplectic matrix using the default symplectic heuristic with Numba."""
    # Compute R1 and R2 matrices using the Numba-optimized r1_r2_numba
    r1, r2 = r1_r2(symplectic)

    # Precompute sums for columns and rows
    r1_col_sum = r1.sum(axis=0)
    r2_col_sum = r2.sum(axis=0)
    r1_row_sum = r1.sum(axis=1)
    r2_row_sum = r2.sum(axis=1)

    # Combine column and row sums into a single vector
    vec = np.empty(2 * n, dtype=np.int32)
    for i in range(n):
        vec[i] = n * r2_col_sum[i] + r1_col_sum[i]
        vec[n + i] = n * r2_row_sum[i] + r1_row_sum[i]

    # Sort the vector for the heuristic score
    h_vec = np.sort(vec)

    # Compute the scalar score
    h_scalar = int(r1_col_sum.sum() + r2_col_sum.sum())

    return h_vec, h_scalar


def score_symplectic(tableau: PauliTableau) -> tuple[np.ndarray, int]:
    """Numba-optimized score_symplectic function."""
    n = get_n(tableau)
    symplectic = tableau.tableau.data
    h_vec, h_scalar = score_symplectic_numba(symplectic, n)
    return h_vec, h_scalar


def reduce_with_swaps(
    tableau: PauliTableau,
) -> tuple[EliminationSequence, PauliTableau]:
    """Reduce a TERMINAL symplectic matrix by applying SWAPs to align blocks on diagonal."""
    tableau_copy = tableau.copy()
    get_n(tableau)
    perm, _blocks = _extract_perm_in_to_out_and_blocks(tableau_copy)

    _perm_inverse(perm)
    swaps = _perm_to_swaps(perm)
    p = list(range(tableau.n))
    for swap in reversed(swaps):
        a, b = (swap.qubit_a, swap.qubit_b)
        p[a], p[b] = p[b], p[a]

    swap_sequence = EliminationSequence([])

    for swap in swaps:
        result = swap.apply(tableau_copy, inplace=True)
        if isinstance(result, PauliTableau):
            tableau_copy = result
        swap_sequence.add_operation(swap)
    return swap_sequence, tableau_copy


def reduce_with_single_qubit_cliffords(
    tableau: PauliTableau,
) -> tuple[EliminationSequence, PauliTableau]:
    """Reduce diagonal blocks to identity using single-qubit Cliffords and Paulis."""
    tableau_copy = tableau.copy()
    n = get_n(tableau)

    clifford_sequence = EliminationSequence([])

    for q in range(n):
        f = tableau_copy.symplectic_submatrix(q)
        clifford_op = SingleQubitClifford.from_symplectic_block(f, q)
        clifford_sequence.add_operation(clifford_op)
        result = clifford_op.apply(tableau_copy, inplace=True)
        if isinstance(result, PauliTableau):
            tableau_copy = result

    pauli_ops = fix_tableau_signs_in_place(tableau_copy)
    for op in pauli_ops:
        clifford_sequence.add_operation(op)
    return clifford_sequence, tableau_copy


def reduce_single_qubit_gates_and_swaps(
    operations: EliminationSequence,
    tableau: PauliTableau,
) -> tuple[EliminationSequence, PauliTableau]:
    """Reduce a TERMINAL symplectic matrix to identity using SWAP/H/S/Pauli gates."""
    swap_seq, tableau_after_swaps = reduce_with_swaps(tableau)

    clifford_seq, final_tableau = reduce_with_single_qubit_cliffords(tableau_after_swaps)

    operations.extend(
        EliminationSequence(list(swap_seq.operations) + list(clifford_seq.operations)), ignore_depth_impact=True
    )

    return operations, final_tableau


def reduce_without_swaps(
    tableau: PauliTableau,
) -> tuple[EliminationSequence, PauliTableau]:
    """Reduce a TERMINAL symplectic matrix to a permuted identity using only single-qubit gates."""
    return reduce_with_single_qubit_cliffords(tableau)


def _extract_perm_in_to_out_and_blocks(tableau: PauliTableau) -> tuple[np.ndarray, list[npt.NDArray[np.int8]]]:
    """Extract the permutation and corresponding 2x2 blocks from a terminal symplectic matrix."""
    n = get_n(tableau)
    symplectic = tableau.tableau.data
    r2 = _compute_r2_matrix(symplectic)

    perm = np.full(n, -1, dtype=int)
    blocks: list[npt.NDArray[np.int8]] = []

    for i in range(n):
        js = np.flatnonzero(r2[i])
        if len(js) != 1:
            msg = "Not terminal: R2 row is not one-hot."
            raise ValueError(msg)
        j = int(js[0])
        perm[i] = j
        block = np.array(
            [
                [int(symplectic[i, j]), int(symplectic[i, j + n])],
                [int(symplectic[i + n, j]), int(symplectic[i + n, j + n])],
            ],
            dtype=np.int8,
        )
        blocks.append(block)

    if len(set(perm.tolist())) != n:
        msg = "Not terminal: R2 columns not one-hot."
        raise ValueError(msg)
    return perm, blocks


def _perm_inverse(perm_in_to_out: np.ndarray) -> np.ndarray:
    n = len(perm_in_to_out)
    inv = np.empty(n, dtype=int)
    for i, j in enumerate(perm_in_to_out):
        inv[int(j)] = i
    return inv


def _perm_to_swaps(perm_in_to_out: np.ndarray) -> list[Swap]:
    """Convert a permutation of qubits to a sequence of SWAP operations."""
    n = len(perm_in_to_out)
    swaps: list[Swap] = []
    current = list(range(n))

    for target_idx in range(n):
        desired_wire = perm_in_to_out[target_idx]
        current_idx = current.index(desired_wire)

        if current_idx != target_idx:
            swaps.append(Swap(current_idx, target_idx))
            current[current_idx], current[target_idx] = current[target_idx], current[current_idx]

    return swaps


def fix_tableau_signs_in_place(tableau: PauliTableau) -> list[PauliOperation]:
    """Determine Pauli corrections so that the tableau matches the desired sign bits."""
    n = get_n(tableau)
    x_part = tableau.tableau.data[:, :n]
    z_part = tableau.tableau.data[:, n:]

    phase = tableau.phase.copy()

    if not np.any(phase):
        return []

    tableau_with_phase = np.hstack((x_part, z_part, np.array([phase]).T))
    ker = mod2.nullspace(tableau_with_phase)
    # A valid correction is any kernel vector with a 1 in the phase (final) column.
    # Locate it explicitly rather than assuming a particular nullspace row ordering.
    phase_rows = np.flatnonzero(ker[:, -1] == 1)
    assert phase_rows.size > 0, "Kernel must contain a vector with a 1 in the phase column."
    correction_symplectic = ker[phase_rows[-1]]
    xc = correction_symplectic[:n]
    zc = correction_symplectic[n:-1]
    ops: list[PauliOperation] = []
    for i, (xv, zv) in enumerate(zip(xc, zc, strict=False)):
        if xv == 1 and zv == 1:
            op = PauliOperation(i, "Y")
        elif xv == 1:
            op = PauliOperation(i, "Z")
        elif zv == 1:
            op = PauliOperation(i, "X")
        else:
            continue
        ops.append(op)
        op.apply(tableau, inplace=True)

    return ops
