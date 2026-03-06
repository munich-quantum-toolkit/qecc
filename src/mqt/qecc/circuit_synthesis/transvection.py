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

import ldpc.mod2.mod2_numpy as mod2
import numpy as np

from ..codes.pauli import StabilizerTableau
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
        if not isinstance(tableau, StabilizerTableau):
            return []

        unscored_candidates = _generate_transvection_operations(tableau)
        filtered_candidates = self._apply_filters(unscored_candidates)
        scored = _score_transvections(filtered_candidates, tableau)
        if not scored:
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


def _generate_transvection_operations(tableau: StabilizerTableau) -> list[Transvection]:
    """Generate all transvection operations without scoring."""
    n = get_n(tableau)
    symplectic = tableau.tableau.matrix
    pairs = _sp_gate_options(symplectic)

    if not pairs:
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    transvections = Transvection.all_two_qubit_transvections()
    operations: list[Transvection] = []
    for i, j in pairs:
        operations.extend(Transvection(v, i, j) for v in transvections)

    return operations


def _score_transvections(
    operations: Sequence[Transvection], tableau: StabilizerTableau
) -> list[tuple[TableauOperation, int | tuple[int, ...]]]:
    """Score transvection operations and return sorted list."""
    base_score, _ = score_symplectic(tableau)
    scored: list[tuple[TableauOperation, int | tuple[int, ...]]] = []

    for op in operations:
        tableau_op_applied = op.apply(tableau)
        if not isinstance(tableau_op_applied, StabilizerTableau):
            continue
        h_vec, _ = score_symplectic(tableau_op_applied)
        if h_vec < base_score:
            # Convert tuple to int for compatibility with the return type
            score_value = sum(h_vec)
            scored.append((op, score_value))

    scored.sort(key=operator.itemgetter(1))
    return scored


def get_candidate_transvections(
    tableau: StabilizerTableau,
) -> list[tuple[Transvection, tuple[int, ...]]]:
    """Score all possible operations and return scored operations."""
    n = get_n(tableau)
    symplectic = tableau.tableau.matrix

    pairs = _sp_gate_options(symplectic)

    if not pairs:
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    transvections = Transvection.all_two_qubit_transvections()
    scores: list[tuple[Transvection, tuple[int, ...]]] = []
    base_score, _ = score_symplectic(tableau)
    for i, j in pairs:
        for v in transvections:
            op = Transvection(v, i, j)
            tableau_op_applied = op.apply(tableau)
            if not isinstance(tableau_op_applied, StabilizerTableau):
                continue
            h_vec, _ = score_symplectic(tableau_op_applied)
            if h_vec < base_score:
                scores.append((op, h_vec))

    scores.sort(key=operator.itemgetter(1))
    return scores


def _compute_r2_matrix(symplectic: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    n = symplectic.shape[0] // 2
    a_xx = symplectic[:n, :n]
    a_xz = symplectic[:n, n:]
    a_zx = symplectic[n:, :n]
    a_zz = symplectic[n:, n:]
    det = (a_xx & a_zz) ^ (a_xz & a_zx)
    return det.astype(np.int8)


def _compute_r0_matrix(symplectic: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    n = symplectic.shape[0] // 2
    a_xx = symplectic[:n, :n]
    a_xz = symplectic[:n, n:]
    a_zx = symplectic[n:, :n]
    a_zz = symplectic[n:, n:]
    zero = (a_xx == 0) & (a_xz == 0) & (a_zx == 0) & (a_zz == 0)
    result: npt.NDArray[np.int8] = zero.astype(np.int8)
    return result


def _compute_r1_matrix_from_r2_r0(r2: npt.NDArray[np.int8], r0: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    return (1 ^ (r2 | r0)).astype(np.int8)


def r1_r2(symplectic: npt.NDArray[np.int8]) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """Compute R1 and R2 matrices from a symplectic matrix."""
    n = symplectic.shape[0] // 2

    a_xx = symplectic[:n, :n]
    a_xz = symplectic[:n, n:]
    a_zx = symplectic[n:, :n]
    a_zz = symplectic[n:, n:]

    r2 = (a_xx & a_zz) ^ (a_xz & a_zx)
    r0 = ~(a_xx | a_xz | a_zx | a_zz)
    r1 = ~(r2 | r0)

    return r1.astype(np.int8), r2.astype(np.int8)


def is_terminal_transvection(tableau: StabilizerTableau) -> bool:
    """Check if the given stabilizer tableau is in terminal form for transvection elimination."""
    r1, r2 = r1_r2(tableau.tableau.matrix)
    if np.any(r1):
        return False
    if not np.all(r2.sum(axis=0) == 1):
        return False
    return bool(np.all(r2.sum(axis=1) == 1))


def score_symplectic(tableau: StabilizerTableau) -> tuple[tuple[int, ...], int]:
    """Score the given symplectic matrix using the default symplectic heuristic."""
    n = get_n(tableau)

    symplectic = tableau.tableau.matrix
    r1, r2 = r1_r2(symplectic)

    c1 = r1.sum(axis=0).astype(int)
    c2 = r2.sum(axis=0).astype(int)

    c1t = r1.sum(axis=1).astype(int)
    c2t = r2.sum(axis=1).astype(int)
    vec = np.concatenate([n * c2 + c1, n * c2t + c1t])

    h_vec = tuple(sorted(int(x) for x in vec))

    h_scalar = int(r1.sum() + r2.sum())
    return h_vec, h_scalar


def reduce_with_swaps(
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
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
        if isinstance(result, StabilizerTableau):
            tableau_copy = result
        swap_sequence.add_operation(swap)
    return swap_sequence, tableau_copy


def reduce_with_single_qubit_cliffords(
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce diagonal blocks to identity using single-qubit Cliffords and Paulis."""
    tableau_copy = tableau.copy()
    n = get_n(tableau)

    clifford_sequence = EliminationSequence([])

    for q in range(n):
        f = tableau_copy.symplectic_submatrix(q)
        clifford_op = SingleQubitClifford.from_symplectic_block(f, q)
        clifford_sequence.add_operation(clifford_op)
        result = clifford_op.apply(tableau_copy, inplace=True)
        if isinstance(result, StabilizerTableau):
            tableau_copy = result

    pauli_ops = fix_tableau_signs_in_place(tableau_copy)
    for op in pauli_ops:
        clifford_sequence.add_operation(op)
    return clifford_sequence, tableau_copy


def reduce_single_qubit_gates_and_swaps(
    operations: EliminationSequence,
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce a TERMINAL symplectic matrix to identity using SWAP/H/S/Pauli gates."""
    swap_seq, tableau_after_swaps = reduce_with_swaps(tableau)

    clifford_seq, final_tableau = reduce_with_single_qubit_cliffords(tableau_after_swaps)

    operations.extend(EliminationSequence(list(swap_seq.operations) + list(clifford_seq.operations)))

    return operations, final_tableau


def reduce_without_swaps(
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce a TERMINAL symplectic matrix to a permuted identity using only single-qubit gates."""
    return reduce_with_single_qubit_cliffords(tableau)


def _extract_perm_in_to_out_and_blocks(tableau: StabilizerTableau) -> tuple[np.ndarray, list[npt.NDArray[np.int8]]]:
    """Extract the permutation and corresponding 2x2 blocks from a terminal symplectic matrix."""
    n = get_n(tableau)
    symplectic = tableau.tableau.matrix
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


def fix_tableau_signs_in_place(tableau: StabilizerTableau) -> list[PauliOperation]:
    """Determine Pauli corrections so that the tableau matches the desired sign bits."""
    n = get_n(tableau)
    x_part = tableau.tableau.matrix[:, :n]
    z_part = tableau.tableau.matrix[:, n:]

    phase = tableau.phase.copy()

    if not np.any(phase):
        return []

    tableau_with_phase = np.hstack((x_part, z_part, np.array([phase]).T))
    ker = mod2.nullspace(tableau_with_phase)
    assert ker[-1, -1] == 1, "Last entry of kernel vector must be 1."
    correction_symplectic = ker[-1]
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
