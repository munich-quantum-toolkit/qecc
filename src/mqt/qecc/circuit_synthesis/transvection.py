# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Transvection-based elimination for non-CSS stabilizer codes."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2
import numpy as np

from .elimination import (
    CandidateGenerator,
    EliminationConfig,
    EliminationSequence,
    eliminate,
    get_n,
)
from .operations import PauliOperation, SingleQubitClifford, Swap, Transvection

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..codes.pauli import StabilizerTableau
    from .elimination import (
        BinaryMatrix,
        OperationFilter,
        TableauOperation,
    )


class GreedyTransvectionGenerator(CandidateGenerator):
    """Generates transvection candidates using greedy heuristic."""

    def __init__(self, filters: list[OperationFilter] | None = None) -> None:
        """Initialize the greedy transvection generator.

        Args:
            filters: Optional list of filters to apply during candidate generation.
        """
        self.operation_history: list[TableauOperation] = []
        self.filters = filters or []

    def get_candidates(self, tableau: BinaryMatrix) -> list[tuple[TableauOperation, int]]:
        """Generate transvection candidates sorted by heuristic score.

        Args:
            tableau: The current stabilizer tableau.

        Returns:
            List of transvection operations sorted by preference.
        """
        all_candidates = get_candidate_transvections(tableau)
        return self._apply_filters(all_candidates)

    def _apply_filters(self, candidates: list[tuple[TableauOperation, int]]) -> list[tuple[TableauOperation, int]]:
        """Apply all filters to candidate list.

        Args:
            candidates: List of candidate operations with scores.

        Returns:
            Filtered list of candidates.
        """
        if not self.filters:
            return candidates

        filtered = []
        for op, score in candidates:
            if score > 0 and all(f.should_include(op) for f in self.filters):
                filtered.append((op, score))

        if not filtered:
            for f in self.filters:
                if hasattr(f, "_reset"):
                    f._reset()
            return candidates

        return filtered

    def update(self, op: TableauOperation, tableau: BinaryMatrix) -> None:  # noqa: ARG002
        """Update operation history and filters after applying an operation.

        Args:
            op: The operation that was applied.
            tableau: The resulting tableau after applying the operation.
        """
        self.operation_history.append(op)
        for f in self.filters:
            f.update(op)

    def reset(self) -> None:
        """Reset the operation history."""
        self.operation_history.clear()


class GreedyTransvectionGeneratorStateprep(CandidateGenerator):
    """Generates transvection candidates using greedy heuristic for state preparation."""

    def __init__(self, filters: list[OperationFilter] | None = None) -> None:
        """Initialize the greedy transvection generator.

        Args:
            filters: Optional list of filters to apply during candidate generation.
        """
        self.operation_history: list[TableauOperation] = []
        self.filters = filters or []

    def get_candidates(self, tableau: BinaryMatrix) -> list[tuple[TableauOperation, int]]:
        """Generate transvection candidates sorted by heuristic score.

        Args:
            tableau: The current stabilizer tableau.

        Returns:
            List of transvection operations sorted by preference.
        """
        all_candidates = get_candidate_transvections_stateprep(tableau)
        return self._apply_filters(all_candidates)

    def _apply_filters(self, candidates: list[tuple[TableauOperation, int]]) -> list[tuple[TableauOperation, int]]:
        """Apply all filters to candidate list.

        Args:
            candidates: List of candidate operations with scores.

        Returns:
            Filtered list of candidates.
        """
        if not self.filters:
            return candidates

        filtered = []
        for op, score in candidates:
            if score > 0 and all(f.should_include(op) for f in self.filters):
                filtered.append((op, score))

        if not filtered:
            for f in self.filters:
                if hasattr(f, "_reset"):
                    f._reset()
            return candidates

        return filtered

    def update(self, op: TableauOperation, tableau: BinaryMatrix) -> None:  # noqa: ARG002
        """Update operation history and filters after applying an operation.

        Args:
            op: The operation that was applied.
            tableau: The resulting tableau after applying the operation.
        """
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
    """Return a reduced set of candidate pairs (i,j) to consider, based on R2/R1 structure.

    Args:
        symplectic: The symplectic matrix (2n x 2n).

    Returns:
        A sorted list of (i, j) pairs where i < j, representing candidate qubit pairs.
    """
    R1, R2 = r1_r2(symplectic)
    n = symplectic.shape[0] // 2
    pairs: set[tuple[int, int]] = set()

    for row in range(n):
        r2_cols = _bin2set(R2[row])
        r1_cols = _bin2set(R1[row])

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


def get_candidate_transvections_stateprep(
    tableau: StabilizerTableau,
) -> list[Transvection]:
    """Score all possible operations and return the top k scored operations for state preparation.

    Args:
        tableau: The current symplectic matrix.

    Returns:
        A list of scored operations, each represented as a tuple of (operation, score).
    """
    n = get_n(tableau)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    transvections = Transvection.all_two_qubit_transvections()
    scores: list[tuple(Transvection, list[int, ...])] = []
    for i, j in pairs:
        for v in transvections:
            op = Transvection(v, i, j)
            tablea_op_applied = op.apply(tableau)
            s = score_stateprep(tablea_op_applied)
            if s == 0:
                pass

            scores.append((op, s))

    scores.sort(key=operator.itemgetter(1))
    return [(tv, score) for tv, score in scores]


def get_candidate_transvections(
    tableau: StabilizerTableau,
) -> list[Transvection]:
    """Score all possible operations and return scored operations.

    Args:
        tableau: The current symplectic matrix.

    Returns:
        A list of scored operations, each represented as a tuple of (operation, score).
    """
    n = get_n(tableau)
    symplectic = tableau.tableau.matrix

    pairs = _sp_gate_options(symplectic)

    if not pairs:
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    transvections = Transvection.all_two_qubit_transvections()
    scores: list[tuple(Transvection, list[int, ...])] = []
    base_score, _ = score_symplectic(tableau)
    for i, j in pairs:
        for v in transvections:
            op = Transvection(v, i, j)
            tablea_op_applied = op.apply(tableau)
            h_vec, _ = score_symplectic(tablea_op_applied)
            if h_vec < base_score:
                scores.append((op, h_vec))

    scores.sort(key=operator.itemgetter(1))
    return [(tv, score) for tv, score in scores]


def eliminate_non_css_state(
    tableau: StabilizerTableau, optimization_criterion: str = "gates"
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Eliminate a non-CSS stabilizer tableau to state preparation form using transvections.

    Args:
        tableau: The stabilizer tableau to eliminate.
        optimization_criterion: Either "gates" or "depth" for optimization objective.

    Returns:
        A tuple of (operations, final_tableau) where operations is the sequence
        of tableau operations and final_tableau is the reduced tableau.
    """
    config = EliminationConfig.for_non_css_stateprep(optimization_criterion=optimization_criterion)

    operations, final_tableau = eliminate(tableau, config)

    return operations, final_tableau


def eliminate_non_css(
    tableau: StabilizerTableau, optimization_criterion: str = "gates"
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Eliminate a non-CSS stabilizer tableau using transvections.

    Args:
        tableau: The stabilizer tableau to eliminate.
        optimization_criterion: Either "gates" or "depth" for optimization objective.

    Returns:
        A tuple of (operations, final_tableau) where operations is the sequence
        of tableau operations and final_tableau is the reduced tableau.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    config = EliminationConfig.for_non_css(optimization_criterion=optimization_criterion)

    operations, final_tableau = eliminate(tableau, config)

    return operations, final_tableau


def eliminate_non_css_with_lookahead(
    tableau: StabilizerTableau,
    optimization_criterion: str = "gates",
    lookahead: int = 1,
    num_lookahead_candidates: int | list[int] = 10,
    enable_early_termination: bool = True,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Eliminate a non-CSS stabilizer tableau using transvections with lookahead.

    Args:
        tableau: The stabilizer tableau to eliminate.
        optimization_criterion: Either "gates" or "depth" for optimization objective.
        lookahead: Number of steps to look ahead in the synthesis.
        num_lookahead_candidates: Number of top candidates to explore at each lookahead layer.
            Can be a single int (same limit for all layers) or a list of ints (one per layer).
        enable_early_termination: If True, allows early termination when no improving candidates found.

    Returns:
        A tuple of (operations, final_tableau) where operations is the sequence
        of tableau operations and final_tableau is the reduced tableau.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    config = EliminationConfig.for_non_css_with_lookahead(
        optimization_criterion=optimization_criterion,
        lookahead=lookahead,
        num_lookahead_candidates=num_lookahead_candidates,
        enable_early_termination=enable_early_termination,
    )
    operations, final_tableau = eliminate(tableau, config)
    return operations, final_tableau


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
    return zero.astype(np.int8)


def _compute_r1_matrix_from_r2_r0(R2: npt.NDArray[np.int8], R0: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    return (1 ^ (R2 | R0)).astype(np.int8)


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


def is_terminal_stateprep(tableau: StabilizerTableau) -> bool:
    """Check if the given stabilizer tableau is in terminal form for state preparation.

    This is the case when there are no overlaps between any pair of qubits.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to check.

    Returns:
        bool: True if the tableau is in terminal form, False otherwise.
    """
    return score_stateprep(tableau) == 0


def is_terminal_transvection(tableau: StabilizerTableau) -> bool:
    """Check if the given stabilizer tableau is in terminal form for transvection elimination.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to check.

    Returns:
        bool: True if the tableau is in terminal form, False otherwise.
    """
    r1, r2 = r1_r2(tableau.tableau.matrix)
    if np.any(r1):
        return False
    if not np.all(r2.sum(axis=0) == 1):
        return False
    return np.all(r2.sum(axis=1) == 1)


def score_stateprep(tableau: StabilizerTableau) -> int:
    r"""Score the given symplectic matrix representing a state.

    The score is the total number of "overlap" between qubit pairs, i.e., where there is a
    "1" for both qubits.

    Args:
        tableau: The stabilizer tableau to score.

    Returns:
        An integer score used for comparing tableaus.
    """
    n = get_n(tableau)
    symplectic = tableau.tableau.matrix
    symplectic.shape[0]
    score = 0
    for q1 in range(n):
        for q2 in range(q1 + 1, n):
            x1 = symplectic[:, q1]
            z1 = symplectic[:, q1 + n]
            x2 = symplectic[:, q2]
            z2 = symplectic[:, q2 + n]

            score += ((x1 & x2) | (x1 & z2) | (z1 & x2) | (z1 & z2)).sum()

    return score


def score_symplectic(tableau: StabilizerTableau) -> tuple[tuple[int, ...], int]:
    """Score the given symplectic matrix using the default symplectic heuristic.

    Args:
        tableau: The stabilizer tableau to score.

    Returns:
        A tuple of (heuristic_vector, scalar_score) used for comparing tableaus.
    """
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
    """Reduce a TERMINAL symplectic matrix by applying SWAPs to align blocks on diagonal.

    Args:
        tableau: A stabilizer tableau in terminal form (permutation matrix of 2x2 blocks).

    Returns:
        A tuple of (swap_sequence, tableau_after_swaps) where the blocks are now diagonal.
    """
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
        tableau_copy = swap.apply(tableau_copy, inplace=True)
        swap_sequence.add_operation(swap)
    return swap_sequence, tableau_copy


def reduce_with_single_qubit_cliffords(
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce diagonal blocks to identity using single-qubit Cliffords and Paulis.

    Args:
        tableau: A stabilizer tableau where each qubit has a 2x2 block on its diagonal.

    Returns:
        A tuple of (clifford_sequence, final_tableau) where final_tableau should be identity.
    """
    tableau_copy = tableau.copy()
    n = get_n(tableau)

    clifford_sequence = EliminationSequence([])

    for q in range(n):
        f = tableau_copy.symplectic_submatrix(q)
        op = SingleQubitClifford.from_symplectic_block(f, q)
        clifford_sequence.add_operation(op)
        tableau_copy = op.apply(tableau_copy, inplace=True)

    pauli_ops = fix_tableau_signs_in_place(tableau_copy)
    for op in pauli_ops:
        clifford_sequence.add_operation(op)
    return clifford_sequence, tableau_copy


def reduce_single_qubit_gates_and_swaps(
    operations: EliminationSequence,
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce a TERMINAL symplectic matrix to identity using SWAP/H/S/Pauli gates.

    This function combines swap-based permutation correction with single-qubit Clifford
    reduction to bring a terminal-form tableau to the identity.

    Args:
        operations: The elimination sequence (unused but required by post_process_fn signature).
        tableau: A stabilizer tableau in terminal form.

    Returns:
        A tuple of (operation_sequence, final_tableau) where final_tableau is identity.
    """
    swap_seq, tableau_after_swaps = reduce_with_swaps(tableau)

    clifford_seq, final_tableau = reduce_with_single_qubit_cliffords(tableau_after_swaps)

    operations.extend(EliminationSequence(swap_seq.operations + clifford_seq.operations))

    return operations, final_tableau


def reduce_without_swaps(
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce a TERMINAL symplectic matrix to a permuted identity using only single-qubit gates.

    This variant does NOT apply SWAPs, so the final tableau will be a permutation of the
    identity (i.e., blocks aligned but possibly permuted).

    Args:
        tableau: A stabilizer tableau in terminal form.

    Returns:
        A tuple of (operation_sequence, final_tableau) where final_tableau is a
        permuted identity.
    """
    return reduce_with_single_qubit_cliffords(tableau)


def reduce_single_qubit_gates_stateprep(
    operations: EliminationSequence,
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce a state preparation tableau using only single-qubit gates.

    Args:
        operations: The elimination sequence.
        tableau: A stabilizer tableau in terminal form for state preparation.

    Returns:
        A tuple of (operation_sequence, final_tableau).
    """
    return reduce_without_swaps(tableau)


def _extract_perm_in_to_out_and_blocks(tableau: StabilizerTableau) -> tuple[EliminationSequence, StabilizerTableau]:
    """Extract the permutation and corresponding 2×2 blocks from a terminal symplectic matrix.

    This function processes a terminal symplectic matrix `U` to determine the permutation
    of input qubits to output qubits and the associated 2×2 symplectic blocks.

    Args:
        U: A 2n×2n symplectic matrix in terminal form.

    Returns:
        A tuple containing:
        - perm: A 1D array where `perm[i]` gives the index `j` such that the determinant
          of the 2×2 block F_ij is 1 (indicating a valid symplectic transformation).
        - blocks: A list of 2×2 symplectic blocks corresponding to the permutation.
    """
    n = get_n(tableau)
    symplectic = tableau.tableau.matrix
    r2 = _compute_r2_matrix(symplectic)

    perm = np.full(n, -1, dtype=int)
    blocks: list[np.ndarray] = [None] * n

    for i in range(n):
        js = np.flatnonzero(r2[i])
        if len(js) != 1:
            msg = "Not terminal: R2 row is not one-hot."
            raise ValueError(msg)
        j = int(js[0])
        perm[i] = j
        blocks[i] = np.array(
            [
                [int(symplectic[i, j]), int(symplectic[i, j + n])],
                [int(symplectic[i + n, j]), int(symplectic[i + n, j + n])],
            ],
            dtype=np.int8,
        )

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
    """Return a SWAP list that realizes perm_in_to_out when right-multiplying
    the symplectic matrix, i.e. permuting columns (wires).
    """
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


def fix_tableau_signs_in_place(tableau: StabilizerTableau) -> EliminationSequence:
    """Determine Pauli corrections so that the tableau matches the desired sign bits.

    This function ensures that the tableau matches the target signs
    by appending the necessary Pauli corrections.
    """
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
    ops = []
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
