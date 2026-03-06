# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""CNOT-based candidate generation for CSS codes."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from ..codes.pauli import CheckMatrix
from .elimination import (
    CandidateGenerator,
    get_n,
)
from .operations import CNOT

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from .elimination import OperationFilter
    from .operations import TableauOperation
    from .types import BinaryMatrix


class GreedyCNOTGenerator(CandidateGenerator):
    """Generates CNOT candidates using greedy heuristic for CSS codes."""

    def __init__(self, filters: Sequence[OperationFilter] | None = None) -> None:
        """Initialize the greedy CNOT generator.

        Args:
            filters: Optional list of filters to apply during candidate generation.
        """
        self.operation_history: list[TableauOperation] = []
        self.filters = list(filters) if filters else []

    def get_candidates(self, tableau: BinaryMatrix) -> Sequence[tuple[TableauOperation, int | tuple[int, ...]]]:
        """Generate CNOT candidates sorted by heuristic score.

        Args:
            tableau: The current check matrix.

        Returns:
            List of (operation, score) tuples sorted by preference.
        """
        assert isinstance(tableau, CheckMatrix), "Input must be a CheckMatrix."
        unscored_candidates = _generate_cnot_operations(tableau)
        filtered_candidates = self._apply_filters(unscored_candidates)
        scored = _score_cnots(filtered_candidates, tableau)
        if scored:
            return scored

        self._reset_filters()
        filtered_candidates = self._apply_filters(unscored_candidates)
        return _score_cnots(unscored_candidates, tableau)

    def _reset_filters(self) -> None:
        """Reset all filters to their initial state."""
        for f in self.filters:
            f.reset()

    def _apply_filters(self, candidates: Sequence[CNOT]) -> list[CNOT]:
        """Apply all filters to candidate list.

        Args:
            candidates: List of candidate operations.

        Returns:
            Filtered list of candidates.
        """
        if not self.filters:
            return list(candidates)

        filtered = [op for op in candidates if all(f.should_include(op) for f in self.filters)]

        if not filtered:
            self._reset_filters()
            return list(candidates)

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


_CNOT_CACHE: dict[int, list[CNOT]] = {}


def _generate_cnot_operations(matrix: BinaryMatrix) -> list[CNOT]:
    """Generate all possible CNOT operations without scoring.

    Args:
        matrix: The CSS check matrix.

    Returns:
        List of all possible CNOT operations.
    """
    n = get_n(matrix)

    if n not in _CNOT_CACHE:
        _CNOT_CACHE[n] = [CNOT(i, j) for i in range(n) for j in range(n) if i != j]

    return _CNOT_CACHE[n]


@nb.njit(nb.int64[:](nb.int64[:, :], nb.int64[:], nb.int64[:], nb.int64[:]), cache=True)  # type: ignore[untyped-decorator]
def _compute_scores_numba(
    mat: npt.NDArray[np.int64],
    controls: npt.NDArray[np.int64],
    targets: npt.NDArray[np.int64],
    col_weights: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Compute scores using numba for speed."""
    scores = np.empty(len(controls), dtype=np.int64)
    for i in range(len(controls)):
        old_weight = col_weights[targets[i]]
        new_weight = 0
        for j in range(mat.shape[0]):
            new_weight += mat[j, targets[i]] ^ mat[j, controls[i]]
        scores[i] = old_weight - new_weight
    return scores


def _score_cnots(
    operations: Sequence[CNOT], matrix: CheckMatrix
) -> list[tuple[TableauOperation, int | tuple[int, ...]]]:
    """Score CNOT operations and return sorted list."""
    mat = matrix.matrix
    col_weights = mat.sum(axis=0, dtype=np.int64)

    controls = np.array([op.control for op in operations], dtype=np.int64)
    targets = np.array([op.target for op in operations], dtype=np.int64)

    scores = _compute_scores_numba(mat, controls, targets, col_weights)

    # Filter positive scores
    positive_mask = scores > 0

    if not positive_mask.any():
        return []

    positive_indices = np.where(positive_mask)[0]
    positive_scores = scores[positive_indices]
    sorted_order = np.argsort(-positive_scores)  # Negate for descending order

    return [(operations[positive_indices[i]], int(positive_scores[i])) for i in sorted_order]


def greedy_matrix_elimination_candidates(matrix: CheckMatrix) -> list[tuple[CNOT, int]]:
    """Get all possible CNOT candidates for a CSS check matrix.

    Args:
        matrix (BinaryMatrix): The CSS check matrix.

    Returns:
        list[tuple[CNOT, int]]: A list of (CNOT, score) tuples that can be applied.
    """
    matrix_copy = matrix.copy()
    n = get_n(matrix)
    candidates: list[tuple[CNOT, int]] = []
    weight_before = int(matrix_copy.matrix.sum())
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            op = CNOT(i, j)
            matrix_copy = op.apply_check_matrix(matrix_copy, inplace=True)
            weight_after = int(matrix_copy.matrix.sum())
            candidates.append((op, weight_before - weight_after))
            matrix_copy = op.apply_check_matrix(matrix_copy, inplace=True)

    candidates.sort(key=operator.itemgetter(1), reverse=True)
    return candidates
