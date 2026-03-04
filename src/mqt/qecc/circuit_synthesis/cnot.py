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

from .elimination import (
    CandidateGenerator,
    get_n,
)
from .operations import CNOT

if TYPE_CHECKING:
    from .elimination import OperationFilter
    from .operations import TableauOperation
    from .types import BinaryMatrix


class GreedyCNOTGenerator(CandidateGenerator):
    """Generates CNOT candidates using greedy heuristic for CSS codes."""

    def __init__(self, filters: list[OperationFilter] | None = None) -> None:
        """Initialize the greedy CNOT generator.

        Args:
            filters: Optional list of filters to apply during candidate generation.
        """
        self.operation_history: list[TableauOperation] = []
        self.filters = filters or []

    def get_candidates(self, tableau: BinaryMatrix) -> list[TableauOperation]:
        """Generate CNOT candidates sorted by heuristic score.

        Args:
            tableau: The current check matrix.

        Returns:
            List of CNOT operations sorted by preference.
        """
        unscored_candidates = _generate_cnot_operations(tableau)
        filtered_candidates = self._apply_filters(unscored_candidates)
        return _score_cnots(filtered_candidates, tableau)

    def _apply_filters(self, candidates: list[TableauOperation]) -> list[TableauOperation]:
        """Apply all filters to candidate list.

        Args:
            candidates: List of candidate operations.

        Returns:
            Filtered list of candidates.
        """
        if not self.filters:
            return candidates

        if len(self.filters) == 1:
            f = self.filters[0]
            filtered = [op for op in candidates if f.should_include(op)]
        else:
            filtered = []
            for op in candidates:
                for f in self.filters:
                    if not f.should_include(op):
                        break
                else:
                    filtered.append(op)

        if not filtered:
            for f in self.filters:
                if hasattr(f, "_reset"):
                    f._reset()
            return candidates

        return filtered

    def update(self, op: TableauOperation, tableau: BinaryMatrix) -> None:
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


def _generate_cnot_operations(matrix: BinaryMatrix) -> list[CNOT]:
    """Generate all possible CNOT operations without scoring.

    Args:
        matrix: The CSS check matrix.

    Returns:
        List of all possible CNOT operations.
    """
    n = get_n(matrix)
    operations = []
    for i in range(n):
        operations.extend(CNOT(i, j) for j in range(n) if i != j)
    return operations


def _score_cnots(operations: list[CNOT], matrix: BinaryMatrix) -> list[tuple[CNOT, int]]:
    """Score CNOT operations and return sorted list.

    Args:
        operations: List of CNOT operations to score.
        matrix: The current check matrix.

    Returns:
        List of (operation, score) tuples sorted by score.
    """
    matrix = matrix.copy()
    weight_before = int(matrix.matrix.sum())
    scored = []

    for op in operations:
        matrix = op.apply(matrix, inplace=True, css=True)
        weight_after = int(matrix.matrix.sum())
        score = weight_before - weight_after
        scored.append((op, score))
        matrix = op.apply(matrix, inplace=True, css=True)

    scored.sort(key=operator.itemgetter(1), reverse=True)
    return scored


def greedy_matrix_elimination_candidates(matrix: BinaryMatrix) -> list[CNOT]:
    """Get all possible CNOT candidates for a CSS check matrix.

    Args:
        matrix (BinaryMatrix): The CSS check matrix.

    Returns:
        list[CNOT]: A list of CNOT operations that can be applied.
    """
    matrix = matrix.copy()
    n = get_n(matrix)
    candidates: list[tuple[CNOT, int]] = []
    weight_before = int(matrix.matrix.sum())
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            op = CNOT(i, j)
            matrix = op.apply(matrix, inplace=True)
            weight_after = int(matrix.matrix.sum())
            candidates.append((op, weight_before - weight_after))
            matrix = op.apply(matrix, inplace=True)

    candidates.sort(key=operator.itemgetter(1), reverse=True)
    return [(op, score) for op, score in candidates]
