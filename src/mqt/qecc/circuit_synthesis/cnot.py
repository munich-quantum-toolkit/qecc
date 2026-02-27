# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""CNOT-based elimination for CSS codes."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2

from .elimination import (
    CandidateGenerator,
    EliminationConfig,
    EliminationSequence,
    eliminate,
    get_n,
)
from .operations import CNOT

if TYPE_CHECKING:
    from ..codes.pauli import CheckMatrix
    from .elimination import (
        BinaryMatrix,
        OperationFilter,
        TableauOperation,
    )


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
        all_candidates = greedy_matrix_elimination_candidates(tableau)
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


def eliminate_cnot_lookahead(
    matrix: CheckMatrix,
    optimization_criterion: str = "gates",
    lookahead: int = 1,
    num_lookahead_candidates: int | list[int] = 10,
) -> tuple[EliminationSequence, CheckMatrix]:
    """Eliminate a CSS check matrix using CNOT operations with lookahead.

    Args:
        matrix: The CSS check matrix to eliminate.
        optimization_criterion: Either "gates" or "depth" for optimization objective.
        lookahead: Number of steps to look ahead in the synthesis.
        num_lookahead_candidates: Number of top candidates to explore at each lookahead layer.
            Can be a single int (same limit for all layers) or a list of ints (one per layer).

    Returns:
        A tuple of (operations, final_matrix) where operations is the sequence
        of CNOT operations and final_matrix is the reduced check matrix.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    config = EliminationConfig.for_cnot_with_lookahead(
        optimization_criterion=optimization_criterion,
        lookahead=lookahead,
        num_lookahead_candidates=num_lookahead_candidates,
    )
    operations, final_matrix = eliminate(matrix, config)
    return operations, final_matrix


def eliminate_cnot(
    matrix: CheckMatrix,
    optimization_criterion: str = "gates",
    exact: bool = True,
    lookahead: int = 0,
    num_lookahead_candidates: int | list[int] = 10,
    enable_early_termination: bool = True,
) -> tuple[EliminationSequence, CheckMatrix]:
    """Eliminate a CSS check matrix using CNOT operations.

    Args:
        matrix: The CSS check matrix to eliminate.
        optimization_criterion: Either "gates" or "depth" for optimization objective.
        exact: If True, eliminate to echelon form. If False, eliminate only up to row operations.
        lookahead: Number of steps to look ahead (0 = greedy).
        num_lookahead_candidates: Number of candidates to explore at each lookahead layer.
        enable_early_termination: If True, allows early termination when no improving candidates found.

    Returns:
        A tuple of (operations, final_matrix) where operations is the sequence
        of CNOT operations and final_matrix is the reduced check matrix.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    if matrix.num_rows() == 0:
        return EliminationSequence([]), matrix.copy()

    target_rank = mod2.rank(matrix.matrix)
    if exact:
        if lookahead > 0:
            config = EliminationConfig.for_cnot_with_lookahead_exact(
                target_rank=target_rank,
                optimization_criterion=optimization_criterion,
                lookahead=lookahead,
                num_lookahead_candidates=num_lookahead_candidates,
                enable_early_termination=enable_early_termination,
            )
        else:
            config = EliminationConfig.for_cnot_exact(
                target_rank=target_rank, optimization_criterion=optimization_criterion
            )
    elif lookahead > 0:
        config = EliminationConfig.for_cnot_with_lookahead_up_to_row_ops(
            optimization_criterion=optimization_criterion,
            lookahead=lookahead,
            num_lookahead_candidates=num_lookahead_candidates,
            target_rank=target_rank,
            enable_early_termination=enable_early_termination,
        )
    else:
        config = EliminationConfig.for_cnot_up_to_row_ops(
            target_rank=target_rank, optimization_criterion=optimization_criterion
        )

    operations, final_matrix = eliminate(matrix, config)

    if matrix.is_z_type():
        for op in operations.operations:
            if isinstance(op, CNOT):
                op.control, op.target = op.target, op.control

    return operations, final_matrix
