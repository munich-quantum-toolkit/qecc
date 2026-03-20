# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Factory methods for creating elimination strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2
import numpy as np

from ..codes.pauli import StabilizerTableau
from .cnot import GreedyCNOTGenerator
from .elimination import EliminationStrategy, ParallelFilter
from .rollout import (
    AdditiveCachePolicy,
    NonAdditiveCachePolicy,
    RolloutCandidateGenerator,
    close_rollout_cache_session,
    open_rollout_cache_session,
)
from .transvection import (
    GreedyTransvectionGenerator,
    is_terminal_transvection,
    reduce_single_qubit_gates_and_swaps,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .elimination import EliminationSequence
    from .operations import TableauOperation
    from .rollout import CachePolicy
    from .types import BinaryMatrix


def for_cnot_up_to_row_ops(
    n_stabs: int,
    n: int,
    optimization_criterion: str = "gates",
    callback: Callable[[int, TableauOperation, BinaryMatrix], None] | None = None,
) -> EliminationStrategy:
    """Create strategy for CSS code elimination.

    Args:
        n_stabs: The number of stabilizers.
        n: The number of qubits (columns) in the check matrix
        optimization_criterion: Either "gates" (minimize gate count) or "depth" (minimize circuit depth).
        callback: Optional callback function invoked after each elimination step.

    Returns:
        EliminationStrategy configured for CSS code elimination.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    if optimization_criterion not in {"gates", "depth"}:
        msg = f"Unsupported optimization criterion: {optimization_criterion}"
        raise ValueError(msg)

    filters = [ParallelFilter(n)] if optimization_criterion == "depth" else []

    cached_rank: int | None = None

    def termination_criterion(tbl: BinaryMatrix) -> bool:
        nonlocal cached_rank
        matrix = tbl.matrix  # type: ignore[union-attr]

        if cached_rank is None:
            cached_rank = mod2.rank(matrix)

        target_rank = cached_rank

        # Fast early rejection: check total non-zero columns first
        col_nonzero = np.any(matrix != 0, axis=0)
        total_non_zero_columns = np.sum(col_nonzero)
        if total_non_zero_columns != target_rank:
            return False

        if n_stabs != 0:
            first_rows = matrix[:n_stabs, :]

            # Fast check: rank and non-zero columns must match
            first_nonzero = np.any(first_rows != 0, axis=0)
            non_zero_columns_first = np.sum(first_nonzero)
            rnk_first_rows = mod2.rank(first_rows)
            if non_zero_columns_first != rnk_first_rows:
                return False

            # Get pivot columns from the first n_stabs rows
            first_rows_reduced, _, _, pivot_cols = mod2.row_echelon(first_rows, full=True)
            pivot_cols_set = set(pivot_cols)
        else:
            pivot_cols_set = set()
            first_rows_reduced = None

        # Check remaining rows
        if matrix.shape[0] > n_stabs:
            remaining_rows = matrix[n_stabs:, :]

            if n_stabs > 0:
                # Reduce remaining rows using first n_stabs rows (vectorized)
                remaining_rows = remaining_rows.copy()
                for pivot_row in first_rows_reduced:
                    pivot_col = np.argmax(pivot_row != 0)
                    mask = remaining_rows[:, pivot_col] == 1
                    if np.any(mask):
                        remaining_rows[mask] = (remaining_rows[mask] + pivot_row) % 2

            # Fast check: all rows must have exactly one non-zero
            row_nonzero_count = np.sum(remaining_rows != 0, axis=1)
            if not np.all(row_nonzero_count == 1):
                return False

            # Check non-pivot columns for duplicates using vectorized operation
            if pivot_cols_set:
                non_pivot_mask = np.array([c not in pivot_cols_set for c in range(matrix.shape[1])])
                non_pivot_remaining = remaining_rows[:, non_pivot_mask]
            else:
                non_pivot_remaining = remaining_rows

            if non_pivot_remaining.shape[1] > 0:
                col_sums = np.sum(non_pivot_remaining != 0, axis=0)
                if np.any(col_sums > 1):
                    return False

        return True

    return EliminationStrategy(
        termination_criterion=termination_criterion,
        candidate_generator=GreedyCNOTGenerator(n_stabs, filters),
        filters=filters,
        callback=callback,
    )


def for_non_css(
    n: int,
    optimization_criterion: str = "gates",
    callback: Callable[[int, TableauOperation, BinaryMatrix], None] | None = None,
) -> EliminationStrategy:
    """Create strategy for non-CSS stabilizer code elimination.

    Args:
        n: Number of qubits.
        optimization_criterion: Either "gates" (minimize gate count) or "depth" (minimize circuit depth).

        callback: Optional callback function invoked after each elimination step.

    Returns:
        EliminationStrategy configured for non-CSS code elimination with post-processing.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    if optimization_criterion not in {"gates", "depth"}:
        msg = f"Unsupported optimization criterion: {optimization_criterion}"
        raise ValueError(msg)

    filters = [ParallelFilter(n)] if optimization_criterion == "depth" else []

    def termination_criterion(tbl: BinaryMatrix) -> bool:
        if not isinstance(tbl, StabilizerTableau):
            return False
        return is_terminal_transvection(tbl)

    def post_process_fn(ops: EliminationSequence, tbl: BinaryMatrix) -> tuple[EliminationSequence, BinaryMatrix]:
        if not isinstance(tbl, StabilizerTableau):
            return ops, tbl
        return reduce_single_qubit_gates_and_swaps(ops, tbl)

    return EliminationStrategy(
        termination_criterion=termination_criterion,
        candidate_generator=GreedyTransvectionGenerator(filters),
        filters=filters,
        callback=callback,
        post_process_fn=post_process_fn,
    )


def for_non_css_with_rollout(
    n: int,
    optimization_criterion: str = "gates",
    rollout: int = 1,
    num_rollout_candidates: int | list[int] = 10,
    enable_early_termination: bool = True,
    callback: Callable[[int, TableauOperation, BinaryMatrix], None] | None = None,
    cache_max_weight: int = 1_000_000,
    num_cached_rollout_subsequences: int = 10,
) -> EliminationStrategy:
    """Create strategy for non-CSS elimination with rollout.

    Args:
        n: Number of qubits.
        optimization_criterion: Either "gates" (minimize gate count) or "depth" (minimize circuit depth).
        rollout: Number of steps to look ahead when selecting operations.
        num_rollout_candidates: Number of top candidates to explore at each rollout layer.
            Can be a single int (same limit for all layers) or a list of ints (one per layer).
        enable_early_termination: If True, allows early termination when no improving candidates found.
        callback: Optional callback function invoked after each elimination step.
        cache_max_weight: Maximum weight for cached subsequences in rollout.
        num_cached_rollout_subsequences: Number of subsequences to cache for rollout.

    Returns:
        EliminationStrategy configured for non-CSS code elimination with rollout and post-processing.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    if optimization_criterion not in {"gates", "depth"}:
        msg = f"Unsupported optimization criterion: {optimization_criterion}"
        raise ValueError(msg)

    filters = [ParallelFilter(n)] if optimization_criterion == "depth" else []

    def termination_criterion(tbl: BinaryMatrix) -> bool:
        if not isinstance(tbl, StabilizerTableau):
            return False
        return is_terminal_transvection(tbl)

    base_strategy = EliminationStrategy(
        termination_criterion=termination_criterion,
        candidate_generator=GreedyTransvectionGenerator(filters),
        filters=filters,
    )

    if optimization_criterion == "gates":

        def score_fn(ops: EliminationSequence) -> tuple[int, int]:
            return ops.num_transvections(), ops.depth()

        policy: CachePolicy = AdditiveCachePolicy()
    else:

        def score_fn(ops: EliminationSequence) -> tuple[int, int]:
            return ops.depth(), ops.num_transvections()

        policy = NonAdditiveCachePolicy()

    def post_process_fn(ops: EliminationSequence, tbl: BinaryMatrix) -> tuple[EliminationSequence, BinaryMatrix]:
        if not isinstance(tbl, StabilizerTableau):
            return ops, tbl
        return reduce_single_qubit_gates_and_swaps(ops, tbl)

    return EliminationStrategy(
        termination_criterion=termination_criterion,
        candidate_generator=RolloutCandidateGenerator(
            base_strategy,
            rollout,
            num_rollout_candidates,
            score_fn,
            enable_early_termination=enable_early_termination,
            cache_policy=policy,
            num_cached_subsequences=num_cached_rollout_subsequences,
        ),
        filters=filters,
        callback=callback,
        post_process_fn=post_process_fn,
        setup_fn=lambda: open_rollout_cache_session(cache_max_weight),
        cleanup_fn=close_rollout_cache_session,
    )


def for_cnot_with_rollout_up_to_row_ops(
    n_stabs: int,
    n: int,
    rollout: int = 1,
    num_rollout_candidates: int | list[int] = 10,
    optimization_criterion: str = "gates",
    enable_early_termination: bool = True,
    callback: Callable[[int, TableauOperation, BinaryMatrix], None] | None = None,
    cache_max_weight: int = 1_000_000,
    num_cached_rollout_subsequences: int = 10,
) -> EliminationStrategy:
    r"""Create strategy for CSS elimination with rollout.

    Args:
        n_stabs: The target rank of the check matrix after elimination.
        n: The number of qubits (columns) in the check matrix
        rollout: Number of steps to look ahead when selecting operations.
        num_rollout_candidates: Number of top candidates to explore at each rollout layer.
            Can be a single int (same limit for all layers) or a list of ints (one per layer).
        optimization_criterion: Either "gates" or "depth" for optimization objective.
        enable_early_termination: If True, allows early termination when no improving candidates found.
        callback: Optional callback function invoked after each elimination step.
        cache_max_weight: Maximum weight for cached subsequences in rollout.
        num_cached_rollout_subsequences: Number of subsequences to cache for rollout.

    Returns:
        EliminationStrategy configured for CSS code elimination with rollout.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    base_strategy = for_cnot_up_to_row_ops(
        n_stabs=n_stabs,
        n=n,
        optimization_criterion=optimization_criterion,
        callback=None,
    )

    if optimization_criterion == "gates":

        def _score_fn(ops: EliminationSequence) -> tuple[int, int]:
            return (ops.num_cnots(), ops.depth())

        policy: CachePolicy = AdditiveCachePolicy()

    else:

        def _score_fn(ops: EliminationSequence) -> tuple[int, int]:
            return (ops.depth(), ops.num_cnots())

        policy = NonAdditiveCachePolicy()

    return EliminationStrategy(
        termination_criterion=base_strategy.termination_criterion,
        candidate_generator=RolloutCandidateGenerator(
            base_strategy,
            rollout,
            num_rollout_candidates,
            _score_fn,
            enable_early_termination=enable_early_termination,
            cache_policy=policy,
            num_cached_subsequences=num_cached_rollout_subsequences,
        ),
        filters=None,
        callback=callback,
        setup_fn=lambda: open_rollout_cache_session(cache_max_weight),
        cleanup_fn=close_rollout_cache_session,
    )
