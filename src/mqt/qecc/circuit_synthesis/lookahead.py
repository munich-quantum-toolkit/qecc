# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Lookahead-based candidate generation for circuit synthesis."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from ..codes.pauli import StabilizerTableau
from .elimination import CandidateGenerator, EliminationSequence, EliminationStrategy, eliminate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .elimination import OperationFilter
    from .operations import TableauOperation
    from .types import BinaryMatrix


def _normalize_lookahead_candidates(num_candidates: int | list[int], lookahead: int) -> list[int]:
    """Normalize num_lookahead_candidates to a list of proper length.

    Args:
        num_candidates: Either a single int or a list of ints.
        lookahead: The lookahead depth.

    Returns:
        A list of length lookahead where each element is the number of candidates to consider at that layer.
    """
    if isinstance(num_candidates, int):
        return [num_candidates] * lookahead
    if len(num_candidates) < lookahead:
        return list(num_candidates) + [num_candidates[-1]] * (lookahead - len(num_candidates))
    return list(num_candidates[:lookahead])


def _create_tableau_cache_key(tableau: BinaryMatrix) -> bytes:
    """Create a hashable cache key from a tableau.

    Args:
        tableau: The binary matrix or stabilizer tableau.

    Returns:
        A bytes representation suitable for dictionary keys.
    """
    if isinstance(tableau, StabilizerTableau):
        return tableau.tableau.matrix.tobytes()
    return tableau.matrix.tobytes()


def _create_fresh_lookahead_strategy(strategy: EliminationStrategy) -> EliminationStrategy:
    """Create a fresh copy of the lookahead strategy with copied filter state.

    Args:
        strategy: The original lookahead strategy.

    Returns:
        A new strategy with fresh filter copies.
    """
    fresh_filters = None
    if strategy.filters:
        fresh_filters = [f.copy() for f in strategy.filters]

    return EliminationStrategy(
        termination_criterion=strategy.termination_criterion,
        candidate_generator=strategy.candidate_generator,
        selection_strategy=strategy.selection_strategy,
        filters=fresh_filters,
        callback=strategy.callback,
        post_process_fn=strategy.post_process_fn,
    )


def _simulate_and_score_operation(
    op: TableauOperation,
    tableau: BinaryMatrix,
    lookahead_strategy: EliminationStrategy,
    score_fn: Callable[[EliminationSequence], tuple[int, ...]],
    prefix_sequence: EliminationSequence,
    generator: LookaheadCandidateGenerator | None = None,
) -> tuple[int, ...] | None:
    """Simulate operation and return score tuple, or None if simulation fails.

    Args:
        op: The operation to simulate.
        tableau: The current tableau state.
        lookahead_strategy: Strategy for lookahead elimination.
        score_fn: Function to compute score tuple from a sequence.
        prefix_sequence: The elimination sequence built so far (for depth calculation).
        generator: The lookahead generator to record complete solutions.

    Returns:
        A tuple containing scores if simulation succeeds, None otherwise.
    """
    try:
        new_tableau = op.apply(tableau)

        if lookahead_strategy.filters:
            for f in lookahead_strategy.filters:
                f.update(op)

        sequence, final_tableau = eliminate(new_tableau, lookahead_strategy)
        full_sequence = EliminationSequence([*prefix_sequence.operations, op, *sequence.operations])
        score = score_fn(full_sequence)

        if generator is not None:
            generator.record_complete_solution(full_sequence, final_tableau, score)
        else:
            return score

    except RuntimeError:
        return None

    return None


def _score_candidates_with_lookahead(
    tableau: BinaryMatrix,
    candidates: list[TableauOperation],
    num_candidates: int,
    lookahead_strategy: EliminationStrategy,
    score_fn: Callable[[EliminationSequence], tuple[int, ...]],
    prefix_sequence: EliminationSequence,
    best_known_score: tuple[int, ...] | None = None,
    generator: LookaheadCandidateGenerator | None = None,
) -> list[tuple[TableauOperation, tuple[int, ...]]]:
    """Score candidates using lookahead simulation.

    Args:
        tableau: The current tableau state.
        candidates: List of candidate operations to score.
        num_candidates: Maximum number of candidates to evaluate.
        lookahead_strategy: Strategy for lookahead elimination.
        score_fn: Function to compute score tuple from a sequence.
        prefix_sequence: The elimination sequence built so far (for depth calculation).
        best_known_score: Best score found so far; used to prune candidates that can't improve.
        generator: The lookahead generator to record complete solutions.

    Returns:
        List of (operation, score) tuples sorted by score.
    """
    candidates_to_evaluate = candidates[:num_candidates]
    scored_candidates: list[tuple[TableauOperation, tuple[int, ...]]] = []

    for op in candidates_to_evaluate:
        fresh_strategy = _create_fresh_lookahead_strategy(lookahead_strategy)
        result = _simulate_and_score_operation(op, tableau, fresh_strategy, score_fn, prefix_sequence, generator)
        if result is not None:
            score_tuple = result
            is_minimal = score_tuple[-1] if isinstance(score_tuple[-1], bool) else False

            if best_known_score is None or score_tuple <= best_known_score:
                scored_candidates.append((op, score_tuple))

            if is_minimal:
                break

    scored_candidates.sort(key=operator.itemgetter(1), reverse=False)
    return scored_candidates


class LookaheadCandidateGenerator(CandidateGenerator):
    """Generates candidates using lookahead simulation.

    This generator tracks the best complete solution found during lookahead exploration
    to ensure that greedy local choices lead to globally good solutions.
    """

    def __init__(
        self,
        base_strategy: EliminationStrategy,
        lookahead: int,
        num_lookahead_candidates: int | list[int],
        score_fn: Callable[[EliminationSequence], tuple[int, ...]],
        track_best_solution: bool = True,
        enable_early_termination: bool = True,
    ) -> None:
        """Initialize the lookahead candidate generator.

        Args:
            base_strategy: Base strategy for greedy candidate generation
            lookahead: Number of steps to look ahead
            num_lookahead_candidates: Number of candidates to explore per layer
            score_fn: Function to score complete elimination sequences
            track_best_solution: If True, tracks best complete solution found during exploration
            enable_early_termination: If True, allows early termination when no improving candidates found
        """
        self.base_strategy = base_strategy
        self.lookahead = lookahead
        self.num_lookahead_candidates_per_layer = _normalize_lookahead_candidates(num_lookahead_candidates, lookahead)
        self.score_fn = score_fn
        self.track_best_solution = track_best_solution
        self.enable_early_termination = enable_early_termination
        self.use_best_if_better = track_best_solution and not enable_early_termination
        self._cache: dict[bytes, list[tuple[TableauOperation, int]]] = {}
        self._current_sequence = EliminationSequence([])
        self._best_known_score: tuple[int, ...] | None = None
        self._best_known_sequence: EliminationSequence | None = None
        self._best_known_tableau: BinaryMatrix | None = None
        self._should_terminate = False

    def get_candidates(self, tableau: BinaryMatrix) -> Sequence[tuple[TableauOperation, int | tuple[int, ...]]]:
        """Generate candidates using lookahead simulation.

        Args:
            tableau: The current binary matrix or tableau.

        Returns:
            List of operations sorted by lookahead score.
        """
        if self.lookahead <= 0:
            return self.base_strategy.candidate_generator.get_candidates(tableau)

        if self.base_strategy.filters:
            for f in self.base_strategy.filters:
                f.reset()

        base_candidates = [cand for cand, _ in self.base_strategy.candidate_generator.get_candidates(tableau)]

        num_candidates_this_layer = self.num_lookahead_candidates_per_layer[0]

        current_filter_state = None
        if self.base_strategy.filters:
            current_filter_state = [f.copy() for f in self.base_strategy.filters]

        scored_candidates = _score_candidates_with_lookahead(
            tableau,
            base_candidates,
            num_candidates_this_layer,
            self._create_lookahead_strategy(current_filter_state),
            self.score_fn,
            self._current_sequence,
            self._best_known_score if self.track_best_solution else None,
            self if self.track_best_solution else None,
        )

        if (
            not scored_candidates
            and self.track_best_solution
            and self.enable_early_termination
            and self._best_known_sequence is not None
        ):
            self._should_terminate = True
            return []

        if not scored_candidates:
            scored_candidates = _score_candidates_with_lookahead(
                tableau,
                base_candidates,
                num_candidates_this_layer,
                self._create_lookahead_strategy(current_filter_state),
                self.score_fn,
                self._current_sequence,
                best_known_score=None,
                generator=None,
            )

        if self.track_best_solution and scored_candidates:
            best_candidate_score = scored_candidates[0][1]
            if self._best_known_score is None or best_candidate_score < self._best_known_score:
                self._best_known_score = best_candidate_score

        return [(op, score) for op, score in scored_candidates]

    def should_terminate_early(self) -> bool:
        """Check if elimination should terminate early.

        Returns:
            True if early termination is requested, False otherwise.
        """
        return self._should_terminate

    def get_best_solution(self) -> tuple[EliminationSequence, BinaryMatrix] | None:
        """Get the best complete solution found during lookahead exploration.

        Returns:
            Tuple of (sequence, tableau) if a solution is available, None otherwise.
        """
        if self._best_known_sequence is not None and self._best_known_tableau is not None:
            return self._best_known_sequence, self._best_known_tableau
        return None

    def record_complete_solution(
        self, sequence: EliminationSequence, tableau: BinaryMatrix, score: tuple[int, ...]
    ) -> None:
        """Record a complete solution if it's better than the current best.

        Args:
            sequence: The complete elimination sequence
            tableau: The final tableau
            score: The score of this solution
        """
        if self._best_known_score is None or score < self._best_known_score:
            self._best_known_score = score
            self._best_known_sequence = sequence
            self._best_known_tableau = tableau

    def _create_lookahead_strategy(self, initial_filter_state: list[OperationFilter] | None) -> EliminationStrategy:
        """Create a fresh lookahead strategy for recursive simulation.

        Args:
            initial_filter_state: Initial state of filters to copy.

        Returns:
            A new EliminationStrategy with fresh generator and filters.
        """
        fresh_base_filters = [f.copy() for f in self.base_strategy.filters] if self.base_strategy.filters else []
        fresh_base_generator = type(self.base_strategy.candidate_generator)(fresh_base_filters)

        fresh_base_strategy = EliminationStrategy(
            termination_criterion=self.base_strategy.termination_criterion,
            candidate_generator=fresh_base_generator,
            filters=self.base_strategy.filters,
        )

        return EliminationStrategy(
            termination_criterion=self.base_strategy.termination_criterion,
            candidate_generator=LookaheadCandidateGenerator(
                fresh_base_strategy,
                self.lookahead - 1,
                self.num_lookahead_candidates_per_layer[1:] if len(self.num_lookahead_candidates_per_layer) > 1 else [],
                self.score_fn,
                track_best_solution=self.track_best_solution,
                enable_early_termination=self.enable_early_termination,
            ),
            filters=initial_filter_state,
        )

    def update(self, op: TableauOperation, tableau: BinaryMatrix) -> None:
        """Update internal state by delegating to the base generator.

        Args:
            op: The operation that was applied.
            tableau: The resulting tableau after applying the operation.
        """
        self._current_sequence.add_operation(op)
        self._cache.clear()
        self.base_strategy.candidate_generator.update(op, tableau)

    def reset(self) -> None:
        """Reset internal state by delegating to the base generator."""
        self._current_sequence = EliminationSequence([])
        self._best_known_score = None
        self._best_known_sequence = None
        self._best_known_tableau = None
        self._should_terminate = False
        self.base_strategy.candidate_generator.reset()
