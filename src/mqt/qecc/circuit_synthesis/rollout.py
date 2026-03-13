# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Rollout-based candidate generation for circuit synthesis."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from .elimination import CandidateGenerator, EliminationSequence, EliminationStrategy, eliminate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .elimination import OperationFilter
    from .operations import TableauOperation
    from .types import BinaryMatrix


def _normalize_rollout_candidates(num_candidates: int | list[int], rollout: int) -> list[int]:
    """Normalize num_rollout_candidates to a list of proper length.

    Args:
        num_candidates: Either a single int or a list of ints.
        rollout: The rollout depth.

    Returns:
        A list of length rollout where each element is the number of candidates to consider at that layer.
    """
    if isinstance(num_candidates, int):
        return [num_candidates] * rollout
    if len(num_candidates) < rollout:
        return list(num_candidates) + [num_candidates[-1]] * (rollout - len(num_candidates))
    return list(num_candidates[:rollout])


def _create_fresh_rollout_strategy(strategy: EliminationStrategy) -> EliminationStrategy:
    """Create a fresh copy of the rollout strategy with copied filter state.

    Args:
        strategy: The original rollout strategy.

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
    rollout_strategy: EliminationStrategy,
    score_fn: Callable[[EliminationSequence], tuple[int, ...]],
    prefix_sequence: EliminationSequence,
    generator: RolloutCandidateGenerator | None = None,
    rollout: int = 0,
) -> tuple[int, ...] | None:
    """Simulate operation and return score tuple, or None if simulation fails.

    Args:
        op: The operation to simulate.
        tableau: The current tableau state.
        rollout_strategy: Strategy for rollout elimination.
        score_fn: Function to compute score tuple from a sequence.
        prefix_sequence: The elimination sequence built so far (for depth calculation).
        generator: The rollout generator to record complete solutions.
        rollout: The current rollout depth (used for caching).

    Returns:
        A tuple containing scores if simulation succeeds, None otherwise.
    """
    try:
        new_tableau = op.apply(tableau)
        cached_result = memoization_cache.get(new_tableau, rollout)
        if cached_result is not None:
            seq = EliminationSequence([*prefix_sequence.operations, op, *cached_result])
            score = score_fn(seq)
            if generator is not None:
                generator.record_complete_solution(seq, score)
            return score  # Return the cached score

        if rollout_strategy.filters:
            for f in rollout_strategy.filters:
                f.update(op)

        sequence, _final_tableau = eliminate(new_tableau, rollout_strategy)
        full_sequence = EliminationSequence([*prefix_sequence.operations, op, *sequence.operations])
        score = score_fn(full_sequence)
        new_tableau = tableau
        for i, op_sequence in enumerate(sequence):
            new_tableau = op_sequence.apply(new_tableau)
            memoization_cache.set(new_tableau, rollout, sequence.operations[i:])

        if generator is not None:
            generator.record_complete_solution(full_sequence, score)
        else:
            return score

    except RuntimeError:
        return None

    return None


def _score_candidates_with_rollout(
    tableau: BinaryMatrix,
    candidates: list[TableauOperation],
    num_candidates: int,
    rollout_strategy: EliminationStrategy,
    score_fn: Callable[[EliminationSequence], tuple[int, ...]],
    prefix_sequence: EliminationSequence,
    best_known_score: tuple[int, ...] | None = None,
    generator: RolloutCandidateGenerator | None = None,
    rollout: int = 0,
) -> list[tuple[TableauOperation, tuple[int, ...]]]:
    """Score candidates using rollout simulation.

    Args:
        tableau: The current tableau state.
        candidates: List of candidate operations to score.
        num_candidates: Maximum number of candidates to evaluate.
        rollout_strategy: Strategy for rollout elimination.
        score_fn: Function to compute score tuple from a sequence.
        prefix_sequence: The elimination sequence built so far (for depth calculation).
        best_known_score: Best score found so far; used to prune candidates that can't improve.
        generator: The rollout generator to record complete solutions.
        rollout: The current rollout depth (used for caching).

    Returns:
        List of (operation, score) tuples sorted by score.
    """
    candidates_to_evaluate = candidates[:num_candidates]
    scored_candidates: list[tuple[TableauOperation, tuple[int, ...]]] = []

    for op in candidates_to_evaluate:
        fresh_strategy = _create_fresh_rollout_strategy(rollout_strategy)
        result = _simulate_and_score_operation(
            op, tableau, fresh_strategy, score_fn, prefix_sequence, generator, rollout
        )
        if result is not None:
            score_tuple = result
            is_minimal = score_tuple[-1] if isinstance(score_tuple[-1], bool) else False

            if best_known_score is None or score_tuple <= best_known_score:
                scored_candidates.append((op, score_tuple))

            if is_minimal:
                break

    scored_candidates.sort(key=operator.itemgetter(1), reverse=False)
    return scored_candidates


class RolloutCandidateGenerator(CandidateGenerator):
    """Generates candidates using rollout simulation.

    This generator tracks the best complete solution found during rollout exploration
    to ensure that greedy local choices lead to globally good solutions.
    """

    def __init__(
        self,
        base_strategy: EliminationStrategy,
        rollout: int,
        num_rollout_candidates: int | list[int],
        score_fn: Callable[[EliminationSequence], tuple[int, ...]],
        track_best_solution: bool = True,
        enable_early_termination: bool = True,
    ) -> None:
        """Initialize the rollout candidate generator.

        Args:
            base_strategy: Base strategy for greedy candidate generation
            rollout: Number of steps to look ahead
            num_rollout_candidates: Number of candidates to explore per layer
            score_fn: Function to score complete elimination sequences
            track_best_solution: If True, tracks best complete solution found during exploration
            enable_early_termination: If True, allows early termination when no improving candidates found
        """
        self.base_strategy = base_strategy
        self.rollout = rollout
        self.num_rollout_candidates_per_layer = _normalize_rollout_candidates(num_rollout_candidates, rollout)
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
        """Generate candidates using rollout simulation.

        Args:
            tableau: The current binary matrix or tableau.

        Returns:
            List of operations sorted by rollout score.
        """
        if self.rollout <= 0:
            return self.base_strategy.candidate_generator.get_candidates(tableau)

        if self.base_strategy.filters:
            for f in self.base_strategy.filters:
                f.reset()

        base_candidates = [cand for cand, _ in self.base_strategy.candidate_generator.get_candidates(tableau)]

        num_candidates_this_layer = self.num_rollout_candidates_per_layer[0]

        current_filter_state = None
        if self.base_strategy.filters:
            current_filter_state = [f.copy() for f in self.base_strategy.filters]

        scored_candidates = _score_candidates_with_rollout(
            tableau,
            base_candidates,
            num_candidates_this_layer,
            self._create_rollout_strategy(current_filter_state),
            self.score_fn,
            self._current_sequence,
            self._best_known_score if self.track_best_solution else None,
            self if self.track_best_solution else None,
            rollout=self.rollout,
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
            scored_candidates = _score_candidates_with_rollout(
                tableau,
                base_candidates,
                num_candidates_this_layer,
                self._create_rollout_strategy(current_filter_state),
                self.score_fn,
                self._current_sequence,
                best_known_score=None,
                generator=None,
                rollout=self.rollout,
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
        """Get the best complete solution found during rollout exploration.

        Returns:
            Tuple of (sequence, tableau) if a solution is available, None otherwise.
        """
        if self._best_known_sequence is not None and self._best_known_tableau is not None:
            return self._best_known_sequence, self._best_known_tableau
        return None

    def record_complete_solution(self, sequence: EliminationSequence, score: tuple[int, ...]) -> None:
        """Record a complete solution if it's better than the current best.

        Args:
            sequence: The complete elimination sequence
            tableau: The final tableau
            score: The score of this solution
        """
        if self._best_known_score is None or score < self._best_known_score:
            self._best_known_score = score
            self._best_known_sequence = sequence

    def _create_rollout_strategy(self, initial_filter_state: list[OperationFilter] | None) -> EliminationStrategy:
        """Create a fresh rollout strategy for recursive simulation.

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
            candidate_generator=RolloutCandidateGenerator(
                fresh_base_strategy,
                self.rollout - 1,
                self.num_rollout_candidates_per_layer[1:] if len(self.num_rollout_candidates_per_layer) > 1 else [],
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


class MemoizationCache:
    """Class to handle memoization for rollout synthesis."""

    def __init__(self) -> None:
        """Initialize the memoization cache."""
        self._cache: dict[tuple[int, int], list[TableauOperation]] = {}
        self._hit_count = 0
        self._miss_count = 0

    @staticmethod
    def generate_key(tableau: BinaryMatrix, rollout: int) -> tuple[int, int]:
        """Generate a unique cache key based on the tableau state and rollout depth.

        Args:
            tableau: The current tableau state.
            rollout: The current rollout depth.

        Returns:
            A tuple representing the unique cache key.
        """
        return (hash(tableau), rollout)

    def get(self, tableau: BinaryMatrix, rollout: int) -> list[TableauOperation] | None:
        """Retrieve a cached result if it exists.

        Args:
            tableau: The current tableau state.
            rollout: The current rollout depth.

        Returns:
            The cached result, or None if not found.
        """
        key = self.generate_key(tableau, rollout)
        if key in self._cache:
            self._hit_count += 1
        else:
            self._miss_count += 1
        return self._cache.get(key)

    def set(self, tableau: BinaryMatrix, rollout: int, result: list[TableauOperation]) -> None:
        """Store a result in the cache.

        Args:
            tableau: The current tableau state.
            rollout: The current rollout depth.
            result: The result to cache.
        """
        key = self.generate_key(tableau, rollout)
        self._cache[key] = result

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0

    def size(self) -> int:
        """Get the current size of the cache.

        Returns:
            The number of entries in the cache.
        """
        return len(self._cache)

    def hit_rate(self) -> float:
        """Calculate the cache hit rate.

        Returns:
            The hit rate as a percentage.
        """
        total = self._hit_count + self._miss_count
        return (self._hit_count / total) * 100 if total > 0 else 0.0


memoization_cache = MemoizationCache()  # global cache instance for rollout synthesis
