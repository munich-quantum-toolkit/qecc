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
        enable_early_termination: bool = True,
        current_sequence: EliminationSequence | None = None,
    ) -> None:
        """Initialize the rollout candidate generator.

        Args:
            base_strategy: Base strategy for greedy candidate generation
            rollout: Number of steps to look ahead
            num_rollout_candidates: Number of candidates to explore per layer
            score_fn: Function to score complete elimination sequences
            track_best_solution: If True, tracks best complete solution found during exploration
            enable_early_termination: If True, allows early termination when no improving candidates found
            current_sequence: The elimination sequence built so far (used for depth calculation)
        """
        self.base_strategy = base_strategy
        self.rollout = rollout
        self.num_rollout_candidates_per_layer = _normalize_rollout_candidates(num_rollout_candidates, rollout)
        self.score_fn = score_fn
        self.enable_early_termination = enable_early_termination
        self._current_sequence = current_sequence.copy() if current_sequence is not None else EliminationSequence([])
        self._best_known_score: tuple[int, ...] | None = None
        self._best_known_sequence: EliminationSequence | None = None
        self._best_known_tableau: BinaryMatrix | None = None
        self._should_terminate = False

    def get_base_candidates(self, tableau: BinaryMatrix) -> Sequence[TableauOperation]:
        """Get base candidates from the underlying strategy without scoring.

        Args:
            tableau: The current binary matrix.

        Returns:
            List of candidate operations from the base strategy.
        """
        return [cand for cand, _ in self.base_strategy.candidate_generator.get_candidates(tableau)][
            : self.num_rollout_candidates_per_layer[0]
        ]

    def _update_best_scored_candidates(self, scored_candidates: list[tuple[TableauOperation, tuple[int, ...]]]) -> bool:
        """Update the best known complete solution based on scored candidates.

        Args:
            scored_candidates: List of (operation, score) tuples from the current rollout layer.

        Returns:
            True if an improvement was found, False otherwise.
        """
        improvement_found = False
        for op, score in scored_candidates:
            if self._best_known_score is None or score < self._best_known_score:
                self._best_known_score = score
                new_sequence = self._current_sequence.copy()
                new_sequence.add_operation(op)
                self._best_known_sequence = new_sequence
                improvement_found = True
        return improvement_found

    def score_rollout_candidates(
        self, tableau: BinaryMatrix, base_candidates: Sequence[TableauOperation]
    ) -> list[tuple[TableauOperation, tuple[int, ...]]]:
        """Score base candidates using rollout simulation.

        Args:
            tableau: The current binary matrix.
            base_candidates: List of candidate operations from the base strategy.

        Returns:
            List of (operation, score) tuples sorted by score.
        """
        scored: list[tuple[TableauOperation, tuple[int, ...]]] = []
        for op in base_candidates:
            new_tableau = op.apply(tableau)
            lower_level_strategy = self._create_rollout_strategy(op, None)  # type: ignore[attr-defined]
            seq, _final_tableau = eliminate(new_tableau, lower_level_strategy)
            completed = EliminationSequence([*self._current_sequence.operations, op, *seq.operations])
            score = self.score_fn(completed)
            scored.append((op, score))

        scored.sort(key=operator.itemgetter(1))
        return scored

    def get_candidates(self, tableau: BinaryMatrix) -> Sequence[tuple[TableauOperation, int | tuple[int, ...]]]:
        """Generate candidates using rollout simulation.

        Args:
            tableau: The current binary matrix or tableau.

        Returns:
            List of operations sorted by rollout score.
        """
        if self.rollout <= 0:
            return self.base_strategy.candidate_generator.get_candidates(tableau)

        base_candidates = self.get_base_candidates(tableau)

        scored_candidates = self.score_rollout_candidates(tableau, base_candidates)

        is_improvement = self._update_best_scored_candidates(scored_candidates)

        if not is_improvement and self.enable_early_termination:
            self._should_terminate = True
            return []

        return [(op, score) for op, score in scored_candidates]

    def should_terminate_early(self) -> bool:
        """Check if elimination should terminate early.

        Returns:
            True if early termination is requested, False otherwise.
        """
        return self._should_terminate

    def get_best_solution(self) -> EliminationSequence | None:
        """Get the best complete solution found during rollout exploration.

        Returns:
            Tuple of (sequence, tableau) if a solution is available, None otherwise.
        """
        if self._best_known_sequence is not None:
            return self._best_known_sequence
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
            self._best_known_sequence = sequence.copy()

    def _create_rollout_strategy(
        self, op: TableauOperation, initial_filter_state: list[OperationFilter] | None
    ) -> EliminationStrategy:
        """Create a fresh rollout strategy for recursive simulation.

        Args:
            initial_filter_state: Initial state of filters to copy.

        Returns:
            A new EliminationStrategy with fresh generator and filters.
        """
        # TODO initialize filter state
        return EliminationStrategy(
            termination_criterion=self.base_strategy.termination_criterion,
            candidate_generator=RolloutCandidateGenerator(
                self.base_strategy,
                self.rollout - 1,
                self.num_rollout_candidates_per_layer[1:] if len(self.num_rollout_candidates_per_layer) > 1 else [],
                self.score_fn,
                enable_early_termination=self.enable_early_termination,
                current_sequence=EliminationSequence([*self._current_sequence.operations, op]),
            ),
        )

    def update(self, op: TableauOperation, tableau: BinaryMatrix) -> None:
        """Update internal state based on the operation applied.

        Args:
            op: The operation that was applied to the tableau.
        """
        self._current_sequence.add_operation(op)

    def reset(self) -> None:
        """Reset internal state by delegating to the base generator."""
        self._current_sequence = EliminationSequence([])
        self._best_known_score = None
        self._best_known_sequence = None
        self._best_known_tableau = None
        self._should_terminate = False
        self.base_strategy.candidate_generator.reset()


class RolloutCache:
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


def clear_cache() -> None:
    """Clear global search cache used for rollout synthesis."""
    cache.clear()


cache = RolloutCache()  # global cache instance for rollout synthesis
