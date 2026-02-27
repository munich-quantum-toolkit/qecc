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

from .elimination import CandidateGenerator, EliminationSequence, EliminationStrategy, eliminate

if TYPE_CHECKING:
    from collections.abc import Callable

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


def _create_fresh_base_generator(base_generator: CandidateGenerator) -> CandidateGenerator:
    """Create a fresh copy of the base candidate generator.

    Args:
        base_generator: The original base generator.

    Returns:
        A new generator instance with fresh state.
    """
    fresh_filters = (
        [f.copy() for f in base_generator.filters]
        if hasattr(base_generator, "filters") and base_generator.filters
        else []
    )
    return type(base_generator)(fresh_filters)


def _create_fresh_filters(filters: list | None) -> list | None:
    """Create fresh copies of filters.

    Args:
        filters: List of filters to copy, or None.

    Returns:
        List of copied filters, or None if input was None.
    """
    if filters is None:
        return None
    return [f.copy() for f in filters]


def _simulate_single_step(
    op: TableauOperation,
    tableau: BinaryMatrix,
    base_strategy: EliminationStrategy,
) -> tuple[EliminationSequence, BinaryMatrix] | None:
    """Simulate a single operation followed by greedy elimination to completion.

    Args:
        op: The operation to simulate.
        tableau: The current tableau state.
        base_strategy: Base strategy with fresh generator and filters.

    Returns:
        Tuple of (sequence, final_tableau) if simulation succeeds, None otherwise.
    """
    try:
        new_tableau = op.apply(tableau)

        if base_strategy.filters:
            for f in base_strategy.filters:
                f.update(op)

        sequence, final_tableau = eliminate(new_tableau, base_strategy)
        return sequence, final_tableau
    except RuntimeError:
        return None


def _score_operation(
    op: TableauOperation,
    tableau: BinaryMatrix,
    base_strategy: EliminationStrategy,
    score_fn: Callable[[EliminationSequence], tuple[int, ...]],
    prefix_sequence: EliminationSequence,
) -> tuple[int, ...] | None:
    """Score a single operation by simulating to completion.

    Args:
        op: The operation to score.
        tableau: The current tableau state.
        base_strategy: Base strategy with fresh state.
        score_fn: Function to compute score from sequence.
        prefix_sequence: The elimination sequence built so far.

    Returns:
        Score tuple if simulation succeeds, None otherwise.
    """
    result = _simulate_single_step(op, tableau, base_strategy)
    if result is None:
        return None

    sequence, _ = result
    full_sequence = EliminationSequence([*prefix_sequence.operations, op, *sequence.operations])
    return score_fn(full_sequence)


def _should_prune_candidate(score: tuple[int, ...], best_known_score: tuple[int, ...] | None) -> bool:
    """Check if a candidate should be pruned based on its score.

    Args:
        score: The candidate's score.
        best_known_score: The best score found so far.

    Returns:
        True if the candidate should be pruned, False otherwise.
    """
    if best_known_score is None:
        return False
    return score > best_known_score


class SolutionTracker:
    """Tracks the best complete solution found during lookahead exploration."""

    def __init__(self) -> None:
        """Initialize the solution tracker."""
        self.best_score: tuple[int, ...] | None = None
        self.best_sequence: EliminationSequence | None = None
        self.best_tableau: BinaryMatrix | None = None

    def record_solution(self, sequence: EliminationSequence, tableau: BinaryMatrix, score: tuple[int, ...]) -> None:
        """Record a complete solution if it's better than the current best.

        Args:
            sequence: The complete elimination sequence.
            tableau: The final tableau.
            score: The score of this solution.
        """
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_sequence = sequence
            self.best_tableau = tableau

    def update_best_score(self, score: tuple[int, ...]) -> None:
        """Update best score without storing full solution.

        Args:
            score: The new best score.
        """
        if self.best_score is None or score < self.best_score:
            self.best_score = score

    def get_solution(self) -> tuple[EliminationSequence, BinaryMatrix] | None:
        """Get the best solution found.

        Returns:
            Tuple of (sequence, tableau) if available, None otherwise.
        """
        if self.best_sequence is not None and self.best_tableau is not None:
            return self.best_sequence, self.best_tableau
        return None


class LayerSimulator:
    """Simulates lookahead layers for candidate scoring."""

    def __init__(
        self,
        base_strategy: EliminationStrategy,
        score_fn: Callable[[EliminationSequence], tuple[int, ...]],
        track_solutions: bool,
    ) -> None:
        """Initialize the layer simulator.

        Args:
            base_strategy: Strategy for base candidate generation.
            score_fn: Function to score elimination sequences.
            track_solutions: Whether to track complete solutions.
        """
        self.base_strategy = base_strategy
        self.score_fn = score_fn
        self.track_solutions = track_solutions
        self.solution_tracker = SolutionTracker() if track_solutions else None

    def simulate_layer(
        self,
        tableau: BinaryMatrix,
        candidates: list[TableauOperation],
        num_candidates: int,
        prefix_sequence: EliminationSequence,
        remaining_depth: int,
        initial_filters: list | None,
    ) -> list[tuple[TableauOperation, tuple[int, ...]]]:
        """Simulate one layer of lookahead.

        Args:
            tableau: Current tableau state.
            candidates: Candidate operations to evaluate.
            num_candidates: Maximum number of candidates to evaluate.
            prefix_sequence: Sequence of operations applied so far.
            remaining_depth: Remaining lookahead depth after this layer.
            initial_filters: Filter state at start of this layer.

        Returns:
            List of (operation, score) tuples sorted by score.
        """
        if remaining_depth == 0:
            return self._simulate_terminal_layer(tableau, candidates, num_candidates, prefix_sequence, initial_filters)

        return self._simulate_recursive_layer(
            tableau, candidates, num_candidates, prefix_sequence, remaining_depth, initial_filters
        )

    def _simulate_terminal_layer(
        self,
        tableau: BinaryMatrix,
        candidates: list[TableauOperation],
        num_candidates: int,
        prefix_sequence: EliminationSequence,
        initial_filters: list | None,
    ) -> list[tuple[TableauOperation, tuple[int, ...]]]:
        """Simulate terminal layer using base strategy only.

        Args:
            tableau: Current tableau state.
            candidates: Candidate operations to evaluate.
            num_candidates: Maximum number of candidates to evaluate.
            prefix_sequence: Sequence of operations applied so far.
            initial_filters: Filter state at start of this layer.

        Returns:
            List of (operation, score) tuples sorted by score.
        """
        scored_candidates: list[tuple[TableauOperation, tuple[int, ...]]] = []
        best_score = self.solution_tracker.best_score if self.solution_tracker else None

        for op in candidates[:num_candidates]:
            fresh_filters = _create_fresh_filters(initial_filters)
            fresh_base_strategy = self._create_base_strategy(fresh_filters)

            result = _simulate_single_step(op, tableau, fresh_base_strategy)
            if result is None:
                continue

            sequence, final_tableau = result
            full_sequence = EliminationSequence([*prefix_sequence.operations, op, *sequence.operations])
            score = self.score_fn(full_sequence)

            if self.solution_tracker:
                self.solution_tracker.record_solution(full_sequence, final_tableau, score)

            if not _should_prune_candidate(score, best_score):
                scored_candidates.append((op, score))
                best_score = score if best_score is None or score < best_score else best_score

        scored_candidates.sort(key=operator.itemgetter(1))
        return scored_candidates

    def _simulate_recursive_layer(
        self,
        tableau: BinaryMatrix,
        candidates: list[TableauOperation],
        num_candidates: int,
        prefix_sequence: EliminationSequence,
        remaining_depth: int,
        initial_filters: list | None,
    ) -> list[tuple[TableauOperation, tuple[int, ...]]]:
        """Simulate layer with recursive lookahead.

        Args:
            tableau: Current tableau state.
            candidates: Candidate operations to evaluate.
            num_candidates: Maximum number of candidates to evaluate.
            prefix_sequence: Sequence of operations applied so far.
            remaining_depth: Remaining lookahead depth after this layer.
            initial_filters: Filter state at start of this layer.

        Returns:
            List of (operation, score) tuples sorted by score.
        """
        scored_candidates: list[tuple[TableauOperation, tuple[int, ...]]] = []
        best_score = self.solution_tracker.best_score if self.solution_tracker else None

        for op in candidates[:num_candidates]:
            fresh_filters = _create_fresh_filters(initial_filters)
            lookahead_strategy = self._create_recursive_strategy(remaining_depth - 1, fresh_filters)

            result = _simulate_single_step(op, tableau, lookahead_strategy)
            if result is None:
                continue

            sequence, final_tableau = result
            full_sequence = EliminationSequence([*prefix_sequence.operations, op, *sequence.operations])
            score = self.score_fn(full_sequence)

            if self.solution_tracker:
                self.solution_tracker.record_solution(full_sequence, final_tableau, score)

            if not _should_prune_candidate(score, best_score):
                scored_candidates.append((op, score))
                best_score = score if best_score is None or score < best_score else best_score

        scored_candidates.sort(key=operator.itemgetter(1))
        return scored_candidates

    def _create_base_strategy(self, filters: list | None) -> EliminationStrategy:
        """Create base strategy with fresh generator and filters.

        Args:
            filters: Filters to use for the strategy.

        Returns:
            A new base strategy with fresh state.
        """
        fresh_generator = _create_fresh_base_generator(self.base_strategy.candidate_generator)
        return EliminationStrategy(
            termination_criterion=self.base_strategy.termination_criterion,
            candidate_generator=fresh_generator,
            filters=filters,
        )

    def _create_recursive_strategy(self, remaining_depth: int, filters: list | None) -> EliminationStrategy:
        """Create strategy with recursive lookahead generator.

        Args:
            remaining_depth: Remaining depth for recursive lookahead.
            filters: Filters to use for the strategy.

        Returns:
            A new strategy with lookahead generator.
        """
        fresh_base_strategy = self._create_base_strategy(None)

        lookahead_generator = LookaheadCandidateGenerator(
            base_strategy=fresh_base_strategy,
            lookahead=remaining_depth,
            num_lookahead_candidates=self._get_remaining_candidate_limits(remaining_depth),
            score_fn=self.score_fn,
            track_best_solution=self.track_solutions,
            enable_early_termination=False,
        )

        return EliminationStrategy(
            termination_criterion=self.base_strategy.termination_criterion,
            candidate_generator=lookahead_generator,
            filters=filters,
        )

    def _get_remaining_candidate_limits(self, remaining_depth: int) -> list[int]:
        """Get candidate limits for remaining depth.

        Args:
            remaining_depth: Remaining lookahead depth.

        Returns:
            List of candidate limits for each remaining layer.
        """
        return [10] * remaining_depth


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

        self.simulator = LayerSimulator(base_strategy, score_fn, track_best_solution)
        self.current_sequence = EliminationSequence([])
        self.should_terminate = False

    def get_candidates(self, tableau: BinaryMatrix) -> list[tuple[TableauOperation, int]]:
        """Generate candidates using lookahead simulation.

        Args:
            tableau: The current binary matrix or tableau.

        Returns:
            List of operations sorted by lookahead score.
        """
        if self.lookahead <= 0:
            return self.base_strategy.candidate_generator.get_candidates(tableau)

        base_candidates = [cand for cand, _ in self.base_strategy.candidate_generator.get_candidates(tableau)]
        num_candidates_first_layer = self.num_lookahead_candidates_per_layer[0]

        initial_filters = _create_fresh_filters(self.base_strategy.filters)

        scored_candidates = self.simulator.simulate_layer(
            tableau=tableau,
            candidates=base_candidates,
            num_candidates=num_candidates_first_layer,
            prefix_sequence=self.current_sequence,
            remaining_depth=self.lookahead - 1,
            initial_filters=initial_filters,
        )

        if not scored_candidates and self.track_best_solution and self.enable_early_termination:
            if self.simulator.solution_tracker and self.simulator.solution_tracker.get_solution() is not None:
                self.should_terminate = True
                return []

        if not scored_candidates:
            scored_candidates = self.simulator.simulate_layer(
                tableau=tableau,
                candidates=base_candidates,
                num_candidates=num_candidates_first_layer,
                prefix_sequence=self.current_sequence,
                remaining_depth=self.lookahead - 1,
                initial_filters=initial_filters,
            )

        if self.track_best_solution and scored_candidates and self.simulator.solution_tracker:
            best_candidate_score = scored_candidates[0][1]
            self.simulator.solution_tracker.update_best_score(best_candidate_score)

        return [(op, score) for op, score in scored_candidates]

    def should_terminate_early(self) -> bool:
        """Check if elimination should terminate early.

        Returns:
            True if early termination is requested, False otherwise.
        """
        return self.should_terminate

    def get_best_solution(self) -> tuple[EliminationSequence, BinaryMatrix] | None:
        """Get the best complete solution found during lookahead exploration.

        Returns:
            Tuple of (sequence, tableau) if a solution is available, None otherwise.
        """
        if self.simulator.solution_tracker:
            return self.simulator.solution_tracker.get_solution()
        return None

    def update(self, op: TableauOperation, tableau: BinaryMatrix) -> None:
        """Update internal state by delegating to the base generator.

        Args:
            op: The operation that was applied.
            tableau: The resulting tableau after applying the operation.
        """
        self.current_sequence.add_operation(op)
        self.base_strategy.candidate_generator.update(op, tableau)

    def reset(self) -> None:
        """Reset internal state by delegating to the base generator."""
        self.current_sequence = EliminationSequence([])
        self.simulator.solution_tracker = SolutionTracker() if self.track_best_solution else None
        self.should_terminate = False
        self.base_strategy.candidate_generator.reset()
