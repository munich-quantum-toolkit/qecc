# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Rollout-based candidate generation for circuit synthesis."""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .cnot import GreedyCNOTGenerator
from .elimination import CandidateGenerator, EliminationSequence, EliminationStrategy, ParallelFilter, eliminate

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

    from .elimination import OperationFilter
    from .operations import TableauOperation
    from .types import BinaryMatrix

logger = logging.getLogger(__name__)


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
    fresh_filters = None
    if strategy.filters:
        fresh_filters = [f.copy() for f in strategy.filters]

    candidate_generator_cls = type(strategy.candidate_generator)
    if isinstance(strategy.candidate_generator, GreedyCNOTGenerator):
        fresh_candidate_generator: CandidateGenerator = GreedyCNOTGenerator(
            strategy.candidate_generator.n_stabs, fresh_filters
        )
    else:
        fresh_candidate_generator = candidate_generator_cls(fresh_filters)

    return EliminationStrategy(
        termination_criterion=strategy.termination_criterion,
        candidate_generator=fresh_candidate_generator,
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
        cache_policy: CachePolicy | None = None,
    ) -> None:
        """Initialize the rollout candidate generator.

        Args:
            base_strategy: Base strategy for greedy candidate generation
            rollout: Number of steps to look ahead
            num_rollout_candidates: Number of candidates to explore per layer
            score_fn: Function to score complete elimination sequences
            enable_early_termination: If True, allows early termination when no improving candidates found
            current_sequence: The elimination sequence built so far (used for depth calculation)
            cache_policy: Policy for caching rollout results
        """
        self.base_strategy = base_strategy
        self.rollout = rollout
        self.num_rollout_candidates_per_layer = _normalize_rollout_candidates(num_rollout_candidates, rollout)
        self.score_fn = score_fn
        self.enable_early_termination = enable_early_termination
        self._evaluation_prefix = current_sequence.copy() if current_sequence is not None else EliminationSequence([])
        self._local_prefix = EliminationSequence([])
        self._best_known_score: tuple[int, ...] | None = None
        self._best_known_sequence: EliminationSequence | None = None
        self._best_known_tableau: BinaryMatrix | None = None
        self._should_terminate = False
        self.cache_policy = cache_policy

    @dataclass
    class ScoredCandidate:
        """Data class to hold scored candidate information."""

        op: TableauOperation
        score: tuple[int, ...]
        completed_sequence: EliminationSequence
        local_sequence: EliminationSequence

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

    def _update_best_scored_candidates(self, scored_candidates: list[ScoredCandidate]) -> bool:
        """Update the best known complete solution based on scored candidates.

        Args:
            scored_candidates: List of (operation, score) tuples from the current rollout layer.

        Returns:
            True if an improvement was found, False otherwise.
        """
        improvement_found = False
        for cand in scored_candidates:
            score = cand.score
            if self._best_known_score is None or score < self._best_known_score:
                self._best_known_score = score
                self._best_known_sequence = cand.local_sequence.copy()
                improvement_found = True
        return improvement_found

    def score_rollout_candidates(
        self, tableau: BinaryMatrix, base_candidates: Sequence[TableauOperation]
    ) -> list[ScoredCandidate]:
        """Score base candidates using rollout simulation.

        Args:
            tableau: The current binary matrix.
            base_candidates: List of candidate operations from the base strategy.

        Returns:
            List of (operation, score) tuples sorted by score.
        """
        scored: list[RolloutCandidateGenerator.ScoredCandidate] = []
        for op in base_candidates:
            new_tableau = op.apply(tableau)

            prefix = EliminationSequence([*self._evaluation_prefix.operations, op])
            cache_key = self.cache_policy.key(new_tableau, self.rollout - 1, prefix) if self.cache_policy else None
            cached = cache.get(cache_key) if self.cache_policy else None

            if cached is not None:
                seq = cached.copy()
            else:
                lower_level_strategy = self._create_rollout_strategy(op)
                seq, _final_tableau = eliminate(new_tableau, lower_level_strategy)

                cache.set(cache_key, seq.copy())
                for i, op_sequence in enumerate(seq.operations):
                    new_tableau = op_sequence.apply(new_tableau)
                    child_prefix = EliminationSequence([*prefix.operations, *seq.operations[: i + 1]])
                    key = (
                        self.cache_policy.key(new_tableau, self.rollout - 1, child_prefix)
                        if self.cache_policy
                        else None
                    )
                    child_suffix = EliminationSequence(seq.operations[i + 1 :])
                    cache.set(key, EliminationSequence([*child_suffix.operations]))

            completed = EliminationSequence([*prefix.operations, *seq.operations])
            local_seq = EliminationSequence([*self._local_prefix.operations, op, *seq.operations])

            score = self.score_fn(completed)
            scored.append(self.ScoredCandidate(op, score, completed, local_seq))

        scored.sort(key=operator.attrgetter("score"))
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

        if not is_improvement and self.enable_early_termination and self._best_known_sequence is not None:
            self._should_terminate = True
            return []

        return [(cand.op, cand.score) for cand in scored_candidates]

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

    def _create_rollout_strategy(self, op: TableauOperation) -> EliminationStrategy:
        """Create a fresh rollout strategy for recursive simulation.

        Args:
        op: The operation to be applied at the current layer, used to build the child evaluation prefix.

        Returns:
            A new EliminationStrategy with fresh generator and filters.
        """
        child_eval_prefix = EliminationSequence([*self._evaluation_prefix.operations, op])
        child_base_strategy = _create_fresh_rollout_strategy(self.base_strategy)

        if self.rollout - 1 == 0:
            child_base_strategy.filters = self._initialize_filter_state_from_prefix(
                child_eval_prefix,
                child_base_strategy.filters,
            )
        # also make sure the candidate generator sees those filters
        if hasattr(child_base_strategy.candidate_generator, "filters"):
            child_base_strategy.candidate_generator.filters = list(child_base_strategy.filters or [])

        return EliminationStrategy(
            termination_criterion=self.base_strategy.termination_criterion,
            candidate_generator=RolloutCandidateGenerator(
                child_base_strategy,
                self.rollout - 1,
                self.num_rollout_candidates_per_layer[1:] if len(self.num_rollout_candidates_per_layer) > 1 else [],
                self.score_fn,
                enable_early_termination=self.enable_early_termination,
                current_sequence=child_eval_prefix,
                cache_policy=self.cache_policy,
            ),
        )

    def update(self, op: TableauOperation, tableau: BinaryMatrix) -> None:
        """Update internal state based on the operation applied.

        Args:
            op: The operation that was applied to the tableau.
            tableau: The resulting tableau after applying the operation.
        """
        self._evaluation_prefix.add_operation(op)
        self._local_prefix.add_operation(op)
        self.base_strategy.candidate_generator.update(op, tableau)

    def reset(self) -> None:
        """Reset internal state by delegating to the base generator."""
        self._evaluation_prefix = EliminationSequence([])
        self._local_prefix = EliminationSequence([])
        self._best_known_score = None
        self._best_known_sequence = None
        self._best_known_tableau = None
        self._should_terminate = False
        self.base_strategy.candidate_generator.reset()

    @staticmethod
    def _initialize_filter_state_from_prefix(
        prefix: EliminationSequence,
        filters: Sequence[OperationFilter] | None,
    ) -> list[OperationFilter] | None:
        """Initialize filter state based on the operations in the prefix sequence.

        Args:
            prefix: The elimination sequence prefix to initialize from.
            filters: The list of filters to initialize, or None if no filters are used.
        """
        if not filters:
            return None

        initialized = [f.copy() for f in filters]
        last_layer_qubits = prefix.last_layer_qubits()

        for f in initialized:
            f.reset()
            if isinstance(f, ParallelFilter):
                f.block_qubits(last_layer_qubits)

        return initialized

    def escape_local_minimum(self, tableau: BinaryMatrix) -> Sequence[TableauOperation] | None:
        """Escape local minimum by applying a random operation.

        Args:
            tableau: The current binary matrix.

        Returns:
            A sequence of operations to escape the local minimum, or an empty sequence if no escape is possible.
        """
        return self.base_strategy.candidate_generator.escape_local_minimum(tableau)


class CachePolicy(Protocol):
    """Protocol for defining cache policies for rollout candidate generation."""

    def key(
        self,
        tableau: BinaryMatrix,
        rollout: int,
        current_sequence: EliminationSequence,
    ) -> Hashable | None:
        """Generate a cache key based on the tableau, rollout depth, and current elimination sequence.

        Args:
            tableau: The current binary matrix.
            rollout: The remaining rollout depth.
            current_sequence: The current elimination sequence.

        Returns:
            A hashable key for caching rollout results, or None if caching is disabled.
        """


@dataclass(frozen=True)
class AdditiveCachePolicy:
    """Cache policy that ignores the current elimination sequence for key generation, i.e. the cached value is independent of the synthesis context."""

    @staticmethod
    def key(tableau: BinaryMatrix, rollout: int, current_sequence: EliminationSequence) -> Hashable:  # noqa: ARG004
        """Generate a cache key based on the tableau and rollout depth, ignoring the current sequence.

        tableau: The current binary matrix.
        rollout: The remaining rollout depth.
        current_sequence: The current elimination sequence (ignored for this policy).

        Returns:
            A hashable key for caching rollout results.
        """
        return (hash(tableau), rollout)


@dataclass(frozen=True)
class NonAdditiveCachePolicy:
    """Cache policy that includes the current elimination sequence in the key, making cached values dependent on the synthesis context."""

    @staticmethod
    def key(tableau: BinaryMatrix, rollout: int, current_sequence: EliminationSequence) -> Hashable:
        """Generate a cache key based on the tableau, rollout depth, and current elimination sequence.

        tableau: The current binary matrix.
        rollout: The remaining rollout depth.
        current_sequence: The current elimination sequence.

        Returns:
            A hashable key for caching rollout results that includes the current sequence.
        """
        return (hash(tableau), rollout, frozenset(current_sequence.last_layer_qubits()))


class RolloutCache:
    """Simple cache for storing rollout results based on a hashable key."""

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._cache: dict[Hashable, EliminationSequence] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable | None) -> EliminationSequence | None:
        """Retrieve a cached result for the given key.

        Args:
            key: The cache key to look up.

        Returns:
            The cached (score, sequence) tuple if found, or None if not present.
        """
        if key is None:
            return None
        value = self._cache.get(key, None)
        if value is not None:
            self.hits += 1
            return value
        self.misses += 1
        return None

    def set(self, key: Hashable | None, value: EliminationSequence) -> None:
        """Store a result in the cache under the given key."""
        if key is None:
            return
        self._cache[key] = value.copy()

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()

    def hit_rate(self) -> float:
        """Calculate the cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


cache = RolloutCache()


def clear_cache() -> None:
    """Clear the global rollout cache."""
    logger.info(
        "Clearing rollout cache: %d hits, %d misses, hit rate %.2f%%", cache.hits, cache.misses, cache.hit_rate() * 100
    )
    cache.clear()
