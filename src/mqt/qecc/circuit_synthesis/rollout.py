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
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .types import BinaryMatrix


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
        *,
        enable_early_termination: bool = True,
        current_sequence: EliminationSequence | None = None,
        cache_policy: CachePolicy | None = None,
        num_cached_subsequences: int = 10,
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
            num_cached_subsequences: Number of subsequences of rollout continuations to cache for reuse
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
        self.num_cached_subsequences = num_cached_subsequences

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
                for i, op_sequence in enumerate(seq.operations[: self.num_cached_subsequences]):
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
            The best `EliminationSequence` found during rollout, or `None` if no
            improving candidate was scored.
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
        if hasattr(child_base_strategy.candidate_generator, "filters") and child_base_strategy.filters is not None:
            child_base_strategy.filters = list(child_base_strategy.filters)

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
                num_cached_subsequences=self.num_cached_subsequences,
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
    """Protocol for rollout cache key generation.

    Implementations define which parts of the current synthesis state are
    relevant for cache reuse. For additive objectives, the key may depend only
    on the tableau and remaining rollout depth. For non-additive objectives,
    the key may also depend on a compact summary of the current synthesis
    context.
    """

    def key(
        self,
        tableau: BinaryMatrix,
        rollout: int,
        current_sequence: EliminationSequence,
    ) -> Hashable | None:
        """Return a cache key for the given rollout subproblem.

        Args:
            tableau: The current tableau or binary matrix describing the
                remaining elimination problem.
            rollout: The remaining rollout depth for the subproblem.
            current_sequence: The elimination sequence built so far. Depending
                on the cache policy, this may be ignored or used to derive a
                compact synthesis context.

        Returns:
            A hashable cache key, or ``None`` if caching should be disabled for
            this subproblem.
        """


@dataclass(frozen=True)
class AdditiveCachePolicy:
    """Cache policy for additive objectives.

    This policy assumes that the best continuation depends only on the current
    tableau and the remaining rollout depth, and not on the synthesis history.
    This is appropriate for additive or prefix-independent objectives.
    """

    @staticmethod
    def key(
        tableau: BinaryMatrix,
        rollout: int,
        current_sequence: EliminationSequence,  # ruff:ignore[unused-static-method-argument]
    ) -> Hashable:
        """Return a cache key independent of the current sequence.

        Args:
            tableau: The current tableau or binary matrix.
            rollout: The remaining rollout depth.
            current_sequence: The elimination sequence built so far. It is
                ignored by this policy.

        Returns:
            A hashable key based only on the tableau and remaining rollout
            depth.
        """
        return (hash(tableau), rollout)


@dataclass(frozen=True)
class NonAdditiveCachePolicy:
    """Cache policy for depth-like non-additive objectives.

    This policy includes a compact summary of the current synthesis context in
    the cache key. Here, the context is represented by the set of qubits used
    in the current last layer, which is sufficient for the current depth-guided
    tail policy.
    """

    @staticmethod
    def key(
        tableau: BinaryMatrix,
        rollout: int,
        current_sequence: EliminationSequence,
    ) -> Hashable:
        """Return a cache key that includes the last-layer qubit context.

        Args:
            tableau: The current tableau or binary matrix.
            rollout: The remaining rollout depth.
            current_sequence: The elimination sequence built so far. The cache
                key uses only the qubits occupied in its current last layer.

        Returns:
            A hashable key based on the tableau, rollout depth, and the current
            last-layer qubit occupancy.
        """
        return (hash(tableau), rollout, frozenset(current_sequence.last_layer_qubits()))


class RolloutCache:
    """Weighted least-recently-used cache for rollout continuations.

    Cache entries are evicted according to an LRU policy once the configured
    maximum total weight is exceeded. The weight of an entry is defined as the
    number of operations stored in its cached continuation.

    This design is useful for rollout synthesis because cached continuations can
    vary significantly in length. A weighted cache controls memory usage more
    effectively than a cache limited only by the number of entries.
    """

    def __init__(self, max_weight: int = 200_000) -> None:
        """Initialize an empty weighted LRU cache.

        Args:
            max_weight: Maximum allowed total cache weight. The weight of one
                cached entry is the number of operations in its stored
                continuation.
        """
        self._cache: OrderedDict[Hashable, EliminationSequence] = OrderedDict()
        self._weights: dict[Hashable, int] = {}
        self._lock = RLock()
        self.max_weight = max_weight
        self.current_weight = 0
        self.hits = 0
        self.misses = 0

    def configure(self, max_weight: int) -> None:
        """Configure the cache parameters.

        Args:
            max_weight: Maximum allowed total cache weight. The weight of one
                cached entry is the number of operations in its stored
                continuation.
        """
        with self._lock:
            self.max_weight = max_weight

    @staticmethod
    def _weight(value: EliminationSequence) -> int:
        """Return the cache weight of a stored continuation.

        Args:
            value: The continuation sequence to be stored.

        Returns:
            The weight of the cache entry, measured as the number of operations
            in the sequence.
        """
        return len(value.operations)

    def get(self, key: Hashable | None) -> EliminationSequence | None:
        """Retrieve a cached continuation for the given key.

        Accessing an entry marks it as recently used.

        Args:
            key: The cache key to look up. If ``None``, caching is treated as
                disabled for this lookup.

        Returns:
            A copy of the cached continuation if present, otherwise ``None``.
        """
        if key is None:
            return None

        with self._lock:
            value = self._cache.get(key)
            if value is None:
                self.misses += 1
                return None

            self.hits += 1
            self._cache.move_to_end(key)
            return value.copy()

    def set(self, key: Hashable | None, value: EliminationSequence) -> None:
        """Store a continuation under the given key.

        If inserting the new value exceeds the configured cache budget, the
        least recently used entries are evicted until the cache is within the
        allowed weight bound again.

        Args:
            key: The cache key under which the continuation should be stored.
                If ``None``, the value is not cached.
            value: The continuation sequence to store.
        """
        if key is None:
            return

        with self._lock:
            stored = value.copy()
            weight = self._weight(stored)

            if key in self._cache:
                self.current_weight -= self._weights[key]
                self._cache.move_to_end(key)

            self._cache[key] = stored
            self._weights[key] = weight
            self.current_weight += weight

            while self.current_weight > self.max_weight and self._cache:
                old_key, _old_value = self._cache.popitem(last=False)
                self.current_weight -= self._weights.pop(old_key, 0)

    def clear(self) -> None:
        """Remove all cache entries and reset cache statistics."""
        with self._lock:
            self._cache.clear()
            self._weights.clear()
            self.current_weight = 0
            self.hits = 0
            self.misses = 0

    def hit_rate(self) -> float:
        """Return the fraction of successful cache lookups.

        Returns:
            The cache hit rate in the interval ``[0.0, 1.0]``.
        """
        with self._lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0

    def size(self) -> int:
        """Return the number of currently stored cache entries.

        Returns:
            The number of entries currently held in the cache.
        """
        with self._lock:
            return len(self._cache)


cache = RolloutCache()


_cache_lock = RLock()
_cache_session_depth = 0


def open_rollout_cache_session(max_weight: int) -> None:
    """Open a rollout cache session.

    Nested calls are supported. The global rollout cache is shared across all
    nested rollout evaluations within one synthesis run.
    """
    global _cache_session_depth  # ruff:ignore[global-statement]

    with _cache_lock:
        if _cache_session_depth == 0:
            cache.configure(max_weight)

        _cache_session_depth += 1


def close_rollout_cache_session() -> None:
    """Close a rollout cache session.

    When the outermost session is closed, the global rollout cache is cleared.
    """
    global _cache_session_depth  # ruff:ignore[global-statement]

    should_clear = False
    hits = 0
    misses = 0
    hit_rate = 0.0
    with _cache_lock:
        _cache_session_depth -= 1

        if _cache_session_depth < 0:
            _cache_session_depth = 0
            msg = "Rollout cache session depth became negative."
            raise RuntimeError(msg)

        if _cache_session_depth == 0:
            hits = cache.hits
            misses = cache.misses
            hit_rate = cache.hit_rate()
            cache.clear()
            should_clear = True

    if should_clear:
        logger.info(
            "Clearing rollout cache: %d hits, %d misses, hit rate %.2f%%",
            hits,
            misses,
            hit_rate * 100,
        )
