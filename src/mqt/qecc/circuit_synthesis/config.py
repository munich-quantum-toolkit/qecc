# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Configuration classes for elimination methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from .elimination import CandidateGenerator, EliminationSequence, OperationFilter, SelectionStrategy
    from .operations import TableauOperation
    from .types import BinaryMatrix


@dataclass
class EliminationStrategy:
    """Configuration for elimination methods."""

    termination_criterion: Callable[[BinaryMatrix], bool]
    candidate_generator: CandidateGenerator
    selection_strategy: SelectionStrategy | None = None
    filters: list[OperationFilter] | None = None
    callback: Callable[[int, TableauOperation, BinaryMatrix], None] | None = None
    post_process_fn: Callable[[EliminationSequence, BinaryMatrix], tuple[EliminationSequence, BinaryMatrix]] = (
        lambda ops, tbl: (ops, tbl)
    )

    @classmethod
    def for_cnot_up_to_row_ops(
        cls,
        target_rank: int,
        optimization_criterion: str = "gates",
        callback: Callable[[int, TableauOperation, BinaryMatrix], None] | None = None,
    ) -> EliminationStrategy:
        """Create configuration for CSS code elimination.

        Args:
            target_rank: The target rank of the check matrix after elimination.
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

        from .cnot import GreedyCNOTGenerator
        from .elimination import ParallelFilter

        filters = [ParallelFilter()]

        def termination_criterion(tbl: BinaryMatrix) -> bool:
            from ..codes.pauli import CheckMatrix

            if not isinstance(tbl, CheckMatrix):
                msg = "CSS elimination can only be applied to CheckMatrix instances."
                raise TypeError(msg)

            matrix = tbl.matrix
            non_zero_columns = np.sum(np.any(matrix != 0, axis=0))
            return non_zero_columns == target_rank

        return cls(
            termination_criterion=termination_criterion,
            candidate_generator=GreedyCNOTGenerator(filters),
            filters=filters,
            callback=callback,
        )

    @classmethod
    def for_cnot_exact(
        cls,
        target_rank: int,
        optimization_criterion: str = "gates",
        callback: Callable[[int, TableauOperation, BinaryMatrix], None] | None = None,
    ) -> EliminationStrategy:
        """Create configuration for CSS code elimination.

        Args:
            target_rank: The target rank of the check matrix after elimination.
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

        from .cnot import GreedyCNOTGenerator
        from .elimination import ParallelFilter

        filters = [ParallelFilter()] if optimization_criterion == "depth" else []

        def termination_criterion(tbl: BinaryMatrix) -> bool:
            from ..codes.pauli import CheckMatrix

            if not isinstance(tbl, CheckMatrix):
                msg = "CSS elimination can only be applied to CheckMatrix instances."
                raise TypeError(msg)

            matrix = tbl.matrix
            one_columns = (np.sum(matrix, axis=0) == 1).sum()
            return one_columns == target_rank

        return cls(
            termination_criterion=termination_criterion,
            candidate_generator=GreedyCNOTGenerator(filters),
            filters=filters,
            callback=callback,
        )

    @classmethod
    def for_non_css(
        cls,
        optimization_criterion: str = "gates",
        callback: Callable[[int, TableauOperation, BinaryMatrix], None] | None = None,
    ) -> EliminationStrategy:
        """Create configuration for non-CSS stabilizer code elimination.

        Args:
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

        from .elimination import ParallelFilter
        from .transvection import (
            GreedyTransvectionGenerator,
            is_terminal_transvection,
            reduce_single_qubit_gates_and_swaps,
        )

        filters = [ParallelFilter()] if optimization_criterion == "depth" else []

        return cls(
            termination_criterion=is_terminal_transvection,
            candidate_generator=GreedyTransvectionGenerator(filters),
            filters=filters,
            callback=callback,
            post_process_fn=reduce_single_qubit_gates_and_swaps,
        )

    @classmethod
    def for_non_css_stateprep(
        cls,
        optimization_criterion: str = "gates",
        callback: Callable[[int, TableauOperation, BinaryMatrix], None] | None = None,
    ) -> EliminationStrategy:
        """Create configuration for non-CSS stabilizer code elimination.

        Args:
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

        from .elimination import ParallelFilter
        from .transvection import (
            GreedyTransvectionGeneratorStateprep,
            is_terminal_stateprep,
            reduce_single_qubit_gates_stateprep,
        )

        filters = [ParallelFilter()] if optimization_criterion == "depth" else []

        return cls(
            termination_criterion=is_terminal_stateprep,
            candidate_generator=GreedyTransvectionGeneratorStateprep(filters),
            filters=filters,
            callback=callback,
            post_process_fn=reduce_single_qubit_gates_stateprep,
        )

    @classmethod
    def for_non_css_with_lookahead(
        cls,
        optimization_criterion: str = "gates",
        lookahead: int = 1,
        num_lookahead_candidates: int | list[int] = 10,
        enable_early_termination: bool = True,
        callback: Callable[[int, TableauOperation, BinaryMatrix], None] | None = None,
    ) -> EliminationStrategy:
        """Create configuration for non-CSS elimination with lookahead.

        Args:
            optimization_criterion: Either "gates" (minimize gate count) or "depth" (minimize circuit depth).
            lookahead: Number of steps to look ahead when selecting operations.
            num_lookahead_candidates: Number of top candidates to explore at each lookahead layer.
                Can be a single int (same limit for all layers) or a list of ints (one per layer).
            enable_early_termination: If True, allows early termination when no improving candidates found.
            callback: Optional callback function invoked after each elimination step.

        Returns:
            EliminationStrategy configured for non-CSS code elimination with lookahead and post-processing.

        Raises:
            ValueError: If optimization_criterion is not "gates" or "depth".
        """
        if optimization_criterion not in {"gates", "depth"}:
            msg = f"Unsupported optimization criterion: {optimization_criterion}"
            raise ValueError(msg)

        from .elimination import ParallelFilter
        from .lookahead import LookaheadCandidateGenerator
        from .transvection import (
            GreedyTransvectionGenerator,
            is_terminal_transvection,
            reduce_single_qubit_gates_and_swaps,
        )

        filters = [ParallelFilter()] if optimization_criterion == "depth" else []

        base_strategy = EliminationStrategy(
            termination_criterion=is_terminal_transvection,
            candidate_generator=GreedyTransvectionGenerator(filters),
            filters=filters,
        )

        def score_fn(ops: EliminationSequence) -> tuple[int, bool]:
            n_transvections = ops.num_transvections()
            return n_transvections, n_transvections <= 1

        return cls(
            termination_criterion=is_terminal_transvection,
            candidate_generator=LookaheadCandidateGenerator(
                base_strategy,
                lookahead,
                num_lookahead_candidates,
                score_fn,
                enable_early_termination=enable_early_termination,
            ),
            filters=filters,
            callback=callback,
            post_process_fn=reduce_single_qubit_gates_and_swaps,
        )

    @classmethod
    def for_cnot_with_lookahead_up_to_row_ops(
        cls,
        target_rank: int,
        lookahead: int = 1,
        num_lookahead_candidates: int | list[int] = 10,
        optimization_criterion: str = "gates",
        enable_early_termination: bool = True,
        callback: Callable[[int, TableauOperation, BinaryMatrix], None] | None = None,
    ) -> EliminationStrategy:
        r"""Create configuration for CSS elimination with lookahead.

        Args:
            target_rank: The target rank of the check matrix after elimination.
            lookahead: Number of steps to look ahead when selecting operations.
            num_lookahead_candidates: Number of top candidates to explore at each lookahead layer.
                Can be a single int (same limit for all layers) or a list of ints (one per layer).
            optimization_criterion: Either "gates" or "depth" for optimization objective.
            enable_early_termination: If True, allows early termination when no improving candidates found.
            callback: Optional callback function invoked after each elimination step.

        Returns:
            EliminationStrategy configured for CSS code elimination with lookahead.

        Raises:
            ValueError: If optimization_criterion is not "gates" or "depth".
        """
        from .lookahead import LookaheadCandidateGenerator

        def termination_criterion(tbl: BinaryMatrix) -> bool:
            from ..codes.pauli import CheckMatrix

            if not isinstance(tbl, CheckMatrix):
                msg = "CSS elimination can only be applied to CheckMatrix instances."
                raise TypeError(msg)

            matrix = tbl.matrix
            non_zero_columns = np.sum(np.any(matrix != 0, axis=0))
            return non_zero_columns == target_rank

        base_strategy = EliminationStrategy.for_cnot_up_to_row_ops(
            target_rank=target_rank,
            optimization_criterion=optimization_criterion,
            callback=None,
        )

        if optimization_criterion == "gates":

            def _score_fn(ops: EliminationSequence) -> tuple[int, int, bool]:
                n_cnots = ops.num_cnots()
                return (n_cnots, ops.depth(), n_cnots <= 1)

        else:

            def _score_fn(ops: EliminationSequence) -> tuple[int, int, bool]:
                depth = ops.depth()
                return (depth, ops.num_cnots(), depth <= 1)

        return EliminationStrategy(
            termination_criterion=base_strategy.termination_criterion,
            candidate_generator=LookaheadCandidateGenerator(
                base_strategy,
                lookahead,
                num_lookahead_candidates,
                _score_fn,
                enable_early_termination=enable_early_termination,
            ),
            filters=None,
            callback=callback,
        )

    @classmethod
    def for_cnot_with_lookahead_exact(
        cls,
        target_rank: int,
        lookahead: int = 1,
        num_lookahead_candidates: int | list[int] = 10,
        optimization_criterion: str = "gates",
        enable_early_termination: bool = True,
        callback: Callable[[int, TableauOperation, BinaryMatrix], None] | None = None,
    ) -> EliminationStrategy:
        r"""Create configuration for CSS elimination with lookahead.

        Args:
            target_rank: The target rank of the check matrix after elimination.
            lookahead: Number of steps to look ahead when selecting operations.
            num_lookahead_candidates: Number of top candidates to explore at each lookahead layer.
                Can be a single int (same limit for all layers) or a list of ints (one per layer).
            optimization_criterion: Either "gates" or "depth" for optimization objective.
            enable_early_termination: If True, allows early termination when no improving candidates found.
            callback: Optional callback function invoked after each elimination step.

        Returns:
            EliminationStrategy configured for CSS code elimination with lookahead.

        Raises:
            ValueError: If optimization_criterion is not "gates" or "depth".
        """
        from .lookahead import LookaheadCandidateGenerator

        def termination_criterion(tbl: BinaryMatrix) -> bool:
            from ..codes.pauli import CheckMatrix

            if not isinstance(tbl, CheckMatrix):
                msg = "CSS elimination can only be applied to CheckMatrix instances."
                raise TypeError(msg)

            matrix = tbl.matrix
            one_columns = (np.sum(matrix, axis=0) == 1).sum()
            return one_columns == target_rank

        base_strategy = EliminationStrategy.for_cnot_exact(
            target_rank=target_rank,
            optimization_criterion=optimization_criterion,
            callback=None,
        )

        if optimization_criterion == "gates":

            def _score_fn(ops: EliminationSequence) -> tuple[int, int, bool]:
                n_cnots = ops.num_cnots()
                return (n_cnots, ops.depth(), n_cnots <= 1)

        else:

            def _score_fn(ops: EliminationSequence) -> tuple[int, int, bool]:
                depth = ops.depth()
                return (depth, ops.num_cnots(), depth <= 1)

        return EliminationStrategy(
            termination_criterion=base_strategy.termination_criterion,
            candidate_generator=LookaheadCandidateGenerator(
                base_strategy,
                lookahead,
                num_lookahead_candidates,
                _score_fn,
                enable_early_termination=enable_early_termination,
            ),
            filters=None,
            callback=callback,
        )

    @classmethod
    def for_cnot_with_lookahead(
        cls,
        optimization_criterion: str = "gates",
        lookahead: int = 1,
        num_lookahead_candidates: int | list[int] = 10,
    ) -> EliminationStrategy:
        """Create configuration for CSS elimination with lookahead.

        Args:
            optimization_criterion: Either "gates" or "depth" for optimization objective.
            lookahead: Number of steps to look ahead when selecting operations.
            num_lookahead_candidates: Number of top candidates to explore at each lookahead layer.

        Returns:
            EliminationStrategy configured for CSS code elimination with lookahead.

        Raises:
            ValueError: If optimization_criterion is not "gates" or "depth".
        """
        msg = "for_cnot_with_lookahead is deprecated. Use for_cnot_with_lookahead_up_to_row_ops or for_cnot_with_lookahead_exact instead."
        raise NotImplementedError(msg)
