# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods for performing Gaussian elimination on GUI2 and symplectic matrices."""

from __future__ import annotations

import operator
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import stim

from .operations import CNOT, Swap, Transvection
from .types import BinaryMatrix

if TYPE_CHECKING:
    from ..codes.pauli import StabilizerTableau
    from .operations import TableauOperation


class EliminationSequence:
    """Class representing a sequence of tableau operations."""

    def __init__(self, operations: list[TableauOperation]) -> None:
        """Initialize the elimination sequence.

        Args:
            operations: A list of tableau operations.
        """
        self.operations = operations

    def to_circuit(self) -> stim.Circuit:
        """Convert the elimination sequence to a Stim circuit.

        Returns:
            stim.Circuit: The Stim circuit representing the elimination sequence.
        """
        circuit = stim.Circuit()
        for op in self.operations:
            op.append_to_circuit(circuit)
        return circuit

    def to_circuit_inverse(self) -> stim.Circuit:
        """Convert the inverse of the elimination sequence to a Stim circuit.

        Returns:
            stim.Circuit: The Stim circuit representing the inverse elimination sequence.
        """
        return self.to_circuit().inverse()

    def num_two_qubit_gates(self) -> int:
        """Count the number of two-qubit gates in the elimination sequence.

        Returns:
            int: The number of two-qubit gates.
        """
        count = 0
        for op in self.operations:
            if isinstance(op, (Transvection, CNOT, Swap)):
                count += 1
        return count

    def num_transvections(self) -> int:
        """Count the number of transvections in the elimination sequence.

        Returns: int: The number of transvections.
        """
        count = 0
        for op in self.operations:
            if isinstance(op, Transvection):
                count += 1
        return count

    def num_cnots(self) -> int:
        """Count the number of CNOT gates in the elimination sequence.

        Returns:
            int: The number of CNOT gates.
        """
        count = 0
        for op in self.operations:
            if isinstance(op, CNOT):
                count += 1
        return count

    def add_operation(self, op: TableauOperation) -> None:
        """Add a tableau operation to the elimination sequence.

        Args:
            op: The tableau operation to add.
        """
        self.operations.append(op)

    def apply(self, tableau: BinaryMatrix, inplace: bool = False) -> BinaryMatrix:
        """Apply the elimination sequence to a stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the sequence to.
            inplace (bool): If True, modifies the tableau in place. If False, returns a new tableau.

        Returns:
            BinaryMatrix: The resulting stabilizer tableau after applying the sequence.
        """
        out = tableau if inplace else tableau.copy()
        for op in self.operations:
            out = op.apply(out, inplace=True)
        return out

    def extend(self, other: EliminationSequence) -> None:
        """Extend the elimination sequence with another sequence.

        Args:
            other: The other elimination sequence to append.
        """
        self.operations.extend(other.operations)

    def __iter__(self) -> iter[TableauOperation]:
        """Return iterator over the tableau operations in the sequence."""
        return iter(self.operations)

    def __reversed__(self) -> iter[TableauOperation]:
        """Return reversed iterator over the tableau operations in the sequence."""
        return reversed(self.operations)

    def depth(self) -> int:
        """Estimate the circuit depth of the elimination sequence.

        Returns:
            int: The estimated circuit depth.
        """
        depth = 0
        qubit_last_used: dict[int, int] = {}
        for op in self.operations:
            involved_qubits = op.qubits()
            earliest_start = 0
            for q in involved_qubits:
                if q in qubit_last_used:
                    earliest_start = max(earliest_start, qubit_last_used[q] + 1)
            for q in involved_qubits:
                qubit_last_used[q] = earliest_start

        if qubit_last_used:
            depth = max(qubit_last_used.values()) + 1
        return depth


def eliminate(target_tableau: BinaryMatrix, config: EliminationConfig) -> tuple[EliminationSequence, BinaryMatrix]:
    """Perform Gaussian elimination on the given stabilizer tableau.

    This is the main elimination engine that iteratively reduces a binary matrix or
    stabilizer tableau by applying a sequence of operations (e.g., CNOTs, transvections,
    single-qubit Cliffords) until a termination criterion is met. The function serves as
    the workhorse for synthesizing quantum circuits from stabilizer codes and check matrices.

    Args:
        target_tableau: The input binary matrix or stabilizer tableau to reduce.
            Can be either a CheckMatrix (for CSS codes) or StabilizerTableau
            (for general stabilizer codes).
        config: Configuration object specifying:
            - termination_criterion: Function that returns True when elimination is complete
            - candidate_generator: Strategy for generating candidate operations from current tableau
            - selection_strategy: Strategy for selecting from candidate operations (optional)
            - filters: List of filters to constrain candidate operations (optional)
            - callback: Function called after each operation for monitoring (optional)
            - post_process_fn: Function to finalize the result (optional)

    Returns:
        A tuple containing:
        - EliminationSequence: The sequence of operations applied during elimination,
          which can be converted to a quantum circuit.
        - BinaryMatrix: The final reduced tableau after all operations are applied.

    Raises:
        RuntimeError: If no candidate operations are available but the termination
            criterion has not been met, indicating the elimination cannot proceed.

    Examples:
        >>> # CSS code elimination with greedy selection
        >>> config = EliminationConfig(
        ...     termination_criterion=lambda tbl: mod2.rank(tbl.matrix) == k,
        ...     candidate_generator=GreedyTransvectionGenerator(),
        ... )
        >>> operations, final_tableau = eliminate(check_matrix, config)

        >>> # Non-CSS code elimination with depth optimization
        >>> config = EliminationConfig(
        ...     termination_criterion=is_terminal_transvection,
        ...     candidate_generator=GreedyTransvectionGenerator(),
        ...     filters=[ParallelFilter()],
        ... )
        >>> operations, final_tableau = eliminate(stabilizer_tableau, config)

        >>> # Lookahead-based elimination
        >>> config = EliminationConfig(
        ...     termination_criterion=is_terminal_transvection,
        ...     candidate_generator=LookaheadCandidateGenerator(...),
        ...     post_process_fn=lambda ops, tbl: reduce_single_qubit_gates_and_swaps(tbl),
        ... )
        >>> operations, final_tableau = eliminate(tableau, config)

    See Also:
        - eliminate_css: High-level function for CSS code elimination
        - eliminate_non_css: High-level function for non-CSS code elimination
        - eliminate_non_css_with_lookahead: Lookahead-based non-CSS elimination
        - EliminationConfig: Configuration dataclass for elimination parameters
        - CandidateGenerator: Abstract base class for candidate generation strategies
        - SelectionStrategy: Abstract base class for operation selection strategies
    """
    tableau = target_tableau.copy()
    operations = EliminationSequence([])
    selection_strategy = config.selection_strategy or GreedySelection()
    iteration = 0

    while not config.termination_criterion(tableau):
        candidate_ops = config.candidate_generator.get_candidates(tableau)

        if _should_terminate_early(config.candidate_generator):
            return _get_early_termination_result(config.candidate_generator, config.post_process_fn)

        _validate_candidates(candidate_ops)

        op = selection_strategy.select(candidate_ops)
        tableau = op.apply(tableau, inplace=True)
        operations.add_operation(op)

        _update_elimination_state(op, tableau, config)
        _invoke_callback(iteration, op, tableau, config)
        iteration += 1

    result_ops, result_tableau = config.post_process_fn(operations, tableau)

    if hasattr(config.candidate_generator, "use_best_if_better"):
        return _maybe_use_best_solution(config.candidate_generator, result_ops, result_tableau, config.post_process_fn)

    return result_ops, result_tableau


def _maybe_use_best_solution(
    generator: CandidateGenerator,
    current_ops: EliminationSequence,
    current_tableau: BinaryMatrix,
    post_process_fn: Callable[[EliminationSequence, BinaryMatrix], tuple[EliminationSequence, BinaryMatrix]],
) -> tuple[EliminationSequence, BinaryMatrix]:
    """Compare current solution with best tracked solution and return the better one.

    Args:
        generator: The candidate generator that may have tracked a best solution.
        current_ops: The operation sequence from normal elimination.
        current_tableau: The tableau from normal elimination.
        post_process_fn: Function to post-process solutions.

    Returns:
        The better of the two solutions (current vs best tracked).
    """
    if not hasattr(generator, "get_best_solution") or not hasattr(generator, "score_fn"):
        return current_ops, current_tableau

    best_solution = generator.get_best_solution()
    if best_solution is None:
        return current_ops, current_tableau

    best_ops, best_tableau = best_solution
    current_score = generator.score_fn(current_ops)
    best_score = generator.score_fn(best_ops)

    if best_score < current_score:
        return best_ops, best_tableau

    return current_ops, current_tableau


def _should_terminate_early(generator: CandidateGenerator) -> bool:
    """Check if the generator wants to terminate early.

    Args:
        generator: The candidate generator.

    Returns:
        True if early termination is requested, False otherwise.
    """
    return hasattr(generator, "should_terminate_early") and generator.should_terminate_early()


def _get_early_termination_result(
    generator: CandidateGenerator,
    post_process_fn: Callable[[EliminationSequence, BinaryMatrix], tuple[EliminationSequence, BinaryMatrix]],
) -> tuple[EliminationSequence, BinaryMatrix]:
    """Get the result when terminating early.

    Args:
        generator: The candidate generator that requested early termination.
        post_process_fn: Function to post-process the result.

    Returns:
        The best solution found by the generator.
    """
    if not hasattr(generator, "get_best_solution"):
        msg = "Generator requested early termination but does not provide get_best_solution()"
        raise RuntimeError(msg)

    best_solution = generator.get_best_solution()
    if best_solution is None:
        msg = "Generator requested early termination but has no best solution"
        raise RuntimeError(msg)

    best_sequence, best_tableau = best_solution
    return post_process_fn(best_sequence, best_tableau)


class CandidateGenerator(ABC):
    """Abstract base class for generating candidate operations."""

    @abstractmethod
    def get_candidates(self, tableau: BinaryMatrix) -> list[tuple[TableauOperation, int]]:
        """Generate sorted candidate operations for the current tableau.

        Args:
            tableau: The current binary matrix/tableau

        Returns:
            A list of candidate operations, sorted by preference
        """

    @abstractmethod
    def update(self, op: TableauOperation, tableau: BinaryMatrix) -> None:
        """Update internal state after an operation is applied.

        Args:
            op: The operation that was just applied
            tableau: The resulting tableau after applying the operation
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (useful for lookahead simulations)."""

    def should_terminate_early(self) -> bool:
        """Check if elimination should terminate early and use the best solution found.

        Returns:
            True if elimination should terminate early, False otherwise.
        """
        return False

    def get_best_solution(self) -> tuple[EliminationSequence, BinaryMatrix] | None:
        """Get the best complete solution found during lookahead exploration.

        Returns:
            Tuple of (sequence, tableau) if a solution is available, None otherwise.
        """
        return None


class SelectionStrategy(ABC):
    """Abstract base class for selecting the best operation from candidates."""

    @abstractmethod
    def select(self, candidates: list[TableauOperation]) -> TableauOperation:
        """Select the best operation from candidates.

        Args:
            candidates: List of candidate operations, typically sorted by preference.

        Returns:
            The selected operation to apply.
        """


class GreedySelection(SelectionStrategy):
    """Always select the first candidate."""

    def select(self, candidates: list[tuple[TableauOperation, int]]) -> TableauOperation:
        """Select the first (best-scored) candidate.

        Args:
            candidates: List of candidate operations.

        Returns:
            The first candidate in the list.
        """
        return candidates[0][0]


class RandomSelection(SelectionStrategy):
    """Select a random candidate from top-k."""

    def __init__(self, k: int = 3) -> None:
        """Initialize the random selection strategy.

        Args:
            k: Number of top candidates to randomly choose from.
        """
        self.k = k

    def select(self, candidates: list[tuple[TableauOperation, int]]) -> TableauOperation:
        """Randomly select one of the top-k candidates.

        Args:
            candidates: List of candidate operations.

        Returns:
            A randomly chosen candidate from the first k candidates.
        """
        return random.choice(candidates[: self.k])[0]


class OperationFilter(ABC):
    """Abstract base class for filtering tableau operations."""

    @abstractmethod
    def should_include(self, op: TableauOperation) -> bool:
        """Check if an operation should be included in candidates.

        Args:
            op: The tableau operation to check.

        Returns:
            True if the operation should be included, False otherwise.
        """

    @abstractmethod
    def update(self, op: TableauOperation) -> None:
        """Update the filter state with the given operation.

        Args:
            op: The tableau operation to update the filter with.
        """

    @abstractmethod
    def copy(self) -> OperationFilter:
        """Create a copy of the filter with the same state.

        Returns:
            A new filter instance with copied state.
        """


class ParallelFilter(OperationFilter):
    """Filter that blocks operations on qubits already used in current layer."""

    def __init__(self, n_qubits: int | None = None) -> None:
        """Initialize the parallel filter.

        Args:
            n_qubits: Total number of qubits in the circuit. If None, will be inferred from operations.
        """
        self.blocked_qubits: set[int] = set()
        self.n_qubits = n_qubits

    def should_include(self, op: TableauOperation) -> bool:
        """Check if operation uses any blocked qubits.

        Args:
            op: The operation to check.

        Returns:
            True if no qubits are blocked, False otherwise.
        """
        return not any(qubit in self.blocked_qubits for qubit in op.qubits())

    def update(self, op: TableauOperation) -> None:
        """Block qubits involved in the operation.

        Args:
            op: The tableau operation to update the filter with.
        """
        qubits_involved = op.qubits()
        self.blocked_qubits.update(qubits_involved)

        if self.n_qubits is None:
            max_qubit = max(qubits_involved) if qubits_involved else 0
            self.n_qubits = max_qubit + 1

        if not self.has_available_qubits():
            self._reset()

    def has_available_qubits(self) -> bool:
        """Check if there are qubits available for operations."""
        if self.n_qubits is None:
            return True
        return len(self.blocked_qubits) < self.n_qubits

    def _reset(self) -> None:
        """Unblock all qubits."""
        self.blocked_qubits.clear()

    def copy(self) -> ParallelFilter:
        """Create a copy of the filter with the same state.

        Returns:
            A new ParallelFilter with copied blocked_qubits state.
        """
        new_filter = ParallelFilter(n_qubits=self.n_qubits)
        new_filter.blocked_qubits = self.blocked_qubits.copy()
        return new_filter


elimination_candidate_fn = Callable[[BinaryMatrix], EliminationSequence]


@dataclass
class EliminationConfig:
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
    ) -> EliminationConfig:
        """Create configuration for CSS code elimination.

        Args:
            target_rank: The target rank of the check matrix after elimination.
            optimization_criterion: Either "gates" (minimize gate count) or "depth" (minimize circuit depth).
            callback: Optional callback function invoked after each elimination step.

        Returns:
            EliminationConfig configured for CSS code elimination.

        Raises:
            ValueError: If optimization_criterion is not "gates" or "depth".
        """
        if optimization_criterion not in {"gates", "depth"}:
            msg = f"Unsupported optimization criterion: {optimization_criterion}"
            raise ValueError(msg)

        from .cnot import GreedyCNOTGenerator

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
    ) -> EliminationConfig:
        """Create configuration for CSS code elimination.

        Args:
            target_rank: The target rank of the check matrix after elimination.
            optimization_criterion: Either "gates" (minimize gate count) or "depth" (minimize circuit depth).
            callback: Optional callback function invoked after each elimination step.

        Returns:
            EliminationConfig configured for CSS code elimination.

        Raises:
            ValueError: If optimization_criterion is not "gates" or "depth".
        """
        if optimization_criterion not in {"gates", "depth"}:
            msg = f"Unsupported optimization criterion: {optimization_criterion}"
            raise ValueError(msg)

        from .cnot import GreedyCNOTGenerator

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
    ) -> EliminationConfig:
        """Create configuration for non-CSS stabilizer code elimination.

        Args:
            optimization_criterion: Either "gates" (minimize gate count) or "depth" (minimize circuit depth).
            callback: Optional callback function invoked after each elimination step.

        Returns:
            EliminationConfig configured for non-CSS code elimination with post-processing.
        -        Raises:
            ValueError: If optimization_criterion is not "gates" or "depth".
        """
        if optimization_criterion not in {"gates", "depth"}:
            msg = f"Unsupported optimization criterion: {optimization_criterion}"
            raise ValueError(msg)

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
    ) -> EliminationConfig:
        """Create configuration for non-CSS stabilizer code elimination.

        Args:
            optimization_criterion: Either "gates" (minimize gate count) or "depth" (minimize circuit depth).
            callback: Optional callback function invoked after each elimination step.

        Returns:
            EliminationConfig configured for non-CSS code elimination with post-processing.
        -        Raises:
            ValueError: If optimization_criterion is not "gates" or "depth".
        """
        if optimization_criterion not in {"gates", "depth"}:
            msg = f"Unsupported optimization criterion: {optimization_criterion}"
            raise ValueError(msg)

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
    ) -> EliminationConfig:
        """Create configuration for non-CSS elimination with lookahead.

        Args:
            optimization_criterion: Either "gates" (minimize gate count) or "depth" (minimize circuit depth).
            lookahead: Number of steps to look ahead when selecting operations.
            num_lookahead_candidates: Number of top candidates to explore at each lookahead layer.
                Can be a single int (same limit for all layers) or a list of ints (one per layer).
            enable_early_termination: If True, allows early termination when no improving candidates found.
            callback: Optional callback function invoked after each elimination step.

        Returns:
            EliminationConfig configured for non-CSS code elimination with lookahead and post-processing.

        Raises:
            ValueError: If optimization_criterion is not "gates" or "depth".
        """
        if optimization_criterion not in {"gates", "depth"}:
            msg = f"Unsupported optimization criterion: {optimization_criterion}"
            raise ValueError(msg)

        from .transvection import (
            GreedyTransvectionGenerator,
            is_terminal_transvection,
            reduce_single_qubit_gates_and_swaps,
        )

        filters = [ParallelFilter()] if optimization_criterion == "depth" else []

        base_config = EliminationConfig(
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
                base_config,
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
    ) -> EliminationConfig:
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
            EliminationConfig configured for CSS code elimination with lookahead.

        Raises:
            ValueError: If optimization_criterion is not "gates" or "depth".
        """

        def termination_criterion(tbl: BinaryMatrix) -> bool:
            from ..codes.pauli import CheckMatrix

            if not isinstance(tbl, CheckMatrix):
                msg = "CSS elimination can only be applied to CheckMatrix instances."
                raise TypeError(msg)

            matrix = tbl.matrix
            non_zero_columns = np.sum(np.any(matrix != 0, axis=0))
            return non_zero_columns == target_rank

        base_config = EliminationConfig.for_cnot_up_to_row_ops(
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

        return EliminationConfig(
            termination_criterion=base_config.termination_criterion,
            candidate_generator=LookaheadCandidateGenerator(
                base_config,
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
    ) -> EliminationConfig:
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
            EliminationConfig configured for CSS code elimination with lookahead.

        Raises:
            ValueError: If optimization_criterion is not "gates" or "depth".
        """

        def termination_criterion(tbl: BinaryMatrix) -> bool:
            from ..codes.pauli import CheckMatrix

            if not isinstance(tbl, CheckMatrix):
                msg = "CSS elimination can only be applied to CheckMatrix instances."
                raise TypeError(msg)

            matrix = tbl.matrix
            one_columns = (np.sum(matrix, axis=0) == 1).sum()
            return one_columns == target_rank

        base_config = EliminationConfig.for_cnot_exact(
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

        return EliminationConfig(
            termination_criterion=base_config.termination_criterion,
            candidate_generator=LookaheadCandidateGenerator(
                base_config,
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
    ) -> EliminationConfig:
        """Create configuration for CSS elimination with lookahead.

        Args:
            optimization_criterion: Either "gates" or "depth" for optimization objective.
            lookahead: Number of steps to look ahead when selecting operations.
            num_lookahead_candidates: Number of top candidates to explore at each lookahead layer.

        Returns:
            EliminationConfig configured for CSS code elimination with lookahead.

        Raises:
            ValueError: If optimization_criterion is not "gates" or "depth".
        """
        msg = "for_cnot_with_lookahead is deprecated. Use for_cnot_with_lookahead_up_to_row_ops or for_cnot_with_lookahead_exact instead."
        raise NotImplementedError(msg)


def _validate_candidates(candidates: list[TableauOperation]) -> None:
    """Ensure at least one candidate is available.

    Args:
        candidates: List of candidate operations.

    Raises:
        RuntimeError: If no candidates are available.
    """
    if not candidates:
        msg = "No more candidate operations available, but termination criterion not met."
        raise RuntimeError(msg)


def _update_elimination_state(op: TableauOperation, tableau: BinaryMatrix, config: EliminationConfig) -> None:
    """Update generator and filter state after applying an operation.

    Args:
        op: The operation that was applied.
        tableau: The resulting tableau after applying the operation.
        config: Elimination configuration containing generator and filters.
    """
    config.candidate_generator.update(op, tableau)


def _invoke_callback(iteration: int, op: TableauOperation, tableau: BinaryMatrix, config: EliminationConfig) -> None:
    """Invoke callback if configured.

    Args:
        iteration: Current iteration number.
        op: The operation that was just applied.
        tableau: The resulting tableau after applying the operation.
        config: Elimination configuration potentially containing a callback.
    """
    if config.callback:
        config.callback(iteration, op, tableau)


def is_identity(tableau: StabilizerTableau) -> bool:
    """Check if the given stabilizer tableau is the identity tableau.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to check.

    Returns:
        bool: True if the tableau is the identity tableau, False otherwise.
    """
    n = get_n(tableau)
    identity_matrix = np.eye(2 * n, dtype=np.int8)
    return np.array_equal(tableau.tableau.matrix, identity_matrix)


def get_n(tableau: BinaryMatrix) -> int:
    """Get the number of qubits in the stabilizer tableau.

    Args:
        tableau (BinaryMatrix): The stabilizer tableau.

    Returns:
        int: The number of qubits.
    """
    from ..codes.pauli import StabilizerTableau

    if isinstance(tableau, StabilizerTableau):
        return tableau.n

    return tableau.matrix.shape[1]


def has_k_non_zero_columns(matrix: BinaryMatrix, k: int) -> bool:
    """Check if the given binary matrix has at least k non-zero columns.

    Args:
        matrix (BinaryMatrix): The binary matrix to check.
        k (int): The number of non-zero columns to check for.

    Returns:
        bool: True if the matrix has at least k non-zero columns, False otherwise.
    """
    non_zero_columns = np.sum(np.any(matrix != 0, axis=0))
    return non_zero_columns >= k


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


class LookaheadCandidateGenerator(CandidateGenerator):
    """Generates candidates using lookahead simulation.

    This generator tracks the best complete solution found during lookahead exploration
    to ensure that greedy local choices lead to globally good solutions.
    """

    def __init__(
        self,
        base_config: EliminationConfig,
        lookahead: int,
        num_lookahead_candidates: int | list[int],
        score_fn: Callable[[EliminationSequence], tuple[int, ...]],
        track_best_solution: bool = True,
        enable_early_termination: bool = True,
    ) -> None:
        """Initialize the lookahead candidate generator.

        Args:
            base_config: Base configuration for greedy candidate generation
            lookahead: Number of steps to look ahead
            num_lookahead_candidates: Number of candidates to explore per layer
            score_fn: Function to score complete elimination sequences
            track_best_solution: If True, tracks best complete solution found during exploration
            enable_early_termination: If True, allows early termination when no improving candidates found
        """
        self.base_config = base_config
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

    def get_candidates(self, tableau: BinaryMatrix) -> list[TableauOperation]:
        """Generate candidates using lookahead simulation.

        Args:
            tableau: The current binary matrix or tableau.

        Returns:
            List of operations sorted by lookahead score.
        """
        if self.lookahead <= 0:
            return self.base_config.candidate_generator.get_candidates(tableau)

        base_candidates = [cand for cand, _ in self.base_config.candidate_generator.get_candidates(tableau)]
        num_candidates_this_layer = self.num_lookahead_candidates_per_layer[0]

        current_filter_state = None
        if self.base_config.filters:
            current_filter_state = [f.copy() for f in self.base_config.filters]

        scored_candidates = _score_candidates_with_lookahead(
            tableau,
            base_candidates,
            num_candidates_this_layer,
            self._create_lookahead_config(current_filter_state),
            self.score_fn,
            self._current_sequence,
            self._best_known_score if self.track_best_solution else None,
            self if self.track_best_solution else None,
        )

        if not scored_candidates and self.track_best_solution and self.enable_early_termination:
            if self._best_known_sequence is not None:
                self._should_terminate = True
                return []

        if not scored_candidates:
            scored_candidates = _score_candidates_with_lookahead(
                tableau,
                base_candidates,
                num_candidates_this_layer,
                self._create_lookahead_config(current_filter_state),
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

    def _create_lookahead_config(self, initial_filter_state: list[OperationFilter] | None) -> EliminationConfig:
        fresh_base_filters = (
            [f.copy() for f in self.base_config.candidate_generator.filters]
            if self.base_config.candidate_generator.filters
            else []
        )
        fresh_base_generator = type(self.base_config.candidate_generator)(fresh_base_filters)

        fresh_base_config = EliminationConfig(
            termination_criterion=self.base_config.termination_criterion,
            candidate_generator=fresh_base_generator,
            filters=self.base_config.filters,
        )

        return EliminationConfig(
            termination_criterion=self.base_config.termination_criterion,
            candidate_generator=LookaheadCandidateGenerator(
                fresh_base_config,
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
        self.base_config.candidate_generator.update(op, tableau)

    def reset(self) -> None:
        """Reset internal state by delegating to the base generator."""
        self._current_sequence = EliminationSequence([])
        self._best_known_score = None
        self._best_known_sequence = None
        self._best_known_tableau = None
        self._should_terminate = False
        self.base_config.candidate_generator.reset()


def _create_tableau_cache_key(tableau: BinaryMatrix) -> bytes:
    """Create a hashable cache key from a tableau.

    Args:
        tableau: The binary matrix or stabilizer tableau.

    Returns:
        A bytes representation suitable for dictionary keys.
    """
    from ..codes.pauli import StabilizerTableau

    if isinstance(tableau, StabilizerTableau):
        return tableau.tableau.matrix.tobytes()
    return tableau.matrix.tobytes()


def _score_candidates_with_lookahead(
    tableau: BinaryMatrix,
    candidates: list[TableauOperation],
    num_candidates: int,
    lookahead_config: EliminationConfig,
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
        lookahead_config: Configuration for lookahead elimination.
        score_fn: Function to compute score tuple from a sequence.
        prefix_sequence: The elimination sequence built so far (for depth calculation).
        best_known_score: Best score found so far; used to prune candidates that can't improve.
        generator: The lookahead generator to record complete solutions.

    Returns:
        List of (operation, score) tuples sorted by score.
    """
    scored_candidates: list[tuple[TableauOperation, tuple[int, ...]]] = []

    for op in candidates[:num_candidates]:
        fresh_config = _create_fresh_lookahead_config(lookahead_config)
        result = _simulate_and_score_operation(op, tableau, fresh_config, score_fn, prefix_sequence, generator)
        if result is not None:
            score_tuple = result
            is_minimal = score_tuple[-1] if isinstance(score_tuple[-1], bool) else False

            if best_known_score is None or score_tuple <= best_known_score:
                scored_candidates.append((op, score_tuple))

            if is_minimal:
                break

    scored_candidates.sort(key=operator.itemgetter(1), reverse=False)
    return scored_candidates


def _create_fresh_lookahead_config(config: EliminationConfig) -> EliminationConfig:
    """Create a fresh copy of the lookahead config with copied filter state.

    Args:
        config: The original lookahead configuration.

    Returns:
        A new config with fresh filter copies.
    """
    fresh_filters = None
    if config.filters:
        fresh_filters = [f.copy() for f in config.filters]

    return EliminationConfig(
        termination_criterion=config.termination_criterion,
        candidate_generator=config.candidate_generator,
        selection_strategy=config.selection_strategy,
        filters=fresh_filters,
        callback=config.callback,
        post_process_fn=config.post_process_fn,
    )


def _simulate_and_score_operation(
    op: TableauOperation,
    tableau: BinaryMatrix,
    lookahead_config: EliminationConfig,
    score_fn: Callable[[EliminationSequence], tuple[int, ...]],
    prefix_sequence: EliminationSequence,
    generator: LookaheadCandidateGenerator | None = None,
) -> tuple[int, ...] | None:
    """Simulate operation and return score tuple, or None if simulation fails.

    Args:
        op: The operation to simulate.
        tableau: The current tableau state.
        lookahead_config: Configuration for lookahead elimination.
        score_fn: Function to compute score tuple from a sequence.
        prefix_sequence: The elimination sequence built so far (for depth calculation).
        generator: The lookahead generator to record complete solutions.

    Returns:
        A tuple containing scores if simulation succeeds, None otherwise.
    """
    try:
        new_tableau = op.apply(tableau)

        if lookahead_config.filters:
            for f in lookahead_config.filters:
                f.update(op)

        sequence, final_tableau = eliminate(new_tableau, lookahead_config)
        full_sequence = EliminationSequence([*prefix_sequence.operations, op, *sequence.operations])
        score = score_fn(full_sequence)

        if generator is not None:
            generator.record_complete_solution(full_sequence, final_tableau, score)

        return score
    except RuntimeError:
        return None
