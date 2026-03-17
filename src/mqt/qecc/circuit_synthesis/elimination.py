# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods for performing Gaussian elimination on GUI2 and symplectic matrices."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import stim

from ..codes.pauli import StabilizerTableau
from .operations import CNOT, Swap, Transvection

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from .operations import TableauOperation
    from .types import BinaryMatrix


class EliminationSequence:
    """Class representing a sequence of tableau operations."""

    def __init__(self, operations: Sequence[TableauOperation]) -> None:
        """Initialize the elimination sequence.

        Args:
            operations: A list of tableau operations.
        """
        self.operations: list[TableauOperation] = []
        self._depth = 0
        self._qubit_depths: defaultdict[int, int] = defaultdict(int)
        for op in operations:
            self.add_operation(op)

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
        # find maximum depth of involved qubits
        involved_qubits = op.qubits()
        earliest_start = 0
        for q in involved_qubits:
            if q in self._qubit_depths:
                earliest_start = max(earliest_start, self._qubit_depths[q] + 1)
        for q in involved_qubits:
            self._qubit_depths[q] = earliest_start
        self._depth = max(self._depth, earliest_start + 1)

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

    def __iter__(self) -> Iterator[TableauOperation]:
        """Return iterator over the tableau operations in the sequence."""
        return iter(self.operations)

    def __reversed__(self) -> Iterator[TableauOperation]:
        """Return reversed iterator over the tableau operations in the sequence."""
        return reversed(self.operations)

    def depth(self) -> int:
        """Estimate the circuit depth of the elimination sequence.

        Returns:
            int: The estimated circuit depth.
        """
        return self._depth

    def copy(self) -> EliminationSequence:
        """Create a copy of the elimination sequence.

        Returns:
            EliminationSequence: A new instance with the same operations.
        """
        return EliminationSequence(self.operations.copy())

    def __eq__(self, other: object) -> bool:
        """Check equality of two elimination sequences based on their operations.

        Args:
            other: The other object to compare with.

        Returns:
            bool: True if the other object is an EliminationSequence with the same operations, False otherwise.
        """
        if not isinstance(other, EliminationSequence):
            return NotImplemented
        return self.operations == other.operations

    def __len__(self) -> int:
        """Return the number of operations in the elimination sequence.

        Returns:
            int: The number of operations.
        """
        return len(self.operations)

    def __hash__(self) -> int:
        """Compute a hash of the elimination sequence based on its operations.

        Returns:
            int: The hash value of the elimination sequence.
        """
        return hash(tuple(self.operations))

    def last_layer_qubits(self) -> set[int]:
        """Get the set of qubits involved in the last layer of operations."""
        if self._depth == 0:
            return set()
        last_layer = self._depth - 1
        return {q for q, d in self._qubit_depths.items() if d == last_layer}


@dataclass
class EliminationStrategy:
    """Strategy for elimination methods."""

    termination_criterion: Callable[[BinaryMatrix], bool]
    candidate_generator: CandidateGenerator
    selection_strategy: SelectionStrategy | None = None
    filters: Sequence[OperationFilter] | None = None
    callback: Callable[[int, TableauOperation, BinaryMatrix], None] | None = None
    post_process_fn: Callable[[EliminationSequence, BinaryMatrix], tuple[EliminationSequence, BinaryMatrix]] = (
        lambda ops, tbl: (ops, tbl)
    )


def eliminate(target_tableau: BinaryMatrix, strategy: EliminationStrategy) -> tuple[EliminationSequence, BinaryMatrix]:
    """Perform Gaussian elimination on the given stabilizer tableau.

    This is the main elimination engine that iteratively reduces a binary matrix or
    stabilizer tableau by applying a sequence of operations (e.g., CNOTs, transvections,
    single-qubit Cliffords) until a termination criterion is met. The function serves as
    the workhorse for synthesizing quantum circuits from stabilizer codes and check matrices.

    Args:
        target_tableau: The input binary matrix or stabilizer tableau to reduce.
            Can be either a CheckMatrix (for CSS codes) or StabilizerTableau
            (for general stabilizer codes).
        strategy: Strategy object specifying:
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
        >>> strategy = EliminationStrategy(
        ...     termination_criterion=lambda tbl: mod2.rank(tbl.matrix) == k,
        ...     candidate_generator=GreedyCNOTGenerator(),
        ... )
        >>> operations, final_tableau = eliminate(check_matrix, strategy)

        >>> # Non-CSS code elimination with depth optimization
        >>> strategy = EliminationStrategy(
        ...     termination_criterion=is_terminal_transvection,
        ...     candidate_generator=GreedyTransvectionGenerator(),
        ...     filters=[ParallelFilter()],
        ... )
        >>> operations, final_tableau = eliminate(stabilizer_tableau, strategy)

        >>> # Lookahead-based elimination
        >>> strategy = EliminationStrategy(
        ...     termination_criterion=is_terminal_transvection,
        ...     candidate_generator=LookaheadCandidateGenerator(...),
        ...     post_process_fn=lambda ops, tbl: reduce_single_qubit_gates_and_swaps(tbl),
        ... )
        >>> operations, final_tableau = eliminate(tableau, strategy)

    See Also:
        - eliminate_css: High-level function for CSS code elimination
        - eliminate_non_css: High-level function for non-CSS code elimination
        - eliminate_non_css_with_lookahead: Lookahead-based non-CSS elimination
        - EliminationStrategy: Configuration dataclass for elimination parameters
        - CandidateGenerator: Abstract base class for candidate generation strategies
        - SelectionStrategy: Abstract base class for operation selection strategies
    """
    tableau = target_tableau.copy()
    operations = EliminationSequence([])
    selection_strategy = strategy.selection_strategy or GreedySelection()
    iteration = 0

    while not strategy.termination_criterion(tableau):
        candidate_ops = strategy.candidate_generator.get_candidates(tableau)

        if _should_terminate_early(strategy.candidate_generator):
            return _get_early_termination_result(strategy.candidate_generator, strategy.post_process_fn, target_tableau)

        if not candidate_ops:
            pass

        if not candidate_ops:
            pass
        _validate_candidates([op for op, _score in candidate_ops])

        op = selection_strategy.select(candidate_ops)

        tableau = op.apply(tableau, inplace=True)

        operations.add_operation(op)

        _update_elimination_state(op, tableau, strategy)
        _invoke_callback(iteration, op, tableau, strategy)
        iteration += 1

    result_ops, result_tableau = strategy.post_process_fn(operations, tableau)

    if hasattr(strategy.candidate_generator, "use_best_if_better"):
        result_ops, result_tableau = _maybe_use_best_solution(
            strategy.candidate_generator, result_ops, result_tableau, target_tableau
        )

    return result_ops, result_tableau


def _maybe_use_best_solution(
    generator: CandidateGenerator,
    current_ops: EliminationSequence,
    current_tableau: BinaryMatrix,
    original_tableau: BinaryMatrix,
) -> tuple[EliminationSequence, BinaryMatrix]:
    """Compare current solution with best tracked solution and return the better one.

    Args:
        generator: The candidate generator that may have tracked a best solution.
        current_ops: The operation sequence from normal elimination.
        current_tableau: The tableau from normal elimination.
        original_tableau: The original tableau before elimination, used to apply the best solution if needed.

    Returns:
        The better of the two solutions (current vs best tracked).
    """
    if not hasattr(generator, "get_best_solution") or not hasattr(generator, "score_fn"):
        return current_ops, current_tableau

    best_solution = generator.get_best_solution()
    if best_solution is None:
        return current_ops, current_tableau

    best_ops = best_solution
    current_score = generator.score_fn(current_ops)
    best_score = generator.score_fn(best_ops)

    if best_score < current_score:
        return best_ops, best_ops.apply(original_tableau, inplace=False)

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
    tableau: BinaryMatrix,
) -> tuple[EliminationSequence, BinaryMatrix]:
    """Get the result when terminating early.

    Args:
        generator: The candidate generator that requested early termination.
        post_process_fn: Function to post-process the result.
        tableau: The original tableau before elimination.

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

    tab = best_solution.apply(tableau, inplace=False)

    return post_process_fn(best_solution, tab)


class CandidateGenerator(ABC):
    """Abstract base class for generating candidate operations."""

    @abstractmethod
    def __init__(self, filters: Sequence[OperationFilter] | None = None) -> None:
        """Initialize the greedy CNOT generator.

        Args:
            filters: Optional list of filters to apply during candidate generation.
        """

    @abstractmethod
    def get_candidates(self, tableau: BinaryMatrix) -> Sequence[tuple[TableauOperation, int | tuple[int, ...]]]:
        """Generate sorted candidate operations for the current tableau.

        Args:
            tableau: The current binary matrix/tableau

        Returns:
            A list of (operation, score) tuples, sorted by preference
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

    def should_terminate_early(self) -> bool:  # noqa: PLR6301
        """Check if elimination should terminate early and use the best solution found.

        Returns:
            True if elimination should terminate early, False otherwise.
        """
        return False

    def get_best_solution(self) -> EliminationSequence | None:  # noqa: PLR6301
        """Get the best complete solution found during lookahead exploration.

        Returns:
            Tuple of (sequence, tableau) if a solution is available, None otherwise.
        """
        return None


class SelectionStrategy(ABC):
    """Abstract base class for selecting the best operation from candidates."""

    @abstractmethod
    def select(self, candidates: Sequence[tuple[TableauOperation, int | tuple[int, ...]]]) -> TableauOperation:
        """Select the best operation from candidates.

        Args:
            candidates: List of (operation, score) tuples, typically sorted by preference.

        Returns:
            The selected operation to apply.
        """


class GreedySelection(SelectionStrategy):
    """Always select the first candidate."""

    def select(self, candidates: Sequence[tuple[TableauOperation, int | tuple[int, ...]]]) -> TableauOperation:  # noqa: PLR6301
        """Select the first (best-scored) candidate.

        Args:
            candidates: List of (operation, score) tuples.

        Returns:
            The first candidate in the list.
        """
        return candidates[0][0]


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

    @abstractmethod
    def reset(self) -> None:
        """Reset the filter to its initial state."""


class ParallelFilter(OperationFilter):
    """Filter that blocks operations on qubits already used in current layer."""

    def __init__(self, n_qubits: int) -> None:
        """Initialize the parallel filter.

        Args:
            n_qubits: Total number of qubits in the circuit. If None, will be inferred from operations.
        """
        self.blocked_qubits: list[bool] = [False] * n_qubits
        self._n_blocked = 0
        self.n_qubits = n_qubits

    def should_include(self, op: TableauOperation) -> bool:
        """Check if operation uses any blocked qubits.

        Args:
            op: The operation to check.

        Returns:
            True if no qubits are blocked, False otherwise.
        """
        return not any(self.blocked_qubits[q] for q in op.qubits())

    def update(self, op: TableauOperation) -> None:
        """Block qubits involved in the operation.

        Args:
            op: The tableau operation to update the filter with.
        """
        qubits_involved = op.qubits()
        for q in qubits_involved:
            if not self.blocked_qubits[q]:
                self.blocked_qubits[q] = True
                self._n_blocked += 1

        if not self.has_available_qubits():
            self.reset()

    def block_qubits(self, qubits: set[int]) -> None:
        """Manually block a list of qubits.

        Args:
            qubits: List of qubit indices to block.
        """
        for q in qubits:
            if not self.blocked_qubits[q]:
                self.blocked_qubits[q] = True
                self._n_blocked += 1

        if not self.has_available_qubits():
            self.reset()

    def has_available_qubits(self) -> bool:
        """Check if there are qubits available for operations."""
        return self._n_blocked < self.n_qubits - 1  # two qubits should be free

    def reset(self) -> None:
        """Unblock all qubits."""
        self._n_blocked = 0
        self.blocked_qubits = [False] * self.n_qubits

    def copy(self) -> ParallelFilter:
        """Create a copy of the filter with the same state.

        Returns:
            A new ParallelFilter with copied blocked_qubits state.
        """
        new_filter = ParallelFilter(n_qubits=self.n_qubits)
        new_filter.blocked_qubits = self.blocked_qubits.copy()
        new_filter._n_blocked = self._n_blocked
        return new_filter


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


def _update_elimination_state(op: TableauOperation, tableau: BinaryMatrix, strategy: EliminationStrategy) -> None:
    """Update generator and filter state after applying an operation.

    Args:
        op: The operation that was applied.
        tableau: The resulting tableau after applying the operation.
        strategy: Elimination strategy containing generator and filters.
    """
    strategy.candidate_generator.update(op, tableau)


def _invoke_callback(
    iteration: int, op: TableauOperation, tableau: BinaryMatrix, strategy: EliminationStrategy
) -> None:
    """Invoke callback if configured.

    Args:
        iteration: Current iteration number.
        op: The operation that was just applied.
        tableau: The resulting tableau after applying the operation.
        strategy: Elimination strategy potentially containing a callback.
    """
    if strategy.callback:
        strategy.callback(iteration, op, tableau)


def is_identity(tableau: StabilizerTableau) -> bool:
    """Check if the given stabilizer tableau is the identity tableau.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to check.

    Returns:
        bool: True if the tableau is the identity tableau, False otherwise.
    """
    n = get_n(tableau)
    identity_matrix = np.eye(2 * n, dtype=np.int8)
    return bool(np.array_equal(tableau.tableau.matrix, identity_matrix))


def get_n(tableau: BinaryMatrix) -> int:
    """Get the number of qubits in the stabilizer tableau.

    Args:
        tableau (BinaryMatrix): The stabilizer tableau.

    Returns:
        int: The number of qubits.
    """
    if isinstance(tableau, StabilizerTableau):
        return int(tableau.n)

    return int(tableau.matrix.shape[1])


def has_k_non_zero_columns(matrix: BinaryMatrix, k: int) -> bool:
    """Check if the given binary matrix has at least k non-zero columns.

    Args:
        matrix (BinaryMatrix): The binary matrix to check.
        k (int): The number of non-zero columns to check for.

    Returns:
        bool: True if the matrix has at least k non-zero columns, False otherwise.
    """
    non_zero_columns = int(np.sum(np.any(matrix != 0, axis=0)))
    return non_zero_columns >= k
