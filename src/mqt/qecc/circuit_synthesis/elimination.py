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

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import stim

from ..codes.pauli import CheckMatrix, StabilizerTableau

if TYPE_CHECKING:
    import numpy.typing as npt


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
    
    if hasattr(config.candidate_generator, 'use_best_if_better'):
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
    if not hasattr(generator, 'get_best_solution') or not hasattr(generator, 'score_fn'):
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


class GreedyTransvectionGenerator(CandidateGenerator):
    """Generates transvection candidates using greedy heuristic."""

    def __init__(self, filters: list[OperationFilter] | None = None) -> None:
        """Initialize the greedy transvection generator.

        Args:
            filters: Optional list of filters to apply during candidate generation.
        """
        self.operation_history: list[TableauOperation] = []
        self.filters = filters or []

    def get_candidates(self, tableau: BinaryMatrix) -> list[tuple[TableauOperation, int]]:
        """Generate transvection candidates sorted by heuristic score.

        Args:
            tableau: The current stabilizer tableau.

        Returns:
            List of transvection operations sorted by preference.
        """
        all_candidates = get_candidate_transvections(tableau)
        return self._apply_filters(all_candidates)

    def _apply_filters(self, candidates: list[tuple[TableauOperation, int]]) -> list[tuple[TableauOperation, int]]:
        """Apply all filters to candidate list.

        Args:
            candidates: List of candidate operations with scores.

        Returns:
            Filtered list of candidates.
        """
        if not self.filters:
            return candidates

        filtered = []
        for op, score in candidates:
            if score > 0 and all(f.should_include(op) for f in self.filters):
                filtered.append((op, score))

        if not filtered:
            for f in self.filters:
                if hasattr(f, "_reset"):
                    f._reset()
            return candidates

        return filtered

    def update(self, op: TableauOperation, tableau: BinaryMatrix) -> None:  # noqa: ARG002
        """Update operation history and filters after applying an operation.

        Args:
            op: The operation that was applied.
            tableau: The resulting tableau after applying the operation.
        """
        self.operation_history.append(op)
        for f in self.filters:
            f.update(op)

    def reset(self) -> None:
        """Reset the operation history."""
        self.operation_history.clear()


class GreedyTransvectionGeneratorStateprep(CandidateGenerator):
    """Generates transvection candidates using greedy heuristic."""

    def __init__(self, filters: list[OperationFilter] | None = None) -> None:
        """Initialize the greedy transvection generator.

        Args:
            filters: Optional list of filters to apply during candidate generation.
        """
        self.operation_history: list[TableauOperation] = []
        self.filters = filters or []

    def get_candidates(self, tableau: BinaryMatrix) -> list[tuple[TableauOperation, int]]:
        """Generate transvection candidates sorted by heuristic score.

        Args:
            tableau: The current stabilizer tableau.

        Returns:
            List of transvection operations sorted by preference.
        """
        all_candidates = get_candidate_transvections_stateprep(tableau)
        return self._apply_filters(all_candidates)

    def _apply_filters(self, candidates: list[tuple[TableauOperation, int]]) -> list[tuple[TableauOperation, int]]:
        """Apply all filters to candidate list.

        Args:
            candidates: List of candidate operations with scores.

        Returns:
            Filtered list of candidates.
        """
        if not self.filters:
            return candidates

        filtered = []
        for op, score in candidates:
            if score > 0 and all(f.should_include(op) for f in self.filters):
                filtered.append((op, score))

        if not filtered:
            for f in self.filters:
                if hasattr(f, "_reset"):
                    f._reset()
            return candidates

        return filtered

    def update(self, op: TableauOperation, tableau: BinaryMatrix) -> None:  # noqa: ARG002
        """Update operation history and filters after applying an operation.

        Args:
            op: The operation that was applied.
            tableau: The resulting tableau after applying the operation.
        """
        self.operation_history.append(op)
        for f in self.filters:
            f.update(op)

    def reset(self) -> None:
        """Reset the operation history."""
        self.operation_history.clear()


class GreedyCNOTGenerator(CandidateGenerator):
    """Generates CNOT candidates using greedy heuristic for CSS codes."""

    def __init__(self, filters: list[OperationFilter] | None = None) -> None:
        """Initialize the greedy CNOT generator.

        Args:
            filters: Optional list of filters to apply during candidate generation.
        """
        self.operation_history: list[TableauOperation] = []
        self.filters = filters or []

    def get_candidates(self, tableau: BinaryMatrix) -> list[TableauOperation]:
        """Generate CNOT candidates sorted by heuristic score.

        Args:
            tableau: The current check matrix.

        Returns:
            List of CNOT operations sorted by preference.
        """
        all_candidates = greedy_matrix_elimination_candidates(tableau)
        return self._apply_filters(all_candidates)

    def _apply_filters(self, candidates: list[tuple[TableauOperation, int]]) -> list[tuple[TableauOperation, int]]:
        """Apply all filters to candidate list.

        Args:
            candidates: List of candidate operations with scores.

        Returns:
            Filtered list of candidates.
        """
        if not self.filters:
            return candidates

        filtered = []
        for op, score in candidates:
            if score > 0 and all(f.should_include(op) for f in self.filters):
                filtered.append((op, score))

        if not filtered:
            for f in self.filters:
                if hasattr(f, "_reset"):
                    f._reset()
            return candidates

        return filtered

    def update(self, op: TableauOperation, tableau: BinaryMatrix) -> None:
        """Update operation history and filters after applying an operation.

        Args:
            op: The operation that was applied.
            tableau: The resulting tableau after applying the operation.
        """
        self.operation_history.append(op)
        for f in self.filters:
            f.update(op)

    def reset(self) -> None:
        """Reset the operation history."""
        self.operation_history.clear()


def eliminate_non_css_state(
    tableau: StabilizerTableau, optimization_criterion: str = "gates"
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Eliminate a non-CSS stabilizer tableau to state preparation form using transvections.

    Args:
        tableau: The stabilizer tableau to eliminate.

    Returns:
        A tuple of (operations, final_tableau) where operations is the sequence
        of tableau operations and final_tableau is the reduced tableau.
    """
    config = EliminationConfig.for_non_css_stateprep(optimization_criterion=optimization_criterion)

    operations, final_tableau = eliminate(tableau, config)

    return operations, final_tableau


def eliminate_non_css(
    tableau: StabilizerTableau, optimization_criterion: str = "gates"
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Eliminate a non-CSS stabilizer tableau using transvections.

    Args:
        tableau: The stabilizer tableau to eliminate.
        optimization_criterion: Either "gates" or "depth" for optimization objective.
        lookahead: Currently unused parameter for future lookahead support.

    Returns:
        A tuple of (operations, final_tableau) where operations is the sequence
        of tableau operations and final_tableau is the reduced tableau.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    config = EliminationConfig.for_non_css(optimization_criterion=optimization_criterion)

    operations, final_tableau = eliminate(tableau, config)

    return operations, final_tableau


def eliminate_cnot_lookahead(
    matrix: CheckMatrix,
    optimization_criterion: str = "gates",
    lookahead: int = 1,
    num_lookahead_candidates: int | list[int] = 10,
) -> tuple[EliminationSequence, CheckMatrix]:
    """Eliminate a CSS check matrix using CNOT operations with lookahead.

    Args:
        matrix: The CSS check matrix to eliminate.
        optimization_criterion: Either "gates" or "depth" for optimization objective.
        lookahead: Number of steps to look ahead in the synthesis.
        num_lookahead_candidates: Number of top candidates to explore at each lookahead layer.
            Can be a single int (same limit for all layers) or a list of ints (one per layer).

    Returns:
        A tuple of (operations, final_matrix) where operations is the sequence
        of CNOT operations and final_matrix is the reduced check matrix.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    config = EliminationConfig.for_cnot_with_lookahead(
        optimization_criterion=optimization_criterion,
        lookahead=lookahead,
        num_lookahead_candidates=num_lookahead_candidates,
    )
    operations, final_matrix = eliminate(matrix, config)
    return operations, final_matrix


def eliminate_cnot(
    matrix: CheckMatrix,
    optimization_criterion: str = "gates",
    exact: bool = True,
    lookahead: int = 0,
    num_lookahead_candidates: int | list[int] = 10,
    enable_early_termination: bool = True,
) -> tuple[EliminationSequence, CheckMatrix]:
    """Eliminate a CSS check matrix using CNOT operations.

    Args:
        matrix: The CSS check matrix to eliminate.
        optimization_criterion: Either "gates" or "depth" for optimization objective.
        exact: If True, eliminate to echelon form. If False, eliminate only up to row operations.
        lookahead: Number of steps to look ahead (0 = greedy).
        num_lookahead_candidates: Number of candidates to explore at each lookahead layer.
        enable_early_termination: If True, allows early termination when no improving candidates found.

    Returns:
        A tuple of (operations, final_matrix) where operations is the sequence
        of CNOT operations and final_matrix is the reduced check matrix.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    if matrix.num_rows() == 0:
        return EliminationSequence([]), matrix.copy()

    target_rank = mod2.rank(matrix.matrix)
    if exact:
        if lookahead > 0:
            config = EliminationConfig.for_cnot_with_lookahead_exact(
                target_rank=target_rank,
                optimization_criterion=optimization_criterion,
                lookahead=lookahead,
                num_lookahead_candidates=num_lookahead_candidates,
                enable_early_termination=enable_early_termination,
            )
        else:
            config = EliminationConfig.for_cnot_exact(
                target_rank=target_rank, optimization_criterion=optimization_criterion
            )
    elif lookahead > 0:
        config = EliminationConfig.for_cnot_with_lookahead_up_to_row_ops(
            optimization_criterion=optimization_criterion,
            lookahead=lookahead,
            num_lookahead_candidates=num_lookahead_candidates,
            target_rank=target_rank,
            enable_early_termination=enable_early_termination,
        )
    else:
        config = EliminationConfig.for_cnot_up_to_row_ops(
            target_rank=target_rank, optimization_criterion=optimization_criterion
        )

    operations, final_matrix = eliminate(matrix, config)

    if matrix.is_z_type():
        for op in operations.operations:
            if isinstance(op, CNOT):
                op.control, op.target = op.target, op.control

    return operations, final_matrix


def eliminate_non_css_with_lookahead(
    tableau: StabilizerTableau,
    optimization_criterion: str = "gates",
    lookahead: int = 1,
    num_lookahead_candidates: int | list[int] = 10,
    enable_early_termination: bool = True,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Eliminate a non-CSS stabilizer tableau using transvections with lookahead.

    Args:
        tableau: The stabilizer tableau to eliminate.
        optimization_criterion: Either "gates" or "depth" for optimization objective.
        lookahead: Number of steps to look ahead in the synthesis.
        num_lookahead_candidates: Number of top candidates to explore at each lookahead layer.
            Can be a single int (same limit for all layers) or a list of ints (one per layer).
        enable_early_termination: If True, allows early termination when no improving candidates found.

    Returns:
        A tuple of (operations, final_tableau) where operations is the sequence
        of tableau operations and final_tableau is the reduced tableau.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    config = EliminationConfig.for_non_css_with_lookahead(
        optimization_criterion=optimization_criterion,
        lookahead=lookahead,
        num_lookahead_candidates=num_lookahead_candidates,
        enable_early_termination=enable_early_termination,
    )
    operations, final_tableau = eliminate(tableau, config)
    return operations, final_tableau


BinaryMatrix = CheckMatrix | StabilizerTableau


class TableauOperation(ABC):
    """Represents an operation performed during tableau elimination."""

    @abstractmethod
    def apply(self, tableau: BinaryMatrix, inplace: bool = False) -> BinaryMatrix:
        """Apply the operation to the given stabilizer tableau.

        Args:
            tableau (BinaryMatrix): The stabilizer tableau to apply the operation to.
            inplace (bool): If True, modifies the tableau in place. If False, returns a new tableau.

        Args:
            tableau (BinaryMatrix): The stabilizer tableau to apply the operation to.
        """

    @abstractmethod
    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit to append the operation to.
        """

    def to_stim_circuit(self) -> stim.Circuit:
        """Convert the operation to a Stim circuit representation.

        Returns:
            stim.Circuit: The Stim circuit representing the operation.
        """
        circuit = stim.Circuit()
        self.append_to_circuit(circuit)
        return circuit

    @abstractmethod
    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """

    def __repr__(self) -> str:
        """Return a string representation of the operation."""
        return f"{self.__class__.__name__}(qubits={self.qubits()})"


TV2 = tuple[int, int, int, int]


class Transvection(TableauOperation):
    """Class representing a transvection operation on a stabilizer tableau."""

    def __init__(self, v: TV2, i: int, j: int) -> None:
        """Initialize the transvection operation.

        Args:
            v: A tuple representing the transvection vector (v1, v2, v3, v4).
            i: The index of the first qubit.
            j: The index of the second qubit.
        """
        self.i = i
        self.j = j
        self.v = v

    def apply(self, tableau: BinaryMatrix, inplace: bool = False) -> BinaryMatrix:
        """Apply the transvection operation to the given stabilizer tableau.

        This applies the transvection by simulating the circuit: basis change, CZ, S on both qubits, undo basis change.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace (bool): If True, modifies the tableau in place. If False, returns a new tableau.
        """
        if not isinstance(tableau, StabilizerTableau):
            msg = "Transvection operations can only be applied to StabilizerTableau instances."
            raise TypeError(msg)

        out = tableau if inplace else tableau.copy()

        i = self.i
        j = self.j
        xi, xj, zi, zj = self.v

        paulis = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
        p_i = paulis[xi, zi]
        p_j = paulis[xj, zj]

        if p_i == "I" or p_j == "I":
            msg = f"Expected non-trivial Pauli on both qubits, got {p_i},{p_j}"
            raise ValueError(msg)

        basis_change_map = {"Z": None, "X": "H", "Y": "SH"}
        undo_basis_change_map = {"Z": None, "X": "H", "Y": "HS"}

        basis_i = basis_change_map[p_i]
        basis_j = basis_change_map[p_j]
        undo_i = undo_basis_change_map[p_i]
        undo_j = undo_basis_change_map[p_j]

        if basis_i == "H":
            out.apply_h(i)
        elif basis_i == "SH":
            out.apply_sdg(i)
            out.apply_h(i)

        if basis_j == "H":
            out.apply_h(j)
        elif basis_j == "SH":
            out.apply_sdg(j)
            out.apply_h(j)

        out.apply_cz(i, j)
        out.apply_s(i)
        out.apply_s(j)

        if undo_j == "H":
            out.apply_h(j)
        elif undo_j == "HS":
            out.apply_h(j)
            out.apply_s(j)

        if undo_i == "H":
            out.apply_h(i)
        elif undo_i == "HS":
            out.apply_h(i)
            out.apply_s(i)

        return out

    @staticmethod
    def all_two_qubit_transvections() -> list[TV2]:
        """Get all 9 possible two-qubit transvections.

        The 9 distinct 2-qubit transvections √(P_i P_j) (P∈{X,Y,Z} non-trivial)
        correspond to choosing (x,z) in {(1,0),(0,1),(1,1)} for each of the two qubits.

        Returns:
            List of all 9 transvection vectors as tuples (xi, xj, zi, zj).
        """
        nontrivial = [(1, 0), (0, 1), (1, 1)]
        out: list[TV2] = []
        for xi, zi in nontrivial:
            for xj, zj in nontrivial:
                out.append((xi, xj, zi, zj))
        return out

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit to append the operation to.
        """
        paulis = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
        i = self.i
        j = self.j
        xi, xj, zi, zj = self.v
        p_i = paulis[xi, zi]
        p_j = paulis[xj, zj]
        if p_i == "I" or p_j == "I":
            msg = f"Expected non-trivial Pauli on both qubits, got {p_i},{p_j}"
            raise ValueError(msg)

        basis_change = {"Z": [], "X": ["H"], "Y": ["S_DAG", "H"]}
        undo_basis_change = {"Z": [], "X": ["H"], "Y": ["H", "S"]}
        for g in basis_change[p_i]:
            circuit.append(g, [i])
        for g in basis_change[p_j]:
            circuit.append(g, [j])

        circuit.append("CZ", [i, j])
        circuit.append("S", [i])
        circuit.append("S", [j])

        for g in undo_basis_change[p_j]:
            circuit.append(g, [j])
        for g in undo_basis_change[p_i]:
            circuit.append(g, [i])

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """
        return {self.i, self.j}


class SingleQubitClifford(TableauOperation):
    """Class representing a single-qubit Clifford operation on a stabilizer tableau."""

    def __init__(self, qubit: int, clifford: str) -> None:
        """Initialize the single-qubit Clifford operation.

        Args:
            qubit: The index of the qubit.
            clifford: The Clifford operation to apply {H, S, HS, SH, HSH, I}.
        """
        self.qubit = qubit
        self.clifford = clifford

    def apply(self, tableau: BinaryMatrix, inplace: bool = False) -> BinaryMatrix:
        """Apply the single-qubit Clifford operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace (bool): If True, modifies the tableau in place. If False, returns a new tableau.
        """
        if not isinstance(tableau, StabilizerTableau):
            msg = "SingleQubitClifford operations can only be applied to StabilizerTableau instances."
            raise TypeError(msg)

        q = self.qubit

        out = tableau if inplace else tableau.copy()
        if self.clifford == "H":
            out.apply_h(q)
        elif self.clifford == "S":
            out.apply_s(q)
        elif self.clifford == "HS":
            out.apply_h(q)
            out.apply_s(q)
        elif self.clifford == "SH":
            out.apply_s(q)
            out.apply_h(q)
        elif self.clifford == "HSH":
            out.apply_h(q)
            out.apply_s(q)
            out.apply_h(q)
        elif self.clifford == "I":
            pass
        else:
            msg = f"Unsupported single-qubit Clifford operation: {self.clifford}"
            raise ValueError(msg)
        return out

    def apply_inverse(self, tableau: BinaryMatrix, inplace: bool = False) -> BinaryMatrix:
        """Apply the inverse of the single-qubit Clifford operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace: Whether to modify the tableau in place.

        Returns:
            BinaryMatrix: The resulting stabilizer tableau after applying the inverse operation.
        """
        if not isinstance(tableau, StabilizerTableau):
            msg = "SingleQubitClifford operations can only be applied to StabilizerTableau instances."
            raise TypeError(msg)
        q = self.qubit

        out = tableau if inplace else tableau.copy()
        if self.clifford == "H":
            out.apply_h(q)
        elif self.clifford == "S":
            out.apply_sdg(q)
        elif self.clifford == "HS":
            out.apply_sdg(q)
            out.apply_h(q)
        elif self.clifford == "SH":
            out.apply_h(q)
            out.apply_sdg(q)
        elif self.clifford == "HSH":
            out.apply_h(q)
            out.apply_sdg(q)
            out.apply_h(q)
        elif self.clifford == "I":
            pass
        else:
            msg = f"Unsupported single-qubit Clifford operation: {self.clifford}"
            raise ValueError(msg)
        return out

    def inverse(self) -> SingleQubitClifford:
        """Get the inverse of the single-qubit Clifford operation.

        Returns:
            SingleQubitClifford: The inverse single-qubit Clifford operation.
        """
        inverse_map = {
            "H": "H",
            "S": "SH",
            "HS": "SH",
            "SH": "HS",
            "HSH": "HSH",
            "I": "I",
        }
        if self.clifford not in inverse_map:
            msg = f"Unsupported single-qubit Clifford operation: {self.clifford}"
            raise ValueError(msg)
        return SingleQubitClifford(self.qubit, inverse_map[self.clifford])

    @staticmethod
    def available_cliffords() -> list[str]:
        """Get the list of available single-qubit Clifford operations.

        Returns:
            List of Clifford operation names: H, S, HS, SH, HSH, I.
        """
        return ["H", "S", "HS", "SH", "HSH", "I"]

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit to append the operation to.
        """
        if self.clifford in {"H", "S", "I"}:
            circuit.append(self.clifford, [self.qubit])
        elif self.clifford == "HS":
            circuit.append("H", [self.qubit])
            circuit.append("S", [self.qubit])
        elif self.clifford == "SH":
            circuit.append("S", [self.qubit])
            circuit.append("H", [self.qubit])
        elif self.clifford == "HSH":
            circuit.append("H", [self.qubit])
            circuit.append("S", [self.qubit])
            circuit.append("H", [self.qubit])
        else:
            msg = f"Unsupported single-qubit Clifford operation: {self.clifford}"
            raise ValueError(msg)

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """
        return {self.qubit}

    @staticmethod
    def from_symplectic_block(block: npt.NDArray[np.int8], qubit: int) -> SingleQubitClifford:
        """Create a SingleQubitClifford from a symplectic block.

        Args:
            block: A 2x2 symplectic block representing the single-qubit Clifford operation.
            qubit: The index of the qubit.

        Returns:
            A single-qubit Clifford operation.
        """
        for name, mat in elems.items():
            if np.array_equal(block, mat):
                return SingleQubitClifford(qubit, name)
        msg = f"Unsupported single-qubit Clifford symplectic block:\n{block}"
        raise ValueError(msg)


class PauliOperation(TableauOperation):
    """Class representing a Pauli operation on a stabilizer tableau."""

    def __init__(self, qubit: int, pauli: str) -> None:
        """Initialize the Pauli operation.

        Args:
            qubit: The index of the qubit.
            pauli: The Pauli operation to apply {X, Y, Z}.
        """
        self.qubit = qubit
        self.pauli = pauli

    def apply(self, tableau: BinaryMatrix, inplace: bool = False) -> BinaryMatrix:
        """Apply the Pauli operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace (bool): If True, modifies the tableau in place. If False, returns a new tableau.
        """
        if not isinstance(tableau, StabilizerTableau):
            msg = "Pauli operations can only be applied to StabilizerTableau instances."
            raise TypeError(msg)

        out = tableau if inplace else tableau.copy()
        if self.pauli == "X":
            out.apply_x(self.qubit)
        elif self.pauli == "Y":
            out.apply_y(self.qubit)
        elif self.pauli == "Z":
            out.apply_z(self.qubit)
        else:
            msg = f"Unsupported Pauli operation: {self.pauli}"
            raise ValueError(msg)
        return out

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit to append the operation to.
        """
        circuit.append(self.pauli, [self.qubit])

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """
        return {self.qubit}


def _matmul2(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return ((m1 @ m2) % 2).astype(np.int8)


identity = np.array([[1, 0], [0, 1]], dtype=np.int8)
hadamard = np.array([[0, 1], [1, 0]], dtype=np.int8)
phase = np.array([[1, 1], [0, 1]], dtype=np.int8)

elems: dict[str, np.ndarray] = {
    "I": identity,
    "H": hadamard,
    "S": phase,
    "SH": _matmul2(hadamard, phase),
    "HS": _matmul2(phase, hadamard),
    "HSH": _matmul2(_matmul2(hadamard, phase), hadamard),
}


class CNOT(TableauOperation):
    """Class representing a CNOT operation on a stabilizer tableau."""

    def __init__(self, control: int, target: int) -> None:
        """Initialize the CNOT operation.

        Args:
            control: The index of the control qubit.
            target: The index of the target qubit.
        """
        self.control = control
        self.target = target

    def apply(self, tableau: BinaryMatrix, inplace: bool = False) -> BinaryMatrix:
        """Apply the CNOT operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace (bool): If True, modifies the tableau in place. If False, returns a new tableau.
        """
        if isinstance(tableau, StabilizerTableau):
            return self._apply_stabilizer_tableau(tableau, inplace)
        return self._apply_check_matrix(tableau, inplace)

    def _apply_stabilizer_tableau(self, tableau: StabilizerTableau, inplace: bool = False) -> StabilizerTableau:
        out = tableau if inplace else tableau.copy()
        out.apply_cx(self.control, self.target)
        return out

    def _apply_check_matrix(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:
        """Apply the operation to a CSS check matrix.

        Args:
            check_matrix (CheckMatrix): The CSS check matrix to apply the operation to.
            inplace (bool): If True, modifies the check matrix in place. If False, returns a new check matrix.

        Returns:
            CheckMatrix: The resulting CSS check matrix after applying the operation.
        """
        out = check_matrix if inplace else check_matrix.copy()
        out.matrix[:, self.target] ^= out.matrix[:, self.control]
        return out

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit to append the operation to.
        """
        circuit.append("CNOT", [self.control, self.target])

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """
        return {self.control, self.target}


class Swap(TableauOperation):
    """Class representing a SWAP operation on a stabilizer tableau."""

    def __init__(self, qubit_a: int, qubit_b: int) -> None:
        """Initialize the SWAP operation.

        Args:
            qubit_a: The index of the first qubit.
            qubit_b: The index of the second qubit.
        """
        self.qubit_a = qubit_a
        self.qubit_b = qubit_b

    def apply(self, tableau: BinaryMatrix, inplace: bool = False) -> BinaryMatrix:
        """Apply the SWAP operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace (bool): If True, modifies the tableau in place. If False, returns a new tableau.
        """
        if isinstance(tableau, StabilizerTableau):
            return self._apply_stabilizer_tableau(tableau, inplace)
        return self._apply_check_matrix(tableau, inplace)

    def _apply_stabilizer_tableau(self, tableau: StabilizerTableau, inplace: bool = False) -> StabilizerTableau:
        out = tableau if inplace else tableau.copy()
        out.apply_swap(self.qubit_a, self.qubit_b)
        return out

    def _apply_check_matrix(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:
        """Apply the operation to a CSS check matrix.

        Args:
            check_matrix (CheckMatrix): The CSS check matrix to apply the operation to.
            inplace (bool): If True, modifies the check matrix in place. If False, returns a new check matrix.
        """
        out = check_matrix if inplace else check_matrix.copy()
        out.matrix[:, [self.qubit_a, self.qubit_b]] = out.matrix[:, [self.qubit_b, self.qubit_a]]
        return out

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit to append the operation to.
        """
        circuit.append("SWAP", [self.qubit_a, self.qubit_b])

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """
        return {self.qubit_a, self.qubit_b}


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

        filters = [ParallelFilter()]

        def termination_criterion(tbl: BinaryMatrix) -> bool:
            if not isinstance(tbl, (CheckMatrix)):
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

        filters = [ParallelFilter()] if optimization_criterion == "depth" else []

        def termination_criterion(tbl: BinaryMatrix) -> bool:
            if not isinstance(tbl, (CheckMatrix)):
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

        filters = [ParallelFilter()] if optimization_criterion == "depth" else []

        return cls(
            termination_criterion=is_terminal_stateprep,
            candidate_generator=GreedyTransvectionGeneratorStateprep(filters),
            filters=filters,
            callback=callback,
            post_process_fn=reduce_singe_qubit_gates_stateprep,
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
            if not isinstance(tbl, (CheckMatrix)):
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
            if not isinstance(tbl, (CheckMatrix)):
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


def _compute_r2_matrix(symplectic: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    n = symplectic.shape[0] // 2
    a_xx = symplectic[:n, :n]
    a_xz = symplectic[:n, n:]
    a_zx = symplectic[n:, :n]
    a_zz = symplectic[n:, n:]
    det = (a_xx & a_zz) ^ (a_xz & a_zx)
    return det.astype(np.int8)


def _compute_r0_matrix(symplectic: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    n = symplectic.shape[0] // 2
    a_xx = symplectic[:n, :n]
    a_xz = symplectic[:n, n:]
    a_zx = symplectic[n:, :n]
    a_zz = symplectic[n:, n:]
    zero = (a_xx == 0) & (a_xz == 0) & (a_zx == 0) & (a_zz == 0)
    return zero.astype(np.int8)


def _compute_r1_matrix_from_r2_r0(R2: npt.NDArray[np.int8], R0: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    return (1 ^ (R2 | R0)).astype(np.int8)


def r1_r2(symplectic: npt.NDArray[np.int8]) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """Compute R1 and R2 matrices from a symplectic matrix."""
    n = symplectic.shape[0] // 2

    a_xx = symplectic[:n, :n]
    a_xz = symplectic[:n, n:]
    a_zx = symplectic[n:, :n]
    a_zz = symplectic[n:, n:]

    r2 = (a_xx & a_zz) ^ (a_xz & a_zx)
    r0 = ~(a_xx | a_xz | a_zx | a_zz)
    r1 = ~(r2 | r0)

    return r1.astype(np.int8), r2.astype(np.int8)


def is_terminal_stateprep(tableau: StabilizerTableau) -> bool:
    """Check if the given stabilizer tableau is in terminal form for state preparation.

    This is the case when there are no overlaps between any pair of qubits.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to check.

    Returns:
        bool: True if the tableau is in terminal form, False otherwise.
    """
    return score_stateprep(tableau) == 0


def is_terminal_transvection(tableau: StabilizerTableau) -> bool:
    """Check if the given stabilizer tableau is in terminal form for transvection elimination.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to check.

    Returns:
        bool: True if the tableau is in terminal form, False otherwise.
    """
    r1, r2 = r1_r2(tableau.tableau.matrix)
    if np.any(r1):
        return False
    if not np.all(r2.sum(axis=0) == 1):
        return False
    return np.all(r2.sum(axis=1) == 1)


def score_stateprep(tableau: StabilizerTableau) -> int:
    r"""Score the given symplectic matrix representing a state.

    The score is the total number of "overlap" between qubit pairs, i.e., where there is a
    "1" for both qubits.

    Args:
        tableau: The stabilizer tableau to score.

    Returns:
        An integer score used for comparing tableaus.
    """
    n = get_n(tableau)
    symplectic = tableau.tableau.matrix
    symplectic.shape[0]
    score = 0
    for q1 in range(n):
        for q2 in range(q1 + 1, n):
            x1 = symplectic[:, q1]
            z1 = symplectic[:, q1 + n]
            x2 = symplectic[:, q2]
            z2 = symplectic[:, q2 + n]

            score += ((x1 & x2) | (x1 & z2) | (z1 & x2) | (z1 & z2)).sum()

    return score


def score_symplectic(tableau: StabilizerTableau) -> tuple[tuple[int, ...], int]:
    """Score the given symplectic matrix using the default symplectic heuristic.

    Args:
        tableau: The stabilizer tableau to score.

    Returns:
        A tuple of (heuristic_vector, scalar_score) used for comparing tableaus.
    """
    n = get_n(tableau)

    symplectic = tableau.tableau.matrix
    r1, r2 = r1_r2(symplectic)

    c1 = r1.sum(axis=0).astype(int)
    c2 = r2.sum(axis=0).astype(int)

    c1t = r1.sum(axis=1).astype(int)
    c2t = r2.sum(axis=1).astype(int)
    vec = np.concatenate([n * c2 + c1, n * c2t + c1t])

    h_vec = tuple(sorted(int(x) for x in vec))

    h_scalar = int(r1.sum() + r2.sum())
    return h_vec, h_scalar


def _bin2set(row: npt.NDArray[np.int8]) -> list[int]:
    """Convert a binary row to a list of column indices where the value is 1."""
    return [int(i) for i in np.flatnonzero(row)]


def _sp_gate_options(symplectic: npt.NDArray[np.int8]) -> list[tuple[int, int]]:
    """Return a reduced set of candidate pairs (i,j) to consider, based on R2/R1 structure.

    Args:
        symplectic: The symplectic matrix (2n x 2n).

    Returns:
        A sorted list of (i, j) pairs where i < j, representing candidate qubit pairs.
    """
    n = symplectic.shape[0] // 2
    R1, R2 = r1_r2(symplectic)
    pairs: set[tuple[int, int]] = set()

    for row in range(n):
        r2_cols = _bin2set(R2[row])
        r1_cols = _bin2set(R1[row])

        for a in range(len(r2_cols) - 1):
            for b in range(a + 1, len(r2_cols)):
                i, j = int(r2_cols[a]), int(r2_cols[b])
                if i != j:
                    pairs.add((min(i, j), max(i, j)))

        for i0 in r2_cols:
            for j0 in r1_cols:
                i, j = int(i0), int(j0)
                if i != j:
                    pairs.add((min(i, j), max(i, j)))

    return sorted(pairs)


def get_candidate_transvections_stateprep(
    tableau: StabilizerTableau,
) -> list[Transvection]:
    """Score all possible operations and return the top k scored operations.

    Args:
        tableau: The current symplectic matrix.
        transvections: List of all two-qubit transvections.
        pairs: List of qubit pairs to consider.
        params: Parameters for the greedy synthesis.
        k: Number of top scored operations to return.

    Returns:
        A list of the top k scored operations, each represented as a tuple of
        (operation, heuristic vector and scalar, resulting matrix).
    """
    n = get_n(tableau)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    transvections = Transvection.all_two_qubit_transvections()
    scores: list[tuple(Transvection, list[int, ...])] = []
    for i, j in pairs:
        for v in transvections:
            op = Transvection(v, i, j)
            tablea_op_applied = op.apply(tableau)
            s = score_stateprep(tablea_op_applied)
            if s == 0:
                pass

            scores.append((op, s))

    scores.sort(key=operator.itemgetter(1))
    return [(tv, score) for tv, score in scores]


def get_candidate_transvections(
    tableau: StabilizerTableau,
) -> list[Transvection]:
    """Score all possible operations and return the top k scored operations.

    Args:
        tableau: The current symplectic matrix.
        transvections: List of all two-qubit transvections.
        pairs: List of qubit pairs to consider.
        params: Parameters for the greedy synthesis.
        k: Number of top scored operations to return.

    Returns:
        A list of the top k scored operations, each represented as a tuple of
        (operation, heuristic vector and scalar, resulting matrix).
    """
    n = get_n(tableau)
    symplectic = tableau.tableau.matrix

    pairs = _sp_gate_options(symplectic)

    if not pairs:
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    transvections = Transvection.all_two_qubit_transvections()
    scores: list[tuple(Transvection, list[int, ...])] = []
    base_score, _ = score_symplectic(tableau)
    for i, j in pairs:
        for v in transvections:
            op = Transvection(v, i, j)
            tablea_op_applied = op.apply(tableau)
            h_vec, _ = score_symplectic(tablea_op_applied)
            if h_vec < base_score:
                scores.append((op, h_vec))

    scores.sort(key=operator.itemgetter(1))
    return [(tv, score) for tv, score in scores]


def reduce_with_swaps(
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce a TERMINAL symplectic matrix by applying SWAPs to align blocks on diagonal.

    Args:
        tableau: A stabilizer tableau in terminal form (permutation matrix of 2x2 blocks).

    Returns:
        A tuple of (swap_sequence, tableau_after_swaps) where the blocks are now diagonal.
    """
    tableau_copy = tableau.copy()
    get_n(tableau)
    perm, _blocks = _extract_perm_in_to_out_and_blocks(tableau_copy)

    _perm_inverse(perm)
    swaps = _perm_to_swaps(perm)
    p = list(range(tableau.n))
    for swap in reversed(swaps):
        a, b = (swap.qubit_a, swap.qubit_b)
        p[a], p[b] = p[b], p[a]

    swap_sequence = EliminationSequence([])

    for swap in swaps:
        tableau_copy = swap.apply(tableau_copy, inplace=True)
        swap_sequence.add_operation(swap)
    return swap_sequence, tableau_copy


def reduce_with_single_qubit_cliffords_stateprep(
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce diagonal blocks to identity using single-qubit Cliffords and Paulis for state prep.

    Args:
        tableau: A stabilizer tableau where each qubit has a 2x2 block on its diagonal.

    Returns:
        A tuple of (clifford_sequence, final_tableau) where final_tableau should
        be identity.
    """
    tableau_copy = tableau.copy()
    n = get_n(tableau)

    clifford_sequence = EliminationSequence([])

    for row in range(tableau_copy.n_rows):
        for q in range(n):
            f = tableau_copy.tableau[row, q] + tableau_copy.tableau[row, q + n]
            if f < 2:
                continue
            op = SingleQubitClifford(q, "S")
            clifford_sequence.add_operation(op)
            tableau_copy = op.apply(tableau_copy, inplace=True)

    pauli_ops = fix_tableau_signs_in_place(tableau_copy)
    for op in pauli_ops:
        clifford_sequence.add_operation(op)
    return clifford_sequence, tableau_copy


def reduce_with_single_qubit_cliffords(
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce diagonal blocks to identity using single-qubit Cliffords and Paulis.

    Args:
        tableau: A stabilizer tableau where each qubit has a 2x2 block on its diagonal.

    Returns:
        A tuple of (clifford_sequence, final_tableau) where final_tableau should be identity.
    """
    tableau_copy = tableau.copy()
    n = get_n(tableau)

    clifford_sequence = EliminationSequence([])

    for q in range(n):
        f = tableau_copy.symplectic_submatrix(q)
        op = SingleQubitClifford.from_symplectic_block(f, q)
        clifford_sequence.add_operation(op)
        tableau_copy = op.apply(tableau_copy, inplace=True)

    pauli_ops = fix_tableau_signs_in_place(tableau_copy)
    for op in pauli_ops:
        clifford_sequence.add_operation(op)
    return clifford_sequence, tableau_copy


def reduce_singe_qubit_gates_stateprep(
    operations: EliminationSequence,
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce a TERMINAL symplectic matrix to identity using single-qubit gates for state prep.

    This function applies single-qubit Clifford reduction to bring a terminal-form tableau
    to the identity, suitable for state preparation.

    Args:
        operations: The elimination sequence (unused but required by post_process_fn signature).
        tableau: A stabilizer tableau in terminal form.

    Returns:
        A tuple of (operation_sequence, final_tableau) where final_tableau is identity.
    """
    clifford_seq, final_tableau = reduce_with_single_qubit_cliffords_stateprep(tableau)

    operations.extend(EliminationSequence(clifford_seq.operations))

    return operations, final_tableau


def reduce_single_qubit_gates_and_swaps(
    operations: EliminationSequence,
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce a TERMINAL symplectic matrix to identity using SWAP/H/S/Pauli gates.

    This function combines swap-based permutation correction with single-qubit Clifford
    reduction to bring a terminal-form tableau to the identity.

    Args:
        operations: The elimination sequence (unused but required by post_process_fn signature).
        tableau: A stabilizer tableau in terminal form.

    Returns:
        A tuple of (operation_sequence, final_tableau) where final_tableau is identity.
    """
    swap_seq, tableau_after_swaps = reduce_with_swaps(tableau)

    clifford_seq, final_tableau = reduce_with_single_qubit_cliffords(tableau_after_swaps)

    operations.extend(EliminationSequence(swap_seq.operations + clifford_seq.operations))

    return operations, final_tableau


def reduce_without_swaps(
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce a TERMINAL symplectic matrix to a permuted identity using only single-qubit gates.

    This variant does NOT apply SWAPs, so the final tableau will be a permutation of the
    identity (i.e., blocks aligned but possibly permuted).

    Args:
        tableau: A stabilizer tableau in terminal form.

    Returns:
        A tuple of (operation_sequence, final_tableau) where final_tableau is a
        permuted identity.
    """
    return reduce_with_single_qubit_cliffords(tableau)


def _extract_perm_in_to_out_and_blocks(tableau: StabilizerTableau) -> tuple[EliminationSequence, StabilizerTableau]:
    """Extract the permutation and corresponding 2×2 blocks from a terminal symplectic matrix.

    This function processes a terminal symplectic matrix `U` to determine the permutation
    of input qubits to output qubits and the associated 2×2 symplectic blocks.

    Args:
        U: A 2n×2n symplectic matrix in terminal form.

    Returns:
        A tuple containing:
        - perm: A 1D array where `perm[i]` gives the index `j` such that the determinant
          of the 2×2 block F_ij is 1 (indicating a valid symplectic transformation).
        - blocks: A list of 2×2 symplectic blocks corresponding to the permutation.
    """
    n = get_n(tableau)
    symplectic = tableau.tableau.matrix
    r2 = _compute_r2_matrix(symplectic)

    perm = np.full(n, -1, dtype=int)
    blocks: list[np.ndarray] = [None] * n

    for i in range(n):
        js = np.flatnonzero(r2[i])
        if len(js) != 1:
            msg = "Not terminal: R2 row is not one-hot."
            raise ValueError(msg)
        j = int(js[0])
        perm[i] = j
        blocks[i] = np.array(
            [
                [int(symplectic[i, j]), int(symplectic[i, j + n])],
                [int(symplectic[i + n, j]), int(symplectic[i + n, j + n])],
            ],
            dtype=np.int8,
        )

    if len(set(perm.tolist())) != n:
        msg = "Not terminal: R2 columns not one-hot."
        raise ValueError(msg)
    return perm, blocks


def _perm_inverse(perm_in_to_out: np.ndarray) -> np.ndarray:
    n = len(perm_in_to_out)
    inv = np.empty(n, dtype=int)
    for i, j in enumerate(perm_in_to_out):
        inv[int(j)] = i
    return inv


def _perm_to_swaps(perm_in_to_out: np.ndarray) -> list[Swap]:
    """Return a SWAP list that realizes perm_in_to_out when right-multiplying
    the symplectic matrix, i.e. permuting columns (wires).
    """
    n = len(perm_in_to_out)
    swaps: list[Swap] = []
    current = list(range(n))

    for target_idx in range(n):
        desired_wire = perm_in_to_out[target_idx]
        current_idx = current.index(desired_wire)

        if current_idx != target_idx:
            swaps.append(Swap(current_idx, target_idx))
            current[current_idx], current[target_idx] = current[target_idx], current[current_idx]

    return swaps


def fix_tableau_signs_in_place(tableau: StabilizerTableau) -> EliminationSequence:
    """Determine Pauli corrections so that the tableau matches the desired sign bits.

    This function ensures that the tableau matches the target signs
    by appending the necessary Pauli corrections.
    """
    n = get_n(tableau)
    x_part = tableau.tableau.matrix[:, :n]
    z_part = tableau.tableau.matrix[:, n:]

    phase = tableau.phase.copy()

    if not np.any(phase):
        return []

    tableau_with_phase = np.hstack((x_part, z_part, np.array([phase]).T))
    ker = mod2.nullspace(tableau_with_phase)
    assert ker[-1, -1] == 1, "Last entry of kernel vector must be 1."
    correction_symplectic = ker[-1]
    xc = correction_symplectic[:n]
    zc = correction_symplectic[n:-1]
    ops = []
    for i, (xv, zv) in enumerate(zip(xc, zc, strict=False)):
        if xv == 1 and zv == 1:
            op = PauliOperation(i, "Y")
        elif xv == 1:
            op = PauliOperation(i, "Z")
        elif zv == 1:
            op = PauliOperation(i, "X")
        else:
            continue
        ops.append(op)
        op.apply(tableau, inplace=True)

    return ops


def get_n(tableau: BinaryMatrix) -> int:
    """Get the number of qubits in the stabilizer tableau.

    Args:
        tableau (BinaryMatrix): The stabilizer tableau.

    Returns:
        int: The number of qubits.
    """
    if isinstance(tableau, StabilizerTableau):
        return tableau.n

    return tableau.matrix.shape[1]


def greedy_matrix_elimination_candidates(matrix: BinaryMatrix) -> list[CNOT]:
    """Get all possible CNOT candidates for a CSS check matrix.

    Args:
        matrix (BinaryMatrix): The CSS check matrix.

    Returns:
        list[CNOT]: A list of CNOT operations that can be applied.
    """
    matrix = matrix.copy()
    n = get_n(matrix)
    candidates: list[tuple[CNOT, int]] = []
    weight_before = int(matrix.matrix.sum())
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            op = CNOT(i, j)
            matrix = op.apply(matrix, inplace=True)
            weight_after = int(matrix.matrix.sum())
            candidates.append((op, weight_before - weight_after))
            matrix = op.apply(matrix, inplace=True)

    candidates.sort(key=operator.itemgetter(1), reverse=True)
    return [(op, score) for op, score in candidates]


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
