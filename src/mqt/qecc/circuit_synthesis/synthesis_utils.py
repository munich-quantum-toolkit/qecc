# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for synthesizing circuits."""

from __future__ import annotations

import functools
import logging
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import multiprocess
import numpy as np
import z3
from ldpc import mod2
from qiskit.circuit import AncillaRegister, ClassicalRegister, QuantumCircuit

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import numpy.typing as npt
    from qiskit.circuit import AncillaQubit, Clbit, Qubit

    from mqt.qecc.circuit_synthesis.state_prep import FaultyStatePrepCircuit
    from mqt.qecc.codes.css_code import CSSCode


logger = logging.getLogger(__name__)


def run_with_timeout(func: Callable[[Any], Any], *args: Any, timeout: int = 10) -> Any | str | None:  # noqa: ANN401
    """Run a function with a timeout.

    If the function does not complete within the timeout, return None.

    Args:
        func: The function to run.
        args: The arguments to pass to the function.
        timeout: The maximum time to allow the function to run for in seconds.
    """
    manager = multiprocess.Manager()
    return_list = manager.list()
    p = multiprocess.Process(target=lambda: return_list.append(func(*args)))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        return "timeout"
    return return_list[0]


def iterative_search_with_timeout(
    fun: Callable[[int], QuantumCircuit],
    min_param: int,
    max_param: int,
    min_timeout: int,
    max_timeout: int,
    param_factor: float = 2,
    timeout_factor: float = 2,
) -> tuple[QuantumCircuit | None, int] | None:
    """Geometrically increases the parameter and timeout until a result is found or the maximum timeout is reached.

    Args:
        fun: function to run with increasing parameters and timeouts
        min_param: minimum parameter to start with
        max_param: maximum parameter to reach
        min_timeout: minimum timeout to start with
        max_timeout: maximum timeout to reach
        param_factor: factor to increase the parameter by at each iteration
        timeout_factor: factor to increase the timeout by at each iteration
    """
    curr_timeout = min_timeout
    curr_param = min_param
    while curr_timeout <= max_timeout:
        while curr_param <= max_param:
            logger.info(f"Running iterative search with param={curr_param} and timeout={curr_timeout}")
            res = run_with_timeout(fun, curr_param, timeout=curr_timeout)
            if res is not None and (not isinstance(res, str) or res != "timeout"):
                return res, curr_param
            if curr_param == max_param:
                break

            curr_param = int(curr_param * param_factor)
            curr_param = min(curr_param, max_param)

        curr_timeout = int(curr_timeout * timeout_factor)
        curr_param = min_param
    return None, max_param


class CandidateAction(Enum):
    """Class to help distinguish the control flow of the reference based heuristic gaussian elimination."""

    SKIP = auto()  # Ignore this candidate and continue the loop
    RESTART_SEARCH = auto()  # Reset used_columns and break to restart the search
    TRIGGER_BACKTRACK = auto()  # Call backtrack() and break
    EVALUATE = auto()  # Proceed with full validation (overlap checks)


class EliminationCNOTSynthesizer:
    """Synthesizes a CNOT circuit through check matrix reduction via eliminations.

    This class implements algorithms to find a sequence of CNOT gates that
    transforms a given binary matrix (e.g., a stabilizer generator matrix
    from a quantum error-correcting code) into a reduced row echelon form.
    The core mechanism is based on Gaussian elimination, where CNOT gates
    correspond to column operations.

    Two primary synthesis strategies are provided:
    1.  `greedy_synthesis`: A fast, purely cost-based algorithm that always
        chooses the CNOT gate that reduces the column weights the most.
    2.  `fault_set_guided_synthesis`: A more advanced algorithm that uses a
        backtracking search guided by reference fault sets. This ensures the
        synthesized circuit adheres to specific fault-tolerance constraints,
        with respect to the reference fault sets given.

    """

    def __init__(
        self,
        matrix: npt.NDArray[np.int8],
        code: CSSCode,
        parallel_elimination: bool = True,
    ) -> None:
        """Initialiser for the basic functionality."""
        self.matrix = matrix.copy()
        self.parallel_elimination = parallel_elimination
        self.code = code
        self.rank = mod2.rank(self.matrix)
        self.eliminations: list[tuple[int, int]] = []
        self.used_columns: list[int] = []
        self.costs = self._compute_cost_matrix()

    def _ref_based_init(
        self,
        ref_x_fs: npt.NDArray[np.int8] | None = None,
        ref_z_fs: npt.NDArray[np.int8] | None = None,
        ref_x_1fs: npt.NDArray[np.int8] | None = None,
        ref_z_1fs: npt.NDArray[np.int8] | None = None,
        penalty_cnots: list[tuple[int, int]] | None = None,
        guide_by_x: bool = True,
    ) -> None:
        def _default_array(arr: npt.NDArray[np.int8] | None) -> npt.NDArray[np.int8]:
            return arr if arr is not None else np.empty((0,), dtype=np.int8)

        self.ref_x_fs = _default_array(ref_x_fs)
        self.ref_z_fs = _default_array(ref_z_fs)
        self.ref_x_1fs = _default_array(ref_x_1fs)
        self.ref_z_1fs = _default_array(ref_z_1fs)
        self.guide_by_x = guide_by_x
        self.failed_cnots: list[tuple[int, int]] = (
            penalty_cnots or []
        )  # NOTE: this is already a feature and not necessarily default
        self.x_propagation_matrix: npt.NDArray[np.int8] = np.eye(self.matrix.shape[1], dtype=np.int8)
        self.z_propagation_matrix: npt.NDArray[np.int8] = np.eye(self.matrix.shape[1], dtype=np.int8)
        self.backtrack_required: bool = False
        self.overlapping_errors_x: set[npt.NDArray[np.int8]] = set()
        self.overlapping_errors_z: set[npt.NDArray[np.int8]] = set()
        self.stack: list[
            tuple[
                npt.NDArray[np.int8],
                npt.NDArray[np.int8],
                list[int],
                npt.NDArray[np.int8],
                npt.NDArray[np.int8],
                list[tuple[int, int]],
                list[tuple[int, int]],
            ]
        ] = []
        self.current_x_fs: npt.NDArray[np.int8] = np.eye(self.matrix.shape[1], dtype=np.int8)
        self.current_z_fs: npt.NDArray[np.int8] = np.eye(self.matrix.shape[1], dtype=np.int8)
        self.used_cnots: list[tuple[int, int]] = []

    def greedy_synthesis(self) -> None:
        """Basic heuristic Gaussian elimination.

        Calculates CNOTS and Check matrix.
        """
        while not self.is_reduced():
            costs_unused = self._mask_out_used_qubits()
            if self._reset_if_stuck(costs_unused):
                continue
            i, j = np.unravel_index(np.argmin(costs_unused), self.costs.shape)
            self._apply_cnot_to_matrix(int(i), int(j))

    def fault_set_guided_synthesis(
        self,
        ref_x_fs: npt.NDArray[np.int8] | None = None,
        ref_z_fs: npt.NDArray[np.int8] | None = None,
        ref_x_1fs: npt.NDArray[np.int8] | None = None,
        ref_z_1fs: npt.NDArray[np.int8] | None = None,
        penalty_cnots: list[tuple[int, int]] | None = None,
        guide_by_x: bool = True,
    ) -> None:
        """Reference based heuristic Gaussian elimination.

        This version takes reference fault sets into consideration and only applies cnots that
        do not cause an overlap of the reference fault set and the fault set of the newly
        created circuit.
        """
        self._validate_inputs()
        self._ref_based_init(
            ref_x_fs=ref_x_fs,
            ref_z_fs=ref_z_fs,
            ref_x_1fs=ref_x_1fs,
            ref_z_1fs=ref_z_1fs,
            penalty_cnots=penalty_cnots,
            guide_by_x=guide_by_x,
        )
        while not self.is_reduced():
            costs_unused = self._mask_out_used_qubits()
            if self._reset_if_stuck(costs_unused):
                continue
            candidate_pairs = self._get_candidate_pairs(costs_unused)
            for i, j in candidate_pairs:
                action = self._get_candidate_action(i, j, costs_unused[i, j])
                if action == CandidateAction.SKIP:
                    continue
                if action == CandidateAction.RESTART_SEARCH:
                    self.used_columns = []
                    self.backtrack_required = True
                    break

                if action == CandidateAction.TRIGGER_BACKTRACK:
                    if self.stack:
                        self._backtrack()
                    else:
                        # This case means we backtracked to the beginning and still failed.
                        logger.error("Backtracking stack is empty. No solution found.")
                    break

                if action == CandidateAction.EVALUATE:
                    is_valid, new_errors = self._cnot_is_valid_against_references(i, j)
                    if is_valid:
                        self.used_cnots.append((int(i), int(j)))
                        self._commit_to_cnot(i, j, new_errors)
                        self._apply_cnot_to_matrix(int(i), int(j))
                        break
                    self.failed_cnots.append((int(i), int(j)))
                    continue

    def is_reduced(self) -> bool:
        """Method decides if the Gaussian elimination has successfully ended.

        The matrix is considered reduced when the number of columns actually turned into
        all-zeros matches the number of linearily depended columns.
        """
        return bool(len(np.where(np.all(self.matrix == 0, axis=0))[0]) == self.matrix.shape[1] - self.rank)

    def _get_candidate_action(self, i: int, j: int, costs: int) -> CandidateAction:
        if (int(i), int(j)) in self.failed_cnots or (int(i), int(j)) in self.used_cnots:
            return CandidateAction.SKIP

        if costs >= 0:
            if self.backtrack_required:
                return CandidateAction.TRIGGER_BACKTRACK
            return CandidateAction.RESTART_SEARCH
        return CandidateAction.EVALUATE

    def _cnot_is_valid_against_references(self, i: int, j: int) -> tuple[bool, dict[str, npt.NDArray[np.int8]]]:
        found_cnot: bool = False
        new_x_error, _next_x_propagation_matrix = get_next_error(
            propagation_matrix=self.x_propagation_matrix.copy(), cnot_gate=(int(i), int(j))
        )
        new_z_error, _next_z_propagation_matrix = get_next_error(
            propagation_matrix=self.z_propagation_matrix.copy(), cnot_gate=(int(i), int(j)), x_error=False
        )

        if self._check_error_existence(new_x_error, self.overlapping_errors_x):
            return found_cnot, {}
        if self._check_error_existence(new_z_error, self.overlapping_errors_z):
            return found_cnot, {}

        new_x_error = np.expand_dims(new_x_error, axis=0)
        new_z_error = np.expand_dims(new_z_error, axis=0)

        new_z_error_tuple = tuple(new_z_error.flatten())
        new_x_error_tuple = tuple(new_x_error.flatten())

        x_overlap = self._check_overlap(self.ref_x_fs, self.ref_x_1fs, self.current_x_fs, new_x_error)
        z_overlap = self._check_overlap(self.ref_z_fs, self.ref_z_1fs, self.current_z_fs, new_z_error, x_error=False)

        if self.ref_x_fs.size and self.ref_z_fs.size:
            # Both sets provided: Enter if **both** overlaps fail
            found_cnot = not (x_overlap or z_overlap)
            if not found_cnot:
                self.overlapping_errors_x.add(new_x_error_tuple)
                self.overlapping_errors_z.add(new_z_error_tuple)
        elif self.ref_x_fs.size:
            # Only x-set provided: Enter if **x_overlap** fails
            found_cnot = not x_overlap
            if not found_cnot:
                self.overlapping_errors_x.add(new_x_error_tuple)
        elif self.ref_z_fs.size:
            # Only z-set provided: Enter if **z_overlap** fails
            found_cnot = not z_overlap
            if not found_cnot:
                self.overlapping_errors_z.add(new_z_error_tuple)
        return found_cnot, {
            "new_x_error": new_x_error,
            "new_z_error": new_z_error,
            "x_overlap": x_overlap,
            "z_overlap": z_overlap,
        }

    def _commit_to_cnot(self, i: int, j: int, new_errors: dict[str, npt.NDArray[np.int8]]) -> None:
        self.stack.append((
            self.x_propagation_matrix.copy(),
            self.z_propagation_matrix.copy(),
            self.used_columns.copy(),
            self.matrix.copy(),
            self.costs.copy(),
            self.used_cnots.copy(),
            self.failed_cnots.copy(),
        ))
        if self.guide_by_x:
            self.failed_cnots = [fcnot for fcnot in self.failed_cnots if int(i) != fcnot[0]]
        else:
            self.failed_cnots = [fcnot for fcnot in self.failed_cnots if int(j) != fcnot[1]]

        if self.ref_x_fs.size and not new_errors["x_overlap"]:
            self.current_x_fs = np.vstack((self.current_x_fs, new_errors["new_x_error"]), dtype=np.int8)
        if self.ref_z_fs.size and not new_errors["z_overlap"]:
            self.current_z_fs = np.vstack((self.current_z_fs, new_errors["new_z_error"]), dtype=np.int8)

    def _backtrack(self) -> None:
        (
            self.x_propagation_matrix,
            self.z_propagation_matrix,
            self.used_columns,
            self.matrix,
            self.costs,
            self.used_cnots,
            _,
        ) = self.stack.pop()
        removed_cnot = self.eliminations.pop()
        if self.guide_by_x:
            self.failed_cnots = [fcnot for fcnot in self.failed_cnots if removed_cnot[0] != fcnot[0]]
        else:
            self.failed_cnots = [fcnot for fcnot in self.failed_cnots if removed_cnot[1] != fcnot[1]]
        self.failed_cnots.append(removed_cnot)
        # print_dynamic_eliminations(eliminations, failed_cnots)
        self.backtrack_required = False

    def _validate_inputs(self) -> None:
        """Here are some checks that ensure that the reference based construction can actually be executed."""
        has_refs = bool(self.ref_x_fs.size or self.ref_z_fs.size)
        has_code = self.code is not None
        if has_refs != has_code:
            msg = "Must provide either both a reference fault set and CSS code, or neither."
            raise ValueError(msg)

    def _reset_if_stuck(self, costs_unused: npt.NDArray[np.int8]) -> bool:
        """Handles local minima or full column usage. Returns True if reset occurred."""
        if np.all(costs_unused >= 0) or len(self.used_columns) == self.matrix.shape[1]:
            if not self.used_columns:  # Local minimum
                self._triangular_reset()
            else:
                self.used_columns = []
            return True
        return False

    def _triangular_reset(self) -> None:
        logger.warning("Local minimum reached. Making matrix triangular.")
        self.matrix = mod2.row_echelon(self.matrix, full=True)[0]
        self.costs = self._compute_cost_matrix()

    def _apply_cnot_to_matrix(self, i: int, j: int) -> None:
        self.eliminations.append((i, j))
        if self.parallel_elimination:
            self.used_columns.append(i)
            self.used_columns.append(j)
        # update matrix
        self.matrix[:, j] = (self.matrix[:, i] + self.matrix[:, j]) % 2
        # update costs
        new_weights = np.sum((self.matrix[:, j][:, np.newaxis] + self.matrix) % 2, axis=0)
        self.costs[j, :] = new_weights - np.sum(self.matrix, axis=0)
        self.costs[:, j] = new_weights - np.sum(self.matrix[:, j])
        np.fill_diagonal(self.costs, 1)

    def _compute_cost_matrix(self) -> npt.NDArray[np.int8]:
        costs = np.array([
            [np.sum((self.matrix[:, i] + self.matrix[:, j]) % 2) for j in range(self.matrix.shape[1])]
            for i in range(self.matrix.shape[1])
        ])
        costs -= np.sum(self.matrix, axis=0)
        np.fill_diagonal(costs, 1)
        return costs

    def _mask_out_used_qubits(self) -> npt.NDArray[np.int8]:
        m = np.zeros((self.matrix.shape[1], self.matrix.shape[1]), dtype=bool)  # type: npt.NDArray[np.bool_]
        m[self.used_columns, :] = True
        m[:, self.used_columns] = True
        return np.ma.array(self.costs, mask=m)  # type: ignore[no-untyped-call]

    def _get_candidate_pairs(self, costs_unused: npt.NDArray[np.int8]) -> list[tuple[int, int]]:
        # Get all valid (i, j) pairs sorted by cost
        candidate_indices = np.argsort(costs_unused.flatten())  # Flatten and sort by value
        return [(int(i), int(j)) for idx in candidate_indices for i, j in [np.unravel_index(idx, self.costs.shape)]]
        # return [np.unravel_index(idx, self.costs.shape) for idx in candidate_indices]

    def _check_overlap(
        self,
        ref_fs: npt.NDArray[np.int8],
        ref_1_fs: npt.NDArray[np.int8],
        current_fs: npt.NDArray[np.int8],
        new_error: npt.NDArray[np.int8],
        x_error: bool = True,
    ) -> bool:
        overlap: bool = True
        if ref_fs.size:
            overlap = self.code.check_fs_overlap(ref_fs, new_error, x_error=x_error)
            if ref_1_fs.size and not overlap:
                trial_2_fs = (current_fs + new_error) % 2
                trial_2_fs = self._filter_fault_set(trial_2_fs, int((self.code.distance - 1) / 2))
                overlap = self.code.check_fs_overlap(ref_1_fs, trial_2_fs, x_error=x_error)
        return overlap

    @staticmethod
    def _check_error_existence(error: npt.NDArray[np.int8], error_memory: set[npt.NDArray[np.int8]]) -> bool:
        # INFO: Quickly check if new error is known to cause overlap. This saves another long overlap check.
        error_tuple = tuple(error.flatten())
        return error_tuple in error_memory

    @staticmethod
    def _filter_fault_set(fault_set: npt.NDArray[np.int8], threshold: int) -> npt.NDArray[np.int8]:
        """Filter fault sets based on a sum threshold."""
        return fault_set[np.where(fault_set.sum(axis=1) > threshold)]


def get_next_error(
    propagation_matrix: npt.NDArray[np.int8], cnot_gate: tuple[int, int], x_error: bool = True
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """Propagates a single X or Z error through a CNOT gate in the circuit.

    This function updates the error propagation matrix when a new CNOT gate is applied.
    Each row in the matrix represents the propagated error if a single X or Z error occurs on that qubit.

    Args:
        propagation_matrix (npt.NDArray[np.int8]):
            Current error propagation matrix (shape: [n_qubits, n_qubits]).
        cnot_gate (Tuple[int, int]):
            A tuple representing the new CNOT gate (control, target).
        x_error (bool, optional):
            Flag indicating whether to propagate X errors (True) or Z errors (False).
            Defaults to True.

    Returns:
        Tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
            - The propagated error (affected row of the matrix).
            - The updated error propagation matrix.

    """
    # NOTE: This implementation assumes single error events only. For higher-order errors, adjustments are required.
    control, target = cnot_gate

    control_row = propagation_matrix[control]
    target_row = propagation_matrix[target]

    if x_error:
        propagation_matrix[control] = np.bitwise_xor(control_row, target_row)
        return propagation_matrix[control], propagation_matrix

    propagation_matrix[target] = np.bitwise_xor(control_row, target_row)
    return propagation_matrix[target], propagation_matrix


def check_mutually_disjointness_spcs(
    c_spcs: list[FaultyStatePrepCircuit], p_spcs: list[FaultyStatePrepCircuit], x_error: bool = True
) -> list[FaultyStatePrepCircuit]:
    """Check if potential SPCs have mutually disjoint fault set with all current SPCs.

    Take list of SPCs with already mutually disjoint fault sets and another list of potential candidates for new SPCs that
    have a mutually disjoint fault set with all current SPCs.

    Args:
        c_spcs: current SPCs that are already mutually disjoint in terms of fault sets
        p_spcs: potential SPCs that shall be tested for mutually disjointness against the current set
        x_error: flag that decides if check happens for x errors or z errors
    """
    i = 0
    while i < len(p_spcs):
        pspc = p_spcs[i]
        # NOTE: check if it is not better to use the big fault set of the current SPC here as those sets have already been
        # calculated and do not need to be calculated again
        # FIX: big fs calculation should be only done for the current elements if there are more candidates than current elements
        px_fs, pz_fs = get_fs_based_on_d(pspc)

        # Check against *all* existing and newly added elements in c_spcs
        for _spc in c_spcs:
            # NOTE: for distance 5 and 7 we only compare against the 1 error fault set.
            # This might need adjustments for different distances.
            x_fs = _spc.compute_fault_set()
            z_fs = _spc.compute_fault_set(x_errors=False)
            if x_error:
                if _spc.code.check_fs_overlap(x_fs, px_fs):
                    break
            elif _spc.code.check_fs_overlap(z_fs, pz_fs, x_error=False):
                break  # Stop checking if there's an overlap
        else:  # No overlap found, so add pspc
            c_spcs.append(pspc)

        i += 1  # Move to next element in p_spcs
    return c_spcs


def get_fs_based_on_d(spc: FaultyStatePrepCircuit) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """Get fault sets based on the code distance.

    Only the faults sets that are of interest for the fault tolerant state preparation are being returned.
    Supported are code distance 5 and 7.

    Args:
        spc: State Prep Circiut of which the fault set is to be returned.

    Returns:
        tuple: first entry the x fault set and second entry the z fault set.
    """

    def compute_fault_sets(level: int = 1) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
        """Helper function to compute fault sets at a given level."""
        spc.compute_fault_set(level)
        spc.compute_fault_set(level, x_errors=False)
        return spc.x_fault_sets[level], spc.z_fault_sets[level]

    def filter_fault_set(fault_set: npt.NDArray[np.int8], threshold: int) -> npt.NDArray[np.int8]:
        """Filter fault sets based on a sum threshold."""
        return fault_set[np.where(fault_set.sum(axis=1) > threshold)]

    fault_set_x_1, fault_set_z_1 = compute_fault_sets(1)

    if spc.code.distance == 3:
        return fault_set_x_1, fault_set_z_1

    if spc.code.distance == 5:
        return filter_fault_set(fault_set_x_1, 2), filter_fault_set(fault_set_z_1, 2)

    if spc.code.distance == 7:
        fault_set_x_2, fault_set_z_2 = compute_fault_sets(2)
        return (
            np.vstack([
                filter_fault_set(fault_set_x_1, 2),
                filter_fault_set(fault_set_x_2, 3),
            ]),
            np.vstack([
                filter_fault_set(fault_set_z_1, 2),
                filter_fault_set(fault_set_z_2, 3),
            ]),
        )

    msg = f"Unsupported code distance: {spc.code.distance}"
    raise ValueError(msg)


def gaussian_elimination_min_column_ops(
    matrix: npt.NDArray[np.int8],
    termination_criteria: Callable[[Any], z3.BoolRef],
    max_eliminations: int,
) -> tuple[npt.NDArray[np.int8], list[tuple[int, int]]] | None:
    """Perform Gaussian elimination on the column space of a matrix using at most `max_eliminations` eliminations.

    The algorithm encodes the elimination into an SMT problem and uses Z3 to find the optimal solution.

    Args:
        matrix: The matrix to perform Gaussian elimination on.
        termination_criteria: A function that takes a boolean matrix as input and returns a Z3 boolean expression that is true if the matrix is considered reduced.
        max_eliminations: The maximum number of eliminations to perform.

    Returns:
        The reduced matrix and a list of the elimination steps taken. The elimination steps are represented as tuples of the form (i, j) where i is the column being eliminated with and j is the column being eliminated.
    """
    n = matrix.shape[1]
    columns = np.array([
        [[z3.Bool(f"x_{d}_{i}_{j}") for j in range(n)] for i in range(matrix.shape[0])]
        for d in range(max_eliminations + 1)
    ])

    n_bits = int(np.ceil(np.log2(n)))
    targets = [z3.BitVec(f"target_{d}", n_bits) for d in range(max_eliminations)]
    controls = [z3.BitVec(f"control_{d}", n_bits) for d in range(max_eliminations)]
    s = z3.Solver()

    additions = np.array([
        [[z3.And(controls[d] == col_1, targets[d] == col_2) for col_2 in range(n)] for col_1 in range(n)]
        for d in range(max_eliminations)
    ])

    # create initial matrix
    columns[0, :, :] = matrix.astype(bool)

    if max_eliminations != 0:
        s.add(_column_addition_constraint(columns, additions))

        for d in range(1, max_eliminations + 1):
            # two columns cannot be in two elimination steps at the same time
            s.add(controls[d - 1] != targets[d - 1])

            # control and target must be valid qubits

            if n and (n - 1) != 0 and not ((n & (n - 1) == 0) and n != 0):  # check if n is a power of 2 or 1 or 0
                s.add(z3.ULT(controls[d - 1], n))
                s.add(z3.ULT(targets[d - 1], n))

        # if column is not involved in any addition at certain depth, it is the same as the previous column
        for d in range(1, max_eliminations + 1):
            for col in range(n):
                s.add(z3.Implies(targets[d - 1] != col, symbolic_vector_eq(columns[d, :, col], columns[d - 1, :, col])))

    # assert that final check matrix has n-checks.shape[0] zero columns
    s.add(termination_criteria(columns))

    if s.check() == z3.sat:
        if max_eliminations == 0:
            return matrix, []

        m = s.model()
        eliminations = [(m[controls[d]].as_long(), m[targets[d]].as_long()) for d in range(max_eliminations)]
        reduced = np.array([
            [bool(m[columns[max_eliminations][i][j]]) for j in range(n)] for i in range(matrix.shape[0])
        ]).astype(np.int8)  # type: npt.NDArray[np.int8]
        return reduced, eliminations

    return None


def gaussian_elimination_min_parallel_eliminations(
    matrix: npt.NDArray[np.int8], termination_criteria: Callable[[Any], z3.BoolRef], max_parallel_steps: int
) -> tuple[npt.NDArray[np.int8], list[tuple[int, int]]] | None:
    """Perform Gaussian elimination on the column space of a matrix using at most `max_parallel_steps` parallel column elimination steps.

    The algorithm encodes the elimination into a SAT problem and uses Z3 to find the optimal solution.

    Args:
        matrix: The matrix to perform Gaussian elimination on.
        termination_criteria: A function that takes a boolean matrix as input and returns a Z3 boolean expression that is true if the matrix is considered reduced.
        max_parallel_steps: The maximum number of parallel elimination steps to perform.

    Returns:
        The reduced matrix and a list of the elimination steps taken. The elimination steps are represented as tuples of the form (i, j) where i is the column being eliminated with and j is the column being eliminated.
    """
    columns = np.array([
        [[z3.Bool(f"x_{d}_{i}_{j}") for j in range(matrix.shape[1])] for i in range(matrix.shape[0])]
        for d in range(max_parallel_steps + 1)
    ])

    additions = np.array([
        [[z3.Bool(f"add_{d}_{i}_{j}") for j in range(matrix.shape[1])] for i in range(matrix.shape[1])]
        for d in range(max_parallel_steps)
    ])
    n_cols = matrix.shape[1]
    s = z3.Solver()

    # create initial matrix
    columns[0, :, :] = matrix.astype(bool)

    if max_parallel_steps != 0:
        s.add(_column_addition_constraint(columns, additions))

        # qubit can be involved in at most one addition at each depth
        for d in range(max_parallel_steps):
            for col in range(n_cols):
                s.add(
                    z3.PbLe(
                        [(additions[d, col_1, col], 1) for col_1 in range(n_cols) if col != col_1]
                        + [(additions[d, col, col_2], 1) for col_2 in range(n_cols) if col != col_2],
                        1,
                    )
                )

        # if column is not involved in any addition at certain depth, it is the same as the previous column
        for d in range(1, max_parallel_steps + 1):
            for col in range(n_cols):
                s.add(
                    z3.Implies(
                        z3.Not(
                            z3.Or(
                                list(np.delete(additions[d - 1, :, col], [col]))
                                + list(np.delete(additions[d - 1, col, :], [col]))
                            )
                        ),
                        symbolic_vector_eq(columns[d, :, col], columns[d - 1, :, col]),
                    )
                )

    s.add(termination_criteria(columns))

    if s.check() == z3.sat:
        if max_parallel_steps == 0:
            return matrix, []
        m = s.model()
        eliminations = [
            (i, j)
            for d in range(max_parallel_steps)
            for j in range(matrix.shape[1])
            for i in range(matrix.shape[1])
            if m[additions[d, i, j]]
        ]
        reduced = np.array([
            [bool(m[columns[max_parallel_steps, i, j]]) for j in range(matrix.shape[1])] for i in range(matrix.shape[0])
        ]).astype(np.int8)  # type: npt.NDArray[np.int8]
        return reduced, eliminations

    return None


def build_css_circuit_from_cnot_list(n: int, cnots: list[tuple[int, int]], hadamards: list[int]) -> QuantumCircuit:
    """Build a quantum circuit consisting of Hadamards followed by a layer of CNOTs from a list of CNOTs and a list of checks.

    Args:
        n: Number of qubits in the circuit.
        cnots: List of CNOTs to apply. Each CNOT is a tuple of the form (control, target).
        hadamards: List of qubits to apply Hadamards to.

    Returns:
        The quantum circuit.
    """
    circ = QuantumCircuit(n)
    if hadamards:
        circ.h(hadamards)
    for i, j in cnots:
        circ.cx(i, j)
    return circ


def _column_addition_constraint(
    columns: npt.NDArray[z3.BoolRef | bool],
    col_add_vars: npt.NDArray[z3.BoolRef],
) -> z3.BoolRef:
    assert len(columns.shape) == 3
    max_parallel_steps = col_add_vars.shape[0]
    n_cols = col_add_vars.shape[2]

    constraints = []
    for d in range(1, max_parallel_steps + 1):
        for col_1 in range(n_cols):
            for col_2 in range(col_1 + 1, n_cols):
                col_sum = symbolic_vector_add(columns[d - 1, :, col_1], columns[d - 1, :, col_2])

                # encode col_2 += col_1
                add_col1_to_col2 = z3.Implies(
                    col_add_vars[d - 1, col_1, col_2],
                    z3.And(
                        symbolic_vector_eq(columns[d, :, col_2], col_sum),
                        symbolic_vector_eq(columns[d, :, col_1], columns[d - 1, :, col_1]),
                    ),
                )

                # encode col_1 += col_2
                add_col2_to_col1 = z3.Implies(
                    col_add_vars[d - 1, col_2, col_1],
                    z3.And(
                        symbolic_vector_eq(columns[d, :, col_1], col_sum),
                        symbolic_vector_eq(columns[d, :, col_2], columns[d - 1, :, col_2]),
                    ),
                )

                constraints.extend([add_col1_to_col2, add_col2_to_col1])

    return z3.And(constraints)


def symbolic_vector_eq(v1: npt.NDArray[z3.BoolRef | bool], v2: npt.NDArray[z3.BoolRef | bool]) -> z3.BoolRef:
    """Return assertion that two symbolic vectors should be equal."""
    if len(v1) != len(v2):
        msg = "Vectors must have the same length for equality check."
        raise ValueError(msg)

    # map all numpy bools to Python bools, otherwise z3 will not be able to handle them
    v1 = np.array([bool(v) if isinstance(v, (bool, np.bool_)) else v for v in v1], dtype=object)
    v2 = np.array([bool(v) if isinstance(v, (bool, np.bool_)) else v for v in v2], dtype=object)

    constraints = [False for _ in v1]
    for i in range(len(v1)):
        # If one of the elements is a bool, we can simplify the expression
        v1_i_is_bool = isinstance(v1[i], (bool, np.bool_))
        v2_i_is_bool = isinstance(v2[i], (bool, np.bool_))
        if v1_i_is_bool:
            v1[i] = bool(v1[i])
            if v1[i]:
                constraints[i] = v2[i]
            else:
                constraints[i] = z3.Not(v2[i]) if not v2_i_is_bool else not v2[i]

        elif v2_i_is_bool:
            v2[i] = bool(v2[i])
            if v2[i]:
                constraints[i] = v1[i]
            else:
                constraints[i] = z3.Not(v1[i])
        else:
            constraints[i] = v1[i] == v2[i]
    return z3.And(constraints)


def odd_overlap(v_sym: npt.NDArray[z3.BoolRef | bool], v_con: npt.NDArray[np.int8]) -> z3.BoolRef:
    """Return True if the overlap of symbolic vector with constant vector is odd."""
    if np.array_equal(v_con, np.zeros(len(v_con), dtype=np.int8)):
        return z3.BoolVal(False)

    constraint = False
    for i, c in enumerate(v_con):
        if c != 1:
            continue
        constraint = z3.Xor(constraint, v_sym[i])
    return constraint


def symbolic_scalar_mult(v: npt.NDArray[np.int8], a: z3.BoolRef | bool) -> npt.NDArray[z3.BoolRef]:
    """Multiply a concrete vector by a symbolic scalar."""
    return np.array([a if s == 1 else False for s in v])


def symbolic_vector_add(
    v1: npt.NDArray[z3.BoolRef | bool], v2: npt.NDArray[z3.BoolRef | bool]
) -> npt.NDArray[z3.BoolRef | bool]:
    """Add two symbolic vectors."""
    v_new = [False for _ in range(len(v1))]
    for i in range(len(v1)):
        # If one of the elements is a bool, we can simplify the expression
        v1_i_is_bool = isinstance(v1[i], (bool, np.bool_))
        v2_i_is_bool = isinstance(v2[i], (bool, np.bool_))
        if v1_i_is_bool:
            v1[i] = bool(v1[i])
            if v1[i]:
                v_new[i] = z3.Not(v2[i]) if not v2_i_is_bool else not v2[i]
            else:
                v_new[i] = v2[i]

        elif v2_i_is_bool:
            v2[i] = bool(v2[i])
            if v2[i]:
                v_new[i] = z3.Not(v1[i])
            else:
                v_new[i] = v1[i]

        elif bool(v1[i] == v2[i]):
            v_new[i] = False
        else:
            v_new[i] = z3.Xor(v1[i], v2[i])

    return np.array(v_new)


def optimal_elimination(
    matrix: npt.NDArray[np.int8],
    termination_criteria: Callable[[Any], z3.BoolRef],
    optimization_metric: str = "column_ops",
    min_param: int = 1,
    max_param: int = 10,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> tuple[npt.NDArray[np.int8], list[tuple[int, int]]] | None:
    """Synthesize a state preparation circuit for a CSS code that minimizes the circuit w.r.t. some metric param according to prep_func.

    Args:
        matrix: The stabilizer matrix of the CSS code.
        termination_criteria: The termination criteria for when the matrix is considered reduced.
        optimization_metric: The metric to optimize the circuit w.r.t. to. Can be either "column_ops" or "parallel_ops".
        zero_state: Whether to start from the zero state.
        min_param: The minimum value of the metric parameter.
        max_param: The maximum value of the metric parameter.
        min_timeout: The minimum time to run one search iteration for.
        max_timeout: The maximum time to run one search iteration for.
    """
    if optimization_metric not in {"column_ops", "parallel_ops"}:
        msg = "Invalid optimization metric"
        raise ValueError(msg)

    opt_fun = {
        "column_ops": gaussian_elimination_min_column_ops,
        "parallel_ops": gaussian_elimination_min_parallel_eliminations,
    }[optimization_metric]

    fun = functools.partial(
        opt_fun,
        matrix,
        termination_criteria,
    )

    res = iterative_search_with_timeout(
        fun,
        min_param,
        max_param,
        min_timeout,
        max_timeout,
    )

    if res is None:
        return None
    reduced = res[0]
    if reduced is None:
        return None
    reduced, eliminations = reduced
    curr_param = res[1]

    logger.info(f"Solution found with param {curr_param}")
    # Solving a SAT instance is much faster than proving unsat in this case
    # so we iterate backwards until we find an unsat instance or hit a timeout
    logger.info("Trying to minimize param")
    while True:
        logger.info(f"Trying param {curr_param - 1}")
        opt_res = run_with_timeout(fun, curr_param - 1, timeout=max_timeout)
        if opt_res is None or (isinstance(opt_res, str) and opt_res == "timeout"):
            break
        assert not isinstance(opt_res, str)
        reduced, eliminations = opt_res
        curr_param -= 1

    logger.info(f"Optimal param: {curr_param}")
    return reduced, eliminations


def _ancilla_cnot(qc: QuantumCircuit, qubit: Qubit | AncillaQubit, ancilla: AncillaQubit, z_measurement: bool) -> None:
    if z_measurement:
        qc.cx(qubit, ancilla)
    else:
        qc.cx(ancilla, qubit)


def _flag_measure(qc: QuantumCircuit, flag: AncillaQubit, meas_bit: Clbit, z_measurement: bool) -> None:
    if z_measurement:
        qc.h(flag)
    qc.measure(flag, meas_bit)


def _flag_reset(qc: QuantumCircuit, flag: AncillaQubit, z_measurement: bool) -> None:
    qc.reset(flag)
    if z_measurement:
        qc.h(flag)


def _flag_init(qc: QuantumCircuit, flag: AncillaQubit, z_measurement: bool) -> None:
    if z_measurement:
        qc.h(flag)


def measure_stab_unflagged(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
) -> None:
    """Measure a stabilizer without flags. The measurement is done in place.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: The qubits to measure.
        ancilla: The ancilla qubit to use for the measurement.
        measurement_bit: The classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
    """
    if not z_measurement:
        qc.h(ancilla)
        qc.cx([ancilla] * len(stab), stab)
        qc.h(ancilla)
    else:
        qc.cx(stab, [ancilla] * len(stab))
    qc.measure(ancilla, measurement_bit)


def measure_flagged(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    t: int,
    z_measurement: bool = True,
) -> None:
    """Measure a w-flagged stabilizer.

    The measurement is done in place.

    Args:
        Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        t: The number of errors to protect against.
        z_measurement: Whether to measure the ancilla in the Z basis.
    """
    w = len(stab)
    if w < 3:
        measure_stab_unflagged(qc, stab, ancilla, measurement_bit, z_measurement)
        return

    if t == 1:
        measure_one_flagged(qc, stab, ancilla, measurement_bit, z_measurement)
        return

    if w == 4 and t >= 2:
        measure_two_flagged_4(qc, stab, ancilla, measurement_bit, z_measurement)
        return

    if w in {5, 6}:
        weight_5 = w == 5
        if t == 2:
            measure_two_flagged_5_or_6(qc, stab, ancilla, measurement_bit, z_measurement, weight_5)
            return
        measure_w_flagged_5_or_6(qc, stab, ancilla, measurement_bit, z_measurement, weight_5)
        return

    if w in {7, 8}:
        weight_7 = w == 7
        if t == 2:
            measure_two_flagged_7_or_8(qc, stab, ancilla, measurement_bit, z_measurement, weight_7)
            return
        if t == 3:
            measure_three_flagged_7_or_8(qc, stab, ancilla, measurement_bit, z_measurement, weight_7)
            return

    if w in {11, 12}:
        weight_11 = w == 11
        if t == 2:
            measure_two_flagged_11_or_12(qc, stab, ancilla, measurement_bit, z_measurement, weight_11)
        if t == 3:
            measure_three_flagged_12(qc, stab, ancilla, measurement_bit, z_measurement, weight_11)
        return

    if t == 2:
        measure_two_flagged_general(qc, stab, ancilla, measurement_bit, z_measurement)
        return

    msg = f"Flagged measurement for w={w} and t={t} not implemented."
    raise NotImplementedError(msg)


def measure_one_flagged(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
) -> None:
    """Measure a 1-flagged stabilizer.

    In this case only one flag is required.
    """
    flag_reg = AncillaRegister(1)
    meas_reg = ClassicalRegister(1)
    qc.add_register(flag_reg)
    qc.add_register(meas_reg)
    flag = flag_reg[0]
    flag_meas = meas_reg[0]
    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)
    _flag_init(qc, flag, z_measurement)

    _ancilla_cnot(qc, flag, ancilla, z_measurement)

    for q in stab[1:-1]:
        _ancilla_cnot(qc, q, ancilla, z_measurement)

    _ancilla_cnot(qc, flag, ancilla, z_measurement)
    _flag_measure(qc, flag, flag_meas, z_measurement)

    _ancilla_cnot(qc, stab[-1], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_two_flagged_general(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
) -> None:
    """Measure a 2-flagged stabilizer using the scheme of https://arxiv.org/abs/1708.02246 (page 13).

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
    """
    n_flags = (len(stab) + 1) // 2 - 1
    flag_reg = AncillaRegister(n_flags)
    meas_reg = ClassicalRegister(n_flags)

    qc.add_register(flag_reg)
    qc.add_register(meas_reg)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag_reg[0], z_measurement)
    _ancilla_cnot(qc, flag_reg[0], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)
    _flag_init(qc, flag_reg[1], z_measurement)
    _ancilla_cnot(qc, flag_reg[1], ancilla, z_measurement)

    cnots = 2
    flags = 2
    for q in stab[2:-2]:
        _ancilla_cnot(qc, q, ancilla, z_measurement)
        cnots += 1
        if cnots % 2 == 0 and cnots < len(stab) - 2:
            _flag_init(qc, flag_reg[flags], z_measurement)
            _ancilla_cnot(qc, flag_reg[flags], ancilla, z_measurement)
        if cnots >= 7 and cnots % 2 == 1:
            _ancilla_cnot(qc, flag_reg[flags - 2], ancilla, z_measurement)
            _flag_measure(qc, flag_reg[flags - 2], meas_reg[flags - 2], z_measurement)
        if cnots % 2 == 0 and cnots < len(stab) - 2:
            flags += 1

    _ancilla_cnot(qc, flag_reg[0], ancilla, z_measurement)
    _flag_measure(qc, flag_reg[0], meas_reg[0], z_measurement)

    _ancilla_cnot(qc, stab[-2], ancilla, z_measurement)

    cnots += 1
    if cnots >= 7 and cnots % 2 == 1:
        _ancilla_cnot(qc, flag_reg[flags - 1], ancilla, z_measurement)
        _flag_measure(qc, flag_reg[flags - 1], meas_reg[flags - 1], z_measurement)

    _ancilla_cnot(qc, flag_reg[1], ancilla, z_measurement)
    _flag_measure(qc, flag_reg[1], meas_reg[1], z_measurement)

    _ancilla_cnot(qc, stab[-1], ancilla, z_measurement)

    cnots += 1
    if cnots >= 7 and cnots % 2 == 1:
        _ancilla_cnot(qc, flag_reg[flags - 1], ancilla, z_measurement)
        _flag_measure(qc, flag_reg[flags - 1], meas_reg[flags - 1], z_measurement)
    if not z_measurement:
        qc.h(ancilla)

    qc.measure(ancilla, measurement_bit)


def measure_two_flagged_4(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
) -> None:
    """Measure a 2-flagged weight 4 stabilizer. In this case only one flag is required.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
    """
    assert len(stab) == 4
    flag_reg = AncillaRegister(1)
    meas_reg = ClassicalRegister(1)
    qc.add_register(flag_reg)
    qc.add_register(meas_reg)
    flag = flag_reg[0]
    flag_meas = meas_reg[0]

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)
    _flag_init(qc, flag, z_measurement)

    _ancilla_cnot(qc, flag, ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)

    _ancilla_cnot(qc, flag, ancilla, z_measurement)
    _flag_measure(qc, flag, flag_meas, z_measurement)

    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_two_flagged_5_or_6(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
    weight_5: bool = False,
) -> None:
    """Measure a two-flagged weight 6 stabilizer using an optimized scheme.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
        weight_5: Whether the stabilizer has weight 5.
    """
    assert len(stab) == 6 or (len(stab) == 5 and weight_5)
    flag = AncillaRegister(2)
    meas = ClassicalRegister(2)

    qc.add_register(flag)
    qc.add_register(meas)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag[0], z_measurement)
    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)

    _flag_init(qc, flag[1], z_measurement)
    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)
    _flag_measure(qc, flag[0], meas[0], z_measurement)

    _ancilla_cnot(qc, stab[4], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)
    _flag_measure(qc, flag[1], meas[1], z_measurement)

    if not weight_5:
        _ancilla_cnot(qc, stab[5], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_w_flagged_5_or_6(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
    weight_5: bool = False,
) -> None:
    """Measure a w-flagged weight 6 stabilizer using an optimized scheme.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
        weight_5: Whether the stabilizer has weight 5.
    """
    assert len(stab) == 6 or (len(stab) == 5 and weight_5)
    flag = AncillaRegister(3)
    meas = ClassicalRegister(3)

    qc.add_register(flag)
    qc.add_register(meas)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag[0], z_measurement)
    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)

    _flag_init(qc, flag[1], z_measurement)
    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)

    _flag_init(qc, flag[2], z_measurement)
    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)
    _flag_measure(qc, flag[0], meas[0], z_measurement)

    _ancilla_cnot(qc, stab[4], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)
    _flag_measure(qc, flag[2], meas[2], z_measurement)

    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)
    _flag_measure(qc, flag[1], meas[1], z_measurement)

    if not weight_5:
        _ancilla_cnot(qc, stab[5], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_two_flagged_7_or_8(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
    weight_7: bool = False,
) -> None:
    """Measure a two-flagged weight 8 stabilizer using an optimized scheme.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
        weight_7: Whether the stabilizer has weight 7.
    """
    assert len(stab) == 8 or (len(stab) == 7 and weight_7)
    flag = AncillaRegister(3)
    meas = ClassicalRegister(3)
    qc.add_register(flag)
    qc.add_register(meas)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag[0], z_measurement)
    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)

    _flag_init(qc, flag[1], z_measurement)
    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    _flag_init(qc, flag[2], z_measurement)
    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[4], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[5], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)
    _flag_measure(qc, flag[0], meas[0], z_measurement)

    _ancilla_cnot(qc, stab[6], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)
    _flag_measure(qc, flag[2], meas[2], z_measurement)

    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)
    _flag_measure(qc, flag[1], meas[1], z_measurement)

    if not weight_7:
        _ancilla_cnot(qc, stab[7], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_three_flagged_7_or_8(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
    weight_7: bool = False,
) -> None:
    """Measure a three-flagged weight 8 stabilizer using an optimized scheme.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
        weight_7: Whether the stabilizer has weight 7.
    """
    assert len(stab) == 8 or (len(stab) == 7 and weight_7)
    flag = AncillaRegister(4)
    meas = ClassicalRegister(4)
    qc.add_register(flag)
    qc.add_register(meas)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag[0], z_measurement)
    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)

    _flag_init(qc, flag[1], z_measurement)
    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)

    _flag_init(qc, flag[2], z_measurement)
    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    _flag_init(qc, flag[3], z_measurement)
    _ancilla_cnot(qc, flag[3], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)
    _flag_measure(qc, flag[0], meas[0], z_measurement)

    _ancilla_cnot(qc, stab[4], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)
    _flag_measure(qc, flag[2], meas[2], z_measurement)

    _ancilla_cnot(qc, stab[5], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[6], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)
    _flag_measure(qc, flag[1], meas[1], z_measurement)

    _ancilla_cnot(qc, flag[3], ancilla, z_measurement)
    _flag_measure(qc, flag[3], meas[3], z_measurement)

    if not weight_7:
        _ancilla_cnot(qc, stab[7], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_two_flagged_11_or_12(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
    weight_11: bool = False,
) -> None:
    """Measure a two-flagged weight 12 stabilizer using an optimized scheme.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
        weight_11: Whether the stabilizer has weight 11.
    """
    assert len(stab) == 12 or (len(stab) == 11 and weight_11)
    flag = AncillaRegister(5)
    meas = ClassicalRegister(5)
    qc.add_register(flag)
    qc.add_register(meas)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag[0], z_measurement)
    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)

    _flag_init(qc, flag[1], z_measurement)
    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    _flag_init(qc, flag[2], z_measurement)
    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[4], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[5], ancilla, z_measurement)

    _flag_init(qc, flag[3], z_measurement)
    _ancilla_cnot(qc, flag[3], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[6], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)
    _flag_measure(qc, flag[2], meas[2], z_measurement)

    _ancilla_cnot(qc, stab[7], ancilla, z_measurement)

    _flag_init(qc, flag[4], z_measurement)
    _ancilla_cnot(qc, flag[4], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[8], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[3], ancilla, z_measurement)
    _flag_measure(qc, flag[3], meas[3], z_measurement)

    _ancilla_cnot(qc, stab[9], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)
    _flag_measure(qc, flag[0], meas[0], z_measurement)

    _ancilla_cnot(qc, stab[10], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)
    _flag_measure(qc, flag[1], meas[1], z_measurement)

    _ancilla_cnot(qc, flag[4], ancilla, z_measurement)
    _flag_measure(qc, flag[4], meas[4], z_measurement)

    if not weight_11:
        _ancilla_cnot(qc, stab[11], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def measure_three_flagged_12(
    qc: QuantumCircuit,
    stab: list[Qubit] | npt.NDArray[np.int_],
    ancilla: AncillaQubit,
    measurement_bit: Clbit,
    z_measurement: bool = True,
    weight_11: bool = False,
) -> None:
    """Measure a three-flagged weight 12 stabilizer using an optimized scheme.

    Args:
        qc: The quantum circuit to add the measurement to.
        stab: Support of the stabilizer to measure.
        ancilla: Ancilla qubit to use for the measurement.
        measurement_bit: Classical bit to store the measurement result of the ancilla.
        z_measurement: Whether to measure the ancilla in the Z basis.
        weight_11: Whether the stabilizer has weight 11.
    """
    assert len(stab) == 12 or (len(stab) == 11 and weight_11)
    flag = AncillaRegister(6)
    meas = ClassicalRegister(6)
    qc.add_register(flag)
    qc.add_register(meas)

    if not z_measurement:
        qc.h(ancilla)

    _ancilla_cnot(qc, stab[0], ancilla, z_measurement)

    _flag_init(qc, flag[0], z_measurement)
    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[1], ancilla, z_measurement)

    _flag_init(qc, flag[5], z_measurement)
    _ancilla_cnot(qc, flag[5], ancilla, z_measurement)

    _flag_init(qc, flag[1], z_measurement)
    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[2], ancilla, z_measurement)
    _ancilla_cnot(qc, stab[3], ancilla, z_measurement)

    _flag_init(qc, flag[2], z_measurement)
    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[4], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[5], ancilla, z_measurement)
    _flag_measure(qc, flag[5], meas[5], z_measurement)

    _ancilla_cnot(qc, stab[5], ancilla, z_measurement)

    _flag_init(qc, flag[3], z_measurement)
    _ancilla_cnot(qc, flag[3], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[6], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[2], ancilla, z_measurement)
    _flag_measure(qc, flag[2], meas[2], z_measurement)

    _ancilla_cnot(qc, stab[7], ancilla, z_measurement)

    _flag_init(qc, flag[4], z_measurement)
    _ancilla_cnot(qc, flag[4], ancilla, z_measurement)

    _ancilla_cnot(qc, stab[8], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[3], ancilla, z_measurement)
    _flag_measure(qc, flag[3], meas[3], z_measurement)

    _ancilla_cnot(qc, stab[9], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[0], ancilla, z_measurement)
    _flag_measure(qc, flag[0], meas[0], z_measurement)

    _ancilla_cnot(qc, stab[10], ancilla, z_measurement)

    _ancilla_cnot(qc, flag[4], ancilla, z_measurement)
    _flag_measure(qc, flag[4], meas[4], z_measurement)

    _ancilla_cnot(qc, flag[1], ancilla, z_measurement)
    _flag_measure(qc, flag[1], meas[1], z_measurement)

    if not weight_11:
        _ancilla_cnot(qc, stab[11], ancilla, z_measurement)

    if not z_measurement:
        qc.h(ancilla)
    qc.measure(ancilla, measurement_bit)


def vars_to_stab(
    measurement: list[z3.BoolRef | bool], generators: npt.NDArray[np.int8]
) -> npt.NDArray[z3.BoolRef | bool]:
    """Compute the stabilizer measured giving the generators and the measurement variables."""
    if not measurement:
        msg = "Measurement must not be empty"
        raise ValueError(msg)

    if len(generators) != len(measurement):
        msg = "Generators and measurement must have the same length"
        raise ValueError(msg)

    measurement_stab = symbolic_scalar_mult(generators[0], measurement[0])
    for i, scalar in enumerate(measurement[1:]):
        measurement_stab = symbolic_vector_add(measurement_stab, symbolic_scalar_mult(generators[i + 1], scalar))
    return measurement_stab
