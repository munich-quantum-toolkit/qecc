# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: ANN001
"""Frozen copy of the April 2025 heuristic Gaussian elimination that synthesized the published distance-7 circuits.

Source: mqt-qecc commit a8be98c9 (2025-04-08, "improve CNOT count of C4 for the distance 7 octagon code"),
file src/mqt/qecc/circuit_synthesis/synthesis_utils.py, lines 91-383 (``print_dynamic_eliminations``,
``heuristic_gaussian_elimination``, ``get_next_error``). The functions below are copied verbatim; only this header
and the imports were added.

Why a copy: the C2-C4 circuits of [[31,1,7]] and [[37,1,7]] were synthesized with this function. The class-based
rewrite in June 2025 (``EliminationCNOTSynthesizer``) is not behaviour-preserving:
  - it no longer seeds the elimination with the row echelon form of the check matrix for codes with distance > 5
    when a reference Z fault set is given (C3 and C4), which is what makes the distance-7 search tractable;
  - it bookkeeps the search differently (``used_cnots`` is no longer recorded before the overlap check and reset
    after each accepted CNOT; candidates whose error is already known to overlap are now added to ``failed_cnots``).
With the current library the same inputs give different circuits or do not finish. Used only by ``recipes.py``.
Do not modify.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import numpy as np
from ldpc import mod2

if TYPE_CHECKING:
    import numpy.typing as npt

    from mqt.qecc.codes.css_code import CSSCode

logger = logging.getLogger(__name__)


def print_dynamic_eliminations(eliminations, failed_cnots) -> None:
    """Prints the eliminations list dynamically on a single line.

    Parameters:
    - eliminations: List of (control, target) tuples representing CNOT operations.
    """
    # Clear both lines
    sys.stdout.write("\r" + " " * 1000 + "\r")  # Clear line 1

    # Print the updated lists
    sys.stdout.write(f"\rCurrent Eliminations: {eliminations} | Failed CNOTs: {failed_cnots}")
    sys.stdout.flush()


def heuristic_gaussian_elimination(
    matrix: npt.NDArray[np.int8],
    parallel_elimination: bool = True,
    penalty_cols: list[tuple[int]] | None = None,
    code: CSSCode | None = None,
    ref_x_fs: npt.NDArray[np.int8] | None = None,
    ref_z_fs: npt.NDArray[np.int8] | None = None,
    guide_by_x: bool = True,
    ref_x_1fs: npt.NDArray[np.int8] | None = None,
) -> tuple[npt.NDArray[np.int8], list[tuple[int, int]]]:
    """Perform Gaussian elimination on the column space of a matrix using as few eliminations as possible.

    The algorithm utilizes a greedy heuristic to select the columns to eliminate in order to minimize the number of eliminations required.

    The matrix is reduced until there are exactly rnk(matrix) columns with non-zero entries.

    Args:
        matrix: The matrix to perform Gaussian elimination on.
        parallel_elimination: Whether to prioritize elimination steps that act on disjoint columns.
        penalty_cols: tuples of CNOTs (control, target) which are initially added to the failed_cnots list and hence can only be applied once the control qubit has been used elsewhere
        code: Code based on which the fault sets are checked for overlaps. (Overlap check is based on check matrices)
        ref_x_fs: (Optional) reference x fault set which influences the construction of the circuit
        ref_z_fs: (Optional) reference z fault set which influences the construction of the circuit
        guide_by_x: Flag that decides wether dismissed CNOTs are free to placement again after either the control (x guided) or the target (z guided) has been used elsewhere
        ref_x_1fs: (Optional) reference one error x fault set which ensures that no two error event of the newly constructed circuit cancels a one error event of the reference circuit

    Returns:
        The reduced matrix and a list of the elimination steps taken. The elimination steps are represented as tuples of the form (i, j) where i is the column being eliminated with and j is the column being eliminated.
    """
    if penalty_cols is None:
        penalty_cols = []
    if ref_x_fs is None:
        ref_x_fs = np.empty((0,), dtype=np.int8)
    if ref_x_1fs is None:
        ref_x_1fs = np.empty((0,), dtype=np.int8)
    if ref_z_fs is None:
        ref_z_fs = np.empty((0,), dtype=np.int8)
    # INFO: Code is always provided but if no reference sets are given we act like there is no code.
    if (code is not None) and not (ref_x_fs.size or ref_z_fs.size):
        code = None
    if not ((ref_x_fs.size or ref_z_fs.size) ^ (code is None)):
        msg = "If providing at least one reference set, it is also necessary to provide a reference CSS Code. Or if providing a CSSCode, at least of reference fault set must be given also."
        raise ValueError(msg)

    matrix = matrix.copy()
    if code and code.distance > 5:
        if ref_z_fs.size and not ref_x_fs.size:
            matrix = mod2.row_echelon(matrix, full=True)[0]
        if ref_z_fs.size and ref_x_fs.size:
            matrix = mod2.row_echelon(matrix, full=True)[0]
    rank = mod2.rank(matrix)

    def is_reduced() -> bool:
        return bool(len(np.where(np.all(matrix == 0, axis=0))[0]) == matrix.shape[1] - rank)

    costs = np.array([
        [np.sum((matrix[:, i] + matrix[:, j]) % 2) for j in range(matrix.shape[1])] for i in range(matrix.shape[1])
    ])

    costs -= np.sum(matrix, axis=0)
    np.fill_diagonal(costs, 1)

    used_columns = []  # type: list[np.int_]
    eliminations = []  # type: list[tuple[int, int]]
    failed_cnots = penalty_cols
    used_cnots = []
    # NOTE: set up variables for distinct fault set construction
    if code is not None:
        # matrix that describes how an error happening at qubit i would propagate right now through
        # the circuit
        x_propagation_matrix: npt.NDArray[np.int8] = np.eye(matrix.shape[1], dtype=np.int8)
        z_propagation_matrix: npt.NDArray[np.int8] = np.eye(matrix.shape[1], dtype=np.int8)
        backtrack_required: bool = False
        overlapping_errors_x = set()
        overlapping_errors_z = set()
        stack = []
        # current_x_fs = np.empty((0, matrix.shape[1]), dtype=np.int8)
        current_x_fs = np.eye(matrix.shape[1], dtype=np.int8)
        # current_z_fs = np.empty((0, matrix.shape[1]), dtype=np.int8)
    while not is_reduced():
        # flag in case algorithm can only choose CNOT that would violate t-distinctness
        deadend: bool = False

        m = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=bool)  # type: npt.NDArray[np.bool_]
        m[used_columns, :] = True
        m[:, used_columns] = True
        costs_unused = np.ma.array(costs, mask=m)  # type: ignore[no-untyped-call]

        if np.all(costs_unused >= 0) or len(used_columns) == matrix.shape[1]:  # no more reductions possible
            if used_columns == []:  # local minimum => get out by making matrix triangular
                logger.warning("Local minimum reached. Making matrix triangular.")
                matrix = mod2.row_echelon(matrix, full=True)[0]
                costs = np.array([
                    [np.sum((matrix[:, i] + matrix[:, j]) % 2) for j in range(matrix.shape[1])]
                    for i in range(matrix.shape[1])
                ])
                costs -= np.sum(matrix, axis=0)
                np.fill_diagonal(costs, 1)
            else:  # try to move onto the next layer
                used_columns = []
            continue

        if code is not None:
            # Get all valid (i, j) pairs sorted by cost
            candidate_indices = np.argsort(costs_unused.flatten())  # Flatten and sort by value
            candidate_pairs = [np.unravel_index(idx, costs.shape) for idx in candidate_indices]

            # valid_indices = np.where((costs_unused < 0) & ~costs_unused.mask)
            # negative_values = costs_unused[valid_indices]
            # sorted_indices = np.argsort(negative_values)  # Sort values in ascending order
            # candidate_pairs = list(zip(valid_indices[0][sorted_indices], valid_indices[1][sorted_indices]))

            # Find the first (i, j) that does NOT cause overlap
            for i, j in candidate_pairs:
                if (int(i), int(j)) in failed_cnots or (int(i), int(j)) in used_cnots:
                    continue

                if costs_unused[i, j] >= 0 and backtrack_required:  # level has been checked completely
                    x_propagation_matrix, z_propagation_matrix, used_columns, matrix, costs, used_cnots, _ = stack.pop()
                    removed_cnot = eliminations.pop()
                    if guide_by_x:
                        failed_cnots = [fcnot for fcnot in failed_cnots if removed_cnot[0] != fcnot[0]]
                    else:
                        failed_cnots = [fcnot for fcnot in failed_cnots if removed_cnot[1] != fcnot[1]]
                    failed_cnots.append(removed_cnot)
                    print_dynamic_eliminations(eliminations, failed_cnots)
                    backtrack_required = False
                    deadend = True
                    break
                # level has been checked once but possibly not completely due to parallelization
                if costs_unused[i, j] >= 0:
                    used_columns = []
                    deadend = True
                    backtrack_required = True
                    break

                used_cnots.append((int(i), int(j)))

                new_x_error, next_x_propagation_matrix = get_next_error(
                    propagation_matrix=x_propagation_matrix.copy(), cnot_gate=(int(i), int(j))
                )
                new_z_error, next_z_propagation_matrix = get_next_error(
                    propagation_matrix=z_propagation_matrix.copy(), cnot_gate=(int(i), int(j)), x_error=False
                )

                # INFO: Quickly check if new error is known to cause overlap. This saves another long overlap check.
                new_x_error_tuple = tuple(new_x_error.flatten())
                new_z_error_tuple = tuple(new_z_error.flatten())
                if new_x_error_tuple in overlapping_errors_x:
                    continue
                if new_z_error_tuple in overlapping_errors_z:
                    continue

                new_x_error = np.expand_dims(new_x_error, axis=0)
                new_z_error = np.expand_dims(new_z_error, axis=0)

                # Check if there is no overlap—if true, accept (i, j)
                # NOTE: Now if specifically given a one error x fault set it is possible to also keep track of the current
                # 2 error fault set of the new circuit and check those for overlap.
                # TODO: for z the above functionality is not yet implemented
                found_cnot = False

                def filter_fault_set(fault_set: npt.NDArray[np.int8], threshold: int) -> npt.NDArray[np.int8]:
                    """Filter fault sets based on a sum threshold."""
                    return fault_set[np.where(fault_set.sum(axis=1) > threshold)]

                # only calculate overlap if reference fault set is given (optimization)
                if ref_x_fs.size:
                    x_overlap: bool = code.check_fs_overlap(ref_x_fs, new_x_error)
                    if ref_x_1fs.size and not x_overlap:
                        trial_x_2fs = (current_x_fs + new_x_error) % 2
                        trial_x_2fs = filter_fault_set(trial_x_2fs, int((code.distance - 1) / 2))
                        x_overlap: bool = code.check_fs_overlap(ref_x_1fs, trial_x_2fs)
                if ref_z_fs.size:
                    z_overlap: bool = code.check_fs_overlap(ref_z_fs, new_z_error, x_error=False)

                if ref_x_fs.size and ref_z_fs.size:
                    # Both sets provided: Enter if **both** overlaps fail
                    found_cnot = not (x_overlap or z_overlap)
                    if not found_cnot:
                        overlapping_errors_x.add(new_x_error_tuple)
                        overlapping_errors_z.add(new_z_error_tuple)
                elif ref_x_fs.size:
                    # Only x-set provided: Enter if **x_overlap** fails
                    found_cnot = not x_overlap
                    if not found_cnot:
                        overlapping_errors_x.add(new_x_error_tuple)
                elif ref_z_fs.size:
                    # Only z-set provided: Enter if **z_overlap** fails
                    found_cnot = not z_overlap
                    if not found_cnot:
                        overlapping_errors_z.add(new_z_error_tuple)

                if found_cnot:
                    stack.append((
                        x_propagation_matrix.copy(),
                        z_propagation_matrix.copy(),
                        used_columns.copy(),
                        matrix.copy(),
                        costs.copy(),
                        used_cnots.copy(),
                        failed_cnots.copy(),
                    ))
                    used_cnots = []
                    if guide_by_x:
                        failed_cnots = [fcnot for fcnot in failed_cnots if int(i) != fcnot[0]]
                    else:
                        failed_cnots = [fcnot for fcnot in failed_cnots if int(j) != fcnot[1]]

                    if ref_x_fs.size and not x_overlap:
                        x_propagation_matrix = next_x_propagation_matrix
                        current_x_fs = np.vstack((current_x_fs, new_x_error), dtype=np.int8)
                    if ref_z_fs.size and not z_overlap:
                        z_propagation_matrix = next_z_propagation_matrix
                        # current_z_fs = trial_z_fs

                    break
                failed_cnots.append((int(i), int(j)))
                print_dynamic_eliminations(eliminations, failed_cnots)

        else:
            i, j = np.unravel_index(np.argmin(costs_unused), costs.shape)
        if deadend:
            continue
        eliminations.append((int(i), int(j)))
        print_dynamic_eliminations(eliminations, failed_cnots)

        if parallel_elimination:
            used_columns.append(i)
            used_columns.append(j)

        # update matrix
        matrix[:, j] = (matrix[:, i] + matrix[:, j]) % 2
        # update costs
        new_weights = np.sum((matrix[:, j][:, np.newaxis] + matrix) % 2, axis=0)
        costs[j, :] = new_weights - np.sum(matrix, axis=0)
        costs[:, j] = new_weights - np.sum(matrix[:, j])
        np.fill_diagonal(costs, 1)

    return matrix, eliminations


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
