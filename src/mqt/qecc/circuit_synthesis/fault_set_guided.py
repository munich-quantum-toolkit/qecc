# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Fault-set-guided synthesis of state preparation circuits for Steane-type fault-tolerant state preparation.

A Steane-type protocol connects several state preparation circuits with transversal CNOTs. The protocol is fault
tolerant if the fault sets of the circuits do not overlap in the required way. The synthesis in this module reduces
a check matrix with column operations, as the heuristic state preparation does, but only applies a CNOT if the new
faults do not overlap with given reference fault sets.

The method is described in E. Weilandt, T. Peham, R. Wille, "Synthesis of fault-tolerant state preparation circuits
using Steane-type error detection", Phys. Rev. A (2026), doi:10.1103/y9ln-b4t9.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from mqt.qecc import mod2

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from ..codes.core.css_code import CSSCode
    from .state_prep import FaultyStatePrepCircuit


logger = logging.getLogger(__name__)


class CandidateAction(Enum):
    """Action of the fault-set-guided search for one candidate CNOT."""

    SKIP = auto()
    """The candidate is forbidden or already used. Continue with the next candidate."""
    RESTART_SEARCH = auto()
    """No candidate reduces the matrix in the current layer. Start a new layer."""
    TRIGGER_BACKTRACK = auto()
    """No candidate reduces the matrix after a new layer was started. Undo the last CNOT."""
    EVALUATE = auto()
    """The candidate reduces the matrix. Check its faults against the reference fault sets."""


class EliminationCNOTSynthesizer:
    """Synthesizes a CNOT circuit by reducing a check matrix with column operations.

    Each column operation adds column ``i`` to column ``j`` and corresponds to a CNOT. The matrix is reduced when the
    number of zero columns equals the number of columns minus the rank.

    :meth:`fault_set_guided_synthesis` applies the cheapest column operation whose new faults do not overlap with given
    reference fault sets. If no such operation exists, the search undoes earlier operations (backtracking).
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.int8],
        code: CSSCode,
        parallel_elimination: bool = True,
    ) -> None:
        """Initialize the synthesizer.

        Args:
            matrix: The check matrix to reduce. The matrix is copied.
            code: The code whose stabilizers define equivalent faults.
            parallel_elimination: If True, prefer column operations on columns that are not yet used in the current
                layer. This reduces the depth of the circuit.
        """
        self.matrix: npt.NDArray[np.int8] = matrix.copy()
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
        # Copy the penalty CNOTs, so that the caller's list is not modified.
        self.failed_cnots: list[tuple[int, int]] = list(penalty_cnots or [])
        self.penalty_cnots: list[tuple[int, int]] = list(penalty_cnots or [])
        self.x_propagation_matrix: npt.NDArray[np.int8] = np.eye(self.matrix.shape[1], dtype=np.int8)
        self.z_propagation_matrix: npt.NDArray[np.int8] = np.eye(self.matrix.shape[1], dtype=np.int8)
        self.backtrack_required: bool = False
        self.overlapping_errors_x: set[tuple[np.int8, ...]] = set()
        self.overlapping_errors_z: set[tuple[np.int8, ...]] = set()
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

    def fault_set_guided_synthesis(
        self,
        ref_x_fs: npt.NDArray[np.int8] | None = None,
        ref_z_fs: npt.NDArray[np.int8] | None = None,
        ref_x_1fs: npt.NDArray[np.int8] | None = None,
        ref_z_1fs: npt.NDArray[np.int8] | None = None,
        penalty_cnots: list[tuple[int, int]] | None = None,
        guide_by_x: bool = True,
    ) -> None:
        """Reduce the matrix with column operations whose faults do not overlap with the reference fault sets.

        The search tries the candidate column operations in order of increasing cost. A candidate is accepted if
        the single X and Z faults it adds are not equivalent to a fault of the reference fault sets. If no candidate
        can be accepted, the search undoes the last accepted column operation and forbids it. The result is stored in
        :attr:`matrix` and :attr:`eliminations`.

        Args:
            ref_x_fs: Reference X fault set. The new X faults must not be equivalent to these faults.
            ref_z_fs: Reference Z fault set. The new Z faults must not be equivalent to these faults.
            ref_x_1fs: Reference X fault set of single faults. No sum of two X faults of the new circuit may be
                equivalent to one of these faults.
            ref_z_1fs: Reference Z fault set of single faults. No sum of two Z faults of the new circuit may be
                equivalent to one of these faults.
            penalty_cnots: CNOTs ``(control, target)`` that are initially forbidden.
            guide_by_x: If True, a forbidden CNOT becomes allowed again when its control is used by another CNOT.
                If False, the target releases it.

        Raises:
            ValueError: If no reference fault set is given.
            RuntimeError: If the search exhausts all candidates without reducing the matrix.
        """
        self._ref_based_init(
            ref_x_fs=ref_x_fs,
            ref_z_fs=ref_z_fs,
            ref_x_1fs=ref_x_1fs,
            ref_z_1fs=ref_z_1fs,
            penalty_cnots=penalty_cnots,
            guide_by_x=guide_by_x,
        )
        self._validate_inputs()
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
                    if not self.stack:
                        # The search undid every column operation and still failed.
                        msg = "Backtracking stack is empty. No solution found."
                        raise RuntimeError(msg)
                    self._backtrack()
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
        """Check if the matrix is reduced.

        Returns:
            True if the number of zero columns equals the number of columns minus the rank.
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

    def _cnot_is_valid_against_references(self, i: int, j: int) -> tuple[bool, dict[str, npt.NDArray[np.int8] | bool]]:
        found_cnot: bool = False
        new_x_error, next_x_propagation_matrix = get_next_error(
            propagation_matrix=self.x_propagation_matrix.copy(), cnot_gate=(int(i), int(j))
        )
        new_z_error, next_z_propagation_matrix = get_next_error(
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
            found_cnot = not (x_overlap or z_overlap)
            if not found_cnot:
                self.overlapping_errors_x.add(new_x_error_tuple)
                self.overlapping_errors_z.add(new_z_error_tuple)
        elif self.ref_x_fs.size:
            found_cnot = not x_overlap
            if not found_cnot:
                self.overlapping_errors_x.add(new_x_error_tuple)
        elif self.ref_z_fs.size:
            found_cnot = not z_overlap
            if not found_cnot:
                self.overlapping_errors_z.add(new_z_error_tuple)
        if found_cnot:
            self.x_propagation_matrix = next_x_propagation_matrix
            self.z_propagation_matrix = next_z_propagation_matrix
        return found_cnot, {
            "new_x_error": new_x_error,
            "new_z_error": new_z_error,
            "x_overlap": x_overlap,
            "z_overlap": z_overlap,
        }

    def _commit_to_cnot(self, i: int, j: int, new_errors: dict[str, npt.NDArray[np.int8] | bool]) -> None:
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
        # Restore the penalty CNOTs whose qubit is no longer used by the remaining eliminations.
        k = 0 if self.guide_by_x else 1
        used = {cnot[k] for cnot in self.eliminations}
        self.failed_cnots += [p for p in self.penalty_cnots if p[k] not in used and p not in self.failed_cnots]
        self.backtrack_required = False

    def _validate_inputs(self) -> None:
        """Check that the fault-set-guided synthesis has a reference fault set and a code."""
        has_refs = bool(self.ref_x_fs.size or self.ref_z_fs.size)
        has_code = self.code is not None
        if has_refs != has_code:
            msg = "Must provide either both a reference fault set and CSS code, or neither."
            raise ValueError(msg)

    def _reset_if_stuck(self, costs_unused: npt.NDArray[np.int8]) -> bool:
        """Start a new layer, or leave a local minimum, if no column operation reduces the matrix.

        Returns:
            True if a new layer was started or the matrix was brought into row echelon form.
        """
        if np.all(costs_unused >= 0) or len(self.used_columns) == self.matrix.shape[1]:
            if not self.used_columns:
                self._triangular_reset()
            else:
                self.used_columns = []
            return True
        return False

    def _triangular_reset(self) -> None:
        logger.warning("Local minimum reached. Making matrix triangular.")
        self.matrix = mod2.row_echelon(self.matrix, full=True)[0].astype(np.int8, copy=False)
        self.costs = self._compute_cost_matrix()

    def _apply_cnot_to_matrix(self, i: int, j: int) -> None:
        self.eliminations.append((i, j))
        if self.parallel_elimination:
            self.used_columns.append(i)
            self.used_columns.append(j)
        self.matrix[:, j] = (self.matrix[:, i] + self.matrix[:, j]) % 2
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
        m = np.zeros((self.matrix.shape[1], self.matrix.shape[1]), dtype=bool)
        m[self.used_columns, :] = True
        m[:, self.used_columns] = True
        return np.ma.array(self.costs, mask=m)

    def _get_candidate_pairs(self, costs_unused: npt.NDArray[np.int8]) -> list[tuple[int, int]]:
        """Return all column pairs, sorted by increasing cost. Masked pairs come last."""
        candidate_indices = np.argsort(costs_unused.flatten())
        return [(int(i), int(j)) for idx in candidate_indices for i, j in [np.unravel_index(idx, self.costs.shape)]]

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
    def _check_error_existence(error: npt.NDArray[np.int8], error_memory: set[tuple[np.int8, ...]]) -> bool:
        """Check if an error is already known to overlap. This avoids a second, slow overlap check."""
        error_tuple = tuple(error.flatten())
        return error_tuple in error_memory

    @staticmethod
    def _filter_fault_set(fault_set: npt.NDArray[np.int8], threshold: int) -> npt.NDArray[np.int8]:
        """Return the faults of weight above ``threshold``."""
        return fault_set[np.where(fault_set.sum(axis=1) > threshold)]


def get_next_error(
    propagation_matrix: npt.NDArray[np.int8], cnot_gate: tuple[int, int], x_error: bool = True
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """Propagate the single-qubit errors of a circuit through one more CNOT.

    Row ``q`` of the propagation matrix is the error at the end of the circuit that a single error on qubit ``q``
    causes. The circuit is built from the back, so the new CNOT comes before all CNOTs of the matrix.

    Args:
        propagation_matrix: The propagation matrix of the circuit (shape: number of qubits x number of qubits). The
            matrix is modified in place.
        cnot_gate: The new CNOT ``(control, target)``.
        x_error: If True, propagate X errors. If False, propagate Z errors.

    Returns:
        The propagated error of the qubit that the new CNOT changes (the control for X errors, the target for Z
        errors), and the updated propagation matrix.
    """
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
    """Add the candidate circuits whose fault sets do not overlap with those of the current circuits.

    Each candidate is compared with all current circuits, including candidates added before it. The candidate
    contributes the fault set that :func:`get_fs_based_on_d` selects, and the current circuits contribute their
    single-fault sets.

    Args:
        c_spcs: The current state preparation circuits. The list is extended in place.
        p_spcs: The candidate state preparation circuits.
        x_error: If True, compare the X fault sets. If False, compare the Z fault sets.

    Returns:
        The list ``c_spcs`` with the accepted candidates appended.
    """
    i = 0
    while i < len(p_spcs):
        pspc = p_spcs[i]
        px_fs, pz_fs = get_fs_based_on_d(pspc)

        for spc in c_spcs:
            x_fs = spc.compute_fault_set()
            z_fs = spc.compute_fault_set(x_errors=False)
            if x_error:
                if spc.code.check_fs_overlap(x_fs.faults, px_fs):
                    break
            elif spc.code.check_fs_overlap(z_fs.faults, pz_fs, x_error=False):
                break
        else:
            c_spcs.append(pspc)

        i += 1
    return c_spcs


def get_fs_based_on_d(spc: FaultyStatePrepCircuit) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """Return the faults of a circuit that a Steane-type protocol must detect.

    The selection depends on the distance ``d`` of ``spc.code``:

    - ``d = 3``: all single faults;
    - ``d = 5``: the single faults of weight above 2;
    - ``d = 7``: the single faults of weight above 2 and the double faults of weight above 3.

    Args:
        spc: The state preparation circuit.

    Returns:
        The selected X faults and Z faults.

    Raises:
        ValueError: If the distance is not 3, 5 or 7.
    """

    def compute_fault_sets(level: int = 1) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
        spc.compute_fault_set(level)
        spc.compute_fault_set(level, x_errors=False)
        return spc.x_fault_sets[level - 1].faults, spc.z_fault_sets[level - 1].faults

    def filter_fault_set(fault_set: npt.NDArray[np.int8], threshold: int) -> npt.NDArray[np.int8]:
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

    if spc.code.distance == 1:
        msg = (
            "Code distance unknown (1). Synthesize the circuit from a code with its distance set, e.g. "
            "CSSCode(..., distance=d) or CSSCode.from_file(..., distance=d)."
        )
    else:
        msg = f"Unsupported code distance: {spc.code.distance}. Supported are 3, 5 and 7."
    raise ValueError(msg)
