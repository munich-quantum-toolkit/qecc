# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Synthesizing deterministic state preparation circuits for d<5 CSS codes."""

from __future__ import annotations

import logging
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import z3
from ldpc import mod2

from .state_prep import (
    StatePrepCircuit,
    all_gate_optimal_verification_stabilizers,
    coset_leader,
    get_hook_errors,
    heuristic_verification_stabilizers,
    vars_to_stab,
)
from .synthesis_utils import (
    iterative_search_with_timeout,
    odd_overlap,
    run_with_timeout,
    symbolic_vector_eq,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..codes import CSSCode

Recovery = tuple[list[npt.NDArray[np.int8]], dict[int, npt.NDArray[np.int8]]]
Recoveries = dict[int, Recovery]
DeterministicCorrection = dict[int, tuple[list[npt.NDArray[np.int8]], Recoveries]]
Verification = list[npt.NDArray[np.int8]]


class DeterministicVerification:
    """Class to store deterministic verification stabilizers and corrections."""

    def __init__(
        self,
        and_verification_stabs: Verification,
        det_correction: DeterministicCorrection,
        hook_corrections: list[DeterministicCorrection] | None = None,
    ) -> None:
        """Initialize a deterministic verification object.

        Args:
            and_verification_stabs: The non-deterministic verification stabilizers to be measured.
            det_correction: The deterministic correction for the non-deterministic verification stabilizers.
            hook_corrections: the hook corrections for the non-deterministic verification stabilizers.
        """
        self.stabs = and_verification_stabs
        self.det_correction = det_correction
        self.hook_corrections: list[DeterministicCorrection] = [{}] * len(and_verification_stabs)
        if hook_corrections:
            assert len(hook_corrections) == len(and_verification_stabs)
            self.hook_corrections = hook_corrections

    def copy(self) -> DeterministicVerification:
        """Return a copy of the deterministic verification object."""
        return DeterministicVerification(self.stabs, self.det_correction, self.hook_corrections)

    @staticmethod
    def _stat_cnots_correction(fun: Callable[..., int], correction: DeterministicCorrection) -> int:
        """Return stats on the CNOTs in the deterministic correction."""
        return fun([np.sum([np.sum(m) for m in v[0]]) for v in correction.values()])

    @staticmethod
    def _stat_anc_correction(fun: Callable[..., int], correction: DeterministicCorrection) -> int:
        """Return stats on the ancillas in the deterministic correction."""
        return fun([len(v[0]) for v in correction.values()])

    @staticmethod
    def _num_branches_correction(correction: DeterministicCorrection) -> int:
        """Return the number of branches in the deterministic correction."""
        return sum(len(v[1]) for v in correction.values())

    # Statistics methods
    def num_ancillas_verification(self) -> int:
        """Return the number of ancillas needed for the verification."""
        return len(self.stabs)

    def num_cnots_verification(self) -> int:
        """Return the number of CNOTs needed for the verification."""
        return np.sum([np.sum(m) for m in self.stabs])

    def num_ancillas_correction(self) -> int:
        """Return the number of ancillas needed for the correction."""
        return self._stat_anc_correction(sum, self.det_correction)

    def stat_ancillas_correction(self, fun: Callable[..., int]) -> int:
        """Return some statistics on the ancillas in the deterministic correction using the function fun.

        Args:
            fun: The function to use for the statistics, e.g. sum, max, min, etc.
        """
        return self._stat_anc_correction(fun, self.det_correction)

    def num_cnots_correction(self) -> int:
        """Return the number of CNOTs needed for the correction."""
        return self._stat_cnots_correction(sum, self.det_correction)

    def stat_cnots_correction(self, fun: Callable[..., int]) -> int:
        """Return some statistics on the CNOTs in the deterministic correction using the function fun.

        Args:
            fun: The function to use for the statistics, e.g. sum, max, min, etc.
        """
        return self._stat_cnots_correction(fun, self.det_correction)

    def num_ancillas_hooks(self) -> int:
        """Return the number of ancillas needed for the hook corrections."""
        return len([v for v in self.hook_corrections if v])

    def num_cnots_hooks(self) -> int:
        """Return the number of CNOTs needed for the hook corrections (two per ancilla)."""
        return self.num_ancillas_hooks() * 2

    def num_ancillas_hook_corrections(self) -> int:
        """Return the number of ancillas needed for the hook corrections of the verification stabilizers."""
        return sum(self._stat_anc_correction(sum, c) if c != {} else 0 for c in self.hook_corrections)

    def stat_ancillas_hook_corrections(self, fun: Callable[..., int]) -> int:
        """Return some statistics on the ancillas in the hook corrections of the verification stabilizers using the function fun.

        Args:
            fun: The function to use for the statistics, e.g. sum, max, min, etc.
        """
        return fun([self._stat_anc_correction(fun, c) if c != {} else 0 for c in self.hook_corrections])

    def num_cnots_hook_corrections(self) -> int:
        """Return the number of CNOTs needed for the hook corrections of the verification stabilizers."""
        return sum(self._stat_cnots_correction(sum, c) if c != {} else 0 for c in self.hook_corrections)

    def stat_cnots_hook_corrections(self, fun: Callable[..., int]) -> int:
        """Return some statistics on the CNOTs in the hook corrections of the verification stabilizers using the function fun.

        Args:
            fun: The function to use for the statistics, e.g. sum, max, min, etc.
        """
        return fun([self._stat_cnots_correction(fun, c) if c != {} else 0 for c in self.hook_corrections])

    def num_ancillas_total(self) -> int:
        """Return the total number of ancillas needed for the verification and correction."""
        return (
            self.num_ancillas_verification()
            + self.num_ancillas_correction()
            + self.num_ancillas_hooks()
            + self.num_ancillas_hook_corrections()
        )

    def num_cnots_total(self) -> int:
        """Return the total number of CNOTs needed for the verification and correction."""
        return (
            self.num_cnots_verification()
            + self.num_cnots_correction()
            + self.num_cnots_hooks()
            + self.num_cnots_hook_corrections()
        )

    def num_branches_det_correction(self) -> int:
        """Return the number of branches in the deterministic correction."""
        return self._num_branches_correction(self.det_correction)

    def num_branches_hook_corrections(self) -> int:
        """Return the number of branches in the hook corrections of the verification stabilizers."""
        return sum(self._num_branches_correction(c) for c in self.hook_corrections)

    def num_branches_total(self) -> int:
        """Return the total number of branches in the verification and correction."""
        return self.num_branches_det_correction() + self.num_branches_hook_corrections()


class DeterministicVerificationHelper:
    """Class to compute the deterministic verification stabilizers and corrections for a given state preparation circuit."""

    def __init__(self, state_prep: StatePrepCircuit, use_optimal_verification: bool = True) -> None:
        """Initialize the deterministic verification helper with a given state preparation circuit.

        Args:
            state_prep: The state preparation circuit to compute the deterministic verification for (must be a CSS code and d<5).
            use_optimal_verification: If True, the optimal verification stabilizers are computed, otherwise heuristic verification stabilizers are used.
        """
        self.state_prep = state_prep
        self.code = state_prep.code
        assert self.code.distance < 5, "Only d=3 and d=4 codes are supported."
        self.num_qubits = self.code.n
        self.layer_x_errors = [self.state_prep.zero_state, not self.state_prep.zero_state]
        self.use_optimal_verification = use_optimal_verification
        # Variable to store the deterministic verification stabilizers and corrections for the two layers
        self._layers: list[list[DeterministicVerification]] = [[], []]
        # Variable to store the deterministic verification stabilizers and corrections for the hook propagation solution
        self._hook_propagation_solutions: list[tuple[DeterministicVerification, DeterministicVerification]] = []

    def _compute_non_det_stabs(
        self,
        min_timeout: int = 1,
        max_timeout: int = 3600,
        max_ancillas: int | None = None,
        compute_all_solutions: bool = False,
    ) -> None:
        """Computes the non-deterministic verification stabilizers for both layers (X and Z).

        Args:
            min_timeout: The minimum time in seconds to run the verification stabilizers.
            max_timeout: The maximum time in seconds to run the verification stabilizers.
            max_ancillas: The maximum number of ancillas to use in the verification stabilizers.
            compute_all_solutions: If True, all equivalent verification stabilizers are computed and stored.
        """
        for idx, x_errors in enumerate(self.layer_x_errors):
            logger.info(f"Computing non-deterministic verification stabilizers for layer {idx + 1} / 2.")
            # if self.use_optimal_verification:
            stabs_all = all_gate_optimal_verification_stabilizers(
                self.state_prep,
                x_errors=x_errors,
                min_timeout=min_timeout,
                max_timeout=max_timeout,
                max_ancillas=max_ancillas,
                return_all_solutions=compute_all_solutions,
            )[0]
            for stabs in stabs_all:
                verify = DeterministicVerification(stabs, {})
                self._layers[idx].append(verify)

    def _compute_det_corrections(
        self, min_timeout: int = 1, max_timeout: int = 3600, max_ancillas: int | None = None, layer_idx: int = 0
    ) -> None:
        """Returns the deterministic verification stabilizers for the first layer of non-deterministic verification stabilizers."""
        x_errors = self.layer_x_errors[layer_idx]
        logger.info(f"Computing deterministic verification for layer {layer_idx}.")
        for verify_idx, verify in enumerate(self._layers[layer_idx]):
            logger.info(
                f"Computing deterministic verification for non-det verification {verify_idx + 1} / {len(self._layers[layer_idx])}."
            )
            self._layers[layer_idx][verify_idx].det_correction = deterministic_correction(
                self.state_prep,
                verify.stabs,
                min_timeout=min_timeout,
                max_timeout=max_timeout,
                max_ancillas=max_ancillas,
                zero_state=x_errors,
            )

    @staticmethod
    def _trivial_hook_errors(hook_errors: list[npt.NDArray[np.int8]], code: CSSCode, x_error: bool) -> bool:
        """Checks if the hook errors are trivial (stabilizers) by checking if the rank of the code stabilizers is the same.

        Args:
            hook_errors: The hook errors to check.
            code: The CSS code to check the rank of the stabilizers.
            x_error: If True, the Z stabilizers are checked, otherwise the X stabilizers are checked.

        Returns:
            bool: True if all hook errors are trivial, False otherwise.
        """
        errors_trivial = []
        code_stabs = np.vstack((code.Hz, code.Lz)) if x_error else np.vstack((code.Hx, code.Lx))
        rank = mod2.rank(code_stabs)
        for error in hook_errors:
            single_qubit_deviation = [(error + np.eye(code.n, dtype=np.int8)[i]) % 2 for i in range(code.n)]
            stabs_plus_single_qubit = [np.vstack((code_stabs, single_qubit_deviation[i])) for i in range(code.n)]
            trivial = any(mod2.rank(m) == rank for m in stabs_plus_single_qubit)
            errors_trivial.append(trivial)
        return bool(all(errors_trivial))

    def _compute_hook_corrections(
        self, min_timeout: int = 1, max_timeout: int = 3600, max_ancillas: int | None = None
    ) -> None:
        """Computes the additional stabilizers to measure with corresponding corrections for the hook errors of each stabilizer measurement in layer 2."""
        for layer_idx, x_error in enumerate(self.layer_x_errors):
            logger.info(f"Computing deterministic verification for hook errors of layer {layer_idx + 1} / 2.")
            for verify_idx, verify in enumerate(self._layers[layer_idx]):
                logger.info(
                    f"Computing deterministic hook correction for non-det verification {verify_idx + 1} / {len(self._layers[layer_idx])}."
                )
                if not verify.stabs:
                    self._layers[layer_idx][verify_idx].hook_corrections = [{}] * len(verify.stabs)
                    continue
                for stab_idx, stab in enumerate(verify.stabs):
                    hook_errors = list(get_hook_errors([stab]))
                    if self._trivial_hook_errors(hook_errors, self.code, not x_error):
                        continue

                    # hook errors are non-trivial
                    # add case of error on hook ancilla
                    hook_errors = np.vstack((hook_errors, np.zeros(self.num_qubits, dtype=np.int8)))
                    self._layers[layer_idx][verify_idx].hook_corrections[stab_idx] = {
                        1: deterministic_correction_single_outcome(
                            self.state_prep,
                            hook_errors,
                            min_timeout=min_timeout,
                            max_timeout=max_timeout,
                            max_ancillas=max_ancillas,
                            zero_state=not x_error,
                        )
                    }

    def _filter_and_stabs(self) -> None:
        """Only keep the best non-deterministic verification stabilizers with minimal number of ancillas and CNOTs."""
        self._best_num_anc = 0
        self._best_num_cnots = 0
        for layer_idx in range(2):
            # get best numbers
            best_num_anc = int(1e6)
            best_num_cnots = int(1e6)
            best_case_indices = []
            for idx_verify, verify in enumerate(self._layers[layer_idx]):
                num_anc = verify.num_ancillas_verification() + verify.num_ancillas_hooks()
                num_cnot = verify.num_cnots_verification() + verify.num_cnots_hooks()
                if best_num_anc > num_anc or (best_num_anc == num_anc and best_num_cnots > num_cnot):
                    best_num_anc = num_anc
                    best_num_cnots = num_cnot
                    best_case_indices = [idx_verify]
                elif best_num_anc == num_anc and best_num_cnots == num_cnot:
                    best_case_indices.append(idx_verify)
            # filter out all but the best case
            self._layers[layer_idx] = [self._layers[layer_idx][idx] for idx in best_case_indices]
            # save the best numbers
            self._best_num_anc += best_num_anc
            self._best_num_cnots += best_num_cnots

    def _recompute_hook_propagation_corrections(
        self,
        verify_2_list: list[DeterministicVerification],
        verify: DeterministicVerification,
        stabs_flagged: list[bool],
        min_timeout: int = 1,
        max_timeout: int = 3600,
        max_ancillas: int | None = None,
    ) -> None:
        for verify_2_idx, verify_2 in enumerate(verify_2_list):
            verify_2_list[verify_2_idx].det_correction = deterministic_correction(
                self.state_prep,
                verify_2.stabs,
                min_timeout=min_timeout,
                max_timeout=max_timeout,
                max_ancillas=max_ancillas,
                zero_state=not self.state_prep.zero_state,
            )
            for stab_idx, stab in enumerate(verify_2.stabs):
                hook_errors_2 = list(get_hook_errors([stab]))
                if self._trivial_hook_errors(hook_errors_2, self.code, self.state_prep.zero_state):
                    verify_2_list[verify_2_idx].hook_corrections[stab_idx] = {}
                else:
                    hook_errors_2 = np.vstack((hook_errors_2, np.zeros(self.num_qubits, dtype=np.int8)))
                    verify_2_list[verify_2_idx].hook_corrections[stab_idx] = {
                        1: deterministic_correction_single_outcome(
                            self.state_prep,
                            hook_errors_2,
                            min_timeout=min_timeout,
                            max_timeout=max_timeout,
                            max_ancillas=max_ancillas,
                            zero_state=self.state_prep.zero_state,
                        )
                    }

        # choose the best solution
        verify_2_best = verify_2_list[0]
        num_anc_verify_2 = verify_2_best.num_ancillas_total()
        num_cnots_verify_2 = verify_2_best.num_cnots_total()
        for verify_2 in verify_2_list[1:]:
            if num_anc_verify_2 > verify_2.num_ancillas_total() or (
                num_anc_verify_2 == verify_2.num_ancillas_total() and num_cnots_verify_2 > verify_2.num_cnots_total()
            ):
                num_anc_verify_2 = verify_2.num_ancillas_total()
                num_cnots_verify_2 = verify_2.num_cnots_total()
                verify_2_best = verify_2

        # modify the first layer verification and reduce necessary hooks
        verify_new = verify.copy()
        # compute necessary hooks
        for idx, flag in enumerate(stabs_flagged):
            if flag:
                verify_new.hook_corrections[idx] = {
                    1: deterministic_correction_single_outcome(
                        self.state_prep,
                        get_hook_errors([verify.stabs[idx]]),
                        min_timeout=min_timeout,
                        max_timeout=max_timeout,
                        max_ancillas=max_ancillas,
                        zero_state=not self.state_prep.zero_state,
                    )
                }
            else:
                verify_new.hook_corrections[idx] = {}

        self._hook_propagation_solutions.append((verify_new, verify_2_best))

    def _compute_hook_propagation_solutions(
        self,
        min_timeout: int = 1,
        max_timeout: int = 3600,
        max_ancillas: int | None = None,
        compute_all_solutions: bool = False,
    ) -> None:
        """Computes the second layer assuming the hook errors are not flagged but propagated."""
        if not self._layers[1]:
            # no second layer
            return

        for verify in self._layers[0]:
            logger.info(f"Computing hook propagation solutions for verification {verify} / {len(self._layers[0])}.")
            # create possible combinations of which hook errors are flagged
            # need_hook_corrections = [
            #     not self._trivial_hook_errors(_hook_errors([stab]), self.code, not x_errors) for stab in stabs
            # ]
            stabs_flagged_all = [
                not self._trivial_hook_errors(get_hook_errors([stab]), self.code, not self.layer_x_errors[0])
                for stab in verify.stabs
            ]
            # stabs_flagged_all = [True if hook else False for hook in verify.hook_corrections]
            stabs_flagged_all_indices = [idx for idx, flag in enumerate(stabs_flagged_all) if flag]
            stabs_flagged_indices_combs = list(product([False, True], repeat=len(stabs_flagged_all_indices)))[:-1]
            stabs_flagged_combs = []
            for comb in stabs_flagged_indices_combs:
                stabs_flagged = [False] * len(stabs_flagged_all)
                for idx_comb, idx in enumerate(stabs_flagged_all_indices):
                    stabs_flagged[idx] = comb[idx_comb]
                stabs_flagged_combs.append(stabs_flagged)

            # iterate over combinations:
            for stabs_flagged in stabs_flagged_combs:
                # get hook errors
                hook_errors = np.empty((0, self.num_qubits), dtype=np.int8)
                for idx, flag in enumerate(stabs_flagged):
                    if not flag:
                        hook_errors = np.vstack((hook_errors, get_hook_errors([verify.stabs[idx]])))
                if self._trivial_hook_errors(hook_errors, self.code, not self.state_prep.zero_state):
                    continue
                # hook errors require different verification in second layer
                # compute new verification
                if self.use_optimal_verification:
                    stabs_2_list = all_gate_optimal_verification_stabilizers(
                        self.state_prep,
                        x_errors=not self.state_prep.zero_state,
                        min_timeout=min_timeout,
                        max_timeout=max_timeout,
                        max_ancillas=max_ancillas,
                        additional_faults=hook_errors,
                        return_all_solutions=compute_all_solutions,
                    )[0]
                else:
                    stabs_2_list = heuristic_verification_stabilizers(
                        self.state_prep, x_errors=not self.state_prep.zero_state, additional_faults=hook_errors
                    )[0]
                    stabs_2_list = [stabs_2_list]
                verify_2_list = [DeterministicVerification(stabs_2, {}) for stabs_2 in stabs_2_list]
                # check if better than normal verification
                anc_saved = (
                    sum(stabs_flagged_all)
                    - sum(stabs_flagged)
                    + self._layers[1][0].num_ancillas_verification()
                    - verify_2_list[0].num_ancillas_verification()
                )
                cnots_saved = (
                    2 * sum(stabs_flagged_all)
                    - 2 * sum(stabs_flagged)
                    + self._layers[1][0].num_cnots_verification()
                    - verify_2_list[0].num_cnots_verification()
                )
                if anc_saved > 0 or (anc_saved == 0 and cnots_saved > 0):
                    # hook propagation is better than hook correction
                    # compute deterministic verification
                    self._recompute_hook_propagation_corrections(
                        verify_2_list, verify, stabs_flagged, min_timeout, max_timeout, max_ancillas
                    )

    def get_solution(
        self,
        min_timeout: int = 1,
        max_timeout: int = 3600,
        max_ancillas: int | None = None,
        use_optimal_verification: bool = True,
    ) -> tuple[DeterministicVerification, DeterministicVerification]:
        """Returns a tuple representing the first layer, second layer and hook error corrections."""
        if max_ancillas is None:
            max_ancillas = self.code.Hx.shape[0] + self.code.Hz.shape[0]
        self.use_optimal_verification = use_optimal_verification

        self._compute_non_det_stabs(min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas)
        self._compute_det_corrections(
            min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, layer_idx=0
        )
        self._compute_hook_propagation_solutions(
            min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, compute_all_solutions=False
        )

        # if hook propagation is worse, compute the hook corrections and deterministic corrections
        if len(self._hook_propagation_solutions) == 0:
            self._compute_hook_corrections(min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas)
            self._compute_det_corrections(
                min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, layer_idx=1
            )
            if len(self._layers[1]) == 0:
                return self._layers[0][0], DeterministicVerification([], {})
            return self._layers[0][0], self._layers[1][0]
        # else return the hook propagation solution
        return self._hook_propagation_solutions[0]

    def get_global_solution(
        self,
        min_timeout: int = 1,
        max_timeout: int = 3600,
        max_ancillas: int | None = None,
    ) -> tuple[DeterministicVerification, DeterministicVerification]:
        """Returns the optimal non-deterministic verification stabilizers for the first and second layer regarding the number of ancillas and CNOTs."""
        if max_ancillas is None:
            max_ancillas = self.code.Hx.shape[0] + self.code.Hz.shape[0]

        self._compute_non_det_stabs(
            min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, compute_all_solutions=True
        )
        self._filter_and_stabs()
        self._compute_det_corrections(
            min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, layer_idx=0
        )
        self._compute_hook_propagation_solutions(
            min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, compute_all_solutions=False
        )

        # if hook propagation is worse, compute the hook corrections and deterministic corrections
        if len(self._hook_propagation_solutions) == 0:
            self._compute_hook_corrections(min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas)
            self._compute_det_corrections(
                min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, layer_idx=1
            )

            # compute the best solution
            best_stab_indices = [0, 0]
            for layer_idx in range(2):
                best_num_anc = 3 * max_ancillas
                best_num_cnots = 3 * max_ancillas * self.num_qubits
                for idx_verify, verify in enumerate(self._layers[layer_idx]):
                    num_anc = verify.num_ancillas_total()
                    num_cnots = verify.num_cnots_total()
                    if best_num_anc > num_anc or (best_num_anc == num_anc and best_num_cnots > num_cnots):
                        best_num_anc = num_anc
                        best_num_cnots = num_cnots
                        best_stab_indices[layer_idx] = idx_verify
            if len(self._layers[1]) == 0:
                return self._layers[0][best_stab_indices[0]], DeterministicVerification([], {})
            return self._layers[0][best_stab_indices[0]], self._layers[1][best_stab_indices[1]]

        # else return the hook propagation solution

        best_num_anc = 3 * max_ancillas
        best_num_cnots = 3 * max_ancillas * self.num_qubits
        best_solution = self._hook_propagation_solutions[0]
        for verify, verify_2 in self._hook_propagation_solutions[1:]:
            # check if better than overall best solution
            num_anc = verify.num_ancillas_total() + verify_2.num_ancillas_total()
            num_cnots = verify.num_cnots_total() + verify_2.num_cnots_total()
            if best_num_anc > num_anc or (best_num_anc == num_anc and best_num_cnots > num_cnots):
                best_num_anc = num_anc
                best_num_cnots = num_cnots
                # save the new verification
                best_solution = (verify, verify_2)
        return best_solution


def deterministic_correction(
    sp_circ: StatePrepCircuit,
    and_d3_verification_stabilizers: list[npt.NDArray[np.int8]],
    min_timeout: int = 1,
    max_timeout: int = 3600,
    max_ancillas: int | None = None,
    zero_state: bool = True,
    additional_faults: npt.NDArray[np.int8] | None = None,
) -> DeterministicCorrection:
    """Returns a deterministic verification for non-deterministic verification stabilizers.

    It computes the corresponding fault set and then solves the problem if finding optimal deterministic verification
    stabilizers for each non-deterministic verification outcome separately.

    Args:
        sp_circ: The state preparation circuit to compute the deterministic verification for.
        and_d3_verification_stabilizers: The non-deterministic verification stabilizers to be measured.
        min_timeout: The minimum time in seconds to run the verification stabilizers.
        max_timeout: The maximum time in seconds to run the verification stabilizers.
        max_ancillas: The maximum number of ancillas to use in the verification stabilizers.
        zero_state: If True, the X errors are considered, otherwise the Z errors are considered.
        additional_faults: Additional faults to consider in the fault set (e.g. hook errors).
    """
    num_and_stabs = len(and_d3_verification_stabilizers)
    num_qubits = sp_circ.code.n
    if max_ancillas is None:
        max_ancillas = sp_circ.code.Hx.shape[0] + sp_circ.code.Hz.shape[0]

    # get the fault set
    if additional_faults is not None:
        fault_set = sp_circ.combine_faults(additional_faults=additional_faults, x_errors=zero_state)
    else:
        fault_set = sp_circ.compute_fault_set(1, x_errors=zero_state)

    det_verify = {}
    for verify_outcome_int in range(1, 2**num_and_stabs):
        verify_outcome = _int_to_int8_array(verify_outcome_int, num_and_stabs)
        logger.info(
            f"Computing deterministic verification for non-det outcome {verify_outcome}: {verify_outcome_int}/{2**num_and_stabs - 1}"
        )

        # only consider errors that triggered the verification pattern
        errors_filtered = np.array([
            error
            for error in fault_set
            if np.array_equal(verify_outcome, [np.sum(m * error) % 2 for m in and_d3_verification_stabilizers])
        ])

        # append single-qubit errors that could have triggered the verification pattern
        for qubit in range(num_qubits):
            # compute error pattern of single-qubit error on qubit i
            error_pattern = [
                np.sum(m * np.eye(num_qubits, dtype=np.int8)[qubit]) % 2 for m in and_d3_verification_stabilizers
            ]
            for i in range(num_and_stabs):
                if np.array_equal(verify_outcome, error_pattern):
                    # if not already in the fault set
                    if len(errors_filtered) == 0:
                        errors_filtered = np.array([np.eye(num_qubits, dtype=np.int8)[qubit]])
                    elif not np.any(np.all(errors_filtered == np.eye(num_qubits, dtype=np.int8)[qubit], axis=1)):
                        errors_filtered = np.vstack((errors_filtered, np.eye(num_qubits, dtype=np.int8)[qubit]))
                else:
                    error_pattern[i] = 0

        # add the no-error case for the error being on one of the verification ancillas
        if np.sum(verify_outcome) == 1:
            errors_filtered = np.vstack((errors_filtered, np.zeros(num_qubits, dtype=np.int8)))
        # case of no errors or only one error is trivial
        if errors_filtered.shape[0] == 0:
            det_verify[verify_outcome_int] = (
                np.zeros((num_qubits, 0), dtype=np.int8),
                {0: np.zeros(num_qubits, dtype=np.int8), 1: np.zeros(num_qubits, dtype=np.int8)},
            )
        elif errors_filtered.shape[0] == 1:
            det_verify[verify_outcome_int] = (
                [np.zeros(num_qubits, dtype=np.int8)],
                {0: errors_filtered[0], 1: errors_filtered[0]},
            )
        else:
            det_verify[verify_outcome_int] = deterministic_correction_single_outcome(
                sp_circ, errors_filtered, min_timeout, max_timeout, max_ancillas, zero_state
            )
    return det_verify


def deterministic_correction_single_outcome(
    sp_circ: StatePrepCircuit,
    fault_set: npt.NDArray[np.int8],
    min_timeout: int,
    max_timeout: int,
    max_ancillas: int | None = None,
    zero_state: bool = True,
) -> Recovery:
    """Returns the deterministic recovery for a set of errors.

    Geometrically increases the number of ancilla qubits until a solution is found.
    Then, first the number of ancillas is optimized and then the number of CNOTs.

    Args:
        sp_circ: The state preparation circuit to compute the deterministic verification for.
        fault_set: The set of errors to consider for the deterministic verification.
        min_timeout: The minimum time in seconds to run the verification stabilizers.
        max_timeout: The maximum time in seconds to run the verification stabilizers.
        max_ancillas: The maximum number of ancillas to use in the verification stabilizers.
        zero_state: If True, the X errors are considered, otherwise the Z errors are considered.
    """
    num_anc = 1
    num_qubits = sp_circ.code.n
    if max_ancillas is None:
        max_ancillas = sp_circ.code.Hx.shape[0] + sp_circ.code.Hz.shape[0]

    def _func(num_anc: int) -> Recovery | None:
        return correction_stabilizers(sp_circ, fault_set, num_anc, num_anc * num_qubits, x_errors=zero_state)

    res = iterative_search_with_timeout(_func, num_anc, max_ancillas, min_timeout, max_timeout)

    assert res is not None, "No deterministic verification found."
    assert res[0], "No deterministic verification found."
    optimal_det_verify: Recovery = res[0]

    num_anc = res[1]
    logger.info(f"Found deterministic verification with {num_anc} ancillas.")

    while num_anc > 1:
        logger.info(f"Trying to reduce the number of ancillas to {num_anc - 1}.")
        det_verify: Recovery | str | None = run_with_timeout(_func, num_anc - 1, timeout=max_timeout)
        if det_verify and not isinstance(det_verify, str):
            optimal_det_verify = det_verify
            num_anc -= 1
        else:
            break
    logger.info(f"Optimal number of ancillas: {num_anc}.")

    # try to reduce the number of CNOTs
    def min_cnot_func(num_cnots: int) -> Recovery | None:
        return correction_stabilizers(sp_circ, fault_set, num_anc, num_cnots, x_errors=zero_state)

    num_cnots = 2
    while num_cnots > 1:
        # set the max number of CNOTs to the number returned by the previous step
        num_cnots = np.sum([np.sum(m) for m in optimal_det_verify[0]])

        logger.info(f"Trying to reduce the number of CNOTs to {num_cnots - 1}.")
        det_verify = run_with_timeout(min_cnot_func, num_cnots - 1, timeout=max_timeout)
        if det_verify and not isinstance(det_verify, str):
            optimal_det_verify = det_verify
            num_cnots -= 1
        else:
            break
    logger.info(f"Optimal number of CNOTs: {num_cnots}.")
    return optimal_det_verify


def correction_stabilizers(
    sp_circ: StatePrepCircuit,
    fault_set: npt.NDArray[np.int8],
    num_anc: int,
    num_cnot: int,
    x_errors: bool = True,
) -> Recovery | None:
    """Return deterministic verification stabilizers with corresponding corrections using z3."""
    gens = sp_circ.z_checks if x_errors else sp_circ.x_checks
    correction_gens = sp_circ.x_checks if x_errors else sp_circ.z_checks

    n_gens = gens.shape[0]
    n_corr_gens = correction_gens.shape[0]
    n_qubits = sp_circ.code.n
    n_errors = fault_set.shape[0]

    # Measurements are written as sums of generators
    # The variables indicate which generators are non-zero in the sum
    measurement_vars = [[z3.Bool(f"m_{anc}_{i}") for i in range(n_gens)] for anc in range(num_anc)]
    measurement_stabs = [vars_to_stab(vars_, gens) for vars_ in measurement_vars]

    # create "stabilizer degree of freedom" variables
    free_var = [[z3.Bool(f"free_{e}_{g}") for g in range(n_corr_gens)] for e in range(n_errors)]
    free_stabs = [vars_to_stab(vars_, correction_gens) for vars_ in free_var]

    # correction variables for each possible deterministic verification outcome
    corrections = [[z3.Bool(f"c_{anc}_{i}") for i in range(n_qubits)] for anc in range(2**num_anc)]

    solver = z3.Solver()

    # for each error, the pattern is computed and the corresponding correction is applied
    for idx_error, error in enumerate(fault_set):
        error_pattern = [odd_overlap(measurement, error) for measurement in measurement_stabs]
        for det_pattern, correction in enumerate(corrections):
            det_pattern_bool = _int_to_bool_array(det_pattern, num_anc)
            # check if error triggers the pattern
            triggered = symbolic_vector_eq(error_pattern, det_pattern_bool)
            # constraint: weight(error + correction + arbitrary free stabilizer) <= 1
            final_error = [
                z3.Xor(correction[i] if error[i] == 0 else z3.Not(correction[i]), free_stabs[idx_error][i])
                for i in range(n_qubits)
            ]
            solver.add(z3.If(triggered, z3.Sum(final_error) <= 1, True))

    # assert that not too many CNOTs are used
    solver.add(z3.PbLe([(measurement[q], 1) for measurement in measurement_stabs for q in range(n_qubits)], num_cnot))

    if solver.check() == z3.sat:
        return _extract_measurement_and_correction(
            solver.model(), gens, correction_gens, n_qubits, num_anc, measurement_vars, corrections
        )
    return None


def _extract_measurement_and_correction(
    model: z3.Model,
    gens: list[npt.NDArray[np.int8]],
    correction_gens: list[npt.NDArray[np.int8]],
    n_qubits: int,
    num_anc: int,
    measurement_vars: list[list[z3.BoolRef]],
    corrections: list[list[z3.BoolRef]],
) -> Recovery:
    """Extract deterministic verification stabilizers and corrections from sat z3 solver."""
    # get measurements
    actual_measurements = []
    for m in measurement_vars:
        v = np.zeros(len(gens[0]), dtype=np.int8)
        for g in range(len(gens)):
            if model[m[g]]:
                v += gens[g]
        actual_measurements.append(v % 2)

    # get corrections for each pattern
    actual_corrections = {}
    for outcome in range(2**num_anc):
        actual_correction = np.array(
            [int(bool(model[corrections[outcome][i]])) for i in range(n_qubits)], dtype=np.int8
        )

        if np.sum(actual_correction) == 0:
            actual_corrections[outcome] = actual_correction
        else:
            actual_corrections[outcome] = coset_leader(actual_correction, np.array(correction_gens))
    return actual_measurements, actual_corrections


def _int_to_bool_array(num: int, num_anc: int) -> npt.NDArray[np.bool_]:
    """Convert an integer to a boolean array of length num_anc corresponding to the binary representation of the integer."""
    return np.array([bool(num & (1 << i)) for i in range(num_anc)])[::-1]


def _int_to_int8_array(num: int, n_qubits: int) -> npt.NDArray[np.int8]:
    """Convert an integer to an int8 array of length n_qubits."""
    return np.array([int(bool(num & (1 << i))) for i in range(n_qubits)], dtype=np.int8)[::-1]
