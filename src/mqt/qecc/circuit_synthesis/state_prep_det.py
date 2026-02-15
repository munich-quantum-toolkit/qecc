# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Synthesizing deterministic state preparation circuits for d<5 CSS codes."""

from __future__ import annotations

import logging
from functools import partial
from itertools import product
from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import numpy.typing as npt
import z3

from .faults import PureFaultSet, coset_leader
from .state_prep import (
    all_gate_optimal_verification_stabilizers,
    get_hook_errors,
    heuristic_verification_stabilizers,
)
from .synthesis_utils import (
    iterative_search_with_timeout,
    odd_overlap,
    run_with_timeout,
    symbolic_vector_eq,
    vars_to_stab,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable

    from .state_prep import FaultyStatePrepCircuit


Recovery = tuple[list[npt.NDArray[np.int8]], dict[int, npt.NDArray[np.int8]]]
DeterministicCorrection = dict[
    int,
    tuple[npt.NDArray[np.int8], dict[int, npt.NDArray[np.int8]]]
    | tuple[list[npt.NDArray[np.int8]], dict[int, npt.NDArray[np.int8]]],
]
Verification = list[npt.NDArray[np.int8]]


class DeterministicVerification:
    """Class to store deterministic verification stabilizers and corrections."""

    def __init__(
        self,
        nd_verification_stabs: Verification,
        det_correction: DeterministicCorrection,
        hook_corrections: list[DeterministicCorrection] | None = None,
    ) -> None:
        """Initialize a deterministic verification object.

        Args:
            nd_verification_stabs: The non-deterministic verification stabilizers to be measured.
            det_correction: The deterministic correction for the non-deterministic verification stabilizers.
            hook_corrections: the hook corrections for the non-deterministic verification stabilizers.
        """
        self.stabs = nd_verification_stabs
        self.det_correction = det_correction
        self.hook_corrections: list[DeterministicCorrection] = [{}] * len(nd_verification_stabs)
        if hook_corrections:
            assert len(hook_corrections) == len(nd_verification_stabs)
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
        return int(np.sum([np.sum(m) for m in self.stabs]))

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

    def __init__(
        self, state_prep: FaultyStatePrepCircuit, use_optimal_verification: bool = True, verify_x_first: bool = True
    ) -> None:
        """Initialize the deterministic verification helper with a given state preparation circuit.

        Args:
            state_prep: The state preparation circuit to compute the deterministic verification for (must be a CSS code and d<5).
            use_optimal_verification: If True, the optimal verification stabilizers are computed, otherwise heuristic verification stabilizers are used.
            verify_x_first: If True, X-errors are verified first.
        """
        self.state_prep = state_prep
        self.code = state_prep.circ.get_code()
        assert max(state_prep.max_x_errors, state_prep.max_z_errors) < 5, "Only d=3 and d=4 codes are supported."
        self.num_qubits = self.code.n
        self.verify_x_first = verify_x_first
        self.use_optimal_verification = use_optimal_verification
        # Variable to store the deterministic verification stabilizers and corrections for the two layers
        self._layers: list[list[DeterministicVerification]] = [[], []]
        # Variable to store the deterministic verification stabilizers and corrections for the hook propagation solution
        self._hook_propagation_solutions: list[tuple[DeterministicVerification, DeterministicVerification]] = []

        self.state_prep.compute_fault_sets()
        if self.verify_x_first:
            self.fault_sets = [self.state_prep.x_fault_sets[0], self.state_prep.z_fault_sets[0]]
            self.checks = [self.state_prep.z_checks, self.state_prep.x_checks]
        else:
            self.fault_sets = [self.state_prep.z_fault_sets[0], self.state_prep.x_fault_sets[0]]
            self.checks = [self.state_prep.x_checks, self.state_prep.z_checks]

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
        for layer in range(2):
            logger.info(f"Computing non-deterministic verification stabilizers for layer {layer + 1}.")
            stabs_all = all_gate_optimal_verification_stabilizers(
                [self.fault_sets[layer]],
                self.checks[layer],
                min_timeout=min_timeout,
                max_timeout=max_timeout,
                max_ancillas=max_ancillas,
                return_all_solutions=compute_all_solutions,
            )[0]
            self._layers[layer] = [DeterministicVerification(stabs, {}) for stabs in stabs_all]

    def _compute_det_corrections(
        self, min_timeout: int = 1, max_timeout: int = 3600, max_ancillas: int | None = None, layer_idx: int = 0
    ) -> None:
        """Returns the deterministic verification stabilizers for the first layer of non-deterministic verification stabilizers."""
        logger.info(f"Computing deterministic verification for layer {layer_idx}.")
        for verify_idx, verify in enumerate(self._layers[layer_idx]):
            logger.info(
                f"Computing deterministic verification for non-det verification {verify_idx + 1} / {len(self._layers[layer_idx])}."
            )
            self._layers[layer_idx][verify_idx].det_correction = deterministic_correction(
                self.fault_sets[layer_idx],
                self.checks[layer_idx],
                self.checks[1 - layer_idx],
                verify.stabs,
                min_timeout=min_timeout,
                max_timeout=max_timeout,
                max_ancillas=max_ancillas,
            )

    @staticmethod
    def _trivial_hook_errors(hook_errors: PureFaultSet, stabs: npt.NDArray[np.int8]) -> bool:
        """Checks if the hook errors are trivial (stabilizers) by checking if the rank of the code stabilizers is the same.

        Args:
            hook_errors: The hook errors to check.
            stabs: The CSS code to check the rank of the stabilizers.

        Returns:
            bool: True if all hook errors are trivial, False otherwise.
        """
        rank = mod2.rank(stabs)
        n = stabs.shape[1]
        for error in hook_errors:
            single_qubit_deviation = (error + np.eye(n, dtype=np.int8)) % 2
            stabs_plus_single_qubit = np.concatenate([stabs, single_qubit_deviation], axis=0)
            if not any(mod2.rank(stabs_plus_single_qubit[:, :n]) == rank for _ in range(n)):
                return False
        return True

    def _compute_hook_corrections(
        self, min_timeout: int = 1, max_timeout: int = 3600, max_ancillas: int | None = None
    ) -> None:
        """Computes the additional stabilizers to measure with corresponding corrections for the hook errors of each stabilizer measurement in layer 2."""
        for layer_idx in range(2):
            logger.info(f"Computing deterministic verification for hook errors of layer {layer_idx + 1} / 2.")
            for verify_idx, verify in enumerate(self._layers[layer_idx]):
                logger.info(
                    f"Computing deterministic hook correction for non-det verification {verify_idx + 1} / {len(self._layers[layer_idx])}."
                )

                # No stabilizers are measured, so no hook errors can occur
                if not verify.stabs:
                    self._layers[layer_idx][verify_idx].hook_corrections = [{}] * len(verify.stabs)
                    continue

                for stab_idx, stab in enumerate(verify.stabs):
                    hook_errors = get_hook_errors([stab])
                    if self._trivial_hook_errors(hook_errors, self.checks[layer_idx]):
                        continue

                    # hook errors are non-trivial
                    # add case of error on hook ancilla
                    hook_errors = PureFaultSet.from_fault_array(
                        np.vstack((hook_errors.to_array(), np.zeros(self.num_qubits, dtype=np.int8)))
                    )
                    self._layers[layer_idx][verify_idx].hook_corrections[stab_idx] = {
                        1: deterministic_correction_single_outcome(
                            hook_errors,
                            self.checks[1 - layer_idx],
                            self.checks[layer_idx],
                            min_timeout=min_timeout,
                            max_timeout=max_timeout,
                            max_ancillas=max_ancillas,
                        )
                    }

    def _filter_nd_stabs(self) -> None:
        """Only keep the best non-deterministic verification stabilizers with minimal number of ancillas and CNOTs."""
        self._best_num_anc = 0
        self._best_num_cnots = 0

        for layer_idx, layer in enumerate(self._layers[:2]):
            # Compute best numbers and indices
            best_num_anc = int(1e6)
            best_num_cnots = int(1e6)
            best_case_indices = []

            for idx_verify, verify in enumerate(layer):
                num_anc = verify.num_ancillas_verification() + verify.num_ancillas_hooks()
                num_cnot = verify.num_cnots_verification() + verify.num_cnots_hooks()

                if (num_anc < best_num_anc) or (num_anc == best_num_anc and num_cnot < best_num_cnots):
                    best_num_anc = num_anc
                    best_num_cnots = num_cnot
                    best_case_indices = [idx_verify]
                elif num_anc == best_num_anc and num_cnot == best_num_cnots:
                    best_case_indices.append(idx_verify)

            # Filter out all but the best cases
            self._layers[layer_idx] = [layer[idx] for idx in best_case_indices]

            # Update the best numbers
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
                self.fault_sets[1],
                self.checks[1],
                self.checks[0],
                verify_2.stabs,
                min_timeout=min_timeout,
                max_timeout=max_timeout,
                max_ancillas=max_ancillas,
            )
            for stab_idx, stab in enumerate(verify_2.stabs):
                hook_errors_2 = get_hook_errors([stab])
                if self._trivial_hook_errors(hook_errors_2, self.checks[1]):
                    verify_2_list[verify_2_idx].hook_corrections[stab_idx] = {}
                else:
                    hook_errors_2.add_faults(np.zeros(self.num_qubits, dtype=np.int8))
                    verify_2_list[verify_2_idx].hook_corrections[stab_idx] = {
                        1: deterministic_correction_single_outcome(
                            hook_errors_2,
                            self.checks[1],
                            self.checks[0],
                            min_timeout=min_timeout,
                            max_timeout=max_timeout,
                            max_ancillas=max_ancillas,
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
                        get_hook_errors([verify.stabs[idx]]),
                        self.checks[1],
                        self.checks[0],
                        min_timeout=min_timeout,
                        max_timeout=max_timeout,
                        max_ancillas=max_ancillas,
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
            return

        for verify in self._layers[0]:
            logger.info(f"Computing hook propagation solutions for verification {verify} / {len(self._layers[0])}.")
            stabs_flagged_all = [
                not self._trivial_hook_errors(get_hook_errors([stab]), self.checks[0]) for stab in verify.stabs
            ]

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
                hook_errors = PureFaultSet(self.num_qubits)
                for idx, flag in enumerate(stabs_flagged):
                    if not flag:
                        hook_errors.combine(get_hook_errors([verify.stabs[idx]]), inplace=True)

                if self._trivial_hook_errors(hook_errors, self.checks[1]):
                    continue
                # hook errors require different verification in second layer
                # compute new verification
                fault_set = self.state_prep.combine_faults(hook_errors, x_errors=not self.verify_x_first, reduce=True)
                if self.use_optimal_verification:
                    stabs_2_list = all_gate_optimal_verification_stabilizers(
                        fault_set,
                        self.checks[1],
                        min_timeout=min_timeout,
                        max_timeout=max_timeout,
                        max_ancillas=max_ancillas,
                        return_all_solutions=compute_all_solutions,
                    )[0]
                else:
                    stabs_2_list_ = heuristic_verification_stabilizers(fault_set, self.checks[1])[0]
                    stabs_2_list = [stabs_2_list_]
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
                    logger.info(
                        f"Hook propagation is better than hook correction for verification {verify} / {len(self._layers[0])}."
                    )
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

        # Handle the first layer
        if self._layers[0]:
            self._compute_det_corrections(
                min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, layer_idx=0
            )

            # If the second layer is empty, return the first layer and a trivial deterministic verification
            if not self._layers[1]:
                return self._layers[0][0], DeterministicVerification([], {})

            # Compute hook propagation solutions
            self._compute_hook_propagation_solutions(
                min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, compute_all_solutions=False
            )
            # If no hook propagation solutions exist, compute hook corrections and deterministic corrections
            if not self._hook_propagation_solutions:
                self._compute_hook_corrections(
                    min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas
                )
                self._compute_det_corrections(
                    min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, layer_idx=1
                )
                return self._layers[0][0], self._layers[1][0]

            # Return the best hook propagation solution
            return self._hook_propagation_solutions[0]

        # Handle the second layer if the first layer is empty
        if self._layers[1]:
            self._compute_det_corrections(
                min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, layer_idx=1
            )
            return DeterministicVerification([], {}), self._layers[1][0]

        # Trivial case: no verification stabilizers
        return DeterministicVerification([], {}), DeterministicVerification([], {})

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

        if not self._layers[0] and not self._layers[1]:
            # Trivial case: no verification stabilizers
            return DeterministicVerification([], {}), DeterministicVerification([], {})

        if not self._layers[0] and self._layers[1]:
            self._compute_det_corrections(
                min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, layer_idx=1
            )
            return DeterministicVerification([], {}), self._layers[1][0]

        if self._layers[0] and not self._layers[1]:
            self._compute_det_corrections(
                min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, layer_idx=0
            )
            return self._layers[0][0], DeterministicVerification([], {})

        # Both layers are non-trivial -> find best combination
        self._compute_det_corrections(
            min_timeout=min_timeout, max_timeout=max_timeout, max_ancillas=max_ancillas, layer_idx=0
        )
        self._filter_nd_stabs()
        # Compute hook propagation solutions
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

            if len(self._layers[0]) == 0:
                return DeterministicVerification([], {}), self._layers[1][best_stab_indices[1]]

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
    fault_set: PureFaultSet,
    verification_gens: npt.NDArray[np.int8],
    correction_gens: npt.NDArray[np.int8],
    nd_d3_verification_stabilizers: list[npt.NDArray[np.int8]],
    min_timeout: int = 1,
    max_timeout: int = 3600,
    max_ancillas: int | None = None,
) -> DeterministicCorrection:
    """Returns a deterministic verification for non-deterministic verification stabilizers.

    It computes the corresponding fault set and then solves the problem if finding optimal deterministic verification
    stabilizers for each non-deterministic verification outcome separately.

    Args:
        fault_set: The set of errors to consider for the deterministic verification.
        verification_gens: The stabilizer generators used for verification.
        correction_gens: The stabilizer generators the faults can be multiplied by.
        nd_d3_verification_stabilizers: The non-deterministic verification stabilizers to be measured.
        min_timeout: The minimum time in seconds to run the verification stabilizers.
        max_timeout: The maximum time in seconds to run the verification stabilizers.
        max_ancillas: The maximum number of ancillas to use in the verification stabilizers.
    """
    num_nd_stabs = len(nd_d3_verification_stabilizers)
    num_qubits = fault_set.num_qubits
    if max_ancillas is None:
        max_ancillas = verification_gens.shape[0] + correction_gens.shape[0]

    logger.info("Fault set has %s faults.", len(fault_set))
    logger.info("Non-deterministic verification stabilizers: %s", nd_d3_verification_stabilizers)

    det_verify: DeterministicCorrection = {}
    for verify_outcome_int in range(1, 2**num_nd_stabs):
        verify_outcome = _int_to_int8_array(verify_outcome_int, num_nd_stabs)
        logger.info(
            f"Computing deterministic verification for non-det outcome {verify_outcome}: {verify_outcome_int}/{2**num_nd_stabs - 1}"
        )

        # only consider errors that triggered the verification pattern
        def triggers_pattern(fault: npt.NDArray[np.int8], verify_outcome: npt.NDArray[np.int8]) -> bool:
            return np.array_equal(verify_outcome, nd_d3_verification_stabilizers @ fault % 2)

        errors_filtered = fault_set.filter_faults(
            partial(triggers_pattern, verify_outcome=verify_outcome), inplace=False
        )
        # append single-qubit errors that could have triggered the verification pattern
        identity_matrix = np.eye(num_qubits, dtype=np.int8)
        for qubit in range(num_qubits):
            single_qubit_error = identity_matrix[qubit]
            # compute error pattern of single-qubit error on qubit i
            error_pattern = [np.sum(m * single_qubit_error) % 2 for m in nd_d3_verification_stabilizers]

            for i in range(num_nd_stabs):
                if np.array_equal(verify_outcome, error_pattern):
                    # Add the fault to the set if not already present
                    errors_filtered.add_fault(single_qubit_error)
                else:
                    error_pattern[i] = 0
            errors_filtered.remove_duplicates()

        # add the no-error case for the error being on one of the verification ancillas
        if np.sum(verify_outcome) == 1:
            errors_filtered.add_fault(np.zeros(num_qubits, dtype=np.int8))
        # case of no errors or only one error is trivial
        if len(errors_filtered) == 0:
            det_verify[verify_outcome_int] = (
                np.zeros((num_qubits, 0), dtype=np.int8),
                {0: np.zeros(num_qubits, dtype=np.int8), 1: np.zeros(num_qubits, dtype=np.int8)},
            )
        elif len(errors_filtered) == 1:
            det_verify[verify_outcome_int] = (
                [np.zeros(num_qubits, dtype=np.int8)],
                {0: errors_filtered[0], 1: errors_filtered[0]},
            )
        else:
            det_verify[verify_outcome_int] = deterministic_correction_single_outcome(
                errors_filtered, verification_gens, correction_gens, min_timeout, max_timeout, max_ancillas
            )
    return det_verify


def deterministic_correction_single_outcome(
    fault_set: PureFaultSet,
    verification_gens: npt.NDArray[np.int8],
    correction_gens: npt.NDArray[np.int8],
    min_timeout: int,
    max_timeout: int,
    max_ancillas: int | None = None,
) -> Recovery:
    """Returns the deterministic recovery for a set of errors.

    Geometrically increases the number of ancilla qubits until a solution is found.
    Then, first the number of ancillas is optimized and then the number of CNOTs.

    Args:
        fault_set: The set of errors to consider for the deterministic verification.
        verification_gens: The stabilizer generators used for verification.
        correction_gens: The stabilizer generators the faults can be multiplied by.
        min_timeout: The minimum time in seconds to run the verification stabilizers.
        max_timeout: The maximum time in seconds to run the verification stabilizers.
        max_ancillas: The maximum number of ancillas to use in the verification stabilizers.
    """
    num_anc = 1
    num_qubits = fault_set.num_qubits
    if max_ancillas is None:
        max_ancillas = verification_gens.shape[0] + correction_gens.shape[0]

    def _func(num_anc: int) -> Recovery | None:
        return correction_stabilizers(fault_set, verification_gens, correction_gens, num_anc, num_anc * num_qubits)

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
        return correction_stabilizers(fault_set, verification_gens, correction_gens, num_anc, num_cnots)

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
    fault_set: PureFaultSet,
    measurement_gens: npt.NDArray[np.int8],
    correction_gens: npt.NDArray[np.int8],
    num_anc: int,
    num_cnot: int,
) -> Recovery | None:
    """Return deterministic verification stabilizers with corresponding corrections using z3."""
    n_gens = measurement_gens.shape[0]
    n_corr_gens = correction_gens.shape[0]
    n_qubits = fault_set.num_qubits
    n_errors = len(fault_set)

    # Measurements are written as sums of generators
    # The variables indicate which generators are non-zero in the sum
    measurement_vars = [[z3.Bool(f"m_{anc}_{i}") for i in range(n_gens)] for anc in range(num_anc)]
    measurement_stabs = [vars_to_stab(vars_, measurement_gens) for vars_ in measurement_vars]

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
            solver.model(), measurement_gens, correction_gens, n_qubits, num_anc, measurement_vars, corrections
        )
    return None


def _extract_measurement_and_correction(
    model: z3.Model,
    gens: npt.NDArray[np.int8],
    correction_gens: npt.NDArray[np.int8],
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
