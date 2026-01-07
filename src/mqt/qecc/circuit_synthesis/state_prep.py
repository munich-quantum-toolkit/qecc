# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Synthesizing state preparation circuits for CSS codes."""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import z3
from qiskit.circuit import AncillaRegister, ClassicalRegister, QuantumCircuit, QuantumRegister

from .circuits import CNOTCircuit
from .faults import PureFaultSet, coset_leader, product_fault_set
from .synthesis_utils import (
    heuristic_gaussian_elimination,
    iterative_search_with_timeout,
    measure_flagged,
    odd_overlap,
    optimal_elimination,
    run_with_timeout,
    vars_to_stab,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import numpy.typing as npt

    from ..codes.css_code import CSSCode


class FaultyStatePrepCircuit:
    """Represents a state preparation circuit for a CSS code."""

    def __init__(self, circ: CNOTCircuit, max_x_errors: int, max_z_errors: int) -> None:
        """Initialize a state preparation circuit.

        Args:
            circ: The state preparation circuit.
            max_x_errors: Maximum number of independent x errors that can happen in the circuit.
            max_z_errors: Maximum number of independent z errors that can happen in the circuit.
        """
        if not circ.is_state():
            msg = "Input circuit is not a state!"
            raise ValueError(msg)

        self.circ = circ
        code = circ.get_code()
        self.num_qubits = circ.num_qubits()
        self.max_x_errors = max_x_errors
        self.max_z_errors = max_z_errors

        if self.max_x_errors == 0 and self.max_z_errors == 0:  # pragma: no cover
            warnings.warn(
                "Initializing FaultyStatePrepCircuit with max_errors=0. "
                "This might be a mistake. Use set_max_errors to manually set the maximum number of X and Z errors that can occur in the circuit.",
                UserWarning,
                stacklevel=2,
            )

        self.x_checks = code.Hx
        self.z_checks = code.Hz
        self.x_fault_sets: list[PureFaultSet] = []
        self.z_fault_sets: list[PureFaultSet] = []
        self.x_fault_sets_unreduced: list[PureFaultSet] = []
        self.z_fault_sets_unreduced: list[PureFaultSet] = []

    def compute_fault_sets(self, reduce: bool = True) -> None:
        """Compute the fault sets for the state preparation circuit."""
        self.compute_fault_set(self.max_x_errors, x_errors=True, reduce=reduce)
        self.compute_fault_set(self.max_z_errors, x_errors=False, reduce=reduce)

    def compute_fault_set(self, num_errors: int = 1, x_errors: bool = True, reduce: bool = True) -> PureFaultSet:
        """Compute the fault set of the state.

        Args:
            num_errors: The number of independent errors to propagate through the circuit.
            x_errors: If True, compute the fault set for X errors. If False, compute the fault set for Z errors.
            reduce: If True, reduce the fault set by the stabilizers of the code to reduce weights.

        Returns:
            The fault set of the state.
        """
        if num_errors == 0:
            return PureFaultSet(self.circ.num_qubits())
        fault_sets = self.x_fault_sets if x_errors else self.z_fault_sets
        fault_sets_unreduced = self.x_fault_sets_unreduced if x_errors else self.z_fault_sets_unreduced
        if len(fault_sets) >= num_errors:
            return fault_sets[num_errors - 1]  # return cached value

        if num_errors <= 0:
            msg = "Cannot compute fault set for less than 1 error."
            raise ValueError(msg)

        if num_errors == 1:
            logger.info("Computing fault set for 1 error.")
            fs = PureFaultSet.from_cnot_circuit(self.circ, kind="X" if x_errors else "Z")
        else:
            logger.info(f"Computing fault set for {num_errors} errors.")
            self.compute_fault_set(num_errors - 1, x_errors, reduce=reduce)
            faults = fault_sets[num_errors - 2]
            single_faults = fault_sets_unreduced[0]

            fs = product_fault_set(faults, single_faults)
            fs.remove_zero_rows()
            fs.remove_duplicates()

        fault_sets_unreduced.append(fs.copy())

        # reduce faults by stabilizer
        stabs = self.x_checks if x_errors else self.z_checks

        if reduce:
            logger.info("Removing stabilizer equivalent faults.")
            fs.remove_equivalent(stabs)

        logger.info("Removing low-weight faults.")
        fs.filter_by_weight_at_least(num_errors + 1, stabs)
        fault_sets.append(fs)

        return fs

    def set_max_errors(self, max_x_errors: int, max_z_errors: int) -> None:
        """Set maximum errors. Also resets cached fault sets."""
        self.max_x_errors = max_x_errors
        self.max_z_errors = max_z_errors

    def combine_faults(
        self, additional_faults: PureFaultSet, x_errors: bool = True, reduce: bool = False
    ) -> list[PureFaultSet]:
        """Combine fault sets of circuit with additional independent faults.

        Args:
            additional_faults: The additional faults to combine with the fault set of the circuit.
            x_errors: If True, combine the fault sets for X errors. If False, combine the fault sets for Z errors.
            reduce: If True, stabilizer-equivalent errors will be removed after fault set construction.
        """
        self.compute_fault_sets()
        fault_sets = self.x_fault_sets if x_errors else self.z_fault_sets
        fault_sets_unreduced = self.x_fault_sets_unreduced if x_errors else self.z_fault_sets_unreduced
        stabs = self.x_checks if x_errors else self.z_checks
        max_errors = self.max_x_errors if x_errors else self.max_z_errors

        if len(additional_faults) == 0:
            return fault_sets

        new_products: list[PureFaultSet] = []
        new_products.append(additional_faults)
        for i in range(1, max_errors):
            single_faults = new_products[0]
            last_faults = new_products[i - 1]
            new_products.append(product_fault_set(single_faults, last_faults))

        # new_fault_sets_unreduced: list[PureFaultSet] = []
        new_fault_sets = []
        for i in range(max_errors):
            fs_opt = fault_sets_unreduced[i]
            fs = fs_opt.copy()
            fs.combine(new_products[i], inplace=True)
            for j in range(i):
                k = i - j - 1
                prod = product_fault_set(fault_sets_unreduced[j], new_products[k])
                fs.combine(prod, inplace=True)
            # new_fault_sets_unreduced.append(fs.copy())
            if reduce:
                fs.remove_equivalent(stabs)
            fs.filter_by_weight_at_least(i + 2, stabs)
            new_fault_sets.append(fs)

        return new_fault_sets


def _build_state_prep_circuit_from_back(
    checks: npt.NDArray[np.int8], cnots: list[tuple[int, int]], zero_state: bool = True
) -> CNOTCircuit:
    cnots.reverse()
    if zero_state:
        hadamards = np.where(np.sum(checks, axis=0) != 0)[0]
    else:
        hadamards = np.where(np.sum(checks, axis=0) == 0)[0]
        cnots = [(j, i) for i, j in cnots]
    non_hadamards = [i for i in range(checks.shape[1]) if i not in hadamards]
    return CNOTCircuit.from_cnot_list(cnots, initialize_z=non_hadamards, initialize_x=hadamards)


def heuristic_prep_circuit(
    code: CSSCode, optimize_depth: bool = True, zero_state: bool = True
) -> FaultyStatePrepCircuit:
    """Return a circuit that prepares the +1 eigenstate of the code w.r.t. the Z or X basis.

    Args:
        code: The CSS code to prepare the state for.
        optimize_depth: If True, optimize the depth of the circuit. This may lead to a higher number of CNOTs.
        zero_state: If True, prepare the +1 eigenstate of the Z basis. If False, prepare the +1 eigenstate of the X basis.
    """
    logger.info("Starting heuristic state preparation.")

    checks = code.Hx if zero_state else code.Hz
    assert checks is not None
    checks, cnots = heuristic_gaussian_elimination(checks, parallel_elimination=optimize_depth)

    circ = _build_state_prep_circuit_from_back(checks, cnots, zero_state)
    return FaultyStatePrepCircuit(circ, code.x_distance // 2, code.z_distance // 2)


def depth_optimal_prep_circuit(
    code: CSSCode,
    zero_state: bool = True,
    min_depth: int = 1,
    max_depth: int = 10,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> FaultyStatePrepCircuit | None:
    """Synthesize a state preparation circuit for a CSS code that minimizes the circuit depth.

    Args:
        code: The CSS code to prepare the state for.
        zero_state: If True, prepare the +1 eigenstate of the Z basis. If False, prepare the +1 eigenstate of the X basis.
        min_depth: minimum depth to start with
        max_depth: maximum depth to reach
        min_timeout: minimum timeout to start with
        max_timeout: maximum timeout to reach
    """
    checks = code.Hx if zero_state else code.Hz
    assert checks is not None
    rank = mod2.rank(checks)
    res = optimal_elimination(
        checks,
        lambda checks: final_matrix_constraint(checks, rank),
        "parallel_ops",
        min_param=min_depth,
        max_param=max_depth,
        min_timeout=min_timeout,
        max_timeout=max_timeout,
    )
    if res is None:
        return None
    checks, cnots = res
    circ = _build_state_prep_circuit_from_back(checks, cnots, zero_state)
    return FaultyStatePrepCircuit(circ, code.x_distance // 2, code.z_distance // 2)


def gate_optimal_prep_circuit(
    code: CSSCode,
    zero_state: bool = True,
    min_gates: int = 1,
    max_gates: int = 10,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> FaultyStatePrepCircuit | None:
    """Synthesize a state preparation circuit for a CSS code that minimizes the number of gates.

    Args:
        code: The CSS code to prepare the state for.
        zero_state: If True, prepare the +1 eigenstate of the Z basis. If False, prepare the +1 eigenstate of the X basis.
        min_gates: minimum number of gates to start with
        max_gates: maximum number of gates to reach
        min_timeout: minimum timeout to start with
        max_timeout: maximum timeout to reach
    """
    checks = code.Hx if zero_state else code.Hz
    assert checks is not None
    rank = mod2.rank(checks)
    res = optimal_elimination(
        checks,
        lambda checks: final_matrix_constraint(checks, rank),
        "column_ops",
        min_param=min_gates,
        max_param=max_gates,
        min_timeout=min_timeout,
        max_timeout=max_timeout,
    )
    if res is None:
        return None
    checks, cnots = res
    circ = _build_state_prep_circuit_from_back(checks, cnots, zero_state)
    return FaultyStatePrepCircuit(circ, code.x_distance // 2, code.z_distance // 2)


def gate_optimal_verification_stabilizers(
    fault_sets: list[PureFaultSet],
    stabs: npt.NDArray[np.int8],
    min_timeout: int = 1,
    max_timeout: int = 3600,
    max_ancillas: int | None = None,
) -> list[list[npt.NDArray[np.int8]]]:
    """Return verification stabilizers for the given fault sets..

    The method uses an iterative search to find the optimal set of stabilizers by repeatedly computing the optimal circuit for each number of ancillas and cnots. This is repeated for each number of independent correctable errors in the state preparation circuit. Thus the verification circuit is constructed of multiple "layers" of stabilizers, each layer corresponding to a fault set it verifies.

    Args:
        fault_sets: List of fault sets to verify.
        stabs: The stabilizer generators to verify the fault sets.
        min_timeout: The minimum time to allow each search to run for.
        max_timeout: The maximum time to allow each search to run for.
        max_ancillas: The maximum number of ancillas to allow in each layer verification circuit.

    Returns:
        A list of stabilizers for each number of errors to verify the state preparation circuit.
    """
    return [
        layers[0] if layers != [] else []
        for layers in all_gate_optimal_verification_stabilizers(
            fault_sets,
            stabs,
            min_timeout,
            max_timeout,
            max_ancillas,
            return_all_solutions=False,
        )
    ]


def all_gate_optimal_verification_stabilizers(
    fault_sets: list[PureFaultSet],
    stabs: npt.NDArray[np.int8],
    min_timeout: int = 1,
    max_timeout: int = 3600,
    max_ancillas: int | None = None,
    return_all_solutions: bool = False,
) -> list[list[list[npt.NDArray[np.int8]]]]:
    """Return all equivalent verification stabilizers for the given fault sets.

    The method uses an iterative search to find the optimal set of stabilizers by repeatedly computing the optimal circuit for each number of ancillas and cnots. This is repeated for each number of independent correctable errors in the state preparation circuit. Thus the verification circuit is constructed of multiple "layers" of stabilizers, each layer corresponding to a fault set it verifies.

    Args:
        fault_sets: List of fault sets to verify.
        stabs: The stabilizer generators to verify the fault sets.
        min_timeout: The minimum time to allow each search to run for.
        max_timeout: The maximum time to allow each search to run for.
        max_ancillas: The maximum number of ancillas to allow in each layer verification circuit.
        additional_faults: Faults to verify in addition to the faults propagating in the state preparation circuit.
        return_all_solutions: If False only the first solution for each number of errors is returned. If True all solutions are returned.

    Returns:
        A list of all equivalent stabilizers for each number of errors to verify the state preparation circuit.
    """
    n_layers = len(fault_sets)
    layers: list[list[list[npt.NDArray[np.int8]]]] = [[] for _ in range(n_layers)]
    if max_ancillas is None:
        max_ancillas = stabs.shape[0]

    # Find the optimal circuit for every number of errors in the preparation circuit
    for layer in range(n_layers):
        logger.info(f"Finding verification stabilizers for {layer + 1} errors")
        faults = fault_sets[layer]

        if len(faults) == 0:
            logger.info(f"No non-trivial faults for {layer + 1} errors")
            layers[layer] = []
            continue
        # Start with maximal number of ancillas
        # Minimal CNOT solution must be achievable with these
        num_anc = max_ancillas
        min_cnots: int = np.min(np.sum(stabs, axis=1))
        max_cnots = int(np.sum(stabs))

        logger.info(
            f"Finding verification stabilizers for {layer + 1} errors with {min_cnots} to {max_cnots} CNOTs using {num_anc} ancillas"
        )

        def fun(num_cnots: int) -> list[npt.NDArray[np.int8]] | None:
            return verification_stabilizers(faults, stabs, num_anc, num_cnots)  # noqa: B023

        res = iterative_search_with_timeout(
            fun,
            min_cnots,
            max_cnots,
            min_timeout,
            max_timeout,
        )

        if res is not None:
            measurements, num_cnots = res
        else:
            measurements = None

        if measurements is None:
            logger.info(f"No verification stabilizers found for {layer + 1} errors")
            return []  # No solution found

        logger.info(f"Found verification stabilizers for {layer + 1} errors with {num_cnots} CNOTs")
        # If any measurements are unused we can reduce the number of ancillas at least by that
        measurements = [m for m in measurements if np.any(m)]
        num_anc = len(measurements)
        # Iterate backwards to find the minimal number of cnots
        logger.info(f"Finding minimal number of CNOTs for {layer + 1} errors")

        def search_cnots(num_cnots: int) -> list[npt.NDArray[np.int8]] | None:
            return verification_stabilizers(faults, stabs, num_anc, num_cnots)  # noqa: B023

        while num_cnots - 1 > 0:
            logger.info(f"Trying {num_cnots - 1} CNOTs")

            cnot_opt = run_with_timeout(
                search_cnots,
                num_cnots - 1,
                timeout=max_timeout,
            )
            if cnot_opt and not isinstance(cnot_opt, str):
                num_cnots -= 1
                measurements = cnot_opt
            else:
                break
        logger.info(f"Minimal number of CNOTs for {layer + 1} errors is: {num_cnots}")

        # If the number of CNOTs is minimal, we can reduce the number of ancillas
        logger.info(f"Finding minimal number of ancillas for {layer + 1} errors")
        while num_anc - 1 > 0:
            logger.info(f"Trying {num_anc - 1} ancillas")

            def search_anc(num_anc: int) -> list[npt.NDArray[np.int8]] | None:
                return verification_stabilizers(faults, stabs, num_anc, num_cnots)  # noqa: B023

            anc_opt = run_with_timeout(
                search_anc,
                num_anc - 1,
                timeout=max_timeout,
            )
            if anc_opt and not isinstance(anc_opt, str):
                num_anc -= 1
                measurements = anc_opt
            else:
                break
        logger.info(f"Minimal number of ancillas for {layer + 1} errors is: {num_anc}")
        if not return_all_solutions:
            layers[layer] = [measurements]
        else:
            all_stabs = all_verification_stabilizers(faults, stabs, num_anc, num_cnots, return_all_solutions=True)
            if all_stabs:
                layers[layer] = all_stabs
                logger.info(f"Found {len(layers[layer])} equivalent solutions for {layer} errors")

    return layers


def _verification_circuit(
    sp_circ: FaultyStatePrepCircuit,
    verification_stabs_fun: Callable[[list[PureFaultSet], npt.NDArray[np.int8]], list[list[npt.NDArray[np.int8]]]],
    only_first_layer: bool = True,
    verify_x_first: bool = True,
    flag_first_layer: bool = False,
) -> QuantumCircuit:
    logger.info("Finding verification stabilizers for the state preparation circuit")

    sp_circ.compute_fault_sets(reduce=True)
    if verify_x_first:
        first_fault_sets = sp_circ.x_fault_sets
        first_checks = sp_circ.z_checks
        if not only_first_layer:
            second_fault_sets = sp_circ.z_fault_sets
            second_checks = sp_circ.x_checks
    else:
        first_fault_sets = sp_circ.z_fault_sets
        first_checks = sp_circ.x_checks
        if not only_first_layer:
            second_fault_sets = sp_circ.x_fault_sets
            second_checks = sp_circ.z_checks

    layers_1 = verification_stabs_fun(first_fault_sets, first_checks)
    measurements_1 = [measurement for layer in layers_1 for measurement in layer]
    if not flag_first_layer:
        if measurements_1:
            additional_errors = get_hook_errors(measurements_1)
            extended_fault_sets = sp_circ.combine_faults(
                additional_faults=additional_errors, x_errors=not verify_x_first
            )
        else:
            extended_fault_sets = second_fault_sets

        if not only_first_layer:
            layers_2 = verification_stabs_fun(extended_fault_sets, second_checks)
    elif not only_first_layer:
        layers_2 = verification_stabs_fun(second_fault_sets, second_checks)

    measurements_2 = [measurement for layer in layers_2 for measurement in layer] if not only_first_layer else []

    z_measurements = measurements_1 if verify_x_first else measurements_2
    x_measurements = measurements_2 if verify_x_first else measurements_1
    return _measure_ft_stabs(
        sp_circ,
        np.asarray(x_measurements, dtype=np.int8),
        np.asarray(z_measurements, dtype=np.int8),
        verify_x_first=verify_x_first,
        flag_first_layer=flag_first_layer,
    )


def gate_optimal_verification_circuit(
    sp_circ: FaultyStatePrepCircuit,
    min_timeout: int = 1,
    max_timeout: int = 3600,
    max_ancillas: int | None = None,
    only_first_layer: bool = False,
    verify_x_first: bool = True,
    flag_first_layer: bool = False,
) -> QuantumCircuit:
    r"""Return a verified state preparation circuit.

    The verification circuit is a set of stabilizers such that each propagated error in sp_circ anticommutes with some verification stabilizer.

    The method uses an iterative search to find the optimal set of stabilizers by repeatedly computing the optimal circuit for each number of ancillas and cnots. This is repeated for each number of independent correctable errors in the state preparation circuit. Thus the verification circuit is constructed of multiple "layers" of stabilizers, each layer corresponding to a fault set it verifies.

    Args:
        sp_circ: The state preparation circuit to verify.
        min_timeout: The minimum time to allow each search to run for.
        max_timeout: The maximum time to allow each search to run for.
        max_ancillas: The maximum number of ancillas to allow in each layer verification circuit.
        only_first_layer: If True, only the first error type will be verified. The type depends on the `verify_x_first` argument.
        verify_x_first: If True, X-errors are verified first.
        flag_first_layer: If True, the first verification layer (verifying X or Z errors) will also be flagged. If False, the potential hook errors introduced by the first layer will be caught by the second layer. This is only relevant if full_fault_tolerance is True.

    Returns:
        QuantumCircuit combining the state preparation and verification circuit.
    """

    def verification_stabs_fun(
        fault_sets: list[PureFaultSet], stabs: npt.NDArray[np.int8]
    ) -> list[list[npt.NDArray[np.int8]]]:
        return gate_optimal_verification_stabilizers(fault_sets, stabs, min_timeout, max_timeout, max_ancillas)

    return _verification_circuit(
        sp_circ,
        verification_stabs_fun,
        only_first_layer=only_first_layer,
        verify_x_first=verify_x_first,
        flag_first_layer=flag_first_layer,
    )


def heuristic_verification_circuit(
    sp_circ: FaultyStatePrepCircuit,
    max_covering_sets: int = 10000,
    find_coset_leaders: bool = True,
    only_first_layer: bool = False,
    verify_x_first: bool = True,
    flag_first_layer: bool = False,
) -> QuantumCircuit:
    r"""Return a verified state preparation circuit.

    The method uses a greedy set covering heuristic to find a small set of stabilizers that verifies the state preparation circuit. The heuristic is not guaranteed to find the optimal set of stabilizers.

    Args:
        sp_circ: The state preparation circuit to verify.
        max_covering_sets: The maximum number of covering sets to consider.
        find_coset_leaders: Whether to find coset leaders for the found measurements. This is done using SAT solvers so it can be slow.
        only_first_layer: If True, only the first error type will be verified. The type depends on the `verify_x_first` argument.
        verify_x_first: If True, X-errors are verified first.
        flag_first_layer: If True, the first verification layer (verifying X or Z errors) will also be flagged. If False, the potential hook errors introduced by the first layer will be caught by the second layer. This is only relevant if full_fault_tolerance is True.

    Returns:
        QuantumCircuit combining the state preparation and verification circuit.
    """

    def verification_stabs_fun(
        fault_sets: list[PureFaultSet], stabs: npt.NDArray[np.int8]
    ) -> list[list[npt.NDArray[np.int8]]]:
        return heuristic_verification_stabilizers(fault_sets, stabs, max_covering_sets, find_coset_leaders)

    return _verification_circuit(
        sp_circ,
        verification_stabs_fun,
        only_first_layer=only_first_layer,
        verify_x_first=verify_x_first,
        flag_first_layer=flag_first_layer,
    )


def heuristic_verification_stabilizers(
    fault_sets: list[PureFaultSet],
    stabs: npt.NDArray[np.int8],
    max_covering_sets: int = 10000,
    find_coset_leaders: bool = True,
) -> list[list[npt.NDArray[np.int8]]]:
    """Return verification stabilizers for the given fault sets.

    Args:
        fault_sets: List of fault sets to verify.
        stabs: The stabilizer generators to verify the fault sets.
        max_covering_sets: The maximum number of covering sets to consider.
        find_coset_leaders: Whether to find coset leaders for the found measurements. This is done using SAT solvers so it can be slow.
    """
    logger.info("Finding verification stabilizers using heuristic method")
    n_layers = len(fault_sets)
    layers: list[list[npt.NDArray[np.int8]]] = [[] for _ in range(n_layers)]

    for num_errors in range(n_layers):
        logger.info(f"Finding verification stabilizers for {num_errors + 1} errors")
        faults = fault_sets[num_errors]
        assert faults is not None
        logger.info(f"There are {len(faults)} faults")
        if len(faults) == 0:
            layers[num_errors] = []
            continue

        layers[num_errors] = _heuristic_layer(faults.faults, stabs, find_coset_leaders, max_covering_sets)

    return layers


def _covers(s: npt.NDArray[np.int8], faults: npt.NDArray[np.int8]) -> frozenset[int]:
    return frozenset(np.where(s @ faults.T % 2 != 0)[0])


def _set_cover(
    n: int, cands: set[frozenset[int]], mapping: dict[frozenset[int], list[npt.NDArray[np.int8]]]
) -> set[frozenset[int]]:
    universe = set(range(n))
    cover: set[frozenset[int]] = set()

    while universe:
        best = max(cands, key=lambda stab: (len(stab & universe), -np.sum(mapping[stab])))
        cover.add(best)
        universe -= best
    return cover


def _extend_covering_sets(
    candidate_sets: set[frozenset[int]], size_limit: int, mapping: dict[frozenset[int], list[npt.NDArray[np.int8]]]
) -> set[frozenset[int]]:
    to_remove: set[frozenset[int]] = set()
    to_add: set[frozenset[int]] = set()
    for c1 in candidate_sets:
        for c2 in candidate_sets:
            if len(to_add) >= size_limit:
                break

            comb = c1 ^ c2
            if c1 == c2 or comb in candidate_sets or comb in to_add or comb in to_remove:
                continue

            mapping[comb].extend([(s1 + s2) % 2 for s1 in mapping[c1] for s2 in mapping[c2]])

            if len(c1 & c2) == 0:
                to_remove.add(c1)
                to_remove.add(c2)
            to_add.add(c1 ^ c2)

    return candidate_sets.union(to_add)


def _heuristic_layer(
    faults: npt.NDArray[np.int8], checks: npt.NDArray[np.int8], find_coset_leaders: bool, max_covering_sets: int
) -> list[npt.NDArray[np.int8]]:
    syndromes = checks @ faults.T % 2
    candidates = np.where(np.any(syndromes != 0, axis=1))[0]
    non_candidates = np.where(np.all(syndromes == 0, axis=1))[0]
    candidate_checks = checks[candidates]
    non_candidate_checks = checks[non_candidates]

    logger.info("Converting Stabilizer Checks to covering sets")
    candidate_sets_ordered = [(_covers(s, faults), s, i) for i, s in enumerate(candidate_checks)]
    mapping = defaultdict(list)
    for cand, _, i in candidate_sets_ordered:
        mapping[cand].append(candidate_checks[i])
    candidate_sets = {cand for cand, _, _ in candidate_sets_ordered}

    logger.info("Finding initial set cover")
    cover = _set_cover(len(faults), candidate_sets, mapping)
    logger.info(f"Initial set cover has {len(cover)} sets")

    def cost(cover: set[frozenset[int]]) -> tuple[int, int]:
        cost1 = len(cover)
        cost2 = sum(np.sum(mapping[stab]) for stab in cover)
        return cost1, cost2

    cost1, cost2 = cost(cover)
    prev_candidates = candidate_sets.copy()

    # find good cover
    improved = True
    while improved and len(candidate_sets) < max_covering_sets:
        improved = False
        # add all symmetric differences to candidates
        candidate_sets = _extend_covering_sets(candidate_sets, max_covering_sets, mapping)
        new_cover = _set_cover(len(faults), candidate_sets, mapping)
        logger.info(f"New Covering set has {len(new_cover)} sets")
        new_cost1 = len(new_cover)
        new_cost2 = sum(np.sum(mapping[stab]) for stab in new_cover)
        if new_cost1 < cost1 or (new_cost1 == cost1 and new_cost2 < cost2):
            cover = new_cover
            cost1 = new_cost1
            cost2 = new_cost2
            improved = True
        elif candidate_sets == prev_candidates:
            break
        prev_candidates = candidate_sets

    # reduce stabilizers in cover
    logger.info(f"Found covering set of size {len(cover)}.")
    if find_coset_leaders and len(non_candidates) > 0:
        logger.info("Finding coset leaders.")
        measurements = []
        for c in cover:
            leaders = [coset_leader(m, non_candidate_checks) for m in mapping[c]]
            leaders.sort(key=np.sum)
            measurements.append(leaders[0])
    else:
        measurements = [min(mapping[c], key=np.sum) for c in cover]

    return measurements


def _measure_ft_x(
    qc: QuantumCircuit,
    x_measurements: npt.NDArray[np.int8],
    t: int,
    flags: bool = False,
) -> None:
    if len(x_measurements) == 0:
        return
    num_x_anc = len(x_measurements)
    x_anc = AncillaRegister(num_x_anc, "x_anc")
    x_c = ClassicalRegister(num_x_anc, "x_c")
    qc.add_register(x_anc)
    qc.add_register(x_c)

    for i, m in enumerate(x_measurements):
        stab = np.where(m != 0)[0]
        if flags:
            measure_flagged(qc, stab, x_anc[i], x_c[i], z_measurement=False, t=t)
        else:
            qc.h(x_anc[i])
            qc.cx([x_anc[i]] * len(stab), stab)
            qc.h(x_anc[i])
            qc.measure(x_anc[i], x_c[i])


def _measure_ft_z(qc: QuantumCircuit, z_measurements: npt.NDArray[np.int8], t: int, flags: bool = False) -> None:
    if len(z_measurements) == 0:
        return
    num_z_anc = len(z_measurements)
    z_anc = AncillaRegister(num_z_anc, "z_anc")
    z_c = ClassicalRegister(num_z_anc, "z_c")
    qc.add_register(z_anc)
    qc.add_register(z_c)

    for i, m in enumerate(z_measurements):
        stab = np.where(m != 0)[0]
        if flags:
            measure_flagged(qc, stab, z_anc[i], z_c[i], z_measurement=True, t=t)
        else:
            qc.cx(stab, [z_anc[i]] * len(stab))
    qc.measure(z_anc, z_c)


def _measure_ft_stabs(
    sp_circ: FaultyStatePrepCircuit,
    x_measurements: npt.NDArray[np.int8],
    z_measurements: npt.NDArray[np.int8],
    verify_x_first: bool = True,
    flag_first_layer: bool = False,
) -> QuantumCircuit:
    # Create the verification circuit
    q = QuantumRegister(sp_circ.num_qubits, "q")
    measured_circ = QuantumCircuit(q)
    measured_circ.compose(sp_circ.circ.to_qiskit_circuit(), inplace=True)

    if verify_x_first:
        _measure_ft_z(measured_circ, z_measurements, t=sp_circ.max_z_errors, flags=flag_first_layer)
        _measure_ft_x(measured_circ, x_measurements, flags=True, t=sp_circ.max_x_errors)

    else:
        _measure_ft_x(measured_circ, x_measurements, flags=flag_first_layer, t=sp_circ.max_x_errors)
        _measure_ft_z(measured_circ, z_measurements, t=sp_circ.max_z_errors)

    return measured_circ


def verification_stabilizers(
    fault_set: PureFaultSet,
    stabs: npt.NDArray[np.int8],
    num_anc: int,
    num_cnots: int,
) -> list[npt.NDArray[np.int8]] | None:
    """Return a set of stabilizers detecting all errors in `fault_set` using at most `num_anc` ancillas and at most `num_cnots` cnots.

    Args:
        fault_set: The fault set to verify.
        stabs: The stabilizer generators to verify the fault set.
        num_anc: The maximum number of ancilla qubits to use.
        num_cnots: The maximum number of CNOT gates to use.

    Returns:
        List of stabilizers.
    """
    stabs_list = all_verification_stabilizers(fault_set, stabs, num_anc, num_cnots, return_all_solutions=False)
    if stabs_list:
        return stabs_list[0]
    return None


def all_verification_stabilizers(
    fault_set: PureFaultSet,
    stabs: npt.NDArray[np.int8],
    num_anc: int,
    num_cnots: int,
    return_all_solutions: bool = False,
) -> list[list[npt.NDArray[np.int8]]] | None:
    """Return a list of verification stabilizers for independent errors in the state preparation circuit using z3.

    Args:
        fault_set: The set of errors to verify.
        stabs: Stabilizer generators of the stabilizers measured.
        num_anc: The maximum number of ancilla qubits to use.
        num_cnots: The maximum number of CNOT gates to use.
        return_all_solutions: If True, return all solutions. Otherwise, return the first solution found.
    """
    # Measurements are written as sums of generators
    # The variables indicate which generators are non-zero in the sum
    if fault_set.faults.shape[1] != stabs.shape[1]:
        msg = "Fault set and stabilizers must have the same number of qubits."
        raise ValueError(msg)

    # Check if fault set can be verified, i.e., every fault can be detected by at least one measurement
    if any(np.all(fault_set.faults @ stabs.T % 2 == 0, axis=1)):
        logger.warning(
            "Fault set cannot be verified by the given stabilizers. Some faults are not detectable by the given stabilizers."
        )
        return None

    n_gens = stabs.shape[0]
    n_qubits = stabs.shape[1]

    measurement_vars = [[z3.Bool(f"m_{anc}_{i}") for i in range(n_gens)] for anc in range(num_anc)]
    measurement_stabs = [vars_to_stab(vars_, stabs) for vars_ in measurement_vars]

    solver = z3.Solver()
    # assert that each error is detected
    solver.add(
        z3.And([
            z3.PbGe([(odd_overlap(measurement, error), 1) for measurement in measurement_stabs], 1)
            for error in fault_set
        ])
    )

    # assert that not too many CNOTs are used
    solver.add(z3.PbLe([(measurement[q], 1) for measurement in measurement_stabs for q in range(n_qubits)], num_cnots))

    solutions = []
    while solver.check() == z3.sat:
        model = solver.model()
        # Extract stabilizer measurements from model
        actual_measurements = []
        for m in measurement_vars:
            v = np.zeros(n_qubits, dtype=np.int8)
            for g in range(n_gens):
                if model[m[g]]:
                    v += stabs[g]
            actual_measurements.append(v % 2)
        if not return_all_solutions:
            return [actual_measurements]
        solutions.append(actual_measurements)
        # add constraint to avoid same solution again
        solver.add(z3.Or([vars_[i] != model[vars_[i]] for vars_ in measurement_vars for i in range(n_gens)]))
    if solutions:
        return solutions

    return None


def naive_verification_circuit(sp_circ: FaultyStatePrepCircuit, flag_first_layer: bool = True) -> QuantumCircuit:
    """Naive verification circuit for a state preparation circuit."""
    code = sp_circ.circ.get_code()

    z_measurements = code.Hz
    x_measurements = code.Hx
    return _measure_ft_stabs(
        sp_circ,
        z_measurements=z_measurements * sp_circ.max_z_errors,
        x_measurements=x_measurements * sp_circ.max_x_errors,
        flag_first_layer=flag_first_layer,
    )


def get_hook_errors(measurements: list[npt.NDArray[np.int8]]) -> PureFaultSet:
    """Assuming CNOTs are executed in ascending order of qubit index, this function gives all the hook errors of the given stabilizer measurements."""
    errors = []
    for stab in measurements:
        support = np.where(stab == 1)[0]
        error = stab.copy()
        for qubit in support[:-1]:
            error[qubit] = 0
            errors.append(error.copy())

    if len(errors) == 0:
        return PureFaultSet(measurements[0].shape[1])
    return PureFaultSet.from_fault_array(np.array(errors))


def final_matrix_constraint(columns: npt.NDArray[np.bool_], rank: int) -> z3.BoolRef:
    """Return a z3 constraint that the final matrix has exactly rank non-zero columns."""
    assert len(columns.shape) == 3
    return z3.PbEq(
        [(z3.Not(z3.Or(list(columns[-1, :, col]))), 1) for col in range(columns.shape[2])],
        columns.shape[2] - rank,
    )
