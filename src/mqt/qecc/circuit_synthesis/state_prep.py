# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Synthesizing state preparation circuits for CSS codes."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import z3
from ldpc import mod2
from qiskit.circuit import AncillaRegister, ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag

from ..codes import InvalidCSSCodeError
from .synthesis_utils import (
    EliminationCNOTSynthesizer,
    build_css_circuit_from_cnot_list,
    iterative_search_with_timeout,
    measure_flagged,
    odd_overlap,
    optimal_elimination,
    run_with_timeout,
    symbolic_scalar_mult,
    symbolic_vector_add,
    symbolic_vector_eq,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import numpy.typing as npt
    from qiskit import DAGNode
    from qiskit.quantum_info import PauliList

    from ..codes.css_code import CSSCode


class StatePrepCircuit:
    """Represents a state preparation circuit for a CSS code."""

    def __init__(
        self, circ: QuantumCircuit, code: CSSCode, zero_state: bool = True, error_detection_code: bool = False
    ) -> None:
        """Initialize a state preparation circuit.

        Args:
            circ: The state preparation circuit.
            code: The CSS code to prepare the state for.
            zero_state: If True, prepare the +1 eigenstate of the Z basis. If False, prepare the +1 eigenstate of the X basis.
            error_detection_code: If True, prepare the state for error detection. This ensures that when computing the fault set of the circuit, up to d//2 errors errors can occur in the circuit.
        """
        self.circ = circ
        self.code = code
        self.zero_state = zero_state

        if code.Hx is None or code.Hz is None:
            msg = "The CSS code must have both X and Z checks."
            raise InvalidCSSCodeError(msg)

        self.x_checks = code.Hx.copy() if zero_state else np.vstack((code.Lx.copy(), code.Hx.copy()))
        self.z_checks = code.Hz.copy() if not zero_state else np.vstack((code.Lz.copy(), code.Hz.copy()))

        self.num_qubits = circ.num_qubits

        self.error_detection_code = error_detection_code
        self._set_max_errors()

        self.max_x_measurements = len(self.x_checks)
        self.max_z_measurements = len(self.z_checks)

    def set_error_detection(self, error_detection: bool) -> None:
        """Set whether the state preparation circuit is for error detection."""
        self.error_detection_code = error_detection
        self._set_max_errors()

    def compute_fault_sets(self, reduce: bool = True) -> None:
        """Compute the fault sets for the state preparation circuit."""
        self.compute_fault_set(self.max_errors, x_errors=True, reduce=reduce)
        self.compute_fault_set(self.max_errors, x_errors=False, reduce=reduce)

    def compute_fault_set(
        self, num_errors: int = 1, x_errors: bool = True, reduce: bool = True
    ) -> npt.NDArray[np.int8]:
        """Compute the fault set of the state.

        Args:
            num_errors: The number of independent errors to propagate through the circuit.
            x_errors: If True, compute the fault set for X errors. If False, compute the fault set for Z errors.
            reduce: If True, reduce the fault set by the stabilizers of the code to reduce weights.

        Returns:
            The fault set of the state.
        """
        faults: npt.NDArray[np.int8] | None = (
            self.x_fault_sets[num_errors] if x_errors else self.z_fault_sets[num_errors]
        )
        if faults is not None:
            return faults

        if num_errors == 1:
            logger.info("Computing fault set for 1 error.")
            dag = circuit_to_dag(self.circ)
            for node in dag.front_layer():  # remove hadamards
                dag.remove_op_node(node)
            fault_list = []
            # propagate every error before a control
            nodes = list(dag.topological_op_nodes())
            for i in range(len(nodes)):
                error = _propagate_error(nodes[i:], dag.num_qubits(), x_errors=x_errors)
                fault_list.append(error)
            faults = np.array(fault_list, dtype=np.int8)
            faults = np.unique(faults, axis=0)

            if x_errors and self.x_fault_sets_unreduced[1] is None:
                non_propagated_single_errors = np.eye(self.num_qubits, dtype=np.int8)
                self.x_fault_sets_unreduced[1] = np.vstack((faults, non_propagated_single_errors))
            elif not x_errors and self.z_fault_sets[1] is None:
                non_propagated_single_errors = np.eye(self.num_qubits, dtype=np.int8)
                self.z_fault_sets_unreduced[1] = np.vstack((faults, non_propagated_single_errors))
        else:
            logger.info(f"Computing fault set for {num_errors} errors.")
            self.compute_fault_set(num_errors - 1, x_errors, reduce=reduce)
            if x_errors:
                faults = self.x_fault_sets_unreduced[num_errors - 1]
                single_faults = self.x_fault_sets_unreduced[1]
            else:
                faults = self.z_fault_sets_unreduced[num_errors - 1]
                single_faults = self.z_fault_sets_unreduced[1]

            assert faults is not None
            assert single_faults is not None

            new_faults = (faults[:, np.newaxis, :] + single_faults).reshape(-1, self.num_qubits) % 2
            # remove duplicates
            faults = np.unique(new_faults, axis=0)
            if x_errors:
                self.x_fault_sets_unreduced[num_errors] = faults.copy()
            else:
                self.z_fault_sets_unreduced[num_errors] = faults.copy()

        # reduce faults by stabilizer
        stabs = self.x_checks if x_errors else self.z_checks
        faults = _remove_trivial_faults(faults, stabs, num_errors)

        # remove stabilizer equivalent faults
        if reduce:
            logger.info("Removing stabilizer equivalent faults.")
            faults = _remove_stabilizer_equivalent_faults(faults, stabs)
        if x_errors:
            self.x_fault_sets[num_errors] = faults
        else:
            self.z_fault_sets[num_errors] = faults
        return faults

    def combine_faults(
        self, additional_faults: npt.NDArray[np.int8], x_errors: bool = True
    ) -> list[npt.NDArray[np.int8] | None]:
        """Combine fault sets of circuit with additional independent faults.

        Args:
            additional_faults: The additional faults to combine with the fault set of the circuit.
            x_errors: If True, combine the fault sets for X errors. If False, combine the fault sets for Z errors.
        """
        self.compute_fault_sets()
        if len(additional_faults) == 0:
            return self.x_fault_sets if x_errors else self.z_fault_sets

        fault_sets_unreduced = self.x_fault_sets_unreduced.copy() if x_errors else self.z_fault_sets_unreduced.copy()
        assert fault_sets_unreduced[1] is not None
        fault_sets_unreduced[1] = np.vstack((fault_sets_unreduced[1], additional_faults))

        for i in range(1, self.max_errors):
            uncombined = fault_sets_unreduced[i]
            assert uncombined is not None
            combined = (uncombined[:, np.newaxis, :] + additional_faults).reshape(-1, self.num_qubits) % 2
            next_faults = fault_sets_unreduced[i + 1]
            assert next_faults is not None
            fault_sets_unreduced[i + 1] = np.vstack((next_faults, combined))
        fault_sets: list[npt.NDArray[np.int8] | None] = [None for _ in range(self.max_errors + 1)]
        stabs = self.x_checks if x_errors else self.z_checks
        for num_errors in range(1, self.max_errors + 1):
            fs = fault_sets_unreduced[num_errors]
            assert fs is not None
            fault_sets[num_errors] = _remove_trivial_faults(fs, stabs, num_errors)
        return fault_sets

    def _set_max_errors(self) -> None:
        if self.code.distance == 2:
            logger.warning("Code distance is 2, assuming error detection code.")
            self.error_detection_code = True

        self.max_errors = (self.code.distance - 1) // 2 if not self.error_detection_code else self.code.distance // 2
        self.max_x_errors = (
            (self.code.x_distance - 1) // 2 if not self.error_detection_code else self.code.x_distance // 2
        )
        self.max_z_errors = (
            (self.code.z_distance - 1) // 2 if not self.error_detection_code else self.code.z_distance // 2
        )
        self.x_fault_sets: list[npt.NDArray[np.int8] | None] = [None for _ in range(self.max_errors + 1)]
        self.z_fault_sets: list[npt.NDArray[np.int8] | None] = [None for _ in range(self.max_errors + 1)]
        self.x_fault_sets_unreduced: list[npt.NDArray[np.int8] | None] = [None for _ in range(self.max_errors + 1)]
        self.z_fault_sets_unreduced: list[npt.NDArray[np.int8] | None] = [None for _ in range(self.max_errors + 1)]


def _build_state_prep_circuit_from_back(
    checks: npt.NDArray[np.int8], cnots: list[tuple[int, int]], zero_state: bool = True
) -> QuantumCircuit:
    cnots.reverse()
    if zero_state:
        hadamards = np.where(np.sum(checks, axis=0) != 0)[0]
    else:
        hadamards = np.where(np.sum(checks, axis=0) == 0)[0]
        cnots = [(j, i) for i, j in cnots]

    return build_css_circuit_from_cnot_list(checks.shape[1], cnots, list(hadamards))


def heuristic_prep_circuit(
    code: CSSCode,
    optimize_depth: bool = True,
    zero_state: bool = True,
) -> StatePrepCircuit:
    """Return a circuit that prepares the +1 eigenstate of the code w.r.t. the Z or X basis.

    Args:
        code: The CSS code to prepare the state for.
        optimize_depth: If True, optimize the depth of the circuit. This may lead to a higher number of CNOTs.
        zero_state: If True, prepare the +1 eigenstate of the Z basis. If False, prepare the +1 eigenstate of the X basis.
    """
    logger.info("Starting heuristic state preparation.")
    if code.Hx is None or code.Hz is None:
        msg = "The code must have both X and Z stabilizers defined."
        raise InvalidCSSCodeError(msg)

    checks = code.Hx if zero_state else code.Hz
    assert checks is not None
    ge = EliminationCNOTSynthesizer(matrix=checks, code=code, parallel_elimination=optimize_depth)
    ge.greedy_synthesis()

    circ = _build_state_prep_circuit_from_back(ge.matrix, ge.eliminations, zero_state)
    return StatePrepCircuit(circ, code, zero_state)


def heuristic_reference_prep_circuit(
    code: CSSCode,
    optimize_depth: bool = True,
    zero_state: bool = True,
    penalty_cnots: list[tuple[int, int]] | None = None,
    guide_by_x: bool = True,
    ref_x_fs: npt.NDArray[np.int8] | None = None,
    ref_z_fs: npt.NDArray[np.int8] | None = None,
    ref_x_1fs: npt.NDArray[np.int8] | None = None,
    ref_z_1fs: npt.NDArray[np.int8] | None = None,
) -> StatePrepCircuit:
    """Return a circuit that prepares the +1 eigenstate of the code w.r.t. the Z or X basis and based on (a) reference fault set(s).

    Args:
        code: The CSS code to prepare the state for.
        optimize_depth: If True, optimize the depth of the circuit. This may lead to a higher number of CNOTs.
        zero_state: If True, prepare the +1 eigenstate of the Z basis. If False, prepare the +1 eigenstate of the X basis.
        penalty_cnots: tuples of CNOTs (control, target) which are initially added to the failed_cnots list and hence can only be applied once the control qubit has been used elsewhere
        guide_by_x: Flag that decides whether dismissed CNOTs are free to placement again after either the control (x guided) or the target (z guided) has been used elsewhere
        ref_x_fs: (Optional) reference x fault set which influences the construction of the circuit
        ref_z_fs: (Optional) reference z fault set which influences the construction of the circuit
        ref_x_1fs: (Optional) reference one error x fault set which ensures that no two error event of the newly constructed circuit cancels a one error event of the reference circuit
        ref_z_1fs: (Optional) reference one error z fault set which ensures that no two error event of the newly constructed circuit cancels a one error event of the reference circuit
    """
    if penalty_cnots is None:
        penalty_cnots = []
    logger.info("Starting heuristic state preparation.")
    if code.Hx is None or code.Hz is None:
        msg = "The code must have both X and Z stabilizers defined."
        raise InvalidCSSCodeError(msg)

    checks = code.Hx if zero_state else code.Hz
    assert checks is not None
    ge = EliminationCNOTSynthesizer(
        matrix=checks,
        parallel_elimination=optimize_depth,
        code=code,
        ref_x_fs=ref_x_fs,
        ref_z_fs=ref_z_fs,
        ref_x_1fs=ref_x_1fs,
        ref_z_1fs=ref_z_1fs,
        penalty_cnots=penalty_cnots,
        guide_by_x=guide_by_x,
    )
    ge.fault_set_guided_synthesis()
    circ = _build_state_prep_circuit_from_back(ge.matrix, ge.eliminations, zero_state)
    return StatePrepCircuit(circ, code, zero_state)


def depth_optimal_prep_circuit(
    code: CSSCode,
    zero_state: bool = True,
    min_depth: int = 1,
    max_depth: int = 10,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> StatePrepCircuit | None:
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
    return StatePrepCircuit(circ, code, zero_state)


def gate_optimal_prep_circuit(
    code: CSSCode,
    zero_state: bool = True,
    min_gates: int = 1,
    max_gates: int = 10,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> StatePrepCircuit | None:
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
    return StatePrepCircuit(circ, code, zero_state)


def iter_cyclic_combinations(lists):
    """Generator that yields one combination of cyclic permutations for each list.
    Each list in 'lists' is rotated by an offset given by its counter.
    """
    n = len(lists)
    # counters to track the rotation index for each list
    counters = [0] * n
    # store the lengths of each list so we know when to reset a counter
    lengths = [len(lst) for lst in lists]

    while True:
        # Yield the current combination based on counters.
        current = [lst[i:] + lst[:i] for lst, i in zip(lists, counters)]
        yield current

        # Now update the counters (like an odometer)
        for pos in reversed(range(n)):
            counters[pos] += 1  # increment the current counter
            if counters[pos] < lengths[pos]:
                # Successfully incremented without overflow;
                # we can break out and yield the next combination on the next iteration.
                break
            else:
                # Reset this counter and move to the next (more significant) counter.
                counters[pos] = 0
        else:
            # If we have reset all counters, then we've exhausted all combinations.
            break


def _permute_commuting_cnots(
    trgt_order: dict[int, list[int]], ctrl_order, trgt_perm: dict[int, list[int]], ctrl_perm: dict[int, list[int]]
) -> list[tuple[int, int]]:
    """Permute CNOTs according to the given permutations.

    Args:
        trgt_order: The target order of the CNOTs.
        ctrl_order: The control order of the CNOTs.
        trgt_perm: The target permutation.
        ctrl_perm: The control permutation.

    Returns:
        The CNOTs in correct order.
    """
    n = len(trgt_order) + len(ctrl_order)

    orders = [None for _ in range(n)]
    for ctrl, trgts in trgt_order.items():
        perm = trgt_perm[ctrl]
        orders[ctrl] = [trgts[perm[i]] for i in range(len(trgts))]
    for trgt, ctrls in ctrl_order.items():
        perm = ctrl_perm[trgt]
        orders[trgt] = [ctrls[perm[i]] for i in range(len(ctrls))]

    [0 for _ in range(n)]
    done = [False for _ in range(n)]
    cnots = []

    stack = []
    while not all(done):
        if len(stack) == 0:
            for i in range(n):
                if not done[i]:
                    stack.append(i)
                    break

        q1 = stack[-1]
        q2 = orders[q1][0]

        if q2 in stack:  # conflict -> resolve via cycles
            q1_idx = orders[q2].index(q1)
            orders[q2] = orders[q2][q1_idx:] + orders[q2][:q1_idx]
            # unroll stack to q2
            stack = stack[: stack.index(q2) + 1]

        if orders[q2][0] == q1:
            if q1 in trgt_order:
                cnots.append((q1, q2))
            else:
                cnots.append((q2, q1))
            orders[q1].remove(q2)
            orders[q2].remove(q1)
            done[q1] = len(orders[q1]) == 0
            done[q2] = len(orders[q2]) == 0
            stack.pop()
        else:
            stack.append(q2)

    return cnots


def _canonical_permutation(w: int) -> list[int]:
    """Return canonical ft5 permutation for weight w."""
    pi = list(range(w))
    pi[0], pi[-1] = pi[-1], pi[0]

    # if w <= 5:
    #     return list(range(w))
    # offset = w // 2
    # pi = list(range(w))
    # if w % 2 == 1:
    #     pi.remove(offset)
    # for i in range(offset):
    #     if i % 2 == 0:
    #         swap_idx = (i + w // 2) % len(pi)
    #         pi[i], pi[swap_idx] = (pi[swap_idx], pi[i])
    #     else:
    #         pi[i]

    # if w % 2 == 1:
    #     pi = pi[:offset] + [offset] + pi[offset:]

    return pi


def find_swapping_permutation(lists: list[list[int]]) -> dict[int, int]:
    """Given a list of lists of integers, finds an involution π (a permutation
    composed solely of 2-cycles and fixed points) with the property that for
    every list xs, π swaps one of the first three elements with one of the
    last three elements.

    Parameters:
        lists (list of list of int): The input lists (each assumed to have 7 integers).

    Returns:
        dict or None: A dictionary representing the permutation π, where each key maps
                      to its image under π, or None if no such permutation exists.
    """
    # Step 1: Compute candidate pairs for each list.
    list_candidates = []
    lists = [lst for lst in lists if len(lst) > 4]
    for xs in lists:
        cand = set()
        for i in range(3):  # first three elements
            for j in range(len(xs) - 3, len(xs)):  # last three elements
                # Use sorted tuple to represent an unordered pair
                if i == j:
                    continue
                pair = tuple(sorted((xs[i], xs[j])))
                cand.add(pair)
        list_candidates.append(cand)

    # Step 2: Compute the global candidate set (union of all candidate pairs).
    global_candidates = set()
    for cand in list_candidates:
        global_candidates |= cand
    candidate_edges = list(global_candidates)

    # Collect the full set of numbers (the domain for our permutation).
    global_numbers = set()
    for xs in lists:
        global_numbers |= set(xs)

    # Step 3: Search for a matching (set of disjoint candidate pairs) such that
    # for every list at least one of its candidate pairs is in the matching.
    best_matching = None

    def backtrack(i, current_matching, used) -> None:
        nonlocal best_matching
        # If all candidate edges have been considered,
        # check if current matching "covers" every list.
        if i == len(candidate_edges):
            for cand in list_candidates:
                if not any(edge in current_matching for edge in cand):
                    return
            best_matching = current_matching.copy()
            return

        # Option 1: Skip the current candidate edge.
        backtrack(i + 1, current_matching, used)
        if best_matching is not None:
            return  # solution found

        # Option 2: Try to include candidate_edges[i] if neither number is used yet.
        edge = candidate_edges[i]
        a, b = edge
        if a not in used and b not in used:
            current_matching.add(edge)
            used.add(a)
            used.add(b)
            backtrack(i + 1, current_matching, used)
            if best_matching is not None:
                return
            current_matching.remove(edge)
            used.remove(a)
            used.remove(b)

    backtrack(0, set(), set())

    # Step 4: If a matching was found, build the permutation π.
    if best_matching is None:
        return None
    pi = {x: x for x in global_numbers}  # initially, all elements are fixed
    for a, b in best_matching:
        pi[a] = b
        pi[b] = a
    return pi


def random_steane_type_prep_circuits(
    code: CSSCode, zero_state: bool = True
) -> tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit, QuantumCircuit]:
    """Return four circuits in permuted standard form that prepare the +1 eigenstate of the code w.r.t. the Z or X basis that can implement Steane-type FTSP.

    Args:
        code: The CSS code to prepare the state for.
        zero_state: If True, prepare the +1 eigenstate of the Z basis. If False, prepare the +1 eigenstate of the X basis.

    Returns: A tuple of four state preparation circuits for the code.
    """
    trgt_order = _standard_form_cnots(code, zero_state)
    c1 = [(x, y) for x, ys in trgt_order.items() for y in ys]

    # obtain three random permutations of the cnots
    n = code.n
    p1 = np.random.permutation(len(c1))
    p2 = np.random.permutation(len(c1))
    p3 = np.random.permutation(len(c1))
    c2 = [c1[i] for i in p1]
    c3 = [c1[i] for i in p2]
    c4 = [c1[i] for i in p3]
    ctrls = trgt_order.keys()

    qc1 = build_css_circuit_from_cnot_list(n, c1, ctrls)
    qc2 = build_css_circuit_from_cnot_list(n, c2, ctrls)
    qc3 = build_css_circuit_from_cnot_list(n, c3, ctrls)
    qc4 = build_css_circuit_from_cnot_list(n, c4, ctrls)

    return qc1, qc2, qc3, qc4, [p1, p2, p3]


def canonical_steane_type_prep_circuits(
    code: CSSCode, zero_state: bool = True
) -> tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit, QuantumCircuit]:
    """Return four circuits in standard form that prepare the +1 eigenstate of the code w.r.t. the Z or X basis that can implement Steane-type FTSP.

    Args:
            code: The CSS code to prepare the state for.
            zero_state: If True, prepare the +1 eigenstate of the Z basis. If False, prepare the +1 eigenstate of the X basis.

    Returns: A tuple of four state preparation circuits for the code.
    """
    n = code.n
    trgt_order = _standard_form_cnots(code, zero_state)
    ctrls = list(trgt_order.keys())
    trgts = [trgt for trgt in range(n) if trgt not in ctrls]
    ctrl_order = {trgt: [] for trgt in trgts}
    for ctrl, ctrl_trgts in trgt_order.items():
        for trgt in ctrl_trgts:
            ctrl_order[trgt].append(ctrl)
    for trgt in trgts:
        ctrl_order[trgt].sort()

    id_trgt = {ctrl: list(range(len(trgt_order[ctrl]))) for ctrl in ctrls}
    id_ctrl = {trgt: list(range(len(ctrl_order[trgt]))) for trgt in trgts}

    pi_trgt = {ctrl: _canonical_permutation(len(trgt_order[ctrl])) for ctrl in trgt_order}
    pi_ctrl = {trgt: _canonical_permutation(len(ctrl_order[trgt])) for trgt in trgts}

    c1 = _permute_commuting_cnots(trgt_order, ctrl_order, id_trgt, id_ctrl)
    c2 = _permute_commuting_cnots(trgt_order, ctrl_order, pi_trgt, id_ctrl)
    c3 = _permute_commuting_cnots(trgt_order, ctrl_order, id_trgt, pi_ctrl)

    # collect cnots in c3 with same target into groups
    c3_groups = defaultdict(list)
    for cnot in c3:
        c3_groups[cnot[1]].append(cnot[0])

    # we need to find a permutation that fulfills the following for all groups of targets:
    # - map the first three elements into the second half
    # - map the last element into the first half
    # No conflicts should occur
    lists = list(trgt_order.values())

    pi_swaps = find_swapping_permutation(lists)

    pi = []
    for x in trgts:
        if x in pi_swaps:
            pi.append(pi_swaps[x])
        else:
            pi.append(x)
    # apply pi
    # use pi to order c3_groups
    c4 = []
    for x in pi:
        c4.extend([(y, x) for y in c3_groups[x]])

    qc1 = build_css_circuit_from_cnot_list(n, c1, ctrls)
    qc2 = build_css_circuit_from_cnot_list(n, c2, ctrls)
    qc3 = build_css_circuit_from_cnot_list(n, c3, ctrls)
    qc4 = build_css_circuit_from_cnot_list(n, c4, ctrls)

    return (qc1, qc2, qc3, qc4)


def standard_form_prep_circuit(code: CSSCode, zero_state: bool = True) -> StatePrepCircuit:
    """Return a circuit that prepares the +1 eigenstate of the code w.r.t. the Z or X basis using the circuit's standard form.

    Note that no depth optimization like Steane's latin rectangle method is performed.

    Args:
            code: The CSS code to prepare the state for.
            zero_state: If True, prepare the +1 eigenstate of the Z basis. If False, prepare the +1 eigenstate of the X basis.

    Returns: A state preparation circuit for the code.
    """
    h = code.Hx.copy() if zero_state else code.Hz.copy()
    _h_red, _rk, _, _pivots = mod2.row_echelon(h, full=True)
    cnots = _standard_form_cnots(code, zero_state)
    qc = QuantumCircuit(code.n)
    cnots = _standard_form_cnots(code, zero_state)

    for pivot, trgts in cnots:
        qc.h(pivot)
        qc.cx(pivot, trgts)
    return StatePrepCircuit(qc, code, zero_state)


def _standard_form_cnots(code: CSSCode, zero_state: bool = True) -> dict[int, list[int]]:
    """Return the cnots that prepare the +1 eigenstate of the code w.r.t. the Z or X basis using the circuit's standard form.

    Args:
            code: The CSS code to prepare the state for.
            zero_state: If True, prepare the +1 eigenstate of the Z basis. If False, prepare the +1 eigenstate of the X basis.

    Returns: A list of ctrl, targets tuples
    """
    h = code.Hx.copy() if zero_state else code.Hz.copy()
    h_red, _rk, _, pivots = mod2.row_echelon(h, full=True)

    cnots = {}
    for i, pivot in enumerate(pivots):
        trgts = list(np.where(h_red[i, :])[0])
        trgts.remove(pivot)
        cnots[pivot] = trgts

    return cnots


def gate_optimal_verification_stabilizers(
    sp_circ: StatePrepCircuit,
    x_errors: bool = True,
    min_timeout: int = 1,
    max_timeout: int = 3600,
    max_ancillas: int | None = None,
    additional_faults: npt.NDArray[np.int8] | None = None,
) -> list[list[npt.NDArray[np.int8]]]:
    """Return verification stabilizers for the state preparation circuit.

    The method uses an iterative search to find the optimal set of stabilizers by repeatedly computing the optimal circuit for each number of ancillas and cnots. This is repeated for each number of independent correctable errors in the state preparation circuit. Thus the verification circuit is constructed of multiple "layers" of stabilizers, each layer corresponding to a fault set it verifies.

    Args:
        sp_circ: The state preparation circuit to verify.
        x_errors: If True, verify the X errors. If False, verify the Z errors.
        min_timeout: The minimum time to allow each search to run for.
        max_timeout: The maximum time to allow each search to run for.
        max_ancillas: The maximum number of ancillas to allow in each layer verification circuit.
        additional_faults: Faults to verify in addition to the faults propagating in the state preparation circuit.

    Returns:
        A list of stabilizers for each number of errors to verify the state preparation circuit.
    """
    return [
        stabs[0] if stabs != [] else []
        for stabs in all_gate_optimal_verification_stabilizers(
            sp_circ,
            x_errors,
            min_timeout,
            max_timeout,
            max_ancillas,
            additional_faults,
            return_all_solutions=False,
        )
    ]


def all_gate_optimal_verification_stabilizers(
    sp_circ: StatePrepCircuit,
    x_errors: bool = True,
    min_timeout: int = 1,
    max_timeout: int = 3600,
    max_ancillas: int | None = None,
    additional_faults: npt.NDArray[np.int8] | None = None,
    return_all_solutions: bool = False,
) -> list[list[list[npt.NDArray[np.int8]]]]:
    """Return all equivalent verification stabilizers for the state preparation circuit.

    The method uses an iterative search to find the optimal set of stabilizers by repeatedly computing the optimal circuit for each number of ancillas and cnots. This is repeated for each number of independent correctable errors in the state preparation circuit. Thus the verification circuit is constructed of multiple "layers" of stabilizers, each layer corresponding to a fault set it verifies.

    Args:
        sp_circ: The state preparation circuit to verify.
        x_errors: If True, verify the X errors. If False, verify the Z errors.
        min_timeout: The minimum time to allow each search to run for.
        max_timeout: The maximum time to allow each search to run for.
        max_ancillas: The maximum number of ancillas to allow in each layer verification circuit.
        additional_faults: Faults to verify in addition to the faults propagating in the state preparation circuit.
        return_all_solutions: If False only the first solution for each number of errors is returned. If True all solutions are returned.

    Returns:
        A list of all equivalent stabilizers for each number of errors to verify the state preparation circuit.
    """
    max_errors = sp_circ.max_errors
    layers: list[list[list[npt.NDArray[np.int8]]]] = [[] for _ in range(max_errors)]
    if max_ancillas is None:
        max_ancillas = sp_circ.max_z_measurements if x_errors else sp_circ.max_x_measurements

    sp_circ.compute_fault_sets()
    fault_sets = (
        sp_circ.combine_faults(additional_faults, x_errors)
        if additional_faults is not None
        else sp_circ.x_fault_sets
        if x_errors
        else sp_circ.z_fault_sets
    )

    # Find the optimal circuit for every number of errors in the preparation circuit
    for num_errors in range(1, max_errors + 1):
        logger.info(f"Finding verification stabilizers for {num_errors} errors")
        faults = fault_sets[num_errors]
        assert faults is not None

        if len(faults) == 0:
            logger.info(f"No non-trivial faults for {num_errors} errors")
            layers[num_errors - 1] = []
            continue
        # Start with maximal number of ancillas
        # Minimal CNOT solution must be achievable with these
        num_anc = max_ancillas
        checks = sp_circ.z_checks if x_errors else sp_circ.x_checks
        min_cnots: int = np.min(np.sum(checks, axis=1))
        max_cnots: int = np.sum(checks)

        logger.info(
            f"Finding verification stabilizers for {num_errors} errors with {min_cnots} to {max_cnots} CNOTs using {num_anc} ancillas"
        )

        def fun(num_cnots: int) -> list[npt.NDArray[np.int8]] | None:
            return verification_stabilizers(sp_circ, faults, num_anc, num_cnots, x_errors=x_errors)  # noqa: B023

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
            logger.info(f"No verification stabilizers found for {num_errors} errors")
            return []  # No solution found

        logger.info(f"Found verification stabilizers for {num_errors} errors with {num_cnots} CNOTs")
        # If any measurements are unused we can reduce the number of ancillas at least by that
        measurements = [m for m in measurements if np.any(m)]
        num_anc = len(measurements)
        # Iterate backwards to find the minimal number of cnots
        logger.info(f"Finding minimal number of CNOTs for {num_errors} errors")

        def search_cnots(num_cnots: int) -> list[npt.NDArray[np.int8]] | None:
            return verification_stabilizers(sp_circ, faults, num_anc, num_cnots, x_errors=x_errors)  # noqa: B023

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
        logger.info(f"Minimal number of CNOTs for {num_errors} errors is: {num_cnots}")

        # If the number of CNOTs is minimal, we can reduce the number of ancillas
        logger.info(f"Finding minimal number of ancillas for {num_errors} errors")
        while num_anc - 1 > 0:
            logger.info(f"Trying {num_anc - 1} ancillas")

            def search_anc(num_anc: int) -> list[npt.NDArray[np.int8]] | None:
                return verification_stabilizers(sp_circ, faults, num_anc, num_cnots, x_errors=x_errors)  # noqa: B023

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
        logger.info(f"Minimal number of ancillas for {num_errors} errors is: {num_anc}")
        if not return_all_solutions:
            layers[num_errors - 1] = [measurements]
        else:
            all_stabs = all_verification_stabilizers(
                sp_circ, faults, num_anc, num_cnots, x_errors=x_errors, return_all_solutions=True
            )
            if all_stabs:
                layers[num_errors - 1] = all_stabs
                logger.info(f"Found {len(layers[num_errors - 1])} equivalent solutions for {num_errors} errors")

    return layers


def _verification_circuit(
    sp_circ: StatePrepCircuit,
    verification_stabs_fun: Callable[
        [StatePrepCircuit, bool, npt.NDArray[np.int8] | None], list[list[npt.NDArray[np.int8]]]
    ],
    full_fault_tolerance: bool = True,
    flag_first_layer: bool = False,
) -> QuantumCircuit:
    logger.info("Finding verification stabilizers for the state preparation circuit")
    layers_1 = verification_stabs_fun(sp_circ, sp_circ.zero_state, None)
    measurements_1 = [measurement for layer in layers_1 for measurement in layer]

    if full_fault_tolerance:
        if not flag_first_layer:
            additional_errors = get_hook_errors(measurements_1)
            layers_2 = verification_stabs_fun(sp_circ, not sp_circ.zero_state, additional_errors)
        else:
            layers_2 = verification_stabs_fun(sp_circ, not sp_circ.zero_state, None)
        measurements_2 = [measurement for layer in layers_2 for measurement in layer]
    else:
        measurements_2 = []

    if sp_circ.zero_state:
        return _measure_ft_stabs(
            sp_circ,
            measurements_2,
            measurements_1,
            full_fault_tolerance=full_fault_tolerance,
            flag_first_layer=flag_first_layer,
        )
    return _measure_ft_stabs(
        sp_circ,
        measurements_1,
        measurements_2,
        full_fault_tolerance=full_fault_tolerance,
        flag_first_layer=flag_first_layer,
    )


def gate_optimal_verification_circuit(
    sp_circ: StatePrepCircuit,
    min_timeout: int = 1,
    max_timeout: int = 3600,
    max_ancillas: int | None = None,
    full_fault_tolerance: bool = True,
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
        full_fault_tolerance: If True, the verification circuit will be constructed to be fault tolerant to all errors in the state preparation circuit. If False, the verification circuit will be constructed to be fault tolerant only to the type of errors that can cause a logical error. For a logical \|0> state preparation circuit, this means the verification circuit will be fault tolerant to X errors but not for Z errors. For a logical \|+> state preparation circuit, this means the verification circuit will be fault tolerant to Z errors but not for X errors.
        flag_first_layer: If True, the first verification layer (verifying X or Z errors) will also be flagged. If False, the potential hook errors introduced by the first layer will be caught by the second layer. This is only relevant if full_fault_tolerance is True.
    """

    def verification_stabs_fun(
        sp_circ: StatePrepCircuit,
        zero_state: bool,
        additional_errors: npt.NDArray[np.int8] | None = None,
    ) -> list[list[npt.NDArray[np.int8]]]:
        return gate_optimal_verification_stabilizers(
            sp_circ, zero_state, min_timeout, max_timeout, max_ancillas, additional_errors
        )

    return _verification_circuit(
        sp_circ, verification_stabs_fun, full_fault_tolerance=full_fault_tolerance, flag_first_layer=flag_first_layer
    )


def heuristic_verification_circuit(
    sp_circ: StatePrepCircuit,
    max_covering_sets: int = 10000,
    find_coset_leaders: bool = True,
    full_fault_tolerance: bool = True,
    flag_first_layer: bool = False,
) -> QuantumCircuit:
    r"""Return a verified state preparation circuit.

    The method uses a greedy set covering heuristic to find a small set of stabilizers that verifies the state preparation circuit. The heuristic is not guaranteed to find the optimal set of stabilizers.

    Args:
        sp_circ: The state preparation circuit to verify.
        max_covering_sets: The maximum number of covering sets to consider.
        find_coset_leaders: Whether to find coset leaders for the found measurements. This is done using SAT solvers so it can be slow.
        full_fault_tolerance: If True, the verification circuit will be constructed to be fault tolerant to all errors in the state preparation circuit. If False, the verification circuit will be constructed to be fault tolerant only to the type of errors that can cause a logical error. For a logical \|0> state preparation circuit, this means the verification circuit will be fault tolerant to X errors but not for Z errors. For a logical \|+> state preparation circuit, this means the verification circuit will be fault tolerant to Z errors but not for X errors.
        flag_first_layer: If True, the first verification layer (verifying X or Z errors) will also be flagged. If False, the potential hook errors introduced by the first layer will be caught by the second layer. This is only relevant if full_fault_tolerance is True.
    """

    def verification_stabs_fun(
        sp_circ: StatePrepCircuit, zero_state: bool, additional_errors: npt.NDArray[np.int8] | None = None
    ) -> list[list[npt.NDArray[np.int8]]]:
        return heuristic_verification_stabilizers(
            sp_circ, zero_state, max_covering_sets, find_coset_leaders, additional_errors
        )

    return _verification_circuit(
        sp_circ, verification_stabs_fun, full_fault_tolerance=full_fault_tolerance, flag_first_layer=flag_first_layer
    )


def heuristic_verification_stabilizers(
    sp_circ: StatePrepCircuit,
    x_errors: bool = True,
    max_covering_sets: int = 10000,
    find_coset_leaders: bool = True,
    additional_faults: npt.NDArray[np.int8] | None = None,
) -> list[list[npt.NDArray[np.int8]]]:
    """Return verification stabilizers for the preparation circuit.

    Args:
        sp_circ: The state preparation circuit to verify.
        x_errors: Whether to find verification stabilizers for X errors. If False, find for Z errors.
        max_covering_sets: The maximum number of covering sets to consider.
        find_coset_leaders: Whether to find coset leaders for the found measurements. This is done using SAT solvers so it can be slow.
        additional_faults: Faults to verify in addition to the faults propagating in the state preparation circuit.
    """
    logger.info("Finding verification stabilizers using heuristic method")
    max_errors = sp_circ.max_errors
    layers: list[list[npt.NDArray[np.int8]]] = [[] for _ in range(max_errors)]
    sp_circ.compute_fault_sets()
    fault_sets = (
        sp_circ.combine_faults(additional_faults, x_errors)
        if additional_faults is not None
        else sp_circ.x_fault_sets
        if x_errors
        else sp_circ.z_fault_sets
    )
    orthogonal_checks = sp_circ.z_checks if x_errors else sp_circ.x_checks
    for num_errors in range(1, max_errors + 1):
        logger.info(f"Finding verification stabilizers for {num_errors} errors")
        faults = fault_sets[num_errors]
        assert faults is not None
        logger.info(f"There are {len(faults)} faults")
        if len(faults) == 0:
            layers[num_errors - 1] = []
            continue

        layers[num_errors - 1] = _heuristic_layer(faults, orthogonal_checks, find_coset_leaders, max_covering_sets)

    return layers


def _covers(s: npt.NDArray[np.int8], faults: npt.NDArray[np.int8]) -> frozenset[int]:
    return frozenset(np.where(s @ faults.T % 2 != 0)[0])


def _set_cover(
    n: int, cands: set[frozenset[int]], mapping: dict[frozenset[int], list[npt.NDArray[np.int8]]]
) -> set[frozenset[int]]:
    universe = set(range(n))
    cover: set[frozenset[int]] = set()

    while universe:
        best = max(cands, key=lambda stab: (len(stab & universe), -np.sum(mapping[stab])))  # type: ignore[operator]
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


def _measure_ft_x(qc: QuantumCircuit, x_measurements: list[npt.NDArray[np.int8]], t: int, flags: bool = False) -> None:
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


def _measure_ft_z(qc: QuantumCircuit, z_measurements: list[npt.NDArray[np.int8]], t: int, flags: bool = False) -> None:
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
    sp_circ: StatePrepCircuit,
    x_measurements: list[npt.NDArray[np.int8]],
    z_measurements: list[npt.NDArray[np.int8]],
    full_fault_tolerance: bool = True,
    flag_first_layer: bool = False,
) -> QuantumCircuit:
    # Create the verification circuit
    q = QuantumRegister(sp_circ.num_qubits, "q")
    measured_circ = QuantumCircuit(q)
    measured_circ.compose(sp_circ.circ, inplace=True)

    if sp_circ.zero_state:
        _measure_ft_z(measured_circ, z_measurements, t=sp_circ.max_x_errors, flags=flag_first_layer)
        if full_fault_tolerance:
            _measure_ft_x(measured_circ, x_measurements, flags=True, t=sp_circ.max_x_errors)
    else:
        _measure_ft_x(measured_circ, x_measurements, t=sp_circ.max_z_errors, flags=flag_first_layer)
        if full_fault_tolerance:
            _measure_ft_z(measured_circ, z_measurements, flags=True, t=sp_circ.max_z_errors)

    return measured_circ


def vars_to_stab(
    measurement: list[z3.BoolRef | bool], generators: npt.NDArray[np.int8]
) -> npt.NDArray[z3.BoolRef | bool]:
    """Compute the stabilizer measured giving the generators and the measurement variables."""
    measurement_stab = symbolic_scalar_mult(generators[0], measurement[0])
    for i, scalar in enumerate(measurement[1:]):
        measurement_stab = symbolic_vector_add(measurement_stab, symbolic_scalar_mult(generators[i + 1], scalar))
    return measurement_stab


def verification_stabilizers(
    sp_circ: StatePrepCircuit,
    fault_set: npt.NDArray[np.int8],
    num_anc: int,
    num_cnots: int,
    x_errors: bool = True,
) -> list[npt.NDArray[np.int8]] | None:
    """Return a verification stabilizers for num_errors independent errors in the state preparation circuit using z3.

    Args:
        sp_circ: The state preparation circuit.
        fault_set: The set of errors to verify.
        num_anc: The maximum number of ancilla qubits to use.
        num_cnots: The maximum number of CNOT gates to use.
        x_errors: If True, the errors are X errors. Otherwise, the errors are Z errors.
    """
    stabs_list = all_verification_stabilizers(
        sp_circ, fault_set, num_anc, num_cnots, x_errors, return_all_solutions=False
    )
    if stabs_list:
        return stabs_list[0]
    return None


def all_verification_stabilizers(
    sp_circ: StatePrepCircuit,
    fault_set: npt.NDArray[np.int8],
    num_anc: int,
    num_cnots: int,
    x_errors: bool = True,
    return_all_solutions: bool = False,
) -> list[list[npt.NDArray[np.int8]]] | None:
    """Return a list of verification stabilizers for num_errors independent errors in the state preparation circuit using z3.

    Args:
        sp_circ: The state preparation circuit.
        fault_set: The set of errors to verify.
        num_anc: The maximum number of ancilla qubits to use.
        num_cnots: The maximum number of CNOT gates to use.
        x_errors: If True, the errors are X errors. Otherwise, the errors are Z errors.
        return_all_solutions: If True, return all solutions. Otherwise, return the first solution found.
    """
    # Measurements are written as sums of generators
    # The variables indicate which generators are non-zero in the sum
    gens = sp_circ.z_checks if x_errors else sp_circ.x_checks
    n_gens = gens.shape[0]

    measurement_vars = [[z3.Bool(f"m_{anc}_{i}") for i in range(n_gens)] for anc in range(num_anc)]
    solver = z3.Solver()

    measurement_stabs = [vars_to_stab(vars_, gens) for vars_ in measurement_vars]

    # assert that each error is detected
    solver.add(
        z3.And([
            z3.PbGe([(odd_overlap(measurement, error), 1) for measurement in measurement_stabs], 1)
            for error in fault_set
        ])
    )

    # assert that not too many CNOTs are used
    solver.add(
        z3.PbLe(
            [(measurement[q], 1) for measurement in measurement_stabs for q in range(sp_circ.num_qubits)], num_cnots
        )
    )

    solutions = []
    while solver.check() == z3.sat:
        model = solver.model()
        # Extract stabilizer measurements from model
        actual_measurements = []
        for m in measurement_vars:
            v = np.zeros(sp_circ.num_qubits, dtype=np.int8)
            for g in range(n_gens):
                if model[m[g]]:
                    v += gens[g]
            actual_measurements.append(v % 2)
        if not return_all_solutions:
            return [actual_measurements]
        solutions.append(actual_measurements)
        # add constraint to avoid same solution again
        solver.add(z3.Or([vars_[i] != model[vars_[i]] for vars_ in measurement_vars for i in range(n_gens)]))
    if solutions:
        return solutions
    return None


def coset_leader(error: npt.NDArray[np.int8], generators: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    """Compute the coset leader of an error given a set of generators."""
    if len(generators) == 0:
        return error
    s = z3.Optimize()
    leader = [z3.Bool(f"e_{i}") for i in range(len(error))]
    coeff = [z3.Bool(f"c_{i}") for i in range(len(generators))]

    g = vars_to_stab(coeff, generators)

    s.add(symbolic_vector_eq(np.array(leader), symbolic_vector_add(error.astype(bool), g)))
    s.minimize(z3.Sum(leader))

    s.check()  # always SAT
    m = s.model()
    return np.array([bool(m[leader[i]]) for i in range(len(error))]).astype(int)


def _propagate_error(nodes: list[DAGNode], n_qubits: int, x_errors: bool = True) -> PauliList:
    """Propagates a Pauli error through a circuit beginning from first node.

    Args:
        nodes: List of nodes in the circuit in topological order.
        n_qubits: Number of qubits in the circuit.
        x_errors: If True, propagate X errors. Otherwise, propagate Z errors.
    """
    start = nodes[0]
    error: npt.NDArray[np.int8] = np.array([0] * n_qubits, dtype=np.int8)
    error[start.qargs[0]._index] = 1  # noqa: SLF001
    error[start.qargs[1]._index] = 1  # noqa: SLF001
    # propagate error through circuit via bfs
    for node in nodes[1:]:
        control = node.qargs[0]._index  # noqa: SLF001
        target = node.qargs[1]._index  # noqa: SLF001
        if x_errors:
            error[target] = (error[target] + error[control]) % 2
        else:
            error[control] = (error[target] + error[control]) % 2
    return error


def _remove_trivial_faults(
    faults: npt.NDArray[np.int8], stabs: npt.NDArray[np.int8], num_errors: int
) -> npt.NDArray[np.int8]:
    faults = faults.copy()
    logger.info("Removing trivial faults.")
    max_w = 1
    for i, fault in enumerate(faults):
        faults[i] = coset_leader(fault, stabs)
    faults = faults[np.where(np.sum(faults, axis=1) > max_w * num_errors)[0]]

    # unique faults
    return np.unique(faults, axis=0)


def _remove_stabilizer_equivalent_faults(
    faults: npt.NDArray[np.int8], stabilizers: npt.NDArray[np.int8]
) -> npt.NDArray[np.int8]:
    """Remove stabilizer equivalent faults from a list of faults."""
    faults = faults.copy()
    stabilizers = stabilizers.copy()
    removed = set()

    logger.debug(f"Removing stabilizer equivalent faults from {len(faults)} faults.")
    for i, f1 in enumerate(faults):
        if i in removed:
            continue
        stabs_ext1 = np.vstack((stabilizers, f1))
        if mod2.rank(stabs_ext1) == mod2.rank(stabilizers):
            removed.add(i)
            continue

        for j, f2 in enumerate(faults[i + 1 :]):
            if j + i + 1 in removed:
                continue
            stabs_ext2 = np.vstack((stabs_ext1, f2))

            if mod2.rank(stabs_ext2) == mod2.rank(stabs_ext1):
                removed.add(j + i + 1)

    logger.debug(f"Removed {len(removed)} stabilizer equivalent faults.")
    indices = list(set(range(len(faults))) - removed)
    if len(indices) == 0:
        return np.array([])

    return faults[indices]


def naive_verification_circuit(sp_circ: StatePrepCircuit, flag_first_layer: bool = True) -> QuantumCircuit:
    """Naive verification circuit for a state preparation circuit."""
    if sp_circ.code.Hx is None or sp_circ.code.Hz is None:
        msg = "Code must have stabilizers defined."
        raise ValueError(msg)

    z_measurements = list(sp_circ.code.Hx)
    x_measurements = list(sp_circ.code.Hz)
    reps = sp_circ.max_errors
    return _measure_ft_stabs(sp_circ, z_measurements * reps, x_measurements * reps, flag_first_layer=flag_first_layer)


def get_hook_errors(measurements: list[npt.NDArray[np.int8]]) -> npt.NDArray[np.int8]:
    """Assuming CNOTs are executed in ascending order of qubit index, this function gives all the hook errors of the given stabilizer measurements."""
    errors = []
    for stab in measurements:
        support = np.where(stab == 1)[0]
        error = stab.copy()
        for qubit in support[:-1]:
            error[qubit] = 0
            errors.append(error.copy())

    return np.array(errors)


def final_matrix_constraint(columns: npt.NDArray[z3.BoolRef | bool], rank: int) -> z3.BoolRef:
    """Return a z3 constraint that the final matrix has exactly rank non-zero columns."""
    assert len(columns.shape) == 3
    return z3.PbEq(
        [(z3.Not(z3.Or(list(columns[-1, :, col]))), 1) for col in range(columns.shape[2])],
        columns.shape[2] - rank,
    )
