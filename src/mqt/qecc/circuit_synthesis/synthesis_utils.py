# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for synthesizing circuits."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Literal

import ldpc.mod2.mod2_numpy as mod2
import multiprocess
import numpy as np
import z3
from qiskit.circuit import AncillaRegister, ClassicalRegister, QuantumCircuit

from ..codes.pauli import CheckMatrix
from .circuits import CNOTCircuit
from .synthesis import synthesize_cnot

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import numpy.typing as npt
    from qiskit.circuit import AncillaQubit, Clbit, Qubit

    from .synthesis import CnotSynthesisConfig


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


Objective = Literal["eliminations", "depth"]


def cnot_encoding_circuit(
    checks: CheckMatrix, logicals: CheckMatrix, balance_checks: bool = False, config: CnotSynthesisConfig | None = None
) -> CNOTCircuit:
    """Synthesize an encoding circuit for the given CSS code using a heuristic greedy search.

    Args:
        checks: The stabilizer check matrix of the CSS code.
        logicals: The logical operator matrix of the CSS code.
        balance_checks: Whether to balance the entries of the stabilizer matrix via row operations.
    optimize_depth: Whether to optimize for depth (True) or number of CNOTs (False).

    Returns:
        The synthesized encoding circuit and the qubits that are used to encode the logical qubits.
    """
    logger.info("Starting encoding circuit synthesis.")

    n_stab = checks.num_rows()

    if balance_checks:
        reduce_checks_by_row_ops(checks, logicals)

    mat = CheckMatrix(np.vstack((checks.matrix, logicals.matrix)), type=checks.type)

    config.exact = False
    ops, reduced_checks = synthesize_cnot(mat, config=config)
    assert isinstance(reduced_checks, CheckMatrix)
    encoding_checks = CheckMatrix(reduced_checks.matrix[n_stab:, :], reduced_checks.type)
    config.exact = True
    final_ops, logicals = synthesize_cnot(encoding_checks, config=config)
    cnots = [(c.control, c.target) for c in reversed(ops)] + [(c.control, c.target) for c in reversed(final_ops)]

    return build_css_encoder_from_cnot_list(reduced_checks, logicals, cnots)


def build_css_encoder_from_cnot_list(
    checks: CheckMatrix, logicals: CheckMatrix, cnots: list[tuple[int, int]]
) -> CNOTCircuit:
    """Build a CSS encoding circuit from a list of CNOTs, given the stabilizers and logicals.

    Args:
        checks: The stabilizer check matrix of the CSS code.
        logicals: The logical operator matrix of the CSS code.
        cnots: The list of CNOT operations to apply.

    Returns:
        The synthesized encoding circuit.
    """
    if checks.type != logicals.type:
        msg = "Checks and logicals must be of the same type."
        raise ValueError(msg)

    check_matrix = checks.matrix
    logical_matrix = logicals.matrix
    n = checks.num_qubits()
    encoding_qubits = np.where(logical_matrix.sum(axis=0) != 0)[0]
    if checks.type == "X":
        hadamards = np.where(check_matrix.sum(axis=0) != 0)[0]
    else:
        hadamards = np.where(check_matrix.sum(axis=0) == 0)[0]

    hadamards = np.setdiff1d(hadamards, encoding_qubits)
    non_hadamards = [i for i in range(n) if i not in hadamards and i not in encoding_qubits]
    return CNOTCircuit.from_cnot_list(cnots, initialize_z=non_hadamards, initialize_x=hadamards)


def reduce_checks_by_row_ops(
    stabs: CheckMatrix,
    logicals: CheckMatrix,
) -> None:
    """Try to reduce the total number of 1s in [checks; logicals] by row ops on *checks* only.

    Allowed operation: for check rows i != j,
        checks[j] <- checks[j] + checks[i] (mod 2)

    Constraints:
    - the new row must not have larger weight than *either* of the two rows we used
      (same guard you had before),
    - the *global* number of 1s across checks and logicals must strictly decrease.

    The arrays are modified in place.
    """
    checks = stabs.matrix
    logical_matrix = logicals.matrix
    r, _n = checks.shape
    # logicals can be empty (shape (0, n)), that's fine

    def total_ones() -> int:
        return int(checks.sum() + logical_matrix.sum())

    improved = True
    while improved:
        improved = False
        total_ones()

        best_op: tuple[int, int] | None = None
        best_delta = 0  # positive = global reduction

        # try all check→check additions
        for i in range(r):
            row_i = checks[i]
            w_i = int(row_i.sum())
            for j in range(r):
                if i == j:
                    continue
                row_j = checks[j]
                w_j = int(row_j.sum())

                s = (row_j + row_i) % 2
                w_s = int(s.sum())

                # enforce "don't increase row weight" constraint
                if w_s > w_j or w_s > w_i:
                    continue

                # effect on global #ones:
                # only row_j changes
                delta = w_j - w_s  # positive = improvement
                if delta <= 0:
                    continue

                if delta > best_delta:
                    best_delta = delta
                    best_op = (i, j)

        if best_op is not None:
            i, j = best_op
            checks[j] = (checks[j] + checks[i]) % 2
            improved = True


def heuristic_gaussian_elimination(
    matrix: npt.NDArray[np.int8],
    parallel_elimination: bool = True,
    objective: Objective = "eliminations",
    lookahead_layers: int = 0,  # 0 = greedy, 1 = simulate-to-completion, n = n-layer lookahead
    layer_topks: list[int] | None = None,  # e.g. [4096, 256, 32]
) -> tuple[npt.NDArray[np.int8], list[tuple[int, int]]]:
    """Gaussian elimination over GF(2) column space with arbitrary (layer-based) lookahead.

    - objective="eliminations": minimize total column additions; ties by depth.
    - objective="depth": minimize number of parallel layers; ties by eliminations.

    Depth is counted by the conflict rule:
      start a new layer iff the next step would reuse a column already used in the open layer.

    layer_topks: list of candidate pool sizes per lookahead layer.
      - layer_topks[0] is used for the *current* layer (the real choice).
      - layer_topks[1] for the next layer in the lookahead, etc.
      - if we run out of values, we fall back to "simulate to completion" from there.

    Example:
      lookahead_layers=3, layer_topks=[2048, 256, 32]
      → try 2048 candidates for the first layer, for each try up to 256 candidates in the
        second layer, for each try up to 32 in the third; afterwards simulate to completion.
    """
    mat = matrix.copy()
    rank = mod2.rank(mat)

    if layer_topks is None:
        # sensible default: big first pool, much smaller afterwards
        layer_topks = [4096, 512, 64]

    # ---------- helpers ----------
    def is_reduced(m: npt.NDArray[np.int8]) -> bool:
        return bool(np.sum(~np.all(m == 0, axis=0)) == rank)

    def compute_costs(m: npt.NDArray[np.int8]) -> npt.NDArray[np.int64]:
        c = np.array(
            [[np.sum((m[:, i] + m[:, j]) % 2) for j in range(m.shape[1])] for i in range(m.shape[1])],
            dtype=np.int64,
        )
        c -= np.sum(m, axis=0)
        np.fill_diagonal(c, 1)
        return c

    def apply_elim_inplace(m: npt.NDArray[np.int8], cst: npt.NDArray[np.int64], i: int, j: int) -> None:
        m[:, j] = (m[:, i] + m[:, j]) % 2
        new_weights = np.sum((m[:, j][:, np.newaxis] + m) % 2, axis=0)
        col_weights = np.sum(m, axis=0)
        cst[j, :] = new_weights - col_weights
        cst[:, j] = new_weights - np.sum(m[:, j])
        np.fill_diagonal(cst, 1)

    def mask_used(cst: npt.NDArray[np.int64], used_mask: list[int]) -> np.ma.MaskedArray:
        mm = np.zeros_like(cst, dtype=bool)
        if used_mask:
            mm[used_mask, :] = True
            mm[:, used_mask] = True
        return np.ma.array(cst, mask=mm)  # type: ignore[no-untyped-call]

    def exact_argmin_pair(costs_unused: np.ma.MaskedArray, shape) -> tuple[int, int]:
        i, j = np.unravel_index(np.argmin(costs_unused), shape)
        return int(i), int(j)

    def topk_candidates(
        costs_full: npt.NDArray[np.int64],
        costs_unused: np.ma.MaskedArray,
        k: int,
    ) -> list[tuple[int, int]]:
        """Return up to k best (i,j) by masked costs (ascending), negative-only, including the true argmin."""
        i_star, j_star = exact_argmin_pair(costs_unused, costs_full.shape)
        cf = costs_unused.filled(10**9)
        k_eff = min(k, cf.size - 1) if cf.size > 1 else 1
        idx = np.argpartition(cf.ravel(), k_eff)[:k_eff]
        cand = {(i_star, j_star)}
        for t in idx:
            i, j = np.unravel_index(int(t), cf.shape)
            if costs_unused.mask[i, j]:
                continue
            if costs_full[i, j] < 0:
                cand.add((int(i), int(j)))
        return sorted(cand, key=lambda ij: (costs_full[ij[0], ij[1]], ij[0], ij[1]))

    # ----- simulation primitives -----
    def greedy_pick(c: npt.NDArray[np.int64], used_mask: list[int]) -> tuple[int, int] | None:
        cu = mask_used(c, used_mask) if parallel_elimination else np.ma.array(c, mask=np.zeros_like(c, dtype=bool))  # type: ignore[no-untyped-call]
        if (cu.count() == 0) or np.all(cu >= 0):
            return None
        return exact_argmin_pair(cu, c.shape)

    def rollout_current_layer(
        m0: npt.NDArray[np.int8],
        c0: npt.NDArray[np.int64],
        used_mask0: list[int],
        first_move: tuple[int, int] | None,
    ) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int64], list[int], int, int]:
        """Finish THIS layer (conflict-based), optionally starting with `first_move`."""
        m = m0.copy()
        c = c0.copy()
        used_mask = used_mask0.copy() if parallel_elimination else []
        pack: set[int] = set()
        steps_in_layer = 0
        steps_in_layer_total = 0
        layers_inc = 0

        def close_layer() -> None:
            nonlocal layers_inc, steps_in_layer, steps_in_layer_total
            if steps_in_layer > 0:
                layers_inc += 1
                steps_in_layer_total += steps_in_layer
                steps_in_layer = 0
            pack.clear()
            if parallel_elimination:
                used_mask.clear()

        def apply_step(i: int, j: int) -> None:
            nonlocal steps_in_layer
            if (i in pack) or (j in pack):
                close_layer()
            apply_elim_inplace(m, c, i, j)
            pack.update((i, j))
            if parallel_elimination:
                used_mask.extend([i, j])
            steps_in_layer += 1

        if first_move is not None:
            apply_step(*first_move)

        while True:
            nxt = greedy_pick(c, used_mask)
            stalled = (nxt is None) or (parallel_elimination and len(used_mask) == m.shape[1])
            if stalled:
                close_layer()
                break
            apply_step(*nxt)

        return m, c, used_mask, steps_in_layer_total, layers_inc

    def simulate_to_completion(
        m0: npt.NDArray[np.int8],
        c0: npt.NDArray[np.int64],
        used_mask0: list[int],
        first_move: tuple[int, int] | None,
    ) -> tuple[int, int]:
        """Greedy to the end; returns (steps, layers) with proper conflict-based depth."""
        m = m0.copy()
        c = c0.copy()
        used_mask = used_mask0.copy() if parallel_elimination else []
        total_steps = 0
        total_layers = 0

        # finish current layer from the (optional) first move
        m, c, used_mask, steps_inc, lay_inc = rollout_current_layer(m, c, used_mask, first_move)
        total_steps += steps_inc
        total_layers += lay_inc

        while not is_reduced(m):
            cu = mask_used(c, used_mask) if parallel_elimination else np.ma.array(c, mask=np.zeros_like(c, dtype=bool))  # type: ignore[no-untyped-call]
            if (cu.count() == 0) or np.all(cu >= 0):
                # triangularize at boundary
                m = mod2.row_echelon(m, full=True)[0]
                c = compute_costs(m)
                used_mask = [] if parallel_elimination else []
            m, c, used_mask, steps_lay, lay_cnt = rollout_current_layer(m, c, used_mask, first_move=None)
            total_steps += steps_lay
            total_layers += lay_cnt

        return total_steps, total_layers

    # --------- recursive lookahead over layers ---------
    def score_from_layer(
        m0: npt.NDArray[np.int8],
        c0: npt.NDArray[np.int64],
        used_mask0: list[int],
        layer_idx: int,
    ) -> tuple[int, int]:
        """Recursively score the best future starting from the boundary (m0,c0,used_mask0)
        looking ahead from layer `layer_idx`.
        Returns (steps, layers) from this point on.
        """
        if layer_idx >= lookahead_layers:
            # we've looked ahead far enough → just simulate rest
            return simulate_to_completion(m0, c0, used_mask0, first_move=None)

        # build candidate set for this lookahead layer
        cu = mask_used(c0, used_mask0) if parallel_elimination else np.ma.array(c0, mask=np.zeros_like(c0, dtype=bool))  # type: ignore[no-untyped-call]

        # if no candidates, just simulate to completion
        if (cu.count() == 0) or np.all(cu >= 0):
            return simulate_to_completion(m0, c0, used_mask0, first_move=None)

        k = layer_topks[layer_idx] if layer_idx < len(layer_topks) else layer_topks[-1]
        cands = topk_candidates(c0, cu, k)

        best_score: tuple[int, int, int, int, int] | None = None
        for ci, cj in cands:
            # finish THIS layer starting with (ci,cj)
            m1, c1, used1, steps_inc1, lay_inc1 = rollout_current_layer(m0, c0, used_mask0, (ci, cj))

            # recurse into the next layer
            future_steps, future_layers = score_from_layer(m1, c1, used1, layer_idx + 1)

            total_steps = steps_inc1 + future_steps
            total_layers = lay_inc1 + future_layers

            primary = total_steps if objective == "eliminations" else total_layers
            secondary = total_layers if objective == "eliminations" else total_steps
            tie_cost = int(c0[ci, cj])
            key = (primary, secondary, tie_cost, ci, cj)

            if (best_score is None) or (key < best_score):
                best_score = key

        # strip tie fields
        return (best_score[0], best_score[1])  # type: ignore[index]

    # ---------- main loop ----------
    costs = compute_costs(mat)
    used_mask_main: list[int] = []
    eliminations: list[tuple[int, int]] = []

    while not is_reduced(mat):
        cu_main = (
            mask_used(costs, used_mask_main)
            if parallel_elimination
            else np.ma.array(costs, mask=np.zeros_like(costs, dtype=bool))
        )  # type: ignore[no-untyped-call]

        if (
            (cu_main.count() == 0)
            or np.all(cu_main >= 0)
            or (parallel_elimination and len(used_mask_main) == mat.shape[1])
        ):
            if parallel_elimination and used_mask_main:
                used_mask_main = []
                continue
            logger.warning("Local minimum reached. Making matrix triangular.")
            mat = mod2.row_echelon(mat, full=True)[0]
            costs = compute_costs(mat)
            continue

        if lookahead_layers == 0:
            i, j = exact_argmin_pair(cu_main, costs.shape)
        else:
            # layer 0 lookahead, but we also need the actual (i,j), not just the score
            k0 = layer_topks[0] if len(layer_topks) > 0 else 4096
            cand0 = topk_candidates(costs, cu_main, k0)

            best_key = None
            best_move = None
            for ci, cj in cand0:
                # finish current layer with this real move
                m1, c1, used1, steps_inc1, lay_inc1 = rollout_current_layer(mat, costs, used_mask_main, (ci, cj))

                # recurse into further layers
                fut_steps, fut_layers = score_from_layer(m1, c1, used1, 1)

                total_steps = steps_inc1 + fut_steps
                total_layers = lay_inc1 + fut_layers

                primary = total_steps if objective == "eliminations" else total_layers
                secondary = total_layers if objective == "eliminations" else total_steps
                tie_cost = int(costs[ci, cj])
                key = (primary, secondary, tie_cost, ci, cj)

                if (best_key is None) or (key < best_key):
                    best_key = key
                    best_move = (ci, cj)

            i, j = best_move  # type: ignore[assignment]

        eliminations.append((i, j))
        apply_elim_inplace(mat, costs, i, j)
        if parallel_elimination:
            used_mask_main.extend([i, j])

    return mat, eliminations


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
    circ.h(hadamards)
    for i, j in cnots:
        circ.cx(i, j)
    return circ


def _column_addition_constraint(
    columns: npt.NDArray[np.bool_],
    col_add_vars: npt.NDArray[np.bool_],
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


def symbolic_vector_eq(v1: npt.NDArray[np.bool_] | list[z3.BoolRef], v2: npt.NDArray[np.bool_]) -> z3.BoolRef:
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


def odd_overlap(v_sym: npt.NDArray[np.bool_], v_con: npt.NDArray[np.int8]) -> z3.BoolRef:
    """Return True if the overlap of symbolic vector with constant vector is odd."""
    if np.array_equal(v_con, np.zeros(len(v_con), dtype=np.int8)):
        return z3.BoolVal(False)

    constraint = False
    for i, c in enumerate(v_con):
        if c != 1:
            continue
        constraint = z3.Xor(constraint, v_sym[i])
    return constraint


def symbolic_scalar_mult(v: npt.NDArray[np.int8], a: z3.BoolRef | bool) -> npt.NDArray[np.bool_]:
    """Multiply a concrete vector by a symbolic scalar."""
    return np.array([a if s == 1 else False for s in v])


def symbolic_vector_add(v1: npt.NDArray[np.bool_], v2: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
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


def vars_to_stab(measurement: list[z3.BoolRef | bool], generators: npt.NDArray[np.int8]) -> npt.NDArray[np.bool_]:
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
