# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Functions for deciding local clifford equivalence."""

from __future__ import annotations

from collections import deque
from itertools import combinations, product
from typing import TYPE_CHECKING

import numpy as np
import z3

from ..mod2 import nullspace, rank, row_basis

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt

    from ..codes.core.stabilizer_code import StabilizerCode


def are_local_clifford_equivalent(code1: StabilizerCode, code2: StabilizerCode) -> list[str] | None:
    """Check if two stabilizer codes define the same code space up to local Clifford operations on output qubits, not considering phase information.

    Args:
        code1: First stabilizer code.
        code2: Second stabilizer code.

    Returns:
        A list of strings representing the local Clifford operations on qubits that maps code1 to code2, or None if no such operations exist.
    """
    cheap_invariants = (
        _preserved_n,
        _preserved_k,
        _preserved_d,
    )

    if not all(invariant(code1, code2) for invariant in cheap_invariants):
        return None

    if code1.n < 1:
        return ["I"] * code1.n

    reduced_symplectic_1 = row_basis(code1.symplectic)
    reduced_symplectic_2 = row_basis(code2.symplectic)

    if code1.k < 2:
        return _lse_stabilizer_code(code1, code2, reduced_symplectic_1, reduced_symplectic_2)

    if code1.n <= 30 and not preserved_low_degree_local_invariant(reduced_symplectic_1, reduced_symplectic_2):
        return None

    return _sat_stabilizer_code(code1, code2)


def is_local_clifford_equivalent_to_css(code: StabilizerCode) -> bool:
    """Check if a stabilizer code is local clifford equivalent to a CSS code."""
    if code.n < 1:
        return True

    reduced_symplectic = row_basis(code.symplectic)

    if code.n < 4:
        return _bruteforce_css_code(reduced_symplectic)

    return _sat_css_code(reduced_symplectic)


# ----------------------------------------------------------------------------------------------------
#   Invariants
# ----------------------------------------------------------------------------------------------------


def _preserved_n(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of qubits is preserved, which is a necessary condition for LC-equivalence."""
    return c1.n == c2.n


def _preserved_k(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the number of logical qubits is preserved, which is a necessary condition for LC-equivalence."""
    return c1.k == c2.k


def _preserved_d(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the distance is preserved, which is a necessary condition for LC-equivalence."""
    return c1.distance == c2.distance


def preserved_low_degree_local_invariant(c1: np.ndarray, c2: np.ndarray) -> bool:
    """Check whether the degree-2 local invariant is preserved, which is a necessary condition for LC-equivalence."""
    n = c1.shape[1] // 2
    rk = n - c1.shape[0]

    def _supp_subcode_dim(code: np.ndarray, subset: tuple[int, ...]) -> int:
        g = np.asarray(code, dtype=np.uint8) & 1

        a = set(subset)
        outside = [i for i in range(n) if i not in a]

        cols = outside + [i + n for i in outside]

        if not cols:
            return rk

        restricted = g[:, cols]
        return rk - rank(restricted)

    max_subset_size = 2
    for subset_size in range(max_subset_size + 1):
        for subset in combinations(range(n), subset_size):
            if _supp_subcode_dim(c1, subset) != _supp_subcode_dim(c2, subset):
                return False

    return True


# ----------------------------------------------------------------------------------------------------
#   Brute force algorithms
# ----------------------------------------------------------------------------------------------------


def _bruteforce_css_code(tableau: np.ndarray) -> bool:
    """lc_css_bruteforce.py."""
    r, n = tableau.shape[0], tableau.shape[1] // 2

    def apply_lc(tableau: npt.NDArray[np.int8], lc: str, qubit: int) -> None:
        if lc == "I":
            pass
        elif lc == "H":
            tableau[:, [qubit, qubit + n]] = tableau[:, [qubit + n, qubit]]
        elif lc == "S":
            tableau[:, qubit + n] ^= tableau[:, qubit]
        elif lc == "HS":
            tableau[:, qubit + n] ^= tableau[:, qubit]
            tableau[:, [qubit, qubit + n]] = tableau[:, [qubit + n, qubit]]
        elif lc == "SH":
            tableau[:, qubit] ^= tableau[:, qubit + n]
            tableau[:, [qubit, qubit + n]] = tableau[:, [qubit + n, qubit]]
        elif lc == "HSH":
            tableau[:, qubit] ^= tableau[:, qubit + n]

    for action in product(LOCAL_CLIFFORDS, repeat=n):
        lc_tableau = tableau.copy()

        for qubit, lc in enumerate(action):
            apply_lc(lc_tableau, lc, qubit)

        if rank(lc_tableau[:, :n]) + rank(lc_tableau[:, n:]) == r:
            return True

    return False


# ----------------------------------------------------------------------------------------------------
#   Decision procedures
# ----------------------------------------------------------------------------------------------------
LOCAL_CLIFFORDS = ("I", "H", "S", "HS", "SH", "HSH")


def _lse_stabilizer_code(
    c1: StabilizerCode, c2: StabilizerCode, reduced_symplectic_1: np.ndarray, reduced_symplectic_2: np.ndarray
) -> list[str] | None:
    """Use a linear system of equations to check if two stabilizer codes are equivalent under local clifford operations."""

    def _stab_code_to_stab_state(code: StabilizerCode, reduced_symplectic: np.ndarray) -> np.ndarray:
        """Convert a stabilizer code into a stabilizer state using the Choi-Jamiolkowski isomorphism."""
        if code.k == 0:
            return reduced_symplectic.copy()

        n = code.n
        r = reduced_symplectic.shape[0]
        k = code.k

        stab_x = reduced_symplectic[:, :n]
        stab_z = reduced_symplectic[:, n:]

        log_x_x = code.x_logicals.tableau.data[:, :n]
        log_x_z = code.x_logicals.tableau.data[:, n:]

        log_z_x = code.z_logicals.tableau.data[:, :n]
        log_z_z = code.z_logicals.tableau.data[:, n:]

        stabilizer_part = np.hstack([stab_x, np.zeros((r, k), dtype=np.int8), stab_z, np.zeros((r, k), dtype=np.int8)])
        logical_x_part = np.hstack([log_x_x, np.eye(k, dtype=np.int8), log_x_z, np.zeros((k, k), dtype=np.int8)])
        logical_z_part = np.hstack([log_z_x, np.zeros((k, k), dtype=np.int8), log_z_z, np.eye(k, dtype=np.int8)])

        return np.vstack([stabilizer_part, logical_x_part, logical_z_part]).astype(np.int8)

    def _stab_state_to_graph_state(tableau: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Convert a stabilizer state into a graph state under local Clifford operations."""
        n = tableau.shape[1] // 2
        lc = [""] * n

        def _make_x_invertible(t: np.ndarray) -> np.ndarray:
            old_x_rank = rank(t[:, :n])
            while old_x_rank < n:
                improved = False

                for q in range(n):
                    if old_x_rank == n:
                        break

                    best_rank = old_x_rank
                    best_choice = (None, None)
                    best_op = None

                    x_col = t[:, q].copy()
                    z_col = t[:, q + n].copy()

                    for new_x, new_z, op in [
                        (x_col, z_col, ""),
                        (z_col, x_col, "H"),
                        ((x_col + z_col) % 2, x_col, "HS"),
                    ]:
                        t[:, q] = new_x
                        new_x_rank = rank(t[:, :n])
                        if new_x_rank > best_rank:
                            best_rank = new_x_rank
                            best_choice = (new_x, new_z)
                            best_op = op

                    if best_choice[0] is not None:
                        t[:, q] = best_choice[0]
                        t[:, q + n] = best_choice[1]
                        old_x_rank = best_rank
                        lc[q] = best_op
                        improved = True
                    else:
                        t[:, q] = x_col
                        t[:, q + n] = z_col

                if not improved:
                    break

            return t

        def _extract_adjacency_matrix(tableau: np.ndarray) -> np.ndarray:
            """Extract the adjacency matrix from the stabilizer state."""

            def _rref_no_column_swaps(matrix: np.ndarray) -> tuple[np.ndarray, int]:
                n_rows, n_cols = matrix.shape
                pivot_row = 0
                for col in range(n_cols // 2):
                    if pivot_row >= n_rows:
                        break

                    tail = matrix[pivot_row:, col]
                    pivot_offset = int(np.argmax(tail))

                    if not tail[pivot_offset]:
                        continue

                    pivot = pivot_row + pivot_offset

                    if pivot != pivot_row:
                        matrix[[pivot_row, pivot], :] = matrix[[pivot, pivot_row], :]

                    for r in range(n_rows):
                        if r != pivot_row and matrix[r, col]:
                            matrix[r, :] ^= matrix[pivot_row, :]
                    pivot_row += 1

                return matrix, pivot_row

            rre, rank_x = _rref_no_column_swaps(tableau)

            if rank_x != n:
                msg = "X part of the tableau is not full rank, something went wrong."
                raise ValueError(msg)

            return rre[:, n:]

        def _remove_diagonal(tableau: np.ndarray) -> None:
            """Basically apply S gate on all qubits to remove self-loops in the graph state."""
            for i in range(tableau.shape[0]):
                if tableau[i, i] == 1:
                    lc[i] = "S" + lc[i]
                    tableau[i, i] = 0

        state = _make_x_invertible(tableau)
        gamma = _extract_adjacency_matrix(state)
        _remove_diagonal(gamma)

        if not np.array_equal(gamma, gamma.T):
            msg = "Extracted adjacency matrix is not symmetric, something went wrong."
            raise ValueError(msg)

        return gamma, lc

    def _extract_connected_components(g: np.ndarray) -> list[list[int]]:
        n = g.shape[0]
        connected_components: list[list[int]] = []
        seen: set[int] = set()

        while len(seen) < n:
            start = next(i for i in range(n) if i not in seen)
            comp = []

            queue = deque([start])
            seen.add(start)

            while queue:
                cur: int = queue.popleft()
                comp.append(cur)

                for neighbor in g[cur, :].nonzero()[0]:
                    nb = int(neighbor)

                    if nb not in seen:
                        seen.add(nb)
                        queue.append(nb)

            connected_components.append(sorted(comp))

        return connected_components

    def _extract_lc_operation(x: np.ndarray) -> list[str] | None:
        n = len(x) // 4
        lc = [""] * n
        for i in range(n):  # code1 --op--> code2
            a_i = x[i]
            b_i = x[n + i]
            c_i = x[2 * n + i]
            d_i = x[3 * n + i]

            if (a_i, b_i, c_i, d_i) == (1, 0, 0, 1):
                lc[i] = "I"
            elif (a_i, b_i, c_i, d_i) == (0, 1, 1, 0):
                lc[i] = "H"
            elif (a_i, b_i, c_i, d_i) == (1, 1, 0, 1):
                lc[i] = "S"
            elif (a_i, b_i, c_i, d_i) == (0, 1, 1, 1):
                lc[i] = "HS"
            elif (a_i, b_i, c_i, d_i) == (1, 1, 1, 0):
                lc[i] = "SH"
            elif (a_i, b_i, c_i, d_i) == (1, 0, 1, 1):
                lc[i] = "HSH"
            else:
                return None
        return lc

    def _lc_equiv_connected(g1: np.ndarray, g2: np.ndarray, n: int) -> list[str] | None:
        """Check if two graph states are equivalent under local complementations using an efficient algorithm that considers a linear system of equations."""

        def _build_lse() -> npt.NDArray[np.uint8]:
            """Build the matrix for the following LSE.

            ( sum_{i=0}^{n-1} g1[i,j] * g2[i,k] * c_i ) + g1[j,k] * a_k + g2[j,k] * d_j + delta[j,k] * b_j = 0
            with n^2 equations for j,k = 0...n-1 and the following 4n unknowns:
                [a_0,...,a_{n-1},
                b_0,...,b_{n-1},
                c_0,...,c_{n-1},
                d_0,...,d_{n-1}].
            """
            matrix = np.zeros((n * n, 4 * n), dtype=np.uint8)

            def a_idx(i: int) -> int:
                return i

            def b_idx(i: int) -> int:
                return n + i

            def d_idx(i: int) -> int:
                return 3 * n + i

            row = 0
            for j in range(n):
                for k in range(n):
                    # sum_{i=0}^{n-1} g1[i,j] * g2[i,k] * c_i
                    matrix[row, 2 * n : 3 * n] = g1[j, :] & g2[:, k]
                    # g1[j, k] * a_k
                    matrix[row, a_idx(k)] ^= g1[j, k]
                    # g2[j, k] * d_j
                    matrix[row, d_idx(j)] ^= g2[j, k]
                    # delta[j, k] * b_j
                    if j == k:
                        matrix[row, b_idx(j)] ^= 1
                    row += 1
            return matrix

        def _satisfy_constraints(x: np.ndarray) -> bool:
            """Check that a solution of the LSE satisfies the determinant constraints.

            For each unknown with i = 0...n-1, a_i d_i + b_i c_i = 1.
            """
            x = np.asarray(x, dtype=np.uint8) % 2
            a = x[0:n]
            b = x[n : 2 * n]
            c = x[2 * n : 3 * n]
            d = x[3 * n : 4 * n]
            dets = (a & d) ^ (b & c)
            return bool(np.all(dets == 1))

        lse = _build_lse()
        solution_space = nullspace(lse).astype(np.uint8)

        dim = solution_space.shape[0]

        if dim == 0:  # trivial nullspace
            return None

        if dim > 4:
            for i in range(dim):
                for j in range(i, dim):
                    x = solution_space[i] ^ solution_space[j]
                    if _satisfy_constraints(x):
                        return _extract_lc_operation(x)
        else:
            for coeffs in product([0, 1], repeat=dim):
                x = np.zeros(4 * n, dtype=np.uint8)
                for bit, basis_vec in zip(coeffs, solution_space, strict=False):
                    if bit:
                        x ^= basis_vec

                if _satisfy_constraints(x):
                    return _extract_lc_operation(x)

        return None

    def _lc_equiv_graph_states(graph_1: np.ndarray, graph_2: np.ndarray) -> list[str] | None:
        lc = [""] * graph_1.shape[0]
        connected_components_g1 = sorted(tuple(comp) for comp in _extract_connected_components(graph_1))
        connected_components_g2 = sorted(tuple(comp) for comp in _extract_connected_components(graph_2))

        if connected_components_g1 != connected_components_g2:
            return None

        for comp in connected_components_g1:
            comp_idx = list(comp)

            component_lc = _lc_equiv_connected(
                graph_1[np.ix_(comp_idx, comp_idx)], graph_2[np.ix_(comp_idx, comp_idx)], len(comp_idx)
            )
            if component_lc is None:
                return None

            for global_q, operation in zip(comp_idx, component_lc, strict=True):
                lc[global_q] = operation

        return lc

    def _simplify_lc_operations(lc: list[str]) -> list[str]:
        """Simplify a list of local clifford operations by removing identity operations."""
        identity = np.eye(2, dtype=np.uint8)
        h = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        s = np.array([[1, 0], [1, 1]], dtype=np.uint8)

        representatives = {
            "I": identity,
            "H": h,
            "S": s,
            "HS": (h @ s) % 2,
            "SH": (s @ h) % 2,
            "HSH": (h @ s @ h) % 2,
        }
        name_by_matrix = {tuple(matrix.flat): name for name, matrix in representatives.items()}

        def canonicalize(word: str) -> str:
            matrix = identity.copy()

            for gate in word:
                if gate == "H":
                    matrix = (matrix @ h) % 2
                elif gate == "S":
                    matrix = (matrix @ s) % 2
                elif gate != "I":
                    msg = f"Unknown Clifford gate {gate!r}."
                    raise ValueError(msg)

            return name_by_matrix[tuple(matrix.flat)]

        return [canonicalize(operation) for operation in lc]

    stab_state1 = _stab_code_to_stab_state(c1, reduced_symplectic_1)
    stab_state2 = _stab_code_to_stab_state(c2, reduced_symplectic_2)

    graph_state1, lc1 = _stab_state_to_graph_state(stab_state1)
    graph_state2, lc2 = _stab_state_to_graph_state(stab_state2)

    leq = _lc_equiv_graph_states(graph_state1, graph_state2)

    if leq is None:
        return None

    result = [op2[::-1] + op_eq + op1 for op1, op_eq, op2 in zip(lc1, leq, lc2, strict=True)]
    return _simplify_lc_operations(result[: c1.n])


def _sat_stabilizer_code(c1: StabilizerCode, c2: StabilizerCode) -> list[str] | None:
    """Map the LC equivalence problem of two stabilizer codes to a SAT problem and solve it using Z3."""
    solver = z3.Solver()

    n = c1.n
    k = c1.k
    r = n - k

    # local cliffords
    aux_tableau = [z3.Bool(f"aux_{row}_{col}") for row in range(r) for col in range(2 * n)]

    local_clifford_variables = [
        {operation: z3.Bool(f"lc_{qubit}_{operation}") for operation in LOCAL_CLIFFORDS} for qubit in range(n)
    ]

    for qubit_variables in local_clifford_variables:
        solver.add(_exactly_one(qubit_variables.values()))

    for i in range(n):
        x_column_original = c1.symplectic[:, i]
        z_column_original = c1.symplectic[:, i + n]
        x_z_column_original = (x_column_original + z_column_original) % 2

        x_column_aux = [aux_tableau[row * (2 * n) + i] for row in range(r)]
        z_column_aux = [aux_tableau[row * (2 * n) + i + n] for row in range(r)]

        # I : (x, z) -> (x, z)
        solver.add(
            z3.Implies(
                local_clifford_variables[i]["I"],
                z3.And(
                    _elementwise_map(x_column_original, x_column_aux), _elementwise_map(z_column_original, z_column_aux)
                ),
            )
        )

        # H : (x, z) -> (z, x)
        solver.add(
            z3.Implies(
                local_clifford_variables[i]["H"],
                z3.And(
                    _elementwise_map(z_column_original, x_column_aux), _elementwise_map(x_column_original, z_column_aux)
                ),
            )
        )

        # S : (x, z) -> (x, x + z)
        solver.add(
            z3.Implies(
                local_clifford_variables[i]["S"],
                z3.And(
                    _elementwise_map(x_column_original, x_column_aux),
                    _elementwise_map(x_z_column_original, z_column_aux),
                ),
            )
        )

        # HS : (x, z) -> (x + z, x)
        solver.add(
            z3.Implies(
                local_clifford_variables[i]["HS"],
                z3.And(
                    _elementwise_map(x_z_column_original, x_column_aux),
                    _elementwise_map(x_column_original, z_column_aux),
                ),
            )
        )

        # SH : (x, z) -> (z, x + z)
        solver.add(
            z3.Implies(
                local_clifford_variables[i]["SH"],
                z3.And(
                    _elementwise_map(z_column_original, x_column_aux),
                    _elementwise_map(x_z_column_original, z_column_aux),
                ),
            )
        )

        # HSH : (x, z) -> (x + z, z)
        solver.add(
            z3.Implies(
                local_clifford_variables[i]["HSH"],
                z3.And(
                    _elementwise_map(x_z_column_original, x_column_aux),
                    _elementwise_map(z_column_original, z_column_aux),
                ),
            )
        )

    # row operations
    row_operation_coefficients = [z3.Bool(f"r_{i}_{j}") for i in range(r) for j in range(r)]

    for row in range(r):
        for q in range(2 * n):
            row_contributions = [
                row_operation_coefficients[row * r + contribution]
                for contribution in range(r)
                if c2.symplectic[contribution, q] == 1
            ]

            solver.add(aux_tableau[row * (2 * n) + q] == _xor_list(row_contributions))

    if solver.check() != z3.sat:
        return None

    model = solver.model()
    return [
        next(
            operation
            for operation, variable in qubit_variables.items()
            if z3.is_true(model.eval(variable, model_completion=True))
        )
        for qubit_variables in local_clifford_variables
    ]


def _sat_css_code(c: npt.NDArray[np.integer]) -> bool:
    """lc_css_sat.py."""
    solver = z3.Solver()

    r, n = c.shape[0], c.shape[1] // 2
    n - r

    # cliffords
    aux_tableau = [z3.Bool(f"aux_{row}_{col}") for row in range(r) for col in range(2 * n)]

    local_clifford_variables = [
        {operation: z3.Bool(f"lc_{qubit}_{operation}") for operation in LOCAL_CLIFFORDS} for qubit in range(n)
    ]

    for qubit_variables in local_clifford_variables:
        solver.add(_exactly_one(qubit_variables.values()))

    for i in range(n):
        x_column_original = c[:, i]
        z_column_original = c[:, i + n]
        x_z_column_original = (x_column_original + z_column_original) % 2
        zero_column_original = np.zeros_like(x_column_original)

        x_column_aux = [aux_tableau[row * (2 * n) + i] for row in range(r)]
        z_column_aux = [aux_tableau[row * (2 * n) + i + n] for row in range(r)]

        # I^(-1) P_x I : (x, z) -> (x, 0)
        solver.add(
            z3.Implies(
                local_clifford_variables[i]["I"],
                z3.And(
                    _elementwise_map(x_column_original, x_column_aux),
                    _elementwise_map(zero_column_original, z_column_aux),
                ),
            )
        )

        # H^(-1) P_x H : (x, z) -> (0, z)
        solver.add(
            z3.Implies(
                local_clifford_variables[i]["H"],
                z3.And(
                    _elementwise_map(zero_column_original, x_column_aux),
                    _elementwise_map(z_column_original, z_column_aux),
                ),
            )
        )

        # S^(-1) P_x S : (x, z) -> (x, x)
        solver.add(
            z3.Implies(
                local_clifford_variables[i]["S"],
                z3.And(
                    _elementwise_map(x_column_original, x_column_aux), _elementwise_map(x_column_original, z_column_aux)
                ),
            )
        )

        # (HS)^(-1) P_x (HS)  : (x, z) -> (z, z)
        solver.add(
            z3.Implies(
                local_clifford_variables[i]["HS"],
                z3.And(
                    _elementwise_map(z_column_original, x_column_aux), _elementwise_map(z_column_original, z_column_aux)
                ),
            )
        )

        # (SH)^(-1) P_x (SH) : (x, z) -> (0, x + z)
        solver.add(
            z3.Implies(
                local_clifford_variables[i]["SH"],
                z3.And(
                    _elementwise_map(zero_column_original, x_column_aux),
                    _elementwise_map(x_z_column_original, z_column_aux),
                ),
            )
        )

        # (HSH)^(-1) P_x (HSH) : (x, z) -> (x + z, 0)
        solver.add(
            z3.Implies(
                local_clifford_variables[i]["HSH"],
                z3.And(
                    _elementwise_map(x_z_column_original, x_column_aux),
                    _elementwise_map(zero_column_original, z_column_aux),
                ),
            )
        )

    # row operations
    row_operation_coefficients = [z3.Bool(f"r_{i}_{j}") for i in range(r) for j in range(r)]

    for row in range(r):
        for q in range(2 * n):
            row_contributions = [
                row_operation_coefficients[row * r + contribution]
                for contribution in range(r)
                if c[contribution, q] == 1
            ]

            solver.add(aux_tableau[row * (2 * n) + q] == _xor_list(row_contributions))

    return solver.check() == z3.sat


# ----------------------------------------------------------------------------------------------------
#   Helper functions
# ----------------------------------------------------------------------------------------------------


def _elementwise_map(normal_bool: npt.NDArray[np.integer], variables: Sequence[z3.BoolRef]) -> z3.BoolRef:
    return z3.And([v if bit == 1 else z3.Not(v) for bit, v in zip(normal_bool, variables, strict=False)])


def _exactly_one(variables: Iterable[z3.BoolRef]) -> z3.BoolRef:
    return z3.PbEq([(v, 1) for v in variables], 1)


def _xor_list(variables: Sequence[z3.BoolRef]) -> z3.BoolRef:
    acc = z3.BoolVal(False)
    for v in variables:
        acc = z3.Xor(acc, v)
    return acc
