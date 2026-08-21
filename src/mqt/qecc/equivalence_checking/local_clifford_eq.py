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

from ..mod2 import nullspace, rank
from ._cliffords import CLIFFORD_ACTIONS, LOCAL_CLIFFORDS, CliffordMatrix, ColumnSource
from .utils import elementwise_map, encode_row_operations, exactly_one, reduce_stabilizer_generators

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..codes.core.stabilizer_code import StabilizerCode


def are_local_clifford_equivalent(code1: StabilizerCode, code2: StabilizerCode) -> list[str] | None:
    """Check whether two stabilizer codes are locally Clifford equivalent.

    Phase information is not considered.

    Args:
        code1: First stabilizer code.
        code2: Second stabilizer code.

    Returns:
        The single-qubit Clifford operations mapping ``code1`` to ``code2``,
        or ``None`` if no such operations exist.
    """
    code1 = reduce_stabilizer_generators(code1)
    code2 = reduce_stabilizer_generators(code2)

    cheap_invariants = (
        _preserved_n,
        _preserved_k,
        _preserved_d,
    )

    if not all(invariant(code1, code2) for invariant in cheap_invariants):
        return None

    if code1.k < 2:
        return _lse_stabilizer_code(code1, code2)

    if code1.n <= 30 and not preserved_low_degree_local_invariant(code1, code2):
        return None

    return _sat_stabilizer_code(code1, code2)


def is_local_clifford_equivalent_to_css(code: StabilizerCode) -> bool:
    """Check whether a stabilizer code is locally Clifford equivalent to a CSS code.

    Args:
        code: The stabilizer code to check.

    Returns:
        Whether the code is locally Clifford equivalent to a CSS code.
    """
    code = reduce_stabilizer_generators(code)

    if code.n < 4:
        return _bruteforce_css_code(code)

    return _sat_css_code(code)


# ----------------------------------------------------------------------------------------------------
#   Invariants
# ----------------------------------------------------------------------------------------------------


def _preserved_n(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check the number-of-qubits invariant for LC equivalence."""
    return c1.n == c2.n


def _preserved_k(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check the number-of-logical-qubits invariant for LC equivalence."""
    return c1.k == c2.k


def _preserved_d(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check the code-distance invariant for LC equivalence."""
    return c1.distance == c2.distance


def preserved_low_degree_local_invariant(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the degree-2 local invariant is preserved.

    The invariant is described by Van den Nest, Dehaene, and De Moor:
    https://doi.org/10.1103/PhysRevA.70.032323.
    """
    n = c1.n
    rk = c1.k

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
            if _supp_subcode_dim(c1.symplectic, subset) != _supp_subcode_dim(c2.symplectic, subset):
                return False

    return True


# ----------------------------------------------------------------------------------------------------
#   Decision procedures
# ----------------------------------------------------------------------------------------------------


def _bruteforce_css_code(c: StabilizerCode) -> bool:
    """Check LC equivalence to a CSS code by enumerating local Cliffords."""
    r, n = c.symplectic.shape[0], c.n

    for action in product(LOCAL_CLIFFORDS, repeat=n):
        lc_tableau = c.symplectic.copy()

        for qubit, lc in enumerate(action):
            _apply_local_clifford(lc_tableau, lc, qubit)

        if rank(lc_tableau[:, :n]) + rank(lc_tableau[:, n:]) == r:
            return True

    return False


def _lse_stabilizer_code(c1: StabilizerCode, c2: StabilizerCode) -> list[str] | None:
    """Check LC equivalence by solving a linear system over graph states.

    Implements the efficient algorithm by Van den Nest, Dehaene, and De Moor:
    https://doi.org/10.1103/PhysRevA.70.034302.
    """

    def _stab_code_to_stab_state(code: StabilizerCode) -> np.ndarray:
        """Extend a stabilizer code to a purified stabilizer state."""
        if code.k == 0:
            return code.symplectic.copy()

        n = code.n
        r = code.symplectic.shape[0]
        k = code.k

        stab_x = code.symplectic[:, :n]
        stab_z = code.symplectic[:, n:]

        log_x_x = code.x_logicals.tableau.data[:, :n]
        log_x_z = code.x_logicals.tableau.data[:, n:]

        log_z_x = code.z_logicals.tableau.data[:, :n]
        log_z_z = code.z_logicals.tableau.data[:, n:]

        stabilizer_part = np.hstack([stab_x, np.zeros((r, k), dtype=np.int8), stab_z, np.zeros((r, k), dtype=np.int8)])
        logical_x_part = np.hstack([log_x_x, np.eye(k, dtype=np.int8), log_x_z, np.zeros((k, k), dtype=np.int8)])
        logical_z_part = np.hstack([log_z_x, np.zeros((k, k), dtype=np.int8), log_z_z, np.eye(k, dtype=np.int8)])

        return np.vstack([stabilizer_part, logical_x_part, logical_z_part]).astype(np.int8)

    def _stab_state_to_graph_state(tableau: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Convert a stabilizer state to an LC-equivalent graph state."""
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
            """Apply phase gates to remove self-loops from the graph state."""
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

            operation = _clifford_from_lse_coefficients(int(a_i), int(b_i), int(c_i), int(d_i))
            if operation is None:
                return None
            lc[i] = operation
        return lc

    def _lc_equiv_connected(g1: np.ndarray, g2: np.ndarray, n: int) -> list[str] | None:
        """Check LC equivalence of two connected graph states using a linear system."""

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
        """Replace local Clifford words with canonical representatives."""
        return [_canonicalize_clifford(operation) for operation in lc]

    stab_state1 = _stab_code_to_stab_state(c1)
    stab_state2 = _stab_code_to_stab_state(c2)

    graph_state1, lc1 = _stab_state_to_graph_state(stab_state1)
    graph_state2, lc2 = _stab_state_to_graph_state(stab_state2)

    leq = _lc_equiv_graph_states(graph_state1, graph_state2)

    if leq is None:
        return None

    result = [op2[::-1] + op_eq + op1 for op1, op_eq, op2 in zip(lc1, leq, lc2, strict=True)]
    return _simplify_lc_operations(result[: c1.n])


def _sat_stabilizer_code(c1: StabilizerCode, c2: StabilizerCode) -> list[str] | None:
    """Check LC equivalence of two stabilizer codes using a SAT encoding."""
    solver = z3.Solver()

    r, n = c1.symplectic.shape[0], c1.n
    aux_tableau = [z3.Bool(f"aux_{row}_{col}") for row in range(r) for col in range(2 * n)]
    local_clifford_variables = _encode_local_cliffords(solver, c1.symplectic, aux_tableau)
    encode_row_operations(solver, aux_tableau, c2.symplectic, variable_prefix="r")

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


def _sat_css_code(c: StabilizerCode) -> bool:
    """Check LC equivalence to a CSS code using a SAT encoding.

    This encoding is based on the ideas by Dasu and Burton:
    https://arxiv.org/abs/2507.10519.
    """
    solver = z3.Solver()

    r, n = c.symplectic.shape[0], c.n
    aux_tableau = [z3.Bool(f"aux_{row}_{col}") for row in range(r) for col in range(2 * n)]
    _encode_local_cliffords(solver, c.symplectic, aux_tableau, project_to_css=True)
    encode_row_operations(solver, aux_tableau, c.symplectic, variable_prefix="r")

    return solver.check() == z3.sat


# ----------------------------------------------------------------------------------------------------
#   Helper functions
# ----------------------------------------------------------------------------------------------------


def _encode_local_cliffords(
    solver: z3.Solver,
    source_tableau: npt.NDArray[np.integer],
    auxiliary_tableau: list[z3.BoolRef],
    *,
    project_to_css: bool = False,
) -> list[dict[str, z3.BoolRef]]:
    """Encode a local Clifford action on a symplectic tableau."""
    rows, twice_n = source_tableau.shape
    n = twice_n // 2
    variables = [{operation: z3.Bool(f"lc_{qubit}_{operation}") for operation in LOCAL_CLIFFORDS} for qubit in range(n)]

    for qubit, qubit_variables in enumerate(variables):
        solver.add(exactly_one(qubit_variables.values()))
        x_column = source_tableau[:, qubit]
        z_column = source_tableau[:, qubit + n]
        auxiliary_x = [auxiliary_tableau[row * twice_n + qubit] for row in range(rows)]
        auxiliary_z = [auxiliary_tableau[row * twice_n + qubit + n] for row in range(rows)]

        for operation, action in CLIFFORD_ACTIONS.items():
            x_source, z_source = action.css_projected_columns if project_to_css else action.transformed_columns
            solver.add(
                z3.Implies(
                    qubit_variables[operation],
                    z3.And(
                        elementwise_map(_select_column(x_source, x_column, z_column), auxiliary_x),
                        elementwise_map(_select_column(z_source, x_column, z_column), auxiliary_z),
                    ),
                )
            )

    return variables


def _select_column(
    source: ColumnSource, x_column: npt.NDArray[np.integer], z_column: npt.NDArray[np.integer]
) -> npt.NDArray[np.integer]:
    """Select a binary column expression from two symplectic columns."""
    if source == "zero":
        return np.zeros_like(x_column)
    if source == "x":
        return x_column
    if source == "z":
        return z_column
    return (x_column + z_column) % 2


def _apply_local_clifford(tableau: npt.NDArray[np.int8], operation: str, qubit: int) -> None:
    """Apply a local Clifford representative to one tableau qubit in place."""
    n = tableau.shape[1] // 2
    x_column = tableau[:, qubit].copy()
    z_column = tableau[:, qubit + n].copy()
    x_source, z_source = CLIFFORD_ACTIONS[operation].transformed_columns
    tableau[:, qubit] = _select_column(x_source, x_column, z_column)
    tableau[:, qubit + n] = _select_column(z_source, x_column, z_column)


def _canonicalize_clifford(word: str) -> str:
    """Return the canonical representative of a Clifford word."""
    matrix = np.eye(2, dtype=np.uint8)
    for gate in word:
        if gate not in {"H", "S", "I"}:
            msg = f"Unknown Clifford gate {gate!r}."
            raise ValueError(msg)
        matrix = (matrix @ np.asarray(CLIFFORD_ACTIONS[gate].matrix, dtype=np.uint8)) % 2

    matrices = {action.matrix: name for name, action in CLIFFORD_ACTIONS.items()}
    key: CliffordMatrix = (
        (int(matrix[0, 0]), int(matrix[0, 1])),
        (int(matrix[1, 0]), int(matrix[1, 1])),
    )
    return matrices[key]


def _clifford_from_lse_coefficients(a: int, b: int, c: int, d: int) -> str | None:
    """Decode an LSE solution using the transpose matrix convention."""
    lse_matrix = ((a, b), (c, d))
    return next(
        (name for name, action in CLIFFORD_ACTIONS.items() if tuple(zip(*action.matrix, strict=True)) == lse_matrix),
        None,
    )
