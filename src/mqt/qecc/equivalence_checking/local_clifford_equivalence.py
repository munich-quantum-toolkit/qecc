# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Functions for deciding local clifford equivalence."""

from __future__ import annotations

from itertools import combinations, product
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import z3

from ..mod2 import nullspace, rank, row_echelon
from ._cliffords import (
    CLIFFORD_ACTIONS,
    LOCAL_CLIFFORDS,
    _apply_local_clifford,
    _canonicalize_clifford,
    _select_column,
)
from .utils import _elementwise_map, _encode_row_operations, _exactly_one, _reduce_stabilizer_generators

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..codes.core.stabilizer_code import StabilizerCode


# These algorithms are parameter-dependent dispatchers,
# combining the empirically best-performing algorithms
# for different code sizes and types, based on these thresholds.
BRUTEFORCE_CSS_MAX_QUBITS = 3
LOW_DEGREE_INVARIANT_MAX_QUBITS = 30


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
    code1 = _reduce_stabilizer_generators(code1)
    code2 = _reduce_stabilizer_generators(code2)

    cheap_invariants = (
        _preserved_n,
        _preserved_k,
        _preserved_d,
    )

    if not all(invariant(code1, code2) for invariant in cheap_invariants):
        return None

    if code1.k < 2:
        return _lse_stabilizer_code(code1, code2)

    if code1.n <= LOW_DEGREE_INVARIANT_MAX_QUBITS and not _preserved_low_degree_local_invariant(code1, code2):
        return None

    return _sat_stabilizer_code(code1, code2)


def is_local_clifford_equivalent_to_css(code: StabilizerCode) -> bool:
    """Check whether a stabilizer code is locally Clifford equivalent to a CSS code.

    Args:
        code: The stabilizer code to check.

    Returns:
        Whether the code is locally Clifford equivalent to a CSS code.
    """
    code = _reduce_stabilizer_generators(code)

    if code.n <= BRUTEFORCE_CSS_MAX_QUBITS:
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


def _preserved_low_degree_local_invariant(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check whether the degree-2 local invariant is preserved.

    The invariant is described by Van den Nest, Dehaene, and De Moor:
    https://doi.org/10.1103/PhysRevA.70.032323.
    """
    n = c1.n
    rk = c1.k

    def _supp_subcode_dim(code: npt.NDArray[np.integer], subset: tuple[int, ...]) -> int:
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
    graph_state1, lc1 = _stabilizer_state_to_graph_state(_stabilizer_code_to_state(c1))
    graph_state2, lc2 = _stabilizer_state_to_graph_state(_stabilizer_code_to_state(c2))

    components1 = sorted(
        tuple(sorted(component)) for component in nx.connected_components(nx.from_numpy_array(graph_state1))
    )
    components2 = sorted(
        tuple(sorted(component)) for component in nx.connected_components(nx.from_numpy_array(graph_state2))
    )
    if components1 != components2:
        return None

    equivalence_operations = [""] * graph_state1.shape[0]
    for component in components1:
        indices = list(component)
        component_operations = _locally_equivalent_connected_graphs(
            graph_state1[np.ix_(indices, indices)], graph_state2[np.ix_(indices, indices)]
        )
        if component_operations is None:
            return None
        for qubit, operation in zip(indices, component_operations, strict=True):
            equivalence_operations[qubit] = operation

    result = [
        op2[::-1] + equivalence + op1 for op1, equivalence, op2 in zip(lc1, equivalence_operations, lc2, strict=True)
    ]
    return [_canonicalize_clifford(operation) for operation in result[: c1.n]]


def _sat_stabilizer_code(c1: StabilizerCode, c2: StabilizerCode) -> list[str] | None:
    """Check LC equivalence of two stabilizer codes using a SAT encoding."""
    solver = z3.Solver()

    r, n = c1.symplectic.shape[0], c1.n
    aux_tableau = [z3.Bool(f"aux_{row}_{col}") for row in range(r) for col in range(2 * n)]
    local_clifford_variables = _encode_local_cliffords(solver, c1.symplectic, aux_tableau)
    _encode_row_operations(solver, aux_tableau, c2.symplectic, variable_prefix="r")

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
    _encode_row_operations(solver, aux_tableau, c.symplectic, variable_prefix="r")

    return solver.check() == z3.sat


# ----------------------------------------------------------------------------------------------------
#   Helper functions
# ----------------------------------------------------------------------------------------------------


def _stabilizer_code_to_state(code: StabilizerCode) -> npt.NDArray[np.integer]:
    """Extend a stabilizer code to a purified stabilizer state."""
    if code.k == 0:
        return code.symplectic.copy()

    n = code.n
    num_stabilizers = code.symplectic.shape[0]
    k = code.k
    stabilizer_x = code.symplectic[:, :n]
    stabilizer_z = code.symplectic[:, n:]
    logical_x_x = code.x_logicals.tableau.data[:, :n]
    logical_x_z = code.x_logicals.tableau.data[:, n:]
    logical_z_x = code.z_logicals.tableau.data[:, :n]
    logical_z_z = code.z_logicals.tableau.data[:, n:]

    stabilizer_part = np.hstack([
        stabilizer_x,
        np.zeros((num_stabilizers, k), dtype=np.int8),
        stabilizer_z,
        np.zeros((num_stabilizers, k), dtype=np.int8),
    ])
    logical_x_part = np.hstack([logical_x_x, np.eye(k, dtype=np.int8), logical_x_z, np.zeros((k, k), dtype=np.int8)])
    logical_z_part = np.hstack([logical_z_x, np.zeros((k, k), dtype=np.int8), logical_z_z, np.eye(k, dtype=np.int8)])
    return np.vstack([stabilizer_part, logical_x_part, logical_z_part]).astype(np.int8)


def _make_x_part_invertible(tableau: npt.NDArray[np.integer], operations: list[str]) -> None:
    """Apply local Cliffords until the tableau's X part has full rank."""
    n = tableau.shape[1] // 2
    x_rank = rank(tableau[:, :n])

    while x_rank < n:
        improved = False
        for qubit in range(n):
            if x_rank == n:
                break

            best_rank = x_rank
            best_columns: tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]] | None = None
            best_operation = ""
            x_column = tableau[:, qubit].copy()
            z_column = tableau[:, qubit + n].copy()

            for new_x, new_z, operation in (
                (x_column, z_column, ""),
                (z_column, x_column, "H"),
                ((x_column + z_column) % 2, x_column, "HS"),
            ):
                tableau[:, qubit] = new_x
                new_rank = rank(tableau[:, :n])
                if new_rank > best_rank:
                    best_rank = new_rank
                    best_columns = (new_x, new_z)
                    best_operation = operation

            if best_columns is None:
                tableau[:, qubit] = x_column
                tableau[:, qubit + n] = z_column
                continue

            tableau[:, qubit] = best_columns[0]
            tableau[:, qubit + n] = best_columns[1]
            operations[qubit] = best_operation
            x_rank = best_rank
            improved = True

        if not improved:
            break


def _stabilizer_state_to_graph_state(
    tableau: npt.NDArray[np.integer],
) -> tuple[npt.NDArray[np.integer], list[str]]:
    """Convert a stabilizer state to an LC-equivalent graph state."""
    state = tableau.copy()
    n = state.shape[1] // 2
    operations = [""] * n

    # Make the X part invertible using local Clifford operations
    _make_x_part_invertible(state, operations)

    # Reduce the X part to the identity and extract the adjacency matrix
    _, x_rank, transform, _ = row_echelon(state[:, :n], full=True)
    if x_rank != n:
        msg = "X part of the tableau is not full rank, something went wrong."
        raise ValueError(msg)
    adjacency = ((transform @ state) % 2)[:, n:]

    # Remove self-loops using phase gates
    for qubit in range(n):
        if adjacency[qubit, qubit]:
            operations[qubit] = "S" + operations[qubit]
            adjacency[qubit, qubit] = 0

    if not np.array_equal(adjacency, adjacency.T):
        msg = "Extracted adjacency matrix is not symmetric, something went wrong."
        raise ValueError(msg)
    return adjacency, operations


def _satisfies_lc_determinant_constraints(solution: npt.NDArray[np.integer]) -> bool:
    """Check the single-qubit determinant constraints for an LSE solution."""
    solution = np.asarray(solution, dtype=np.uint8) % 2
    n = len(solution) // 4
    a = solution[:n]
    b = solution[n : 2 * n]
    c = solution[2 * n : 3 * n]
    d = solution[3 * n :]
    return bool(np.all(((a & d) ^ (b & c)) == 1))


def _extract_lc_operations(solution: npt.NDArray[np.integer]) -> list[str] | None:
    """Decode the local Clifford operations represented by an LSE solution."""
    n = len(solution) // 4
    operations = []
    for qubit in range(n):
        lse_matrix = (
            (int(solution[qubit]), int(solution[n + qubit])),
            (int(solution[2 * n + qubit]), int(solution[3 * n + qubit])),
        )
        operation = next(
            (
                name
                for name, action in CLIFFORD_ACTIONS.items()
                if tuple(zip(*action.matrix, strict=True)) == lse_matrix
            ),
            None,
        )
        if operation is None:
            return None
        operations.append(operation)
    return operations


def _locally_equivalent_connected_graphs(
    graph1: npt.NDArray[np.integer], graph2: npt.NDArray[np.integer]
) -> list[str] | None:
    """Check LC equivalence of two connected graph states using a linear system."""
    n = graph1.shape[0]

    # Build the graph-state LC-equivalence linear system of equations
    lse = np.zeros((n * n, 4 * n), dtype=np.uint8)
    row = 0
    for j in range(n):
        for k in range(n):
            lse[row, 2 * n : 3 * n] = graph1[j, :] & graph2[:, k]
            lse[row, k] ^= graph1[j, k]
            lse[row, 3 * n + j] ^= graph2[j, k]
            if j == k:
                lse[row, n + j] ^= 1
            row += 1

    solution_space = nullspace(lse).astype(np.uint8)
    dimension = solution_space.shape[0]
    if dimension == 0:
        return None

    if dimension > 4:
        candidates = (
            solution_space[first] ^ solution_space[second]
            for first in range(dimension)
            for second in range(first, dimension)
        )
    else:
        candidates = (
            np.bitwise_xor.reduce(solution_space[np.flatnonzero(coefficients)], axis=0, initial=0)
            for coefficients in product([0, 1], repeat=dimension)
        )

    return next(
        (
            operations
            for candidate in candidates
            if _satisfies_lc_determinant_constraints(candidate)
            if (operations := _extract_lc_operations(candidate)) is not None
        ),
        None,
    )


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
        solver.add(_exactly_one(qubit_variables.values()))
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
                        _elementwise_map(_select_column(x_source, x_column, z_column), auxiliary_x),
                        _elementwise_map(_select_column(z_source, x_column, z_column), auxiliary_z),
                    ),
                )
            )

    return variables
