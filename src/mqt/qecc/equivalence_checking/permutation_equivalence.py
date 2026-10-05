# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Functions for deciding permutation equivalence."""

from __future__ import annotations

import operator
from collections import Counter, defaultdict
from itertools import permutations
from typing import TYPE_CHECKING, TypeVar

import networkx as nx
import numpy as np
import z3

from ..codes.core.css_code import CSSCode
from ..mod2 import nullspace, rank, row_basis, row_span
from ..mod4 import matmul_gf2_gf4
from ..mod4 import row_basis as gf4_row_basis
from .utils import (
    _colored_graph_isomorphism,
    _elementwise_map,
    _encode_row_operations,
    _exactly_one,
    _preserved_k,
    _preserved_n,
    _reduce_stabilizer_generators,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence

    import numpy.typing as npt

    from ..codes.core.stabilizer_code import StabilizerCode


# Each equivalence check runs up to three stages with the following
# empirically determined parameter thresholds:
# it tries to refute equivalence with invariants,
# refines the qubits into classes that any permutation has to preserve,
# and finally decides equivalence with a complete procedure.

# Refute
LINEAR_DEPENDENCY_MIN_QUBITS_CSS = 23
LINEAR_DEPENDENCY_MIN_GENERATORS_CSS = 10
LINEAR_DEPENDENCY_MIN_QUBITS_STB = 25
LINEAR_DEPENDENCY_MIN_GENERATORS_STB = 8

# Refine
PUNCTURED_HULL_MAX_DIMENSION_CSS = 14
PUNCTURED_HULL_MAX_QUBITS_STB = 25
PUNCTURED_HULL_MAX_GENERATORS_STB = 14

# Decide
BRUTEFORCE_MAX_QUBITS_CSS = 5
MATROID_MAX_QUBITS_CSS = 13
GRAPH_ISOMORPHISM_MAX_GENERATORS_STB = 7

InvariantT = TypeVar("InvariantT", bound="Hashable")
_CSSHullSignature = tuple[tuple[int, ...], tuple[int, ...]]


def are_permutation_equivalent(code1: StabilizerCode | CSSCode, code2: StabilizerCode | CSSCode) -> list[int] | None:
    """Check whether two stabilizer codes are permutation equivalent.

    Phase information is not considered.

    Args:
        code1: First stabilizer code.
        code2: Second stabilizer code.

    Returns:
        The qubit permutation mapping ``code1`` to ``code2``, or ``None`` if no
        such permutation exists. Entry ``i`` gives the target of qubit ``i``.
    """
    code1 = _reduce_stabilizer_generators(code1)
    code2 = _reduce_stabilizer_generators(code2)

    # Refute
    cheap_invariants = (
        _preserved_n,
        _preserved_k,
        _preserved_d,
        _preserved_ranks,
        _preserved_number_zero_columns,
        _preserved_number_duplicate_columns,
    )

    if not all(invariant(code1, code2) for invariant in cheap_invariants):
        return None

    if code1.n == 0 or code1.symplectic.shape[0] == 0:
        return list(range(code1.n))

    if isinstance(code1, CSSCode) and isinstance(code2, CSSCode):
        return _permutation_eq_css_codes(code1, code2)
    return _permutation_eq_stabilizer_codes(code1, code2)


def _permutation_eq_css_codes(code1: CSSCode, code2: CSSCode) -> list[int] | None:
    """Check whether two non-trivial CSS codes with matching cheap invariants are permutation equivalent.

    Refutes equivalence of larger codes with a linear-dependency invariant, refines the qubits into classes
    using a qubit signature, and decides equivalence with a brute-force, matroid, or SAT-based algorithm
    depending on the size of the codes.

    Note: Due to a different graph isomorphism tool (pynauty vs. networkx), the empirical threshold between the matroid and SAT-based algorithms differs between the paper and MQT version.
    """
    n = code1.n
    r = code1.Hx.shape[0] + code1.Hz.shape[0]

    # Refute
    if (
        n >= LINEAR_DEPENDENCY_MIN_QUBITS_CSS
        and r >= LINEAR_DEPENDENCY_MIN_GENERATORS_CSS
        and not _preserved_linear_dependencies(code1, code2)
    ):
        return None

    # Refine
    partitions = _preserved_punctured_hull_weight_enumerator_css_code(code1, code2)
    if partitions is None:
        return None
    partition1, partition2 = partitions

    # Decide
    if n <= BRUTEFORCE_MAX_QUBITS_CSS:
        return _bruteforce_css(code1, code2)
    if n <= MATROID_MAX_QUBITS_CSS:
        return _matroid_css_code(code1, partition1, code2, partition2)
    return _sat_css_code(code1, partition1, code2, partition2)


def _permutation_eq_stabilizer_codes(code1: StabilizerCode, code2: StabilizerCode) -> list[int] | None:
    """Check whether two non-trivial stabilizer codes with matching cheap invariants are permutation equivalent.

    Refutes equivalence of larger codes with a linear-dependency invariant, refines the qubits of smaller codes
    into classes using a qubit signature, and decides equivalence with a graph-isomorphism or SAT-based algorithm
    depending on the number of stabilizer generators.
    """
    n = code1.n
    r = code1.symplectic.shape[0]

    # Refute
    if (
        n >= LINEAR_DEPENDENCY_MIN_QUBITS_STB
        and r >= LINEAR_DEPENDENCY_MIN_GENERATORS_STB
        and not _preserved_linear_dependencies(code1, code2)
    ):
        return None

    # Refine
    partition1: dict[tuple[int, ...], list[int]] = {(): list(range(n))}
    partition2 = partition1
    if n <= PUNCTURED_HULL_MAX_QUBITS_STB and r <= PUNCTURED_HULL_MAX_GENERATORS_STB:
        refined_partitions = _preserved_punctured_hull_weight_enumerator_stabilizer_code(code1, code2)
        if refined_partitions is None:
            return None
        partition1, partition2 = refined_partitions

    # Decide
    if r <= GRAPH_ISOMORPHISM_MAX_GENERATORS_STB:
        return _graph_isomorphism_stabilizer_code(code1, partition1, code2, partition2)
    return _sat_stabilizer_code(code1, partition1, code2, partition2)


# ----------------------------------------------------------------------------------------------------
#   Invariants
# ----------------------------------------------------------------------------------------------------


def _preserved_d(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check the code-distance invariant for permutation equivalence."""
    if isinstance(c1, CSSCode) and isinstance(c2, CSSCode):
        return c1.x_distance == c2.x_distance and c1.z_distance == c2.z_distance
    return c1.distance == c2.distance


def _preserved_ranks(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check the X-part and Z-part rank invariant for permutation equivalence."""

    def _ranks(c: StabilizerCode | CSSCode) -> tuple[int, int]:
        return rank(c.symplectic[:, : c.n]), rank(c.symplectic[:, c.n :])

    return _ranks(c1) == _ranks(c2)


def _preserved_number_zero_columns(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check the zero-column-count invariant for permutation equivalence."""
    return int(np.count_nonzero(np.all(c1.symplectic == 0, axis=0))) == int(
        np.count_nonzero(np.all(c2.symplectic == 0, axis=0))
    )


def _preserved_number_duplicate_columns(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check the duplicate-column-count invariant for permutation equivalence."""

    def _duplicate_column(m: npt.NDArray[np.integer]) -> list[int]:
        columns = [tuple(m[:, j].tolist()) for j in range(m.shape[1])]
        counts = Counter(columns)
        return sorted(counts.values())

    return _duplicate_column(c1.symplectic) == _duplicate_column(c2.symplectic)


def _preserved_linear_dependencies(c1: StabilizerCode | CSSCode, c2: StabilizerCode | CSSCode) -> bool:
    """Check low-order column-rank invariants for permutation equivalence."""

    def _linear_dependencies(c: StabilizerCode | CSSCode) -> tuple[list[int], list[int], list[int]]:
        n = c.n
        m = c.symplectic

        one_columns = [rank(np.column_stack([m[:, q], m[:, q + n]])) for q in range(n)]

        two_columns = [
            rank(np.column_stack([m[:, i], m[:, i + n], m[:, j], m[:, j + n]]))
            for i in range(n)
            for j in range(i + 1, n)
        ]
        three_columns = [
            rank(np.column_stack([m[:, i], m[:, i + n], m[:, j], m[:, j + n], m[:, k], m[:, k + n]]))
            for i in range(n)
            for j in range(i + 1, n)
            for k in range(j + 1, n)
        ]

        return (sorted(one_columns), sorted(two_columns), sorted(three_columns))

    return _linear_dependencies(c1) == _linear_dependencies(c2)


# ----------------------------------------------------------------------------------------------------
#   Signatures
# ----------------------------------------------------------------------------------------------------


def _preserved_punctured_hull_weight_enumerator_css_code(
    c1: CSSCode, c2: CSSCode
) -> tuple[dict[_CSSHullSignature, list[int]], dict[_CSSHullSignature, list[int]]] | None:
    """Partition CSS-code qubits using punctured-hull weight enumerators.

    This invariant is based on Sendrier's support splitting algorithm:
    https://doi.org/10.1109/18.850662.
    """

    def _signatures(parity_check: npt.NDArray[np.integer]) -> list[tuple[int, ...]]:
        return _punctured_hull_weight_enumerators(
            _binary_punctured_hull_bases(nullspace(parity_check)),
            lambda word: int(word.sum()),
            max_dimension=PUNCTURED_HULL_MAX_DIMENSION_CSS,
        )

    signatures_c1 = list(zip(_signatures(c1.Hx), _signatures(c1.Hz), strict=True))
    signatures_c2 = list(zip(_signatures(c2.Hx), _signatures(c2.Hz), strict=True))

    return _matching_invariant_partitions(signatures_c1, signatures_c2)


def _binary_punctured_hull_bases(
    matrix: npt.NDArray[np.integer],
) -> Iterator[npt.NDArray[np.integer]]:
    """Yield a binary hull basis for every punctured column of a matrix."""
    for column in range(matrix.shape[1]):
        punctured = np.delete(matrix, column, axis=1)
        gram = (punctured @ punctured.T) & 1
        yield row_basis((nullspace(gram) @ punctured) & 1).astype(np.uint8)


def _preserved_punctured_hull_weight_enumerator_stabilizer_code(
    c1: StabilizerCode, c2: StabilizerCode
) -> tuple[dict[tuple[int, ...], list[int]], dict[tuple[int, ...], list[int]]] | None:
    """Partition stabilizer-code qubits using punctured-hull weight enumerators.

    This invariant is based on Sendrier's support splitting algorithm:
    https://doi.org/10.1109/18.850662.
    """
    gf4_tableau_c1 = c1.symplectic[:, : c1.n] + 2 * c1.symplectic[:, c1.n :]
    gf4_tableau_c2 = c2.symplectic[:, : c2.n] + 2 * c2.symplectic[:, c2.n :]

    signatures_c1 = _punctured_hull_weight_enumerators(
        _quaternary_punctured_hull_bases(gf4_tableau_c1), lambda word: int(np.count_nonzero(word))
    )
    signatures_c2 = _punctured_hull_weight_enumerators(
        _quaternary_punctured_hull_bases(gf4_tableau_c2), lambda word: int(np.count_nonzero(word))
    )
    return _matching_invariant_partitions(signatures_c1, signatures_c2)


def _quaternary_punctured_hull_bases(
    matrix: npt.NDArray[np.integer],
) -> Iterator[npt.NDArray[np.integer]]:
    """Yield an additive GF(4) hull basis for every punctured column of a matrix."""
    num_rows, num_columns = matrix.shape
    contributions = np.zeros((num_columns, num_rows, num_rows), dtype=np.uint8)
    full_gram = np.zeros((num_rows, num_rows), dtype=np.uint8)

    for column in range(num_columns):
        x_column = matrix[:, column] & 1
        z_column = matrix[:, column] >> 1
        contributions[column] = (x_column[:, None] & z_column[None, :]) ^ (z_column[:, None] & x_column[None, :])
        full_gram ^= contributions[column]

    for column in range(num_columns):
        punctured = np.delete(matrix, column, axis=1)
        gram = full_gram ^ contributions[column]
        yield gf4_row_basis(matmul_gf2_gf4(nullspace(gram.T), punctured))


def _punctured_hull_weight_enumerators(
    hull_bases: Iterator[npt.NDArray[np.integer]],
    weight: Callable[[npt.NDArray[np.integer]], int],
    *,
    max_dimension: int | None = None,
) -> list[tuple[int, ...]]:
    """Compute a weight enumerator for each punctured hull basis.

    Hulls exceeding ``max_dimension`` are too large to enumerate and only described by their dimension.
    """
    return [
        (-1, basis.shape[0])
        if max_dimension is not None and basis.shape[0] > max_dimension
        else tuple(_gray_code_weight_enumerator(basis, weight))
        for basis in hull_bases
    ]


def _gray_code_weight_enumerator(
    basis: npt.NDArray[np.integer], weight: Callable[[npt.NDArray[np.integer]], int]
) -> list[int]:
    """Enumerate the weights of a binary row span in Gray-code order."""
    rows, columns = basis.shape
    enumerator = [1] + [0] * columns
    word = np.zeros(columns, dtype=basis.dtype)
    previous_gray = 0

    for value in range(1, 1 << rows):
        gray = value ^ (value >> 1)
        changed = gray ^ previous_gray
        word ^= basis[changed.bit_length() - 1]
        enumerator[weight(word)] += 1
        previous_gray = gray

    return enumerator


def _matching_invariant_partitions(
    invariants1: Sequence[InvariantT], invariants2: Sequence[InvariantT]
) -> tuple[dict[InvariantT, list[int]], dict[InvariantT, list[int]]] | None:
    """Build and compare column partitions induced by two invariant sequences."""
    partition1 = _partition_columns_by_invariants(invariants1)
    partition2 = _partition_columns_by_invariants(invariants2)

    if partition1.keys() != partition2.keys():
        return None
    if any(len(partition1[invariant]) != len(partition2[invariant]) for invariant in partition1):
        return None
    return partition1, partition2


def _partition_columns_by_invariants(invariants: Sequence[InvariantT]) -> dict[InvariantT, list[int]]:
    """Partition column indices by invariant value."""
    partition: defaultdict[InvariantT, list[int]] = defaultdict(list)
    for index, invariant in enumerate(invariants):
        partition[invariant].append(index)
    return dict(sorted(partition.items(), key=operator.itemgetter(0)))


# ----------------------------------------------------------------------------------------------------
#   Decision procedures
# ----------------------------------------------------------------------------------------------------

# --------------------------------------------------
#   Brute force
# --------------------------------------------------


def _bruteforce_css(c1: CSSCode, c2: CSSCode) -> list[int] | None:
    """Brute-force check for permutation equivalence of two CSS codes."""
    hx_rank = c1.Hx.shape[0]
    hz_rank = c1.Hz.shape[0]

    for perm in permutations(range(c1.n)):
        if hx_rank and hx_rank != rank(np.vstack([c1.Hx, c2.Hx[:, perm]])):
            continue
        if hz_rank and hz_rank != rank(np.vstack([c1.Hz, c2.Hz[:, perm]])):
            continue
        return list(perm)

    return None


# --------------------------------------------------
#   Matroid isomorphism
# --------------------------------------------------


def _matroid_css_code(
    c1: CSSCode,
    partition1: Mapping[InvariantT, Sequence[int]],
    c2: CSSCode,
    partition2: Mapping[InvariantT, Sequence[int]],
) -> list[int] | None:
    """Check permutation equivalence of CSS codes through an isomorphism of the matroid-circuit graph encodings."""
    n = c1.n

    circuits_c1_hx = _circuits_binary_matroid(c1.Hx)
    circuits_c1_hz = _circuits_binary_matroid(c1.Hz)

    graph_c1 = _graph_from_circuits_and_invariants(n, circuits_c1_hx, circuits_c1_hz, partition1)

    len_circuits_c1_hx = len(circuits_c1_hx)
    len_circuits_c1_hz = len(circuits_c1_hz)

    # delete the potentially large lists of circuits to save memory
    del circuits_c1_hx
    del circuits_c1_hz

    circuits_c2_hx = _circuits_binary_matroid(c2.Hx)
    if len_circuits_c1_hx != len(circuits_c2_hx):
        return None

    circuits_c2_hz = _circuits_binary_matroid(c2.Hz)
    if len_circuits_c1_hz != len(circuits_c2_hz):
        return None

    graph_c2 = _graph_from_circuits_and_invariants(n, circuits_c2_hx, circuits_c2_hz, partition2)

    del circuits_c2_hx
    del circuits_c2_hz

    mapping = _colored_graph_isomorphism(graph_c1, graph_c2)

    if mapping is None:
        return None

    return [mapping[q] for q in range(n)]


def _circuits_binary_matroid(matrix: npt.NDArray[np.integer]) -> list[int]:
    """Return the circuits of a binary matroid as support bit masks."""
    kernel = nullspace(matrix)
    kernel_rows = kernel.shape[0]
    row_supports = [sum(1 << int(column) for column in np.flatnonzero(row)) for row in kernel]
    circuits_by_size: list[list[int]] = [[] for _ in range(matrix.shape[1] + 1)]

    support = 0
    previous_gray = 0
    for mask in range(1, 1 << kernel_rows):
        gray = mask ^ (mask >> 1)
        changed = gray ^ previous_gray
        support ^= row_supports[changed.bit_length() - 1]
        previous_gray = gray

        support_size = support.bit_count()
        if any(
            (circuit & support) == circuit for size in range(1, support_size + 1) for circuit in circuits_by_size[size]
        ):
            continue

        for size in range(support_size + 1, len(circuits_by_size)):
            if circuits_by_size[size]:
                circuits_by_size[size] = [
                    circuit for circuit in circuits_by_size[size] if (support & circuit) != support
                ]

        circuits_by_size[support_size].append(support)

    return [circuit for circuits in circuits_by_size for circuit in sorted(circuits)]


def _graph_from_circuits_and_invariants(
    n: int,
    circuits_hx: list[int],
    circuits_hz: list[int],
    partition: Mapping[InvariantT, Sequence[int]],
) -> nx.Graph:
    """Build a colored incidence graph for two binary matroids."""
    n_hx = len(circuits_hx)
    n_hz = len(circuits_hz)
    hx_offset = n
    hz_offset = n + n_hx

    graph = nx.Graph()
    graph.add_nodes_from(range(n + n_hx + n_hz))
    _color_qubits(graph, partition)

    for circuits, offset, kind in ((circuits_hx, hx_offset, "hx"), (circuits_hz, hz_offset, "hz")):
        for index, circuit in enumerate(circuits):
            circuit_vertex = offset + index
            graph.nodes[circuit_vertex]["color"] = (kind,)
            remaining = circuit
            while remaining:
                bit = remaining & -remaining
                graph.add_edge(bit.bit_length() - 1, circuit_vertex)
                remaining ^= bit

    return graph


# --------------------------------------------------
#   Graph isomorphism
# --------------------------------------------------


def _graph_isomorphism_stabilizer_code(
    c1: StabilizerCode,
    partition1: Mapping[InvariantT, Sequence[int]],
    c2: StabilizerCode,
    partition2: Mapping[InvariantT, Sequence[int]],
) -> list[int] | None:
    """Check permutation equivalence of stabilizer codes through an isomorphism of the full-group graph encodings."""
    mapping = _colored_graph_isomorphism(
        _graph_from_stabilizer_group_and_invariants(c1.symplectic, partition1),
        _graph_from_stabilizer_group_and_invariants(c2.symplectic, partition2),
        edge_colors=True,
    )

    if mapping is None:
        return None

    return [mapping[q] for q in range(c1.n)]


def _graph_from_stabilizer_group_and_invariants(
    symplectic: npt.NDArray[np.integer],
    partition: Mapping[InvariantT, Sequence[int]],
) -> nx.MultiGraph:
    """Build a colored incidence multigraph of the qubits and the stabilizer group elements."""
    n = symplectic.shape[1] // 2
    group = row_span(symplectic)

    graph = nx.MultiGraph()
    graph.add_nodes_from(range(n))
    _color_qubits(graph, partition)
    graph.add_nodes_from(range(n, n + group.shape[0]), color=("stabilizer",))

    for color, part in (("x", group[:, :n]), ("z", group[:, n:])):
        elements, qubits = np.nonzero(part)
        graph.add_edges_from(
            ((qubit, n + element, color) for element, qubit in zip(elements.tolist(), qubits.tolist(), strict=True)),
            color=color,
        )

    return graph


def _color_qubits(graph: nx.Graph, partition: Mapping[InvariantT, Sequence[int]]) -> None:
    """Color the qubit vertices of a graph by their class in an invariant partition."""
    for color, (_, columns) in enumerate(sorted(partition.items())):
        for column in columns:
            graph.nodes[column]["color"] = ("qubit", color)


# --------------------------------------------------
#   SAT
# --------------------------------------------------


def _sat_css_code(
    c1: CSSCode,
    partition1: Mapping[InvariantT, Sequence[int]],
    c2: CSSCode,
    partition2: Mapping[InvariantT, Sequence[int]],
) -> list[int] | None:
    """Check permutation equivalence of CSS codes using a SAT encoding."""
    solver = z3.Solver()

    n = c1.n
    rx = c1.Hx.shape[0]
    rz = c1.Hz.shape[0]

    auxiliary_x = [z3.Bool(f"aux_x_{row}_{column}") for row in range(rx) for column in range(n)]
    auxiliary_z = [z3.Bool(f"aux_z_{row}_{column}") for row in range(rz) for column in range(n)]
    permutation_variables = _encode_permutation(solver, n, partition1, partition2)
    _encode_permutation_implications(solver, permutation_variables, c1.Hx, c1.Hz, auxiliary_x, auxiliary_z)
    _encode_row_operations(solver, auxiliary_x, c2.Hx, variable_prefix="r_x")
    _encode_row_operations(solver, auxiliary_z, c2.Hz, variable_prefix="r_z")

    if solver.check() != z3.sat:
        return None

    return _extract_permutation(solver.model(), n, permutation_variables)


def _sat_stabilizer_code(
    c1: StabilizerCode,
    partition1: dict[tuple[int, ...], list[int]],
    c2: StabilizerCode,
    partition2: dict[tuple[int, ...], list[int]],
) -> list[int] | None:
    """Check permutation equivalence of stabilizer codes using a SAT encoding."""
    solver = z3.Solver()

    r, n = c1.symplectic.shape[0], c1.n

    auxiliary_x = [z3.Bool(f"aux_x_{row}_{column}") for row in range(r) for column in range(n)]
    auxiliary_z = [z3.Bool(f"aux_z_{row}_{column}") for row in range(r) for column in range(n)]
    permutation_variables = _encode_permutation(solver, n, partition1, partition2)
    _encode_permutation_implications(
        solver,
        permutation_variables,
        c1.symplectic[:, :n],
        c1.symplectic[:, n:],
        auxiliary_x,
        auxiliary_z,
    )

    auxiliary_tableau = [
        auxiliary_x[row * n + column] if column < n else auxiliary_z[row * n + column - n]
        for row in range(r)
        for column in range(2 * n)
    ]
    _encode_row_operations(solver, auxiliary_tableau, c2.symplectic, variable_prefix="r")

    if solver.check() != z3.sat:
        return None

    return _extract_permutation(solver.model(), n, permutation_variables)


def _encode_permutation(
    solver: z3.Solver,
    n: int,
    partition1: Mapping[InvariantT, Sequence[int]],
    partition2: Mapping[InvariantT, Sequence[int]],
) -> dict[tuple[int, int], z3.BoolRef]:
    """Create a partition-preserving permutation matrix encoding."""
    variables = {
        (source, target): z3.Bool(f"p_{source}_{target}")
        for invariant, source_columns in partition1.items()
        for source in source_columns
        for target in partition2[invariant]
    }

    for source in range(n):
        solver.add(_exactly_one(variable for (src, _), variable in variables.items() if src == source))
    for target in range(n):
        solver.add(_exactly_one(variable for (_, tgt), variable in variables.items() if tgt == target))

    return variables


def _encode_permutation_implications(
    solver: z3.Solver,
    permutation_variables: Mapping[tuple[int, int], z3.BoolRef],
    source_x: npt.NDArray[np.integer],
    source_z: npt.NDArray[np.integer],
    target_x: Sequence[z3.BoolRef],
    target_z: Sequence[z3.BoolRef],
) -> None:
    """Constrain each selected permutation to map corresponding X and Z columns."""
    for (source, target), permutation_variable in permutation_variables.items():
        rows_x, columns_x = source_x.shape
        rows_z, columns_z = source_z.shape
        target_x_column = [target_x[row * columns_x + target] for row in range(rows_x)]
        target_z_column = [target_z[row * columns_z + target] for row in range(rows_z)]
        solver.add(
            z3.Implies(
                permutation_variable,
                z3.And(
                    _elementwise_map(source_x[:, source], target_x_column),
                    _elementwise_map(source_z[:, source], target_z_column),
                ),
            )
        )


def _extract_permutation(model: z3.ModelRef, n: int, variables: Mapping[tuple[int, int], z3.BoolRef]) -> list[int]:
    """Extract a source-to-target permutation from a satisfying model."""
    return [
        next(
            target
            for (src, target), variable in variables.items()
            if src == source and z3.is_true(model.eval(variable, model_completion=True))
        )
        for source in range(n)
    ]
