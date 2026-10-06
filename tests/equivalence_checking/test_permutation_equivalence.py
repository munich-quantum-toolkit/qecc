# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the permutation equivalence function."""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest

from mqt.qecc import CSSCode, StabilizerCode, are_permutation_equivalent
from mqt.qecc.equivalence_checking import permutation_equivalence
from mqt.qecc.equivalence_checking.graphs import (
    _colored_graph_isomorphism,  # ruff: ignore[import-private-name]
    _merge_parallel_edges,  # ruff: ignore[import-private-name]
)
from mqt.qecc.equivalence_checking.invariants import (
    _preserved_k,  # ruff: ignore[import-private-name]
    _preserved_n,  # ruff: ignore[import-private-name]
)
from mqt.qecc.equivalence_checking.permutation_equivalence import (
    _binary_punctured_hull_bases,  # ruff: ignore[import-private-name]
    _circuits_binary_matroid,  # ruff: ignore[import-private-name]
    _graph_from_circuits_and_invariants,  # ruff: ignore[import-private-name]
    _graph_from_stabilizer_group_and_invariants,  # ruff: ignore[import-private-name]
    _graph_isomorphism_stabilizer_code,  # ruff: ignore[import-private-name]
    _matching_invariant_partitions,  # ruff: ignore[import-private-name]
    _preserved_number_duplicate_columns,  # ruff: ignore[import-private-name]
    _preserved_number_zero_columns,  # ruff: ignore[import-private-name]
    _preserved_ranks,  # ruff: ignore[import-private-name]
    _preserved_sector_distances,  # ruff: ignore[import-private-name]
    _punctured_hull_weight_enumerators,  # ruff: ignore[import-private-name]
    _quaternary_punctured_hull_bases,  # ruff: ignore[import-private-name]
)
from mqt.qecc.mod2 import is_in_row_space, rank

from .conftest import assert_same_row_space

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

# The stages of the pipelines that follow the cheap invariants.
STAGES = {
    "linear-dependencies": "_preserved_linear_dependencies",
    "css-hull": "_preserved_punctured_hull_weight_enumerator_css_code",
    "stabilizer-hull": "_preserved_punctured_hull_weight_enumerator_stabilizer_code",
    "bruteforce": "_bruteforce_css_codes",
    "matroid": "_matroid_css_code",
    "css-sat": "_sat_css_code",
    "graph-isomorphism": "_graph_isomorphism_stabilizer_code",
    "stabilizer-sat": "_sat_stabilizer_code",
}


@pytest.fixture
def stages(trace_stages: Callable[[ModuleType, Mapping[str, str]], list[str]]) -> list[str]:
    """Record the stages that the permutation-equivalence pipelines run."""
    return trace_stages(permutation_equivalence, STAGES)


def _apply_permutation(symplectic: np.ndarray, permutation: list[int]) -> np.ndarray:
    # p[source] = target
    n = symplectic.shape[1] // 2
    assert sorted(permutation) == list(range(n))

    transformed = np.empty_like(symplectic)
    for source, target in enumerate(permutation):
        transformed[:, target] = symplectic[:, source]
        transformed[:, target + n] = symplectic[:, source + n]
    return transformed


def _assert_maps_rowspace_stabilizer(
    code1: StabilizerCode,
    code2: StabilizerCode,
    permutation: list[int],
) -> None:
    transformed = _apply_permutation(code1.symplectic, permutation)
    assert_same_row_space(transformed, code2.symplectic)


def _assert_maps_rowspace_css(code1: CSSCode, code2: CSSCode, permutation: list[int]) -> None:
    hx_rank = rank(code1.Hx)
    hz_rank = rank(code1.Hz)
    permuted_hx = code2.Hx[:, permutation]
    permuted_hz = code2.Hz[:, permutation]

    assert rank(code2.Hx) == hx_rank
    assert rank(code2.Hz) == hz_rank
    assert all(is_in_row_space(row, permuted_hx) for row in code1.Hx)
    assert all(is_in_row_space(row, permuted_hz) for row in code1.Hz)


def _permute_generators(generators: list[str], permutation: list[int]) -> list[str]:
    # p[source] = target
    return [
        "".join(generator[permutation.index(target)] for target in range(len(permutation))) for generator in generators
    ]


def _x_checks_with_column_weight(num_rows: int, num_columns: int, weight: int) -> np.ndarray:
    # the first columns form an identity, the remaining ones are distinct columns of the given weight
    supports = [(row,) for row in range(num_rows)] + list(combinations(range(num_rows), weight))
    checks = np.zeros((num_rows, num_columns), dtype=np.int8)
    for column, support in enumerate(supports[:num_columns]):
        checks[list(support), column] = 1
    return checks


def _colored_path(node_colors: str, edge_colors: str = "") -> nx.Graph:
    graph = nx.path_graph(len(node_colors))
    nx.set_node_attributes(graph, dict(enumerate(node_colors)), "color")
    nx.set_edge_attributes(graph, dict(zip(graph.edges, edge_colors, strict=False)), "color")
    return graph


def _colored_multipath(*edge_colors: str) -> nx.MultiGraph:
    # the nodes i and i + 1 are connected by one edge for every character of the i-th argument
    graph = nx.MultiGraph()
    graph.add_nodes_from(range(len(edge_colors) + 1), color="a")
    for node, colors in enumerate(edge_colors):
        for color in colors:
            graph.add_edge(node, node + 1, color=color)
    return graph


def test_binary_hull_edge_cases() -> None:
    """Test binary punctured hulls with empty and self-orthogonal Gram matrices."""
    empty_hulls = list(_binary_punctured_hull_bases(np.zeros((0, 2), dtype=np.uint8)))
    self_orthogonal_hulls = list(_binary_punctured_hull_bases(np.array([[1, 1, 1]], dtype=np.uint8)))

    assert all(hull.shape == (0, 1) for hull in empty_hulls)
    assert all(np.array_equal(hull, np.array([[1, 1]], dtype=np.uint8)) for hull in self_orthogonal_hulls)


def test_quaternary_trivial_hull() -> None:
    """Test that a trivial punctured GF(4) hull has an empty basis."""
    matrix = np.array([[1, 0], [2, 0]], dtype=np.uint8)

    hulls = list(_quaternary_punctured_hull_bases(matrix))

    assert any(hull.shape[0] == 0 for hull in hulls)


def test_hull_dimension_cap() -> None:
    """Test that hulls above the dimension cap are only described by their dimension."""
    hull_bases = [np.eye(2, dtype=np.uint8), np.eye(3, dtype=np.uint8)]

    signatures = _punctured_hull_weight_enumerators(iter(hull_bases), lambda word: int(word.sum()), max_dimension=2)

    assert signatures == [(1, 2, 1), (-1, 3)]


def test_partition_multiplicities() -> None:
    """Test that invariant partitions with different class sizes do not match."""
    assert _matching_invariant_partitions([0, 0, 1], [0, 1, 1]) is None


def test_binary_matroid_circuits() -> None:
    """Test circuit extraction for a simple binary dependency."""
    matrix = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)

    assert _circuits_binary_matroid(matrix) == [0b111]


def test_matroid_incidence_graph() -> None:
    """Test the colored incidence graph for two matroid circuits."""
    graph = _graph_from_circuits_and_invariants(
        3,
        circuits_hx=[0b101],
        circuits_hz=[0b110],
        partition={0: [0, 1], 1: [2]},
    )

    assert set(graph.edges) == {(0, 3), (1, 4), (2, 3), (2, 4)}
    assert [graph.nodes[qubit]["color"] for qubit in range(3)] == [("qubit", 0), ("qubit", 0), ("qubit", 1)]
    assert graph.nodes[3]["color"] == ("hx",)
    assert graph.nodes[4]["color"] == ("hz",)


def test_stabilizer_group_graph() -> None:
    """Test the colored incidence graph of a stabilizer group with two elements."""
    graph = _graph_from_stabilizer_group_and_invariants(
        StabilizerCode(["XYZI"]).symplectic,
        partition={0: [0, 1, 2], 1: [3]},
    )

    # vertex 4 is the identity, vertex 5 the generator, whose Y is represented by two parallel edges
    assert set(graph.nodes) == set(range(6))
    assert set(graph.edges(data="color")) == {(0, 5, "x"), (1, 5, "x"), (1, 5, "z"), (2, 5, "z")}
    assert [graph.nodes[qubit]["color"] for qubit in range(4)] == [("qubit", 0)] * 3 + [("qubit", 1)]
    assert graph.nodes[4]["color"] == graph.nodes[5]["color"] == ("stabilizer",)


def test_isomorphism_node_colors() -> None:
    """Test that isomorphisms preserve node colors."""
    assert _colored_graph_isomorphism(_colored_path("aab"), _colored_path("baa")) == {0: 2, 1: 1, 2: 0}
    assert _colored_graph_isomorphism(_colored_path("aab"), _colored_path("aba")) is None


def test_isomorphism_edge_colors() -> None:
    """Test that isomorphisms preserve edge colors only on request."""
    graph1 = _colored_path("aaa", "xy")
    graph2 = _colored_path("aaa", "xx")

    assert _colored_graph_isomorphism(graph1, graph2) is not None
    assert _colored_graph_isomorphism(graph1, graph2, edge_colors=True) is None
    assert _colored_graph_isomorphism(graph1, _colored_path("aaa", "yx"), edge_colors=True) == {0: 2, 1: 1, 2: 0}


def test_isomorphism_parallel_edges() -> None:
    """Test that isomorphisms of multigraphs preserve the colors of parallel edges."""
    graph = _colored_multipath("xz", "x")

    assert _colored_graph_isomorphism(graph, _colored_multipath("x", "zx"), edge_colors=True) == {0: 2, 1: 1, 2: 0}
    assert _colored_graph_isomorphism(graph, _colored_multipath("xz", "z"), edge_colors=True) is None
    assert _colored_graph_isomorphism(graph, _colored_multipath("xz", "z")) is not None
    assert _colored_graph_isomorphism(graph, _colored_multipath("x", "x"), edge_colors=True) is None


def test_merge_parallel_edges() -> None:
    """Test that merged edges are colored by the sorted colors of their parallel edges."""
    merged = _merge_parallel_edges(_colored_multipath("zx", "x"))

    assert not merged.is_multigraph()
    assert dict(merged.nodes(data="color")) == {0: "a", 1: "a", 2: "a"}
    assert {(u, v): color for u, v, color in merged.edges(data="color")} == {(0, 1): ("x", "z"), (1, 2): ("x",)}


def test_isomorphism_beyond_color_refinement() -> None:
    """Test that graphs with identical color refinements are still told apart."""
    hexagon = nx.cycle_graph(6)
    triangles = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
    for graph in (hexagon, triangles):
        nx.set_node_attributes(graph, "a", "color")

    assert _colored_graph_isomorphism(hexagon, triangles) is None


# ----------------------------------------------------------------------------------------------------
# are_permutation_equivalent - cheap invariants
# ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("violated", "code1", "code2"),
    [
        pytest.param(_preserved_n, StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(4), id="n"),
        pytest.param(_preserved_k, StabilizerCode.get_trivial_code(3), StabilizerCode(["ZII"]), id="k"),
        pytest.param(
            _preserved_sector_distances, StabilizerCode(["ZZ"], distance=1), StabilizerCode(["ZZ"], distance=2), id="d"
        ),
        pytest.param(_preserved_ranks, StabilizerCode(["Z"]), StabilizerCode(["X"]), id="ranks-one-qubit"),
        pytest.param(_preserved_ranks, StabilizerCode(["XII"]), StabilizerCode(["ZII"]), id="ranks-x-vs-z"),
        pytest.param(
            _preserved_ranks, StabilizerCode(["ZI", "IZ"]), StabilizerCode(["XX", "ZZ"]), id="ranks-bell-state"
        ),
        pytest.param(
            _preserved_number_zero_columns,
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            StabilizerCode(["ZI"], z_logicals=["IZ"], x_logicals=["IX"]),
            id="zero-columns",
        ),
        pytest.param(
            _preserved_number_duplicate_columns,
            StabilizerCode(["ZZII", "IIZZ"]),
            StabilizerCode(["ZZZI", "IIIZ"]),
            id="duplicate-columns",
        ),
        pytest.param(_preserved_n, CSSCode.get_trivial_code(3), CSSCode.get_trivial_code(4), id="css-n"),
        pytest.param(
            _preserved_k,
            CSSCode.get_trivial_code(4),
            CSSCode(Hx=np.array([[1, 0, 0, 0]], dtype=np.int8)),
            id="css-k",
        ),
        pytest.param(
            _preserved_sector_distances,
            CSSCode(Hx=np.array([[1, 1, 0]], dtype=np.int8), distance=1, x_distance=2, z_distance=1),
            CSSCode(Hx=np.array([[1, 1, 0]], dtype=np.int8), distance=1, x_distance=1, z_distance=2),
            id="css-d",
        ),
        pytest.param(
            _preserved_ranks,
            CSSCode(Hx=np.array([[1, 0, 0, 0]], dtype=np.int8)),
            CSSCode(Hz=np.array([[1, 0, 0, 0]], dtype=np.int8)),
            id="css-ranks",
        ),
    ],
)
def test_cheap_invariants(
    violated: Callable[[StabilizerCode, StabilizerCode], bool],
    code1: StabilizerCode,
    code2: StabilizerCode,
    stages: list[str],
) -> None:
    """Test that a violated cheap invariant rules out equivalence before any other stage runs."""
    assert not violated(code1, code2)

    assert are_permutation_equivalent(code1, code2) is None
    assert stages == []


@pytest.mark.parametrize(
    "code", [StabilizerCode.get_trivial_code(3), CSSCode.get_trivial_code(4)], ids=["stabilizer", "css"]
)
def test_trivial(code: StabilizerCode, stages: list[str]) -> None:
    """Test that a trivial code returns the identity permutation without running any stage."""
    assert are_permutation_equivalent(code, code) == list(range(code.n))
    assert stages == []


# ----------------------------------------------------------------------------------------------------
# are_permutation_equivalent - stabilizer codes
# ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("generators1", "generators2"),
    [
        pytest.param(["Z"], ["Z"], id="one-qubit"),
        pytest.param(["ZZ"], ["ZZ", "ZZ"], id="redundant-generators"),
        pytest.param(["XXII", "IIZZ"], ["XIXI", "IZIZ"], id="interleaved"),
        pytest.param(["XZ", "IZ"], ["ZX", "ZI"], id="mixed-paulis"),
        pytest.param(["XZYI", "IYXZ"], ["ZIXY", "XZXZ"], id="y-paulis"),
    ],
)
def test_stabilizer_graph_isomorphism_positive(
    generators1: list[str], generators2: list[str], stages: list[str]
) -> None:
    """Test that the stabilizer graph-isomorphism backend finds a permutation."""
    code1 = StabilizerCode(generators1)
    code2 = StabilizerCode(generators2)

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_stabilizer(code1, code2, permutation)
    assert stages == ["stabilizer-hull", "graph-isomorphism"]


def test_stabilizer_graph_isomorphism_most_generators(stages: list[str]) -> None:
    """Test that the stabilizer graph-isomorphism backend decides codes with up to seven generators."""
    generators: list[str] = ["".join("Z" if qubit in {i, i + 1} else "I" for qubit in range(8)) for i in range(7)]
    code1 = StabilizerCode(generators)
    code2 = StabilizerCode(_permute_generators(generators, [3, 6, 1, 4, 7, 2, 5, 0]))

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_stabilizer(code1, code2, permutation)
    assert stages == ["stabilizer-hull", "graph-isomorphism"]


def test_stabilizer_graph_isomorphism_negative(stages: list[str]) -> None:
    """Test that the stabilizer graph-isomorphism backend rejects an inequivalent pair."""
    code1 = StabilizerCode(["ZIIII", "IXZXX", "IZYYY"])
    code2 = StabilizerCode(["IYYYZ", "IZXZX", "ZIIII"])

    assert are_permutation_equivalent(code1, code2) is None
    assert stages == ["stabilizer-hull", "graph-isomorphism"]


def test_stabilizer_graph_isomorphism_unrefined(stages: list[str]) -> None:
    """Test that the stabilizer graph-isomorphism backend decides larger codes without a refinement."""
    generators = [generator + "I" * 22 for generator in ("XZII", "ZXZI", "IZXZ")]
    code1 = StabilizerCode(generators)
    code2 = StabilizerCode(_permute_generators(generators, list(reversed(range(26)))))

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_stabilizer(code1, code2, permutation)
    assert stages == ["graph-isomorphism"]


@pytest.mark.parametrize(("generator1", "generator2"), [("XX", "ZZ"), ("XX", "YY"), ("ZZ", "YY")])
def test_stabilizer_graph_isomorphism_paulis(generator1: str, generator2: str) -> None:
    """Test that the stabilizer graph-isomorphism backend distinguishes X, Y, and Z operators."""
    code1 = StabilizerCode([generator1])
    code2 = StabilizerCode([generator2])
    partition: dict[tuple[int, ...], list[int]] = {(): [0, 1]}

    assert _graph_isomorphism_stabilizer_code(code1, partition, code2, partition) is None


def test_stabilizer_graph_isomorphism_partition() -> None:
    """Test that the stabilizer graph-isomorphism backend only maps qubits within their class."""
    code = StabilizerCode(["ZZ"])

    assert _graph_isomorphism_stabilizer_code(code, {0: [0], 1: [1]}, code, {0: [1], 1: [0]}) == [1, 0]


def test_stabilizer_sat_positive(stages: list[str]) -> None:
    """Test that the stabilizer SAT backend finds a permutation."""
    n = 9
    generators: list[str] = ["".join("Z" if qubit in {i, (i + 1) % n} else "I" for qubit in range(n)) for i in range(n)]
    code1 = StabilizerCode(generators)
    code2 = StabilizerCode(_permute_generators(generators, [4, 2, 0, 8, 6, 3, 1, 7, 5]))

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_stabilizer(code1, code2, permutation)
    assert stages == ["stabilizer-hull", "stabilizer-sat"]


def test_stabilizer_sat_negative(stages: list[str]) -> None:
    """Test that the stabilizer SAT backend rejects an inequivalent pair."""
    code1 = StabilizerCode([
        "ZIYIIIIII",
        "IZIZIIIII",
        "YIXIZZIIY",
        "ZIIZIIIXZ",
        "IIIIXIIXZ",
        "IIIIIZIII",
        "IIIIXIXXZ",
        "IIIIIIIZY",
    ])
    code2 = StabilizerCode([
        "ZIXIZIZXX",
        "XYXIIXIXZ",
        "IIIIIIZYY",
        "YYYIIXZII",
        "XXIIZIIYI",
        "YYYIZXIYY",
        "XXIXIIIIY",
        "ZZYIZIZYI",
    ])

    assert are_permutation_equivalent(code1, code2) is None
    assert stages == ["stabilizer-hull", "stabilizer-sat"]


def test_stabilizer_sat_unrefined(stages: list[str]) -> None:
    """Test that the stabilizer SAT backend decides codes with many generators without a refinement."""
    n = 20
    generators: list[str] = ["".join("Z" if qubit in {i, i + 1} else "I" for qubit in range(n)) for i in range(16)]
    code1 = StabilizerCode(generators)
    code2 = StabilizerCode(_permute_generators(generators, [(7 * qubit + 3) % n for qubit in range(n)]))

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_stabilizer(code1, code2, permutation)
    assert stages == ["stabilizer-sat"]


def test_stabilizer_linear_dependencies(stages: list[str]) -> None:
    """Test that different linear column dependencies rule out equivalence."""
    code1, code2 = (
        StabilizerCode(["".join("X" if entry else "I" for entry in row) for row in checks])
        for checks in (_x_checks_with_column_weight(8, 25, 2), _x_checks_with_column_weight(8, 25, 3))
    )

    assert are_permutation_equivalent(code1, code2) is None
    assert stages == ["linear-dependencies"]


@pytest.mark.parametrize(
    ("n", "expected_stages"),
    [
        pytest.param(25, ["linear-dependencies", "stabilizer-hull", "stabilizer-sat"], id="refined"),
        pytest.param(26, ["linear-dependencies", "stabilizer-sat"], id="unrefined"),
    ],
)
def test_stabilizer_linear_dependencies_preserved(n: int, expected_stages: list[str], stages: list[str]) -> None:
    """Test that larger codes with identical linear column dependencies are passed on to the later stages."""
    generators: list[str] = [
        "".join("X" if entry else "I" for entry in row) for row in _x_checks_with_column_weight(8, n, 2)
    ]
    code1 = StabilizerCode(generators)
    code2 = StabilizerCode(_permute_generators(generators, [(7 * qubit + 3) % n for qubit in range(n)]))

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_stabilizer(code1, code2, permutation)
    assert stages == expected_stages


def test_stabilizer_hull_rejection(stages: list[str]) -> None:
    """Test that different punctured-hull signatures rule out equivalence."""
    code1 = StabilizerCode(["XYZXZZXIXX", "XIZYYXIXZI", "ZZZZXYZIII"])
    code2 = StabilizerCode(["YZXZIZXYIY", "ZIXYXYXXII", "ZYIYIIZZZX"])

    assert are_permutation_equivalent(code1, code2) is None
    assert stages == ["stabilizer-hull"]


# ----------------------------------------------------------------------------------------------------
# are_permutation_equivalent - CSS codes
# ----------------------------------------------------------------------------------------------------


def test_css_linear_dependencies(stages: list[str]) -> None:
    """Test that different linear column dependencies rule out equivalence."""
    code1 = CSSCode(Hx=_x_checks_with_column_weight(10, 23, 2))
    code2 = CSSCode(Hx=_x_checks_with_column_weight(10, 23, 3))

    assert are_permutation_equivalent(code1, code2) is None
    assert stages == ["linear-dependencies"]


def test_css_linear_dependencies_preserved(stages: list[str]) -> None:
    """Test that larger codes with identical linear column dependencies are passed on to the later stages."""
    code1 = CSSCode(Hx=_x_checks_with_column_weight(10, 23, 2))
    permutation_used = [(7 * qubit + 3) % 23 for qubit in range(23)]
    code2 = CSSCode(Hx=code1.Hx[:, permutation_used])

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_css(code1, code2, permutation)
    assert stages == ["linear-dependencies", "css-hull", "css-sat"]


def test_css_hull_rejection(stages: list[str]) -> None:
    """Test that different punctured-hull signatures rule out equivalence."""
    hx1 = np.array([[0, 0, 0, 1, 1, 0, 0, 0]], dtype=np.int8)
    hz1 = np.array([[0, 1, 0, 1, 1, 1, 0, 1]], dtype=np.int8)
    hx2 = np.array([[1, 0, 1, 0, 1, 1, 0, 1]], dtype=np.int8)
    hz2 = np.array([[0, 1, 0, 1, 0, 0, 0, 0]], dtype=np.int8)

    assert are_permutation_equivalent(CSSCode(Hx=hx1, Hz=hz1), CSSCode(Hx=hx2, Hz=hz2)) is None
    assert stages == ["css-hull"]


def test_css_bruteforce_positive(stages: list[str]) -> None:
    """Test that the CSS brute-force backend finds a permutation."""
    code1 = CSSCode(
        Hx=np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8),
        Hz=np.array([[1, 1, 1, 1]], dtype=np.int8),
    )
    code2 = CSSCode(
        Hx=np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int8),
        Hz=np.array([[1, 1, 1, 1]], dtype=np.int8),
    )

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_css(code1, code2, permutation)
    assert stages == ["css-hull", "bruteforce"]


def test_css_bruteforce_negative(stages: list[str]) -> None:
    """Test that the CSS brute-force backend rejects an inequivalent pair."""
    code1 = CSSCode(
        Hx=np.array([[1, 0, 1, 0, 1], [0, 1, 1, 0, 0]], dtype=np.int8),
        Hz=np.array([[0, 1, 1, 0, 1]], dtype=np.int8),
    )
    code2 = CSSCode(
        Hx=np.array([[1, 0, 1, 1, 0], [0, 1, 1, 1, 0]], dtype=np.int8),
        Hz=np.array([[0, 0, 1, 1, 1]], dtype=np.int8),
    )

    assert are_permutation_equivalent(code1, code2) is None
    assert stages == ["css-hull", "bruteforce"]


def test_css_matroid_positive(stages: list[str]) -> None:
    """Test that the CSS matroid backend finds a permutation."""
    code1 = CSSCode.from_code_name("Steane")
    permutation_used = [3, 0, 6, 2, 5, 1, 4]
    code2 = CSSCode(
        Hx=code1.Hx[:, permutation_used],
        Hz=code1.Hz[:, permutation_used],
        distance=code1.distance,
        x_distance=code1.x_distance,
        z_distance=code1.z_distance,
    )

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_css(code1, code2, permutation)
    assert stages == ["css-hull", "matroid"]


def test_css_matroid_negative(stages: list[str]) -> None:
    """Test that the CSS matroid backend rejects an inequivalent pair with identically many circuits."""
    code1 = CSSCode(
        Hx=np.array([[0, 1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0]], dtype=np.int8),
        Hz=np.array([[0, 1, 1, 0, 0, 0, 1], [1, 0, 0, 0, 1, 1, 1]], dtype=np.int8),
    )
    code2 = CSSCode(
        Hx=np.array([[0, 0, 0, 0, 1, 1, 0], [1, 0, 1, 0, 0, 0, 1]], dtype=np.int8),
        Hz=np.array([[0, 0, 1, 0, 1, 1, 1], [1, 1, 1, 0, 1, 1, 0]], dtype=np.int8),
    )

    assert are_permutation_equivalent(code1, code2) is None
    assert stages == ["css-hull", "matroid"]


def test_css_matroid_circuits(stages: list[str]) -> None:
    """Test that the CSS matroid backend rejects a pair with differently many circuits."""
    hx1 = np.array([[1, 0, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    hz1 = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]], dtype=np.int8)
    hx2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    hz2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 1]], dtype=np.int8)

    assert are_permutation_equivalent(CSSCode(Hx=hx1, Hz=hz1), CSSCode(Hx=hx2, Hz=hz2)) is None
    assert stages == ["css-hull", "matroid"]


@pytest.mark.parametrize(
    ("n", "expected_stages"),
    [pytest.param(13, ["css-hull", "matroid"], id="matroid"), pytest.param(14, ["css-hull", "css-sat"], id="sat")],
)
def test_css_qubit_threshold(n: int, expected_stages: list[str], stages: list[str]) -> None:
    """Test that the number of qubits alone selects between the CSS matroid and SAT backends."""
    code1 = CSSCode(Hx=_x_checks_with_column_weight(5, n, 2))
    permutation_used = [(5 * qubit + 2) % n for qubit in range(n)]
    code2 = CSSCode(Hx=code1.Hx[:, permutation_used])

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_css(code1, code2, permutation)
    assert stages == expected_stages


def test_css_sat_positive(stages: list[str]) -> None:
    """Test that the CSS SAT backend finds a permutation."""
    n = 18
    hx = np.zeros((9, n), dtype=np.int8)
    for i in range(8):
        hx[i, i] = 1
    hx[8, 8] = 1
    hx[8, 9] = 1
    code1 = CSSCode(Hx=hx, Hz=None)

    permutation_used = list(reversed(range(n)))
    code2 = CSSCode(Hx=hx[:, permutation_used], Hz=None)

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_css(code1, code2, permutation)
    assert stages == ["css-hull", "css-sat"]


def test_css_sat_negative(stages: list[str]) -> None:
    """Test that the CSS SAT backend rejects an inequivalent pair with identical punctured-hull signatures."""
    hx1 = np.array(
        [
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    hz1 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1],
        ],
        dtype=np.int8,
    )
    hx2 = np.array(
        [
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    hz2 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
        ],
        dtype=np.int8,
    )

    assert are_permutation_equivalent(CSSCode(Hx=hx1, Hz=hz1), CSSCode(Hx=hx2, Hz=hz2)) is None
    assert stages == ["css-hull", "css-sat"]
