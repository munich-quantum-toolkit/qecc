# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the permutation equivalence function."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc import CSSCode, StabilizerCode, are_permutation_equivalent
from mqt.qecc.equivalence_checking.permutation_equivalence import (
    _binary_punctured_hull_bases,  # ruff: ignore[import-private-name]
    _circuits_binary_matroid,  # ruff: ignore[import-private-name]
    _graph_from_circuits_and_invariants,  # ruff: ignore[import-private-name]
    _matching_invariant_partitions,  # ruff: ignore[import-private-name]
    _quaternary_punctured_hull_bases,  # ruff: ignore[import-private-name]
    _sat_stabilizer_code,  # ruff: ignore[import-private-name]
)
from mqt.qecc.mod2 import is_in_row_space, rank

from .conftest import assert_same_row_space


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


# ----------------------------------------------------------------------------------------------------
# are_permutation_equivalent - stabilizer codes
# ----------------------------------------------------------------------------------------------------


def test_stabilizer_preserves_n() -> None:
    """Test that permutation equivalence preserves the number of physical qubits."""
    assert are_permutation_equivalent(StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(4)) is None


def test_stabilizer_preserves_k() -> None:
    """Test that permutation equivalence preserves the number of logical qubits."""
    code1 = StabilizerCode.get_trivial_code(3)
    code2 = StabilizerCode(["ZII"])

    assert are_permutation_equivalent(code1, code2) is None


def test_stabilizer_preserves_paulis() -> None:
    """Test that qubit permutations do not exchange X and Z operators."""
    code1 = StabilizerCode(["XII"])
    code2 = StabilizerCode(["ZII"])

    assert code1.n == code2.n
    assert code1.k == code2.k
    assert are_permutation_equivalent(code1, code2) is None


def test_stabilizer_trivial() -> None:
    """Test that a trivial stabilizer code returns the identity permutation."""
    assert are_permutation_equivalent(StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(3)) == [
        0,
        1,
        2,
    ]


def test_stabilizer_one_qubit() -> None:
    """Test that a one-qubit code is permutation-equivalent to itself."""
    assert are_permutation_equivalent(StabilizerCode(["Z"]), StabilizerCode(["Z"])) == [0]


def test_stabilizer_one_qubit_mismatch() -> None:
    """Test that a one-qubit Pauli mismatch is rejected."""
    assert are_permutation_equivalent(StabilizerCode(["Z"]), StabilizerCode(["X"])) is None


def test_stabilizer_redundant_generators() -> None:
    """Test that redundant stabilizer generators do not affect permutation equivalence."""
    code = StabilizerCode(["ZZ"])
    code_with_redundancy = StabilizerCode(["ZZ", "ZZ"])

    permutation = are_permutation_equivalent(code, code_with_redundancy)

    assert permutation is not None
    _assert_maps_rowspace_stabilizer(code, code_with_redundancy, permutation)


def test_stabilizer_positive() -> None:
    """Test that an interleaving qubit permutation is recognized."""
    code1 = StabilizerCode(["XXII", "IIZZ"])
    code2 = StabilizerCode(["XIXI", "IZIZ"])

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_stabilizer(code1, code2, permutation)


def test_stabilizer_mixed_paulis() -> None:
    """Test that a permutation consistently maps mixed X and Z generators."""
    code1 = StabilizerCode(["XZ", "IZ"])
    code2 = StabilizerCode(["ZX", "ZI"])

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_stabilizer(code1, code2, permutation)


def test_stabilizer_sat_positive() -> None:
    """Test that the stabilizer SAT backend finds a permutation."""
    n = 6
    generators = ["ZZIIII", "IZZIII", "IIZZII", "IIIZZI", "IIIIZZ", "ZIIIIZ"]
    code1 = StabilizerCode(generators)

    permutation_used = [1, 2, 3, 4, 5, 0]
    permuted_generators = ["".join(g[permutation_used.index(i)] for i in range(n)) for g in generators]
    code2 = StabilizerCode(permuted_generators)

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_stabilizer(code1, code2, permutation)


def test_stabilizer_sat_negative() -> None:
    """Test that the stabilizer SAT backend rejects different support weights."""
    code1 = StabilizerCode(["ZI"])
    code2 = StabilizerCode(["ZZ"])
    partition: dict[tuple[int, ...], list[int]] = {(): [0, 1]}

    assert _sat_stabilizer_code(code1, partition, code2, partition) is None


@pytest.mark.parametrize(
    ("code1", "code2"),
    [
        pytest.param(
            StabilizerCode(["ZI", "IZ"]),
            StabilizerCode(["XX", "ZZ"]),
            id="product-vs-bell-state",
        ),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            StabilizerCode(["ZI"], z_logicals=["IZ"], x_logicals=["IX"]),
            id="weight-two-vs-weight-one-stabilizer",
        ),
    ],
)
def test_stabilizer_negative(
    code1: StabilizerCode,
    code2: StabilizerCode,
) -> None:
    """Test that known inequivalent stabilizer-code pairs are rejected."""
    assert are_permutation_equivalent(code1, code2) is None


def test_stabilizer_hull_rejection() -> None:
    """Test that different punctured-hull signatures rule out equivalence."""
    code1 = StabilizerCode(["XYZXZZXIXX", "XIZYYXIXZI", "ZZZZXYZIII"])
    code2 = StabilizerCode(["YZXZIZXYIY", "ZIXYXYXXII", "ZYIYIIZZZX"])

    assert code1.n == code2.n == 10
    assert code1.k == code2.k
    assert code1.distance == code2.distance

    assert are_permutation_equivalent(code1, code2) is None


# ----------------------------------------------------------------------------------------------------
# are_permutation_equivalent - CSS codes
# ----------------------------------------------------------------------------------------------------


def test_css_preserves_n() -> None:
    """Test that CSS permutation equivalence preserves the number of physical qubits."""
    assert are_permutation_equivalent(CSSCode.get_trivial_code(3), CSSCode.get_trivial_code(4)) is None


def test_css_preserves_k() -> None:
    """Test that CSS permutation equivalence preserves the number of logical qubits."""
    code1 = CSSCode.get_trivial_code(4)
    code2 = CSSCode(Hx=np.array([[1, 0, 0, 0]], dtype=np.int8))

    assert are_permutation_equivalent(code1, code2) is None


def test_css_preserves_ranks() -> None:
    """Test that CSS permutation equivalence preserves the X and Z ranks."""
    code1 = CSSCode(Hx=np.array([[1, 0, 0, 0]], dtype=np.int8))
    code2 = CSSCode(Hz=np.array([[1, 0, 0, 0]], dtype=np.int8))

    assert code1.n == code2.n
    assert code1.k == code2.k
    assert are_permutation_equivalent(code1, code2) is None


def test_css_linear_dependencies() -> None:
    """Test that different linear column dependencies rule out equivalence."""
    hx1 = np.array(
        [
            [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        ],
        dtype=np.int8,
    )
    hz1 = np.array(
        [
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.int8,
    )
    hx2 = np.array(
        [
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.int8,
    )
    hz2 = np.array(
        [
            [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
        ],
        dtype=np.int8,
    )

    code1 = CSSCode(Hx=hx1, Hz=hz1)
    code2 = CSSCode(Hx=hx2, Hz=hz2)

    assert code1.n == code2.n == 20
    assert code1.k == code2.k
    assert code1.x_distance == code2.x_distance
    assert code1.z_distance == code2.z_distance

    assert are_permutation_equivalent(code1, code2) is None


def test_css_trivial() -> None:
    """Test that a trivial CSS code returns the identity permutation."""
    code = CSSCode.get_trivial_code(4)

    assert rank(code.Hx) == 0
    assert rank(code.Hz) == 0
    assert are_permutation_equivalent(code, code) == [0, 1, 2, 3]


def test_css_bruteforce_positive() -> None:
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


def test_css_bruteforce_negative() -> None:
    """Test that the CSS brute-force backend rejects an inequivalent pair."""
    hx1 = np.array([[1, 0, 0, 0, 1]], dtype=np.int8)
    hz1 = np.array([[0, 1, 1, 1, 0]], dtype=np.int8)
    hx2 = np.array([[1, 1, 1, 0, 0]], dtype=np.int8)
    hz2 = np.array([[0, 1, 1, 0, 0]], dtype=np.int8)

    code1 = CSSCode(Hx=hx1, Hz=hz1)
    code2 = CSSCode(Hx=hx2, Hz=hz2)

    assert code1.n == code2.n == 5
    assert code1.k == code2.k
    assert code1.x_distance == code2.x_distance
    assert code1.z_distance == code2.z_distance

    assert are_permutation_equivalent(code1, code2) is None


def test_css_matroid_positive() -> None:
    """Test that the CSS matroid backend finds a permutation."""
    code1 = CSSCode.from_code_name("Steane")
    permutation_used = list(reversed(range(code1.n)))
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


def test_css_matroid_negative() -> None:
    """Test that the CSS matroid backend rejects an inequivalent pair."""
    hx1 = np.array([[0, 0, 0, 1, 1, 0, 0, 0]], dtype=np.int8)
    hz1 = np.array([[0, 1, 0, 1, 1, 1, 0, 1]], dtype=np.int8)
    hx2 = np.array([[1, 0, 1, 0, 1, 1, 0, 1]], dtype=np.int8)
    hz2 = np.array([[0, 1, 0, 1, 0, 0, 0, 0]], dtype=np.int8)

    code1 = CSSCode(Hx=hx1, Hz=hz1)
    code2 = CSSCode(Hx=hx2, Hz=hz2)

    assert code1.n == code2.n == 8
    assert code1.k == code2.k
    assert code1.x_distance == code2.x_distance
    assert code1.z_distance == code2.z_distance

    assert are_permutation_equivalent(code1, code2) is None


def test_css_matroid_circuits() -> None:
    """Test that different matroid circuits rule out equivalence."""
    hx1 = np.array([[1, 0, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    hz1 = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]], dtype=np.int8)
    hx2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
    hz2 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 1, 0, 1]], dtype=np.int8)

    code1 = CSSCode(Hx=hx1, Hz=hz1)
    code2 = CSSCode(Hx=hx2, Hz=hz2)

    assert code1.n == code2.n == 10
    assert code1.k == code2.k
    assert code1.x_distance == code2.x_distance
    assert code1.z_distance == code2.z_distance

    assert are_permutation_equivalent(code1, code2) is None


def test_css_sat_positive() -> None:
    """Test that the CSS SAT backend finds a permutation."""
    n = 18
    hx = np.zeros((10, n), dtype=np.int8)
    for i in range(9):
        hx[i, i] = 1
    hx[9, 9] = 1
    hx[9, 10] = 1
    code1 = CSSCode(Hx=hx, Hz=None)

    permutation_used = list(reversed(range(n)))
    code2 = CSSCode(Hx=hx[:, permutation_used], Hz=None)

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_css(code1, code2, permutation)


def test_css_sat_negative() -> None:
    """Test that the CSS SAT backend rejects an inequivalent pair."""
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

    code1 = CSSCode(Hx=hx1, Hz=hz1)
    code2 = CSSCode(Hx=hx2, Hz=hz2)

    assert code1.n == code2.n == 18
    assert code1.k == code2.k
    assert code1.x_distance == code2.x_distance
    assert code1.z_distance == code2.z_distance
    assert rank(hx1) + rank(hz1) >= 10

    assert are_permutation_equivalent(code1, code2) is None
