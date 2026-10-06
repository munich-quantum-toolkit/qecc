# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the local clifford equivalence functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.qecc import StabilizerCode, are_local_clifford_equivalent, is_local_clifford_equivalent_to_css
from mqt.qecc.codes import RotatedSurfaceCode
from mqt.qecc.equivalence_checking import local_clifford_equivalence
from mqt.qecc.equivalence_checking.invariants import (
    _preserved_distance,  # ruff: ignore[import-private-name]
    _preserved_k,  # ruff: ignore[import-private-name]
    _preserved_n,  # ruff: ignore[import-private-name]
)
from mqt.qecc.equivalence_checking.local_clifford_equivalence import (
    CLIFFORD_ACTIONS,
    LOCAL_CLIFFORDS,
    _graph_from_stabilizer_group,  # ruff: ignore[import-private-name]
    _graph_isomorphism_stabilizer_code,  # ruff: ignore[import-private-name]
    _locally_equivalent_connected_graphs,  # ruff: ignore[import-private-name]
    _lse_stabilizer_code,  # ruff: ignore[import-private-name]
    _preserved_low_degree_local_invariant,  # ruff: ignore[import-private-name]
    _stabilizer_code_to_state,  # ruff: ignore[import-private-name]
    _stabilizer_state_to_graph_state,  # ruff: ignore[import-private-name]
)

from .conftest import assert_same_row_space

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

# The stages of the pipeline that follow the cheap invariants.
STAGES = {
    "low-degree": "_preserved_low_degree_local_invariant",
    "lse": "_lse_stabilizer_code",
    "graph-isomorphism": "_graph_isomorphism_stabilizer_code",
    "sat": "_sat_stabilizer_code",
}


@pytest.fixture
def stages(trace_stages: Callable[[ModuleType, Mapping[str, str]], list[str]]) -> list[str]:
    """Record the stages that the LC-equivalence pipeline runs."""
    return trace_stages(local_clifford_equivalence, STAGES)


# ----------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------


def _apply_lc_witness(symplectic: np.ndarray, witness: list[str]) -> np.ndarray:
    transformed = symplectic.copy()

    assert len(witness) == transformed.shape[1] // 2
    assert all(operation in LOCAL_CLIFFORDS for operation in witness)

    for qubit, operation in enumerate(witness):
        n = transformed.shape[1] // 2
        matrix = np.asarray(CLIFFORD_ACTIONS[operation].matrix, dtype=np.int8)
        x_column = transformed[:, qubit].copy()
        z_column = transformed[:, qubit + n].copy()
        transformed[:, qubit] = (matrix[0, 0] * x_column + matrix[0, 1] * z_column) % 2
        transformed[:, qubit + n] = (matrix[1, 0] * x_column + matrix[1, 1] * z_column) % 2

    return transformed


def _zz_chain(n: int, num_generators: int) -> list[str]:
    return ["".join("Z" if qubit in {i, i + 1} else "I" for qubit in range(n)) for i in range(num_generators)]


def _graph_state(n: int, edges: set[tuple[int, int]]) -> list[str]:
    return [
        "".join(
            "X" if qubit == vertex else "Z" if (min(qubit, vertex), max(qubit, vertex)) in edges else "I"
            for qubit in range(n)
        )
        for vertex in range(n)
    ]


def _assert_maps_rowspace(
    code1: StabilizerCode,
    code2: StabilizerCode,
    witness: list[str],
) -> None:
    transformed = _apply_lc_witness(code1.symplectic, witness)
    assert_same_row_space(transformed, code2.symplectic)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        pytest.param(StabilizerCode(["Z"]), np.array([[0, 1]], dtype=np.uint8), id="single-qubit-z-state"),
        pytest.param(StabilizerCode(["X"]), np.array([[1, 0]], dtype=np.uint8), id="single-qubit-x-state"),
        pytest.param(
            StabilizerCode.get_trivial_code(1),
            np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8),
            id="one-qubit-trivial-code",
        ),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            np.array([[0, 0, 0, 1, 1, 0], [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 0, 1]], dtype=np.uint8),
            id="two-qubit-repetition-code",
        ),
        pytest.param(
            StabilizerCode(["ZZI", "IZZ"], z_logicals=["ZII"], x_logicals=["XXX"]),
            np.array(
                [
                    [0, 0, 0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 1],
                ],
                dtype=np.uint8,
            ),
            id="three-qubit-repetition-code",
        ),
    ],
)
def test_code_to_state(code: StabilizerCode, expected: np.ndarray) -> None:
    """Test that stabilizer codes are converted to the expected purified states."""
    assert np.array_equal(_stabilizer_code_to_state(code), expected)


@pytest.mark.parametrize(
    ("tableau", "expected"),
    [
        pytest.param(np.array([[1, 0]], dtype=np.uint8), np.array([[0]], dtype=np.uint8), id="isolated-vertex"),
        pytest.param(
            np.array([[1, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0]], dtype=np.uint8),
            np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.uint8),
            id="triangle",
        ),
        pytest.param(
            np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
            id="hadamard-improvement",
        ),
        pytest.param(
            np.array([[0, 0, 0, 0, 1, 0], [1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 1]], dtype=np.uint8),
            np.zeros((3, 3), dtype=np.uint8),
            id="mixed-columns",
        ),
    ],
)
def test_state_to_graph(tableau: np.ndarray, expected: np.ndarray) -> None:
    """Test that stabilizer states are converted to the expected graph states."""
    original = tableau.copy()
    adjacency, _ = _stabilizer_state_to_graph_state(tableau)

    assert np.array_equal(adjacency, expected)
    assert np.array_equal(tableau, original)


def test_graph_lc_positive() -> None:
    """Test that the star and complete graphs are locally equivalent."""
    star = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.uint8)
    complete = np.ones((4, 4), dtype=np.uint8) ^ np.eye(4, dtype=np.uint8)

    assert _locally_equivalent_connected_graphs(star, complete) is not None


def test_graph_lc_order_three_cliffords() -> None:
    """Test that the operations between two locally equivalent graphs are not inverted."""
    path = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
    triangle = np.ones((3, 3), dtype=np.int8) ^ np.eye(3, dtype=np.int8)
    identity = np.eye(3, dtype=np.int8)

    operations = _locally_equivalent_connected_graphs(path, triangle)

    assert operations is not None
    assert_same_row_space(_apply_lc_witness(np.hstack([identity, path]), operations), np.hstack([identity, triangle]))


def test_graph_lc_negative() -> None:
    """Test that the path and star graphs are not locally equivalent."""
    path = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=np.uint8,
    )
    star = np.array(
        [[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
        dtype=np.uint8,
    )

    assert _locally_equivalent_connected_graphs(path, star) is None


def test_stabilizer_group_graph() -> None:
    """Test the colored incidence graph of a stabilizer group with two elements."""
    graph = _graph_from_stabilizer_group(StabilizerCode(["XYZI"]).symplectic)

    # vertices 12 and 13 are the identity and the generator, vertices 0, 5, and 7 are X_0, Y_1, and Z_2
    assert set(graph.nodes) == set(range(14))
    assert set(graph.edges) == {(0, 13), (5, 13), (7, 13)}
    assert [graph.nodes[vertex]["color"] for vertex in range(12)] == [("qubit", vertex // 3) for vertex in range(12)]
    assert graph.nodes[12]["color"] == graph.nodes[13]["color"] == ("stabilizer",)


# ----------------------------------------------------------------------------------------------------
# are_local_clifford_equivalent
# ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("violated", "code1", "code2"),
    [
        pytest.param(_preserved_n, StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(4), id="n"),
        pytest.param(_preserved_k, StabilizerCode.get_trivial_code(3), StabilizerCode(["ZII"]), id="k"),
        pytest.param(
            _preserved_distance, StabilizerCode(["ZZ"], distance=1), StabilizerCode(["ZZ"], distance=2), id="d"
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

    assert are_local_clifford_equivalent(code1, code2) is None
    assert stages == []


def test_trivial_code(stages: list[str]) -> None:
    """Test that a trivial stabilizer code returns the identity witness without running any stage."""
    code = StabilizerCode.get_trivial_code(3)

    assert are_local_clifford_equivalent(code, code) == ["I", "I", "I"]
    assert stages == []


@pytest.mark.parametrize(
    ("code1", "code2"),
    [
        pytest.param(StabilizerCode(["ZI", "IZ"]), StabilizerCode(["XX", "ZZ"]), id="product-vs-bell-state"),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            StabilizerCode(["ZI"], z_logicals=["IZ"], x_logicals=["IX"]),
            id="weight-two-vs-weight-one-stabilizer",
        ),
        pytest.param(StabilizerCode(["ZZII", "IIZZ"]), StabilizerCode(["ZZZZ", "XXII"]), id="two-logical-qubits"),
    ],
)
def test_low_degree_rejection(code1: StabilizerCode, code2: StabilizerCode, stages: list[str]) -> None:
    """Test that the low-degree local invariant rejects an inequivalent pair of small codes."""
    assert not _preserved_low_degree_local_invariant(code1, code2)

    assert are_local_clifford_equivalent(code1, code2) is None
    assert stages == ["low-degree"]


def test_low_degree_basis_change() -> None:
    """Test that the low-degree local invariant ignores local basis changes."""
    code1 = StabilizerCode(["ZI", "IZ"])
    code2 = StabilizerCode(["YI", "I" + "Y"])

    assert _preserved_low_degree_local_invariant(code1, code2)


@pytest.mark.parametrize(
    ("code1", "code2", "expected_witness"),
    [
        pytest.param(StabilizerCode(["Z"]), StabilizerCode(["Z"]), ["I"], id="one-qubit-identity"),
        pytest.param(StabilizerCode(["Z"]), StabilizerCode(["X"]), ["H"], id="one-qubit-z-vs-x"),
        pytest.param(StabilizerCode(["Z"]), StabilizerCode(["Y"]), None, id="one-qubit-z-vs-y"),
        pytest.param(StabilizerCode(["ZI", "IZ"]), StabilizerCode(["XI", "IX"]), ["H", "H"], id="two-product-bases"),
        pytest.param(StabilizerCode(["XY"]), StabilizerCode(["YZ"]), ["S", "HSH"], id="s-and-hsh"),
        pytest.param(StabilizerCode(["YY"]), StabilizerCode(["ZY"]), None, id="order-three-clifford"),
        pytest.param(StabilizerCode(["ZZ"]), StabilizerCode(["ZZ", "ZZ"]), None, id="redundant-generators"),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            StabilizerCode(["XX"], z_logicals=["XI"], x_logicals=["ZZ"]),
            None,
            id="repetition-code-under-hadamards",
        ),
        pytest.param(
            StabilizerCode(["XZZZ", "ZXZZ", "ZZXZ", "ZZZX"]),
            StabilizerCode(["XZZZ", "ZXZZ", "ZZXZ", "ZZZX"]),
            None,
            id="solution-space-of-dimension-greater-than-four",
        ),
    ],
)
def test_lse_positive(
    code1: StabilizerCode, code2: StabilizerCode, expected_witness: list[str] | None, stages: list[str]
) -> None:
    """Test that the LSE backend finds an LC witness for codes with at most one logical qubit."""
    witness = are_local_clifford_equivalent(code1, code2)

    assert witness is not None
    _assert_maps_rowspace(code1, code2, witness)
    if expected_witness is not None:
        assert witness == expected_witness
    assert stages == ["low-degree", "lse"]


def test_lse_positive_larger_codes(stages: list[str]) -> None:
    """Test that the LSE backend finds an LC witness for larger graph states without checking an invariant."""
    n = 11
    star = StabilizerCode(_graph_state(n, {(0, vertex) for vertex in range(1, n)}))
    complete = StabilizerCode(_graph_state(n, {(u, v) for u in range(n) for v in range(u + 1, n)}))

    witness = are_local_clifford_equivalent(star, complete)

    assert witness is not None
    _assert_maps_rowspace(star, complete, witness)
    assert stages == ["lse"]


def test_lse_negative(stages: list[str]) -> None:
    """Test that the LSE backend rejects an inequivalent pair that preserves the low-degree local invariant."""
    code1 = StabilizerCode(["YXIYZX", "XXYXZI", "ZZZIXZ", "XIZZZI", "IIZXXI", "IXIIIX"])
    code2 = StabilizerCode(["YYYIZZ", "YZIZYZ", "ZZYIYX", "XXYZII", "ZZZXZX", "IYYYXX"])

    assert are_local_clifford_equivalent(code1, code2) is None
    assert stages == ["low-degree", "lse"]


def test_lse_negative_larger_codes(stages: list[str]) -> None:
    """Test that the LSE backend rejects graph states from different LC orbits without checking an invariant."""
    n = 11
    path = StabilizerCode(_graph_state(n, {(vertex, vertex + 1) for vertex in range(n - 1)}))
    star = StabilizerCode(_graph_state(n, {(0, vertex) for vertex in range(1, n)}))

    assert are_local_clifford_equivalent(path, star) is None
    assert stages == ["lse"]


def test_lse_connected_components() -> None:
    """Test that the LSE backend rejects graph states with different connected components."""
    product_state = StabilizerCode(["ZI", "IZ"])
    bell_state = StabilizerCode(["XX", "ZZ"])

    assert _lse_stabilizer_code(product_state, bell_state) is None


@pytest.mark.parametrize(
    ("code1", "code2", "expected_stages"),
    [
        pytest.param(
            StabilizerCode(["ZZII", "IIZZ"]),
            StabilizerCode(["XXII", "IIXX"]),
            ["low-degree", "graph-isomorphism"],
            id="two-repetition-codes",
        ),
        pytest.param(
            StabilizerCode(["ZIYX", "ZIII"]),
            StabilizerCode(["IIYZ", "XIYZ"]),
            ["low-degree", "graph-isomorphism"],
            id="four-qubit-mixed-code",
        ),
        pytest.param(
            StabilizerCode(_zz_chain(11, 7)),
            StabilizerCode([generator.replace("Z", "Y") for generator in _zz_chain(11, 7)]),
            ["graph-isomorphism"],
            id="most-generators-without-invariant",
        ),
    ],
)
def test_graph_isomorphism_positive(
    code1: StabilizerCode, code2: StabilizerCode, expected_stages: list[str], stages: list[str]
) -> None:
    """Test that the stabilizer graph-isomorphism backend finds an LC witness."""
    assert code1.k >= 2

    witness = are_local_clifford_equivalent(code1, code2)

    assert witness is not None
    _assert_maps_rowspace(code1, code2, witness)
    assert stages == expected_stages


@pytest.mark.parametrize(
    ("code1", "code2", "expected_stages"),
    [
        pytest.param(
            StabilizerCode(["YIXXZ", "XXIYI", "XIYYY"]),
            StabilizerCode(["ZYIZY", "YZIYZ", "YIXIZ"]),
            ["low-degree", "graph-isomorphism"],
            id="five-qubit-codes",
        ),
        pytest.param(
            StabilizerCode(_zz_chain(11, 7)),
            StabilizerCode([*_zz_chain(11, 6), "IIIIIIIIIZZ"]),
            ["graph-isomorphism"],
            id="without-invariant",
        ),
    ],
)
def test_graph_isomorphism_negative(
    code1: StabilizerCode, code2: StabilizerCode, expected_stages: list[str], stages: list[str]
) -> None:
    """Test that the stabilizer graph-isomorphism backend rejects an inequivalent pair."""
    assert code1.k >= 2

    assert are_local_clifford_equivalent(code1, code2) is None
    assert stages == expected_stages


@pytest.mark.parametrize("operation", LOCAL_CLIFFORDS)
def test_graph_isomorphism_witness(operation: str) -> None:
    """Test that the stabilizer graph-isomorphism backend extracts a valid witness for every local Clifford."""
    code1 = StabilizerCode(["XZYI", "IYXZ"])
    transformed = _apply_lc_witness(code1.symplectic, [operation] * code1.n)
    paulis = transformed[:, : code1.n] + 2 * transformed[:, code1.n :]
    code2 = StabilizerCode(["".join("IXZY"[pauli] for pauli in row) for row in paulis])

    witness = _graph_isomorphism_stabilizer_code(code1, code2)

    assert witness is not None
    _assert_maps_rowspace(code1, code2, witness)


@pytest.mark.parametrize(
    ("n", "expected_stages"),
    [pytest.param(10, ["low-degree", "sat"], id="with-invariant"), pytest.param(11, ["sat"], id="without-invariant")],
)
def test_sat_positive(n: int, expected_stages: list[str], stages: list[str]) -> None:
    """Test that the stabilizer SAT backend finds an LC witness for codes with many generators."""
    code1 = StabilizerCode(_zz_chain(n, 8))
    code2 = StabilizerCode([generator.replace("Z", "X") for generator in _zz_chain(n, 8)])

    assert code1.k >= 2

    witness = are_local_clifford_equivalent(code1, code2)

    assert witness is not None
    _assert_maps_rowspace(code1, code2, witness)
    assert stages == expected_stages


@pytest.mark.parametrize(
    ("code1", "code2", "expected_stages"),
    [
        pytest.param(
            StabilizerCode([
                "YIIIIIZZYZ",
                "ZYIZYXXZYI",
                "IIXIIIZIZX",
                "XIIZIZIZYY",
                "ZIIZXZIYZX",
                "IZIXIXIIII",
                "YIIXIIYXIZ",
                "ZIIIYXXXXI",
            ]),
            StabilizerCode([
                "ZXXYZZYZYI",
                "ZYIZIZYIIY",
                "ZIZYIZZIYI",
                "IXIXXZYZYI",
                "IIIYXZZZYI",
                "XXYZIYIZIZ",
                "IZIYXZIIIY",
                "YYXIZIZYYI",
            ]),
            ["low-degree", "sat"],
            id="with-invariant",
        ),
        pytest.param(
            StabilizerCode(_zz_chain(11, 8)),
            StabilizerCode([*_zz_chain(11, 7), "IIIIIIIIIZZ"]),
            ["sat"],
            id="without-invariant",
        ),
    ],
)
def test_sat_negative(
    code1: StabilizerCode, code2: StabilizerCode, expected_stages: list[str], stages: list[str]
) -> None:
    """Test that the stabilizer SAT backend rejects an inequivalent pair of codes with many generators."""
    assert code1.k >= 2

    assert are_local_clifford_equivalent(code1, code2) is None
    assert stages == expected_stages


# ----------------------------------------------------------------------------------------------------
# is_local_clifford_equivalent_to_css
# ----------------------------------------------------------------------------------------------------


def test_css_trivial_code() -> None:
    """Test that a trivial stabilizer code is LC-equivalent to a CSS code."""
    assert is_local_clifford_equivalent_to_css(StabilizerCode.get_trivial_code(3)) is True


def test_css_code() -> None:
    """Test that a constructed CSS code is LC-equivalent to a CSS code."""
    code = RotatedSurfaceCode(3)

    assert is_local_clifford_equivalent_to_css(code) is True


def test_css_small_positive() -> None:
    """Test that a small stabilizer state is LC-equivalent to a CSS code."""
    assert is_local_clifford_equivalent_to_css(StabilizerCode(["YX"])) is True


def test_css_sat_negative() -> None:
    """Test that the CSS SAT backend rejects a negative instance."""
    code = StabilizerCode(["IZIIII", "IIZZIZ", "ZZIZZZ", "ZIIXIY"])

    assert code.n >= 4
    assert is_local_clifford_equivalent_to_css(code) is False


def test_css_bruteforce_positive() -> None:
    """Test that the CSS brute-force backend accepts a positive instance."""
    code = StabilizerCode(["XYZ"])

    assert code.n < 4
    assert is_local_clifford_equivalent_to_css(code) is True


def test_css_sat_positive() -> None:
    """Test that the CSS SAT backend accepts a positive instance."""
    code = StabilizerCode(["YXII", "IIXX", "ZZZZ"])

    assert code.n >= 4
    assert is_local_clifford_equivalent_to_css(code) is True
