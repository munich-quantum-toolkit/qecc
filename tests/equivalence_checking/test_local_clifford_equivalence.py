# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the local clifford equivalence functions."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qecc import StabilizerCode, are_local_clifford_equivalent, is_local_clifford_equivalent_to_css
from mqt.qecc.codes import RotatedSurfaceCode
from mqt.qecc.equivalence_checking._cliffords import _canonicalize_clifford  # ruff: ignore[import-private-name]

# Import the private mathematical helpers intentionally for focused unit tests.
from mqt.qecc.equivalence_checking.local_clifford_equivalence import (
    CLIFFORD_ACTIONS,
    LOCAL_CLIFFORDS,
    _locally_equivalent_connected_graphs,  # ruff: ignore[import-private-name]
    _preserved_low_degree_local_invariant,  # ruff: ignore[import-private-name]
    _stabilizer_code_to_state,  # ruff: ignore[import-private-name]
    _stabilizer_state_to_graph_state,  # ruff: ignore[import-private-name]
)

from .conftest import assert_same_row_space

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


def test_invalid_clifford() -> None:
    """Test that Clifford words with unsupported gates are rejected."""
    with pytest.raises(ValueError, match="Unknown Clifford gate 'X'"):
        _canonicalize_clifford("HX")


# ----------------------------------------------------------------------------------------------------
# are_local_clifford_equivalent
# ----------------------------------------------------------------------------------------------------


def test_preserves_n() -> None:
    """Test that LC equivalence preserves the number of physical qubits."""
    assert are_local_clifford_equivalent(StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(4)) is None


def test_preserves_k() -> None:
    """Test that LC equivalence preserves the number of logical qubits."""
    code1 = StabilizerCode.get_trivial_code(3)
    code2 = StabilizerCode(["ZII"])

    assert are_local_clifford_equivalent(code1, code2) is None


@pytest.mark.parametrize(
    ("code1", "code2", "expected"),
    [
        pytest.param(StabilizerCode(["Z"]), StabilizerCode(["X"]), True, id="one-qubit-z-vs-x"),
        pytest.param(StabilizerCode(["Z"]), StabilizerCode(["Y"]), True, id="one-qubit-z-vs-y"),
        pytest.param(
            StabilizerCode(["ZI", "IZ"]),
            StabilizerCode(["XI", "IX"]),
            True,
            id="two-product-bases",
        ),
        pytest.param(
            StabilizerCode(["ZI", "IZ"]),
            StabilizerCode(["XX", "ZZ"]),
            False,
            id="product-vs-bell-state",
        ),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            StabilizerCode(["XX"], z_logicals=["XI"], x_logicals=["ZZ"]),
            True,
            id="repetition-code-under-hadamards",
        ),
        pytest.param(
            StabilizerCode(["ZZ"], z_logicals=["ZI"], x_logicals=["XX"]),
            StabilizerCode(["ZI"], z_logicals=["IZ"], x_logicals=["IX"]),
            False,
            id="weight-two-vs-weight-one-stabilizer",
        ),
        pytest.param(
            StabilizerCode(["ZIYX", "ZIII"]),
            StabilizerCode(["IIYZ", "XIYZ"]),
            True,
            id="four-qubit-mixed-code",
        ),
    ],
)
def test_hardcoded_cases(
    code1: StabilizerCode,
    code2: StabilizerCode,
    expected: bool,
) -> None:
    """Test LC equivalence for small stabilizer-code pairs with known outcomes."""
    witness = are_local_clifford_equivalent(code1, code2)

    assert (witness is not None) is expected
    if witness is not None:
        _assert_maps_rowspace(code1, code2, witness)


def test_hadamard_witness() -> None:
    """Test that a one-qubit basis change returns a Hadamard witness."""
    witness = are_local_clifford_equivalent(StabilizerCode(["Z"]), StabilizerCode(["X"]))

    assert witness == ["H"]


def test_two_hadamard_witness() -> None:
    """Test that two product-basis changes return two Hadamards."""
    witness = are_local_clifford_equivalent(StabilizerCode(["ZI", "IZ"]), StabilizerCode(["XI", "IX"]))

    assert witness == ["H", "H"]


def test_sat_positive() -> None:
    """Test that the stabilizer SAT backend finds an LC witness."""
    code1 = StabilizerCode(["ZZII", "IIZZ"])
    code2 = StabilizerCode(["XXII", "IIXX"])

    assert code1.k == 2

    witness = are_local_clifford_equivalent(code1, code2)

    assert witness is not None
    _assert_maps_rowspace(code1, code2, witness)


def test_sat_negative() -> None:
    """Test that the stabilizer SAT backend rejects an inequivalent pair."""
    code1 = StabilizerCode(["YYZYZ", "IXIIX", "ZXZXX"])
    code2 = StabilizerCode(["XIZIY", "XXZII", "IYXIZ"])

    assert code1.k == 2
    assert code1.n == code2.n
    assert code1.k == code2.k
    assert code1.distance == code2.distance

    assert are_local_clifford_equivalent(code1, code2) is None


def test_low_degree_rejection() -> None:
    """Test that the low-degree local invariant rejects an inequivalent pair."""
    code1 = StabilizerCode(["ZZII", "IIZZ"])
    code2 = StabilizerCode(["ZZZZ", "XXII"])

    assert code1.n == code2.n
    assert code1.k == code2.k == 2
    assert code1.distance == code2.distance

    assert not _preserved_low_degree_local_invariant(code1, code2)

    assert are_local_clifford_equivalent(code1, code2) is None


def test_low_degree_basis_change() -> None:
    """Test that the low-degree local invariant ignores local basis changes."""
    code1 = StabilizerCode(["ZI", "IZ"])
    code2 = StabilizerCode(["YI", "I" + "Y"])

    assert _preserved_low_degree_local_invariant(code1, code2)


def test_s_and_hsh_witness() -> None:
    """Test that the LSE backend extracts S and HSH operations."""
    code1 = StabilizerCode(["XY"])
    code2 = StabilizerCode(["YZ"])

    witness = are_local_clifford_equivalent(code1, code2)

    assert witness is not None
    assert witness == ["S", "HSH"]
    _assert_maps_rowspace(code1, code2, witness)


def test_large_lse_nullspace() -> None:
    """Test the LSE branch for solution spaces of dimension greater than four."""
    code = StabilizerCode(["XZZZ", "ZXZZ", "ZZXZ", "ZZZX"])

    witness = are_local_clifford_equivalent(code, code)

    assert witness is not None
    _assert_maps_rowspace(code, code, witness)


def test_lse_negative() -> None:
    """Test that the LSE backend rejects graph states from different LC orbits."""
    path = StabilizerCode(["XZII", "ZXZI", "IZXZ", "IIZX"])
    star = StabilizerCode(["XZZZ", "ZXII", "ZIXI", "ZIIX"])

    assert are_local_clifford_equivalent(path, star) is None


# ----------------------------------------------------------------------------------------------------
# are_local_clifford_equivalent - edge cases
# ----------------------------------------------------------------------------------------------------


def test_trivial_code() -> None:
    """Test that a trivial stabilizer code is LC-equivalent to itself."""
    code = StabilizerCode.get_trivial_code(3)
    witness = are_local_clifford_equivalent(code, code)

    assert witness is not None
    assert len(witness) == code.n
    _assert_maps_rowspace(code, code, witness)


def test_one_qubit_code() -> None:
    """Test that a one-qubit code returns the identity witness."""
    witness = are_local_clifford_equivalent(StabilizerCode(["Z"]), StabilizerCode(["Z"]))

    assert witness == ["I"]


def test_redundant_generators() -> None:
    """Test that redundant stabilizer generators do not affect LC equivalence."""
    code = StabilizerCode(["ZZ"])
    code_with_redundancy = StabilizerCode(["ZZ", "ZZ"])

    witness = are_local_clifford_equivalent(code, code_with_redundancy)

    assert witness is not None
    _assert_maps_rowspace(code, code_with_redundancy, witness)


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
