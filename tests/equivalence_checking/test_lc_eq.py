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

from mqt.qecc import CSSCode, StabilizerCode
from mqt.qecc.equivalence_checking import are_local_clifford_equivalent
from mqt.qecc.equivalence_checking.local_clifford_eq import (
    CLIFFORD_ACTIONS,
    LOCAL_CLIFFORDS,
    is_local_clifford_equivalent_to_css,
    preserved_low_degree_local_invariant,
)
from mqt.qecc.mod2 import is_in_row_space, rank

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

    assert rank(code2.symplectic) == rank(code1.symplectic)
    assert all(is_in_row_space(row, code2.symplectic) for row in transformed)


# ----------------------------------------------------------------------------------------------------
# are_local_clifford_equivalent
# ----------------------------------------------------------------------------------------------------


def test_are_local_clifford_equivalent_preserves_n() -> None:
    """Test precondition that codes have to have the same number of physical qubits to be LC-equivalent."""
    assert are_local_clifford_equivalent(StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(4)) is None


def test_are_local_clifford_equivalent_preserves_k() -> None:
    """Test precondition that codes have to have the same number of logical qubits to be LC-equivalent."""
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
def test_are_local_clifford_equivalent_hardcoded_cases(
    code1: StabilizerCode,
    code2: StabilizerCode,
    expected: bool,
) -> None:
    """Test LC-equivalence for a set of hardcoded stabilizer-code pairs with known outcomes."""
    witness = are_local_clifford_equivalent(code1, code2)

    assert (witness is not None) is expected
    if witness is not None:
        _assert_maps_rowspace(code1, code2, witness)


def test_are_local_clifford_equivalent_one_qubit_z_vs_x_witness_is_hadamard() -> None:
    """Test correct witness extraction for LC-equivalent stabilizer codes."""
    witness = are_local_clifford_equivalent(StabilizerCode(["Z"]), StabilizerCode(["X"]))

    assert witness == ["H"]


def test_are_local_clifford_equivalent_two_product_bases_witness_is_two_hadamards() -> None:
    """Test correct witness extraction for LC-equivalent stabilizer codes."""
    witness = are_local_clifford_equivalent(StabilizerCode(["ZI", "IZ"]), StabilizerCode(["XI", "IX"]))

    assert witness == ["H", "H"]


def test_are_local_clifford_equivalent_hardcoded_positive_sat_backend() -> None:
    """A positive stabilizer instance exercising the SAT backend."""
    code1 = StabilizerCode(["ZZII", "IIZZ"])
    code2 = StabilizerCode(["XXII", "IIXX"])

    assert code1.k == 2

    witness = are_local_clifford_equivalent(code1, code2)

    assert witness is not None
    _assert_maps_rowspace(code1, code2, witness)


def test_are_local_clifford_equivalent_hardcoded_negative_sat_backend() -> None:
    """A negative stabilizer instance exercising the SAT backend."""
    code1 = StabilizerCode(["YYZYZ", "IXIIX", "ZXZXX"])
    code2 = StabilizerCode(["XIZIY", "XXZII", "IYXIZ"])

    assert code1.k == 2
    assert code1.n == code2.n
    assert code1.k == code2.k
    assert code1.distance == code2.distance

    assert are_local_clifford_equivalent(code1, code2) is None


def test_are_local_clifford_equivalent_low_degree_invariant_rules_out_pair() -> None:
    """A negative stabilizer instance exercising the local invariant."""
    code1 = StabilizerCode(["ZZII", "IIZZ"])
    code2 = StabilizerCode(["ZZZZ", "XXII"])

    assert code1.n == code2.n
    assert code1.k == code2.k == 2
    assert code1.distance == code2.distance

    assert not preserved_low_degree_local_invariant(code1, code2)

    assert are_local_clifford_equivalent(code1, code2) is None


def test_are_local_clifford_equivalent_witness_uses_s_and_hsh() -> None:
    """A stabilizer instance exercising the graph-state LSE backend with different local operations."""
    code1 = StabilizerCode(["XY"])
    code2 = StabilizerCode(["YZ"])

    witness = are_local_clifford_equivalent(code1, code2)

    assert witness is not None
    assert witness == ["S", "HSH"]
    _assert_maps_rowspace(code1, code2, witness)


def test_are_local_clifford_equivalent_complete_graph_state_exercises_large_nullspace() -> None:
    """A stabilizer instance exercising the graph-state LSE backend with simplifying branch."""
    code = StabilizerCode(["XZZZ", "ZXZZ", "ZZXZ", "ZZZX"])

    witness = are_local_clifford_equivalent(code, code)

    assert witness is not None
    _assert_maps_rowspace(code, code, witness)


# ----------------------------------------------------------------------------------------------------
# are_local_clifford_equivalent - edge cases
# ----------------------------------------------------------------------------------------------------


def test_are_local_clifford_equivalent_trivial_codes_are_equivalent() -> None:
    """The trivial code is LC-equivalent to itself under some (not necessarily identity) witness."""
    code = StabilizerCode.get_trivial_code(3)
    witness = are_local_clifford_equivalent(code, code)

    assert witness is not None
    assert len(witness) == code.n
    _assert_maps_rowspace(code, code, witness)


def test_are_local_clifford_equivalent_one_qubit_codes_are_equivalent() -> None:
    """Test correct witness extraction for LC-equivalent stabilizer codes."""
    witness = are_local_clifford_equivalent(StabilizerCode(["Z"]), StabilizerCode(["Z"]))

    assert witness == ["I"]


def test_are_local_clifford_equivalent_ignores_redundant_generators() -> None:
    """Equivalent codes may use different numbers of stabilizer generators."""
    code = StabilizerCode(["ZZ"])
    code_with_redundancy = StabilizerCode(["ZZ", "ZZ"])

    witness = are_local_clifford_equivalent(code, code_with_redundancy)

    assert witness is not None
    _assert_maps_rowspace(code, code_with_redundancy, witness)


# ----------------------------------------------------------------------------------------------------
# is_local_clifford_equivalent_to_css
# ----------------------------------------------------------------------------------------------------


def test_is_local_clifford_equivalent_to_css_accepts_trivial_code() -> None:
    """The trivial code is (trivially) LC-equivalent to a CSS code."""
    assert is_local_clifford_equivalent_to_css(StabilizerCode.get_trivial_code(3)) is True


def test_is_local_clifford_equivalent_to_css_accepts_css_code() -> None:
    """A CSS code is LC-equivalent to a CSS code via the identity."""
    code = CSSCode(
        Hx=np.array([[1, 1, 0, 0]], dtype=np.int8),
        Hz=np.array([[0, 0, 1, 1]], dtype=np.int8),
    )

    assert is_local_clifford_equivalent_to_css(code) is True


def test_is_local_clifford_equivalent_to_css_hardcoded_positive() -> None:
    """A two-qubit stabilizer state is LC-equivalent to a CSS code."""
    assert is_local_clifford_equivalent_to_css(StabilizerCode(["YX"])) is True


def test_is_local_clifford_equivalent_to_css_hardcoded_negative_sat_backend() -> None:
    """A negative stabilizer instance exercising the SAT backend."""
    code = StabilizerCode(["IZIIII", "IIZZIZ", "ZZIZZZ", "ZIIXIY"])

    assert code.n >= 4
    assert is_local_clifford_equivalent_to_css(code) is False


def test_is_local_clifford_equivalent_to_css_hardcoded_positive_bruteforce_backend() -> None:
    """A positive stabilizer instance exercising the bruteforce backend."""
    code = StabilizerCode(["XYZ"])

    assert code.n < 4
    assert is_local_clifford_equivalent_to_css(code) is True


def test_is_local_clifford_equivalent_to_css_hardcoded_positive_sat_backend() -> None:
    """A positive stabilizer instance exercising the SAT backend."""
    code = StabilizerCode(["YXII", "IIXX", "ZZZZ"])

    assert code.n >= 4
    assert is_local_clifford_equivalent_to_css(code) is True
