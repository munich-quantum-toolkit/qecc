# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the local clifford equivalence functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mqt.qecc import StabilizerCode
from mqt.qecc.equivalence_checking import are_local_clifford_equivalent
from mqt.qecc.mod2 import is_in_row_space, rank

if TYPE_CHECKING:
    import numpy as np

LOCAL_CLIFFORDS = ("I", "H", "S", "HS", "SH", "HSH")

# ----------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------


def _apply_lc_witness(symplectic: np.ndarray, witness: list[str]) -> np.ndarray:
    """Apply matrix-ordered LC words to a symplectic tableau."""
    transformed = symplectic.copy()
    n = transformed.shape[1] // 2

    assert len(witness) == n
    assert all(operation in LOCAL_CLIFFORDS for operation in witness)

    for qubit, operation in enumerate(witness):
        # strings denote matrix products
        for gate in reversed(operation):
            if gate == "H":
                transformed[:, [qubit, qubit + n]] = transformed[:, [qubit + n, qubit]]
            elif gate == "S":
                transformed[:, qubit + n] ^= transformed[:, qubit]

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
    """Codes on a different number of qubits can never be LC-equivalent."""
    assert are_local_clifford_equivalent(StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(4)) is None


def test_are_local_clifford_equivalent_preserves_k() -> None:
    """Codes with a different number of logical qubits can never be LC-equivalent."""
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
    """Check LC-equivalence for a set of hardcoded stabilizer-code pairs with known outcomes."""
    witness = are_local_clifford_equivalent(code1, code2)

    assert (witness is not None) is expected
    if witness is not None:
        _assert_maps_rowspace(code1, code2, witness)


def test_are_local_clifford_equivalent_one_qubit_z_vs_x_witness_is_hadamard() -> None:
    """The witness mapping the Z stabilizer to the X stabilizer is a single Hadamard."""
    witness = are_local_clifford_equivalent(StabilizerCode(["Z"]), StabilizerCode(["X"]))

    assert witness == ["H"]


def test_are_local_clifford_equivalent_two_product_bases_witness_is_two_hadamards() -> None:
    """The witness mapping a two-qubit Z-basis product state to the X-basis one is two Hadamards."""
    witness = are_local_clifford_equivalent(StabilizerCode(["ZI", "IZ"]), StabilizerCode(["XI", "IX"]))

    assert witness == ["H", "H"]


def test_are_local_clifford_equivalent_hardcoded_positive_sat_backend() -> None:
    """A 4-qubit, 2-logical-qubit code pair exercises the SAT backend."""
    code1 = StabilizerCode(["ZZII", "IIZZ"])
    code2 = StabilizerCode(["XXII", "IIXX"])

    assert code1.k == 2

    witness = are_local_clifford_equivalent(code1, code2)

    assert witness is not None
    _assert_maps_rowspace(code1, code2, witness)


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
    """A code is LC-equivalent to itself via the identity witness."""
    witness = are_local_clifford_equivalent(StabilizerCode(["Z"]), StabilizerCode(["Z"]))

    assert witness == ["I"]


# ----------------------------------------------------------------------------------------------------
# is_local_clifford_equivalent_to_css
# ----------------------------------------------------------------------------------------------------
