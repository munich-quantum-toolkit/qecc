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

from mqt.qecc import CSSCode, StabilizerCode
from mqt.qecc.equivalence_checking import are_permutation_equivalent
from mqt.qecc.mod2 import is_in_row_space, rank


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

    assert rank(code2.symplectic) == rank(code1.symplectic)
    assert all(is_in_row_space(row, code2.symplectic) for row in transformed)


def _assert_maps_rowspace_css(code1: CSSCode, code2: CSSCode, permutation: list[int]) -> None:
    hx_rank = rank(code1.Hx)
    hz_rank = rank(code1.Hz)
    permuted_hx = code2.Hx[:, permutation]
    permuted_hz = code2.Hz[:, permutation]

    assert rank(code2.Hx) == hx_rank
    assert rank(code2.Hz) == hz_rank
    assert all(is_in_row_space(row, permuted_hx) for row in code1.Hx)
    assert all(is_in_row_space(row, permuted_hz) for row in code1.Hz)


# ----------------------------------------------------------------------------------------------------
# are_permutation_equivalent - stabilizer codes
# ----------------------------------------------------------------------------------------------------


def test_are_permutation_equivalent_stabilizer_preserves_n() -> None:
    """Codes on a different number of qubits can never be permutation-equivalent."""
    assert are_permutation_equivalent(StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(4)) is None


def test_are_permutation_equivalent_stabilizer_preserves_k() -> None:
    """Codes with a different number of logical qubits can never be permutation-equivalent."""
    code1 = StabilizerCode.get_trivial_code(3)
    code2 = StabilizerCode(["ZII"])

    assert are_permutation_equivalent(code1, code2) is None


def test_are_permutation_equivalent_stabilizer_does_not_swap_x_and_z() -> None:
    """A permutation may reorder qubits but must not conflate X-type and Z-type stabilizers."""
    code1 = StabilizerCode(["XII"])
    code2 = StabilizerCode(["ZII"])

    assert code1.n == code2.n
    assert code1.k == code2.k
    assert are_permutation_equivalent(code1, code2) is None


def test_are_permutation_equivalent_stabilizer_trivial() -> None:
    """The trivial code is permutation-equivalent to itself via the identity permutation."""
    assert are_permutation_equivalent(StabilizerCode.get_trivial_code(3), StabilizerCode.get_trivial_code(3)) == [
        0,
        1,
        2,
    ]


def test_are_permutation_equivalent_one_qubit_codes_are_equivalent() -> None:
    """A single-qubit code is permutation-equivalent to itself via the identity permutation."""
    assert are_permutation_equivalent(StabilizerCode(["Z"]), StabilizerCode(["Z"])) == [0]


def test_are_permutation_equivalent_one_qubit_codes_reject_pauli_mismatch() -> None:
    """A single-qubit permutation cannot turn a Z stabilizer into an X stabilizer."""
    assert are_permutation_equivalent(StabilizerCode(["Z"]), StabilizerCode(["X"])) is None


def test_are_permutation_equivalent_stabilizer_hardcoded_positive() -> None:
    """A hardcoded pair of codes related by an interleaving permutation is recognized as equivalent."""
    code1 = StabilizerCode(["XXII", "IIZZ"])
    code2 = StabilizerCode(["XIXI", "IZIZ"])

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_stabilizer(code1, code2, permutation)


def test_are_permutation_equivalent_stabilizer_hardcoded_positive_with_x_z_swap_within_qubit() -> None:
    """A witness permutation can also reorder mixed X/Z stabilizer generators consistently."""
    code1 = StabilizerCode(["XZ", "IZ"])
    code2 = StabilizerCode(["ZX", "ZI"])

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_stabilizer(code1, code2, permutation)


def test_are_permutation_equivalent_stabilizer_hardcoded_positive_sat_backend() -> None:
    """A positive stabilizer instance exercising the SAT backend."""
    n = 6
    generators = ["ZZIIII", "IZZIII", "IIZZII", "IIIZZI", "IIIIZZ", "ZIIIIZ"]
    code1 = StabilizerCode(generators)

    permutation_used = [1, 2, 3, 4, 5, 0]
    permuted_generators = ["".join(g[permutation_used.index(i)] for i in range(n)) for g in generators]
    code2 = StabilizerCode(permuted_generators)

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_stabilizer(code1, code2, permutation)


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
def test_are_permutation_equivalent_stabilizer_hardcoded_negative(
    code1: StabilizerCode,
    code2: StabilizerCode,
) -> None:
    """Check rejection for a set of hardcoded stabilizer-code pairs that are not permutation-equivalent."""
    assert are_permutation_equivalent(code1, code2) is None


def test_are_permutation_equivalent_stabilizer_punctured_hull_weight_enumerator() -> None:
    """Test precondition that codes have to have the same signature distribution to be P-equivalent."""
    code1 = StabilizerCode(["XYZXZZXIXX", "XIZYYXIXZI", "ZZZZXYZIII"])
    code2 = StabilizerCode(["YZXZIZXYIY", "ZIXYXYXXII", "ZYIYIIZZZX"])

    assert code1.n == code2.n == 10
    assert code1.k == code2.k
    assert code1.distance == code2.distance

    assert are_permutation_equivalent(code1, code2) is None


# ----------------------------------------------------------------------------------------------------
# are_permutation_equivalent - CSS codes
# ----------------------------------------------------------------------------------------------------


def test_are_permutation_equivalent_css_preserves_n() -> None:
    """Test precondition that codes have to have the same number of physical qubits to be P-equivalent."""
    assert are_permutation_equivalent(CSSCode.get_trivial_code(3), CSSCode.get_trivial_code(4)) is None


def test_are_permutation_equivalent_css_preserves_k() -> None:
    """Test precondition that codes have to have the same number of logical qubits to be P-equivalent."""
    code1 = CSSCode.get_trivial_code(4)
    code2 = CSSCode(Hx=np.array([[1, 0, 0, 0]], dtype=np.int8))

    assert are_permutation_equivalent(code1, code2) is None


def test_are_permutation_equivalent_css_preserves_x_and_z_ranks() -> None:
    """Test precondition that codes have to have the same X and Z ranks to be P-equivalent."""
    code1 = CSSCode(Hx=np.array([[1, 0, 0, 0]], dtype=np.int8))
    code2 = CSSCode(Hz=np.array([[1, 0, 0, 0]], dtype=np.int8))

    assert code1.n == code2.n
    assert code1.k == code2.k
    assert are_permutation_equivalent(code1, code2) is None


def test_are_permutation_equivalent_css_preserves_linear_dependency() -> None:
    """Test precondition that codes have to have the same linear column dependencies to be P-equivalent."""
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


def test_are_permutation_equivalent_css_trivial() -> None:
    """Test trivial CSS code with trivial permutation."""
    code = CSSCode.get_trivial_code(4)

    assert rank(code.Hx) == 0
    assert rank(code.Hz) == 0
    assert are_permutation_equivalent(code, code) == [0, 1, 2, 3]


def test_are_permutation_equivalent_css_bruteforce_positive() -> None:
    """A positive CSS instance exercising the bruteforce backend."""
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


def test_are_permutation_equivalent_css_bruteforce_negative() -> None:
    """A negative CSS instance exercising the bruteforce backend."""
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


def test_are_permutation_equivalent_css_matroid_positive() -> None:
    """A positive CSS instance exercising the matroid-isomorphism backend."""
    n = 10
    hx = np.zeros((5, n), dtype=np.int8)
    for i in range(5):
        hx[i, 2 * i] = 1
        hx[i, 2 * i + 1] = 1
    code1 = CSSCode(Hx=hx, Hz=None)

    permutation_used = list(reversed(range(n)))
    code2 = CSSCode(Hx=hx[:, permutation_used], Hz=None)

    permutation = are_permutation_equivalent(code1, code2)

    assert permutation is not None
    _assert_maps_rowspace_css(code1, code2, permutation)


def test_are_permutation_equivalent_css_matroid_negative() -> None:
    """A negative CSS instance exercising the matroid-isomorphism backend."""
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


def test_are_permutation_equivalent_css_matroid_negative_circuits() -> None:
    """A negative CSS instance exercising the matroid-isomorphism backend (different circuits)."""
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


def test_are_permutation_equivalent_css_sat_positive() -> None:
    """A positive CSS instance exercising the SAT backend."""
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


def test_are_permutation_equivalent_css_sat_negative() -> None:
    """A negative CSS instance exercising the SAT backend."""
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
