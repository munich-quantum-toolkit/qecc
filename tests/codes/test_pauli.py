# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests the Pauli and PauliTableau classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import stim
from numpy.testing import assert_array_equal

from mqt.qecc.codes.core.pauli import (
    CheckMatrix,
    InvalidPauliError,
    Pauli,
    PauliTableau,
    StabilizerTableau,
    complete_stabilizer_tableau_with_destabilizers,
    pauli_rank,
    pauli_row_echelon,
)
from mqt.qecc.codes.core.symplectic import (
    SymplecticMatrix,
    SymplecticVector,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt


def test_pauli() -> None:
    """Test the Pauli class."""
    p1 = Pauli.from_pauli_string("XIZ")
    p2 = Pauli(SymplecticVector(np.array([1, 0, 0, 0, 0, 1])))
    assert p1 == p2
    p3 = p1 * p2
    assert p3 == Pauli.from_pauli_string("III")
    p4 = Pauli.from_pauli_string("-X")
    p5 = Pauli.from_pauli_string("+Z")
    p6 = Pauli.from_pauli_string("Y")
    assert p4 * p5 != p6
    assert p4 * p5 != -p6
    assert p4 * p5 == Pauli.from_pauli_string("+iY")

    assert np.array_equal(p1.x_part(), np.array([1, 0, 0]))
    assert np.array_equal(p1.z_part(), np.array([0, 0, 1]))
    assert np.array_equal(p6.x_part(), np.array([1]))
    assert np.array_equal(p6.z_part(), np.array([1]))
    assert len(p1) == 3
    assert len(p6) == 1

    assert p4.anticommute(p5)
    p7 = Pauli.from_pauli_string("XI")
    p8 = Pauli.from_pauli_string("IZ")
    assert p8.commute(p7)

    obj = "abc"
    assert p1 != obj
    assert_array_equal(p1.as_vector(), np.array([1, 0, 0, 0, 0, 1, 0], dtype=np.int8))
    assert hash(p1) == hash((SymplecticVector(np.array([1, 0, 0, 0, 0, 1], dtype=np.int8)), 0))


def test_pauli_phases() -> None:
    """Test parsing, printing, and multiplication with all four Pauli phases."""
    for pauli_string in ("X", "+iX", "-X", "-iX"):
        assert repr(Pauli.from_pauli_string(pauli_string)) == pauli_string

    x = Pauli.from_pauli_string("X")
    z = Pauli.from_pauli_string("Z")
    assert Pauli.from_pauli_string("Y").phase_exponent == 1
    assert Pauli.from_pauli_string("-Y").phase_exponent == 3
    assert x * z == Pauli.from_pauli_string("-iY")
    assert z * x == Pauli.from_pauli_string("+iY")


def test_default_pauli_phase() -> None:
    """An omitted phase creates the positive Hermitian Pauli for the given support."""
    y_support = Pauli.from_pauli_string("Y").symplectic
    assert Pauli(y_support) == Pauli.from_pauli_string("Y")
    assert Pauli.from_symplectic_and_sign(y_support, 0) == Pauli.from_pauli_string("Y")
    assert Pauli.from_symplectic_and_sign(y_support, 1) == Pauli.from_pauli_string("-Y")


def test_pauli_sign() -> None:
    """Test conversion between phase exponents and binary signs."""
    assert Pauli.from_pauli_string("Y").sign() == 0
    assert Pauli.from_pauli_string("-Y").sign() == 1
    with pytest.raises(InvalidPauliError):
        Pauli.from_pauli_string("+iY").sign()


def test_invalid_arithmetic() -> None:
    """Test that invalid arithmetic operations raise an error."""
    p1 = Pauli.from_pauli_string("XZ")
    p2 = Pauli.from_pauli_string("ZIX")

    with pytest.raises(InvalidPauliError):
        p1 * p2

    with pytest.raises(IndexError):
        p1[3]


@pytest.mark.parametrize(
    ("tableau_matrix", "expected"),
    [
        (np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int8), True),
        (np.array([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=np.int8), False),
    ],
)
def test_is_css(tableau_matrix: npt.NDArray[np.int8], expected: bool) -> None:
    """Test the is_css method."""
    tableau = PauliTableau(SymplecticMatrix(tableau_matrix))
    assert tableau.is_css() == expected


@pytest.mark.parametrize(
    ("tableau_fixture", "expected_x_part", "expected_z_part"),
    [
        ("identity_tableau", np.array([[1], [0]], dtype=np.int8), np.array([[0], [1]], dtype=np.int8)),
        ("hadamard_tableau", np.array([[0], [1]], dtype=np.int8), np.array([[1], [0]], dtype=np.int8)),
    ],
)
def test_get_x_and_z_parts(
    tableau_fixture: str,
    expected_x_part: npt.NDArray[np.int8],
    expected_z_part: npt.NDArray[np.int8],
    request: pytest.FixtureRequest,
) -> None:
    """Test the get_x_part and get_z_part methods with various tableaus."""
    tableau = request.getfixturevalue(tableau_fixture)
    x_part = tableau.get_x_part()
    z_part = tableau.get_z_part()
    assert_array_equal(x_part, expected_x_part)
    assert_array_equal(z_part, expected_z_part)


def test_stabilizer_tableau() -> None:
    """Test the PauliTableau class."""
    with pytest.raises(InvalidPauliError):
        PauliTableau.from_pauli_strings([])

    with pytest.raises(InvalidPauliError):
        PauliTableau.from_paulis([])

    m = SymplecticMatrix(np.array([[1, 0], [0, 1]]))
    with pytest.raises(InvalidPauliError):
        PauliTableau(m, np.array([1]))

    p1 = Pauli.from_pauli_string("XIZ")
    p2 = Pauli.from_pauli_string("ZIX")
    p3 = Pauli.from_pauli_string("IZX")
    t1 = PauliTableau.from_paulis([p1, p2, p3])
    t2 = PauliTableau(np.array([[1, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 0]]), np.array([0, 0, 0]))
    assert t1 == t2
    assert str(t1) == "XIZ\nZIX\nIZX"
    assert repr(t1) == f"PauliTableau(n=3, n_rows=3, tableau=\n{t1.tableau.data},\nphase={t1.phase_exponents})"
    assert not t1.is_row(Pauli.from_pauli_string("III"))

    t3 = PauliTableau.from_pauli_strings(["ZII", "IZI", "IIZ"])
    assert t1 != t3

    t4 = PauliTableau.from_pauli_strings(["ZII"])
    assert t1 != t4

    assert len(t1) == 3

    with pytest.raises(AssertionError):
        PauliTableau.from_matrix(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int8))

    obj = "abc"
    assert t1 != obj

    assert_array_equal(
        t1.to_numpy(), np.array([[1, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0]], dtype=np.int8)
    )

    t5 = PauliTableau.from_matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 1]], dtype=np.int8))
    with pytest.raises(ValueError, match="full"):
        t5.symplectic_submatrix(1)


def test_tableau_signs_and_phase_from_signs() -> None:
    """Test conversion between binary signs and tableau phase exponents."""
    tableau = PauliTableau.from_pauli_strings(["Y", "-Y"])
    assert_array_equal(tableau.signs(), np.array([0, 1], dtype=np.int8))
    assert_array_equal(PauliTableau.phase_from_signs(tableau.tableau.data, tableau.signs()), tableau.phase_exponents)


def test_default_tableau_phase() -> None:
    """An omitted phase creates positive Hermitian rows."""
    tableau = PauliTableau.from_matrix(np.array([[1, 1]], dtype=np.int8))
    assert tableau[0] == Pauli.from_pauli_string("Y")


def test_tableau_rejects_sign_of_non_hermitian_row() -> None:
    """A non-Hermitian tableau row has no binary sign."""
    tableau = PauliTableau.from_pauli_strings(["+iY"])
    with pytest.raises(InvalidPauliError):
        tableau.signs()


@pytest.mark.parametrize("pauli_string", ["+iX", "-iY"])
def test_as_hermitian_matrix_rejects_non_hermitian_rows(pauli_string: str) -> None:
    """A tableau containing a non-Hermitian row has no Hermitian matrix representation."""
    tableau = PauliTableau.from_pauli_strings(["Z", pauli_string])

    with pytest.raises(InvalidPauliError):
        tableau.as_hermitian_matrix()


def test_tableau_gate_phases() -> None:
    """Test phase updates for representative Clifford conjugations."""
    tableau = PauliTableau.from_pauli_strings(["X", "Y", "Z"])
    tableau.apply_s(0)
    assert list(tableau) == [
        Pauli.from_pauli_string("Y"),
        Pauli.from_pauli_string("-X"),
        Pauli.from_pauli_string("Z"),
    ]


def test_tableau_getitem_does_not_alias() -> None:
    """A Pauli taken from a tableau owns its support and does not write back."""
    tableau = PauliTableau.from_pauli_strings(["XX", "ZZ"])

    p = tableau[0]
    p.symplectic[0] = 0

    assert tableau[0] == Pauli.from_pauli_string("XX")
    assert tableau.to_pauli_list()[0] == Pauli.from_pauli_string("XX")


def test_multiply_rows() -> None:
    """Multiplying one row onto another tracks the phase and leaves other rows alone."""
    tableau = PauliTableau.from_pauli_strings(["XX", "ZZ"])

    tableau.multiply_rows(0, 1)

    assert tableau[0] == Pauli.from_pauli_string("-YY")  # XX * ZZ == -YY
    assert tableau[1] == Pauli.from_pauli_string("ZZ")


def test_multiply_rows_is_not_commutative() -> None:
    """Row multiplication is left-multiplication, so the operand order matters."""
    forward = PauliTableau.from_pauli_strings(["X", "Z"])
    backward = PauliTableau.from_pauli_strings(["X", "Z"])

    forward.multiply_rows(0, 1)
    backward.multiply_rows(1, 0)

    assert forward[0] == Pauli.from_pauli_string("-iY")  # X * Z
    assert backward[1] == Pauli.from_pauli_string("+iY")  # Z * X


def test_independent_rows() -> None:
    """Test that independent_rows returns a reduced copy with phases preserved."""
    tableau = PauliTableau.from_pauli_strings(["ZII", "-IZI", "ZZI"])

    reduced = tableau.independent_rows()

    assert reduced == PauliTableau.from_pauli_strings(["ZII", "-IZI"])
    assert tableau == PauliTableau.from_pauli_strings(["ZII", "-IZI", "ZZI"])
    assert reduced is not tableau


def test_stabilizer_tableau_to_css() -> None:
    """Test the function to_css of the PauliTableau class."""
    p1 = Pauli.from_pauli_string("XII")
    p2 = Pauli.from_pauli_string("IXI")
    p3 = Pauli.from_pauli_string("ZIZ")
    p4 = Pauli.from_pauli_string("YIZ")

    t1 = PauliTableau.from_paulis([p1, p2, p3])
    assert t1.is_css()
    cx, cz = t1.to_css()
    assert_array_equal(cx.matrix, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int8))
    assert_array_equal(cz.matrix, np.array([[1, 0, 1]], dtype=np.int8))

    t2 = PauliTableau.from_paulis([p1, p2, p4])
    assert not t2.is_css()
    with pytest.raises(InvalidPauliError):
        t2.to_css()


@pytest.fixture
def identity_tableau() -> PauliTableau:
    """Fixture for the identity stabilizer tableau."""
    return PauliTableau.from_matrix(np.eye(2, dtype=np.int8))


@pytest.fixture
def hadamard_tableau() -> PauliTableau:
    """Fixture for the Hadamard stabilizer tableau."""
    return PauliTableau.from_matrix(np.array([[0, 1], [1, 0]], dtype=np.int8))


@pytest.fixture
def cnot_tableau() -> PauliTableau:
    """Fixture for the CNOT stabilizer tableau."""
    return PauliTableau.from_matrix(
        np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.int8,
        )
    )


@pytest.mark.parametrize(
    ("stim_tableau", "expected_tableau_fixture"),
    [
        (stim.Tableau.from_named_gate("I"), "identity_tableau"),
        (stim.Tableau.from_named_gate("H"), "hadamard_tableau"),
        (stim.Tableau.from_named_gate("CX"), "cnot_tableau"),
    ],
)
def test_stabilizer_tableau_from_stim_tableau(
    stim_tableau: stim.Tableau, expected_tableau_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Test the from_stim_tableau method of PauliTableau."""
    expected_tableau = request.getfixturevalue(expected_tableau_fixture)
    tableau = PauliTableau.from_stim_tableau(stim_tableau)
    assert tableau == expected_tableau


@pytest.mark.parametrize(
    ("stim_circuit", "expected_tableau_fixture"),
    [
        (stim.Circuit("I 0"), "identity_tableau"),
        (stim.Circuit("H 0"), "hadamard_tableau"),
        (stim.Circuit("CX 0 1"), "cnot_tableau"),
    ],
)
def test_stabilizer_tableau_from_stim_circuit(
    stim_circuit: stim.Circuit, expected_tableau_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Test the from_stim_circuit method of PauliTableau."""
    expected_tableau = request.getfixturevalue(expected_tableau_fixture)
    tableau = PauliTableau.from_stim_circuit(stim_circuit)
    assert tableau == expected_tableau


def test_complete_stabilizer_tableau_invalid_cases():
    """Test error handling for invalid inputs."""
    stabs = PauliTableau.from_pauli_strings(["XX", "ZZ", "YY"])

    with pytest.raises(ValueError, match="Cannot have more stabilizers than qubits"):
        complete_stabilizer_tableau_with_destabilizers(stabs)


def test_pauli_tableau_alias() -> None:
    """StabilizerTableau remains available as a deprecated alias of PauliTableau."""
    assert StabilizerTableau is PauliTableau
    t = PauliTableau.from_pauli_strings(["XX", "ZZ"])
    assert t.symplectic.shape == (2, 4)
    assert_array_equal(t.symplectic, t.tableau.data)


def test_pauli_row_echelon_preserves_phases() -> None:
    """Row reduction propagates phases introduced by Pauli multiplication."""
    tableau = PauliTableau.from_pauli_strings(["-YY", "ZZ"])

    reduced, rank, _, transform, pivots = pauli_row_echelon(tableau)

    assert reduced.to_pauli_list() == [
        Pauli.from_pauli_string("XX"),
        Pauli.from_pauli_string("ZZ"),
    ]
    assert rank == 2
    assert pivots == [0, 2]
    assert_array_equal((transform @ tableau.symplectic) % 2, reduced.symplectic)
    assert tableau.to_pauli_list() == [
        Pauli.from_pauli_string("-YY"),
        Pauli.from_pauli_string("ZZ"),
    ]


def test_pauli_row_echelon_keeps_scalar_rows() -> None:
    """Dependent rows reduce to their scalar Pauli product."""
    tableau = PauliTableau.from_pauli_strings(["XX", "ZZ", "-YY"])

    reduced, rank, _, _, pivots = pauli_row_echelon(tableau)

    assert reduced.to_pauli_list() == [
        Pauli.from_pauli_string("XX"),
        Pauli.from_pauli_string("ZZ"),
        Pauli.from_pauli_string("II"),
    ]
    assert rank == 2
    assert pivots == [0, 2]


def test_pauli_row_echelon_on_many_qubits() -> None:
    """Row reduction stays correct when x.z sums exceed the range of a single byte."""
    n = 200  # a 200-qubit all-Y row has x.z == 200, which overflows int8
    tableau = PauliTableau.from_pauli_strings(["Y" * n, "Z" * n, "X" * n])

    reduced, rank, n_global_phases, transform, pivots = pauli_row_echelon(tableau)

    # Y^200 == i^200 X^200 Z^200 and 200 % 4 == 0, so the row is +YY..Y with exponent 0.
    assert tableau[0].phase_exponent == 0
    assert rank == 2
    assert n_global_phases == 1
    assert pivots == [0, n]
    assert_array_equal((transform @ tableau.symplectic) % 2, reduced.symplectic)
    assert reduced[2] == Pauli.from_pauli_string("I" * n)


def test_pauli_rank() -> None:
    """The Pauli rank includes independent support and central phases."""
    assert pauli_rank(PauliTableau.from_pauli_strings(["XX", "ZZ", "-YY"])) == 2
    assert pauli_rank(PauliTableau.from_pauli_strings(["XX", "ZZ", "YY"])) == 3
    assert pauli_rank(PauliTableau.from_pauli_strings(["+iX"])) == 2
    assert pauli_rank(PauliTableau.from_pauli_strings(["X", "Z"])) == 3
    assert pauli_rank(PauliTableau.empty(2)) == 0


def test_is_in_pauli_subgroup() -> None:
    """Subgroup membership includes the phase of the Pauli operator."""
    subgroup = PauliTableau.from_pauli_strings(["XX", "ZZ"])

    assert subgroup.is_in_subgroup(Pauli.from_pauli_string("XX"))
    assert subgroup.is_in_subgroup(Pauli.from_pauli_string("-YY"))
    assert not subgroup.is_in_subgroup(Pauli.from_pauli_string("YY"))
    assert not subgroup.is_in_subgroup(Pauli.from_pauli_string("-II"))
    assert not subgroup.is_in_subgroup(Pauli.from_pauli_string("XI"))

    negative_subgroup = PauliTableau.from_pauli_strings(["-X"])
    assert negative_subgroup.is_in_subgroup(Pauli.from_pauli_string("-X"))
    assert not negative_subgroup.is_in_subgroup(Pauli.from_pauli_string("X"))


def test_is_in_general_pauli_subgroup() -> None:
    """Subgroup membership supports non-Hermitian and anticommuting generators."""
    phased_subgroup = PauliTableau.from_pauli_strings(["+iX"])
    assert phased_subgroup.is_in_subgroup(Pauli.from_pauli_string("-I"))
    assert not phased_subgroup.is_in_subgroup(Pauli.from_pauli_string("X"))

    anticommuting_subgroup = PauliTableau.from_pauli_strings(["X", "Z"])
    assert anticommuting_subgroup.is_in_subgroup(Pauli.from_pauli_string("-I"))
    assert anticommuting_subgroup.is_in_subgroup(Pauli.from_pauli_string("-iY"))


def test_complete_stabilizer_tableau_with_destabilizers():
    """Test completing a stabilizer tableau with destabilizers."""
    stabs = PauliTableau.from_pauli_strings(["ZZ"])
    completed = complete_stabilizer_tableau_with_destabilizers(stabs)

    assert completed.num_rows() == 2

    destab = completed[0]
    stab = completed[1]
    assert destab.anticommute(stab)

    stabs = PauliTableau.from_pauli_strings(["XXXXIII", "ZZZZIII"])
    completed = complete_stabilizer_tableau_with_destabilizers(stabs)

    assert completed.num_rows() == 4

    for i in range(2):
        destab_i = completed[i]
        for j in range(2):
            stab_j = completed[2 + j]
            if i == j:
                assert destab_i.anticommute(stab_j)
            else:
                assert destab_i.commute(stab_j)

    for i in range(2):
        for j in range(i + 1, 2):
            assert completed[i].commute(completed[j])

    stabs = PauliTableau.from_pauli_strings(["XII", "IZI", "IIX"])
    completed = complete_stabilizer_tableau_with_destabilizers(stabs)

    assert completed.num_rows() == 6

    for i in range(3):
        destab_i = completed[i]
        for j in range(3):
            stab_j = completed[3 + j]
            if i == j:
                assert destab_i.anticommute(stab_j)
            else:
                assert destab_i.commute(stab_j)


def test_check_matrix() -> None:
    """Test the CheckMatrix class."""
    matrix = np.eye(2, dtype=np.int8)
    x_checks = CheckMatrix(matrix, "X")
    z_checks = CheckMatrix(matrix, "Z")

    with pytest.raises(ValueError, match="must be either 'X' or 'Z'"):
        CheckMatrix(matrix, "Y")

    assert x_checks.is_x_type()
    assert not x_checks.is_z_type()
    assert z_checks.is_z_type()
    assert x_checks.is_identity()
    assert x_checks.num_qubits() == 2
    assert x_checks.num_rows() == 2
    assert x_checks.copy() == x_checks
    assert x_checks != z_checks
    assert x_checks != "not a check matrix"
    assert hash(x_checks) == hash(x_checks.copy())
    assert x_checks.equ_span(z_checks)
    assert repr(x_checks) == "CheckMatrix(type=X, matrix=\n[[1 0]\n [0 1]])"


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.bool_])
def test_check_matrix_normalizes_dtype(dtype: npt.DTypeLike) -> None:
    """CheckMatrix normalizes any binary integer dtype to int8 (regression test for #775).

    Non-int8 dtypes (e.g. uint8/int16) previously propagated into numba-dispatched synthesis
    routines and raised a "No matching definition" dispatch error.
    """
    matrix = np.eye(2, dtype=dtype)
    checks = CheckMatrix(matrix, "X")
    assert checks.matrix.dtype == np.int8
    assert_array_equal(checks.matrix, np.eye(2, dtype=np.int8))
