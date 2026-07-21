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
    assert p4 * p5 == -p6

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
    tableau = StabilizerTableau(SymplecticMatrix(tableau_matrix))
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
    """Test the StabilizerTableau class."""
    with pytest.raises(InvalidPauliError):
        StabilizerTableau.from_pauli_strings([])

    with pytest.raises(InvalidPauliError):
        StabilizerTableau.from_paulis([])

    m = SymplecticMatrix(np.array([[1, 0], [0, 1]]))
    with pytest.raises(InvalidPauliError):
        StabilizerTableau(m, np.array([1]))

    p1 = Pauli.from_pauli_string("XIZ")
    p2 = Pauli.from_pauli_string("ZIX")
    p3 = Pauli.from_pauli_string("IZX")
    t1 = StabilizerTableau.from_paulis([p1, p2, p3])
    t2 = StabilizerTableau(np.array([[1, 0, 0, 0, 0, 1], [0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 1, 0]]), np.array([0, 0, 0]))
    assert t1 == t2
    assert str(t1) == "XIZ\nZIX\nIZX"
    assert repr(t1) == f"PauliTableau(n=3, n_rows=3, tableau=\n{t1.tableau.data},\nphase={t1.phase})"
    assert not t1.is_row(Pauli.from_pauli_string("III"))

    t3 = StabilizerTableau.from_pauli_strings(["ZII", "IZI", "IIZ"])
    assert t1 != t3

    t4 = StabilizerTableau.from_pauli_strings(["ZII"])
    assert t1 != t4

    assert len(t1) == 3

    with pytest.raises(AssertionError):
        StabilizerTableau.from_matrix(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int8))

    obj = "abc"
    assert t1 != obj

    assert_array_equal(
        t1.to_numpy(), np.array([[1, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 0]], dtype=np.int8)
    )

    t5 = StabilizerTableau.from_matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 1]], dtype=np.int8))
    with pytest.raises(ValueError, match="full"):
        t5.symplectic_submatrix(1)


def test_stabilizer_tableau_to_css() -> None:
    """Test the function to_css of the StabilizerTableau class."""
    p1 = Pauli.from_pauli_string("XII")
    p2 = Pauli.from_pauli_string("IXI")
    p3 = Pauli.from_pauli_string("ZIZ")
    p4 = Pauli.from_pauli_string("YIZ")

    t1 = StabilizerTableau.from_paulis([p1, p2, p3])
    assert t1.is_css()
    cx, cz = t1.to_css()
    assert_array_equal(cx.matrix, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int8))
    assert_array_equal(cz.matrix, np.array([[1, 0, 1]], dtype=np.int8))

    t2 = StabilizerTableau.from_paulis([p1, p2, p4])
    assert not t2.is_css()
    with pytest.raises(InvalidPauliError):
        t2.to_css()


@pytest.fixture
def identity_tableau() -> StabilizerTableau:
    """Fixture for the identity stabilizer tableau."""
    return StabilizerTableau.from_matrix(np.eye(2, dtype=np.int8))


@pytest.fixture
def hadamard_tableau() -> StabilizerTableau:
    """Fixture for the Hadamard stabilizer tableau."""
    return StabilizerTableau.from_matrix(np.array([[0, 1], [1, 0]], dtype=np.int8))


@pytest.fixture
def cnot_tableau() -> StabilizerTableau:
    """Fixture for the CNOT stabilizer tableau."""
    return StabilizerTableau.from_matrix(
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
    """Test the from_stim_tableau method of StabilizerTableau."""
    expected_tableau = request.getfixturevalue(expected_tableau_fixture)
    tableau = StabilizerTableau.from_stim_tableau(stim_tableau)
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
    """Test the from_stim_circuit method of StabilizerTableau."""
    expected_tableau = request.getfixturevalue(expected_tableau_fixture)
    tableau = StabilizerTableau.from_stim_circuit(stim_circuit)
    assert tableau == expected_tableau


def test_complete_stabilizer_tableau_invalid_cases():
    """Test error handling for invalid inputs."""
    stabs = StabilizerTableau.from_pauli_strings(["XX", "ZZ", "YY"])

    with pytest.raises(ValueError, match="Cannot have more stabilizers than qubits"):
        complete_stabilizer_tableau_with_destabilizers(stabs)


def test_pauli_tableau_alias() -> None:
    """StabilizerTableau remains available as a deprecated alias of PauliTableau."""
    assert StabilizerTableau is PauliTableau
    t = PauliTableau.from_pauli_strings(["XX", "ZZ"])
    assert t.symplectic.shape == (2, 4)
    assert_array_equal(t.symplectic, t.tableau.data)


def test_complete_stabilizer_tableau_with_destabilizers():
    """Test completing a stabilizer tableau with destabilizers."""
    stabs = StabilizerTableau.from_pauli_strings(["ZZ"])
    completed = complete_stabilizer_tableau_with_destabilizers(stabs)

    assert completed.num_rows() == 2

    destab = completed[0]
    stab = completed[1]
    assert destab.anticommute(stab)

    stabs = StabilizerTableau.from_pauli_strings(["XXXXIII", "ZZZZIII"])
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

    stabs = StabilizerTableau.from_pauli_strings(["XII", "IZI", "IIX"])
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
