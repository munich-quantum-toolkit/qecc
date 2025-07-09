# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the CSSCode class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.qecc import CSSCode, StabilizerCode
from mqt.qecc.codes import (
    ConcatenatedCode,
    ConcatenatedCSSCode,
    InvalidCSSCodeError,
    InvalidStabilizerCodeError,
    construct_bb_code,
    construct_iceberg_code,
    construct_many_hypercube_code,
    construct_quantum_hamming_code,
)
from mqt.qecc.codes.pauli import InvalidPauliError, Pauli, StabilizerTableau
from mqt.qecc.codes.symplectic import SymplecticMatrix, SymplecticVector

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

    with pytest.raises(IndexError):
        p1[3]


def test_symplectic() -> None:
    """Test the SymplecticMatrix and SymplecticVector classes."""
    ones = SymplecticVector.ones(3)
    zeros = SymplecticVector.zeros(3)
    assert ones - ones == zeros
    assert ones + ones == zeros

    v = SymplecticVector(np.array([1, 0, 0, 0, 0, 1]))
    w = SymplecticVector(np.array([0, 1, 0, 0, 0, 1]))
    assert w + v == v + w
    assert w - v == -v + w

    obj = "abc"
    assert v != obj

    assert v @ w == 0
    u = SymplecticVector(np.array([0, 0, 1, 0, 0, 0]))
    assert v @ u == 1

    eye = SymplecticMatrix.identity(3)
    zero_mat = SymplecticMatrix.zeros(6, 3)
    assert eye + eye == zero_mat
    assert eye - eye == zero_mat

    vs = [v.vector, w.vector, u.vector, ones.vector, zeros.vector, v.vector]
    m = SymplecticMatrix(np.array(vs))
    assert eye @ m.transpose() == m
    assert m @ eye == m

    for i, row in enumerate(m):
        assert np.array_equal(row, vs[i])

    assert m != obj
    assert len(m) == 6
    assert m.shape == (6, 6)
    assert m.n == 3


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

    t3 = StabilizerTableau.from_pauli_strings(["ZII", "IZI", "IIZ"])
    assert t1 != t3

    t4 = StabilizerTableau.from_pauli_strings(["ZII"])
    assert t1 != t4

    assert t1 == [Pauli.from_pauli_string("XIZ"), Pauli.from_pauli_string("ZIX"), Pauli.from_pauli_string("IZX")]
    assert len(t1) == 3


@pytest.fixture
def rep_code_checks() -> tuple[npt.NDArray[np.int8] | None, npt.NDArray[np.int8] | None]:
    """Return the parity check matrices for the repetition code."""
    hx = np.array([[1, 1, 0], [0, 1, 1]])
    hz = None
    return hx, hz


@pytest.fixture
def rep_code_checks_reverse() -> tuple[npt.NDArray[np.int8] | None, npt.NDArray[np.int8] | None]:
    """Return the parity check matrices for the repetition code."""
    hz = np.array([[1, 1, 0], [0, 0, 1]])
    hx = None
    return hx, hz


@pytest.fixture
def steane_code_checks() -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """Return the check matrices for the Steane code."""
    hx = np.array([[1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 1, 0]])
    hz = hx
    return hx, hz


@pytest.fixture
def steane_code() -> CSSCode:
    """Return the Steane code."""
    hx = np.array([[1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 1, 0]])
    hz = hx
    return CSSCode(distance=3, Hx=hx, Hz=hz)


@pytest.fixture
def five_qubit_code_stabs() -> list[str]:
    """Return the five qubit code."""
    return ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]


@pytest.fixture
def five_qubit_code() -> StabilizerCode:
    """Return the five qubit code."""
    return StabilizerCode(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"], 3, z_logicals=["ZZZZZ"], x_logicals=["XXXXX"])


def test_invalid_css_codes() -> None:
    """Test that an invalid CSS code raises an error."""
    # Violates CSS condition
    hx = np.array([[1, 1, 1]])
    hz = np.array([[1, 0, 0]])
    with pytest.raises(InvalidCSSCodeError):
        CSSCode(distance=3, Hx=hx, Hz=hz)

    # Distances don't match
    hz = np.array([[1, 1, 0]])
    with pytest.raises(InvalidCSSCodeError):
        CSSCode(distance=3, Hx=hx, Hz=hz, x_distance=4, z_distance=1)

    # Checks not over the same number of qubits
    hz = np.array([[1, 1]])
    with pytest.raises(InvalidCSSCodeError):
        CSSCode(distance=3, Hx=hx, Hz=hz)

    # Checks not provided
    with pytest.raises(InvalidCSSCodeError):
        CSSCode(distance=3)


@pytest.mark.parametrize("checks", ["steane_code_checks", "rep_code_checks", "rep_code_checks_reverse"])
def test_logicals(checks: tuple[npt.NDArray[np.int8] | None, npt.NDArray[np.int8] | None], request) -> None:  # type: ignore[no-untyped-def]
    """Test the logical operators of the CSSCode class."""
    hx, hz = request.getfixturevalue(checks)
    code = CSSCode(distance=3, Hx=hx, Hz=hz)
    assert code.Lx is not None
    assert code.Lz is not None
    assert code.Lx.shape[1] == code.Lz.shape[1] == code.n
    assert code.Lx.shape[0] == code.Lz.shape[0]

    # assert that logicals anticommute
    assert code.Lx @ code.Lz.T % 2 != 0

    # assert that logicals commute with stabilizers
    if code.Hz is not None:
        assert np.all(code.Lx @ code.Hz.T % 2 == 0)
    if code.Hx is not None:
        assert np.all(code.Lz @ code.Hx.T % 2 == 0)


def test_errors(steane_code_checks: tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]) -> None:
    """Test error detection and symdromes."""
    hx, hz = steane_code_checks
    code = CSSCode(distance=3, Hx=hx, Hz=hz)
    e1 = np.array([1, 0, 0, 0, 0, 0, 0])
    e2 = np.array([0, 1, 0, 0, 1, 0, 0])
    e3 = np.array([0, 0, 0, 0, 0, 1, 1])
    e4 = np.array([0, 1, 1, 1, 0, 0, 0])

    assert np.array_equal(code.get_x_syndrome(e1), code.get_z_syndrome(e2))
    assert np.array_equal(code.get_x_syndrome(e2), code.get_z_syndrome(e2))

    x_syndrome_1 = code.get_x_syndrome(e1)
    x_syndrome_2 = code.get_x_syndrome(e2)
    x_syndrome_3 = code.get_x_syndrome(e3)
    x_syndrome_4 = code.get_x_syndrome(e4)

    assert np.array_equal(x_syndrome_1, x_syndrome_2)
    assert not np.array_equal(x_syndrome_1, x_syndrome_3)
    assert np.array_equal(x_syndrome_1, x_syndrome_4)

    # e1 and e2 have same syndrome but if we add them we get a logical error
    assert code.check_if_logical_x_error((e1 + e2) % 2)
    assert code.check_if_logical_z_error((e1 + e2) % 2)
    assert not code.stabilizer_eq_x_error(e1, e2)
    assert not code.stabilizer_eq_z_error(e1, e2)

    # e1 and e4 on the other hand do not induce a logical error because they are stabilizer equivalent
    assert not code.check_if_logical_x_error((e1 + e4) % 2)
    assert not code.check_if_logical_z_error((e1 + e4) % 2)
    assert code.stabilizer_eq_x_error(e1, e4)
    assert code.stabilizer_eq_z_error(e1, e4)


def test_rep_code(rep_code_checks: tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]) -> None:
    """Test utility functions and correctness of the repetition code."""
    hx, hz = rep_code_checks
    code = CSSCode(distance=1, Hx=hx, Hz=hz)
    assert code.n == 3
    assert code.k == 1
    assert code.distance == 1
    assert not code.is_self_dual()

    e1 = np.array([1, 0, 0], dtype=np.int8)
    e2 = np.array([0, 1, 0], dtype=np.int8)
    e3 = np.array([0, 0, 1], dtype=np.int8)
    assert np.array_equal(code.get_x_syndrome(e1), np.array([1, 0]))
    assert np.array_equal(code.get_x_syndrome(e2), np.array([1, 1]))
    assert np.array_equal(code.get_x_syndrome(e3), np.array([0, 1]))

    assert code.get_z_syndrome(e1).size == 0

    assert code.check_if_logical_z_error((e1 + e2 + e3) % 2)
    assert not code.check_if_x_stabilizer((e1 + e2 + e3) % 2)
    assert code.check_if_x_stabilizer((e1 + e2) % 2)
    assert not code.check_if_z_stabilizer((e1 + e2 + e3) % 2)
    assert not code.check_if_z_stabilizer((e1 + e3) % 2)

    assert code.stabilizer_eq_x_error(e1, (e1 + e2 + e3) % 2)
    assert not code.stabilizer_eq_z_error(e1, (e1 + e2 + e3) % 2)
    assert code.stabilizer_eq_z_error(e1, e1)


def test_steane(steane_code_checks: tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]) -> None:
    """Test utility functions and correctness of the Steane code."""
    hx, hz = steane_code_checks
    code = CSSCode(distance=3, Hx=hx, Hz=hz)
    assert code.n == 7
    assert code.k == 1
    assert code.distance == 3
    assert code.is_self_dual()

    x_paulis = code.x_checks_as_pauli_strings()
    z_paulis = code.z_checks_as_pauli_strings()
    assert x_paulis is not None
    assert z_paulis is not None
    assert len(x_paulis) == len(z_paulis) == 3
    assert x_paulis == ["XXXXIII", "XIXIXIX", "IXXIXXI"]
    assert z_paulis == ["ZZZZIII", "ZIZIZIZ", "IZZIZZI"]

    hx_reordered = hx[::-1, :]
    code_reordered = CSSCode(distance=3, Hx=hx_reordered, Hz=hz)
    assert code == code_reordered


@pytest.mark.parametrize("n", [72, 90, 108, 144, 288])
def test_bb_codes(n: int) -> None:
    """Test that BB codes are constructed as valid CSS codes."""
    code = construct_bb_code(n)
    assert code.n == n
    assert code.Hx is not None
    assert code.Hz is not None
    assert np.all(code.Hx @ code.Hx.T % 2) == 0


def test_five_qubit_code(five_qubit_code_stabs: list[str]) -> None:
    """Test that the five qubit code is constructed as a valid stabilizer code."""
    z_logicals = ["ZZZZZ"]
    x_logicals = ["XXXXX"]

    # Many assertions are already made in the constructor
    code = StabilizerCode(five_qubit_code_stabs, distance=3, x_logicals=x_logicals, z_logicals=z_logicals)
    assert code.n == 5
    assert code.k == 1
    assert code.distance == 3

    error = "XIIII"
    syndrome = code.get_syndrome(error)
    assert np.array_equal(syndrome, np.array([0, 0, 0, 1]))

    stabilizer_eq_error = "IZZXI"
    assert code.stabilizer_equivalent(error, stabilizer_eq_error)

    different_error = "IZIII"
    assert not code.stabilizer_equivalent(error, different_error)

    strings = code.stabs_as_pauli_strings()
    assert strings == five_qubit_code_stabs


def test_stabilizer_sign() -> None:
    """Test that (negative) signs are correctly handled in stabilizer codes."""
    s = ["-ZZZZ", "-XXXX"]
    code = StabilizerCode(s)
    assert code.n == 4
    assert code.k == 2

    error = "XIII"
    syndrome = code.get_syndrome(error)
    assert np.array_equal(syndrome, np.array([1, 0]))


def test_trivial_code() -> None:
    """Test code with no stabilizers."""
    code = StabilizerCode.get_trivial_code(3)
    assert code.n == 3
    assert code.k == 3
    assert code.x_logicals == ["XII", "IXI", "IIX"]
    assert code.z_logicals == ["ZII", "IZI", "IIZ"]
    assert code.generators.n_rows == 0


def test_negative_distance() -> None:
    """Test that an error is raised if a negative distance is provided."""
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["ZZZZ", "XXXX"], distance=-1)


def test_different_length_stabilizers() -> None:
    """Test that an error is raised if stabilizers have different lengths."""
    with pytest.raises(InvalidPauliError):
        StabilizerCode(["ZZZZ", "X", "Y"])


def test_invalid_pauli_strings() -> None:
    """Test that invalid Pauli strings raise an error."""
    with pytest.raises(InvalidPauliError):
        StabilizerCode(["ABCD", "XIXI", "YIYI"])


def test_no_x_logical() -> None:
    """Test that an error is raised if no X logical is provided when a Z logical is provided."""
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["ZZZZ", "XXXX"], x_logicals=["XXII"])


def test_no_z_logical() -> None:
    """Test that an error is raised if no Z logical is provided when an X logical is provided."""
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["ZZZZ", "XXXX"], z_logicals=["ZZII"])


def test_logicals_wrong_length() -> None:
    """Test that an error is raised if the logicals have the wrong length."""
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["ZZZZ", "XXXX"], x_logicals=["XX"], z_logicals=["IZZI"])
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["ZZZZ", "XXXX"], x_logicals=["IXXI"], z_logicals=["ZZ"])


def test_commuting_logicals() -> None:
    """Test that an error is raised if the logicals commute."""
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["ZZZZ", "XXXX"], z_logicals=["ZZII"], x_logicals=["XXII"])


def test_anticommuting_logicals() -> None:
    """Test that an error is raised if the logicals anticommute with the stabilizer generators."""
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["ZZZZ", "XXXX"], z_logicals=["ZIII"], x_logicals=["IXXI"])
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["ZZZZ", "XXXX"], z_logicals=["IZZI"], x_logicals=["XIII"])


def test_too_many_logicals() -> None:
    """Test that an error is raised if too many logicals are provided."""
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["ZZZZ", "XXXX"], z_logicals=["ZZII", "ZZII", "ZZII"], x_logicals=["IXXI"])
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["ZZZZ", "XXXX"], z_logicals=["IZZI"], x_logicals=["XXII", "XXII", "XXII"])


def test_trivial_concatenation(five_qubit_code: StabilizerCode) -> None:
    """Test that the trivial concatenation of a code is the code itself."""
    inner_code = StabilizerCode.get_trivial_code(1)
    concatenated = ConcatenatedCode(five_qubit_code, inner_code)

    assert concatenated.n == 5
    assert concatenated.k == 1
    assert concatenated.distance == 3
    assert concatenated == five_qubit_code


def test_trivial_css_concatenation(steane_code: CSSCode) -> None:
    """Test that the trivial concatenation of a CSS code is the code itself."""
    inner_code = CSSCode.get_trivial_code(1)
    concatenated = ConcatenatedCSSCode(steane_code, inner_code)

    assert concatenated.n == 7
    assert concatenated.k == 1
    assert concatenated.distance == 3
    assert concatenated == steane_code


def test_hamming_code() -> None:
    """Test that the Hamming code is constructed as a valid CSS code."""
    code = construct_quantum_hamming_code(3)
    assert code.n == 7
    assert code.k == 1
    assert code.distance == 3


def test_many_hypercube_code_level_1() -> None:
    """Test that the many-hypercube code."""
    code = construct_many_hypercube_code(1)
    assert code.n == 6
    assert code.k == 4
    assert code.distance == 2
    iceberg = construct_iceberg_code(3)
    assert code == iceberg


def test_many_hypercube_code_level_2() -> None:
    """Test that the many-hypercube code."""
    code = construct_many_hypercube_code(2)
    assert code.n == 36
    assert code.k == 16
    assert code.distance == 4


def test_many_hypercube_code_level_3() -> None:
    """Test that the many-hypercube code."""
    code = construct_many_hypercube_code(3)
    assert code.n == 6**3
    assert code.k == 4**3
    assert code.distance == 2**3


def test_stabilizer_code_from_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test that a stabilizer code can be constructed from a file."""
    file_content = "IXXZZ\nZIXXZ\nZZIXX\nXZZIX"
    file_path = tmp_path / "test_file.txt"
    file_path.write_text(file_content)

    code = StabilizerCode.from_file(file_path)

    assert code.n == 5
    assert code.k == 1


def test_css_code_from_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test that a CSS code can be constructed from a file."""
    file_content = "XIIXXXI\nIXIIXXX\nIIXXIXX\nZIIZZZI\nIZIIZZZ\nIIZZIZZ"
    file_path = tmp_path / "test_file.txt"
    file_path.write_text(file_content)

    code = CSSCode.from_file(file_path)

    assert code.n == 7
    assert code.k == 1


def test_css_code_from_file_empty_line(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test that a CSS code can be constructed from a file."""
    file_content = "XIIXXXI\nIXIIXXX\n\nIIXXIXX\nZIIZZZI\nIZIIZZZ\nIIZZIZZ"
    file_path = tmp_path / "test_file.txt"
    file_path.write_text(file_content)

    code = CSSCode.from_file(file_path)

    assert code.n == 7
    assert code.k == 1
