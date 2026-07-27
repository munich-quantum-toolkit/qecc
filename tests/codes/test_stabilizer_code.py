# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests the StabilizerCode class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.qecc import StabilizerCode
from mqt.qecc.codes import InvalidStabilizerCodeError
from mqt.qecc.codes.core.pauli import (
    InvalidPauliError,
    Pauli,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def five_qubit_code_stabs() -> list[str]:
    """Return the five qubit code."""
    return ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]


@pytest.fixture
def five_qubit_code() -> StabilizerCode:
    """Return the five qubit code."""
    return StabilizerCode(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"], 3, z_logicals=["ZZZZZ"], x_logicals=["XXXXX"])


def test_five_qubit_code(five_qubit_code_stabs: list[str]) -> None:
    """Test that the five qubit code is constructed as a valid stabilizer code."""
    z_logicals = ["ZZZZZ"]
    x_logicals = ["XXXXX"]

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
    assert code.x_logicals.to_pauli_list() == [
        Pauli.from_pauli_string("XII"),
        Pauli.from_pauli_string("IXI"),
        Pauli.from_pauli_string("IIX"),
    ]
    assert code.z_logicals.to_pauli_list() == [
        Pauli.from_pauli_string("ZII"),
        Pauli.from_pauli_string("IZI"),
        Pauli.from_pauli_string("IIZ"),
    ]
    assert code.generators.n_rows == 0


def test_negative_distance() -> None:
    """Test that an error is raised if a negative distance is provided."""
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["ZZZZ", "XXXX"], distance=-1)


def test_mismatched_n() -> None:
    """Test that an error is raised if the explicit n does not match the generator width."""
    with pytest.raises(InvalidStabilizerCodeError, match="does not match generator width"):
        StabilizerCode(["ZZZZ", "XXXX"], n=5)


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
        StabilizerCode(["ZZZZ", "XXXX"], z_logicals=["XXII"])


def test_no_z_logical() -> None:
    """Test that an error is raised if no Z logical is provided when an X logical is provided."""
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["ZZZZ", "XXXX"], x_logicals=["ZZII"])


def test_not_compatible_logicals() -> None:
    """Test that an error is raised if no Z logical is provided when an X logical is provided."""
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode([], n=2, z_logicals=["ZI", "IZ"], x_logicals=["XX", "ZZ"])


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


def test_empty_logicals() -> None:
    """Test that an the case of empty logicals is handled gracefully."""
    code = StabilizerCode(["XX", "ZZ"], z_logicals=[], x_logicals=[])
    assert code.k == 0


def test_code_equality() -> None:
    """Test equality of stabilizer codes."""
    # Equal codes
    code1 = StabilizerCode(
        ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"],
        z_logicals=["ZZZZZ"],
        x_logicals=["XXXXX"],
    )
    code2 = StabilizerCode(
        ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"],
        z_logicals=["ZZZZZ"],
        x_logicals=["XXXXX"],
    )
    assert code1 == code2  # literally the same code

    # different basis of stabilizers, logicals automatically computed
    code3 = StabilizerCode(
        ["XZZXI", "IXZZX", "XIXZZ", "YXXYI"],
        z_logicals=["ZIXXI"],
        x_logicals=["XXXXX"],
    )

    assert code1.is_equivalent(code3)

    # Unequal codes (swapped logical operators)
    code4 = StabilizerCode(
        ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"],
        z_logicals=["XXXXX"],
        x_logicals=["ZZZZZ"],
    )
    assert code1 != code4


def test_logical_mapping() -> None:
    """Test the get_logical_mapping method of StabilizerCode."""
    # Define two equivalent codes
    code1 = StabilizerCode(
        ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"],
        z_logicals=["ZZZZZ"],
        x_logicals=["XXXXX"],
    )
    code2 = StabilizerCode(
        ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"],
        z_logicals=["ZZZZZ"],
        x_logicals=["XXXXX"],
    )
    mapping = code1.get_logical_mapping(code2)
    assert mapping == [0]

    code3 = StabilizerCode(["XXXX", "ZZZZ"], z_logicals=["ZZII", "IZZI"], x_logicals=["IXXI", "XXII"])
    code4 = StabilizerCode(["XXXX", "ZZZZ"], z_logicals=["IZZI", "ZZII"], x_logicals=["XXII", "IXXI"])

    mapping = code3.get_logical_mapping(code4)
    assert mapping == [1, 0]

    code_5 = StabilizerCode(  # first logical is product of logicals
        ["XXXX", "ZZZZ"], z_logicals=["ZIZI", "IZZI"], x_logicals=["IXXI", "XIXI"]
    )

    mapping = code3.get_logical_mapping(code_5)
    assert mapping is None


def test_stabilizer_code_from_file(tmp_path: Path) -> None:
    """Test that a stabilizer code can be constructed from a file."""
    file_content = "XZZXI\nIXZZX\nXIXZZ\nZXIXZ"
    file_path = tmp_path / "test_file.txt"
    file_path.write_text(file_content)

    code = StabilizerCode.from_file(file_path)

    assert code.n == 5
    assert code.k == 1


def test_stabilizer_code_from_binary_matrix_comma_separated(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test loading a stabilizer code from a comma-separated binary matrix file."""
    file_content = """1,0,0,1,0,0,1,1,0,0
0,1,0,0,1,0,0,1,1,0
1,0,1,0,0,0,0,0,1,1
0,1,0,1,0,1,0,0,0,1"""
    file_path = tmp_path / "test_binary_comma.txt"
    file_path.write_text(file_content)

    code = StabilizerCode.from_file(file_path)

    assert code.n == 5
    assert code.generators.n_rows == 4


def test_stabilizer_code_from_binary_matrix_correctness(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test that binary matrix loading produces the correct Pauli strings."""
    file_content = """1 1 1 1"""
    file_path = tmp_path / "test_binary_correctness.txt"
    file_path.write_text(file_content)

    code = StabilizerCode.from_file(file_path)

    assert code.n == 2
    stabs = code.stabs_as_pauli_strings()
    assert stabs == ["YY"]


def test_stabilizer_code_from_binary_matrix_five_qubit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test loading the five-qubit code from binary matrix format."""
    file_content = """1 0 0 1 0 0 1 1 0 0
    0 1 0 0 1 0 0 1 1 0
    1 0 1 0 0 0 0 0 1 1
    0 1 0 1 0 1 0 0 0 1"""
    file_path = tmp_path / "test_five_qubit.txt"
    file_path.write_text(file_content)

    code = StabilizerCode.from_file(file_path)
    print(code.stabs_as_pauli_strings())

    assert code.n == 5
    assert code.generators.n_rows == 4


def test_stabilizer_code_from_binary_matrix_invalid_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test that an error is raised for binary matrices with odd number of rows."""
    file_content = """1 1 0 0
0 0 1 0"""
    file_path = tmp_path / "test_binary_invalid.txt"
    file_path.write_text(file_content)

    with pytest.raises(InvalidStabilizerCodeError, match=r"Stabilizer generators must commute with each other."):
        StabilizerCode.from_file(file_path)


def test_stabilizer_code_from_binary_matrix_empty(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test that an error is raised for empty binary matrix files."""
    file_content = ""
    file_path = tmp_path / "test_binary_empty.txt"
    file_path.write_text(file_content)

    with pytest.raises(InvalidStabilizerCodeError, match="File is empty"):
        StabilizerCode.from_file(file_path)


def test_stabilizer_code_to_tableau(five_qubit_code: StabilizerCode) -> None:
    """Test the to_tableau method of StabilizerCode."""
    tableau = five_qubit_code.to_tableau()

    assert tableau.n == 5
    assert tableau.num_rows() == 6

    x_log_pauli = tableau[0]
    z_log_pauli = tableau[1]
    assert not x_log_pauli.commute(z_log_pauli)

    for i in range(2, 6):
        stab = tableau[i]
        assert x_log_pauli.commute(stab)
        assert z_log_pauli.commute(stab)


def test_stabilizer_code_to_tableau_trivial() -> None:
    """Test the to_tableau method for trivial code."""
    code = StabilizerCode.get_trivial_code(3)
    tableau = code.to_tableau()

    assert tableau.n == 3
    assert tableau.num_rows() == 6

    for i in range(3):
        x_log = tableau[i]
        z_log = tableau[3 + i]
        assert not x_log.commute(z_log)


def test_compute_logical_steane():
    """Test that compute_logical correctly finds logicals for Steane code."""
    stabs = ["XXXXIII", "XIXIXIX", "IXXIXXI", "ZZZZIII", "ZIZIZIZ", "IZZIZZI"]
    code = StabilizerCode(stabs, distance=3)

    assert code.k == 1
    assert code.z_logicals is not None
    assert code.x_logicals is not None
    assert code.z_logicals.n_rows == 1
    assert code.x_logicals.n_rows == 1

    assert not code.z_logicals[0].commute(code.x_logicals[0])


def test_compute_logical_multiple_pairs():
    """Test compute_logical with k>1."""
    stabs = ["XXXX", "ZZZZ"]
    code = StabilizerCode(stabs, distance=2)

    assert code.k == 2
    assert code.z_logicals.n_rows == 2
    assert code.x_logicals.n_rows == 2


def test_independent_of_generator_signs() -> None:
    """Test that the stabilizer group is independent of the signs of the generators.

    Note: Depending on the demand of a sign-aware matrix multiplication, this test can be removed.
    """
    plus = StabilizerCode(["XXXX", "ZZZZ"])
    minus = StabilizerCode(["-XXXX", "-ZZZZ"])
    assert plus.k == minus.k == 2
    assert not plus.equal_stabilizer_group(minus)


def test_rejects_non_hermitian_stabilizer() -> None:
    """Test that an error is raised if a non-Hermitian stabilizer is provided."""
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["+iX"])


def test_rejects_explicit_minus_identity() -> None:
    """Test that an error is raised if the identity operator is given a negative sign."""
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["-I"])


def test_rejects_generators_that_generate_minus_identity() -> None:
    """Test that an error is raised if the generators together generate the negative identity."""
    with pytest.raises(InvalidStabilizerCodeError):
        StabilizerCode(["X", "-X"])


def test_accepts_redundant_consistent_generators() -> None:
    """Test that redundant generators that are consistent are accepted."""
    code = StabilizerCode(["XI", "IX", "XX"])
    assert code.generators.n_rows == 2


def test_inequality_due_to_incompatible_codes() -> None:
    """Test that the stabilizer group equality depends on the number of qubits."""
    c1 = StabilizerCode(["XXXX", "ZZZZ"])
    c2 = StabilizerCode(["XX", "ZZ"])
    assert c1.n == 4
    assert c2.n == 2
    assert c1.k == 2
    assert c2.k == 0
    assert not c1.equal_stabilizer_group(c2)
    assert not c1.equal_logical_basis(c2)
    assert c1.get_logical_mapping(c2) is None


def test_logical_operator_checks() -> None:
    """Test checks for logical and stabilizer operators."""
    c1 = StabilizerCode(["XXXX", "ZZZZ"])
    p_str = "XIII"
    p = Pauli.from_pauli_string(p_str)

    assert not c1.is_z_logical(p)
    assert not c1.is_z_logical(p_str)
    assert not c1.is_x_logical(p)
    assert not c1.is_x_logical(p_str)
    assert not c1.is_logical(p)
    assert not c1.is_logical(p_str)

    r_str = "ZZII"
    r = Pauli.from_pauli_string(r_str)
    assert not c1.is_stabilizer(r)
    assert not c1.is_stabilizer(r_str)


def test_string_representation() -> None:
    """Test the human-readable string representation of a stabilizer code."""
    code = StabilizerCode(["ZZ"], distance=1, z_logicals=["ZI"], x_logicals=["XX"])
    assert str(code) == (
        "Stabilizer Code: n=2, k=1, distance=1\n"
        "Stabilizer Generators:\n"
        "  ZZ\n"
        "Logical Z operators:\n"
        "  ZI\n"
        "Logical X operators:\n"
        "  XX"
    )
