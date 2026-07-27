# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
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

from mqt.qecc import CSSCode
from mqt.qecc.codes import InvalidCSSCodeError

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt


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
def steane_code() -> CSSCode:
    """Return the Steane code."""
    hx = np.array([[1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 1, 0]])
    hz = hx
    return CSSCode(distance=3, Hx=hx, Hz=hz)


@pytest.fixture
def steane_code_checks() -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """Return the check matrices for the Steane code."""
    hx = np.array([[1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 1, 0]])
    hz = hx
    return hx, hz


def test_invalid_css_codes() -> None:
    """Test that an invalid CSS code raises an error."""
    hx = np.array([[1, 1, 1]])
    hz = np.array([[1, 0, 0]])
    with pytest.raises(InvalidCSSCodeError):
        CSSCode(distance=3, Hx=hx, Hz=hz)

    hz = np.array([[1, 1, 0]])
    with pytest.raises(InvalidCSSCodeError):
        CSSCode(distance=3, Hx=hx, Hz=hz, x_distance=4, z_distance=1)

    hz = np.array([[1, 1]])
    with pytest.raises(InvalidCSSCodeError):
        CSSCode(distance=3, Hx=hx, Hz=hz)

    with pytest.raises(InvalidCSSCodeError):
        CSSCode(distance=3)

    with pytest.raises(InvalidCSSCodeError, match="does not match check-matrix width"):
        CSSCode(Hx=hx, n=4)


def test_partial_logicals_rejected(steane_code_checks: tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]) -> None:
    """Test that providing only one of Lx/Lz raises an error."""
    hx, hz = steane_code_checks
    lx = np.array([[1, 1, 0, 0, 1, 0, 0]], dtype=np.int8)
    with pytest.raises(InvalidCSSCodeError, match="Both Lx and Lz must be provided together"):
        CSSCode(distance=3, Hx=hx, Hz=hz, Lx=lx)
    with pytest.raises(InvalidCSSCodeError, match="Both Lx and Lz must be provided together"):
        CSSCode(distance=3, Hx=hx, Hz=hz, Lz=lx)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.bool_])
def test_check_matrix_dtype_normalized(
    dtype: npt.DTypeLike, steane_code_checks: tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]
) -> None:
    """Any binary check-matrix dtype yields a consistent int8 code (regression test for #775).

    Non-int8 dtypes previously propagated into ``Hx``/``Hz`` and broke numba-dispatched synthesis
    (uint8/int16) or the boolean-``@`` orthogonality check (bool).
    """
    hx, hz = steane_code_checks
    code = CSSCode(distance=3, Hx=hx.astype(dtype), Hz=hz.astype(dtype))
    assert code.Hx.dtype == np.int8
    assert code.Hz.dtype == np.int8
    # Downstream matmul must still compute the correct mod-2 syndrome regardless of input dtype.
    error = np.zeros(code.n, dtype=np.int8)
    error[0] = 1
    np.testing.assert_array_equal(code.get_x_syndrome(error), hx[:, 0].astype(np.int8))


def test_orthogonality_check_rejects_non_orthogonal_bool() -> None:
    """Genuinely non-orthogonal boolean check matrices are still rejected (regression test for #775).

    A boolean ``@`` computes a logical product, so the mod-2 orthogonality test must widen the dtype
    to remain correct — without over-accepting non-orthogonal codes.
    """
    hx = np.array([[1, 1, 0]], dtype=np.bool_)
    hz = np.array([[1, 0, 0]], dtype=np.bool_)
    with pytest.raises(InvalidCSSCodeError, match="orthogonal"):
        CSSCode(distance=1, Hx=hx, Hz=hz)


@pytest.mark.parametrize("checks", ["steane_code_checks", "rep_code_checks", "rep_code_checks_reverse"])
def test_logicals(checks: str, request: pytest.FixtureRequest) -> None:
    """Test the logical operators of the CSSCode class."""
    hx, hz = request.getfixturevalue(checks)
    code = CSSCode(distance=3, Hx=hx, Hz=hz)
    assert code.Lx is not None
    assert code.Lz is not None
    assert code.Lx.shape[1] == code.Lz.shape[1] == code.n
    assert code.Lx.shape[0] == code.Lz.shape[0]

    assert np.array_equal(code.Lx @ code.Lz.T % 2, np.eye(code.Lx.shape[0], dtype=np.int8))

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

    assert code.check_if_logical_x_error((e1 + e2) % 2)
    assert code.check_if_logical_z_error((e1 + e2) % 2)
    assert not code.stabilizer_eq_x_error(e1, e2)
    assert not code.stabilizer_eq_z_error(e1, e2)

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

    x_logicals = code.x_logicals_as_pauli_strings()
    z_logicals = code.z_logicals_as_pauli_strings()
    assert x_logicals is not None
    assert z_logicals is not None
    assert len(x_logicals) == len(z_logicals) == 1
    assert x_logicals == ["XXIIXII"]
    assert z_logicals == ["ZZIIZII"]

    hx_reordered = hx[::-1, :]
    code_reordered = CSSCode(distance=3, Hx=hx_reordered, Hz=hz)
    assert code.is_equivalent(code_reordered)


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


def test_css_code_from_binary_matrix_space_separated(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test loading a CSS code from space-separated binary matrix format."""
    file_content = """1 1 1 1 0 0 0
1 0 1 0 1 0 1
0 1 1 0 1 1 0

1 1 1 1 0 0 0
1 0 1 0 1 0 1
0 1 1 0 1 1 0"""
    file_path = tmp_path / "test_css_binary_space.txt"
    file_path.write_text(file_content)

    code = CSSCode.from_file(file_path)

    assert code.n == 7
    assert code.Hx.shape == (3, 7)
    assert code.Hz.shape == (3, 7)
    assert np.array_equal(
        code.Hx, np.array([[1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 1, 0]], dtype=np.int8)
    )
    assert np.array_equal(
        code.Hz, np.array([[1, 1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 1, 0]], dtype=np.int8)
    )


def test_css_code_from_binary_matrix_comma_separated(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test loading a CSS code from comma-separated binary matrix format."""
    file_content = """1,1,1,1,0,0,0
1,0,1,0,1,0,1
0,1,1,0,1,1,0

1,1,1,1,0,0,0
1,0,1,0,1,0,1
0,1,1,0,1,1,0
"""
    file_path = tmp_path / "test_css_binary_comma.txt"
    file_path.write_text(file_content)

    code = CSSCode.from_file(file_path)

    assert code.n == 7
    assert code.Hx.shape == (3, 7)
    assert code.Hz.shape == (3, 7)


def test_css_code_from_binary_matrix_list_notation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test loading a CSS code from list notation binary matrix format."""
    file_content = """[[1,1,1,1,0,0,0],
[1,0,1,0,1,0,1],
[0,1,1,0,1,1,0]]

[[1,1,1,1,0,0,0],
[1,0,1,0,1,0,1],
[0,1,1,0,1,1,0]]"""
    file_path = tmp_path / "test_css_binary_list.txt"
    file_path.write_text(file_content)

    code = CSSCode.from_file(file_path)

    assert code.n == 7
    assert code.Hx.shape == (3, 7)
    assert code.Hz.shape == (3, 7)


def test_css_code_from_binary_matrix_numpy_notation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test loading a CSS code from numpy array notation format."""
    file_content = """[[1 1 1 1 0 0 0]
 [1 0 1 0 1 0 1]
 [0 1 1 0 1 1 0]]

[[1 1 1 1 0 0 0]
 [1 0 1 0 1 0 1]
 [0 1 1 0 1 1 0]]"""
    file_path = tmp_path / "test_css_binary_numpy.txt"
    file_path.write_text(file_content)

    code = CSSCode.from_file(file_path)

    assert code.n == 7
    assert code.Hx.shape == (3, 7)
    assert code.Hz.shape == (3, 7)


def test_css_code_from_binary_matrix_x_only(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test loading a CSS code with only X stabilizers from binary format."""
    file_content = """1,1,1,1,0,0,0
1,0,1,0,1,0,1
0,1,1,0,1,1,0"""
    file_path = tmp_path / "test_css_x_only.txt"
    file_path.write_text(file_content)

    code = CSSCode.from_file(file_path)

    assert code.n == 7
    assert code.Hx.shape == (3, 7)
    assert code.Hz.shape == (0, 7)


def test_css_code_from_binary_matrix_empty(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test that an error is raised for empty CSS code files."""
    file_content = ""
    file_path = tmp_path / "test_css_empty.txt"
    file_path.write_text(file_content)

    with pytest.raises(InvalidCSSCodeError, match="File is empty"):
        CSSCode.from_file(file_path)


def test_css_code_from_invalid_pauli_string(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Test that an error is raised for invalid Pauli strings in CSS code files."""
    file_content = "XIIX\nIIII\nZZII"
    file_path = tmp_path / "test_css_invalid.txt"
    file_path.write_text(file_content)

    with pytest.raises(InvalidCSSCodeError, match="Invalid stabilizer"):
        CSSCode.from_file(file_path)


def test_css_code_from_invalid_name() -> None:
    """Test that an error is raised for an invalid code name."""
    with pytest.raises(InvalidCSSCodeError, match="Unknown code name"):
        CSSCode.from_code_name("mqt-invalid-code")


def test_set_logicals():
    """Test that set_logicals correctly sets the logical operators."""
    h = np.array([[1, 1, 1, 1]])
    code = CSSCode(h, h)
    assert code.k == 2

    lxs = code.Lx.copy()
    lzs = code.Lz.copy()

    lxs[0] ^= lxs[1]
    lzs[1] ^= lzs[0]

    code.set_x_logicals(lxs)
    code.set_z_logicals(lzs)

    assert np.array_equal(code.Lx, lxs)
    assert np.array_equal(code.Lz, lzs)

    with pytest.raises(InvalidCSSCodeError, match="Number of logicals"):
        code.set_x_logicals(lxs[:1])

    with pytest.raises(InvalidCSSCodeError, match="Number of logicals"):
        code.set_z_logicals(lzs[:1])

    invalid = np.array([[1, 0, 0, 0], [1, 1, 0, 0]], dtype=np.int8)
    with pytest.raises(InvalidCSSCodeError, match="commute"):
        code.set_x_logicals(invalid)
    with pytest.raises(InvalidCSSCodeError, match="commute"):
        code.set_z_logicals(invalid)


def test_css_matrices_are_views_of_tableaus(steane_code: CSSCode) -> None:
    """CSS check and logical matrices share storage with their canonical tableaus."""
    assert np.shares_memory(steane_code.Hx, steane_code.generators.symplectic)
    assert np.shares_memory(steane_code.Hz, steane_code.generators.symplectic)
    assert np.shares_memory(steane_code.Lx, steane_code.x_logicals.symplectic)
    assert np.shares_memory(steane_code.Lz, steane_code.z_logicals.symplectic)


def test_trivial_code_syndromes_and_cosets() -> None:
    """Trivial codes (no checks) behave sensibly without special-case guards."""
    code = CSSCode.get_trivial_code(3)
    e = np.array([1, 0, 0], dtype=np.int8)
    zero = np.zeros(3, dtype=np.int8)
    assert code.get_x_syndrome(e).size == 0
    assert code.get_z_syndrome(e).size == 0
    assert code.check_if_x_stabilizer(zero)
    assert not code.check_if_x_stabilizer(e)
    assert code.check_if_z_stabilizer(zero)
    assert not code.check_if_z_stabilizer(e)
    assert code.stabilizer_eq_x_error(e, e)
    assert not code.stabilizer_eq_x_error(e, zero)
    assert code.check_if_logical_x_error(e)
    assert code.check_if_logical_z_error(e)
