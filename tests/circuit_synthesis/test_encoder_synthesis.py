# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test synthesis of encoding circuit synthesis."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.qecc import CSSCode, StabilizerCode
from mqt.qecc.circuit_synthesis import (
    depth_optimal_encoding_circuit,
    gate_optimal_encoding_circuit,
    gottesman_encoding_circuit,
    heuristic_encoding_circuit,
)
from mqt.qecc.codes.pauli import Pauli

from .utils import eq_span, in_span

if TYPE_CHECKING:  # pragma: no cover
    import stim

    from mqt.qecc.circuit_synthesis.circuits import CNOTCircuit


@pytest.fixture
def steane_code() -> CSSCode:
    """Return the Steane code."""
    return CSSCode.from_code_name("Steane")


@pytest.fixture
def surface_3() -> CSSCode:
    """Return the surface code."""
    return CSSCode.from_code_name("surface", 3)


@pytest.fixture
def tetrahedral() -> CSSCode:
    """Return the tetrahedral code."""
    return CSSCode.from_code_name("tetrahedral")


@pytest.fixture
def hamming() -> CSSCode:
    """Return the Hamming code."""
    return CSSCode.from_code_name("Hamming")


@pytest.fixture
def shor() -> CSSCode:
    """Return the Shor code."""
    return CSSCode.from_code_name("Shor")


@pytest.fixture
def css_4_2_2_code() -> CSSCode:
    """Return the 4,2,2  code."""
    return CSSCode(np.array([[1] * 4]), np.array([[1] * 4]), 2)


@pytest.fixture
def css_6_2_2_code() -> CSSCode:
    """Return the 4,2,2  code."""
    return CSSCode(
        np.array([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1]]), np.array([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1]]), 2
    )


@pytest.fixture
def non_css_5_qubit() -> StabilizerCode:
    """Return the 5-qubit code."""
    return StabilizerCode(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])


@pytest.fixture
def non_css_8_qubit() -> StabilizerCode:  # from https://arxiv.org/abs/quant-ph/9705052
    """Return a non-CSS 8-qubit code."""
    return StabilizerCode(["XXXXXXXX", "ZZZZZZZZ", "IXIXYZYZ", "IXZYIXZY", "IYXZXZIY"])


def _assert_correct_encoding_circuit_non_css(
    encoder: stim.Circuit, message_qs: list[int], code: StabilizerCode
) -> None:
    assert encoder.num_qubits == code.n
    assert len(message_qs) == code.k
    stabs = encoder.to_tableau().to_stabilizers()
    paulis = [Pauli.from_pauli_string(str(s)) for s in stabs]
    paulis = [pauli for i, pauli in enumerate(paulis) if i not in message_qs]

    circuit_code = StabilizerCode(paulis)
    assert code == circuit_code


def _assert_correct_encoding_circuit(encoder: CNOTCircuit, code: CSSCode) -> None:
    assert encoder.num_qubits() == code.n
    circuit_code = encoder.get_code()

    # assert correct propagation of stabilizers
    assert eq_span(code.Hx, circuit_code.Hx)
    assert eq_span(code.Hz, circuit_code.Hz)

    # assert correct propagation of logicals
    for logical in circuit_code.Lz:
        assert in_span(np.vstack((code.Hz, code.Lz)), logical)

    for logical in circuit_code.Lx:
        assert in_span(np.vstack((code.Hx, code.Lx)), logical)


@pytest.mark.parametrize(
    "code", ["steane_code", "css_4_2_2_code", "css_6_2_2_code", "tetrahedral", "surface_3", "hamming", "shor"]
)
def test_heuristic_encoding_consistent(code: CSSCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check that heuristic_encoding_circuit returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code)

    encoder = heuristic_encoding_circuit(code)
    encoder.get_uninitialized()
    assert encoder.num_qubits() == code.n

    _assert_correct_encoding_circuit(encoder, code)


@pytest.mark.skipif(
    os.getenv("CI") is not None and (sys.platform == "win32" or sys.platform == "darwin"),
    reason="Too slow for CI on Windows or MacOS",
)
@pytest.mark.parametrize("code", ["steane_code", "css_4_2_2_code", "css_6_2_2_code"])
def test_gate_optimal_encoding_consistent(code: CSSCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check that `gate_optimal_encoding_circuit` returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code)

    encoder = gate_optimal_encoding_circuit(code, max_timeout=8, min_gates=3, max_gates=10)
    assert encoder is not None
    encoder.get_uninitialized()
    assert encoder.num_qubits() == code.n

    _assert_correct_encoding_circuit(encoder, code)


@pytest.mark.skipif(
    os.getenv("CI") is not None and (sys.platform == "win32" or sys.platform == "darwin"),
    reason="Too slow for CI on Windows or MacOS",
)
@pytest.mark.parametrize("code", ["steane_code", "css_4_2_2_code", "css_6_2_2_code"])
def test_depth_optimal_encoding_consistent(code: CSSCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check that `gate_optimal_encoding_circuit` returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code)

    encoder = depth_optimal_encoding_circuit(code, max_timeout=5)
    assert encoder is not None
    encoder.get_uninitialized()
    assert encoder.num_qubits() == code.n

    _assert_correct_encoding_circuit(encoder, code)


@pytest.mark.parametrize("code", ["non_css_5_qubit", "non_css_8_qubit", "steane_code"])
def test_gottesman_encoding(code: StabilizerCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check that `gottesman_encoding_circuit` returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code)
    tab = code.generators
    encoder, message_qs = gottesman_encoding_circuit(tab)
    assert encoder is not None

    _assert_correct_encoding_circuit_non_css(encoder, message_qs, code)


def test_gottesman_encoding_invalid() -> None:
    """Check that `gottesman_encoding_circuit` fails for invalid stabilizers."""
    with pytest.raises(ValueError, match=r"Invalid tableau: could not find a valid pivot."):
        gottesman_encoding_circuit(["I"])

    with pytest.raises(ValueError, match=r"Invalid tableau: could not find a valid pivot."):
        gottesman_encoding_circuit(["X", "Z"])
