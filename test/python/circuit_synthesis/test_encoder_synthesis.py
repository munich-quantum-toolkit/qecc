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

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import (
    depth_optimal_encoding_circuit,
    gate_optimal_encoding_circuit,
    heuristic_encoding_circuit,
)

from .utils import eq_span, get_stabs_css_with_indices, in_span

if TYPE_CHECKING:  # pragma: no cover
    from qiskit import QuantumCircuit


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


def _assert_correct_encoding_circuit(encoder: QuantumCircuit, encoding_qubits: list[int], code: CSSCode) -> None:
    assert encoder.num_qubits == code.n

    x_stabs, z_stabs_tmp, x_qubits, z_qubits = get_stabs_css_with_indices(encoder)

    # Since no gate is applied to the encoding qubits at the beginning of the circuit, the propagation of Z-logicals through the circuit can be read off from the Z-stabilizers.
    z_logical_indices = [z_qubits[i] for i in encoding_qubits]
    z_logicals = z_stabs_tmp[z_logical_indices]

    # To get propagation we need to apply a Hadamard to the encoding qubits and propagate again.
    encoder_h = encoder.inverse()
    encoder_h.h(encoding_qubits)
    encoder_h = encoder_h.inverse()
    x_stabs_tmp, z_stabs, x_qubits, _ = get_stabs_css_with_indices(encoder_h)
    print(x_stabs_tmp, x_qubits)
    x_logicals = x_stabs_tmp[[x_qubits[i] for i in encoding_qubits]]

    # assert correct propagation of stabilizers
    assert eq_span(code.Hx, x_stabs)
    assert eq_span(code.Hz, z_stabs)

    # assert correct propagation of logicals
    for logical in z_logicals:
        assert in_span(np.vstack((code.Hz, code.Lz)), logical)

    for logical in x_logicals:
        assert in_span(np.vstack((code.Hx, code.Lx)), logical)


@pytest.mark.parametrize(
    "code", ["steane_code", "css_4_2_2_code", "css_6_2_2_code", "tetrahedral", "surface_3", "hamming", "shor"]
)
def test_heuristic_encoding_consistent(code: CSSCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check that heuristic_encoding_circuit returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code)

    encoder, encoding_qubits = heuristic_encoding_circuit(code)
    assert encoder.num_qubits == code.n

    _assert_correct_encoding_circuit(encoder, encoding_qubits, code)


@pytest.mark.skipif(os.getenv("CI") is not None and sys.platform == "win32", reason="Too slow for CI on Windows")
@pytest.mark.parametrize("code", ["steane_code", "css_4_2_2_code", "css_6_2_2_code"])
def test_gate_optimal_encoding_consistent(code: CSSCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check that `gate_optimal_encoding_circuit` returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code)

    encoder, encoding_qubits = gate_optimal_encoding_circuit(code, max_timeout=5, min_gates=3, max_gates=10)
    assert encoder.num_qubits == code.n

    _assert_correct_encoding_circuit(encoder, encoding_qubits, code)


@pytest.mark.skipif(os.getenv("CI") is not None and sys.platform == "win32", reason="Too slow for CI on Windows")
@pytest.mark.parametrize("code", ["steane_code", "css_4_2_2_code", "css_6_2_2_code"])
def test_depth_optimal_encoding_consistent(code: CSSCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check that `gate_optimal_encoding_circuit` returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code)

    encoder, encoding_qubits = depth_optimal_encoding_circuit(code, max_timeout=5)
    assert encoder.num_qubits == code.n

    _assert_correct_encoding_circuit(encoder, encoding_qubits, code)
