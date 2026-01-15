# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
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
    depth_optimal_encoding_circuit_non_css,
    gate_optimal_encoding_circuit,
    gottesman_encoding_circuit,
    heuristic_encoding_circuit,
)
from mqt.qecc.circuit_synthesis.encoding import (
    greedy_adapted_volanto,
    reduce_with_single_qubit_gates,
    stim_circuit_to_symplectic,
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
    stabs = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    x_logicals = ["XXXXX"]
    z_logicals = ["ZZZZZ"]
    return StabilizerCode(stabs, x_logicals=x_logicals, z_logicals=z_logicals)


@pytest.fixture
def non_css_8_qubit() -> StabilizerCode:  # from https://arxiv.org/abs/quant-ph/9705052
    """Return a non-CSS 8-qubit code."""
    stabs = ["XXXXXXXX", "ZZZZZZZZ", "IXIXYZYZ", "IXZYIXZY", "IYXZXZIY"]
    x_logicals = ["XXIIIZIZ", "XIXZIIZI", "XIIZXZII"]
    z_logicals = ["IZIZIZIZ", "IIZZIIZZ", "IIIIZZZZ"]
    return StabilizerCode(stabs, x_logicals=x_logicals, z_logicals=z_logicals)


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


@pytest.mark.parametrize("code", ["non_css_5_qubit", "non_css_8_qubit"])
def test_depth_optimal_encoding_non_css_consistent(code: StabilizerCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check that `depth_optimal_encoding_circuit_non_css` returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code)

    encoder, message_qs = depth_optimal_encoding_circuit_non_css(code, max_depth=10)
    assert encoder is not None
    assert encoder.num_qubits == code.n

    # Assert correct propagation of stabilizers and logicals
    stabs = encoder.to_tableau().to_stabilizers()
    paulis = [str(s) for s in stabs]
    paulis = [pauli for i, pauli in enumerate(paulis) if i not in message_qs]

    circuit_code = StabilizerCode(paulis)
    assert code == circuit_code


@pytest.mark.parametrize("code", ["non_css_5_qubit"])
def test_depth_optimal_encoding_non_css_edge_cases(code: StabilizerCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check edge cases for `depth_optimal_encoding_circuit_non_css`."""
    code = request.getfixturevalue(code)

    # Test with minimal depth
    encoder, message_qs = depth_optimal_encoding_circuit_non_css(code, max_depth=1)
    assert encoder is not None
    assert encoder.num_qubits == code.n

    # Test with maximal depth
    encoder, _message_qs = depth_optimal_encoding_circuit_non_css(code, max_depth=20)
    assert encoder is not None
    assert encoder.num_qubits == code.n


@pytest.mark.skipif(
    os.getenv("CI") is not None and (sys.platform == "win32" or sys.platform == "darwin"),
    reason="Too slow for CI on Windows or MacOS",
)
@pytest.mark.parametrize("code", ["non_css_5_qubit", "non_css_8_qubit"])
def test_depth_optimal_encoding_non_css_timeout(code: StabilizerCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check that `depth_optimal_encoding_circuit_non_css` respects timeout constraints."""
    code = request.getfixturevalue(code)

    # Test with a short timeout
    encoder = depth_optimal_encoding_circuit_non_css(code, max_depth=10, max_two_qubit_gates=5)
    assert encoder is not None


@pytest.mark.parametrize(
    "n, operations, expected_transvections, expected_single_qubit_ops",
    [
        (4, [], 0, 0),  # Identity
        (2, [(0, 1, "CNOT")], 1, 0),  # CNOT
        (1, [(0, 1, "HADAMARD")], 0, 1),  # Hadamard
        (2, [(1, 3, "HADAMARD"), (0, 1, "CNOT"), (3, 2, "CNOT")], 1, 0),  # Bell state
        (2, [(0, 1, "CNOT"), (3, 2, "CNOT"), (1, 0, "CNOT"), (2, 3, "CNOT")], 1, 1),  # SWAP
        (3, [(0, 3, "HADAMARD"), (1, 0, "CNOT"), (3, 4, "CNOT"), (2, 1, "CNOT"), (4, 5, "CNOT")], 2, 0),  # GHZ
    ],
)
def test_volanto_cases(n, operations, expected_transvections, expected_single_qubit_ops):
    """Test greedy adapted Volanto on various cases."""
    U = np.eye(2 * n, dtype=np.int8)

    for op in operations:
        if op[2] == "CNOT":
            U[:, op[0]] ^= U[:, op[1]]
            U[:, n + op[1]] ^= U[:, n + op[0]]
        elif op[2] == "HADAMARD":
            U[:, [op[0], op[1]]] = U[:, [op[1], op[0]]]

    transvections, final_U = greedy_adapted_volanto(U, use_all_pairs=True)
    single_qubit_ops, final_U = reduce_with_single_qubit_gates(final_U)

    assert len(transvections) == expected_transvections
    assert len(single_qubit_ops[0]) == expected_single_qubit_ops
    assert np.array_equal(final_U, np.eye(2 * n, dtype=np.int8))


def test_stim_circuit_to_symplectic_empty() -> None:
    """Test conversion from stim circuit to symplectic matrix."""
    import stim

    circuit = stim.Circuit()

    U = stim_circuit_to_symplectic(circuit)
    n = circuit.num_qubits
    expected_U = np.eye(2 * n, dtype=np.int8)
    assert np.array_equal(U, expected_U)


def test_stim_circuit_to_symplectic_cnot() -> None:
    """Test conversion from stim circuit to symplectic matrix."""
    import stim

    circuit = stim.Circuit()
    circuit.append_operation("CNOT", [0, 1])

    U = stim_circuit_to_symplectic(circuit)
    n = circuit.num_qubits
    expected_U = np.eye(2 * n, dtype=np.int8)
    expected_U[:, 0] ^= expected_U[:, 1]  # CNOT from qubit 1 to qubit 0
    expected_U[:, n + 1] ^= expected_U[:, n]

    assert np.array_equal(U, expected_U)


def test_stim_circuit_to_symplectic_hadamard() -> None:
    """Test conversion from stim circuit to symplectic matrix."""
    import stim

    circuit = stim.Circuit()
    circuit.append_operation("H", [0])

    U = stim_circuit_to_symplectic(circuit)
    n = circuit.num_qubits
    expected_U = np.eye(2 * n, dtype=np.int8)
    expected_U[:, [0, 1]] = expected_U[:, [1, 0]]  # Hadamard on qubit 0

    assert np.array_equal(U, expected_U)
