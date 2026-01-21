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
import stim

from mqt.qecc import CSSCode, StabilizerCode
from mqt.qecc.circuit_synthesis import (
    depth_optimal_encoding_circuit,
    depth_optimal_encoding_circuit_non_css,
    gate_optimal_encoding_circuit,
    gottesman_encoding_circuit,
    heuristic_encoding_circuit,
)
from mqt.qecc.circuit_synthesis.circuit_utils import num_two_qubit_gates
from mqt.qecc.circuit_synthesis.encoding import (
    fix_tableau_signs_in_place,
    greedy_adapted_volanto,
    lookahead_volanto,
    reduce_single_qubit_gates_and_swaps,
    resynthesize_stim_circuit_with_volanto,
)
from mqt.qecc.codes.pauli import Pauli, StabilizerTableau

from .utils import eq_span, in_span

if TYPE_CHECKING:  # pragma: no cover
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
@pytest.mark.parametrize("code", ["css_4_2_2_code"])
def test_gate_optimal_encoding_consistent(code: CSSCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check that `gate_optimal_encoding_circuit` returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code)

    encoder = gate_optimal_encoding_circuit(code, max_timeout=1, min_gates=3, max_gates=10)
    assert encoder is not None
    encoder.get_uninitialized()
    assert encoder.num_qubits() == code.n

    _assert_correct_encoding_circuit(encoder, code)


@pytest.mark.skipif(
    os.getenv("CI") is not None and (sys.platform == "win32" or sys.platform == "darwin"),
    reason="Too slow for CI on Windows or MacOS",
)
@pytest.mark.parametrize("code", ["css_4_2_2_code"])
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


@pytest.mark.parametrize("code", ["non_css_5_qubit"])
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
    result = depth_optimal_encoding_circuit_non_css(code, max_depth=1)
    assert result == "UNSAT"


@pytest.mark.parametrize(
    ("tableau", "expected_transvections", "min_single_qubit_ops", "expected_swaps"),
    [
        # Add test cases here with StabilizerTableau instances
        # Example: (StabilizerTableau.from_matrix(np.eye(8, dtype=np.int8)), 0, 0, 0),  # Identity
        (StabilizerTableau.from_matrix(np.eye(8, dtype=np.int8)), 0, 0, 0),  # Identity
        (StabilizerTableau.from_stim_circuit(stim.Circuit("CX 0 1")), 1, 0, 0),  # Single CNOT
        (StabilizerTableau.from_stim_circuit(stim.Circuit("H 0\nCX 0 1")), 1, 1, 0),  # Bell pair
    ],
)
def test_volanto_cases(
    tableau: StabilizerTableau, expected_transvections: int, min_single_qubit_ops: int, expected_swaps: int
):
    """Test greedy adapted Volanto on various cases."""
    transvections, final_u = greedy_adapted_volanto(tableau.tableau.matrix, use_all_pairs=True)
    prefix, final_u = reduce_single_qubit_gates_and_swaps(final_u)
    single_qubit_ops = prefix[1]
    swaps = prefix[0]

    assert len(transvections) == expected_transvections
    assert len(single_qubit_ops) >= min_single_qubit_ops
    assert len(swaps) == expected_swaps
    assert np.array_equal(final_u, np.eye(2 * tableau.n, dtype=np.int8))


@pytest.mark.parametrize(
    "tableau",
    [
        StabilizerTableau.from_matrix(np.eye(8, dtype=np.int8)),  # Identity
        StabilizerTableau.from_stim_circuit(stim.Circuit("CX 0 1")),  # Single CNOT
        StabilizerTableau.from_stim_circuit(stim.Circuit("H 0\nCX 0 1")),  # Bell pair
    ],
)
@pytest.mark.parametrize("lookahead_depth", [0, 1, 2])
def test_lookahead_volanto_cases(
    tableau: StabilizerTableau, lookahead_depth: int
):
    """Test lookahead Volanto synthesis on various cases."""
    transvections, final_u = lookahead_volanto(tableau.tableau.matrix, lookahead_depth=lookahead_depth, use_all_pairs=True)
    prefix, final_u = reduce_single_qubit_gates_and_swaps(final_u)
    single_qubit_ops = prefix[1]
    swaps = prefix[0]

    # Define expected results based on tableau and lookahead_depth
    if np.array_equal(tableau.tableau.matrix, np.eye(8, dtype=np.int8)):
        expected_transvections, min_single_qubit_ops, expected_swaps = 0, 0, 0
    elif tableau == StabilizerTableau.from_stim_circuit(stim.Circuit("CX 0 1")):
        expected_transvections, min_single_qubit_ops, expected_swaps = 1, 0, 0
    elif tableau == StabilizerTableau.from_stim_circuit(stim.Circuit("H 0\nCX 0 1")):
        expected_transvections, min_single_qubit_ops, expected_swaps = 1, 1, 0

    assert len(transvections) == expected_transvections
    assert len(single_qubit_ops) >= min_single_qubit_ops
    assert len(swaps) == expected_swaps
    assert np.array_equal(final_u, np.eye(2 * tableau.n, dtype=np.int8))
    
@pytest.mark.parametrize(
    "circuit",
    [
        stim.Circuit(),
        stim.Circuit("H 0"),
        stim.Circuit("CX 0 1"),
        stim.Circuit("H 0\nCX 0 1"),
        stim.Circuit("H 0\nCX 0 1\nH 1\nCX 1 2"),  # Simple circuit with H and CX gates
        stim.Circuit("H 0\nCX 0 1\nCX 1 2\nCX 2 3\nH 3"),  # Circuit with more gates
        stim.Circuit("H 0\nCX 0 1\nCX 1 2\nH 2\nCX 2 3\nH 3"),  # Circuit with interleaved H and CX gates
    ],
)
@pytest.mark.parametrize("lookahead_depth", [0,1,2])
def test_resynthesize_stim_circuit_with_volanto(circuit: stim.Circuit, lookahead_depth: int) -> None:
    """Test that resynthesized circuit has the same tableau and fewer or equal two-qubit gates."""
    original_tableau = circuit.to_tableau()
    original_two_qubit_gates = num_two_qubit_gates(circuit)

    resynthesized_circuit = resynthesize_stim_circuit_with_volanto(
        circuit, lookahead_depth=lookahead_depth, fix_signs=True
    )
    resynthesized_tableau = resynthesized_circuit.to_tableau()
    resynthesized_two_qubit_gates = num_two_qubit_gates(resynthesized_circuit)

    # Assert that the tableaus are identical
    assert original_tableau == resynthesized_tableau

    # Assert that the resynthesized circuit has fewer or equal two-qubit gates
    assert resynthesized_two_qubit_gates <= original_two_qubit_gates

    
@pytest.mark.parametrize(
    "circuit, target_phase",
    [
        (
            stim.Circuit("H 0\nCX 0 1\nH 1\nCX 1 2"),  # Simple circuit
            np.array([0, 1, 0, 1, 0, 1]),  # Target phase
        ),
        (
            stim.Circuit("H 0\nCX 0 1\nCX 1 2\nCX 2 3\nH 3"),  # Circuit with more gates
            np.array([1, 0, 1, 0, 0, 1, 0, 1]),  # Target phase
        ),
    ],
)
def test_fix_tableau_signs_in_place(circuit: stim.Circuit, target_phase: np.ndarray):
    """Test that fix_tableau_signs_in_place correctly adjusts tableau signs."""
    # Apply the fix_tableau_signs_in_place function
    fix_tableau_signs_in_place(circuit, target_phase)

    # Convert circuit to StabilizerTableau
    tableau = StabilizerTableau.from_stim_circuit(circuit)

    # Assert that the phase is correctly updated
    assert np.array_equal(tableau.phase, target_phase)
