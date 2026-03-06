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

import numpy as np
import pytest
import stim

from mqt.qecc.circuit_synthesis import (
    depth_optimal_encoding_circuit,
    depth_optimal_encoding_circuit_non_css,
    gate_optimal_encoding_circuit,
    gottesman_encoding_circuit,
)
from mqt.qecc.circuit_synthesis.circuit_utils import num_two_qubit_gates
from mqt.qecc.circuit_synthesis.circuits import CNOTCircuit
from mqt.qecc.circuit_synthesis.encoding import (
    encoder_from_stabilizers_and_logicals,
    resynthesize_stim_circuit,
    synthesize_encoding_circuit,
)
from mqt.qecc.circuit_synthesis.synthesis import SynthesisConfig
from mqt.qecc.codes import CSSCode, SquareOctagonColorCode, StabilizerCode
from mqt.qecc.codes.pauli import Pauli, StabilizerTableau

from .utils import eq_span, in_span


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


@pytest.fixture
def cnot_synthesis_config() -> SynthesisConfig:
    """Fixture to create a CNOT synthesis configuration."""
    return SynthesisConfig(
        optimization_criterion="gates",
        lookahead=0,
        num_lookahead_candidates=10,
        enable_early_termination=False,
    )


@pytest.fixture
def clifford_synthesis_config() -> SynthesisConfig:
    """Fixture to create a Clifford synthesis configuration."""
    return SynthesisConfig(
        optimization_criterion="gates",
        lookahead=0,
        num_lookahead_candidates=10,
        enable_early_termination=False,
    )


def _assert_correct_encoding_circuit_non_css(
    encoder: stim.Circuit, message_qs: dict[int, int], code: StabilizerCode
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
    "code_fixture", ["steane_code", "css_4_2_2_code", "css_6_2_2_code", "tetrahedral", "surface_3", "hamming", "shor"]
)
def test_heuristic_encoding_consistent(code_fixture: str, request: pytest.FixtureRequest) -> None:
    """Check that heuristic_encoding_circuit returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code_fixture)

    encoder = synthesize_encoding_circuit(code)
    encoder.get_uninitialized()
    assert encoder.num_qubits() == code.n

    assert isinstance(encoder, CNOTCircuit)
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
    encoder = gottesman_encoding_circuit(tab)
    assert encoder is not None

    _assert_correct_encoding_circuit_non_css(encoder.to_stim_circuit(), encoder.inputs(), code)


def test_gottesman_encoding_invalid() -> None:
    """Check that `gottesman_encoding_circuit` fails for invalid stabilizers."""
    with pytest.raises(ValueError, match=r"Invalid tableau: could not find a valid pivot."):
        gottesman_encoding_circuit(["I"])

    with pytest.raises(ValueError, match=r"Invalid tableau: could not find a valid pivot."):
        gottesman_encoding_circuit(["X", "Z"])


@pytest.mark.parametrize("code_fixture", ["non_css_5_qubit"])
def test_depth_optimal_encoding_non_css_consistent(code_fixture: str, request) -> None:  # type: ignore[no-untyped-def]
    """Check that `depth_optimal_encoding_circuit_non_css` returns a valid circuit with the correct stabilizers."""
    code = request.getfixturevalue(code_fixture)
    result = depth_optimal_encoding_circuit_non_css(code, max_depth=10)
    assert result != "UNSAT"
    assert not isinstance(result, str)
    encoder = result
    assert encoder.to_stim_circuit().num_qubits == code.n

    # Assert correct propagation of stabilizers and logicals
    circuit_code = encoder.get_code()
    print(circuit_code.stabs_as_pauli_strings())
    assert code == circuit_code


@pytest.mark.parametrize("code", ["non_css_5_qubit"])
def test_depth_optimal_encoding_non_css_edge_cases(code: StabilizerCode, request) -> None:  # type: ignore[no-untyped-def]
    """Check edge cases for `depth_optimal_encoding_circuit_non_css`."""
    code = request.getfixturevalue(code)

    # Test with minimal depth
    result = depth_optimal_encoding_circuit_non_css(code, max_depth=1)
    assert result == "UNSAT"


@pytest.mark.parametrize(
    "circuit",
    [
        stim.Circuit(),
        stim.Circuit("H 0"),
        stim.Circuit("CX 0 1"),
        stim.Circuit("H 0\nCX 0 1"),
        stim.Circuit("H 0\nCX 0 1\nH 1\nCX 1 2"),
        stim.Circuit("H 0\nCX 0 1\nCX 1 2\nCX 2 3\nH 3"),
        stim.Circuit("H 0\nCX 0 1\nCX 1 2\nH 2\nCX 2 3\nH 3"),
    ],
)
@pytest.mark.parametrize("lookahead_depth", [0, 1, 2])
def test_resynthesize_stim_circuit(
    circuit: stim.Circuit, lookahead_depth: int, clifford_synthesis_config: SynthesisConfig
) -> None:
    """Test that resynthesized circuit has the same tableau and fewer or equal two-qubit gates."""
    original_tableau = circuit.to_tableau()
    original_two_qubit_gates = num_two_qubit_gates(circuit)
    clifford_synthesis_config.lookahead = lookahead_depth
    clifford_synthesis_config.num_lookahead_candidates = 5

    resynthesized_circuit = resynthesize_stim_circuit(circuit, config=clifford_synthesis_config, use_cnots_if_css=False)
    resynthesized_tableau = resynthesized_circuit.to_tableau()
    resynthesized_two_qubit_gates = num_two_qubit_gates(resynthesized_circuit)

    curr = circuit.copy()
    for gate in resynthesized_circuit.inverse():
        curr.append(gate)

    assert original_tableau == resynthesized_tableau

    assert resynthesized_two_qubit_gates <= original_two_qubit_gates


def test_encoder_from_stabilizers_and_logicals(clifford_synthesis_config: SynthesisConfig) -> None:
    """Test encoder_from_stabilizers_and_logicals function."""
    stabilizers = StabilizerTableau.from_pauli_strings(["ZZZZ", "XXXX"])
    logicals = StabilizerTableau.from_pauli_strings(["XXII", "IXXI", "IZZI", "ZZII"])

    iso = encoder_from_stabilizers_and_logicals(stabilizers, logicals, config=clifford_synthesis_config)
    tab = StabilizerTableau.from_stim_circuit(iso.to_stim_circuit())
    for stab in stabilizers:
        tab.is_row(Pauli.from_pauli_string(str(stab)))
    for logical in logicals:
        assert tab.is_row(Pauli.from_pauli_string(str(logical)))


def test_encoder_from_stabilizers_and_logicals_five_qubit(clifford_synthesis_config: SynthesisConfig) -> None:
    """Test encoder_from_stabilizers_and_logicals function."""
    stabilizers = StabilizerTableau.from_pauli_strings(["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"])
    x_logicals = ["XXXXX"]
    z_logicals = ["ZZZZZ"]
    logicals = StabilizerTableau.from_pauli_strings(x_logicals + z_logicals)

    iso = encoder_from_stabilizers_and_logicals(stabilizers, logicals, config=clifford_synthesis_config)
    StabilizerTableau.from_stim_circuit(iso.to_stim_circuit())

    StabilizerTableau.from_stim_circuit(iso.to_stim_circuit())
    code = iso.get_code()
    for stab in stabilizers:
        code.is_stabilizer(stab)

    for expected_log in x_logicals:
        is_expected_logical = False
        for circuit_log in code.x_logicals:
            if code.stabilizer_equivalent(expected_log, circuit_log):
                is_expected_logical = True
                break
        assert is_expected_logical, f"Expected logical {expected_log} not found in code logicals."

    for expected_log in z_logicals:
        is_expected_logical = False
        for circuit_log in code.z_logicals:
            if code.stabilizer_equivalent(expected_log, circuit_log):
                is_expected_logical = True
                break
        assert is_expected_logical, f"Expected logical {expected_log} not found in code logicals."


def test_encoder_from_stabilizers_and_logicals_gottesman() -> None:
    """Test encoder_from_stabilizers_and_logicals function."""
    stabilizers = StabilizerTableau.from_pauli_strings(["XXXXXXXX", "ZZZZZZZZ", "IXIXYZYZ", "IXZYIXZY", "IYXZXZIY"])
    x_logicals = ["XXIIIZIZ", "XIXZIIZI", "XIIZXZII"]
    z_logicals = ["IZIZIZIZ", "IIZZIIZZ", "IIIIZZZZ"]
    logicals_tab = StabilizerTableau.from_pauli_strings(x_logicals + z_logicals)

    iso = encoder_from_stabilizers_and_logicals(stabilizers, logicals_tab)
    StabilizerTableau.from_stim_circuit(iso.to_stim_circuit())
    code = iso.get_code()
    for stab in stabilizers:
        code.is_stabilizer(stab)

    for expected_log in x_logicals:
        is_expected_logical = False
        for circuit_log in code.x_logicals:
            if code.stabilizer_equivalent(expected_log, circuit_log):
                is_expected_logical = True
                break
        assert is_expected_logical, f"Expected logical {expected_log} not found in code logicals."

    for expected_log in z_logicals:
        is_expected_logical = False
        for circuit_log in code.z_logicals:
            if code.stabilizer_equivalent(expected_log, circuit_log):
                is_expected_logical = True
                break
        assert is_expected_logical, f"Expected logical {expected_log} not found in code logicals."


def test_cc_4_8_8():
    """Test encoding circuit synthesis for the 4.8.8 color code."""
    code = SquareOctagonColorCode(5)
    config = SynthesisConfig(
        optimization_criterion="gates",
        lookahead=1,
        num_lookahead_candidates=3,
        enable_early_termination=False,
    )
    enc = synthesize_encoding_circuit(code, config=config)
    assert isinstance(enc, CNOTCircuit)
    _assert_correct_encoding_circuit(enc, code)
