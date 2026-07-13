# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for gate operation classes."""

from __future__ import annotations

import numpy as np
import pytest
import z3

from mqt.qecc.circuit_synthesis.exact.gate_operations import (
    CNOTGate,
    GateRegistry,
    HGate,
    IdentityGate,
    SGate,
    get_standard_clifford_gate_set,
    get_standard_css_gate_set,
)


def test_h_gate_properties() -> None:
    """Test H gate basic properties."""
    h = HGate(0)
    assert h.qubit == 0
    assert h.qubits() == {0}
    assert h.to_stim_gate() == ("H", [0])
    assert h.inverse_stim_gate() == ("H", [0])


def test_s_gate_properties() -> None:
    """Test S gate basic properties."""
    s = SGate(1)
    assert s.qubit == 1
    assert s.qubits() == {1}
    assert s.to_stim_gate() == ("S", [1])
    assert s.inverse_stim_gate() == ("S_DAG", [1])


def test_cnot_gate_properties() -> None:
    """Test CNOT gate basic properties."""
    cx = CNOTGate(0, 1)
    assert cx.control == 0
    assert cx.target == 1
    assert cx.qubits() == {0, 1}
    assert cx.to_stim_gate() == ("CX", [0, 1])
    assert cx.inverse_stim_gate() == ("CX", [0, 1])


def test_identity_gate_properties() -> None:
    """Test identity gate basic properties."""
    i = IdentityGate(2)
    assert i.qubit == 2
    assert i.qubits() == {2}
    assert i.to_stim_gate() == ("I", [2])


def test_h_gate_clifford_transition() -> None:
    """Test H gate Clifford tableau transition."""
    solver = z3.Solver()

    curr_x = np.array([[z3.Bool("curr_x_0_0")], [z3.Bool("curr_x_1_0")]], dtype=object)
    curr_z = np.array([[z3.Bool("curr_z_0_0")], [z3.Bool("curr_z_1_0")]], dtype=object)
    next_x = np.array([[z3.Bool("next_x_0_0")], [z3.Bool("next_x_1_0")]], dtype=object)
    next_z = np.array([[z3.Bool("next_z_0_0")], [z3.Bool("next_z_1_0")]], dtype=object)

    solver.add(curr_x[0, 0])
    solver.add(z3.Not(curr_z[0, 0]))

    h = HGate(0)
    solver.add(h.clifford_tableau_effect(curr_x, curr_z, next_x, next_z))

    assert solver.check() == z3.sat
    model = solver.model()

    assert not model.eval(next_x[0, 0], model_completion=True)
    assert model.eval(next_z[0, 0], model_completion=True)


def test_s_gate_not_applicable_to_css() -> None:
    """Test that S gate raises error for CSS encoding."""
    s = SGate(0)

    matrix_curr = np.array([[z3.Bool("m_0_0")]], dtype=object)
    matrix_next = np.array([[z3.Bool("m_1_0")]], dtype=object)

    with pytest.raises(NotImplementedError, match="S gate cannot be applied in CSS"):
        s.css_matrix_effect(matrix_curr, matrix_next)


def test_cnot_gate_css_transition() -> None:
    """Test CNOT gate CSS matrix transition."""
    solver = z3.Solver()

    matrix_curr = np.array([[z3.Bool("curr_0_0"), z3.Bool("curr_0_1")]], dtype=object)
    matrix_next = np.array([[z3.Bool("next_0_0"), z3.Bool("next_0_1")]], dtype=object)

    solver.add(matrix_curr[0, 0])
    solver.add(z3.Not(matrix_curr[0, 1]))

    cx = CNOTGate(0, 1)
    solver.add(cx.css_matrix_effect(matrix_curr, matrix_next))

    assert solver.check() == z3.sat
    model = solver.model()

    assert model.eval(matrix_next[0, 0], model_completion=True)
    assert model.eval(matrix_next[0, 1], model_completion=True)


def test_register_and_retrieve_clifford_gate() -> None:
    """Test registering and retrieving Clifford gates."""
    registry = GateRegistry()
    registry.register_clifford_gate("H", HGate)
    registry.register_clifford_gate("S", SGate)

    gates = registry.get_clifford_gates()
    assert "H" in gates
    assert "S" in gates
    assert gates["H"] == HGate


def test_register_and_retrieve_css_gate() -> None:
    """Test registering and retrieving CSS gates."""
    registry = GateRegistry()
    registry.register_css_gate("CX", CNOTGate)

    gates = registry.get_css_gates()
    assert "CX" in gates
    assert gates["CX"] == CNOTGate


def test_create_gate_instance() -> None:
    """Test creating gate instances from registry."""
    registry = GateRegistry()
    registry.register_clifford_gate("H", HGate)

    h = registry.create_gate("H", 0, for_css=False)
    assert isinstance(h, HGate)
    assert h.qubit == 0


def test_create_unregistered_gate_raises_error() -> None:
    """Test that creating unregistered gate raises KeyError."""
    registry = GateRegistry()

    with pytest.raises(KeyError, match="Gate 'X' not registered"):
        registry.create_gate("X", 0, for_css=False)


def test_standard_clifford_gate_set() -> None:
    """Test standard Clifford gate set contains expected gates."""
    gate_set = get_standard_clifford_gate_set()

    assert "H" in gate_set
    assert "S" in gate_set
    assert "CX" in gate_set
    assert "ID" in gate_set

    assert gate_set["H"] == HGate
    assert gate_set["S"] == SGate
    assert gate_set["CX"] == CNOTGate
    assert gate_set["ID"] == IdentityGate


def test_standard_css_gate_set() -> None:
    """Test standard CSS gate set contains expected gates."""
    gate_set = get_standard_css_gate_set()

    assert "CX" in gate_set
    assert "ID" in gate_set
    assert "H" not in gate_set
    assert "S" not in gate_set

    assert gate_set["CX"] == CNOTGate
    assert gate_set["ID"] == IdentityGate
