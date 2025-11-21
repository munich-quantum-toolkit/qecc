# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit and integration tests for the code switching compilation module."""

import pytest
from qiskit import QuantumCircuit

from mqt.qecc.circuit_compilation import (
    CodeSwitchGraph,
    random_universal_circuit,
)
from mqt.qecc.circuit_compilation.compilation_utils import parse_node_id


@pytest.fixture
def simple_graph():
    """Fixture for a fresh graph instance."""
    return CodeSwitchGraph()


# =============================================================
# Unit tests


def test_parse_node_id():
    """Test the regex parsing of node IDs."""
    q, d = parse_node_id("H_q0_d10")
    assert q == 0
    assert d == 10

    with pytest.raises(ValueError, match="Invalid node_id format"):
        parse_node_id("Invalid_String")


def test_idle_bonus_logic(simple_graph):
    """Test that idle bonus is calculated correctly."""
    # Case 1: Short idle (less than SWITCHING_LENGTH=2) -> 0 bonus
    # depths 0 and 2 implies gap of 1 (depth 1 is empty)
    bonus = simple_graph.compute_idle_bonus(previous_depth=0, current_depth=2)
    assert bonus == 0.0

    # Case 2: Long idle greater than max_bonus
    # depths 0 and 10 implies gap of 9
    bonus = simple_graph.compute_idle_bonus(previous_depth=0, current_depth=10)
    assert bonus == 5 * simple_graph.base_unary_capacity

    # Ensure capacity reduction logic works
    cap = simple_graph._edge_capacity_with_idle_bonus([0, 10], base_capacity=1.0)  # noqa: SLF001
    assert cap < 1.0


# =============================================================
# Integration tests


def test_simple_switch_constraint():
    """Test a circuit that MUST switch: H (Source) -> T (Sink)."""
    qc = QuantumCircuit(1)
    qc.h(0)  # Source code transversal
    qc.t(0)  # Sink code transversal

    csg = CodeSwitchGraph()
    csg.build_from_qiskit(qc)

    # We expect the cut to sever the link between H and T or source/sink
    num_switches, switch_pos, _, _ = csg.compute_min_cut()

    # H -> T requires 1 switch
    assert num_switches > 0
    assert len(switch_pos) == 1
    # The switch should likely happen between depth 0 and 1
    assert switch_pos[0][0] == 0  # Qubit 0


def test_same_code_no_switch():
    """Test a circuit that stays in one code: H -> H -> CX."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)  # All compatible with 2D Color Code (Source)

    csg = CodeSwitchGraph()
    csg.build_from_qiskit(qc)

    num_switches, switch_pos, _, _ = csg.compute_min_cut()

    # Should flow entirely through Source
    assert len(switch_pos) == 0
    assert num_switches == 0.0


def test_one_way_transversal():
    """Test capability to cover one-way transversal gates."""
    qc = QuantumCircuit(2)
    qc.t(0)
    qc.h(1)
    qc.cx(0, 1)

    csg = CodeSwitchGraph()
    csg.build_from_qiskit(qc)
    num_switches, switch_pos, _, _ = csg.compute_min_cut()

    # We expect at least one switch due to T gate
    assert num_switches == 1
    assert switch_pos == [(0, 0)]  # Switch on qubit 0 at depth 0

    csg = CodeSwitchGraph()
    csg.build_from_qiskit(qc, one_way_transversal_cnot=True)
    num_switches, switch_pos, _, _ = csg.compute_min_cut()

    # Now, no switches should be needed
    assert num_switches == 0
    assert switch_pos == []  # Switch on qubit 0 at depth 0


# =============================================================
# Stress tests


def test_random_circuits_robustness():
    """Generate random circuits and ensure the compiler runs without error."""
    for seed in range(10):
        qc = random_universal_circuit(num_qubits=3, depth=10, seed=seed)

        csg = CodeSwitchGraph()
        csg.build_from_qiskit(qc)

        num_switches, switch_pos, S, T = csg.compute_min_cut()  # noqa: N806

        # Invariants
        assert len(switch_pos) >= 0
        assert num_switches >= 0
        assert len(S) + len(T) == csg.G.number_of_nodes()

        # Ensure Source is in S and Sink is in T
        assert csg.source in S
        assert csg.sink in T


# =============================================================
