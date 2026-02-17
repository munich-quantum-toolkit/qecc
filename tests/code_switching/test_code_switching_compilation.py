# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit and integration tests for the code switching compilation module."""

import pytest
from qiskit import QuantumCircuit

from mqt.qecc.code_switching import (
    MinimalCodeSwitchingCompiler,
)
from mqt.qecc.code_switching.code_switching_compiler import CompilerConfig


@pytest.fixture
def simple_graph():
    """Fixture for a fresh graph instance."""
    code_set_source = {"H", "CX"}
    code_set_sink = {"T", "CX"}
    return MinimalCodeSwitchingCompiler(code_set_source, code_set_sink)


# =============================================================
# Unit tests


def test_parse_node_id(simple_graph):
    """Test the regex parsing of node IDs."""
    q, d = simple_graph.parse_node_id("H_q0_d10")
    assert q == 0
    assert d == 10

    with pytest.raises(ValueError, match="Invalid node_id format"):
        simple_graph.parse_node_id("Invalid_String")


def test_idle_bonus_logic(simple_graph):
    """Test that idle bonus is calculated correctly using the normalized formula."""
    simple_graph.config.switching_time = 2
    base_cap = 1.0
    N = 100  # noqa: N806

    bonus = simple_graph.compute_idle_bonus(previous_depth=0, current_depth=2, total_edges=N)
    assert bonus == pytest.approx(0.0, abs=1e-8)

    bonus = simple_graph.compute_idle_bonus(previous_depth=0, current_depth=3, total_edges=N)
    assert bonus == pytest.approx(0.0, abs=1e-8)

    # --- Bonus Active (Calculation Check) ---
    # prev=0, curr=10 -> idle = 9.
    # Formula: 9 / (100 * (9 + 1)) = 9 / 1000 = 0.009
    idle_length = 9
    expected_bonus = idle_length / (N * (idle_length + 1))

    bonus = simple_graph.compute_idle_bonus(previous_depth=0, current_depth=10, total_edges=N)

    # Use approx for float safety
    assert bonus == pytest.approx(expected_bonus, rel=1e-9)
    assert bonus > 0.0

    eff_cap = simple_graph._edge_capacity_with_idle_bonus(  # noqa: SLF001
        depths=[0, 10], total_edges=N, base_capacity=base_cap
    )

    assert eff_cap < base_cap
    assert eff_cap == pytest.approx(base_cap - expected_bonus, rel=1e-9)

    # --- Large Graph Scaling (Safety Check) ---
    # In a huge circuit, the bonus should become very small to avoid distorting the graph,
    # but it must remain non-zero to act as a tie-breaker.
    huge_N = 1_000_000  # noqa: N806

    bonus_huge = simple_graph.compute_idle_bonus(previous_depth=0, current_depth=10, total_edges=huge_N)

    assert bonus_huge > 0.0
    assert bonus_huge < 1e-5  # Must be tiny


# =============================================================
# Integration tests


def test_simple_switch_constraint(simple_graph):
    """Test a circuit that MUST switch: H (Source) -> T (Sink)."""
    qc = QuantumCircuit(1)
    qc.h(0)  # Source code transversal
    qc.t(0)  # Sink code transversal

    simple_graph.build_from_qiskit(qc)

    # We expect the cut to sever the link between H and T or source/sink
    num_switches, switch_pos, _, _ = simple_graph.compute_min_cut()

    # H -> T requires 1 switch
    assert num_switches > 0
    assert len(switch_pos) == 1
    # The switch should likely happen between depth 0 and 1
    assert switch_pos[0][0] == 0  # Qubit 0


def test_same_code_no_switch(simple_graph):
    """Test a circuit that stays in one code: H -> H -> CX."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.cx(0, 1)  # All compatible with 2D Color Code (Source)

    simple_graph.build_from_qiskit(qc)

    num_switches, switch_pos, _, _ = simple_graph.compute_min_cut()

    # Should flow entirely through Source
    assert len(switch_pos) == 0
    assert num_switches == 0


def test_one_way_transversal(simple_graph):
    """Test capability to cover one-way transversal gates."""
    qc = QuantumCircuit(2)
    qc.t(0)
    qc.h(1)
    qc.cx(0, 1)

    simple_graph.build_from_qiskit(qc)
    num_switches, switch_pos, _, _ = simple_graph.compute_min_cut()

    # We expect at least one switch due to T gate
    assert num_switches == 1
    assert switch_pos == [(0, 0)]  # Switch on qubit 0 at depth 0

    simple_graph = MinimalCodeSwitchingCompiler({"H", "CX"}, {"T", "CX"})
    simple_graph.build_from_qiskit(qc, one_way_gates={"CX": ("SNK", "SRC")})
    num_switches, switch_pos, _, _ = simple_graph.compute_min_cut()

    # Now, no switches should be needed
    assert num_switches == 0
    assert switch_pos == []

    # One-way transversal should be invariant of source/sink definition
    simple_graph = MinimalCodeSwitchingCompiler({"T", "CX"}, {"H", "CX"})
    simple_graph.build_from_qiskit(qc, one_way_gates={"CX": ("SRC", "SNK")})
    num_switches, switch_pos, _, _ = simple_graph.compute_min_cut()

    # Again, no switches should be needed
    assert num_switches == 0
    assert switch_pos == []


def test_code_bias(simple_graph):
    """Test code bias effects switching positions."""
    qc = QuantumCircuit(2)
    qc.t(0)
    qc.h(1)
    qc.cx(0, 1)

    # Check min-cuts build in source bias
    simple_graph.build_from_qiskit(qc)
    num_switches_source_bias, switch_pos_source_bias, _, _ = simple_graph.compute_min_cut()

    assert num_switches_source_bias == 1
    assert switch_pos_source_bias == [(0, 0)]

    # To equalize min-cuts build in source bias, change CompilerConfig to bias sink.
    config = CompilerConfig(biased_code="SNK")
    simple_graph = MinimalCodeSwitchingCompiler({"H", "CX"}, {"T", "CX"}, config=config)
    simple_graph.build_from_qiskit(qc, code_bias=True)
    num_switches_sink_bias, switch_pos_sink_bias, _, _ = simple_graph.compute_min_cut()

    assert num_switches_sink_bias == 1
    assert switch_pos_sink_bias != switch_pos_source_bias
    assert switch_pos_sink_bias == [(1, 0)]
