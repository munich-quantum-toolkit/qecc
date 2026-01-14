# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for utility functions in compilation_utils."""

import pytest
from qiskit import QuantumCircuit

from mqt.qecc.code_switching.compilation_utils import (
    insert_switch_placeholders,
    naive_switching,
)

# ==============================================================================
# Tests for naive_switching
# ==============================================================================


def test_count_switches_single_qubit_simple():
    """Test simple H -> T transition on a single qubit."""
    qc = QuantumCircuit(1)
    qc.h(0)  # Code A
    qc.t(0)  # Code B -> Expect 1 switch

    count, final_codes = naive_switching(qc)

    assert count == 1
    assert final_codes[0] == "B"


def test_count_switches_no_switch_needed():
    """Test a sequence that stays within one code."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)  # H and CX are both supported by Code A

    count, final_codes = naive_switching(qc)

    assert count == 0
    assert final_codes[0] == "A"
    assert final_codes[1] == "A"


def test_count_switches_cnot_synchronization():
    """Test that CNOT forces qubits into the same code."""
    qc = QuantumCircuit(2)
    # Q0 starts in Code A (H gate)
    qc.h(0)
    # Q1 starts in Code B (T gate)
    qc.t(1)

    # CNOT requires them to be in the SAME code.
    # The implementation defaults to switching to A if possible, or B.
    # Here, one of them MUST switch.
    qc.cx(0, 1)

    count, _ = naive_switching(qc)
    assert count == 1


def test_count_switches_invalid_gate():
    """Test that an unsupported gate raises ValueError."""
    qc = QuantumCircuit(1)
    qc.x(0)  # 'x' is not in the defined sets for Code A or B in the utils

    with pytest.raises(ValueError, match="not supported by any code"):
        naive_switching(qc)


def test_count_switches_ignore_id():
    """Test that ID gates do not trigger initialization or switches."""
    qc = QuantumCircuit(1)
    qc.id(0)  # Should be ignored
    qc.h(0)  # Sets code to A
    qc.id(0)  # Should be ignored
    qc.h(0)  # Still A

    count, _ = naive_switching(qc)
    assert count == 0


# ==============================================================================
# Tests for insert_switch_placeholders
# ==============================================================================


def test_insert_placeholders_increases_depth():
    """Test that inserting placeholders actually changes the circuit."""
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.t(0)

    # Original Depth is 2
    original_depth = qc.depth()

    # Insert a placeholder after layer 0 (the H gate)
    # This usually adds 'id' gates.
    new_qc = insert_switch_placeholders(qc, switch_positions=[(0, 0)], placeholder_depth=1)

    # New depth should be Original + Placeholder Depth
    assert new_qc.depth() == original_depth + 1
    # Ensure the circuit name is updated
    assert new_qc.name == qc.name + "_with_switches"


def test_insert_placeholders_correct_location():
    """Verify placeholders are inserted on the correct qubit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    # Both H gates are likely in Layer 0 of the DAG

    # Insert placeholder ONLY on qubit 1 at layer 0
    new_qc = insert_switch_placeholders(qc, switch_positions=[(1, 0)], placeholder_depth=5)

    # Qubit 0 should remain depth 1 (just the H)
    # Qubit 1 should be depth 1 (H) + 5 (IDs) = 6
    # Note: Qiskit depth calculation can be tricky with IDs,
    # so we check the operations count on the specific qubit.

    # All placeholder IDs should be on qubit 1 only.
    id_targets = {
        new_qc.find_bit(q).index for instr in new_qc.data if instr.operation.name == "id" for q in instr.qubits
    }
    assert id_targets == {1}
    # And we still expect exactly 5 such IDs.
    id_count = sum(1 for instr in new_qc.data if instr.operation.name == "id")
    assert id_count == 5


def test_insert_placeholders_out_of_bounds():
    """Test behavior when insertion index is beyond the last layer."""
    qc = QuantumCircuit(1)
    qc.h(0)  # Layer 0

    # Request insertion at Layer 99
    # Logic dictates it should append at the end
    new_qc = insert_switch_placeholders(qc, switch_positions=[(0, 99)], placeholder_depth=1)

    # Should have H then ID
    assert new_qc.data[-1].operation.name == "id"
    assert new_qc.depth() == 2


def test_insert_multiple_placeholders():
    """Test inserting multiple placeholders at different layers."""
    qc = QuantumCircuit(1)
    qc.h(0)  # Layer 0
    qc.t(0)  # Layer 1

    # Insert after H and after T
    new_qc = insert_switch_placeholders(qc, switch_positions=[(0, 0), (0, 1)], placeholder_depth=1)

    # Original (2) + 2 placeholders = 4
    assert new_qc.depth() == 4

    # Verify order: H -> ID -> T -> ID
    op_names = [instr.operation.name for instr in new_qc.data]
    assert op_names == ["h", "id", "t", "id"]
