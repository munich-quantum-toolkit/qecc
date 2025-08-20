# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
from mqt.qecc.circuit_synthesis.synthesis_utils import CandidateAction, EliminationCNOTSynthesizer
from mqt.qecc.codes.css_code import CSSCode

STEANE_CHECK_MATRIX = np.array([[1, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=np.int8)
STEANE_CODE = CSSCode(Hx=STEANE_CHECK_MATRIX)


@pytest.fixture
def get_instance():
    """A basic fixture to provide a fresh GaussianElimination instance for each test."""
    return EliminationCNOTSynthesizer(matrix=STEANE_CHECK_MATRIX.copy(), code=STEANE_CODE)


# Use parametrize to test all logical branches
@pytest.mark.parametrize(
    ("cost", "backtrack_required", "used_cnots", "failed_cnots", "expected_action"),
    [
        # Case 1: Greedy step, should evaluate
        (-1, False, [], [], CandidateAction.EVALUATE),
        # Case 2: Already used, should skip
        (-1, False, [(0, 1)], [], CandidateAction.SKIP),
        # Case 3: Already failed, should skip
        (-1, False, [], [(0, 1)], CandidateAction.SKIP),
        # Case 4: First time hitting non-negative cost, should restart search
        (0, False, [], [], CandidateAction.RESTART_SEARCH),
        (1, False, [], [], CandidateAction.RESTART_SEARCH),
        # Case 5: Already in backtrack mode and hitting non-negative cost, should trigger backtrack
        (0, True, [], [], CandidateAction.TRIGGER_BACKTRACK),
        (1, True, [], [], CandidateAction.TRIGGER_BACKTRACK),
    ],
)
def test_get_candidate_action(get_instance, cost, backtrack_required, used_cnots, failed_cnots, expected_action):
    """Verify that _get_candidate_action returns the correct action based on state."""
    # Arrange: Set up the instance state for the test case
    get_instance.backtrack_required = backtrack_required
    get_instance.used_cnots = used_cnots
    get_instance.failed_cnots = failed_cnots

    # Act: Call the method
    action = get_instance._get_candidate_action(i=0, j=1, costs=cost)

    # Assert: Check the result
    assert action == expected_action


def test_apply_cnot_to_matrix_updates_matrix_and_costs_correctly(get_instance):
    """Verify the target column of the matrix is correctly updated after a CNOT."""
    expected_matrix = np.array([[1, 1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=np.int8)
    get_instance._apply_cnot_to_matrix(i=0, j=4)
    np.testing.assert_array_equal(get_instance.matrix, expected_matrix)
    expected_instance = EliminationCNOTSynthesizer(matrix=expected_matrix, code=CSSCode(expected_matrix))
    expected_costs = expected_instance.costs
    np.testing.assert_array_equal(get_instance.costs, expected_costs)


def test_mask_out_used_qubits(get_instance):
    """Check that the matrix is masked out cleanly based on the used columns."""
    used_qubit_index = 1
    unused_qubit_index = 0
    get_instance.used_columns.append(used_qubit_index)
    masked_costs = get_instance._mask_out_used_qubits()
    # 1. Is the data of the masked array still the same as the original costs?
    np.testing.assert_array_equal(masked_costs.data, get_instance.costs)

    # 2. Is the entire row for the used qubit masked?
    assert np.all(masked_costs.mask[used_qubit_index, :]), "Row for used qubit should be fully masked"

    # 3. Is the entire column for the used qubit masked?
    assert np.all(masked_costs.mask[:, used_qubit_index]), "Column for used qubit should be fully masked"

    # 4. Is a cell NOT involving the used qubit left unmasked?
    assert not masked_costs.mask[unused_qubit_index, unused_qubit_index], "Unused cell should not be masked"


@pytest.mark.parametrize(
    (
        "scenario",
        "initial_used_cols",
        "costs_are_positive",
        "expected_return",
        "expect_reset_called",
        "expected_final_used_cols",
    ),
    [
        # Scenario 1: Stagnated and no used columns -> should call triangular_reset
        ("local_minimum", [], True, True, True, []),
        # Scenario 2: Stagnated with used columns -> should clear used_columns
        ("parallel_deadend", [1, 2], True, True, False, []),
        # Scenario 3: Not stagnated (negative cost exists) -> should do nothing
        ("not_stagnated", [1, 2], False, False, False, [1, 2]),
    ],
)
def test_reset_if_stuck(
    get_instance,
    monkeypatch,
    scenario,
    initial_used_cols,
    costs_are_positive,
    expected_return,
    expect_reset_called,
    expected_final_used_cols,
):
    """Verify that _reset_if_stuck behaves correctly in all scenarios."""
    costs_unused = np.array([[0, 1], [1, 0]]) if costs_are_positive else np.array([[0, -1], [1, 0]])
    get_instance.used_columns = initial_used_cols

    reset_call_log = []  # This list will act as our spy.
    # We replace the real, complex _triangular_reset with a fake function
    # that just appends to our log list when it's called.
    monkeypatch.setattr(get_instance, "_triangular_reset", lambda: reset_call_log.append(True))

    return_value = get_instance._reset_if_stuck(costs_unused)

    assert return_value == expected_return
    assert get_instance.used_columns == expected_final_used_cols

    if expect_reset_called:
        assert len(reset_call_log) == 1
    else:
        assert len(reset_call_log) == 0


def test_greedy_synthesis_integration_with_known_matrix(get_instance):
    """Tests the full greedy_synthesis method from start to finish.

    It verifies that the correct sequence of eliminations is produced
    and that the matrix is fully reduced at the end.
    """
    expected_final_matrix = np.array(
        [[0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]], dtype=np.int8
    )

    expected_eliminations = [(0, 4), (1, 5), (2, 6), (1, 0), (3, 4), (5, 6), (0, 2), (3, 5)]

    get_instance.greedy_synthesis()

    # 1. Did it produce the correct sequence of operations?
    assert get_instance.eliminations == expected_eliminations

    # 2. Is the algorithm correctly reporting that it's finished?
    assert get_instance.is_reduced() is True

    # 3. Is the final matrix what we expect?
    np.testing.assert_array_equal(get_instance.matrix, expected_final_matrix)


@pytest.mark.parametrize(
    ("guide_by_x", "expected_failed_cnots"),
    [
        # Scenario 1: Guided by the control qubit (X-errors)
        (True, [(6, 4), (3, 4)]),
        # Scenario 2: Guided by the target qubit (Z-errors)
        (False, [(3, 5), (3, 4)]),
    ],
)
def test_backtrack_restores_state_and_updates_failed_cnots(get_instance, guide_by_x, expected_failed_cnots):
    """Verify that _backtrack correctly pops state from the stack, restores it, and updates the failed_cnots list based on the guide_by_x flag."""
    # ARRANGE
    prev_x_propagation_matrix = np.array([[-1]])
    prev_z_propagation_matrix = np.array([[-2]])
    prev_matrix = np.array([[-3]])
    prev_costs = np.array([[-4]])
    prev_used_cols = [-5]
    prev_used_cnots = [(-6, -6)]

    previous_state_tuple = (
        prev_x_propagation_matrix,
        prev_z_propagation_matrix,
        prev_used_cols,
        prev_matrix,
        prev_costs,
        prev_used_cnots,
        None,
    )

    get_instance._ref_based_init()
    get_instance.matrix = np.array([[99]])
    get_instance.costs = np.array([[98]])
    get_instance.used_columns = [97]
    get_instance.used_cnots = [(96, 96)]
    get_instance.x_propagation_matrix = [(95, 95)]
    get_instance.z_propagation_matrix = [(94, 94)]
    get_instance.eliminations = [(1, 2), (3, 4)]  # Last element (3, 4) will be popped
    get_instance.backtrack_required = True
    get_instance.guide_by_x = guide_by_x

    get_instance.failed_cnots = [(3, 5), (6, 4)]

    get_instance.stack.append(previous_state_tuple)

    # ACT
    get_instance._backtrack()

    # ASSERT

    # 1. Check that the core attributes were restored from the stack.
    np.testing.assert_array_equal(get_instance.matrix, prev_matrix)
    np.testing.assert_array_equal(get_instance.costs, prev_costs)
    np.testing.assert_array_equal(get_instance.x_propagation_matrix, prev_x_propagation_matrix)
    np.testing.assert_array_equal(get_instance.z_propagation_matrix, prev_z_propagation_matrix)
    assert get_instance.used_columns == prev_used_cols
    assert get_instance.used_cnots == prev_used_cnots

    # 2. Check that the backtrack flag was reset.
    assert get_instance.backtrack_required is False

    # 3. Check that the eliminations list had the last item popped.
    assert get_instance.eliminations == [(1, 2)]

    # 4. Check the conditional logic: Was failed_cnots updated correctly?
    #    The `_` in the pop restores `prev_failed_cnots`, but then the method
    #    recalculates it based on the popped elimination.
    #    NOTE: Your implementation replaces the list entirely, so we check against
    #    the expected final list, not the restored one.
    assert get_instance.failed_cnots == expected_failed_cnots


def test_compute_cost_matrix_golden_master():
    """Tests if the cost matrix is calculatet correctly.

    The test is based on the initial check matrix of the Steane Code.
    """
    input_matrix = np.array([[1, 1, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 0, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=np.int8)
    expected_cost_matrix = np.array([
        [1, 0, 0, 2, -2, 0, 0],
        [-1, 1, 1, 1, -1, -1, 1],
        [-1, 1, 1, 1, -1, 1, -1],
        [1, 1, 1, 1, -1, -1, -1],
        [-1, 1, 1, 1, 1, -1, -1],
        [0, 0, 2, 0, -2, 1, 0],
        [0, 2, 0, 0, -2, 0, 1],
    ])
    instance = EliminationCNOTSynthesizer(matrix=input_matrix, code=CSSCode(input_matrix))
    cost_matrix = instance._compute_cost_matrix()
    np.testing.assert_array_equal(cost_matrix, expected_cost_matrix)


def test_compute_cost_matrix_diagonal_is_always_one(get_instance):
    """Verify that the diagonal of the cost matrix is always 1."""
    cost_matrix = get_instance._compute_cost_matrix()

    assert np.all(np.diag(cost_matrix) == 1)


def test_compute_cost_matrix_off_diagonal_logic(get_instance):
    """Verify the core logic for a single off-diagonal element of the cost matrix.

    cost(i->j) = sum(col_i + col_j) - weight(col_j)
    """
    # Let's test the cost of applying column 0 onto column 1: costs[0, 1]
    matrix = get_instance.matrix
    col_0 = matrix[:, 0]
    col_1 = matrix[:, 1]

    # Manually calculate the expected value for just one cell.
    xor_sum: npt.NDArray[np.int8] = np.sum((col_0 + col_1) % 2)
    weight_col_1: npt.NDArray[np.int8] = np.sum(col_1)
    expected_cost_0_1 = xor_sum - weight_col_1

    cost_matrix = get_instance._compute_cost_matrix()

    assert cost_matrix[0, 1] == expected_cost_0_1
