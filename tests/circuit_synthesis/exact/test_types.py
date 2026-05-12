# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for exact synthesis type definitions."""

from __future__ import annotations

import pytest

from mqt.qecc.circuit_synthesis.exact import (
    ExactSynthesisOptions,
    GateFamily,
    Objective,
    SearchStrategy,
    TargetKind,
)


def test_target_kind_enum() -> None:
    """Test TargetKind enum values."""
    assert TargetKind.CLIFFORD_UNITARY.value == "clifford_unitary"
    assert TargetKind.STABILIZER_STATE.value == "stabilizer_state"
    assert TargetKind.CLIFFORD_ISOMETRY.value == "clifford_isometry"
    assert TargetKind.CSS_STATE_PREP.value == "css_state_prep"
    assert TargetKind.CSS_ISOMETRY.value == "css_isometry"


def test_gate_family_enum() -> None:
    """Test GateFamily enum values."""
    assert GateFamily.CLIFFORD.value == "clifford"
    assert GateFamily.CNOT.value == "cnot"


def test_objective_enum() -> None:
    """Test Objective enum values."""
    assert Objective.GATE_COUNT.value == "gate_count"
    assert Objective.DEPTH.value == "depth"
    assert Objective.DEPTH_THEN_TWO_QUBIT_COUNT.value == "depth_then_two_qubit_count"


def test_search_strategy_enum() -> None:
    """Test SearchStrategy enum values."""
    assert SearchStrategy.LINEAR.value == "linear"
    assert SearchStrategy.BINARY.value == "binary"
    assert SearchStrategy.GEOMETRIC.value == "geometric"


def test_options_validation() -> None:
    """Test ExactSynthesisOptions validation."""
    # Valid options
    opts = ExactSynthesisOptions(max_bound=10, lower_bound=0)
    assert opts.max_bound == 10
    assert opts.lower_bound == 0

    # Invalid: negative max_bound
    with pytest.raises(ValueError, match="max_bound must be non-negative"):
        ExactSynthesisOptions(max_bound=-1)

    # Invalid: lower_bound > max_bound
    with pytest.raises(ValueError, match="lower_bound cannot exceed max_bound"):
        ExactSynthesisOptions(max_bound=5, lower_bound=10)

    # Invalid: upper_bound < lower_bound
    with pytest.raises(ValueError, match="upper_bound cannot be less than lower_bound"):
        ExactSynthesisOptions(max_bound=20, lower_bound=5, upper_bound=3)

    # Invalid: binary search without upper_bound
    with pytest.raises(ValueError, match="Binary search requires an upper_bound"):
        ExactSynthesisOptions(max_bound=10, search_strategy=SearchStrategy.BINARY)

    # Valid: binary search with upper_bound
    opts = ExactSynthesisOptions(max_bound=10, upper_bound=8, search_strategy=SearchStrategy.BINARY)
    assert opts.upper_bound == 8


def test_options_defaults() -> None:
    """Test ExactSynthesisOptions default values."""
    opts = ExactSynthesisOptions(max_bound=10)
    assert opts.lower_bound == 0
    assert opts.upper_bound is None
    assert opts.search_strategy == SearchStrategy.LINEAR
    assert opts.enable_symmetry_breaking is False
    assert opts.timeout_per_bound is None
    assert opts.solver_params == {}
    assert opts.allow_qubit_permutation is True
    assert opts.verify_result is True
