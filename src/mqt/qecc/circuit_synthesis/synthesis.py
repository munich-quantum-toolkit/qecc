# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""High-level synthesis functions for quantum circuits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..codes.core.pauli import CheckMatrix, PauliTableau
from . import strategy
from .elimination import EliminationSequence, eliminate
from .operations import CNOT

if TYPE_CHECKING:
    from .operations import TableauOperation


@dataclass
class SynthesisConfig:
    """Base class for synthesis configuration."""

    optimization_criterion: str = "gates"
    rollout: int = 0
    num_rollout_candidates: int | list[int] = 10
    enable_early_termination: bool = False

    cache_max_weight: int = 1_000_000
    num_cached_rollout_subsequences: int = 10


def synthesize_cnot(
    matrix: CheckMatrix,
    config: SynthesisConfig | None = None,
    n_stabs: int = 0,
) -> tuple[EliminationSequence, CheckMatrix]:
    """Eliminate a CSS check matrix using CNOT operations.

    Args:
        matrix: The CSS check matrix to eliminate.
        config: Configuration for the synthesis process.
        n_stabs: Optional number of stabilizers

    Returns:
        A tuple of (operations, final_matrix) where operations is the sequence
        of CNOT operations and final_matrix is the reduced check matrix.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    if config is None:
        config = SynthesisConfig()

    optimization_criterion = config.optimization_criterion
    rollout = config.rollout
    num_rollout_candidates = config.num_rollout_candidates
    enable_early_termination = config.enable_early_termination
    n = matrix.num_qubits()

    if matrix.num_rows() == 0:
        return EliminationSequence([]), matrix.copy()

    if rollout > 0:
        strat = strategy.for_cnot_with_rollout_up_to_row_ops(
            n_stabs=n_stabs,
            n=n,
            optimization_criterion=optimization_criterion,
            rollout=rollout,
            num_rollout_candidates=num_rollout_candidates,
            enable_early_termination=enable_early_termination,
            cache_max_weight=config.cache_max_weight,
            num_cached_rollout_subsequences=config.num_cached_rollout_subsequences,
        )
    else:
        strat = strategy.for_cnot_up_to_row_ops(n_stabs=n_stabs, n=n, optimization_criterion=optimization_criterion)

    operations, final_matrix = eliminate(matrix, strat)

    assert isinstance(final_matrix, CheckMatrix), "Expected CheckMatrix from CSS elimination"

    if matrix.is_z_type():
        swapped_operations: list[TableauOperation] = []
        for op in operations.operations:
            assert isinstance(op, CNOT)
            swapped_operations.append(CNOT(op.target, op.control))
        operations.operations = swapped_operations

    return operations, final_matrix


def synthesize_non_css(
    tableau: PauliTableau,
    config: SynthesisConfig | None = None,
) -> tuple[EliminationSequence, PauliTableau]:
    """Eliminate a non-CSS stabilizer tableau using transvections.

    Args:
        tableau: The stabilizer tableau to eliminate.
        config: Configuration for the synthesis process.

    Returns:
        A tuple of (operations, final_tableau) where operations is the sequence
        of tableau operations and final_tableau is the reduced tableau.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    if config is None:
        config = SynthesisConfig()
    if config.rollout > 0:
        strat = strategy.for_non_css_with_rollout(
            n=tableau.n,
            optimization_criterion=config.optimization_criterion,
            rollout=config.rollout,
            num_rollout_candidates=config.num_rollout_candidates,
            enable_early_termination=config.enable_early_termination,
            cache_max_weight=config.cache_max_weight,
            num_cached_rollout_subsequences=config.num_cached_rollout_subsequences,
        )
    else:
        strat = strategy.for_non_css(n=tableau.n, optimization_criterion=config.optimization_criterion)

    operations, final_tableau = eliminate(tableau, strat)
    assert isinstance(final_tableau, PauliTableau), "Expected PauliTableau from non-CSS elimination"

    return operations, final_tableau
