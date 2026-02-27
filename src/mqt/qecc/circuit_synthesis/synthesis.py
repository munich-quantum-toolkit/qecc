# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""High-level synthesis functions for quantum circuits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2

from . import strategy
from .elimination import EliminationSequence, eliminate
from .operations import CNOT

if TYPE_CHECKING:
    from ..codes.pauli import CheckMatrix, StabilizerTableau


def synthesize_cnot(
    matrix: CheckMatrix,
    optimization_criterion: str = "gates",
    exact: bool = True,
    lookahead: int = 0,
    num_lookahead_candidates: int | list[int] = 10,
    enable_early_termination: bool = False,
) -> tuple[EliminationSequence, CheckMatrix]:
    """Eliminate a CSS check matrix using CNOT operations.

    Args:
        matrix: The CSS check matrix to eliminate.
        optimization_criterion: Either "gates" or "depth" for optimization objective.
        exact: If True, eliminate to echelon form. If False, eliminate only up to row operations.
        lookahead: Number of steps to look ahead (0 = greedy).
        num_lookahead_candidates: Number of candidates to explore at each lookahead layer.
        enable_early_termination: If True, allows early termination when no improving candidates found.

    Returns:
        A tuple of (operations, final_matrix) where operations is the sequence
        of CNOT operations and final_matrix is the reduced check matrix.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    if matrix.num_rows() == 0:
        return EliminationSequence([]), matrix.copy()

    target_rank = mod2.rank(matrix.matrix)
    if exact:
        if lookahead > 0:
            strat = strategy.for_cnot_with_lookahead_exact(
                target_rank=target_rank,
                optimization_criterion=optimization_criterion,
                lookahead=lookahead,
                num_lookahead_candidates=num_lookahead_candidates,
                enable_early_termination=enable_early_termination,
            )
        else:
            strat = strategy.for_cnot_exact(target_rank=target_rank, optimization_criterion=optimization_criterion)
    elif lookahead > 0:
        strat = strategy.for_cnot_with_lookahead_up_to_row_ops(
            optimization_criterion=optimization_criterion,
            lookahead=lookahead,
            num_lookahead_candidates=num_lookahead_candidates,
            target_rank=target_rank,
            enable_early_termination=enable_early_termination,
        )
    else:
        strat = strategy.for_cnot_up_to_row_ops(target_rank=target_rank, optimization_criterion=optimization_criterion)

    operations, final_matrix = eliminate(matrix, strat)

    if matrix.is_z_type():
        for op in operations.operations:
            if isinstance(op, CNOT):
                op.control, op.target = op.target, op.control

    return operations, final_matrix


def synthesize_non_css(
    tableau: StabilizerTableau, optimization_criterion: str = "gates"
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Eliminate a non-CSS stabilizer tableau using transvections.

    Args:
        tableau: The stabilizer tableau to eliminate.
        optimization_criterion: Either "gates" or "depth" for optimization objective.

    Returns:
        A tuple of (operations, final_tableau) where operations is the sequence
        of tableau operations and final_tableau is the reduced tableau.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    strat = strategy.for_non_css(optimization_criterion=optimization_criterion)
    operations, final_tableau = eliminate(tableau, strat)
    return operations, final_tableau


def synthesize_non_css_with_lookahead(
    tableau: StabilizerTableau,
    optimization_criterion: str = "gates",
    lookahead: int = 1,
    num_lookahead_candidates: int | list[int] = 10,
    enable_early_termination: bool = True,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Eliminate a non-CSS stabilizer tableau using transvections with lookahead.

    Args:
        tableau: The stabilizer tableau to eliminate.
        optimization_criterion: Either "gates" or "depth" for optimization objective.
        lookahead: Number of steps to look ahead in the synthesis.
        num_lookahead_candidates: Number of top candidates to explore at each lookahead layer.
            Can be a single int (same limit for all layers) or a list of ints (one per layer).
        enable_early_termination: If True, allows early termination when no improving candidates found.

    Returns:
        A tuple of (operations, final_tableau) where operations is the sequence
        of tableau operations and final_tableau is the reduced tableau.

    Raises:
        ValueError: If optimization_criterion is not "gates" or "depth".
    """
    strat = strategy.for_non_css_with_lookahead(
        optimization_criterion=optimization_criterion,
        lookahead=lookahead,
        num_lookahead_candidates=num_lookahead_candidates,
        enable_early_termination=enable_early_termination,
    )
    operations, final_tableau = eliminate(tableau, strat)
    return operations, final_tableau
