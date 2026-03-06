# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""High-level synthesis functions for quantum circuits."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import ldpc.mod2.mod2_numpy as mod2

from ..codes.pauli import CheckMatrix, StabilizerTableau
from . import strategy
from .elimination import EliminationSequence, eliminate
from .operations import CNOT


class SynthesisConfig(ABC):  # noqa:B024
    """Base class for synthesis configuration."""


@dataclass
class CnotSynthesisConfig(SynthesisConfig):
    """Configuration for CNOT-based synthesis."""

    optimization_criterion: str = "gates"
    exact: bool = True
    lookahead: int = 0
    num_lookahead_candidates: int | list[int] = 10
    enable_early_termination: bool = False


@dataclass
class CliffordSynthesisConfig(SynthesisConfig):
    """Configuration for non-CSS synthesis."""

    optimization_criterion: str = "gates"
    lookahead: int = 0
    num_lookahead_candidates: int | list[int] = 10
    enable_early_termination: bool = False


def synthesize_cnot(
    matrix: CheckMatrix,
    config: CnotSynthesisConfig | None = None,
    n_stabs: int | None = None,
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
        config = CnotSynthesisConfig()

    exact = config.exact
    optimization_criterion = config.optimization_criterion
    lookahead = config.lookahead
    num_lookahead_candidates = config.num_lookahead_candidates
    enable_early_termination = config.enable_early_termination
    n = matrix.num_qubits()

    if matrix.num_rows() == 0:
        return EliminationSequence([]), matrix.copy()

    target_rank = mod2.rank(matrix.matrix) if n_stabs is None else n_stabs

    if exact:
        if lookahead > 0:
            strat = strategy.for_cnot_with_lookahead_exact(
                target_rank=target_rank,
                n=n,
                optimization_criterion=optimization_criterion,
                lookahead=lookahead,
                num_lookahead_candidates=num_lookahead_candidates,
                enable_early_termination=enable_early_termination,
            )
        else:
            strat = strategy.for_cnot_exact(target_rank=target_rank, n=n, optimization_criterion=optimization_criterion)
    elif lookahead > 0:
        strat = strategy.for_cnot_with_lookahead_up_to_row_ops(
            target_rank=target_rank,
            n=n,
            optimization_criterion=optimization_criterion,
            lookahead=lookahead,
            num_lookahead_candidates=num_lookahead_candidates,
            enable_early_termination=enable_early_termination,
        )
    else:
        strat = strategy.for_cnot_up_to_row_ops(
            target_rank=target_rank, n=n, optimization_criterion=optimization_criterion
        )

    operations, final_matrix = eliminate(matrix, strat)

    assert isinstance(final_matrix, CheckMatrix), "Expected CheckMatrix from CSS elimination"

    if matrix.is_z_type():
        for op in operations.operations:
            if isinstance(op, CNOT):
                op.control, op.target = op.target, op.control

    return operations, final_matrix


def synthesize_non_css(
    tableau: StabilizerTableau,
    config: CliffordSynthesisConfig | None = None,
) -> tuple[EliminationSequence, StabilizerTableau]:
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
        config = CliffordSynthesisConfig()
    if config.lookahead > 0:
        strat = strategy.for_non_css_with_lookahead(
            n=tableau.n,
            optimization_criterion=config.optimization_criterion,
            lookahead=config.lookahead,
            num_lookahead_candidates=config.num_lookahead_candidates,
            enable_early_termination=config.enable_early_termination,
        )
    else:
        strat = strategy.for_non_css(n=tableau.n, optimization_criterion=config.optimization_criterion)

    operations, final_tableau = eliminate(tableau, strat)

    assert isinstance(final_tableau, StabilizerTableau), "Expected StabilizerTableau from non-CSS elimination"

    return operations, final_tableau
