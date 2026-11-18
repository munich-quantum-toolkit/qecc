# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Solvers: given a design and an open scope (a node), propose new nodes for that scope."""

from __future__ import annotations

from .ancilla import BlockSolver, GuideSolver
from .checks import CrigerSolver
from .common import (
    CHOICE_CTX_DEPS,
    QeccSolver,
    block_slack,
    circuit_choices,
    part_cost,
    scope_slack,
)
from .edit import (
    NeighbourhoodSolver,
    OrderSolver,
    PrepMoveSolver,
    apply_prep_move,
    prep_moves,
)
from .merge import MergeSolver
from .polish import DepthOrderSolver, PolishSolver, PruneSolver, SharedFlagSolver, TakeoverSolver

__all__ = [
    "CHOICE_CTX_DEPS",
    "BlockSolver",
    "CrigerSolver",
    "DepthOrderSolver",
    "GuideSolver",
    "MergeSolver",
    "NeighbourhoodSolver",
    "OrderSolver",
    "PolishSolver",
    "PrepMoveSolver",
    "PruneSolver",
    "QeccSolver",
    "SharedFlagSolver",
    "TakeoverSolver",
    "apply_prep_move",
    "block_slack",
    "circuit_choices",
    "part_cost",
    "prep_moves",
    "scope_slack",
]
