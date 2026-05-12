# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unified exact synthesis framework for Clifford circuits.

This module provides optimal synthesis for:
- Clifford unitaries
- Stabilizer-state preparation
- Clifford encoding isometries
- CSS state preparation
- CSS encoding isometries

The framework supports gate-count optimization, depth optimization,
and optional lexicographic depth-then-two-qubit-count optimization.
"""

from __future__ import annotations

from .search import synthesize_exact
from .types import (
    ExactSynthesisOptions,
    ExactSynthesisResult,
    GateFamily,
    Objective,
    SearchStrategy,
    SynthesisStatus,
    TargetKind,
)

__all__ = [
    "ExactSynthesisOptions",
    "ExactSynthesisResult",
    "GateFamily",
    "Objective",
    "SearchStrategy",
    "SynthesisStatus",
    "TargetKind",
    "synthesize_exact",
]
