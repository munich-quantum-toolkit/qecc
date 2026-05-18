# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact synthesis module for Clifford circuits."""

from __future__ import annotations

from .gate_operations import get_clifford_cz_gate_set
from .search import synthesize_exact
from .types import (
    Objective,
    SynthesisResult,
    SynthesisStatus,
    TargetKind,
)
from .vars import CliffordDepthVars, CliffordGateCountVars

__all__ = [
    "CliffordDepthVars",
    "CliffordGateCountVars",
    "Objective",
    "SynthesisResult",
    "SynthesisStatus",
    "TargetKind",
    "get_clifford_cz_gate_set",
    "synthesize_exact",
]
