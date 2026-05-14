# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact synthesis module for Clifford circuits."""

from __future__ import annotations

from .search import synthesize_exact
from .types import (
    Objective,
    SynthesisResult,
    SynthesisStatus,
    TargetKind,
)

__all__ = [
    "Objective",
    "SynthesisResult",
    "SynthesisStatus",
    "TargetKind",
    "synthesize_exact",
]
