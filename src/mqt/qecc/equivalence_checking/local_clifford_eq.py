# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Functions for deciding local clifford equivalence."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..codes.stabilizer_code import StabilizerCode


def are_local_clifford_equivalent(code1: StabilizerCode, code2: StabilizerCode) -> bool:
    """Check if two stabilizer codes are local clifford equivalent."""
    raise NotImplementedError


def is_local_clifford_equivalent_to_css(code: StabilizerCode) -> bool:
    """Check if a stabilizer code is local clifford equivalent to a CSS code."""
    raise NotImplementedError
