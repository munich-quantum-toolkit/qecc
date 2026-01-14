# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utilities for the CC decoder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..codes.color_code import LatticeType
from ..codes.hexagonal_color_code import HexagonalColorCode
from ..codes.square_octagon_color_code import SquareOctagonColorCode

if TYPE_CHECKING:
    from ..codes.color_code import ColorCode


def code_from_string(lattice_type: str, distance: int) -> ColorCode:
    """Construct a color code from a string defining the lattice and a distance."""
    if lattice_type == LatticeType.HEXAGON:
        return HexagonalColorCode(distance)
    if lattice_type == LatticeType.SQUARE_OCTAGON:
        return SquareOctagonColorCode(distance)
    msg = f"Unknown lattice type {lattice_type}. Please choose either hexagon or square_octagon."
    raise ValueError(msg)
