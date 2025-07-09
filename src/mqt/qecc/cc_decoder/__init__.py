# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Contains the implementation of the tensor network decoder for the hexagonal color code."""

from __future__ import annotations

from ..codes.color_code import ColorCode, LatticeType
from ..codes.hexagonal_color_code import HexagonalColorCode
from ..codes.square_octagon_color_code import SquareOctagonColorCode
from .comparison import tn_decoder


def code_from_string(lattice_type: str, distance: int) -> ColorCode:
    """Construct a color code from a string defining the lattice and a distance."""
    if lattice_type == LatticeType.HEXAGON:
        return HexagonalColorCode(distance)
    if lattice_type == LatticeType.SQUARE_OCTAGON:
        return SquareOctagonColorCode(distance)
    msg = f"Unknown lattice type {lattice_type}. Please choose either hexagon or square_octagon."
    raise ValueError(msg)


__all__ = [
    "ColorCode",
    "HexagonalColorCode",
    "LatticeType",
    "SquareOctagonColorCode",
    "code_from_string",
    "tn_decoder",
]
