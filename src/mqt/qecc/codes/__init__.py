# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Module for constructing and manipulating CSS codes."""

from __future__ import annotations

from .constructions.bb_codes import construct_bb_code
from .constructions.color_codes.color_code import ColorCode, LatticeType
from .constructions.color_codes.hexagonal_color_code import HexagonalColorCode
from .constructions.color_codes.square_octagon_color_code import SquareOctagonColorCode
from .constructions.concatenation import ConcatenatedCode, ConcatenatedCSSCode
from .constructions.constructions import (
    construct_iceberg_code,
    construct_many_hypercube_code,
    construct_quantum_hamming_code,
)
from .constructions.rotated_surface_code import InvalidDistanceError, RotatedSurfaceCode
from .core.css_code import CSSCode, InvalidCSSCodeError
from .core.stabilizer_code import InvalidStabilizerCodeError, StabilizerCode

__all__ = [
    "CSSCode",
    "ColorCode",
    "ConcatenatedCSSCode",
    "ConcatenatedCode",
    "HexagonalColorCode",
    "InvalidCSSCodeError",
    "InvalidDistanceError",
    "InvalidStabilizerCodeError",
    "LatticeType",
    "RotatedSurfaceCode",
    "SquareOctagonColorCode",
    "StabilizerCode",
    "construct_bb_code",
    "construct_iceberg_code",
    "construct_many_hypercube_code",
    "construct_quantum_hamming_code",
]
