# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Representatives of the single-qubit Clifford group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ColumnSource = Literal["zero", "x", "z", "xz"]
CliffordMatrix = tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class CliffordAction:
    """Binary symplectic data for a single-qubit Clifford representative."""

    matrix: CliffordMatrix
    transformed_columns: tuple[ColumnSource, ColumnSource]
    css_projected_columns: tuple[ColumnSource, ColumnSource]


CLIFFORD_ACTIONS = {
    "I": CliffordAction(((1, 0), (0, 1)), ("x", "z"), ("x", "zero")),
    "H": CliffordAction(((0, 1), (1, 0)), ("z", "x"), ("zero", "z")),
    "S": CliffordAction(((1, 0), (1, 1)), ("x", "xz"), ("x", "x")),
    "HS": CliffordAction(((1, 1), (1, 0)), ("xz", "x"), ("z", "z")),
    "SH": CliffordAction(((0, 1), (1, 1)), ("z", "xz"), ("zero", "xz")),
    "HSH": CliffordAction(((1, 1), (0, 1)), ("xz", "z"), ("xz", "zero")),
}
LOCAL_CLIFFORDS = tuple(CLIFFORD_ACTIONS)
