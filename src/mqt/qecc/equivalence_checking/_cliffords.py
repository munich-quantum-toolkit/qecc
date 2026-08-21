# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Representatives of the single-qubit Clifford group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

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


def _select_column(
    source: ColumnSource, x_column: npt.NDArray[np.integer], z_column: npt.NDArray[np.integer]
) -> npt.NDArray[np.integer]:
    """Select a binary column expression from two symplectic columns."""
    if source == "zero":
        return np.zeros_like(x_column)
    if source == "x":
        return x_column
    if source == "z":
        return z_column
    return (x_column + z_column) % 2


def _apply_local_clifford(tableau: npt.NDArray[np.int8], operation: str, qubit: int) -> None:
    """Apply a local Clifford representative to one tableau qubit in place."""
    n = tableau.shape[1] // 2
    x_column = tableau[:, qubit].copy()
    z_column = tableau[:, qubit + n].copy()
    x_source, z_source = CLIFFORD_ACTIONS[operation].transformed_columns
    tableau[:, qubit] = _select_column(x_source, x_column, z_column)
    tableau[:, qubit + n] = _select_column(z_source, x_column, z_column)


def _canonicalize_clifford(word: str) -> str:
    """Return the canonical representative of a Clifford word."""
    matrix = np.eye(2, dtype=np.uint8)
    for gate in word:
        if gate not in {"H", "S", "I"}:
            msg = f"Unknown Clifford gate {gate!r}."
            raise ValueError(msg)
        matrix = (matrix @ np.asarray(CLIFFORD_ACTIONS[gate].matrix, dtype=np.uint8)) % 2

    matrices = {action.matrix: name for name, action in CLIFFORD_ACTIONS.items()}
    key: CliffordMatrix = (
        (int(matrix[0, 0]), int(matrix[0, 1])),
        (int(matrix[1, 0]), int(matrix[1, 1])),
    )
    return matrices[key]
