# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""A color code class."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from .css_code import CSSCode

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt


class LatticeType(str, Enum):
    """The type of lattice used for the color code."""

    HEXAGON = "hexagon"
    SQUARE_OCTAGON = "square_octagon"


class ColorCode(CSSCode):
    """A base class for color codes on a three-valent, three-colourable lattice."""

    def __init__(self, distance: int, lattice_type: LatticeType) -> None:
        """Initialize the color code."""
        self.distance = distance
        self.ancilla_qubits: set[tuple[int, int]] = set()
        self.data_qubits: set[tuple[int, int]] = set()
        self.qubits_to_faces: dict[int, list[int]] = {}
        self.faces_to_qubits: dict[int, list[int]] = {}
        self.lattice_type = lattice_type
        self.add_qubits()
        self.H: npt.NDArray[np.int_] = np.zeros((len(self.ancilla_qubits), len(self.data_qubits)), dtype=int)
        self.construct_layout()
        CSSCode.__init__(self, self.H, self.H, distance)
        self.L = self.Lz

    def __hash__(self) -> int:
        """Compute a hash for the color code."""
        return hash(self.H.tobytes())

    def __eq__(self, other: object) -> bool:
        """Check if two color codes are equal."""
        if not isinstance(other, ColorCode):
            return NotImplemented
        return np.array_equal(self.H, other.H)

    def add_qubits(self) -> None:
        """Compute the ancilla and data qubits lists from the lattice type."""

    def construct_layout(self) -> None:
        """Construct the adjacency lists of the code from the qubits lists. Assumes add_qubits was called."""

    def compute_logical(self) -> None:
        """Compute the logical operators of the code."""
        self.L = self._compute_logical(self.H, self.H)

    def get_syndrome(self, error: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Compute the syndrome of the error."""
        return self.H @ error % 2

    def check_if_logical_error(self, residual: npt.NDArray[np.int_]) -> bool:
        """Check if the residual is a logical error."""
        return (self.L @ residual % 2).any() is True
