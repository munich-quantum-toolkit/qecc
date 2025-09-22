# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Hexagonal Color Code class.

Created by Peter-Jan Derks
Hexagonal Color Code layout construction adapted from https://github.com/peter-janderks/restriction_decoder_domain_wall_colour_code
"""

from __future__ import annotations

from .color_code import ColorCode, LatticeType


class HexagonalColorCode(ColorCode):
    """Hexagonal Color Code."""

    def __init__(self, distance: int) -> None:
        """Hexagonal Color Code initialization from base class."""
        super().__init__(lattice_type=LatticeType.HEXAGON, distance=distance)

    def add_qubits(self) -> None:
        """Add qubits to the code."""
        colour = ["r", "b", "g"]
        y = 0

        x_max = self.distance + self.distance // 2
        while x_max > 0:
            ancilla_colour = colour[y % 3]
            if ancilla_colour == "r":
                self.red_row(x_max, y)
            elif ancilla_colour == "b":
                self.blue_row(x_max, y)
            else:
                self.green_row(x_max, y)
            x_max -= 1
            y += 1

    def red_row(self, x_max: int, y: int) -> None:
        """Create red ancilla qubits."""
        i = 0
        x_row = y
        while i < x_max:
            data_or_ancilla = i % 3
            if data_or_ancilla in {0, 2}:
                self.data_qubits.add((x_row, y))
            else:
                self.ancilla_qubits.add((x_row, y))
            i += 1
            x_row += 2

    def blue_row(self, x_max: int, y: int) -> None:
        """Create blue ancilla qubits."""
        i = 0
        x_row = y
        while i < x_max:
            data_or_ancilla = i % 3
            if data_or_ancilla in {0, 1}:
                self.data_qubits.add((x_row, y))
            else:
                self.ancilla_qubits.add((x_row, y))
            i += 1
            x_row += 2

    def green_row(self, x_max: int, y: int) -> None:
        """Create green ancilla qubits."""
        i = 0
        x_row = y
        while i < x_max:
            data_or_ancilla = i % 3
            if data_or_ancilla in {1, 2}:
                self.data_qubits.add((x_row, y))
            else:
                self.ancilla_qubits.add((x_row, y))
            x_row += 2
            i += 1

    def construct_layout(self) -> None:
        """Construct the layout of the hexagonal color code."""
        coords_to_idx: dict[tuple[int, int], int] = {}
        # builds a map: {(x,y): index} for each qubit with coordinates (x,y)
        # initializes the {qubit_index: [faces]} adjacency list
        for idx, q in enumerate(self.data_qubits):
            coords_to_idx[q] = idx
            self.qubits_to_faces[idx] = []

        for idx, (x, y) in enumerate(self.ancilla_qubits):
            qbts: list[int] = []
            for coord in [
                (x - 1, y + 1),
                (x - 2, y),
                (x - 1, y - 1),
                (x + 1, y - 1),
                (x + 2, y),
                (x + 1, y + 1),
            ]:
                if coord in coords_to_idx:
                    qubit_idx = coords_to_idx[coord]
                    qbts.append(qubit_idx)
                    self.qubits_to_faces[qubit_idx].append(idx)
            self.faces_to_qubits[idx] = qbts
            for qb in qbts:
                self.H[idx, qb] = 1

        # L is the matrix of logicals of the code
        self.compute_logical()
