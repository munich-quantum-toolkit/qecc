# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Square Octagon Color Code. Created by Peter-Jan Derks."""

from __future__ import annotations

from .color_code import ColorCode, LatticeType


class SquareOctagonColorCode(ColorCode):
    """4.8.8 triangular colour code."""

    def __init__(self, distance: int) -> None:
        """4.8.8 triangular colour code.

        This class can be used to generate the parity check matrix of 4.8.8 triangular colour code.
        This code has parameters [n, k, d] = [1/2*d**2 + d - 1/2, 1, d].

        param:  distance: Distance of the code to generate. Must be an odd integer.
        """
        # additionally to ancilla_qubits (on squares) we have the ones on octagons
        self.octagon_ancilla_qubits: set[tuple[int, int]] = set()
        self.square_ancilla_qubits: set[tuple[int, int]] = set()
        super().__init__(lattice_type=LatticeType.SQUARE_OCTAGON, distance=distance)

    def add_qubits(self) -> None:
        """Add qubits to the code."""
        self.bottom_row_ancillas()
        y = 1
        x_max = self.distance

        while y <= (self.distance + self.distance // 2):
            self.even_data_qubit_row(x_max, y)
            if y == 1:
                x_max -= 1
            else:
                x_max -= 2
            y += 1
            if y <= (self.distance + self.distance // 2):
                self.odd_data_qubit_row(x_max, y)
                y += 1
            if y <= (self.distance + self.distance // 2):
                if y % 2 == 0:
                    self.even_ancilla_qubit_row(x_max, y)
                else:
                    self.odd_ancilla_qubit_row(x_max, y)
                y += 1

        self.ancilla_qubits = self.square_ancilla_qubits.union(self.octagon_ancilla_qubits)

    def even_ancilla_qubit_row(self, x_max: int, y: int) -> None:
        """Create even ancilla qubits."""
        x = y + 1
        n_qubits = 0
        while True:
            if n_qubits >= x_max:
                break
            self.square_ancilla_qubits.add((x, y))
            n_qubits += 1
            x += 3
            if n_qubits >= x_max:
                break
            self.octagon_ancilla_qubits.add((x, y))
            n_qubits += 1
            x += 3

    def odd_ancilla_qubit_row(self, x_max: int, y: int) -> None:
        """Create odd ancilla qubits."""
        x = y - 2
        n_qubits = 0
        while True:
            if n_qubits >= x_max:
                break
            self.octagon_ancilla_qubits.add((x, y))
            n_qubits += 1
            x += 3
            if n_qubits >= x_max:
                break
            self.square_ancilla_qubits.add((x, y))
            n_qubits += 1
            x += 3

    def even_data_qubit_row(self, x_max: int, y: int) -> None:
        """Create even data qubits."""
        x = y - 1
        n_qubits = 0
        while True:
            if n_qubits >= x_max:
                break
            self.data_qubits.add((x, y))
            n_qubits += 1
            x += 2
            if n_qubits >= x_max:
                break
            self.data_qubits.add((x, y))
            n_qubits += 1
            x += 4

    def odd_data_qubit_row(self, x_max: int, y: int) -> None:
        """Create odd data qubits."""
        x = y + 1
        n_qubits = 0
        while True:
            if n_qubits >= x_max:
                break
            self.data_qubits.add((x, y))
            n_qubits += 1
            x += 2
            if n_qubits >= x_max:
                break
            self.data_qubits.add((x, y))
            n_qubits += 1
            x += 4

    def bottom_row_ancillas(self) -> None:
        """Create ancilla qubits on the bottom row of the lattice."""
        for x in range(4, self.distance // 2 * 6, 6):
            self.octagon_ancilla_qubits.add((x, 0))

    def construct_layout(self) -> None:
        """Construct the layout of the code."""
        coords_to_idx: dict[tuple[int, int], int] = {}
        # builds a map: {(x,y): index} for each qubit with coordinates (x,y)
        # initializes the {qubit_index: [faces]} adjacency list
        for idx, q in enumerate(self.data_qubits):
            coords_to_idx[q] = idx
            self.qubits_to_faces[idx] = []

        for idx, (x, y) in enumerate(self.square_ancilla_qubits):
            qbts: list[int] = []
            for coord in [
                (x - 1, y - 1),
                (x + 1, y - 1),
                (x + 1, y + 1),
                (x - 1, y + 1),
            ]:
                if coord in coords_to_idx:
                    qubit_idx = coords_to_idx[coord]
                    qbts.append(qubit_idx)
                    self.qubits_to_faces[qubit_idx].append(idx)

            self.faces_to_qubits[idx] = qbts
            for qb in qbts:
                self.H[idx, qb] = 1

        for idx, (x, y) in enumerate(self.octagon_ancilla_qubits):
            idx += len(self.square_ancilla_qubits)  # noqa: PLW2901
            qbts = []
            for coord in [
                (x - 2, y - 1),
                (x - 1, y - 2),
                (x + 1, y - 2),
                (x + 2, y - 1),
                (x + 2, y + 1),
                (x + 1, y + 2),
                (x - 1, y + 2),
                (x - 2, y + 1),
            ]:
                if coord in coords_to_idx:
                    qubit_idx = coords_to_idx[coord]
                    qbts.append(qubit_idx)
                    self.qubits_to_faces[qubit_idx].append(idx)

            self.faces_to_qubits[idx] = qbts
            for qb in qbts:
                self.H[idx, qb] = 1
