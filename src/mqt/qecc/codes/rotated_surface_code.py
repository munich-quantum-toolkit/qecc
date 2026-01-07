# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""A rotated surface code class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .css_code import CSSCode

if TYPE_CHECKING:
    from numpy.typing import NDArray


class InvalidDistanceError(ValueError):
    """Custom error for invalid distance in rotated surface code."""

    def __init__(self, message: str) -> None:
        """Initialize the InvalidDistanceError."""
        super().__init__(message)
        self.message = message


class RotatedSurfaceCode(CSSCode):
    """A rotated surface code class."""

    def __init__(
        self, distance: int | None = None, x_distance: int | None = None, z_distance: int | None = None
    ) -> None:
        """Initialize the rotated surface code."""
        if distance is not None:
            if distance % 2 == 0:
                msg = "Distance must be odd."
                raise InvalidDistanceError(msg)
            super().__init__(
                self._generate_h("x", distance, distance), self._generate_h("z", distance, distance), distance
            )
        elif x_distance is None or z_distance is None:
            msg = "Either distance or both x_distance and z_distance must be provided."
            raise InvalidDistanceError(msg)
        else:
            if x_distance % 2 == 0 or z_distance % 2 == 0:
                msg = "x_distance and z_distance must be odd."
                raise InvalidDistanceError(msg)
            super().__init__(
                self._generate_h("x", x_distance, z_distance),
                self._generate_h("z", x_distance, z_distance),
                x_distance=x_distance,
                z_distance=z_distance,
            )

    @staticmethod
    def _generate_h(stab_type: str, x_distance: int, z_distance: int) -> NDArray[np.int8]:
        """Generate the check matrix for the rotated surface code."""
        n = x_distance * z_distance
        n_stabs = ((x_distance - 1) * (z_distance - 1)) // 2
        n_stabs += z_distance - 1 if stab_type == "x" else x_distance - 1
        h: NDArray[np.int8] = np.zeros((n_stabs, n), dtype=np.int8)

        # squares
        row = 0
        for i in range(x_distance - 1):
            for j in range(z_distance - 1):
                if (stab_type == "x" and (i + j) % 2 == 0) or (stab_type == "z" and (i + j) % 2 == 1):
                    stab = h[row]
                    base_index = i + j * x_distance
                    h[row, base_index : base_index + 2] = 1
                    h[row, base_index + x_distance : base_index + x_distance + 2] = 1
                    row += 1
        # boundaries
        if stab_type == "x":
            for i in range(z_distance - 1):  # rows
                stab = h[row]
                if i % 2 == 0:
                    stab[i * x_distance + x_distance - 1] = 1
                    stab[(i + 1) * x_distance + x_distance - 1] = 1
                else:
                    stab[i * x_distance] = 1
                    stab[(i + 1) * x_distance] = 1
                row += 1
        else:
            for i in range(x_distance - 1):  # columns
                stab = h[row]
                if i % 2 == 0:
                    stab[i] = 1
                    stab[i + 1] = 1
                else:
                    stab[i + x_distance * (z_distance - 1)] = 1
                    stab[i + x_distance * (z_distance - 1) + 1] = 1
                row += 1
        return h
