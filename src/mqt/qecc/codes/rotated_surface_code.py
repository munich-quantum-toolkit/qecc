# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
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
        if stab_type not in {"x", "z"}:
            msg = "Type must be either 'x' or 'z'."
            raise ValueError(msg)

        n = x_distance * z_distance
        h: NDArray[np.int8] = np.empty((0, n), dtype=np.int8)

        # squares
        for i in range(x_distance - 1):
            for j in range(z_distance - 1):
                if (stab_type == "x" and (i + j) % 2 == 0) or (stab_type == "z" and (i + j) % 2 == 1):
                    stab = np.zeros(n, dtype=np.int8)
                    stab[i + j * x_distance] = 1
                    stab[i + j * x_distance + 1] = 1
                    stab[i + j * x_distance + x_distance] = 1
                    stab[i + j * x_distance + x_distance + 1] = 1
                    h = np.vstack((h, stab), dtype=np.int8)
        # boundaries
        if stab_type == "x":
            for i in range(z_distance - 1):  # rows
                stab = np.zeros(n, dtype=np.int8)
                if i % 2 == 0:
                    stab[i * x_distance + x_distance - 1] = 1
                    stab[(i + 1) * x_distance + x_distance - 1] = 1
                else:
                    stab[i * x_distance] = 1
                    stab[(i + 1) * x_distance] = 1
                h = np.vstack((h, stab), dtype=np.int8)
        else:
            for i in range(x_distance - 1):  # columns
                stab = np.zeros(n, dtype=np.int8)
                if i % 2 == 0:
                    stab[i] = 1
                    stab[i + 1] = 1
                else:
                    stab[i + x_distance * (z_distance - 1)] = 1
                    stab[i + x_distance * (z_distance - 1) + 1] = 1
                h = np.vstack((h, stab), dtype=np.int8)
        return h
