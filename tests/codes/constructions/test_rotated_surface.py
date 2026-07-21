# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the CSSCode class."""

from __future__ import annotations

import pytest

from mqt.qecc.codes.constructions.rotated_surface_code import InvalidDistanceError, RotatedSurfaceCode


@pytest.mark.parametrize(("x_distance", "z_distance"), [(3, 3), (7, 3), (5, 7)])
def test_rotated_surface_code_params(x_distance: int, z_distance: int) -> None:
    """Test the RotatedSurfaceCode class."""
    code = RotatedSurfaceCode(x_distance=x_distance, z_distance=z_distance)
    n = x_distance * z_distance
    assert code.n == n
    assert code.k == 1
    assert code.x_distance == x_distance
    assert code.z_distance == z_distance

    num_patches = (x_distance - 1) * (z_distance - 1)
    num_x_stabs = num_patches // 2 + num_patches % 2 + (z_distance - 1)
    num_z_stabs = num_patches // 2 + (x_distance - 1)
    assert code.Hx.shape == (num_x_stabs, n)
    assert code.Hz.shape == (num_z_stabs, n)


def test_rotated_surface_code_invalid_distance() -> None:
    """Test that an error is raised if the distance is not odd."""
    with pytest.raises(InvalidDistanceError):
        RotatedSurfaceCode(distance=6)
    with pytest.raises(InvalidDistanceError):
        RotatedSurfaceCode(x_distance=4, z_distance=5)
    with pytest.raises(InvalidDistanceError):
        RotatedSurfaceCode(x_distance=5, z_distance=4)
    with pytest.raises(InvalidDistanceError):
        RotatedSurfaceCode(x_distance=5, z_distance=None)
