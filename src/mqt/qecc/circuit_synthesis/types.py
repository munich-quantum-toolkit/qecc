# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared type definitions for circuit synthesis."""

from __future__ import annotations

from ..codes.core.pauli import CheckMatrix, PauliTableau

# Type alias for matrices that can be used in elimination
BinaryMatrix = CheckMatrix | PauliTableau
