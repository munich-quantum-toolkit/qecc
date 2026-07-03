# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Functions for deciding permutation equivalence."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..codes.css_code import CSSCode
    from ..codes.stabilizer_code import StabilizerCode


def are_permutation_equivalent(code1: StabilizerCode | CSSCode, code2: StabilizerCode | CSSCode) -> bool:
    """Check if two stabilizer codes are permutation equivalent."""
    raise NotImplementedError
