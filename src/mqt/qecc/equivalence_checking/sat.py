# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""SAT encoding primitives shared by the equivalence-checking decision procedures."""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import z3

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np
    import numpy.typing as npt


def _elementwise_map(normal_bool: npt.NDArray[np.integer], variables: Sequence[z3.BoolRef]) -> z3.BoolRef:
    """Constrain Boolean variables to equal a binary vector."""
    return z3.And([
        variable if bit == 1 else z3.Not(variable) for bit, variable in zip(normal_bool, variables, strict=True)
    ])


def _exactly_one(variables: Iterable[z3.BoolRef]) -> z3.BoolRef:
    """Constrain exactly one of the given Boolean variables to hold."""
    return z3.PbEq([(variable, 1) for variable in variables], 1)


def _encode_row_operations(
    solver: z3.Solver,
    auxiliary_matrix: Sequence[z3.BoolRef],
    target_matrix: npt.NDArray[np.integer],
    *,
    variable_prefix: str,
) -> None:
    """Constrain an auxiliary matrix to lie in the target matrix's row space."""
    rows, columns = target_matrix.shape
    coefficients = [z3.Bool(f"{variable_prefix}_{row}_{source}") for row in range(rows) for source in range(rows)]

    for row in range(rows):
        for column in range(columns):
            contributions = (
                coefficients[row * rows + source] for source in range(rows) if target_matrix[source, column] == 1
            )
            solver.add(auxiliary_matrix[row * columns + column] == reduce(z3.Xor, contributions, z3.BoolVal(False)))
