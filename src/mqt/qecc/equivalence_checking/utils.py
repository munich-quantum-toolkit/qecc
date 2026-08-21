# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared helpers for equivalence-checking decision procedures."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np
import z3

from ..codes.core.css_code import CSSCode
from ..codes.core.pauli import PauliTableau
from ..codes.core.stabilizer_code import StabilizerCode
from ..mod2 import row_basis

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt


# ----------------------------------------------------------------------------------------------------
#   Helper functions
# ----------------------------------------------------------------------------------------------------


def elementwise_map(normal_bool: npt.NDArray[np.integer], variables: Sequence[z3.BoolRef]) -> z3.BoolRef:
    """Constrain Boolean variables to equal a binary vector."""
    return z3.And([
        variable if bit == 1 else z3.Not(variable) for bit, variable in zip(normal_bool, variables, strict=True)
    ])


def exactly_one(variables: Iterable[z3.BoolRef]) -> z3.BoolRef:
    """Constrain exactly one of the given Boolean variables to hold."""
    return z3.PbEq([(variable, 1) for variable in variables], 1)


def xor_list(variables: Iterable[z3.BoolRef]) -> z3.BoolRef:
    """Return the exclusive-or of an iterable of Boolean variables."""
    result = z3.BoolVal(False)
    for variable in variables:
        result = z3.Xor(result, variable)
    return result


def encode_row_operations(
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
            solver.add(auxiliary_matrix[row * columns + column] == xor_list(contributions))


@overload
def reduce_stabilizer_generators(code: CSSCode) -> CSSCode: ...


@overload
def reduce_stabilizer_generators(code: StabilizerCode) -> StabilizerCode: ...


def reduce_stabilizer_generators(code: StabilizerCode) -> StabilizerCode:
    """Return an equivalent code with a minimal independent generator set."""
    if isinstance(code, CSSCode):
        return CSSCode(
            Hx=row_basis(code.Hx).astype(np.int8),
            Hz=row_basis(code.Hz).astype(np.int8),
            distance=code.distance,
            x_distance=code.x_distance,
            z_distance=code.z_distance,
            Lx=code.Lx,
            Lz=code.Lz,
        )

    reduced_symplectic = row_basis(code.symplectic).astype(np.int8)
    return StabilizerCode(
        generators=PauliTableau.from_matrix(reduced_symplectic),
        distance=code.distance,
        z_logicals=code.z_logicals,
        x_logicals=code.x_logicals,
    )
