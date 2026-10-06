# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Code normalization and the invariants shared by both equivalence notions.

Every decision procedure first normalizes its inputs with
:func:`_reduce_stabilizer_generators` and then tries to refute equivalence with the
cheap invariants below. Invariants that hold for only one equivalence notion live in
the module implementing that notion.
"""

from __future__ import annotations

from typing import overload

import numpy as np

from ..codes.core.css_code import CSSCode
from ..codes.core.pauli import PauliTableau
from ..codes.core.stabilizer_code import StabilizerCode
from ..mod2 import row_basis


@overload
def _reduce_stabilizer_generators(code: CSSCode) -> CSSCode: ...


@overload
def _reduce_stabilizer_generators(code: StabilizerCode) -> StabilizerCode: ...


def _reduce_stabilizer_generators(code: StabilizerCode) -> StabilizerCode:
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


def _preserved_n(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check the number-of-qubits invariant."""
    return c1.n == c2.n


def _preserved_k(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check the number-of-logical-qubits invariant."""
    return c1.k == c2.k


def _preserved_distance(c1: StabilizerCode, c2: StabilizerCode) -> bool:
    """Check the total-code-distance invariant.

    Permutation equivalence additionally preserves the X- and Z-distances of a CSS code
    separately, which local Clifford equivalence does not, since local Cliffords may
    exchange the X and Z sector of a qubit. That stronger invariant therefore lives in
    :mod:`mqt.qecc.equivalence_checking.permutation_equivalence`.
    """
    return c1.distance == c2.distance
