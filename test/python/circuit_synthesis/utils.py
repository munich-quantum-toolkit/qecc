# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utilities for the circuit synthesis unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ldpc import mod2
from qiskit.quantum_info import Clifford

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt
    from qiskit import QuantumCircuit


def eq_span(a: npt.NDArray[np.int_], b: npt.NDArray[np.int_]) -> bool:
    """Check if two matrices have the same row space."""
    return (a.shape[1] == b.shape[1]) and (int(mod2.rank(np.vstack((a, b)))) == int(mod2.rank(a)) == int(mod2.rank(b)))


def in_span(m: npt.NDArray[np.int_], v: npt.NDArray[np.int_]) -> bool:
    """Check if a vector is in the row space of a matrix over GF(2)."""
    return bool(mod2.rank(np.vstack((m, v))) == mod2.rank(m))


def get_stabs_css_with_indices(
    qc: QuantumCircuit,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], dict[int, int], dict[int, int]]:
    """Return the stabilizers of a quantum circuit.

    Assumes that stabilizers are CSS.

    Args:
        qc: The quantum circuit.
    returns:
        x: The X stabilizers.
        z: The Z stabilizers.
        x_indices: The indices of the X stabilizers.
        z_indices: The indices of the Z stabilizers.
    """
    cliff = Clifford(qc)
    x = cliff.stab_x.astype(int)
    x_indices = np.where(np.logical_not(np.all(x == 0, axis=1)))[0]
    qubit_to_x_pos = {x_indices[i]: i for i in range(len(x_indices))}
    x = x[x_indices]
    z = cliff.stab_z.astype(int)
    z_indices = np.where(np.logical_not(np.all(z == 0, axis=1)))[0]
    qubit_to_z_pos = {z_indices[i]: i for i in range(len(z_indices))}
    z = z[z_indices]

    return x, z, qubit_to_x_pos, qubit_to_z_pos


def get_stabs_css(qc: QuantumCircuit) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Return the stabilizers of a quantum circuit.

    Assumes that stabilizers are CSS.

    Args:
        qc: The quantum circuit.

    Returns:
        x: The X stabilizers.
        z: The Z stabilizers.

    """
    cliff = Clifford(qc)
    x = cliff.stab_x.astype(int)
    x_indices = np.where(np.logical_not(np.all(x == 0, axis=1)))[0]

    x = x[x_indices]
    z = cliff.stab_z.astype(int)
    z_indices = np.where(np.logical_not(np.all(z == 0, axis=1)))[0]

    z = z[z_indices]
    return x, z
