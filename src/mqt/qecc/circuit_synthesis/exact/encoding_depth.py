# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Depth encoding builders for exact synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import z3

from .gate_operations import get_standard_clifford_gate_set, get_standard_css_gate_set
from .terminal import add_clifford_isometry_terminal, add_css_isometry_terminal

if TYPE_CHECKING:
    from ...codes.pauli import CheckMatrix, StabilizerTableau
    from .gate_operations import SymbolicGateOperation


def encode_clifford_depth(
    target: StabilizerTableau,
    k: int,
    max_depth: int,
    allow_qubit_permutation: bool = True,
    gate_set: dict[str, type[SymbolicGateOperation]] | None = None,
) -> tuple[z3.Solver, list[list[z3.BoolRef]], list[list[z3.BoolRef]], list[list[z3.BoolRef]], list[list[z3.BoolRef]]]:
    """Encode Clifford isometry synthesis with depth optimization.

    Uses the provided gate set to dynamically support registered Clifford gates.
    Defaults to {H, S, CX, ID} if no gate set is provided.

    Args:
        target: Target stabilizer tableau (2k+m rows, where m=n-k stabilizers).
        k: Number of logical qubits.
        max_depth: Maximum circuit depth.
        allow_qubit_permutation: Allow final qubit permutation.
        gate_set: Optional custom gate set. If None, uses standard {H, S, CX, ID}.

    Returns:
        Tuple of (solver, h_vars, s_vars, cx_vars, id_vars).
    """
    if gate_set is None:
        gate_set = get_standard_clifford_gate_set()

    n = target.n
    num_rows = target.n_rows

    solver = z3.Solver()

    h_vars = [[z3.Bool(f"h_{layer}_{q}") for q in range(n)] for layer in range(max_depth)]
    s_vars = [[z3.Bool(f"s_{layer}_{q}") for q in range(n)] for layer in range(max_depth)]
    id_vars = [[z3.Bool(f"id_{layer}_{q}") for q in range(n)] for layer in range(max_depth)]

    cx_vars = [
        [z3.Bool(f"cx_{layer}_{ctrl}_{tgt}") for ctrl in range(n) for tgt in range(n) if ctrl != tgt]
        for layer in range(max_depth)
    ]

    tableau_x = np.array(
        [
            [[z3.Bool(f"tx_{layer}_{row}_{q}") for q in range(n)] for row in range(num_rows)]
            for layer in range(max_depth + 1)
        ],
        dtype=object,
    )

    tableau_z = np.array(
        [
            [[z3.Bool(f"tz_{layer}_{row}_{q}") for q in range(n)] for row in range(num_rows)]
            for layer in range(max_depth + 1)
        ],
        dtype=object,
    )

    for row in range(num_rows):
        for q in range(n):
            solver.add(tableau_x[0, row, q] == bool(target.tableau.matrix[row, q]))
            solver.add(tableau_z[0, row, q] == bool(target.tableau.matrix[row, q + n]))

    for layer in range(max_depth):
        for q in range(n):
            cx_involving_q = []
            for ctrl in range(n):
                if ctrl == q:
                    continue
                cx_idx = ctrl * (n - 1) + (q if q < ctrl else q - 1)
                cx_involving_q.append(cx_vars[layer][cx_idx])

            for tgt in range(n):
                if tgt == q:
                    continue
                cx_idx = q * (n - 1) + (tgt if tgt < q else tgt - 1)
                cx_involving_q.append(cx_vars[layer][cx_idx])

            solver.add(
                z3.PbEq(
                    [(h_vars[layer][q], 1), (s_vars[layer][q], 1), (id_vars[layer][q], 1)]
                    + [(v, 1) for v in cx_involving_q],
                    1,
                )
            )

        curr_x = tableau_x[layer]
        curr_z = tableau_z[layer]
        next_x = tableau_x[layer + 1]
        next_z = tableau_z[layer + 1]

        for q in range(n):
            for row in range(num_rows):
                h_effect = z3.If(
                    h_vars[layer][q],
                    z3.And(next_x[row, q] == curr_z[row, q], next_z[row, q] == curr_x[row, q]),
                    True,
                )
                solver.add(h_effect)

                s_effect = z3.If(
                    s_vars[layer][q],
                    z3.And(next_x[row, q] == curr_x[row, q], next_z[row, q] == z3.Xor(curr_z[row, q], curr_x[row, q])),
                    True,
                )
                solver.add(s_effect)

                id_effect = z3.If(
                    id_vars[layer][q],
                    z3.And(next_x[row, q] == curr_x[row, q], next_z[row, q] == curr_z[row, q]),
                    True,
                )
                solver.add(id_effect)

        cx_idx = 0
        for ctrl in range(n):
            for tgt in range(n):
                if ctrl == tgt:
                    continue

                for row in range(num_rows):
                    cx_effect = z3.If(
                        cx_vars[layer][cx_idx],
                        z3.And(
                            next_x[row, ctrl] == curr_x[row, ctrl],
                            next_x[row, tgt] == z3.Xor(curr_x[row, tgt], curr_x[row, ctrl]),
                            next_z[row, ctrl] == z3.Xor(curr_z[row, ctrl], curr_z[row, tgt]),
                            next_z[row, tgt] == curr_z[row, tgt],
                        ),
                        True,
                    )
                    solver.add(cx_effect)

                cx_idx += 1

    add_clifford_isometry_terminal(
        solver,
        n,
        k,
        tableau_x[max_depth],
        tableau_z[max_depth],
        allow_qubit_permutation,
    )

    return solver, h_vars, s_vars, cx_vars, id_vars


def encode_css_depth(
    target: CheckMatrix,
    k: int,
    m_x: int,
    max_depth: int,
    gate_set: dict[str, type[SymbolicGateOperation]] | None = None,
) -> tuple[z3.Solver, list[list[z3.BoolRef]], list[list[z3.BoolRef]]]:
    """Encode CSS CNOT isometry synthesis with depth optimization.

    Supports both X-type and Z-type check matrices.  For X-type targets,
    CNOT(ctrl, tgt) adds column ctrl to column tgt (X propagates ctrl→tgt).
    For Z-type targets, CNOT(ctrl, tgt) adds column tgt to column ctrl
    (Z propagates tgt→ctrl), so the pivot columns of the reduced matrix
    are the qubits initialized in |0⟩.

    Uses the provided gate set to dynamically support registered CSS gates.
    Defaults to {CX, ID} if no gate set is provided.

    Args:
        target: Target CSS matrix [L; H] (X-type or Z-type).
        k: Number of logical qubits.
        m_x: Number of stabilizer rows (rank of the stabilizer block).
        max_depth: Maximum circuit depth.
        gate_set: Optional custom gate set. If None, uses standard {CX, ID}.

    Returns:
        Tuple of (solver, cx_vars, id_vars).
    """
    if gate_set is None:
        gate_set = get_standard_css_gate_set()

    n = target.num_qubits()
    num_rows = target.num_rows()
    is_x_type = target.is_x_type()

    solver = z3.Solver()

    id_vars = [[z3.Bool(f"id_{layer}_{q}") for q in range(n)] for layer in range(max_depth)]

    cx_vars = [
        [z3.Bool(f"cx_{layer}_{ctrl}_{tgt}") for ctrl in range(n) for tgt in range(n) if ctrl != tgt]
        for layer in range(max_depth)
    ]

    matrix = np.array(
        [
            [[z3.Bool(f"m_{layer}_{row}_{q}") for q in range(n)] for row in range(num_rows)]
            for layer in range(max_depth + 1)
        ],
        dtype=object,
    )

    for row in range(num_rows):
        for q in range(n):
            solver.add(matrix[0, row, q] == bool(target.matrix[row, q]))

    for layer in range(max_depth):
        for q in range(n):
            cx_involving_q = []
            for ctrl in range(n):
                if ctrl == q:
                    continue
                cx_idx = ctrl * (n - 1) + (q if q < ctrl else q - 1)
                cx_involving_q.append(cx_vars[layer][cx_idx])

            for tgt in range(n):
                if tgt == q:
                    continue
                cx_idx = q * (n - 1) + (tgt if tgt < q else tgt - 1)
                cx_involving_q.append(cx_vars[layer][cx_idx])

            solver.add(z3.PbEq([(id_vars[layer][q], 1)] + [(v, 1) for v in cx_involving_q], 1))

        curr = matrix[layer]
        next_m = matrix[layer + 1]

        for q in range(n):
            for row in range(num_rows):
                id_effect = z3.If(
                    id_vars[layer][q],
                    next_m[row, q] == curr[row, q],
                    True,
                )
                solver.add(id_effect)

        cx_idx = 0
        for ctrl in range(n):
            for tgt in range(n):
                if ctrl == tgt:
                    continue

                # X-type: CNOT(ctrl,tgt) adds column ctrl to column tgt  (col_tgt ^= col_ctrl)
                # Z-type: CNOT(ctrl,tgt) adds column tgt to column ctrl  (col_ctrl ^= col_tgt)
                src, dst = (ctrl, tgt) if is_x_type else (tgt, ctrl)
                for row in range(num_rows):
                    cx_effect = z3.If(
                        cx_vars[layer][cx_idx],
                        z3.And(
                            next_m[row, src] == curr[row, src],
                            next_m[row, dst] == z3.Xor(curr[row, dst], curr[row, src]),
                        ),
                        True,
                    )
                    solver.add(cx_effect)

                cx_idx += 1

    add_css_isometry_terminal(
        solver,
        n,
        k,
        m_x,
        matrix[max_depth],
    )

    return solver, cx_vars, id_vars
