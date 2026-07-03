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
from .initialization import constrain_initial_clifford_tableau, constrain_initial_css_matrix
from .terminal import add_clifford_isometry_terminal, add_css_isometry_terminal
from .vars import CliffordDepthVars

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...codes.pauli import CheckMatrix, StabilizerTableau
    from .gate_operations import SymbolicGateOperation


def _ordered_pair_pb_terms(
    layer_vars: list[z3.BoolRef],
    q: int,
    n: int,
) -> list[tuple[z3.BoolRef, int]]:
    """PbEq terms for an ordered-pair gate (CX-like) involving qubit q."""
    terms: list[tuple[z3.BoolRef, int]] = []
    for other in range(n):
        if other == q:
            continue
        terms.extend((
            (layer_vars[q * (n - 1) + (other if other < q else other - 1)], 1),
            (layer_vars[other * (n - 1) + (q if q < other else q - 1)], 1),
        ))
    return terms


def _symmetric_pair_pb_terms(
    layer_vars: list[z3.BoolRef],
    q: int,
    n: int,
) -> list[tuple[z3.BoolRef, int]]:
    """PbEq terms for a symmetric-pair gate (CZ-like) involving qubit q."""
    terms: list[tuple[z3.BoolRef, int]] = []
    for other in range(n):
        if other == q:
            continue
        pi, pj = min(q, other), max(q, other)
        terms.append((layer_vars[pi * (2 * n - pi - 1) // 2 + (pj - pi - 1)], 1))
    return terms


def _declare_clifford_depth_vars(
    n: int,
    num_rows: int,
    max_depth: int,
    gate_set: dict[str, type[SymbolicGateOperation]],
) -> tuple[dict[str, list[list[z3.BoolRef]]], npt.NDArray[np.object_], npt.NDArray[np.object_]]:
    """Allocate the per-layer gate and tableau SAT variables for a Clifford depth encoding.

    The gate-variable index structure per layer is: single-qubit (H, S, SX, ID, …) →
    ``[qubit]``; ordered two-qubit (CX-like) → ``[ordered-pair]``; symmetric two-qubit
    (CZ-like) → ``[unordered-pair]``. Returns ``(gate_vars, tableau_x, tableau_z)``.
    """
    gate_vars: dict[str, list[list[z3.BoolRef]]] = {}
    for gate_name, gate_cls in gate_set.items():
        key = gate_name.lower()
        if not gate_cls.IS_TWO_QUBIT:
            gate_vars[gate_name] = [[z3.Bool(f"{key}_{layer}_{q}") for q in range(n)] for layer in range(max_depth)]
        elif not gate_cls.IS_SYMMETRIC:
            gate_vars[gate_name] = [
                [z3.Bool(f"{key}_{layer}_{ctrl}_{tgt}") for ctrl in range(n) for tgt in range(n) if ctrl != tgt]
                for layer in range(max_depth)
            ]
        else:
            gate_vars[gate_name] = [
                [z3.Bool(f"{key}_{layer}_{i}_{j}") for i in range(n) for j in range(i + 1, n)]
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
    return gate_vars, tableau_x, tableau_z


def _add_clifford_depth_layer_constraints(
    solver: z3.Solver,
    layer: int,
    n: int,
    gate_vars: dict[str, list[list[z3.BoolRef]]],
    gate_set: dict[str, type[SymbolicGateOperation]],
) -> None:
    """Require exactly one gate to act on each qubit in the given layer."""
    for q in range(n):
        pb_terms: list[tuple[z3.BoolRef, int]] = []
        for gate_name, all_layer_vars in gate_vars.items():
            gate_cls = gate_set[gate_name]
            if not gate_cls.IS_TWO_QUBIT:
                pb_terms.append((all_layer_vars[layer][q], 1))
            elif not gate_cls.IS_SYMMETRIC:
                pb_terms.extend(_ordered_pair_pb_terms(all_layer_vars[layer], q, n))
            else:
                pb_terms.extend(_symmetric_pair_pb_terms(all_layer_vars[layer], q, n))
        solver.add(z3.PbEq(pb_terms, 1))


def _add_clifford_depth_transitions(
    solver: z3.Solver,
    layer: int,
    n: int,
    num_rows: int,
    gate_vars: dict[str, list[list[z3.BoolRef]]],
    tableau_x: npt.NDArray[np.object_],
    tableau_z: npt.NDArray[np.object_],
) -> None:
    """Add the tableau-transition constraints for one depth layer (H, S, SX, ID, CX, CZ)."""
    h_vars = gate_vars["H"]
    s_vars = gate_vars.get("S", [])
    sx_vars = gate_vars.get("SX", [])
    id_vars = gate_vars["ID"]
    cx_vars = gate_vars["CX"]
    cz_vars = gate_vars.get("CZ", [])

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

            if s_vars:
                s_effect = z3.If(
                    s_vars[layer][q],
                    z3.And(
                        next_x[row, q] == curr_x[row, q],
                        next_z[row, q] == z3.Xor(curr_z[row, q], curr_x[row, q]),
                    ),
                    True,
                )
                solver.add(s_effect)

            if sx_vars:
                sx_effect = z3.If(
                    sx_vars[layer][q],
                    z3.And(
                        next_x[row, q] == z3.Xor(curr_x[row, q], curr_z[row, q]),
                        next_z[row, q] == curr_z[row, q],
                    ),
                    True,
                )
                solver.add(sx_effect)

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

    if cz_vars:
        cz_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                for row in range(num_rows):
                    cz_effect = z3.If(
                        cz_vars[layer][cz_idx],
                        z3.And(
                            next_x[row, i] == curr_x[row, i],
                            next_z[row, i] == z3.Xor(curr_z[row, i], curr_x[row, j]),
                            next_x[row, j] == curr_x[row, j],
                            next_z[row, j] == z3.Xor(curr_z[row, j], curr_x[row, i]),
                        ),
                        True,
                    )
                    solver.add(cz_effect)
                cz_idx += 1


def encode_clifford_depth(
    target: StabilizerTableau,
    k: int,
    max_depth: int,
    allow_qubit_permutation: bool = True,
    gate_set: dict[str, type[SymbolicGateOperation]] | None = None,
) -> CliffordDepthVars:
    """Encode Clifford isometry synthesis with depth optimization.

    Uses the provided gate set to dynamically support registered Clifford gates.
    Defaults to {H, S, CX, ID} if no gate set is provided.  Pass a gate set
    containing ``"CZ"`` to also allow CZ gates.

    Args:
        target: Target stabilizer tableau (2k+m rows, where m=n-k stabilizers).
        k: Number of logical qubits.
        max_depth: Maximum circuit depth.
        allow_qubit_permutation: Allow final qubit permutation.
        gate_set: Optional custom gate set. If None, uses standard {H, S, CX, ID}.

    Returns:
        A :class:`CliffordDepthVars` container holding the solver and all SAT
        variables.  ``gate_vars`` maps each gate name to a
        ``[layer][idx]`` boolean array; ``n`` holds the qubit count needed by
        consumers to decode pair indices.
    """
    if gate_set is None:
        gate_set = get_standard_clifford_gate_set()

    n = target.n
    num_rows = target.n_rows

    solver = z3.Solver()
    gate_vars, tableau_x, tableau_z = _declare_clifford_depth_vars(n, num_rows, max_depth, gate_set)
    constrain_initial_clifford_tableau(solver, target, tableau_x, tableau_z, n, num_rows)

    for layer in range(max_depth):
        _add_clifford_depth_layer_constraints(solver, layer, n, gate_vars, gate_set)
        _add_clifford_depth_transitions(solver, layer, n, num_rows, gate_vars, tableau_x, tableau_z)

    add_clifford_isometry_terminal(
        solver,
        n,
        k,
        tableau_x[max_depth],
        tableau_z[max_depth],
        allow_qubit_permutation,
    )

    return CliffordDepthVars(solver, gate_vars, n, gate_set)


def _declare_css_depth_vars(
    n: int,
    num_rows: int,
    max_depth: int,
) -> tuple[list[list[z3.BoolRef]], list[list[z3.BoolRef]], npt.NDArray[np.object_]]:
    """Allocate the per-layer ID/CX gate and check-matrix SAT variables for a CSS depth encoding.

    Returns ``(id_vars, cx_vars, matrix)`` where ``matrix`` has ``max_depth + 1`` slices.
    """
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
    return id_vars, cx_vars, matrix


def _add_css_depth_layer_constraints(
    solver: z3.Solver,
    layer: int,
    n: int,
    id_vars: list[list[z3.BoolRef]],
    cx_vars: list[list[z3.BoolRef]],
) -> None:
    """Require exactly one gate (an ID or a CX involving it) to act on each qubit in the layer."""
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


def _add_css_depth_transitions(
    solver: z3.Solver,
    layer: int,
    n: int,
    num_rows: int,
    id_vars: list[list[z3.BoolRef]],
    cx_vars: list[list[z3.BoolRef]],
    is_x_type: bool,
    matrix: npt.NDArray[np.object_],
) -> None:
    """Add the ID/CNOT column-operation constraints for one depth layer."""
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
    id_vars, cx_vars, matrix = _declare_css_depth_vars(n, num_rows, max_depth)
    constrain_initial_css_matrix(solver, target, matrix, n, num_rows)

    for layer in range(max_depth):
        _add_css_depth_layer_constraints(solver, layer, n, id_vars, cx_vars)
        _add_css_depth_transitions(solver, layer, n, num_rows, id_vars, cx_vars, is_x_type, matrix)

    add_css_isometry_terminal(
        solver,
        n,
        k,
        m_x,
        matrix[max_depth],
    )

    return solver, cx_vars, id_vars
