# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Symmetry-breaking constraint builders for exact synthesis."""

from __future__ import annotations

import z3


def _cx_idx(ctrl: int, tgt: int, n: int) -> int:
    """Compute the flat index of CX(ctrl, tgt) in the cx_vars list.

    cx_vars are ordered as [(ctrl, tgt) for ctrl in range(n) for tgt in range(n) if ctrl != tgt],
    giving n*(n-1) entries with index = ctrl*(n-1) + (tgt if tgt < ctrl else tgt-1).
    """
    return ctrl * (n - 1) + (tgt if tgt < ctrl else tgt - 1)


def add_clifford_depth_symmetry_breaking(
    solver: z3.Solver,
    n: int,
    max_depth: int,
    h_vars: list[list[z3.BoolRef]],
    s_vars: list[list[z3.BoolRef]],
    cx_vars: list[list[z3.BoolRef]],
    id_vars: list[list[z3.BoolRef]],
) -> None:
    """Add symmetry-breaking constraints for Clifford depth encoding.

    Constraints:
    - Adjacent H cancellation: h_i^(l) => not h_i^(l+1)
    - Adjacent identical CNOT cancellation: cx_{i,j}^(l) => not cx_{i,j}^(l+1)
    - Left alignment: if id_i^(l), then no single-qubit gate on i in l+1
    - Left alignment: if id_i^(l) and id_j^(l), then no CNOT between i,j in l+1

    Args:
        solver: Z3 solver instance.
        n: Number of qubits.
        max_depth: Maximum depth bound.
        h_vars: H gate variables [depth][qubit].
        s_vars: S gate variables [depth][qubit].
        cx_vars: CNOT gate variables [depth][cx_idx] where cx_idx = ctrl*(n-1)+(tgt if tgt<ctrl else tgt-1).
        id_vars: Identity variables [depth][qubit].
    """
    # Adjacent H cancellation
    for ell in range(max_depth - 1):
        for i in range(n):
            solver.add(z3.Implies(h_vars[ell][i], z3.Not(h_vars[ell + 1][i])))

    # Adjacent identical CNOT cancellation
    for ell in range(max_depth - 1):
        for cx_i in range(len(cx_vars[ell])):
            solver.add(z3.Implies(cx_vars[ell][cx_i], z3.Not(cx_vars[ell + 1][cx_i])))

    # Left alignment for single-qubit gates
    for ell in range(max_depth - 1):
        for i in range(n):
            solver.add(
                z3.Implies(
                    id_vars[ell][i],
                    z3.And(z3.Not(h_vars[ell + 1][i]), z3.Not(s_vars[ell + 1][i])),
                )
            )

    # Left alignment for CNOTs
    for ell in range(max_depth - 1):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                solver.add(
                    z3.Implies(
                        z3.And(id_vars[ell][i], id_vars[ell][j]),
                        z3.And(
                            z3.Not(cx_vars[ell + 1][_cx_idx(i, j, n)]),
                            z3.Not(cx_vars[ell + 1][_cx_idx(j, i, n)]),
                        ),
                    )
                )


def add_clifford_gate_count_symmetry_breaking(
    solver: z3.Solver,
    max_gates: int,
    h_vars: list[z3.BoolRef],
    _s_vars: list[z3.BoolRef],
    c_vars: list[z3.BoolRef],
    alpha_vars: list[z3.BitVecRef],
    beta_vars: list[z3.BitVecRef],
) -> None:
    """Add symmetry-breaking constraints for Clifford gate-count encoding.

    Constraints:
    - Adjacent H cancellation: not (h^(l) and h^(l+1) and alpha^(l) = alpha^(l+1))
    - Adjacent identical CNOT cancellation: similar for c^(l)

    Args:
        solver: Z3 solver instance.
        max_gates: Maximum gate count bound.
        h_vars: H gate selection variables [slot].
        s_vars: S gate selection variables [slot].
        c_vars: CNOT gate selection variables [slot].
        alpha_vars: Index variables for H/S/CNOT control [slot].
        beta_vars: Index variables for CNOT target [slot].
    """
    for ell in range(max_gates - 1):
        # Adjacent H cancellation
        solver.add(z3.Not(z3.And(h_vars[ell], h_vars[ell + 1], alpha_vars[ell] == alpha_vars[ell + 1])))

        # Adjacent identical CNOT cancellation
        solver.add(
            z3.Not(
                z3.And(
                    c_vars[ell],
                    c_vars[ell + 1],
                    alpha_vars[ell] == alpha_vars[ell + 1],
                    beta_vars[ell] == beta_vars[ell + 1],
                )
            )
        )


def add_css_depth_symmetry_breaking(
    solver: z3.Solver,
    n: int,
    max_depth: int,
    cx_vars: list[list[z3.BoolRef]],
    id_vars: list[list[z3.BoolRef]],
) -> None:
    """Add symmetry-breaking constraints for CSS CNOT depth encoding.

    Constraints:
    - Adjacent identical CNOT cancellation
    - Left alignment for CNOTs

    Args:
        solver: Z3 solver instance.
        n: Number of qubits.
        max_depth: Maximum depth bound.
        cx_vars: CNOT gate variables [depth][cx_idx] where cx_idx = ctrl*(n-1)+(tgt if tgt<ctrl else tgt-1).
        id_vars: Identity variables [depth][qubit].
    """
    # Adjacent identical CNOT cancellation
    for ell in range(max_depth - 1):
        for cx_i in range(len(cx_vars[ell])):
            solver.add(z3.Implies(cx_vars[ell][cx_i], z3.Not(cx_vars[ell + 1][cx_i])))

    # Left alignment for CNOTs
    for ell in range(max_depth - 1):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                solver.add(
                    z3.Implies(
                        z3.And(id_vars[ell][i], id_vars[ell][j]),
                        z3.And(
                            z3.Not(cx_vars[ell + 1][_cx_idx(i, j, n)]),
                            z3.Not(cx_vars[ell + 1][_cx_idx(j, i, n)]),
                        ),
                    )
                )


def add_css_gate_count_symmetry_breaking(
    solver: z3.Solver,
    max_gates: int,
    alpha_vars: list[z3.BitVecRef],
    beta_vars: list[z3.BitVecRef],
) -> None:
    """Add symmetry-breaking constraints for CSS CNOT gate-count encoding.

    Constraints:
    - Adjacent identical CNOT cancellation

    Args:
        solver: Z3 solver instance.
        max_gates: Maximum gate count bound.
        alpha_vars: CNOT control index variables [slot].
        beta_vars: CNOT target index variables [slot].
    """
    for ell in range(max_gates - 1):
        solver.add(z3.Not(z3.And(alpha_vars[ell] == alpha_vars[ell + 1], beta_vars[ell] == beta_vars[ell + 1])))
