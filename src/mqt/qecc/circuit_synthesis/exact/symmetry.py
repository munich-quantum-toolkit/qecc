# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Symmetry-breaking constraint builders for exact synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import z3

if TYPE_CHECKING:
    from .vars import CliffordDepthVars, CliffordGateCountVars


def _cx_idx(ctrl: int, tgt: int, n: int) -> int:
    """Compute the flat index of CX(ctrl, tgt) in the cx_vars list.

    cx_vars are ordered as [(ctrl, tgt) for ctrl in range(n) for tgt in range(n) if ctrl != tgt],
    giving n*(n-1) entries with index = ctrl*(n-1) + (tgt if tgt < ctrl else tgt-1).
    """
    return ctrl * (n - 1) + (tgt if tgt < ctrl else tgt - 1)


def cz_pair_idx(i: int, j: int, n: int) -> int:
    """Compute the flat index of CZ(i, j) in the cz_vars list.

    cz_vars are ordered as [(i, j) for i in range(n) for j in range(i+1, n)],
    giving n*(n-1)//2 entries.  Callers may pass either ordering; the function
    normalises to i < j.
    """
    if i > j:
        i, j = j, i
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def add_clifford_gate_count_symmetry_breaking(
    solver: z3.Solver,
    max_gates: int,
    enc: CliffordGateCountVars,
) -> None:
    """Add symmetry-breaking constraints for Clifford gate-count encoding.

    For every self-inverse gate, adjacent identical applications with the same
    qubit arguments are forbidden (since the two gates would cancel).  This
    applies to single-qubit gates (same alpha) and two-qubit gates (same alpha
    and beta).  Non-self-inverse gates (e.g. S) are left unconstrained because
    consecutive applications are not redundant.

    Args:
        solver: Z3 solver instance.
        max_gates: Maximum gate count bound.
        enc: Variable container returned by :func:`encode_clifford_gate_count`.
    """
    for gate_name, sel in enc.gate_sel.items():
        gate_cls = enc.gate_set[gate_name]
        if not gate_cls.IS_SELF_INVERSE:
            continue
        for ell in range(max_gates - 1):
            same_gate = z3.And(sel[ell], sel[ell + 1])
            same_alpha = enc.alpha[ell] == enc.alpha[ell + 1]
            if gate_cls.IS_TWO_QUBIT:
                same_beta = enc.beta[ell] == enc.beta[ell + 1]
                solver.add(z3.Not(z3.And(same_gate, same_alpha, same_beta)))
            else:
                solver.add(z3.Not(z3.And(same_gate, same_alpha)))


def _add_ordered_pair_left_alignment(
    solver: z3.Solver,
    all_layer_vars: list[list[z3.BoolRef]],
    id_vars: list[list[z3.BoolRef]],
    ell: int,
    n: int,
) -> None:
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            solver.add(
                z3.Implies(
                    z3.And(id_vars[ell][i], id_vars[ell][j]),
                    z3.And(
                        z3.Not(all_layer_vars[ell + 1][_cx_idx(i, j, n)]),
                        z3.Not(all_layer_vars[ell + 1][_cx_idx(j, i, n)]),
                    ),
                )
            )


def _add_unordered_pair_left_alignment(
    solver: z3.Solver,
    all_layer_vars: list[list[z3.BoolRef]],
    id_vars: list[list[z3.BoolRef]],
    ell: int,
    n: int,
) -> None:
    for i in range(n):
        for j in range(i + 1, n):
            solver.add(
                z3.Implies(
                    z3.And(id_vars[ell][i], id_vars[ell][j]),
                    z3.Not(all_layer_vars[ell + 1][cz_pair_idx(i, j, n)]),
                )
            )


def add_clifford_depth_symmetry_breaking(
    solver: z3.Solver,
    max_depth: int,
    enc: CliffordDepthVars,
) -> None:
    """Add symmetry-breaking constraints for Clifford depth encoding.

    Two classes of constraints are added:

    *Adjacent identical gate cancellation* — for every self-inverse gate,
    the same gate at the same qubit/pair index cannot appear in two
    consecutive layers.

    *Left alignment* — if both qubits involved in a potential two-qubit gate
    are idle (identity) in layer ``ell``, that gate is forbidden in layer
    ``ell+1``. Additionally, if a qubit is idle in layer ``ell``, no
    single-qubit gate (H or S) may target it in layer ``ell+1``.

    Args:
        solver: Z3 solver instance.
        max_depth: Maximum depth bound.
        enc: Variable container returned by :func:`encode_clifford_depth`.
    """
    n = enc.n
    id_vars = enc.gate_vars.get("ID", [])

    # Adjacent identical gate cancellation for all self-inverse non-identity gates.
    for gate_name, all_layer_vars in enc.gate_vars.items():
        if gate_name == "ID":
            continue
        gate_cls = enc.gate_set[gate_name]
        if not gate_cls.IS_SELF_INVERSE:
            continue
        for ell in range(max_depth - 1):
            for idx in range(len(all_layer_vars[ell])):
                solver.add(z3.Implies(all_layer_vars[ell][idx], z3.Not(all_layer_vars[ell + 1][idx])))

    if not id_vars:
        return

    # Left alignment: idle qubit → no single-qubit gate on that qubit next layer.
    for gate_name, all_layer_vars in enc.gate_vars.items():
        gate_cls = enc.gate_set[gate_name]
        if gate_cls.IS_TWO_QUBIT or gate_name == "ID":
            continue
        for ell in range(max_depth - 1):
            for q in range(n):
                solver.add(z3.Implies(id_vars[ell][q], z3.Not(all_layer_vars[ell + 1][q])))

    # Left alignment: both qubits idle → no two-qubit gate between them next layer.
    for gate_name, all_layer_vars in enc.gate_vars.items():
        gate_cls = enc.gate_set[gate_name]
        if not gate_cls.IS_TWO_QUBIT:
            continue
        for ell in range(max_depth - 1):
            if not gate_cls.IS_SYMMETRIC:
                _add_ordered_pair_left_alignment(solver, all_layer_vars, id_vars, ell, n)
            else:
                _add_unordered_pair_left_alignment(solver, all_layer_vars, id_vars, ell, n)


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
