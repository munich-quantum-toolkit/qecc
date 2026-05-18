# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Gate-count encoding builders for exact synthesis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import z3

from .gate_operations import get_standard_clifford_gate_set, get_standard_css_gate_set
from .terminal import add_clifford_isometry_terminal, add_css_isometry_terminal
from .vars import CliffordGateCountVars

if TYPE_CHECKING:
    from ...codes.pauli import CheckMatrix, StabilizerTableau
    from .gate_operations import SymbolicGateOperation


def encode_clifford_gate_count(
    target: StabilizerTableau,
    k: int,
    max_gates: int,
    allow_qubit_permutation: bool = True,
    gate_set: dict[str, type[SymbolicGateOperation]] | None = None,
) -> CliffordGateCountVars:
    """Encode Clifford isometry synthesis with gate-count optimization.

    Args:
        target: Target stabilizer tableau (2k+m rows, where m=n-k stabilizers).
        k: Number of logical qubits.
        max_gates: Maximum number of gates.
        allow_qubit_permutation: Allow final qubit permutation.
        gate_set: Gate set to use. If None, uses standard {H, S, CX, ID}.
            Pass a gate set containing ``"CZ"`` to enable CZ gates.

    Returns:
        A :class:`CliffordGateCountVars` container holding the solver and all
        SAT variables.  The ``gate_sel`` dict has one entry per gate type
        (excluding ``"ID"``); ``alpha`` and ``beta`` hold per-slot qubit-index
        variables.
    """
    if gate_set is None:
        gate_set = get_standard_clifford_gate_set()

    use_cz = "CZ" in gate_set

    n = target.n
    num_rows = target.n_rows

    solver = z3.Solver()

    n_bits = max(1, int(np.ceil(np.log2(n)))) if n > 1 else 1

    # One selection boolean per slot for every non-ID gate in the gate set.
    # "ID" is a depth-encoding concept; gate-count slots are always occupied.
    gate_sel: dict[str, list[z3.BoolRef]] = {
        name: [z3.Bool(f"{name.lower()}_{slot}") for slot in range(max_gates)] for name in gate_set if name != "ID"
    }

    # Convenient local aliases for the inline transition logic below.
    h_vars = gate_sel["H"]
    s_vars = gate_sel["S"]
    cx_vars = gate_sel["CX"]
    cz_vars = gate_sel.get("CZ", [])

    alpha_vars = [z3.BitVec(f"alpha_{slot}", n_bits) for slot in range(max_gates)]
    beta_vars = [z3.BitVec(f"beta_{slot}", n_bits) for slot in range(max_gates)]

    tableau_x = np.array(
        [
            [[z3.Bool(f"tx_{slot}_{row}_{q}") for q in range(n)] for row in range(num_rows)]
            for slot in range(max_gates + 1)
        ],
        dtype=object,
    )

    tableau_z = np.array(
        [
            [[z3.Bool(f"tz_{slot}_{row}_{q}") for q in range(n)] for row in range(num_rows)]
            for slot in range(max_gates + 1)
        ],
        dtype=object,
    )

    for row in range(num_rows):
        for q in range(n):
            solver.add(tableau_x[0, row, q] == bool(target.tableau.matrix[row, q]))
            solver.add(tableau_z[0, row, q] == bool(target.tableau.matrix[row, q + n]))

    for slot in range(max_gates):
        pb_terms: list[tuple[z3.BoolRef, int]] = [(v[slot], 1) for v in gate_sel.values()]
        solver.add(z3.PbEq(pb_terms, 1))

        if n > 1 and n & n - 1 != 0:
            solver.add(z3.ULT(alpha_vars[slot], n))
            solver.add(z3.ULT(beta_vars[slot], n))

        solver.add(z3.Implies(cx_vars[slot], alpha_vars[slot] != beta_vars[slot]))

        # CZ is symmetric: canonically enforce alpha < beta so we only need i<j transitions.
        if use_cz:
            solver.add(z3.Implies(cz_vars[slot], z3.ULT(alpha_vars[slot], beta_vars[slot])))

        curr_x = tableau_x[slot]
        curr_z = tableau_z[slot]
        next_x = tableau_x[slot + 1]
        next_z = tableau_z[slot + 1]

        for i in range(n):
            h_condition = z3.And(h_vars[slot], alpha_vars[slot] == i)

            for row in range(num_rows):
                solver.add(z3.Implies(h_condition, next_x[row, i] == curr_z[row, i]))
                solver.add(z3.Implies(h_condition, next_z[row, i] == curr_x[row, i]))

            s_condition = z3.And(s_vars[slot], alpha_vars[slot] == i)

            for row in range(num_rows):
                solver.add(z3.Implies(s_condition, next_x[row, i] == curr_x[row, i]))
                solver.add(z3.Implies(s_condition, next_z[row, i] == z3.Xor(curr_z[row, i], curr_x[row, i])))

            for j in range(n):
                if i == j:
                    continue

                cx_condition = z3.And(cx_vars[slot], alpha_vars[slot] == i, beta_vars[slot] == j)

                for row in range(num_rows):
                    solver.add(z3.Implies(cx_condition, next_x[row, i] == curr_x[row, i]))
                    solver.add(z3.Implies(cx_condition, next_x[row, j] == z3.Xor(curr_x[row, j], curr_x[row, i])))
                    solver.add(z3.Implies(cx_condition, next_z[row, i] == z3.Xor(curr_z[row, i], curr_z[row, j])))
                    solver.add(z3.Implies(cx_condition, next_z[row, j] == curr_z[row, j]))

        # CZ transitions: only i < j pairs (alpha < beta enforced above).
        if use_cz:
            for i in range(n):
                for j in range(i + 1, n):
                    cz_condition = z3.And(cz_vars[slot], alpha_vars[slot] == i, beta_vars[slot] == j)
                    for row in range(num_rows):
                        solver.add(z3.Implies(cz_condition, next_x[row, i] == curr_x[row, i]))
                        solver.add(z3.Implies(cz_condition, next_z[row, i] == z3.Xor(curr_z[row, i], curr_x[row, j])))
                        solver.add(z3.Implies(cz_condition, next_x[row, j] == curr_x[row, j]))
                        solver.add(z3.Implies(cz_condition, next_z[row, j] == z3.Xor(curr_z[row, j], curr_x[row, i])))

        for q in range(n):
            not_h_on_q = z3.Not(z3.And(h_vars[slot], alpha_vars[slot] == q))
            not_s_on_q = z3.Not(z3.And(s_vars[slot], alpha_vars[slot] == q))

            not_cx_involving_q = []
            for other in range(n):
                if other == q:
                    continue
                not_cx_involving_q.append(
                    z3.Not(
                        z3.And(
                            cx_vars[slot],
                            z3.Or(
                                z3.And(alpha_vars[slot] == q, beta_vars[slot] == other),
                                z3.And(alpha_vars[slot] == other, beta_vars[slot] == q),
                            ),
                        )
                    )
                )

            if use_cz:
                not_cz_involving_q = z3.Not(z3.And(cz_vars[slot], z3.Or(alpha_vars[slot] == q, beta_vars[slot] == q)))
                qubit_untouched = z3.And(not_h_on_q, not_s_on_q, *not_cx_involving_q, not_cz_involving_q)
            else:
                qubit_untouched = z3.And(not_h_on_q, not_s_on_q, *not_cx_involving_q)

            for row in range(num_rows):
                solver.add(z3.Implies(qubit_untouched, next_x[row, q] == curr_x[row, q]))
                solver.add(z3.Implies(qubit_untouched, next_z[row, q] == curr_z[row, q]))

    add_clifford_isometry_terminal(
        solver,
        n,
        k,
        tableau_x[max_gates],
        tableau_z[max_gates],
        allow_qubit_permutation,
    )

    return CliffordGateCountVars(solver, gate_sel, alpha_vars, beta_vars, gate_set)


def encode_css_gate_count(
    target: CheckMatrix,
    k: int,
    m_x: int,
    max_gates: int,
    gate_set: dict[str, type[SymbolicGateOperation]] | None = None,
) -> tuple[z3.Solver, list[z3.BitVecRef], list[z3.BitVecRef]]:
    """Encode CSS CNOT isometry synthesis with gate-count optimization.

    Supports both X-type and Z-type check matrices.  For X-type targets,
    CNOT(ctrl, tgt) adds column ctrl to column tgt (X propagates ctrl→tgt).
    For Z-type targets, CNOT(ctrl, tgt) adds column tgt to column ctrl
    (Z propagates tgt→ctrl), so the pivot columns of the reduced matrix
    are the qubits initialized in |0⟩.

    Uses the provided gate set to dynamically support registered CSS gates.
    Defaults to {CX} if no gate set is provided.

    Args:
        target: Target CSS matrix [L; H] (X-type or Z-type).
        k: Number of logical qubits.
        m_x: Number of stabilizer rows (rank of the stabilizer block).
        max_gates: Maximum number of gates.
        gate_set: Optional custom gate set. If None, uses standard {CX, ID}.

    Returns:
        Tuple of (solver, alpha_vars, beta_vars).
    """
    if gate_set is None:
        gate_set = get_standard_css_gate_set()

    n = target.num_qubits()
    num_rows = target.num_rows()
    is_x_type = target.is_x_type()

    solver = z3.Solver()

    n_bits = max(1, int(np.ceil(np.log2(n)))) if n > 1 else 1

    alpha_vars = [z3.BitVec(f"alpha_{slot}", n_bits) for slot in range(max_gates)]
    beta_vars = [z3.BitVec(f"beta_{slot}", n_bits) for slot in range(max_gates)]

    matrix = np.array(
        [
            [[z3.Bool(f"m_{slot}_{row}_{q}") for q in range(n)] for row in range(num_rows)]
            for slot in range(max_gates + 1)
        ],
        dtype=object,
    )

    for row in range(num_rows):
        for q in range(n):
            solver.add(matrix[0, row, q] == bool(target.matrix[row, q]))

    for slot in range(max_gates):
        if n > 1 and n & n - 1 != 0:
            solver.add(z3.ULT(alpha_vars[slot], n))
            solver.add(z3.ULT(beta_vars[slot], n))

        solver.add(alpha_vars[slot] != beta_vars[slot])

        curr = matrix[slot]
        next_m = matrix[slot + 1]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                cx_condition = z3.And(alpha_vars[slot] == i, beta_vars[slot] == j)

                # X-type: CNOT(i,j) adds column i to column j  (col_j ^= col_i)
                # Z-type: CNOT(i,j) adds column j to column i  (col_i ^= col_j)
                src, dst = (i, j) if is_x_type else (j, i)
                for row in range(num_rows):
                    solver.add(z3.Implies(cx_condition, next_m[row, src] == curr[row, src]))
                    solver.add(z3.Implies(cx_condition, next_m[row, dst] == z3.Xor(curr[row, dst], curr[row, src])))

        for q in range(n):
            not_control = z3.Not(alpha_vars[slot] == q)
            not_target = z3.Not(beta_vars[slot] == q)
            qubit_untouched = z3.And(not_control, not_target)

            for row in range(num_rows):
                solver.add(z3.Implies(qubit_untouched, next_m[row, q] == curr[row, q]))

    add_css_isometry_terminal(
        solver,
        n,
        k,
        m_x,
        matrix[max_gates],
    )

    return solver, alpha_vars, beta_vars
