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

from .gate_operations import CNOTGate, HGate, SGate
from .terminal import add_clifford_isometry_terminal, add_css_isometry_terminal

if TYPE_CHECKING:
    from ...codes.pauli import CheckMatrix, StabilizerTableau


def encode_clifford_gate_count(
    target: StabilizerTableau,
    k: int,
    max_gates: int,
    allow_qubit_permutation: bool = True,
) -> tuple[z3.Solver, list, list, list, list, list]:
    """Encode Clifford isometry synthesis with gate-count optimization.

    Args:
        target: Target stabilizer tableau (2k+m rows, where m=n-k stabilizers).
        k: Number of logical qubits.
        max_gates: Maximum number of gates.
        allow_qubit_permutation: Allow final qubit permutation.

    Returns:
        Tuple of (solver, h_vars, s_vars, c_vars, alpha_vars, beta_vars).
    """
    n = target.n
    num_rows = target.n_rows

    solver = z3.Solver()

    n_bits = max(1, int(np.ceil(np.log2(n)))) if n > 1 else 1

    h_vars = [z3.Bool(f"h_{slot}") for slot in range(max_gates)]
    s_vars = [z3.Bool(f"s_{slot}") for slot in range(max_gates)]
    c_vars = [z3.Bool(f"c_{slot}") for slot in range(max_gates)]

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
        solver.add(z3.PbEq([(h_vars[slot], 1), (s_vars[slot], 1), (c_vars[slot], 1)], 1))

        if n > 1 and n & n - 1 != 0:
            solver.add(z3.ULT(alpha_vars[slot], n))
            solver.add(z3.ULT(beta_vars[slot], n))

        solver.add(z3.Implies(c_vars[slot], alpha_vars[slot] != beta_vars[slot]))

        curr_x = tableau_x[slot]
        curr_z = tableau_z[slot]
        next_x = tableau_x[slot + 1]
        next_z = tableau_z[slot + 1]

        for i in range(n):
            HGate(i)
            h_condition = z3.And(h_vars[slot], alpha_vars[slot] == i)
            for row in range(num_rows):
                solver.add(z3.Implies(h_condition, next_x[row, i] == curr_z[row, i]))
                solver.add(z3.Implies(h_condition, next_z[row, i] == curr_x[row, i]))

            SGate(i)
            s_condition = z3.And(s_vars[slot], alpha_vars[slot] == i)
            for row in range(num_rows):
                solver.add(z3.Implies(s_condition, next_x[row, i] == curr_x[row, i]))
                solver.add(z3.Implies(s_condition, next_z[row, i] == z3.Xor(curr_z[row, i], curr_x[row, i])))

            for j in range(n):
                if i == j:
                    continue

                CNOTGate(i, j)
                cx_condition = z3.And(c_vars[slot], alpha_vars[slot] == i, beta_vars[slot] == j)
                for row in range(num_rows):
                    solver.add(z3.Implies(cx_condition, next_x[row, i] == curr_x[row, i]))
                    solver.add(z3.Implies(cx_condition, next_x[row, j] == z3.Xor(curr_x[row, j], curr_x[row, i])))
                    solver.add(z3.Implies(cx_condition, next_z[row, i] == z3.Xor(curr_z[row, i], curr_z[row, j])))
                    solver.add(z3.Implies(cx_condition, next_z[row, j] == curr_z[row, j]))

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
                            c_vars[slot],
                            z3.Or(
                                z3.And(alpha_vars[slot] == q, beta_vars[slot] == other),
                                z3.And(alpha_vars[slot] == other, beta_vars[slot] == q),
                            ),
                        )
                    )
                )

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

    return solver, h_vars, s_vars, c_vars, alpha_vars, beta_vars


def encode_css_gate_count(
    target: CheckMatrix,
    k: int,
    m_x: int,
    max_gates: int,
) -> tuple[z3.Solver, list, list]:
    """Encode CSS CNOT isometry synthesis with gate-count optimization.

    Args:
        target: Target CSS matrix [L; H].
        k: Number of logical qubits.
        m_x: Number of X-stabilizers.
        max_gates: Maximum number of gates.

    Returns:
        Tuple of (solver, alpha_vars, beta_vars).
    """
    n = target.num_qubits()
    num_rows = target.num_rows()

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

                CNOTGate(i, j)
                cx_condition = z3.And(alpha_vars[slot] == i, beta_vars[slot] == j)

                for row in range(num_rows):
                    solver.add(z3.Implies(cx_condition, next_m[row, i] == curr[row, i]))
                    solver.add(z3.Implies(cx_condition, next_m[row, j] == z3.Xor(curr[row, j], curr[row, i])))

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
