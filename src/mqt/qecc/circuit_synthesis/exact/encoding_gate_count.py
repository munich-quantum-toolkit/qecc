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

from .gate_operations import (
    CNOTGate,
    IdentityGate,
    get_standard_clifford_gate_set,
    get_standard_css_gate_set,
)
from .initialization import constrain_initial_clifford_tableau, constrain_initial_css_matrix
from .terminal import add_clifford_isometry_terminal, add_css_isometry_terminal
from .vars import CliffordGateCountVars

if TYPE_CHECKING:
    import numpy.typing as npt

    from ...codes.core.pauli import CheckMatrix, StabilizerTableau
    from .gate_operations import SymbolicGateOperation


def _declare_clifford_gate_count_vars(
    target: StabilizerTableau,
    n: int,
    num_rows: int,
    max_gates: int,
    n_bits: int,
    gate_set: dict[str, type[SymbolicGateOperation]],
) -> tuple[
    dict[str, list[z3.BoolRef]],
    list[z3.BitVecRef],
    list[z3.BitVecRef],
    npt.NDArray[np.object_],
    npt.NDArray[np.object_],
]:
    """Allocate the SAT variables for a Clifford gate-count encoding.

    Returns ``(gate_sel, alpha_vars, beta_vars, tableau_x, tableau_z)``. ``gate_sel`` has
    one per-slot selection-boolean list for every non-ID gate ("ID" is a depth-only
    concept); ``alpha``/``beta`` are the per-slot qubit-index bitvectors; the tableaus have
    ``max_gates + 1`` slices.

    When ``max_gates > 0``, slice 0 holds the target's Boolean constants directly (rather
    than fresh pinned variables) so the first transition folds against them. For the
    degenerate 0-gate case slice 0 is also the terminal slice and stays symbolic (pinned to
    the target by the caller).
    """
    gate_sel: dict[str, list[z3.BoolRef]] = {
        name: [z3.Bool(f"{name.lower()}_{slot}") for slot in range(max_gates)] for name in gate_set if name != "ID"
    }
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
    if max_gates > 0:
        for row in range(num_rows):
            for q in range(n):
                tableau_x[0, row, q] = bool(target.tableau.data[row, q])
                tableau_z[0, row, q] = bool(target.tableau.data[row, q + n])
    return gate_sel, alpha_vars, beta_vars, tableau_x, tableau_z


def _add_clifford_gate_count_slot_constraints(
    solver: z3.Solver,
    slot: int,
    n: int,
    n_bits: int,
    gate_sel: dict[str, list[z3.BoolRef]],
    alpha_vars: list[z3.BitVecRef],
    beta_vars: list[z3.BitVecRef],
    gate_set: dict[str, type[SymbolicGateOperation]],
) -> None:
    """Add the well-formedness constraints for one gate slot.

    Requires exactly one gate in the slot, bounds the qubit indices, forces distinct
    control/target for CX, and canonically orders symmetric two-qubit gates (alpha < beta).
    """
    solver.add(z3.PbEq([(v[slot], 1) for v in gate_sel.values()], 1))

    if (1 << n_bits) > n:
        solver.add(z3.ULT(alpha_vars[slot], n))
        solver.add(z3.ULT(beta_vars[slot], n))

    solver.add(z3.Implies(gate_sel["CX"][slot], alpha_vars[slot] != beta_vars[slot]))

    for gate_name, sel_list in gate_sel.items():
        if gate_set[gate_name].IS_TWO_QUBIT and gate_set[gate_name].IS_SYMMETRIC:
            solver.add(z3.Implies(sel_list[slot], z3.ULT(alpha_vars[slot], beta_vars[slot])))


def _add_clifford_gate_count_transitions(
    solver: z3.Solver,
    slot: int,
    n: int,
    gate_sel: dict[str, list[z3.BoolRef]],
    alpha_vars: list[z3.BitVecRef],
    beta_vars: list[z3.BitVecRef],
    gate_set: dict[str, type[SymbolicGateOperation]],
    tableau_x: npt.NDArray[np.object_],
    tableau_z: npt.NDArray[np.object_],
) -> None:
    """Add the tableau-transition constraints for one gate slot.

    For each gate and candidate qubit assignment, guards the gate's own
    :meth:`~SymbolicGateOperation.clifford_tableau_effect` with the corresponding selection
    condition. Qubits not touched by the chosen gate are preserved (treated as identity).
    """
    curr_x, curr_z = tableau_x[slot], tableau_z[slot]
    next_x, next_z = tableau_x[slot + 1], tableau_z[slot + 1]

    for gate_name, sel in gate_sel.items():
        gate_cls = gate_set[gate_name]
        if gate_cls.IS_TWO_QUBIT:
            for i in range(n):
                for j in range(n):
                    if i == j or (gate_cls.IS_SYMMETRIC and i > j):
                        continue
                    condition = z3.And(sel[slot], alpha_vars[slot] == i, beta_vars[slot] == j)
                    effect = gate_cls.from_qubits(i, j).clifford_tableau_effect(curr_x, curr_z, next_x, next_z)
                    solver.add(z3.Implies(condition, effect))
        else:
            for i in range(n):
                condition = z3.And(sel[slot], alpha_vars[slot] == i)
                effect = gate_cls.from_qubits(i, i).clifford_tableau_effect(curr_x, curr_z, next_x, next_z)
                solver.add(z3.Implies(condition, effect))

    # Preserve qubits not touched by any gate at this slot (an untouched qubit is an identity).
    for q in range(n):
        untouched = [
            z3.Not(z3.And(sel_list[slot], alpha_vars[slot] == q))
            if not gate_set[gate_name].IS_TWO_QUBIT
            else z3.Not(z3.And(sel_list[slot], z3.Or(alpha_vars[slot] == q, beta_vars[slot] == q)))
            for gate_name, sel_list in gate_sel.items()
        ]
        effect = IdentityGate(q).clifford_tableau_effect(curr_x, curr_z, next_x, next_z)
        solver.add(z3.Implies(z3.And(*untouched), effect))


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

    n = target.n
    num_rows = target.n_rows
    n_bits = max(1, int(np.ceil(np.log2(n)))) if n > 1 else 1

    solver = z3.Solver()
    gate_sel, alpha_vars, beta_vars, tableau_x, tableau_z = _declare_clifford_gate_count_vars(
        target, n, num_rows, max_gates, n_bits, gate_set
    )
    if max_gates == 0:
        constrain_initial_clifford_tableau(solver, target, tableau_x, tableau_z, n, num_rows)

    for slot in range(max_gates):
        _add_clifford_gate_count_slot_constraints(solver, slot, n, n_bits, gate_sel, alpha_vars, beta_vars, gate_set)
        _add_clifford_gate_count_transitions(
            solver, slot, n, gate_sel, alpha_vars, beta_vars, gate_set, tableau_x, tableau_z
        )

    add_clifford_isometry_terminal(
        solver,
        n,
        k,
        tableau_x[max_gates],
        tableau_z[max_gates],
        allow_qubit_permutation,
    )

    return CliffordGateCountVars(solver, gate_sel, alpha_vars, beta_vars, gate_set)


def _declare_css_gate_count_vars(
    target: CheckMatrix,
    n: int,
    num_rows: int,
    max_gates: int,
    n_bits: int,
) -> tuple[list[z3.BitVecRef], list[z3.BitVecRef], npt.NDArray[np.object_]]:
    """Allocate the qubit-index and check-matrix SAT variables for a CSS gate-count encoding.

    Returns ``(alpha_vars, beta_vars, matrix)`` where ``matrix`` has ``max_gates + 1`` slices.
    When ``max_gates > 0``, slice 0 holds the target's Boolean constants directly; for the
    degenerate 0-gate case it stays symbolic (pinned to the target by the caller).
    """
    alpha_vars = [z3.BitVec(f"alpha_{slot}", n_bits) for slot in range(max_gates)]
    beta_vars = [z3.BitVec(f"beta_{slot}", n_bits) for slot in range(max_gates)]
    matrix = np.array(
        [
            [[z3.Bool(f"m_{slot}_{row}_{q}") for q in range(n)] for row in range(num_rows)]
            for slot in range(max_gates + 1)
        ],
        dtype=object,
    )
    if max_gates > 0:
        for row in range(num_rows):
            for q in range(n):
                matrix[0, row, q] = bool(target.matrix[row, q])
    return alpha_vars, beta_vars, matrix


def _add_css_gate_count_slot_constraints(
    solver: z3.Solver,
    slot: int,
    n: int,
    alpha_vars: list[z3.BitVecRef],
    beta_vars: list[z3.BitVecRef],
) -> None:
    """Bound the qubit indices and require distinct control/target for the slot's CNOT."""
    if n > 1 and n & n - 1 != 0:
        solver.add(z3.ULT(alpha_vars[slot], n))
        solver.add(z3.ULT(beta_vars[slot], n))

    solver.add(alpha_vars[slot] != beta_vars[slot])


def _add_css_gate_count_transitions(
    solver: z3.Solver,
    slot: int,
    n: int,
    alpha_vars: list[z3.BitVecRef],
    beta_vars: list[z3.BitVecRef],
    is_x_type: bool,
    matrix: npt.NDArray[np.object_],
) -> None:
    """Add the CNOT column-operation constraints for one gate slot and preserve untouched columns.

    Delegates to :meth:`CNOTGate.css_matrix_effect`; for a Z-type target the propagation
    direction is reversed simply by swapping the control and target at instantiation.
    """
    curr, next_m = matrix[slot], matrix[slot + 1]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            condition = z3.And(alpha_vars[slot] == i, beta_vars[slot] == j)
            control, target = (i, j) if is_x_type else (j, i)
            effect = CNOTGate(control, target).css_matrix_effect(curr, next_m)
            solver.add(z3.Implies(condition, effect))

    for q in range(n):
        qubit_untouched = z3.And(z3.Not(alpha_vars[slot] == q), z3.Not(beta_vars[slot] == q))
        solver.add(z3.Implies(qubit_untouched, IdentityGate(q).css_matrix_effect(curr, next_m)))


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
    n_bits = max(1, int(np.ceil(np.log2(n)))) if n > 1 else 1

    solver = z3.Solver()
    alpha_vars, beta_vars, matrix = _declare_css_gate_count_vars(target, n, num_rows, max_gates, n_bits)
    if max_gates == 0:
        constrain_initial_css_matrix(solver, target, matrix, n, num_rows)

    for slot in range(max_gates):
        _add_css_gate_count_slot_constraints(solver, slot, n, alpha_vars, beta_vars)
        _add_css_gate_count_transitions(solver, slot, n, alpha_vars, beta_vars, is_x_type, matrix)

    add_css_isometry_terminal(
        solver,
        n,
        k,
        m_x,
        matrix[max_gates],
    )

    return solver, alpha_vars, beta_vars
