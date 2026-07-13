# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for SqrtXGate and the {H, SX, CX, ID} gate set."""

from __future__ import annotations

import numpy as np
import pytest
import z3

from mqt.qecc.circuit_synthesis.exact.gate_operations import (
    SqrtXGate,
    get_clifford_sx_gate_set,
    get_standard_clifford_gate_set,
)
from mqt.qecc.circuit_synthesis.exact.search import synthesize_isometry_exact
from mqt.qecc.circuit_synthesis.exact.types import Objective, SynthesisStatus, TargetKind
from mqt.qecc.circuit_synthesis.exact.verification import verify_stabilizer_state
from mqt.qecc.codes.core.pauli import StabilizerTableau

# ---------------------------------------------------------------------------
# SqrtXGate class properties
# ---------------------------------------------------------------------------


def test_sqrt_x_gate_properties() -> None:
    """SX gate has correct class attributes and Stim names."""
    sx = SqrtXGate(0)
    assert sx.qubit == 0
    assert sx.qubits() == {0}
    assert sx.to_stim_gate() == ("SQRT_X", [0])
    assert sx.inverse_stim_gate() == ("SQRT_X_DAG", [0])
    assert not SqrtXGate.IS_TWO_QUBIT
    assert not SqrtXGate.IS_SYMMETRIC
    assert SqrtXGate.IS_SELF_INVERSE


def test_sqrt_x_not_applicable_to_css() -> None:
    """SX gate raises NotImplementedError in CSS context."""
    sx = SqrtXGate(0)
    matrix_curr = np.array([[z3.Bool("m_0")]], dtype=object)
    matrix_next = np.array([[z3.Bool("m_1")]], dtype=object)
    with pytest.raises(NotImplementedError, match="SX gate cannot be applied"):
        sx.css_matrix_effect(matrix_curr, matrix_next)


def test_sqrt_x_gate_set_contents() -> None:
    """get_clifford_sx_gate_set returns {H, SX, CX, ID}."""
    gs = get_clifford_sx_gate_set()
    assert set(gs) == {"H", "SX", "CX", "ID"}
    assert gs["SX"] is SqrtXGate


def test_sqrt_x_not_in_standard_gate_set() -> None:
    """SX is not in the standard Clifford gate set."""
    assert "SX" not in get_standard_clifford_gate_set()


# ---------------------------------------------------------------------------
# SX tableau transition
# ---------------------------------------------------------------------------


def test_sqrt_x_transition_x_unchanged() -> None:
    """SX leaves X column unchanged: X_out = X ⊕ Z; with Z=0 → X_out = X."""
    n, _num_rows = 1, 1
    solver = z3.Solver()
    curr_x = np.array([[z3.Bool("cx_0")]], dtype=object)
    curr_z = np.array([[z3.Bool(f"cz_{i}") for i in range(n)]], dtype=object)
    next_x = np.array([[z3.Bool("nx_0")]], dtype=object)
    next_z = np.array([[z3.Bool("nz_0")]], dtype=object)

    solver.add(curr_x[0, 0])
    solver.add(z3.Not(curr_z[0, 0]))

    solver.add(SqrtXGate(0).clifford_tableau_effect(curr_x, curr_z, next_x, next_z))

    assert solver.check() == z3.sat
    model = solver.model()
    assert model.eval(next_x[0, 0], model_completion=True)  # X ⊕ 0 = X


def test_sqrt_x_transition_z_maps_to_xz() -> None:
    """SX maps Z to Y=XZ: with X=0, Z=1 → X_out=1, Z_out=1."""
    _n, _num_rows = 1, 1
    solver = z3.Solver()
    curr_x = np.array([[z3.Bool("cx2_0")]], dtype=object)
    curr_z = np.array([[z3.Bool("cz2_0")]], dtype=object)
    next_x = np.array([[z3.Bool("nx2_0")]], dtype=object)
    next_z = np.array([[z3.Bool("nz2_0")]], dtype=object)

    solver.add(z3.Not(curr_x[0, 0]))
    solver.add(curr_z[0, 0])

    solver.add(SqrtXGate(0).clifford_tableau_effect(curr_x, curr_z, next_x, next_z))

    assert solver.check() == z3.sat
    model = solver.model()
    assert model.eval(next_x[0, 0], model_completion=True)  # Z → Y: X_out = 1
    assert model.eval(next_z[0, 0], model_completion=True)  # Z_out = 1


def test_sqrt_x_transition_matches_stim() -> None:
    """SX binary tableau matches H·S·H action on identity tableau."""
    n = 2
    num_rows = 4
    init = StabilizerTableau.identity(n)

    # SX = H · S · H
    stim_tab = init.copy()
    stim_tab.apply_h(0)
    stim_tab.apply_s(0)
    stim_tab.apply_h(0)

    solver = z3.Solver()
    curr_x = np.array([[z3.Bool(f"cx_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    curr_z = np.array([[z3.Bool(f"cz_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    next_x = np.array([[z3.Bool(f"nx_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    next_z = np.array([[z3.Bool(f"nz_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)

    for r in range(num_rows):
        for q in range(n):
            solver.add(curr_x[r, q] == bool(init.tableau.matrix[r, q]))
            solver.add(curr_z[r, q] == bool(init.tableau.matrix[r, q + n]))

    # Only qubit 0 is transformed; copy qubit 1 explicitly.
    sx = SqrtXGate(0)
    solver.add(sx.clifford_tableau_effect(curr_x, curr_z, next_x, next_z))
    for r in range(num_rows):
        solver.add(next_x[r, 1] == curr_x[r, 1])
        solver.add(next_z[r, 1] == curr_z[r, 1])

    assert solver.check() == z3.sat
    model = solver.model()
    for r in range(num_rows):
        for q in range(n):
            sym_x = z3.is_true(model.eval(next_x[r, q], model_completion=True))
            sym_z = z3.is_true(model.eval(next_z[r, q], model_completion=True))
            stim_x = bool(stim_tab.tableau.matrix[r, q])
            stim_z = bool(stim_tab.tableau.matrix[r, q + n])
            assert sym_x == stim_x, f"X mismatch at row={r}, q={q}"
            assert sym_z == stim_z, f"Z mismatch at row={r}, q={q}"


# ---------------------------------------------------------------------------
# End-to-end synthesis with SX gate set
# ---------------------------------------------------------------------------


def test_sx_gate_count_synthesis_stabilizer_state(
    bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """SX gate set synthesizes the Bell state."""
    stabilizers, _xl, _zl = bell_state
    sx_gs = get_clifford_sx_gate_set()

    result = synthesize_isometry_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=5,
        gate_set=sx_gs,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.gate_count is not None
    assert result.gate_count <= 3
    assert result.verified
    assert result.circuit is not None
    assert verify_stabilizer_state(result.circuit, stabilizers)


def test_sx_depth_synthesis_stabilizer_state(
    bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """SX depth synthesis synthesizes the Bell state."""
    stabilizers, _xl, _zl = bell_state
    sx_gs = get_clifford_sx_gate_set()

    result = synthesize_isometry_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        objective=Objective.DEPTH,
        lower_bound=0,
        upper_bound=5,
        gate_set=sx_gs,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.depth is not None
    assert result.depth <= 2
    assert result.verified
    assert result.circuit is not None
    assert verify_stabilizer_state(result.circuit, stabilizers)


def test_sx_circuit_contains_only_valid_gates(
    bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Synthesized circuit uses only gates from the SX-extended set."""
    stabilizers, _xl, _zl = bell_state
    sx_gs = get_clifford_sx_gate_set()

    result = synthesize_isometry_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=5,
        gate_set=sx_gs,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    stim_circ = result.circuit.to_stim_circuit()
    gate_names = {inst.name for inst in stim_circ}
    assert gate_names <= {"R", "H", "SQRT_X", "SQRT_X_DAG", "CX", "I", "X", "Y", "Z"}


def test_sx_synthesis_unsat(
    bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """SX gate set with 0 upper bound returns UNSAT for Bell state."""
    stabilizers, _xl, _zl = bell_state
    sx_gs = get_clifford_sx_gate_set()

    result = synthesize_isometry_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=0,
        gate_set=sx_gs,
    )

    assert result.status == SynthesisStatus.UNSAT
