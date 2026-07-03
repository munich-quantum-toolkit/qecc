# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for CZ gate encoding, transitions, symmetry breaking, and end-to-end synthesis."""

from __future__ import annotations

import numpy as np
import pytest
import z3

from mqt.qecc.circuit_synthesis.exact.gate_operations import (
    CNOTGate,
    CZGate,
    HGate,
    IdentityGate,
    SGate,
    get_clifford_cz_gate_set,
    get_standard_clifford_gate_set,
)
from mqt.qecc.circuit_synthesis.exact.search import synthesize_isometry_exact
from mqt.qecc.circuit_synthesis.exact.symmetry import (
    add_clifford_depth_symmetry_breaking,
    add_clifford_gate_count_symmetry_breaking,
    cz_pair_idx,
)
from mqt.qecc.circuit_synthesis.exact.types import Objective, SynthesisStatus, TargetKind
from mqt.qecc.circuit_synthesis.exact.vars import CliffordDepthVars, CliffordGateCountVars
from mqt.qecc.circuit_synthesis.exact.verification import verify_stabilizer_state
from mqt.qecc.codes.pauli import StabilizerTableau

# ---------------------------------------------------------------------------
# CZGate class
# ---------------------------------------------------------------------------


def test_cz_gate_properties() -> None:
    """Test CZ gate basic properties."""
    cz = CZGate(0, 1)
    assert cz.qubit1 == 0
    assert cz.qubit2 == 1
    assert cz.qubits() == {0, 1}
    assert cz.to_stim_gate() == ("CZ", [0, 1])
    assert cz.inverse_stim_gate() == ("CZ", [0, 1])  # CZ is self-inverse


def test_cz_gate_not_applicable_to_css() -> None:
    """Test that CZ gate raises NotImplementedError for CSS encoding."""
    cz = CZGate(0, 1)
    z3.Solver()
    matrix_curr = np.array([[z3.Bool("m_0_0"), z3.Bool("m_0_1")]], dtype=object)
    matrix_next = np.array([[z3.Bool("m_1_0"), z3.Bool("m_1_1")]], dtype=object)

    with pytest.raises(NotImplementedError, match="CZ gate cannot be applied in CSS"):
        cz.css_matrix_effect(matrix_curr, matrix_next)


def test_cz_gate_set_contains_cz() -> None:
    """Test that get_clifford_cz_gate_set includes CZ in addition to standard gates."""
    gate_set = get_clifford_cz_gate_set()
    assert "H" in gate_set
    assert "S" in gate_set
    assert "CX" in gate_set
    assert "CZ" in gate_set
    assert "ID" in gate_set
    assert gate_set["CZ"] is CZGate


# ---------------------------------------------------------------------------
# CZ tableau transition
# ---------------------------------------------------------------------------


@pytest.fixture
def tableau_vars_2q() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Symbolic 2-qubit, 2-row tableau variables."""
    n, num_rows = 2, 2
    curr_x = np.array([[z3.Bool(f"cx_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    curr_z = np.array([[z3.Bool(f"cz_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    next_x = np.array([[z3.Bool(f"nx_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    next_z = np.array([[z3.Bool(f"nz_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    return curr_x, curr_z, next_x, next_z


def test_cz_transition_x_columns_unchanged(
    tableau_vars_2q: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """CZ leaves X columns intact."""
    curr_x, curr_z, next_x, next_z = tableau_vars_2q
    solver = z3.Solver()
    solver.add(curr_x[0, 0])  # X[0,0] = 1
    solver.add(z3.Not(curr_x[0, 1]))  # X[0,1] = 0

    solver.add(CZGate(0, 1).clifford_tableau_effect(curr_x, curr_z, next_x, next_z))

    assert solver.check() == z3.sat
    model = solver.model()
    assert model.eval(next_x[0, 0], model_completion=True)  # unchanged
    assert not model.eval(next_x[0, 1], model_completion=True)  # unchanged


def test_cz_transition_z_columns_updated(
    tableau_vars_2q: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """CZ XORs each Z column with the opposite X column.

    With curr_x[0,0]=1, curr_x[0,1]=0, curr_z[0,0]=0, curr_z[0,1]=0:
    - next_z[0,0] = Z[0,0] XOR X[0,1] = 0 XOR 0 = 0
    - next_z[0,1] = Z[0,1] XOR X[0,0] = 0 XOR 1 = 1
    """
    curr_x, curr_z, next_x, next_z = tableau_vars_2q
    solver = z3.Solver()
    solver.add(curr_x[0, 0])
    solver.add(z3.Not(curr_x[0, 1]))
    solver.add(z3.Not(curr_z[0, 0]))
    solver.add(z3.Not(curr_z[0, 1]))

    solver.add(CZGate(0, 1).clifford_tableau_effect(curr_x, curr_z, next_x, next_z))

    assert solver.check() == z3.sat
    model = solver.model()
    assert not model.eval(next_z[0, 0], model_completion=True)  # 0 XOR 0 = 0
    assert model.eval(next_z[0, 1], model_completion=True)  # 0 XOR 1 = 1


def test_cz_transition_matches_stim() -> None:
    """CZ symbolic transition matches Stim tableau simulation."""
    n = 2
    num_rows = 4
    init = StabilizerTableau.identity(n)

    stim_tab = init.copy()
    stim_tab.apply_cz(0, 1)

    solver = z3.Solver()
    curr_x = np.array([[z3.Bool(f"cx_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    curr_z = np.array([[z3.Bool(f"czz_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    next_x = np.array([[z3.Bool(f"nx_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)
    next_z = np.array([[z3.Bool(f"nz_{r}_{q}") for q in range(n)] for r in range(num_rows)], dtype=object)

    for r in range(num_rows):
        for q in range(n):
            solver.add(curr_x[r, q] == bool(init.tableau.matrix[r, q]))
            solver.add(curr_z[r, q] == bool(init.tableau.matrix[r, q + n]))

    solver.add(CZGate(0, 1).clifford_tableau_effect(curr_x, curr_z, next_x, next_z))
    # qubit 1 is not touched by CZ(0,1)... wait, both are. Leave identity for no other qubit needed.

    assert solver.check() == z3.sat
    model = solver.model()

    for r in range(num_rows):
        for q in range(n):
            sym_x = z3.is_true(model.eval(next_x[r, q], model_completion=True))
            sym_z = z3.is_true(model.eval(next_z[r, q], model_completion=True))
            stim_x = bool(stim_tab.tableau.matrix[r, q])
            stim_z = bool(stim_tab.tableau.matrix[r, q + n])
            assert sym_x == stim_x, f"row={r}, q={q}: X mismatch"
            assert sym_z == stim_z, f"row={r}, q={q}: Z mismatch"


# ---------------------------------------------------------------------------
# Symmetry breaking helpers
# ---------------------------------------------------------------------------


def test_cz_pair_idx_canonical() -> None:
    """cz_pair_idx normalises to i < j and gives correct flat index."""
    n = 4
    # Pairs in order: (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5
    assert cz_pair_idx(0, 1, n) == 0
    assert cz_pair_idx(0, 2, n) == 1
    assert cz_pair_idx(0, 3, n) == 2
    assert cz_pair_idx(1, 2, n) == 3
    assert cz_pair_idx(1, 3, n) == 4
    assert cz_pair_idx(2, 3, n) == 5
    # Symmetry: same result regardless of argument order
    assert cz_pair_idx(1, 0, n) == cz_pair_idx(0, 1, n)
    assert cz_pair_idx(3, 1, n) == cz_pair_idx(1, 3, n)


def test_adjacent_cz_cancellation_gate_count() -> None:
    """Gate-count symmetry breaking forbids two identical adjacent CZ gates."""
    max_gates = 2
    n_bits = 2

    h_vars = [z3.Bool(f"h_{i}") for i in range(max_gates)]
    s_vars = [z3.Bool(f"s_{i}") for i in range(max_gates)]
    c_vars = [z3.Bool(f"c_{i}") for i in range(max_gates)]
    cz_vars = [z3.Bool(f"cz_{i}") for i in range(max_gates)]
    alpha_vars = [z3.BitVec(f"alpha_{i}", n_bits) for i in range(max_gates)]
    beta_vars = [z3.BitVec(f"beta_{i}", n_bits) for i in range(max_gates)]

    solver = z3.Solver()
    enc = CliffordGateCountVars(
        solver=solver,
        gate_sel={"H": h_vars, "S": s_vars, "CX": c_vars, "CZ": cz_vars},
        alpha=alpha_vars,
        beta=beta_vars,
        gate_set={"H": HGate, "S": SGate, "CX": CNOTGate, "CZ": CZGate},
    )
    add_clifford_gate_count_symmetry_breaking(solver, max_gates, enc)

    # Two identical CZ(0,1) gates → should be UNSAT
    solver.add(cz_vars[0])
    solver.add(cz_vars[1])
    solver.add(alpha_vars[0] == 0)
    solver.add(beta_vars[0] == 1)
    solver.add(alpha_vars[1] == 0)
    solver.add(beta_vars[1] == 1)

    assert solver.check() == z3.unsat


def test_adjacent_cz_distinct_pairs_allowed_gate_count() -> None:
    """Adjacent CZ gates on different qubit pairs are still allowed."""
    max_gates = 2
    n_bits = 2

    h_vars = [z3.Bool(f"h2_{i}") for i in range(max_gates)]
    s_vars = [z3.Bool(f"s2_{i}") for i in range(max_gates)]
    c_vars = [z3.Bool(f"c2_{i}") for i in range(max_gates)]
    cz_vars = [z3.Bool(f"cz2_{i}") for i in range(max_gates)]
    alpha_vars = [z3.BitVec(f"alpha2_{i}", n_bits) for i in range(max_gates)]
    beta_vars = [z3.BitVec(f"beta2_{i}", n_bits) for i in range(max_gates)]

    solver = z3.Solver()
    enc = CliffordGateCountVars(
        solver=solver,
        gate_sel={"H": h_vars, "S": s_vars, "CX": c_vars, "CZ": cz_vars},
        alpha=alpha_vars,
        beta=beta_vars,
        gate_set={"H": HGate, "S": SGate, "CX": CNOTGate, "CZ": CZGate},
    )
    add_clifford_gate_count_symmetry_breaking(solver, max_gates, enc)

    # CZ(0,1) followed by CZ(0,2) — different pairs, allowed
    solver.add(cz_vars[0])
    solver.add(cz_vars[1])
    solver.add(alpha_vars[0] == 0)
    solver.add(beta_vars[0] == 1)
    solver.add(alpha_vars[1] == 0)
    solver.add(beta_vars[1] == 2)

    assert solver.check() == z3.sat


def test_adjacent_cz_cancellation_depth() -> None:
    """Depth symmetry breaking forbids two identical adjacent CZ layers."""
    n = 3
    max_depth = 2
    n_pairs = n * (n - 1) // 2  # 3 pairs for n=3

    h_vars = [[z3.Bool(f"hd_{l}_{q}") for q in range(n)] for l in range(max_depth)]
    s_vars = [[z3.Bool(f"sd_{l}_{q}") for q in range(n)] for l in range(max_depth)]
    cx_vars = [[z3.Bool(f"cxd_{l}_{k}") for k in range(n * (n - 1))] for l in range(max_depth)]
    id_vars = [[z3.Bool(f"idd_{l}_{q}") for q in range(n)] for l in range(max_depth)]
    cz_vars = [[z3.Bool(f"czd_{l}_{k}") for k in range(n_pairs)] for l in range(max_depth)]

    solver = z3.Solver()
    enc = CliffordDepthVars(
        solver=solver,
        gate_vars={"H": h_vars, "S": s_vars, "CX": cx_vars, "ID": id_vars, "CZ": cz_vars},
        n=n,
        gate_set={"H": HGate, "S": SGate, "CX": CNOTGate, "ID": IdentityGate, "CZ": CZGate},
    )
    add_clifford_depth_symmetry_breaking(solver, max_depth, enc)

    # CZ pair 0 = (0,1) in both layers → UNSAT
    solver.add(cz_vars[0][0])
    solver.add(cz_vars[1][0])

    assert solver.check() == z3.unsat


def test_left_alignment_cz_depth() -> None:
    """Depth left alignment: both qubits idle in layer l → no CZ in layer l+1."""
    n = 3
    max_depth = 2
    n_pairs = n * (n - 1) // 2

    h_vars = [[z3.Bool(f"hla_{l}_{q}") for q in range(n)] for l in range(max_depth)]
    s_vars = [[z3.Bool(f"sla_{l}_{q}") for q in range(n)] for l in range(max_depth)]
    cx_vars = [[z3.Bool(f"cxla_{l}_{k}") for k in range(n * (n - 1))] for l in range(max_depth)]
    id_vars = [[z3.Bool(f"idla_{l}_{q}") for q in range(n)] for l in range(max_depth)]
    cz_vars = [[z3.Bool(f"czla_{l}_{k}") for k in range(n_pairs)] for l in range(max_depth)]

    solver = z3.Solver()
    enc = CliffordDepthVars(
        solver=solver,
        gate_vars={"H": h_vars, "S": s_vars, "CX": cx_vars, "ID": id_vars, "CZ": cz_vars},
        n=n,
        gate_set={"H": HGate, "S": SGate, "CX": CNOTGate, "ID": IdentityGate, "CZ": CZGate},
    )
    add_clifford_depth_symmetry_breaking(solver, max_depth, enc)

    # Both qubits 0 and 1 idle in layer 0, then CZ(0,1) in layer 1 → UNSAT
    solver.add(id_vars[0][0])
    solver.add(id_vars[0][1])
    solver.add(cz_vars[1][cz_pair_idx(0, 1, n)])  # pair index for (0,1)

    assert solver.check() == z3.unsat


# ---------------------------------------------------------------------------
# End-to-end synthesis with CZ gate set
# ---------------------------------------------------------------------------


def test_cz_gate_count_synthesis_stabilizer_state(
    bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Synthesis with CZ gate set finds a circuit for the Bell state.

    Bell state stabilizers: XX, ZZ.
    Optimal with {H, S, CX, CZ, ID}: 1 H + 1 CZ = 2 gates.
    """
    stabilizers, _xl, _zl = bell_state
    cz_gs = get_clifford_cz_gate_set()

    result = synthesize_isometry_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=5,
        gate_set=cz_gs,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.gate_count is not None
    assert result.gate_count <= 2  # should match or beat {H, S, CX}
    assert result.verified
    assert result.circuit is not None
    assert verify_stabilizer_state(result.circuit, stabilizers)


def test_cz_depth_synthesis_stabilizer_state(
    bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Depth synthesis with CZ gate set finds a circuit for the Bell state."""
    stabilizers, _xl, _zl = bell_state
    cz_gs = get_clifford_cz_gate_set()

    result = synthesize_isometry_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        objective=Objective.DEPTH,
        lower_bound=0,
        upper_bound=5,
        gate_set=cz_gs,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.depth is not None
    assert result.depth <= 2
    assert result.verified
    assert result.circuit is not None
    assert verify_stabilizer_state(result.circuit, stabilizers)


def test_cz_gate_count_synthesis_clifford_unitary(
    cnot_unitary: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Synthesis with CZ gate set finds a circuit for the CNOT unitary.

    CNOT = H(1) CZ(0,1) H(1), so optimal gate count with CZ should be ≤3.
    """
    stabilizers, x_logicals, z_logicals = cnot_unitary
    cz_gs = get_clifford_cz_gate_set()

    result = synthesize_isometry_exact(
        target=stabilizers,
        target_kind=TargetKind.CLIFFORD_UNITARY,
        objective=Objective.GATE_COUNT,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
        lower_bound=0,
        upper_bound=5,
        gate_set=cz_gs,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None
    assert result.verified


def test_cz_gate_count_circuit_contains_cz(
    bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """The synthesized circuit may contain CZ instructions when CZ gate set is used."""
    stabilizers, _xl, _zl = bell_state
    cz_gs = get_clifford_cz_gate_set()

    result = synthesize_isometry_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=5,
        gate_set=cz_gs,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.circuit is not None

    # The circuit should be valid; we don't assert CZ is always used (the solver
    # may also choose H+CX), but the extracted circuit must be accepted by Stim.
    stim_circ = result.circuit.to_stim_circuit()
    gate_names = {inst.name for inst in stim_circ}
    # Must only contain gates from the CZ-extended Clifford set
    assert gate_names <= {"R", "H", "S", "S_DAG", "CX", "CZ", "I", "X", "Y", "Z"}


def test_cz_synthesis_with_symmetry_breaking(
    bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Synthesis with CZ gate set and symmetry breaking finds the same result."""
    stabilizers, _xl, _zl = bell_state
    cz_gs = get_clifford_cz_gate_set()

    result = synthesize_isometry_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=5,
        gate_set=cz_gs,
        use_symmetry_breaking=True,
    )

    assert result.status == SynthesisStatus.SUCCESS
    assert result.verified
    assert result.circuit is not None
    assert verify_stabilizer_state(result.circuit, stabilizers)


def test_cz_synthesis_unsat_returns_unsat(
    bell_state: tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau],
) -> None:
    """Synthesis with CZ gate set and too-tight upper bound returns UNSAT."""
    stabilizers, _xl, _zl = bell_state
    cz_gs = get_clifford_cz_gate_set()

    result = synthesize_isometry_exact(
        target=stabilizers,
        target_kind=TargetKind.STABILIZER_STATE,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=0,  # 0 gates can't prepare Bell state
        gate_set=cz_gs,
    )

    assert result.status == SynthesisStatus.UNSAT


def test_cz_not_in_standard_gate_set() -> None:
    """Standard Clifford gate set does not contain CZ."""
    gs = get_standard_clifford_gate_set()
    assert "CZ" not in gs
