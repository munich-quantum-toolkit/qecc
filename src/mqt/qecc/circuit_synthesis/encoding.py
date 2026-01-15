# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods for synthesizing encoding circuits for CSS codes."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import stim
import z3

from ..codes.pauli import StabilizerTableau
from .circuits import CNOTCircuit
from .synthesis_utils import (
    heuristic_gaussian_elimination,
    optimal_elimination,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..codes import CSSCode
    from ..codes.css_code import CSSCode


logger = logging.getLogger(__name__)


def heuristic_encoding_circuit(code: CSSCode, balance_checks: bool = False, **kwargs) -> CNOTCircuit:
    """Synthesize an encoding circuit for the given CSS code using a heuristic greedy search.

    Args:
        code: The CSS code to synthesize the encoding circuit for.
        balance_checks: Whether to balance the entries of the stabilizer matrix via row operations.

    Returns:
        The synthesized encoding circuit and the qubits that are used to encode the logical qubits.
    """
    logger.info("Starting encoding circuit synthesis.")

    checks, logicals, use_x_checks = _get_matrix_with_fewest_checks(code)
    n_checks = checks.shape[0]

    if balance_checks:
        reduce_checks_by_row_ops(checks, logicals)

    checks, cnots = heuristic_gaussian_elimination(np.vstack((checks, logicals)), **kwargs)

    # after reduction there still might be some overlap between initialized qubits and encoding qubits, we simply perform CNOTs to correct this

    encoding_qubits = np.where(checks[n_checks:, :].sum(axis=0) != 0)[0]
    initialization_qubits = np.where(checks[:n_checks, :].sum(axis=0) != 0)[0]
    # remove encoding qubits from initialization qubits
    initialization_qubits = np.setdiff1d(initialization_qubits, encoding_qubits)
    # TODO: this can fail, need a more robust way
    rows = []  # type: list[int]
    qubit_to_row = {}
    for qubit in initialization_qubits:
        cand = np.where(checks[:n_checks, qubit] == 1)[0]
        np.setdiff1d(cand, np.array(rows))
        rows.append(cand[0])
        qubit_to_row[qubit] = cand[0]

    for init_qubit in initialization_qubits:
        for encoding_qubit in encoding_qubits:
            row = qubit_to_row[init_qubit]
            if checks[row, encoding_qubit] == 1:
                cnots.append((init_qubit, encoding_qubit))
                checks[row, encoding_qubit] = 0

    cnots = cnots[::-1]
    return _build_css_encoder_from_cnot_list(n_checks, checks, cnots, use_x_checks)


from collections.abc import Callable

import numpy.typing as npt
from ortools.sat.python import cp_model


def depth_optimal_encoding_circuit_non_css(
    code,
    max_depth: int,
    max_two_qubit_gates: int | None = None,
    exact_two_qubit_count: bool = False,
):
    """OR-Tools (CP-SAT) version of your Z3 model.
    Matches gate codes: I=0, H=1, S=2, SQRTX=3, CXCTRL=4, CXTAR=5, CZ=6, CZ2=7.
    """
    n = code.n
    k = code.k
    m = n - k
    assert code.x_logicals is not None
    assert code.z_logicals is not None

    # Constants (just for readability)
    I, H, Sg, SX, CXCTRL, CXTAR, CZ, CZ2 = 0, 1, 2, 3, 4, 5, 6, 7

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def xor_eq(model: cp_model.CpModel, a, b, c, cond=None) -> None:
        """Enforce c == a XOR b. If cond provided, enforce only when cond is True."""
        # CNF for c <-> a xor b
        clauses = [
            [a, b, c.Not()],  # a ∨ b ∨ ¬c
            [a, b.Not(), c],  # a ∨ ¬b ∨ c
            [a.Not(), b, c],  # ¬a ∨ b ∨ c
            [a.Not(), b.Not(), c.Not()],  # ¬a ∨ ¬b ∨ ¬c
        ]
        for list in clauses:
            ct = model.AddBoolOr(list)
            if cond is not None:
                ct.OnlyEnforceIf(cond)

    def eq_if(model: cp_model.CpModel, a, b, cond) -> None:
        """A == b only if cond is True."""
        model.Add(a == b).OnlyEnforceIf(cond)

    def or_equal(model: cp_model.CpModel, out, list) -> None:
        """Out <-> Or(list)."""
        if not list:
            model.Add(out == 0)
            return
        # Or(list) => out
        # out => Or(list)
        for L in list:
            model.AddImplication(L, out)
        model.Add(sum(list) >= 1).OnlyEnforceIf(out)
        model.Add(sum(list) == 0).OnlyEnforceIf(out.Not())

    # ----------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------
    model = cp_model.CpModel()

    # Gate code per layer/qubit
    sqgs = [[model.NewIntVar(0, 7, f"sqg_{t}_{q}") for q in range(n)] for t in range(max_depth)]

    # Convenience: code==const booleans (reified equalities)
    isI = [[model.NewBoolVar(f"isI_{t}_{q}") for q in range(n)] for t in range(max_depth)]
    isH = [[model.NewBoolVar(f"isH_{t}_{q}") for q in range(n)] for t in range(max_depth)]
    isS = [[model.NewBoolVar(f"isS_{t}_{q}") for q in range(n)] for t in range(max_depth)]
    isSX = [[model.NewBoolVar(f"isSX_{t}_{q}") for q in range(n)] for t in range(max_depth)]
    isCXc = [[model.NewBoolVar(f"isCXc_{t}_{q}") for q in range(n)] for t in range(max_depth)]
    isCXt = [[model.NewBoolVar(f"isCXt_{t}_{q}") for q in range(n)] for t in range(max_depth)]
    isCZr = [
        [model.NewBoolVar(f"isCZr_{t}_{q}") for q in range(n)] for t in range(max_depth)
    ]  # CZ role (either 6 or 7)

    def bind_code(bv, code, lit) -> None:
        model.Add(bv == code).OnlyEnforceIf(lit)
        model.Add(bv != code).OnlyEnforceIf(lit.Not())

    for t in range(max_depth):
        for q in range(n):
            bind_code(sqgs[t][q], I, isI[t][q])
            bind_code(sqgs[t][q], H, isH[t][q])
            bind_code(sqgs[t][q], Sg, isS[t][q])
            bind_code(sqgs[t][q], SX, isSX[t][q])
            # 2q roles are grouped:
            # CXCTRL, CXTAR, CZ or CZ2
            # We'll derive isCZr by (sqg in {CZ,CZ2})
            tmp_cz = model.NewBoolVar(f"isCZ_{t}_{q}")
            tmp_cz2 = model.NewBoolVar(f"isCZ2_{t}_{q}")
            bind_code(sqgs[t][q], CXCTRL, isCXc[t][q])
            bind_code(sqgs[t][q], CXTAR, isCXt[t][q])
            bind_code(sqgs[t][q], CZ, tmp_cz)
            bind_code(sqgs[t][q], CZ2, tmp_cz2)
            # isCZr <-> tmp_cz OR tmp_cz2
            or_equal(model, isCZr[t][q], [tmp_cz, tmp_cz2])

            # Exclusivity: exactly one of the 8 codes
            model.Add(
                sum([isI[t][q], isH[t][q], isS[t][q], isSX[t][q], isCXc[t][q], isCXt[t][q], tmp_cz, tmp_cz2]) == 1
            )

    # 2q gate variables
    cxs = [[[model.NewBoolVar(f"cx_{t}_{u}_{v}") for v in range(n)] for u in range(n)] for t in range(max_depth)]
    czs = [
        [{v: model.NewBoolVar(f"cz_{t}_{u}_{v}") for v in range(u + 1, n)} for u in range(n)] for t in range(max_depth)
    ]

    # Per-layer: each qubit participates in at most one 2q gate (matching)
    for t in range(max_depth):
        for q in range(n):
            incident = []
            incident += [cxs[t][q][r] for r in range(n) if r != q]
            incident += [cxs[t][r][q] for r in range(n) if r != q]
            for r in range(n):
                if r == q:
                    continue
                u, v = (r, q) if r < q else (q, r)
                if u < v:
                    incident.append(czs[t][u][v])
            if incident:
                model.Add(sum(incident) <= 1)

    # Bidirectional consistency (your fix)
    for t in range(max_depth):
        for q in range(n):
            # compute Or of incident edges
            inc_ctrl = [cxs[t][q][r] for r in range(n) if r != q]
            inc_targ = [cxs[t][r][q] for r in range(n) if r != q]
            inc_cz = list(czs[t][q].values()) + [czs[t][u][q] for u in range(q)]

            or_equal(model, isCXc[t][q], inc_ctrl)
            or_equal(model, isCXt[t][q], inc_targ)
            or_equal(model, isCZr[t][q], inc_cz)

            # If any 2q incident, forbid 1q codes; if 1q code, forbid 2q incident
            any2q = model.NewBoolVar(f"any2q_{t}_{q}")
            or_equal(model, any2q, inc_ctrl + inc_targ + inc_cz)
            is1q = model.NewBoolVar(f"is1q_{t}_{q}")
            or_equal(model, is1q, [isI[t][q], isH[t][q], isS[t][q], isSX[t][q]])

            # any2q => not(1q)
            model.Add(is1q == 0).OnlyEnforceIf(any2q)
            # 1q => no incident 2q
            for e in inc_ctrl + inc_targ + inc_cz:
                model.Add(e == 0).OnlyEnforceIf(is1q)

    # "No two single-qubit gates in a row" (excluding I)
    for t in range(1, max_depth):
        for q in range(n):
            # If H/S/SX at t then NONE of H/S/SX at t-1
            model.Add(isH[t - 1][q] == 0).OnlyEnforceIf(isH[t][q])
            model.Add(isS[t - 1][q] == 0).OnlyEnforceIf(isH[t][q])
            model.Add(isSX[t - 1][q] == 0).OnlyEnforceIf(isH[t][q])

            model.Add(isH[t - 1][q] == 0).OnlyEnforceIf(isS[t][q])
            model.Add(isS[t - 1][q] == 0).OnlyEnforceIf(isS[t][q])
            model.Add(isSX[t - 1][q] == 0).OnlyEnforceIf(isS[t][q])

            model.Add(isH[t - 1][q] == 0).OnlyEnforceIf(isSX[t][q])
            model.Add(isS[t - 1][q] == 0).OnlyEnforceIf(isSX[t][q])
            model.Add(isSX[t - 1][q] == 0).OnlyEnforceIf(isSX[t][q])

    # the same 2q gate can't be applied in a row
    for t in range(1, max_depth):
        for u in range(n):
            for v in range(u + 1, n):
                model.AddImplication(cxs[t][u][v], cxs[t - 1][u][v].Not())
                model.AddImplication(cxs[t][v][u], cxs[t - 1][v][u].Not())
                model.AddImplication(czs[t][u][v], czs[t - 1][u][v].Not())

    # Two-qubit budget
    if max_two_qubit_gates is not None:
        twoq = []
        for t in range(max_depth):
            for u in range(n):
                twoq.extend(cxs[t][u][v] for v in range(n) if u != v)
            for u in range(n):
                twoq.extend(czs[t][u][v] for v in range(u + 1, n))
        if exact_two_qubit_count:
            model.Add(sum(twoq) == max_two_qubit_gates)
        else:
            model.Add(sum(twoq) <= max_two_qubit_gates)

    # ----------------------------------------------------------------------
    # Tableau variables: Bool for each entry
    # Order rows: stabilizers (m), X-logicals (k), Z-logicals (k)
    # ----------------------------------------------------------------------
    def make_tableau():
        return [
            np.array([[model.NewBoolVar(f"tx_{t}_{r}_{q}") for q in range(n)] for r in range(m + 2 * k)], dtype=object),
            np.array([[model.NewBoolVar(f"tz_{t}_{r}_{q}") for q in range(n)] for r in range(m + 2 * k)], dtype=object),
        ]

    # initialize t=0 with constants
    S = code.symplectic.astype(int)
    LX = code.x_logicals.tableau.matrix.astype(int)
    LZ = code.z_logicals.tableau.matrix.astype(int)

    rows_X0 = np.vstack([S[:, :n], LX[:, :n], LZ[:, :n]])  # (m+2k) x n
    rows_Z0 = np.vstack([S[:, n:], LX[:, n:], LZ[:, n:]])  # (m+2k) x n

    Tx = []
    Tz = []
    for t in range(max_depth + 1):
        x, z = make_tableau()
        Tx.append(x)
        Tz.append(z)

    for r in range(m + 2 * k):
        for q in range(n):
            model.Add(Tx[0][r, q] == rows_X0[r, q])
            model.Add(Tz[0][r, q] == rows_Z0[r, q])

    # ----------------------------------------------------------------------
    # Gate semantics (reified)
    # ----------------------------------------------------------------------
    # Single-qubit
    for t in range(1, max_depth + 1):
        tm1 = t - 1
        for q in range(n):
            for r in range(m + 2 * k):
                # Identity
                eq_if(model, Tx[t][r, q], Tx[tm1][r, q], isI[tm1][q])
                eq_if(model, Tz[t][r, q], Tz[tm1][r, q], isI[tm1][q])

                # H: swap
                eq_if(model, Tx[t][r, q], Tz[tm1][r, q], isH[tm1][q])
                eq_if(model, Tz[t][r, q], Tx[tm1][r, q], isH[tm1][q])

                # S: Z ^= X
                xor_eq(model, Tz[tm1][r, q], Tx[tm1][r, q], Tz[t][r, q], cond=isS[tm1][q])
                eq_if(model, Tx[t][r, q], Tx[tm1][r, q], isS[tm1][q])

                # SQRT_X: X ^= Z
                xor_eq(model, Tx[tm1][r, q], Tz[tm1][r, q], Tx[t][r, q], cond=isSX[tm1][q])
                eq_if(model, Tz[t][r, q], Tz[tm1][r, q], isSX[tm1][q])

    # Two-qubit (CNOT/CZ)
    for t in range(1, max_depth + 1):
        tm1 = t - 1

        # CX(u->v)
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                e = cxs[tm1][u][v]
                for r in range(m + 2 * k):
                    # X_v ^= X_u ; Z_u ^= Z_v ; others unchanged (handled by identity / exclusivity)
                    xor_eq(model, Tx[tm1][r, v], Tx[tm1][r, u], Tx[t][r, v], cond=e)
                    xor_eq(model, Tz[tm1][r, u], Tz[tm1][r, v], Tz[t][r, u], cond=e)
                    # Keep the untouched halves when e is active:
                    eq_if(model, Tx[t][r, u], Tx[tm1][r, u], e)
                    eq_if(model, Tz[t][r, v], Tz[tm1][r, v], e)

        # CZ(u,v), with u < v
        for u in range(n):
            for v in range(u + 1, n):
                e = czs[tm1][u][v]
                for r in range(m + 2 * k):
                    # X unchanged:
                    eq_if(model, Tx[t][r, u], Tx[tm1][r, u], e)
                    eq_if(model, Tx[t][r, v], Tx[tm1][r, v], e)
                    # Z_u ^= X_v ; Z_v ^= X_u
                    xor_eq(model, Tz[tm1][r, u], Tx[tm1][r, v], Tz[t][r, u], cond=e)
                    xor_eq(model, Tz[tm1][r, v], Tx[tm1][r, u], Tz[t][r, v], cond=e)

        # ----------------------------------------------------------------------

    # Final constraints (logicals up to stabilizer multiplications)
    # ----------------------------------------------------------------------
    # Helpers for AND/XOR over BoolVars into a BoolVar
    def and2(a, b, name):
        v = model.NewBoolVar(name)
        model.Add(v <= a)
        model.Add(v <= b)
        model.Add(v >= a + b - 1)
        return v

    def xor_list_to_var(list, name):
        v = model.NewBoolVar(name)
        if not list:
            model.Add(v == 0)
        else:
            # Enforce v == XOR(list)  via XOR(list + [~v]) == True
            model.AddBoolXOr([*list, v.Not()])
        return v

    # Stabilizer halves at final time (rows 0..m-1)
    # Tx/Tz are (max_depth+1) arrays of shape [(m+2k) x n]
    Sx_fin = Tx[max_depth]
    Sz_fin = Tz[max_depth]

    # Witnesses: which stabilizers are multiplied into each logical
    Wx = [[model.NewBoolVar(f"Wx_{i}_{s}") for s in range(m)] for i in range(k)]
    Wz = [[model.NewBoolVar(f"Wz_{i}_{s}") for s in range(m)] for i in range(k)]

    # Adjusted logical rows
    #   X-logical i is at row rx = m + i
    #   Z-logical i is at row rz = m + k + i
    LxX_adj = [[None for q in range(n)] for i in range(k)]
    LxZ_adj = [[None for q in range(n)] for i in range(k)]
    LzX_adj = [[None for q in range(n)] for i in range(k)]
    LzZ_adj = [[None for q in range(n)] for i in range(k)]

    for i in range(k):
        rx = m + i
        rz = m + k + i

        for q in range(n):
            # --- X-logical adjusted by Wx:  (LxX', LxZ') = (LxX, LxZ) ⊕ ⨁_s Wx[i,s]*(Sx[s,*], Sz[s,*])
            terms_x = [and2(Wx[i][s], Sx_fin[s, q], f"ax_{i}_{s}_{q}") for s in range(m)]
            terms_z = [and2(Wx[i][s], Sz_fin[s, q], f"az_{i}_{s}_{q}") for s in range(m)]
            acc_x = xor_list_to_var(terms_x, f"accx_{i}_{q}")
            acc_z = xor_list_to_var(terms_z, f"accz_{i}_{q}")

            LxX_adj[i][q] = xor_list_to_var([Tx[max_depth][rx, q], acc_x], f"LxXadj_{i}_{q}")
            LxZ_adj[i][q] = xor_list_to_var([Tz[max_depth][rx, q], acc_z], f"LxZadj_{i}_{q}")

            # --- Z-logical adjusted by Wz:  (LzX', LzZ') = (LzX, LzZ) ⊕ ⨁_s Wz[i,s]*(Sx[s,*], Sz[s,*])
            terms_x2 = [and2(Wz[i][s], Sx_fin[s, q], f"bx_{i}_{s}_{q}") for s in range(m)]
            terms_z2 = [and2(Wz[i][s], Sz_fin[s, q], f"bz_{i}_{s}_{q}") for s in range(m)]
            acc_x2 = xor_list_to_var(terms_x2, f"accx2_{i}_{q}")
            acc_z2 = xor_list_to_var(terms_z2, f"accz2_{i}_{q}")

            LzX_adj[i][q] = xor_list_to_var([Tx[max_depth][rz, q], acc_x2], f"LzXadj_{i}_{q}")
            LzZ_adj[i][q] = xor_list_to_var([Tz[max_depth][rz, q], acc_z2], f"LzZadj_{i}_{q}")

        # Canonical conditions on the adjusted logicals
        # X-logical: Z-part == 0, X-part is one-hot
        model.Add(sum(LxZ_adj[i][q] for q in range(n)) == 0)
        model.Add(sum(LxX_adj[i][q] for q in range(n)) == 1)

        # Z-logical: X-part == 0, Z-part is one-hot
        model.Add(sum(LzX_adj[i][q] for q in range(n)) == 0)
        model.Add(sum(LzZ_adj[i][q] for q in range(n)) == 1)

        # Positions match: enforce equality of the one-hot vectors
        for q in range(n):
            model.Add(LxX_adj[i][q] == LzZ_adj[i][q])

    # Stabilizers "up to row ops" — keep your original column-count version
    col_has_Z = [model.NewBoolVar(f"colHasZ_{q}") for q in range(n)]
    col_has_X = [model.NewBoolVar(f"colHasX_{q}") for q in range(n)]

    for q in range(n):
        z_rows = [Tz[max_depth][i, q] for i in range(m)]
        x_rows = [Tx[max_depth][i, q] for i in range(m)]
        or_equal(model, col_has_Z[q], z_rows)
        or_equal(model, col_has_X[q], x_rows)

    # Sum of chosen single-qubit-looking columns equals m
    model.Add(sum(col_has_Z + col_has_X) == m)

    # ----------------------------------------------------------------------
    # Solve
    # ----------------------------------------------------------------------
    solver = cp_model.CpSolver()
    # A couple of params that often help
    solver.parameters.num_search_workers = 8
    solver.parameters.cp_model_presolve = True
    solver.parameters.linearization_level = 2

    status = solver.Solve(model)
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        return "UNSAT"

    # ----------------------------------------------------------------------
    # Extract circuit
    # ----------------------------------------------------------------------
    circ = stim.Circuit()

    for t in range(max_depth):
        for q in range(n):
            code = solver.Value(sqgs[t][q])
            if code == H:
                circ.append("H", [q])
            elif code == Sg:
                circ.append("S_DAG", [q])  # match your original and invert later
            elif code == SX:
                circ.append("SQRT_X_DAG", [q])

        # 2q
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                if solver.Value(cxs[t][u][v]) == 1:
                    circ.append("CX", [u, v])
        for u in range(n):
            for v in range(u + 1, n):
                if solver.Value(czs[t][u][v]) == 1:
                    circ.append("CZ", [u, v])

    # Optional: add initial Hs derived from final tableau (as in your code)
    # (We use the model values here.)
    final_tableau_x = np.array([[int(solver.Value(Tx[max_depth][i, q])) for q in range(n)] for i in range(m)])
    first_layer_hadamards = np.where(final_tableau_x.sum(axis=0) >= 1)[0]
    if len(first_layer_hadamards) > 0:
        circ.append("H", list(first_layer_hadamards))

        # --- Encoding qubits from ADJUSTED logicals (use witness Wx/Wz) ---
    # Extract final stabs & logical rows from the model as plain ints
    Sx_fin_val = np.array([[int(solver.Value(Tx[max_depth][s, q])) for q in range(n)] for s in range(m)], dtype=int)
    Sz_fin_val = np.array([[int(solver.Value(Tz[max_depth][s, q])) for q in range(n)] for s in range(m)], dtype=int)

    LxX_raw = np.array([[int(solver.Value(Tx[max_depth][m + i, q])) for q in range(n)] for i in range(k)], dtype=int)
    LxZ_raw = np.array([[int(solver.Value(Tz[max_depth][m + i, q])) for q in range(n)] for i in range(k)], dtype=int)
    LzX_raw = np.array(
        [[int(solver.Value(Tx[max_depth][m + k + i, q])) for q in range(n)] for i in range(k)], dtype=int
    )
    LzZ_raw = np.array(
        [[int(solver.Value(Tz[max_depth][m + k + i, q])) for q in range(n)] for i in range(k)], dtype=int
    )

    Wx_val = np.array([[int(solver.Value(Wx[i][s])) for s in range(m)] for i in range(k)], dtype=int)
    Wz_val = np.array([[int(solver.Value(Wz[i][s])) for s in range(m)] for i in range(k)], dtype=int)

    # Adjust: Lx' = Lx ⊕ (Wx · S), Lz' = Lz ⊕ (Wz · S)   over GF(2)
    # (matrix multiply mod 2; we'll do it explicitly)
    def adjust(LX_raw, LZ_raw, W, Sx, Sz):
        LX_adj = LX_raw.copy()
        LZ_adj = LZ_raw.copy()
        for i in range(k):
            for s in range(m):
                if W[i, s] == 1:
                    LX_adj[i, :] ^= Sx[s, :]
                    LZ_adj[i, :] ^= Sz[s, :]
        return LX_adj, LZ_adj

    LxX_adj, LxZ_adj = adjust(LxX_raw, LxZ_raw, Wx_val, Sx_fin_val, Sz_fin_val)
    LzX_adj, LzZ_adj = adjust(LzX_raw, LzZ_raw, Wz_val, Sx_fin_val, Sz_fin_val)

    # Now each row should be canonical: LxZ_adj[i]==0 and LxX_adj[i] one-hot,
    # LzX_adj[i]==0 and LzZ_adj[i] one-hot, and positions match.
    positions = []
    for i in range(k):
        # Be defensive in case of numerical oddities
        if LxX_adj[i].sum() != 1:
            # fallback: pick argmax (shouldn't happen if constraints are satisfied)
            pos = int(np.argmax(LxX_adj[i]))
        else:
            pos = int(np.flatnonzero(LxX_adj[i])[0])
        positions.append(pos)

    encoding_qubits = np.array(positions, dtype=int)

    # Invert & sign fix (your original tail)
    enc_circ = circ.inverse()
    stabs_numpy = enc_circ.to_tableau().to_numpy()
    x_part = stabs_numpy[2].astype(int)
    z_part = stabs_numpy[3].astype(int)
    signs = stabs_numpy[-1].astype(int)
    tableau = np.hstack((x_part, z_part, np.array([signs]).T))
    if not np.all(tableau[:, -1] == 0):
        ker = mod2.nullspace(tableau)
        assert ker[-1, -1] == 1, "Last entry of kernel vector must be 1."
        correction_symplectic = ker[-1]
        zc = correction_symplectic[:n]
        xc = correction_symplectic[n:-1]
        for i, (xv, zv) in enumerate(zip(xc, zc, strict=False)):
            if xv == 1 and zv == 1:
                enc_circ.append("Y", [i])
            elif xv == 1:
                enc_circ.append("X", [i])
            elif zv == 1:
                enc_circ.append("Z", [i])

    return enc_circ, encoding_qubits


def gate_optimal_encoding_circuit(
    code: CSSCode,
    min_gates: int = 1,
    max_gates: int = 10,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> CNOTCircuit | None:
    """Synthesize an encoding circuit for the given CSS code using the minimal number of gates.

    Args:
        code: The CSS code to synthesize the encoding circuit for.
        min_gates: The minimum number of gates to use in the circuit.
        max_gates: The maximum number of gates to use in the circuit.
        min_timeout: The minimum time to spend on the synthesis.
        max_timeout: The maximum time to spend on the synthesis.

    Returns:
        The synthesized encoding circuit and the qubits that are used to encode the logical qubits.
    """
    logger.info("Starting optimal encoding circuit synthesis.")
    checks, logicals, use_x_checks = _get_matrix_with_fewest_checks(code)
    assert checks is not None
    n_checks = checks.shape[0]
    checks_and_logicals = np.vstack((checks, logicals))
    rank = mod2.rank(checks_and_logicals)
    termination_criteria = functools.partial(
        _final_matrix_constraint_partially_full_reduction,
        full_reduction_rows=list(range(checks.shape[0], checks.shape[0] + logicals.shape[0])),
        rank=rank,
    )

    res = optimal_elimination(
        checks_and_logicals,
        termination_criteria,
        "column_ops",
        min_param=min_gates,
        max_param=max_gates,
        min_timeout=min_timeout,
        max_timeout=max_timeout,
    )
    if res is None:
        return None
    reduced_checks_and_logicals, cnots = res
    cnots = cnots[::-1]

    return _build_css_encoder_from_cnot_list(n_checks, reduced_checks_and_logicals, cnots, use_x_checks)


def depth_optimal_encoding_circuit(
    code: CSSCode,
    min_depth: int = 1,
    max_depth: int = 10,
    min_timeout: int = 1,
    max_timeout: int = 3600,
) -> CNOTCircuit | None:
    """Synthesize an encoding circuit for the given CSS code using minimal depth.

    Args:
        code: The CSS code to synthesize the encoding circuit for.
        min_depth: The minimum number of gates to use in the circuit.
        max_depth: The maximum number of gates to use in the circuit.
        min_timeout: The minimum time to spend on the synthesis.
        max_timeout: The maximum time to spend on the synthesis.

    Returns:
        The synthesized encoding circuit and the qubits that are used to encode the logical qubits.
    """
    logger.info("Starting optimal encoding circuit synthesis.")
    checks, logicals, use_x_checks = _get_matrix_with_fewest_checks(code)
    assert checks is not None
    n_checks = checks.shape[0]
    checks_and_logicals = np.vstack((checks, logicals))
    rank = mod2.rank(checks_and_logicals)
    termination_criteria = functools.partial(
        _final_matrix_constraint_partially_full_reduction,
        full_reduction_rows=list(range(checks.shape[0], checks.shape[0] + logicals.shape[0])),
        rank=rank,
    )
    res = optimal_elimination(
        checks_and_logicals,
        termination_criteria,
        "parallel_ops",
        min_param=min_depth,
        max_param=max_depth,
        min_timeout=min_timeout,
        max_timeout=max_timeout,
    )
    if res is None:
        return None
    reduced_checks_and_logicals, cnots = res
    cnots = cnots[::-1]

    return _build_css_encoder_from_cnot_list(n_checks, reduced_checks_and_logicals, cnots, use_x_checks)


def _get_matrix_with_fewest_checks(code: CSSCode) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8], bool]:
    """Return the stabilizer matrix with the fewest checks, the corresponding logicals and a bool indicating whether X- or Z-checks have been returned."""
    use_x_checks = code.Hx.shape[0] < code.Hz.shape[0]
    checks = code.Hx if use_x_checks else code.Hz
    logicals = code.Lx if use_x_checks else code.Lz
    return checks, logicals, use_x_checks


def _final_matrix_constraint_partially_full_reduction(
    columns: npt.NDArray[np.bool_], full_reduction_rows: list[int], rank: int
) -> z3.BoolRef:
    assert len(columns.shape) == 3

    partial_reduction_rows = list(set(range(columns.shape[1])) - set(full_reduction_rows))

    # assert that the partial_reduction_rows are partially reduced, i.e. there are at least columns.shape[2] - (columns.shape[1] - len(full_reduction_rows)) non-zero columns
    partially_reduced = z3.PbEq(
        [(z3.Not(z3.Or(list(columns[-1, partial_reduction_rows, col]))), 1) for col in range(columns.shape[2])],
        columns.shape[2] - (rank - len(full_reduction_rows)),
    )

    # assert that there is no overlap between the full_reduction_rows and the partial_reduction_rows
    overlap_constraints = [True]
    for col in range(columns.shape[2]):
        has_entry_partial = z3.Or(list(columns[-1, partial_reduction_rows, col]))
        has_entry_full = z3.Or(list(columns[-1, full_reduction_rows, col]))
        overlap_constraints.append(z3.Not(z3.And(has_entry_partial, has_entry_full)))

    # assert that the full_reduction_rows are fully reduced
    fully_reduced = z3.PbEq(
        [
            (z3.PbEq([(columns[-1, row, col], 1) for col in range(columns.shape[2])], 1), 1)
            for row in full_reduction_rows
        ],
        len(full_reduction_rows),
    )

    return z3.And(fully_reduced, partially_reduced, z3.And(overlap_constraints))


def _build_css_encoder_from_cnot_list(
    n_checks: int, checks_and_logicals: npt.NDArray[np.int8], cnots: list[tuple[int, int]], use_x_checks: bool
) -> CNOTCircuit:
    encoding_qubits = np.where(checks_and_logicals[n_checks:, :].sum(axis=0) != 0)[0]
    if use_x_checks:
        hadamards = np.where(checks_and_logicals[:n_checks, :].sum(axis=0) != 0)[0]
    else:
        hadamards = np.where(checks_and_logicals[:n_checks, :].sum(axis=0) == 0)[0]
        cnots = [(j, i) for i, j in cnots]

    hadamards = np.setdiff1d(hadamards, encoding_qubits)
    non_hadamards = [i for i in range(checks_and_logicals.shape[1]) if i not in hadamards and i not in encoding_qubits]
    return CNOTCircuit.from_cnot_list(cnots, initialize_z=non_hadamards, initialize_x=hadamards)


def reduce_checks_by_row_ops(
    checks: npt.NDArray[np.int8],
    logicals: npt.NDArray[np.int8],
) -> None:
    """Try to reduce the total number of 1s in [checks; logicals] by row ops on *checks* only.

    Allowed operation: for check rows i != j,
        checks[j] <- checks[j] + checks[i] (mod 2)

    Constraints:
    - the new row must not have larger weight than *either* of the two rows we used
      (same guard you had before),
    - the *global* number of 1s across checks and logicals must strictly decrease.

    The arrays are modified in place.
    """
    r, _n = checks.shape
    # logicals can be empty (shape (0, n)), that's fine

    def total_ones() -> int:
        return int(checks.sum() + logicals.sum())

    improved = True
    while improved:
        improved = False
        total_ones()

        best_op: tuple[int, int] | None = None
        best_delta = 0  # positive = global reduction

        # try all check→check additions
        for i in range(r):
            row_i = checks[i]
            w_i = int(row_i.sum())
            for j in range(r):
                if i == j:
                    continue
                row_j = checks[j]
                w_j = int(row_j.sum())

                s = (row_j + row_i) % 2
                w_s = int(s.sum())

                # enforce "don't increase row weight" constraint
                if w_s > w_j or w_s > w_i:
                    continue

                # effect on global #ones:
                # only row_j changes
                delta = w_j - w_s  # positive = improvement
                if delta <= 0:
                    continue

                if delta > best_delta:
                    best_delta = delta
                    best_op = (i, j)

        if best_op is not None:
            i, j = best_op
            checks[j] = (checks[j] + checks[i]) % 2
            improved = True


def gottesman_encoding_circuit(tableau: StabilizerTableau | list[str]) -> tuple[stim.Circuit, list[int]]:
    """Synthesize encoding circuit for a stabilizer code as described in chapter 6.4 of Gottesman's book.

    Assumes all signs of the stabilizers are +1.

    Args:
        tableau: The stabilizer tableau of the code to synthesize the encoding circuit for.

    Returns:
        stim circuit implementing the encoding and a list of qubits that are used to encode the logical qubits.
    """
    if isinstance(tableau, list):
        tableau = StabilizerTableau.from_pauli_strings(tableau)
    nq = tableau.n
    mat = tableau.tableau.matrix.copy()
    x_part = mat[:, :nq]
    z_part = mat[:, nq:]

    circ = stim.Circuit()
    n_rows = mat.shape[0]

    initialized = []
    for row in range(n_rows):
        # find row with either x_part[row][i] = 1 or z_part[row][i] = 1
        pivot = row
        column = row

        while column < nq and x_part[pivot][column] != 1 and z_part[pivot][column] != 1:
            found_pivot = False
            for p in range(row, n_rows):
                if x_part[p][column] == 1 or z_part[p][column] == 1:
                    pivot = p
                    found_pivot = True
                    break
            if not found_pivot:
                column += 1
                pivot = row
        if column >= nq:
            # No valid pivot found, invalid tableau
            msg = "Invalid tableau: could not find a valid pivot."
            raise ValueError(msg)
        initialized.append(column)
        # swap to row i
        t = x_part[pivot].copy()
        x_part[pivot] = x_part[row]
        x_part[row] = t

        t = z_part[pivot].copy()
        z_part[pivot] = z_part[row]
        z_part[row] = t

        if x_part[row][column] == 0:
            circ.append("H", [column])
            t = x_part[:, column].copy()
            x_part[:, column] = z_part[:, column]
            z_part[:, column] = t

        # reduce column
        for q in np.where(x_part[row])[0]:
            if q == column:
                continue
            circ.append("CX", [column, q])
            x_part[:, q] ^= x_part[:, column]
            z_part[:, column] ^= z_part[:, q]

        if z_part[row][column] == 1:
            circ.append("S", [column])
            z_part[:, column] ^= x_part[:, column]

        for q in np.where(z_part[row])[0]:
            if q == column:
                continue
            circ.append("CZ", [column, q])
            z_part[:, q] ^= x_part[:, column]
            z_part[:, column] ^= x_part[:, q]

        # reduce stabilizers below row
        x_part[:, column] = 0
        x_part[row, column] = 1

    circ.append("H", initialized)
    circ = circ.inverse()

    signs = [s.sign for s in circ.to_tableau().to_stabilizers()]
    for row, sign in enumerate(signs):
        if sign == -1:
            circ.insert(0, stim.CircuitInstruction("X", [row]))
    uninitialized = list(set(range(nq)) - set(initialized))
    return circ, uninitialized


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
TV2 = tuple[int, int, int, int]  # (x_i, x_j, z_i, z_j) in {0,1}^4, non-trivial on both qubits
Op = tuple[TV2, tuple[int, int]]  # (v_bits, (i, j))


# ---------------------------------------------------------------------
# Basic symplectic helpers
# ---------------------------------------------------------------------


def sym_shape(U: npt.NDArray[np.int8]) -> int:
    """Return n for a 2n×2n symplectic matrix U."""
    tableau = StabilizerTableau.from_matrix(U)
    return tableau.n


def compute_r2_matrix(U: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    """R2(U)[i,j] = 1 iff det(F_ij)=1 over GF(2), i.e. rank(F_ij)=2 (invertible).
    This matches Eq. (12) via determinant test.
    """
    n = sym_shape(U)
    # det(F_ij) = A_xx[i,j]*A_zz[i,j] XOR A_xz[i,j]*A_zx[i,j]
    A_xx = U[:n, :n]
    A_xz = U[:n, n:]
    A_zx = U[n:, :n]
    A_zz = U[n:, n:]
    det = (A_xx & A_zz) ^ (A_xz & A_zx)
    return det.astype(np.int8)


def compute_r0_matrix(U: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    """R0(U)[i,j]=1 iff F_ij is all-zero (rank 0)."""
    n = sym_shape(U)
    A_xx = U[:n, :n]
    A_xz = U[:n, n:]
    A_zx = U[n:, :n]
    A_zz = U[n:, n:]
    zero = (A_xx == 0) & (A_xz == 0) & (A_zx == 0) & (A_zz == 0)
    return zero.astype(np.int8)


def compute_r1_matrix_from_r2_r0(R2: npt.NDArray[np.int8], R0: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    """R1 = 1 iff rank(F_ij)=1.
    Since rank ∈ {0,1,2}, we can do: R1 = NOT(R2 OR R0).
    This matches Eq. (11) / discussion.
    """
    return (1 ^ (R2 | R0)).astype(np.int8)


def r1_r2(U: npt.NDArray[np.int8]) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """Compute R1 and R2 matrices."""
    R2 = compute_r2_matrix(U)
    R0 = compute_r0_matrix(U)
    R1 = compute_r1_matrix_from_r2_r0(R2, R0)
    return R1, R2


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def bin2set(row: npt.NDArray[np.int8]) -> npt.NDArray[np.int64]:
    """Indices of 1s in a 0/1 row."""
    return np.flatnonzero(row)


def all_two_qubit_transvections() -> list[TV2]:
    """The 9 distinct 2-qubit transvections √(P_i P_j) (P∈{X,Y,Z} non-trivial) correspond to
    choosing (x,z) in {(1,0),(0,1),(1,1)} for each of the two qubits. :contentReference[oaicite:7]{index=7}.
    """
    nontrivial = [(1, 0), (0, 1), (1, 1)]  # X, Z, Y in (x,z)
    out: list[TV2] = []
    for xi, zi in nontrivial:
        for xj, zj in nontrivial:
            out.append((xi, xj, zi, zj))
    return out


def apply_tv2(U: npt.NDArray[np.int8], v: TV2, ij: tuple[int, int]) -> npt.NDArray[np.int8]:
    """Right-multiply U by the 2-qubit transvection corresponding to v on qubits (i,j),
    using the fast update U -> U + (U Ω v^T) v (GF(2)). :contentReference[oaicite:8]{index=8}
    This is the same fast method as the reference implementation's applyTv2. :contentReference[oaicite:9]{index=9}.
    """
    n = sym_shape(U)
    i, j = ij

    # Indices of v support in the 2n columns: [x_i, x_j, z_i, z_j]
    cols_v = [i, j, i + n, j + n]
    # Indices of Ω v^T support: Ω swaps X<->Z blocks, so [z_i, z_j, x_i, x_j]
    cols_ov = [i + n, j + n, i, j]

    v_bits = np.array(v, dtype=np.int8)
    nz = np.flatnonzero(v_bits)  # which of the 4 components are 1

    # Compute C = U * Ω v^T  (a length-2n column vector over GF(2))
    C = np.zeros((2 * n,), dtype=np.int8)
    for k in nz:
        C ^= U[:, cols_ov[k]]

    # Update the columns where v has 1s: col ^= C
    out = U.copy()
    for k in nz:
        out[:, cols_v[k]] ^= C
    return out


# ---------------------------------------------------------------------
# Gate option generation (symplectic case)
# ---------------------------------------------------------------------
def sp_gate_options(U: npt.NDArray[np.int8]) -> list[tuple[int, int]]:
    """Return a reduced set of candidate pairs (i,j) to consider, based on R2/R1 structure.
    The paper's greedy considers all pairs (i,j), but using structure often speeds things up.
    This mirrors the reference's SpOptions pattern.

    If you want *exactly* "all pairs", replace the body with:
        return [(i,j) for i in range(n) for j in range(i+1,n)]
    """
    n = sym_shape(U)
    R1, R2 = r1_r2(U)
    pairs: set[tuple[int, int]] = set()

    for row in range(n):
        r2_cols = bin2set(R2[row])
        r1_cols = bin2set(R1[row])

        # pair up columns that both have rank-2 blocks in the same row
        for a in range(len(r2_cols) - 1):
            for b in range(a + 1, len(r2_cols)):
                i, j = int(r2_cols[a]), int(r2_cols[b])
                if i != j:
                    pairs.add((min(i, j), max(i, j)))

        # pair a rank-2 with a rank-1 in the same row
        for i0 in r2_cols:
            for j0 in r1_cols:
                i, j = int(i0), int(j0)
                if i != j:
                    pairs.add((min(i, j), max(i, j)))

    return sorted(pairs)


# ---------------------------------------------------------------------
# Heuristic (Eq. 13)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class GreedyParams:
    max_wait: int = 10
    use_transpose_terms: bool = True
    # Weighting trick noted under Eq. (13): downweight R1 by /n; implemented as integer n*c2 + c1. :contentReference[oaicite:11]{index=11}
    use_integer_weighting: bool = True


def default_sp_heuristic(U: npt.NDArray[np.int8], params: GreedyParams) -> tuple[tuple[int, ...], int]:
    """Return (h_vector, h_scalar_like) for lexicographic minimization.

    h_vector matches Eq. (13) up to the integer-weighting trick mentioned in text:
        sorted(colSums(R2 + R1/n), colSums(R2^T + R1^T/n)) :contentReference[oaicite:12]{index=12}

    We avoid fractions by using key = n*colSum(R2) + colSum(R1).
    """
    n = sym_shape(U)
    R1, R2 = r1_r2(U)

    c1 = R1.sum(axis=0).astype(int)
    c2 = R2.sum(axis=0).astype(int)

    if params.use_transpose_terms:
        c1t = R1.sum(axis=1).astype(int)  # colSums(R1^T) = rowSums(R1)
        c2t = R2.sum(axis=1).astype(int)
        if params.use_integer_weighting:
            vec = np.concatenate([n * c2 + c1, n * c2t + c1t])
        else:
            # fallback: approximate real weighting
            vec = np.concatenate([c2 + c1 / n, c2t + c1t / n])
    else:
        vec = n * c2 + c1 if params.use_integer_weighting else (c2 + c1 / n)

    h_vec = tuple(sorted(int(x) for x in vec))
    # A simple scalar for termination / monitoring; zero iff already in target form in practice.
    # (You can replace this with a stronger check if you want.)
    h_scalar = int(R1.sum() + R2.sum())
    return h_vec, h_scalar


def is_terminal_form(U: npt.NDArray[np.int8]) -> bool:
    """Terminal condition from Chapter 3.3:
    R2(U) is a permutation matrix and R1(U) is all-zero. :contentReference[oaicite:13]{index=13}.
    """
    sym_shape(U)
    R1, R2 = r1_r2(U)
    if np.any(R1):
        return False
    # permutation matrix: each row/col has exactly one 1
    if not np.all(R2.sum(axis=0) == 1):
        return False
    return np.all(R2.sum(axis=1) == 1)


# ---------------------------------------------------------------------
# Main greedy loop (Algorithm 5 specialized to symplectic transvections)
# ---------------------------------------------------------------------
ChooseOpFn = Callable[
    [npt.NDArray[np.int8], list[tuple[Op, tuple[tuple[int, ...], int], npt.NDArray[np.int8]]]],
    tuple[Op, npt.NDArray[np.int8]],
]


def greedy_adapted_volanto(
    U: npt.NDArray[np.int8],
    params: GreedyParams = GreedyParams(),
    *,
    choose_op: ChooseOpFn | None = None,
    use_all_pairs: bool = False,
) -> tuple[list[Op], npt.NDArray[np.int8]]:
    """Greedy adapted Volanto synthesis:
      input: 2n×2n symplectic matrix U
      output: (op_list, P) where P is in 'perm + 1Q Clifford' form and
              applying op_list (in reverse/inverse sense) reconstructs U.

    This follows Algorithm 5 (greedy loop) and Chapter 3.3's goal state.
    """
    Uc = U.astype(np.int8).copy()
    n = sym_shape(Uc)
    # Uc = np.vstack((U[n:, :], U[:n, :]))

    transvections = all_two_qubit_transvections()
    op_list: list[Op] = []

    # initial heuristic
    h_last_vec, _ = default_sp_heuristic(Uc, params)
    h_min_vec = h_last_vec
    curr_wait = 0

    while not is_terminal_form(Uc):
        # Candidate (i,j) pairs
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)] if use_all_pairs else sp_gate_options(Uc)

        # Evaluate all candidates
        scored: list[tuple[Op, tuple[tuple[int, ...], int], npt.NDArray[np.int8]]] = []
        for i, j in pairs:
            for v in transvections:
                op: Op = (v, (i, j))
                B = apply_tv2(Uc, v, (i, j))
                h_vec, h_s = default_sp_heuristic(B, params)
                scored.append((op, (h_vec, h_s), B))

        if not scored:
            msg = "No gate options generated; cannot proceed."
            raise RuntimeError(msg)

        # Default choice: minimize h_vec lexicographically (Eq. 13).
        if choose_op is None:
            op_best, (h_best_vec, _), B_best = min(scored, key=lambda t: t[1][0])
        else:
            op_best, B_best = choose_op(Uc, scored)
            h_best_vec, _ = default_sp_heuristic(B_best, params)

        # Apply the chosen operation
        op_list.append(op_best)
        Uc = B_best
        h_last_vec = h_best_vec

        # max-wait early exit as in Algorithm 5 :contentReference[oaicite:15]{index=15}
        if h_last_vec < h_min_vec:
            h_min_vec = h_last_vec
            curr_wait = 0
        else:
            curr_wait += 1
            if params.max_wait > 0 and curr_wait > params.max_wait:
                msg = "Greedy synthesis appears stuck (max_wait exceeded)."
                raise RuntimeError(msg)

    # Algorithm 5 returns inverse(opList), P.
    # Here: our ops are involutions (transvections are self-inverse up to phase),
    # and application order reverses.
    op_list_inv = list(reversed(op_list))
    return op_list_inv, Uc


# Encoded 1Q gate sequence for a qubit: list of "H" and "S"
SingleQOp = tuple[int, list[str]]  # (qubit, ["H","S",...])
SwapOp = tuple[int, int]  # (a, b)


def _right_multiply_swap(U: npt.NDArray[np.int8], a: int, b: int) -> npt.NDArray[np.int8]:
    """Right-multiply U by SWAP(a,b) in symplectic form (permutes X_a<->X_b and Z_a<->Z_b columns)."""
    n = sym_shape(U)
    out = U.copy()
    cols = list(range(2 * n))
    # swap X columns
    cols[a], cols[b] = cols[b], cols[a]
    # swap Z columns
    cols[a + n], cols[b + n] = cols[b + n], cols[a + n]
    return out[:, cols]


def _right_multiply_H(U: npt.NDArray[np.int8], q: int) -> npt.NDArray[np.int8]:
    """Right-multiply by H on qubit q (swap X_q and Z_q columns)."""
    n = sym_shape(U)
    out = U.copy()
    out[:, [q, q + n]] = out[:, [q + n, q]]
    return out


def _right_multiply_S(U: npt.NDArray[np.int8], q: int) -> npt.NDArray[np.int8]:
    """Right-multiply by S on qubit q under ROW-IMAGE convention:
      X -> XZ   => column X_q unchanged, column Z_q ^= column X_q
      Z -> Z
    In symplectic matrix columns: Zcol := Zcol + Xcol.
    """
    n = sym_shape(U)
    out = U.copy()
    out[:, q + n] ^= out[:, q]
    return out


def _matmul2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return ((A @ B) % 2).astype(np.int8)


def _one_qubit_group() -> dict[str, np.ndarray]:
    """6 elements generated by H and S (row-image, right-multiply convention):
    matrices act on columns [X, Z] of a single qubit.
    """
    I = np.array([[1, 0], [0, 1]], dtype=np.int8)
    H = np.array([[0, 1], [1, 0]], dtype=np.int8)
    S = np.array([[1, 1], [0, 1]], dtype=np.int8)

    # For right-multiplication, sequence "HS" means multiply by H then S => I*H*S
    elems: dict[str, np.ndarray] = {
        "": I,
        "H": H,
        "S": S,
        "HS": _matmul2(H, S),
        "SH": _matmul2(S, H),
        "HSH": _matmul2(_matmul2(H, S), H),
    }
    return elems


def _inv_word(word: str) -> str:
    """Inverse in the {H,S} group: H^{-1}=H, S^{-1}=S^3=S^(-1) = 'SSS' but inside 6-element group we can table it."""
    # easiest: brute force using the 6-element table
    elems = _one_qubit_group()
    M = elems[word]
    I = elems[""]
    for w2, M2 in elems.items():
        if np.array_equal(_matmul2(M, M2), I):
            return w2
    msg = f"no inverse for {word}"
    raise ValueError(msg)


def _extract_perm_in_to_out_and_blocks(U: npt.NDArray[np.int8]) -> tuple[np.ndarray, list[np.ndarray]]:
    """For terminal U:
    perm[i] = unique j where det(F_ij)=1 (row i of R2).
    blocks[i] = 2×2 block F_ij for that (i,j).
    """
    n = sym_shape(U)
    R2 = compute_r2_matrix(U)

    perm = np.full(n, -1, dtype=int)
    blocks: list[np.ndarray] = [None] * n  # type: ignore

    for i in range(n):
        js = np.flatnonzero(R2[i])
        if len(js) != 1:
            msg = "Not terminal: R2 row is not one-hot."
            raise ValueError(msg)
        j = int(js[0])
        perm[i] = j
        blocks[i] = np.array(
            [
                [int(U[i, j]), int(U[i, j + n])],
                [int(U[i + n, j]), int(U[i + n, j + n])],
            ],
            dtype=np.int8,
        )

    if len(set(perm.tolist())) != n:
        msg = "Not terminal: R2 columns not one-hot."
        raise ValueError(msg)
    return perm, blocks


def _perm_inverse(perm_in_to_out: np.ndarray) -> np.ndarray:
    n = len(perm_in_to_out)
    inv = np.empty(n, dtype=int)
    for i, j in enumerate(perm_in_to_out):
        inv[int(j)] = i
    return inv


def _perm_to_swaps(perm_in_to_out: np.ndarray) -> list[SwapOp]:
    """Return a SWAP list that realizes perm_in_to_out when right-multiplying the symplectic matrix,
    i.e. permuting columns (wires). (Any decomposition is fine for the test.).
    """
    perm = perm_in_to_out.copy().tolist()
    n = len(perm)
    swaps: list[SwapOp] = []
    pos = list(range(n))  # current label at position p

    # We want to permute columns so that wire i ends up at perm[i].
    # Do it via bubble-like swapping on positions.
    for i in range(n):
        target_pos = perm[i]
        cur_pos = pos.index(i)
        while cur_pos != target_pos:
            step = cur_pos + 1 if cur_pos < target_pos else cur_pos - 1
            swaps.append((cur_pos, step))
            # swap labels in pos
            pos[cur_pos], pos[step] = pos[step], pos[cur_pos]
            cur_pos = step

    # This is not minimal, but deterministic and fine for testing.
    # You can replace with cycle decomposition later.
    return swaps


def reduce_single_qubit_gates_and_swaps(
    U: npt.NDArray[np.int8],
) -> tuple[tuple[list[SwapOp], list[SingleQOp]], npt.NDArray[np.int8]]:
    """Reduce a TERMINAL symplectic matrix U to identity using only SWAP/H/S by right-multiplication.

    Returns:
      ((swaps, one_qubit_ops), U_reduced)
    where U_reduced should be identity.
    """
    Uc = U.astype(np.int8).copy()
    n = sym_shape(Uc)

    if not is_terminal_form(Uc):
        msg = "reduce_with_single_qubit_gates expects a terminal-form matrix."
        raise ValueError(msg)

    # 1) Extract permutation and 2×2 blocks
    perm, _blocks = _extract_perm_in_to_out_and_blocks(Uc)

    # 2) Right-multiply by permutation inverse to bring blocks onto the diagonal
    inv = _perm_inverse(perm)
    swaps = _perm_to_swaps(inv)  # realize inv permutation
    for a, b in swaps:
        Uc = _right_multiply_swap(Uc, a, b)

    # After applying inv, each input i should land on output i, and blocks move to diagonal.
    # Re-extract diagonal blocks (now at (i,i)).
    oneq_table = _one_qubit_group()
    inv_words: list[SingleQOp] = []

    for q in range(n):
        F = np.array(
            [
                [int(Uc[q, q]), int(Uc[q, q + n])],
                [int(Uc[q + n, q]), int(Uc[q + n, q + n])],
            ],
            dtype=np.int8,
        )

        # Find which word produces F; then apply its inverse to cancel.
        found = None
        for w, M in oneq_table.items():
            if np.array_equal(M, F):
                found = w
                break
        if found is None:
            msg = f"Diagonal block not a 1Q Clifford in {{H,S}} group:\n{F}"
            raise ValueError(msg)

        w_inv = _inv_word(found)
        if w_inv:
            inv_words.append((q, list(w_inv)))

        # Apply the inverse word by right-multiplication
        for g in w_inv:
            if g == "H":
                Uc = _right_multiply_H(Uc, q)
            elif g == "S":
                Uc = _right_multiply_S(Uc, q)
            else:
                raise ValueError(g)

    return (swaps, inv_words), Uc


# # ---------- helpers for transvections √(P_i P_j) ----------

_PAULI_FROM_XZ = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}


def _c_for_pauli(p: str) -> list[str]:
    """Local Clifford C(P) such that C(P) Z C(P)† = P."""
    if p == "Z":
        return []
    if p == "X":
        return ["H"]
    if p == "Y":
        return ["S", "H"]  # (S·H) Z (S·H)† = Y
    raise ValueError(p)


def _c_dag_for_pauli(p: str) -> list[str]:
    """Inverse of C(P)."""
    if p == "Z":
        return []
    if p == "X":
        return ["H"]
    if p == "Y":
        return ["H", "S_DAG"]  # ["S_DAG", "H"]
    raise ValueError(p)


def _append_transvection_as_hs_cz(
    c: stim.Circuit,
    v_bits: tuple[int, int, int, int],
    ij: tuple[int, int],
) -> None:
    i, j = ij
    xi, xj, zi, zj = v_bits
    Pi = _PAULI_FROM_XZ[xi, zi]
    Pj = _PAULI_FROM_XZ[xj, zj]
    if Pi == "I" or Pj == "I":
        msg = f"Expected non-trivial Pauli on both qubits, got {Pi},{Pj}"
        raise ValueError(msg)

    # Basis change: map Pi,Pj to Z on each qubit
    for g in _c_for_pauli(Pi):
        c.append(g, [i])
    for g in _c_for_pauli(Pj):
        c.append(g, [j])

    # Core: √(Z_i Z_j) == CZ(i,j) then S on i and j (up to global phase)
    c.append("CZ", [i, j])
    c.append("S", [i])
    c.append("S", [j])

    # Undo basis change
    for g in _c_dag_for_pauli(Pj):
        c.append(g, [j])
    for g in _c_dag_for_pauli(Pi):
        c.append(g, [i])


# # ---------- main conversion ----------


def _append_reduction_ops_as_stim(
    circ: stim.Circuit,
    swaps: list[tuple[int, int]],
    oneq_ops: list[tuple[int, list[str]]],
) -> None:
    """Append the SWAP/H/S operations (the right-multiplication reductions) as a stim circuit."""
    # Swaps were produced to be applied (right-multiply) in the given order,
    # and in a stim circuit, appending SWAP applies that gate in that order.
    for a, b in swaps:
        circ.append("SWAP", [int(a), int(b)])

    # oneq_ops is a list of (q, ["H","S",...]) where gates are to be right-multiplied in that order.
    for q, word in oneq_ops:
        q = int(q)
        for g in word:
            if g == "H":
                circ.append("H", [q])
            elif g == "S":
                circ.append("S", [q])
            else:
                msg = f"Unsupported 1Q gate in reduction word: {g}"
                raise ValueError(msg)


def synthesize_clifford_volanto(
    tableau: StabilizerTableau,
    *,
    greedy_params: GreedyParams = GreedyParams(),
    choose_op: ChooseOpFn | None = None,
    use_all_pairs: bool = False,
) -> stim.Circuit:
    """Synthesize a stim circuit implementing a StabilizerTableau using:
      - greedy_adapted_volanto (right-multiply transvections) -> terminal P
      - reduce_with_single_qubit_gates(P) (right-multiply SWAP/H/S) -> I.

    We have: U * (G1 ... Gm) = I  =>  U = (G1 ... Gm)^-1
    Circuit should append G1^-1, G2^-1, ..., Gm^-1 in that order.
    """
    # 1) Greedy: Uc = tableau * T1 * ... * TL = P (terminal)
    ops_inv, P = greedy_adapted_volanto(
        tableau.tableau.matrix,
        params=greedy_params,
        choose_op=choose_op,
        use_all_pairs=use_all_pairs,
    )

    # The actual applied order was op_list = [T1, ..., TL] = reversed(ops_inv)
    op_list = list(reversed(ops_inv))

    # 2) Reduce terminal: P * R1 * ... * Rr = I
    (swaps, oneq_ops), P_reduced = reduce_single_qubit_gates_and_swaps(P)
    if not np.array_equal(P_reduced, np.eye(2 * tableau.n, dtype=np.int8)):
        msg = "Single-qubit reduction failed: did not reach identity."
        raise RuntimeError(msg)

    # 3) Emit circuit for U = (T1..TL R1..Rr)^-1
    # i.e. append inverses in the SAME order they were applied: T1^-1, ..., TL^-1, R1^-1, ..., Rr^-1
    circ = stim.Circuit()

    # Transvections: inverse is itself (symplectic involution), so emit same gate.
    for v_bits, (i, j) in op_list:
        _append_transvection_as_hs_cz(
            circ,
            tuple(int(b) for b in v_bits),
            (int(i), int(j)),
        )

    # Swaps: self-inverse
    for a, b in swaps:
        circ.append("SWAP", [int(a), int(b)])

    # 1Q reductions were applied as right-multiplies by H/S to reduce P.
    # For U we need the inverse gates:
    #   H^-1 = H
    #   S^-1 = S_DAG
    for q, word in oneq_ops:
        q = int(q)
        for g in word:
            if g == "H":
                circ.append("H", [q])
            elif g == "S":
                circ.append("S_DAG", [q])
            else:
                msg = f"Unsupported 1Q gate in reduction word: {g}"
                raise ValueError(msg)

    return circ


def _fix_tableau_signs_in_place(
    out: stim.Circuit,
    target_x_signs: np.ndarray,
    target_z_signs: np.ndarray,
) -> None:
    """Append Pauli corrections so that out.to_tableau() matches the desired sign bits.

    If tableau(X_i) has wrong sign -> append Z_i
    If tableau(Z_i) has wrong sign -> append X_i
    """
    tab = out.to_tableau()
    _, _, _, _, got_x, got_z = tab.to_numpy(bit_packed=False)

    got_x = got_x.astype(np.int8)
    got_z = got_z.astype(np.int8)

    n = len(tab)
    for q in range(n):
        if got_x[q] != target_x_signs[q]:
            out.append("Z", [q])
        if got_z[q] != target_z_signs[q]:
            out.append("X", [q])


def resynthesize_stim_circuit_with_volanto(
    circ: stim.Circuit,
    *,
    greedy_params: GreedyParams = GreedyParams(),
    choose_op: ChooseOpFn | None = None,
    use_all_pairs: bool = False,
    fix_signs: bool = True,
) -> stim.Circuit:
    """Take a stim circuit, convert to symplectic matrix U, resynthesize using our
    Volanto+single-qubit reduction method, and return a new stim circuit.

    If fix_signs=True, the returned circuit matches circ.to_tableau() exactly.
    Otherwise it matches only the symplectic (phase-free) action.
    """
    tableau = StabilizerTableau.from_stim_circuit(circ)
    U = tableau.tableau.matrix
    x_signs = tableau.phase[: tableau.n]
    z_signs = tableau.phase[tableau.n :]

    out = synthesize_clifford_volanto(
        U,
        greedy_params=greedy_params,
        choose_op=choose_op,
        use_all_pairs=use_all_pairs,
    )

    if fix_signs:
        _fix_tableau_signs_in_place(out, x_signs, z_signs)

    return out
