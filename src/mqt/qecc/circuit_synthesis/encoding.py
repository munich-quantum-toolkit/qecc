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
from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import stim
import z3

from ..codes import CSSCode
from ..codes.pauli import CheckMatrix, StabilizerTableau, complete_stabilizer_tableau_with_destabilizers
from .circuits import CliffordIsometry
from .synthesis_utils import build_css_encoder_from_cnot_list, cnot_encoding_circuit, optimal_elimination
from .transvection import (
    eliminate_non_css_with_lookahead,
    score_symplectic,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from ..codes import CSSCode, StabilizerCode
    from .circuits import CNOTCircuit


logger = logging.getLogger(__name__)


from ortools.sat.python import cp_model

from ..codes.css_code import CSSCode


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

    # Invert & sign fix
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
    checks, logicals = _get_matrix_with_fewest_checks(code)
    assert checks is not None
    n_checks = checks.num_rows()

    checks_and_logicals = np.vstack((checks.matrix, logicals.matrix))
    rank = mod2.rank(checks_and_logicals)
    termination_criteria = functools.partial(
        _final_matrix_constraint_partially_full_reduction,
        full_reduction_rows=list(range(n_checks, n_checks + logicals.num_rows())),
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
    if checks.type == "Z":
        cnots = [(j, i) for i, j in cnots]

    return build_css_encoder_from_cnot_list(
        CheckMatrix(reduced_checks_and_logicals[:n_checks], type=checks.type),
        CheckMatrix(reduced_checks_and_logicals[n_checks:], type=logicals.type),
        cnots,
    )


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
    checks, logicals = _get_matrix_with_fewest_checks(code)
    assert checks is not None
    n_checks = checks.num_rows()
    checks_and_logicals = np.vstack((checks.matrix, logicals.matrix))
    rank = mod2.rank(checks_and_logicals)
    termination_criteria = functools.partial(
        _final_matrix_constraint_partially_full_reduction,
        full_reduction_rows=list(range(n_checks, n_checks + logicals.num_rows())),
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
    if checks.type == "Z":
        cnots = [(j, i) for i, j in cnots]

    return build_css_encoder_from_cnot_list(
        CheckMatrix(reduced_checks_and_logicals[:n_checks], type=checks.type),
        CheckMatrix(reduced_checks_and_logicals[n_checks:], type=logicals.type),
        cnots,
    )


def _get_matrix_with_fewest_checks(code: CSSCode) -> tuple[CheckMatrix, CheckMatrix]:
    """Return the stabilizer matrix with the fewest checks, the corresponding logicals and a bool indicating whether X- or Z-checks have been returned."""
    use_x_checks = code.Hx.shape[0] < code.Hz.shape[0]
    checks = code.Hx if use_x_checks else code.Hz
    logicals = code.Lx if use_x_checks else code.Lz
    type_ = "X" if use_x_checks else "Z"
    return CheckMatrix(checks, type_), CheckMatrix(logicals, type_)


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


def gottesman_encoding_circuit(tableau: StabilizerTableau | list[str]) -> CliffordIsometry:
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
    iso = CliffordIsometry.from_stim_circuit(circ)
    for q in initialized:
        iso.initialize_qubit(q, basis="Z")
    return iso


def synthesize_clifford(
    tableau: StabilizerTableau,
    lookahead_depth: int = 1,
    lookahead_top_k: int = 10,
    use_cnots_if_css: bool = True,
    optimization_criterion: str = "gates",
) -> CliffordIsometry:
    """Synthesize a stim circuit implementing a Clifford operation to minimize two-qubit gate count.

    Args:
        tableau: The stabilizer tableau representing the Clifford operation to synthesize.
        lookahead_depth: The depth of lookahead to use in the synthesis.
        lookahead_top_k: The number of candidates to consider during lookahead.
        use_cnots_if_css: Whether to use CNOT-only synthesis if the tableau is CSS.

    Returns:
        A stim.Circuit that implements the same operation as the input tableau but with potentially fewer two
    """
    if tableau.is_css() and use_cnots_if_css:
        x_checks, z_checks = tableau.to_css()
        return cnot_encoding_circuit(
            CheckMatrix(np.empty((0, tableau.n)), type="X"),
            x_checks if x_checks.num_rows() <= z_checks.num_rows() else z_checks,
        )

    ops, _ = eliminate_non_css_with_lookahead(
        tableau,
        lookahead=lookahead_depth,
        num_lookahead_candidates=lookahead_top_k,
        optimization_criterion=optimization_criterion,
    )
    return CliffordIsometry.from_stim_circuit(ops.to_circuit_inverse())


def synthesize_encoding_circuit(
    code: StabilizerCode, lookahead_depth=0, lookahead_top_k=10, optimize_depth: bool = False
) -> CliffordIsometry:
    """Synthesize an encoding circuit for the given stabilizer code.

    Args:
        code: The stabilizer code to synthesize the encoding circuit for.

    Returns:
        A CliffordIsometry that implements the encoding circuit for the given stabilizer code.
    """
    if isinstance(code, CSSCode):
        x_checks = CheckMatrix(code.Hx, type="X")
        z_checks = CheckMatrix(code.Hz, type="Z")
        x_logicals = CheckMatrix(code.Lx, type="X")
        z_logicals = CheckMatrix(code.Lz, type="Z")
        checks, logicals = (
            (x_checks, x_logicals) if x_checks.num_rows() <= z_checks.num_rows() else (z_checks, z_logicals)
        )
        return cnot_encoding_circuit(
            checks,
            logicals,
            lookahead=lookahead_depth,
            lookahead_candidates=lookahead_top_k,
            optimize_depth=optimize_depth,
        )

    tableau = StabilizerTableau.from_stabilizer_code(code)
    return synthesize_clifford(tableau, lookahead_depth=1, lookahead_top_k=10, use_cnots_if_css=True)


def resynthesize_stim_circuit(
    circ: stim.Circuit,
    *,
    top_k: int = 10,
    lookahead_depth: int = 1,
    use_cnots_if_css: bool = True,
) -> stim.Circuit:
    """Resynthesize a stim circuit implementing a Clifford operation to minimize two-qubit gate count.

    Args:
        circ: The stim.Circuit to resynthesize.
        top_k: The number of candidates to consider during lookahead.
        lookahead_depth: The depth of lookahead to use in the synthesis.
        use_cnots_if_css: Whether to use CNOT-only synthesis if the circuit is CSS.

    Returns:
        A stim.Circuit that implements the same operation as the input circuit but with potentially fewer two
    """
    tableau = StabilizerTableau.from_stim_circuit(circ)
    return synthesize_clifford(
        tableau,
        lookahead_depth=lookahead_depth,
        lookahead_top_k=top_k,
        use_cnots_if_css=use_cnots_if_css,
    ).to_stim_circuit()


def encoder_from_stabilizers_and_logicals(
    stabilizers: StabilizerTableau,
    logicals: StabilizerTableau,
    lookahead_depth: int = 1,
    lookahead_top_k: int = 10,
    optimize_tableau_before_synthesis: bool = True,
    optimization_criterion: str = "gates",
) -> CliffordIsometry:
    """Synthesize an encoding circuit for a stabilizer code given its stabilizers and logicals as tableaux.

    Args:
        stabilizers: A tableau representing the stabilizers of the code.
        logicals: A tableau representing the logical operators of the code.

    Returns:
        A CliffordIsometry that implements the encoding circuit for the given stabilizer code.
    """
    if stabilizers.n != logicals.n:
        msg = "Stabilizers and logicals must have the same number of qubits."
        raise ValueError(msg)
    if stabilizers.num_rows() + logicals.num_rows() > stabilizers.n * 2:
        msg = "The total number of stabilizers and logicals must be less than or equal to 2n."
        raise ValueError(msg)

    full_tableau = combine_stabilizer_and_logical_tableau(stabilizers, logicals)
    stab_indices = list(range(logicals.num_rows() // 2, logicals.num_rows() // 2 + stabilizers.num_rows()))
    if optimize_tableau_before_synthesis:
        optimized_tableau = optimize_tableau(full_tableau, stab_rows=stab_indices)
    else:
        optimized_tableau = full_tableau

    iso = synthesize_clifford(
        optimized_tableau,
        use_cnots_if_css=False,
        lookahead_depth=lookahead_depth,
        lookahead_top_k=lookahead_top_k,
        optimization_criterion=optimization_criterion,
    )
    iso.initialize_qubits(stab_indices, basis="Z")
    return iso


def optimize_tableau(tableau: StabilizerTableau, stab_rows: list[int]) -> StabilizerTableau:
    """Optimize a stabilizer tableau by performing row operations to reduce the cost of the initial tableau for synthesis."""
    tab = tableau.copy()

    best = (tab, score_symplectic(tab))
    improved = True
    half = tableau.num_rows() // 2
    x_logical_rows = [i for i in range(half) if i not in stab_rows]
    z_logical_rows = [i + half for i in range(half) if i not in stab_rows]
    logical_rows = x_logical_rows + z_logical_rows
    k = len(logical_rows) // 2
    while improved:
        improved = False
        for i in range(len(stab_rows)):
            for j in range(len(stab_rows)):
                if i == j:
                    continue
                tab = tableau.copy()
                mat = tab.tableau.matrix
                destabs = mat[:half][stab_rows]
                stabs = mat[half:][stab_rows]
                stabs[i] ^= stabs[j]
                destabs[j] ^= destabs[i]
                mat[:half][stab_rows] = destabs
                mat[half:][stab_rows] = stabs
                new_score = score_symplectic(StabilizerTableau(mat, tableau.phase.copy()))
                if new_score < best[1]:
                    best = (tab, new_score)
                    improved = True
            for j in range(len(logical_rows)):
                tab = tableau.copy()
                mat = tab.tableau.matrix
                destabs = mat[:half][stab_rows]
                stabs = mat[half:][stab_rows]

                other_log = mat[logical_rows[(j + k) % (2 * k)]]

                destabs[i] ^= other_log
                logj = mat[logical_rows[j]]

                logj ^= stabs[i]
                mat[:half][stab_rows] = destabs
                mat[logical_rows[j]] = logj
                new_score = score_symplectic(StabilizerTableau(mat, tableau.phase.copy()))
                if new_score < best[1]:
                    best = (tab, new_score)
                    improved = True
        tableau = best[0]

    return best[0]


def combine_stabilizer_and_logical_tableau(
    stabilizers: StabilizerTableau, logicals: StabilizerTableau
) -> StabilizerTableau:
    """Combine a stabilizer tableau and a logical tableau, then complete with destabilizers.

    Args:
        stabilizers: A tableau representing the stabilizers of the code (without destabilizers).
        logicals: A tableau containing logical operators.

    Returns:
        A combined tableau with destabilizers added, suitable for circuit synthesis.
    """
    if stabilizers.n != logicals.n:
        msg = "Stabilizers and logicals must act on the same number of qubits."
        raise ValueError(msg)

    m = stabilizers.num_rows()

    # Combine stabilizers and logicals into a single tableau
    x_logicals = logicals.tableau.matrix[: logicals.num_rows() // 2]
    z_logicals = logicals.tableau.matrix[logicals.num_rows() // 2 :]
    x_logicals_phase = logicals.phase[: logicals.num_rows() // 2]
    z_logicals_phase = logicals.phase[logicals.num_rows() // 2 :]
    combined_matrix = np.vstack([x_logicals, z_logicals, stabilizers.tableau.matrix])

    combined_phase = np.hstack([x_logicals_phase, z_logicals_phase, stabilizers.phase])
    combined_tableau = StabilizerTableau(combined_matrix, combined_phase)

    # Complete with destabilizers for the stabilizers only
    # The stabilizer rows are at indices 0 to m-1
    stab_rows = list(range(logicals.num_rows(), logicals.num_rows() + m))
    return complete_stabilizer_tableau_with_destabilizers(combined_tableau, stab_rows)
