# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests of the fault model, the exact fault-tolerance check and the check engines on small instances."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import stim

from mqt.qecc.circuit_synthesis.ft_search._core import gauge
from mqt.qecc.circuit_synthesis.ft_search._core.blocks import BlockSpec
from mqt.qecc.circuit_synthesis.ft_search._core.circuit import Circuit, Op, layering, metrics
from mqt.qecc.circuit_synthesis.ft_search._core.design import ancilla_states, compile_node, make_node
from mqt.qecc.circuit_synthesis.ft_search._core.faults import fault_table
from mqt.qecc.circuit_synthesis.ft_search._core.gauge import promote
from mqt.qecc.circuit_synthesis.ft_search._core.greedy_ft import greedy_ft
from mqt.qecc.circuit_synthesis.ft_search._core.kernels import enumerate_faults
from mqt.qecc.circuit_synthesis.ft_search._core.schedule import schedule
from mqt.qecc.circuit_synthesis.ft_search._core.solvers.common import peham_verified
from mqt.qecc.circuit_synthesis.ft_search._core.verify import (
    MAX_ROWS,
    CSSState,
    _group_keys,
    _pack,
    check_ft,
    check_noiseless,
    check_size,
    gf2_row_basis,
    leader_weights,
    state_leader,
)

from .helpers import color_17_1_5, heuristic_prep, load_code, prep_circuit, table_keys


def _random_unitary(n: int, depth: int, rng: np.random.Generator) -> Circuit:
    ops = [Op("R", (q,)) for q in range(n)]
    for _ in range(depth):
        kind = rng.choice(["CX", "CZ", "H"])
        if kind == "H":
            ops.append(Op("H", (int(rng.integers(n)),)))
        else:
            a, b = rng.choice(n, size=2, replace=False)
            ops.append(Op(str(kind), (int(a), int(b))))
    return Circuit(ops, n, list(range(n)))


def test_propagation_matches_stim() -> None:
    """The fault propagation of ``fault_table`` agrees with stim."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        circ = _random_unitary(5, 30, rng)
        table = fault_table(circ)
        for f in rng.choice(len(table), size=40, replace=False):
            i = int(table.op_index[f])
            op = circ.ops[i]
            p = stim.PauliString(circ.num_qubits)
            for q, c in zip(op.qubits, table.label[f], strict=True):
                if c != "I":
                    p[q] = c
            suffix = stim.Circuit()
            for later in circ.ops[i + 1 :]:
                suffix.append(later.name, list(later.qubits))
            out = p.after(suffix)
            xs, zs = out.to_numpy()
            assert np.array_equal(xs.astype(np.uint8), table.data_x[f])
            assert np.array_equal(zs.astype(np.uint8), table.data_z[f])


def test_measurement_flips() -> None:
    """A fault before a parity check flips its measurement; a measurement fault flips it too."""
    # Z-parity check of qubits 0,1 via ancilla 2; X fault on qubit 0 before the check flips it
    circ = Circuit(
        [Op("R", (0,)), Op("R", (1,)), Op("R", (2,)), Op("CX", (0, 2)), Op("CX", (1, 2)), Op("M", (2,))],
        3,
        [0, 1],
    )
    table = fault_table(circ)
    k = next(f for f in range(len(table)) if table.op_index[f] == 0 and table.label[f] == "X")
    assert table.flips[k, 0] == 1
    k = next(f for f in range(len(table)) if table.op_index[f] == 5)
    assert table.label[k] == "flip"
    assert table.flips[k, 0] == 1


def test_leader_weights_repetition() -> None:
    """Leader weights of the repetition code."""
    h = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    w = leader_weights(h, 1)
    assert isinstance(w, np.ndarray)  # a dense table for few rows
    assert w[0] == 0
    assert sorted(w.tolist()) == [0, 1, 1, 1]


def _brute(circ: Circuit, state: CSSState, t: int) -> dict[int, int]:
    """Reference: count failing key multisets by direct enumeration."""
    table = fault_table(circ)
    kx, kz, kc, sx, sz = table_keys(table, state)
    keys = np.unique(np.stack([kx, kz, kc], axis=1), axis=0)
    wx, wz = leader_weights(sz, t), leader_weights(sx, t)
    out = {}
    for s in range(1, t + 1):
        combos = (
            itertools.combinations(range(len(keys)), 2)
            if s == 2
            else itertools.combinations_with_replacement(range(len(keys)), s)
        )
        cnt = 0
        for c in combos:
            x = np.bitwise_xor.reduce(keys[list(c), 0])  # ty: ignore[no-matching-overload]  (numpy stub gap)
            z = np.bitwise_xor.reduce(keys[list(c), 1])  # ty: ignore[no-matching-overload]  (numpy stub gap)
            sig = np.bitwise_xor.reduce(keys[list(c), 2])  # ty: ignore[no-matching-overload]  (numpy stub gap)
            if sig == 0 and (wx[int(x)] > s or wz[int(z)] > s):
                cnt += 1
        out[s] = cnt
    return out


@pytest.fixture(scope="module")
def steane():
    """The Steane code, its heuristic prep and the prep verified by MQT QECC."""
    code = load_code("steane")
    sp = heuristic_prep(code)
    return code, prep_circuit(sp), peham_verified(sp)


def test_steane_prep_not_ft(steane) -> None:
    """The plain Steane prep is not FT."""
    code, prep, _ = steane
    check_noiseless(prep, code.zero_state())
    assert not check_ft(prep, code.zero_state(), 1).ok


def test_steane_peham_ft_and_goto_count(steane) -> None:
    """The verified Steane prep is FT with one measurement and at most 13 operations."""
    code, _, ver = steane
    check_noiseless(ver, code.zero_state())
    assert check_ft(ver, code.zero_state(), 1).ok
    m = metrics(ver)
    # one weight-3 check suffices: with an 8-CNOT prep this is Goto's 12 ops, with a 9-CNOT prep 13
    assert m["meas"] == 1
    assert m["ops"] <= 13


def test_fast_enumeration_matches_brute_force(steane) -> None:
    """The fast enumeration counts the same failures as the brute force."""
    code, prep, ver = steane
    for circ in (prep, ver):
        for t in (2, 3):
            rep = check_ft(circ, code.zero_state(), t)
            assert rep.failures == _brute(circ, code.zero_state(), t)


def test_wrong_state_detected(steane) -> None:
    """The noiseless check rejects a prep of the wrong state."""
    code, prep, _ = steane
    plus = CSSState.plus_state(code.hx, code.hz, code.lx)
    with pytest.raises(AssertionError):
        check_noiseless(prep, plus)


def _layered(circ: Circuit) -> Circuit:
    """Same circuit with ops reordered: preps/H first, then CNOT layers (ASAP)."""
    lay = layering(circ)
    pre = [op for op in circ.ops if op.name != "CX"]
    ops = pre + [Op("CX", g) for t in range(1, lay.depth + 1) for g in lay.gates[t]]
    return Circuit(ops, circ.num_qubits, circ.data)


def test_detectors_roundtrip_and_block(steane) -> None:
    """Detectors survive a round trip, and Steane blocks compile to a correct circuit."""
    code, prep, _ = steane
    st = code.zero_state()
    zspec = BlockSpec("Z", gf2_row_basis(st.sz))
    xspec = BlockSpec("X", gf2_row_basis(st.sx))
    # both ancillas are |0_L>: reuse the (non-FT) data prep circuit
    anc = Circuit(list(prep.ops), prep.num_qubits, list(prep.data))
    root = make_node("D", prep, st)
    for spec in (zspec, xspec):
        g = spec.ancilla_generator()
        assert len(gf2_row_basis(np.vstack([g, code.hx]))) == len(gf2_row_basis(code.hx)) == len(g)
        full, target = ancilla_states(spec)
        root.add_block(spec, make_node("A", anc, full, target))
    circ = compile_node(root)
    last_prep_op = circ.base_index[len(root.base.ops) - 1]
    check_noiseless(circ, st)
    back = Circuit.from_stim(circ.to_stim(), data=circ.data)
    assert back.detectors == circ.detectors
    # without detectors the block outcomes are random -> per-record check must fail
    with pytest.raises(AssertionError):
        check_noiseless(Circuit(circ.ops, circ.num_qubits, circ.data), st)
    # a single X fault on the data right after its prep is caught by the Z-block
    tab = fault_table(circ)
    det = circ.detector_matrix()
    k = next(f for f in range(len(tab)) if tab.op_index[f] == last_prep_op and tab.label[f][0] in "XY")
    assert ((tab.flips[k].astype(int) @ det.T) % 2).any()


def test_promotion_steane() -> None:
    """Criger's ILP promotion makes the Steane prep FT."""
    code = load_code("steane")
    prep = prep_circuit(heuristic_prep(code))
    st = code.zero_state()
    circ, _gadgets, log = promote(prep, st, 1, rounds=4, log=lambda *_: None)
    assert log.ok
    check_noiseless(circ, st)
    assert check_ft(circ, st, 1).ok
    assert metrics(circ)["ops"] <= 14


def test_greedy_ft_steane() -> None:
    """The greedy makes the Steane prep FT for every depth weight."""
    code = load_code("steane")
    prep = prep_circuit(heuristic_prep(code))
    st = code.zero_state()
    for lam in (0.0, 2.0):
        circ, _, ok = greedy_ft(prep, st, 1, lam=lam)
        assert ok
        check_noiseless(circ, st)
        assert check_ft(circ, st, 1).ok


@pytest.mark.parametrize("asap_end", [False, True])
def test_check_engines_do_not_depend_on_the_chunk_size(monkeypatch, asap_end: bool) -> None:
    """Stabilizer candidates scored in small chunks give the same checks in the promotion and in the greedy."""
    code = load_code("steane")
    prep = prep_circuit(heuristic_prep(code))
    st = code.zero_state()
    runs = []
    for chunk in (10**9, 2):
        monkeypatch.setattr(gauge, "STAB_CHUNK", chunk)
        _, p_gadgets, plog = promote(prep, st, 1, rounds=4, asap_end=asap_end, log=lambda *_: None)
        _, g_gadgets, g_ok = greedy_ft(prep, st, 1, lam=1.0)
        assert plog.ok
        assert g_ok
        runs.append([[(g.kind, g.pos, g.support, g.flags, g.slots) for g in gs] for gs in (p_gadgets, g_gadgets)])
    assert runs[0] == runs[1]


def test_schedule_same_clifford(steane) -> None:
    """Rescheduling keeps the gates and the state."""
    code, prep, ver = steane
    st = code.zero_state()
    for circ in (prep, ver):
        for seed in range(3):
            c = schedule(circ, seed=seed, noise=0.9)
            assert sorted(map(str, c.ops)) == sorted(map(str, circ.ops))
            check_noiseless(c, st)
            assert metrics(c)["depth"] <= metrics(circ)["depth"] or seed > 0


def test_check_rejects_states_too_large_to_pack() -> None:
    """More independent stabilizers than fit into one 64-bit syndrome key are rejected, not checked wrongly."""
    n = MAX_ROWS + 1
    state = CSSState(np.eye(n, dtype=np.uint8), np.zeros((0, n), dtype=np.uint8))
    with pytest.raises(ValueError, match="at most"):
        check_size(state)
    check_size(CSSState(np.eye(MAX_ROWS, dtype=np.uint8), np.zeros((0, MAX_ROWS), dtype=np.uint8)))


@pytest.fixture(scope="module")
def bare17() -> tuple[Circuit, CSSState]:
    """The bare [[17,1,5]] color prep and its state."""
    return color_17_1_5("depth")


def _brute_triples(circ: Circuit, state: CSSState) -> tuple[int, int, set, set]:
    """Failing sorted key triples: count, raw count, dangerous X and Z syndromes."""
    kx, kz, kc, sx, sz = table_keys(fault_table(circ), state, circ.detector_matrix())
    ux, uz, uc, _, cnt, _ = _group_keys(kx, kz, kc)
    wx, wz = leader_weights(sz, 3), leader_weights(sx, 3)
    cnt = cnt.astype(np.int64)
    n = len(ux)
    fails = raw = 0
    dx, dz = set(), set()
    for i in range(n):
        k, m = np.triu_indices(n - i)
        k, m = k + i, m + i
        ex, ez, ec = ux[i] ^ ux[k] ^ ux[m], uz[i] ^ uz[k] ^ uz[m], uc[i] ^ uc[k] ^ uc[m]
        bx = (ec == 0) & (np.asarray(wx[ex.astype(np.int64)]) > 3)
        bz = (ec == 0) & (np.asarray(wz[ez.astype(np.int64)]) > 3)
        b = bx | bz
        fails += int(b.sum())
        raw += int((cnt[i] * cnt[k] * cnt[m])[b].sum())
        dx |= set(ex[bx].tolist())
        dz |= set(ez[bz].tolist())
    return fails, raw, dx, dz


def test_order_three_matches_brute_force(bare17) -> None:
    """Order 3 matches a count over all sorted key triples, and the collected triples come in key order."""
    c, state = bare17
    partly, _, _ = greedy_ft(c, state, 3, max_steps=3)  # some checks: fewer, other failures
    for circ in (c, partly):
        rep = check_ft(circ, state, 3, collect=True)
        fails, raw, dx, dz = _brute_triples(circ, state)
        assert rep.failures[3] == fails > 0
        assert rep.raw_failures[3] == raw
        wx = leader_weights(gf2_row_basis(state.sz), 3)
        wz = leader_weights(gf2_row_basis(state.sx), 3)
        ent = [(x, z) for x, z, r in zip(rep.failing_x, rep.failing_z, rep.failing_reps, strict=True) if len(r) == 3]
        assert {x for x, _ in ent if wx[x] > 3} == dx
        assert {z for _, z in ent if wz[z] > 3} == dz
        kx, kz, kc, _, _ = table_keys(fault_table(circ), state, circ.detector_matrix())
        ux, uz, uc, _, cnt, _ = _group_keys(kx, kz, kc)
        wx3, wz3 = state_leader(state, "X", 3), state_leader(state, "Z", 3)
        assert isinstance(wx3, np.ndarray)
        assert isinstance(wz3, np.ndarray)
        none = np.zeros(0, dtype=np.int64)
        out = enumerate_faults(
            np.ascontiguousarray(ux),
            np.ascontiguousarray(uz),
            np.ascontiguousarray(uc),
            np.ascontiguousarray(cnt, dtype=np.int64),
            True,
            none,
            wx3,
            True,
            none,
            wz3,
            3,
            True,
            False,
            len(ux),
            np.zeros(len(ux), dtype=np.int64),
        )
        rows = [tuple(r) for r in out[6].reshape(-1, 3).tolist()]
        assert len(rows) == rep.failures[3]
        assert rows == sorted(rows)
        assert all(i <= j <= k for i, j, k in rows)


def test_collected_triples_fail(bare17) -> None:
    """Each collected triple is undetected and leaves the recorded data syndromes."""
    c, state = bare17
    rep = check_ft(c, state, 3, collect=True)
    kx, kz, kc, _, _ = table_keys(fault_table(c), state, c.detector_matrix())
    n = 0
    for x, z, r in zip(rep.failing_x, rep.failing_z, rep.failing_reps, strict=True):
        if len(r) == 3:
            idx = list(r)
            assert np.bitwise_xor.reduce(kc[idx]) == 0  # ty: ignore[no-matching-overload]  (numpy stub gap)
            assert (int(np.bitwise_xor.reduce(kx[idx])), int(np.bitwise_xor.reduce(kz[idx]))) == (x, z)  # ty: ignore[no-matching-overload]  (numpy stub gap)
            n += 1
    assert n > 0


def test_first_fail_witnesses_and_limit(bare17) -> None:
    """The early exit, the witness limit and the limit t <= 3."""
    c, state = bare17
    for t in (2, 3):
        assert check_ft(c, state, t, first_fail=True).ok is False
    rep = check_ft(c, state, 3, max_witnesses=5)
    assert len(rep.witnesses) == 5
    with pytest.raises(ValueError, match="1 to 3"):
        check_ft(c, state, 4)


def test_leader_levels_match_all_supports() -> None:
    """The breadth-first leader tables equal the minimum over all supports of weight <= t (dense and sparse)."""
    rng = np.random.default_rng(3)
    for r, n, t in ((8, 14, 3), (12, 20, 3), (22, 26, 2), (23, 18, 3)):
        h = rng.integers(0, 2, size=(r, n), dtype=np.uint8)
        cols = _pack(h.T).astype(np.int64)
        best = {0: 0}
        for w in range(1, t + 1):
            for supp in itertools.combinations(range(n), w):
                best.setdefault(int(np.bitwise_xor.reduce(cols[list(supp)])), w)  # ty: ignore[no-matching-overload]  (numpy stub gap)
        lw = leader_weights(h, t)
        probe = np.array(sorted(best) + [int(x) for x in rng.integers(0, 1 << r, 200)], dtype=np.int64)
        exp = np.array([best.get(int(x), t + 1) for x in probe])
        assert (np.asarray(lw[probe]) == exp).all()
