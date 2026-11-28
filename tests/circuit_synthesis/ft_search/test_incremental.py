# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the incremental check ``verify.check_added``.

The incremental check (``verify.check_added``): adding a check or a flag to a design gives the same raw failure
counts as the full check, at t = 2 and t = 3, on greedy designs of the [[17,1,5]] color prep.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.qecc.circuit_synthesis.ft_search._core import gauge
from mqt.qecc.circuit_synthesis.ft_search._core.complete import (
    _finish,
)
from mqt.qecc.circuit_synthesis.ft_search._core.config import Context
from mqt.qecc.circuit_synthesis.ft_search._core.design import Design, make_node
from mqt.qecc.circuit_synthesis.ft_search._core.faults import (
    _first_fault,
    fault_table,
)
from mqt.qecc.circuit_synthesis.ft_search._core.gauge import build, flag_probes, layered_base
from mqt.qecc.circuit_synthesis.ft_search._core.greedy_ft import greedy_ft
from mqt.qecc.circuit_synthesis.ft_search._core.verify import (
    CSSState,
    check_added,
    check_ft,
)

from .helpers import color_17_1_5

if TYPE_CHECKING:
    from mqt.qecc.circuit_synthesis.ft_search._core.circuit import Circuit


@pytest.fixture(scope="module")
def bare17() -> tuple[Circuit, CSSState]:
    """The bare [[17,1,5]] color prep and its state."""
    return color_17_1_5("depth")


def _pairs(c: Circuit, state: CSSState, t: int) -> list[tuple[str, Circuit, Circuit]]:
    """Added element, old circuit and new circuit from a greedy design.

    (what was added, old circuit, new circuit) from a greedy design: its last check, each existing flag, and a new
    flag on each check of weight >= 3.
    """
    _, gadgets, _ = greedy_ft(c, state, t, max_steps=6)
    base, _ = layered_base(c)
    out = [("check", build(base, gadgets[:-1])[0], build(base, gadgets)[0])]
    fid = 1 + max((f for g in gadgets for *_, f in g.flags), default=-1)
    for gi, g in enumerate(gadgets):
        for flag in g.flags:
            fewer = copy.deepcopy(gadgets)
            fewer[gi].flags.remove(flag)
            out.append(("flag", build(base, fewer)[0], build(base, gadgets)[0]))
        if len(g.support) >= 3 and not g.drive:
            more = copy.deepcopy(gadgets)
            more[gi].flags.append((0, len(g.support) - 1, fid))
            out.append(("new flag", build(base, gadgets)[0], build(base, more)[0]))
    return out


@pytest.mark.parametrize("t", [2, 3])
def test_added_matches_full_check(bare17, t) -> None:
    """The incremental check gives the same raw failure counts as the full check."""
    c, state = bare17
    pairs = _pairs(c, state, t)
    assert any(kind == "new flag" for kind, _, _ in pairs)
    for kind, old, new in pairs:
        before = check_ft(old, state, t, collect=True)
        delta = check_added(new, state, t, old, before)
        assert delta is not None, kind
        full = check_ft(new, state, t)
        assert delta["raw"] == {s: full.raw_failures[s] for s in range(1, t + 1)}, kind
        assert all(v >= 0 for v in delta["added"].values())


def test_flag_probes_give_new_flag_detection(bare17) -> None:
    """The greedy scores a new flag (j, k) by two ancilla frame probes.

    That equals the flag's record in the circuit with the flag, for every fault of the circuit without it.
    """
    c, state = bare17
    _, gadgets, _ = greedy_ft(c, state, 2, max_steps=6)
    base, _ = layered_base(c)
    old, _ = build(base, gadgets)
    probes, col = flag_probes(old, gadgets, base.num_qubits)
    tab = fault_table(old, probes=probes)
    fid = 1 + max((f for g in gadgets for *_, f in g.flags), default=-1)
    n = 0
    for gi, g in enumerate(gadgets):
        for j in range(len(g.support) - 1):
            for k in range(j + 2, len(g.support) + 1):
                if any((j, k) == (a, b) for a, b, _ in g.flags):
                    continue
                more = copy.deepcopy(gadgets)
                more[gi].flags.append((j, k, fid))
                new, _ = build(base, more)
                assert new.tags is not None
                assert old.tags is not None
                assert tab.probe is not None
                meas = [i for i, op in enumerate(new.ops) if op.name in {"M", "MX", "MR", "MRX"}]
                r_flag = next(r for r, i in enumerate(meas) if new.tags[i] == ("fm", fid))
                flips = fault_table(new).flips[:, r_flag]
                pos = {tg: i for i, tg in enumerate(new.tags)}
                f_new, f_old = _first_fault(new), _first_fault(old)
                got = np.concatenate([flips[f_new[pos[tg]] : f_new[pos[tg] + 1]] for tg in old.tags])
                assert len(got) == f_old[-1]
                assert (got == (tab.probe[:, col[gi, j]] ^ tab.probe[:, col[gi, k]])).all(), (gi, j, k)
                n += 1
    assert n > 5


def test_greedy_by_order_reaches_ft(bare17) -> None:
    """The greedy one fault order at a time ends FT at t = 3.

    The t >= 3 path of the greedy (one fault order at a time, flags ranked with the checks, few incremental trials
    per step) ends FT at t = 3.
    """
    c, state = bare17
    out, _, ok = greedy_ft(c, state, 3, by_order=True, top_k=8)
    assert ok
    assert check_ft(out, state, 3).ok


def test_criger_finisher_first_at_t3(bare17) -> None:
    """At t >= 3 the rollout finishes with Criger's ILP promotion first (``Profile.finishers``)."""
    c, state = bare17
    d = Design(make_node("D", c, state), 3)
    outs = _finish(d, Context(), (4.0,))
    assert outs
    assert outs[0][0] == "criger"
    assert check_ft(outs[0][1].circuit(), state, 3).ok


@pytest.mark.slow
def test_gauge_candidates_in_chunks_change_nothing(bare17, monkeypatch) -> None:
    """Gauge candidates scored in chunks give the same checks.

    The promotion scores gauge-operator candidates in chunks (``gauge.GAUGE_CHUNK``; at d = 7 all at once needed
    4.4 GB): tiny chunks pick the same checks as one chunk.
    """
    c, state = bare17
    runs, logged = [], []
    for chunk in (10**9, 7):
        monkeypatch.setattr(gauge, "GAUGE_CHUNK", chunk)
        _, gadgets, plog = gauge.promote(c, state, 2, rounds=4, gauge_w=6, seed=1, log=logged.append)
        runs.append(([(g.kind, g.pos, g.support, g.flags) for g in gadgets], [r["failures"] for r in plog.rounds]))
    assert runs[0] == runs[1]
    pool = next(e["gauge_candidates"] for e in logged if isinstance(e, dict) and "gauge_candidates" in e)
    assert min(pool.values()) > 7  # several chunks
