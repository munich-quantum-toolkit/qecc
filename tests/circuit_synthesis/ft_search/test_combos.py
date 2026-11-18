# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the class-consistent coverage of ``combos.Combos``.

The vectorized class-consistent coverage (``combos.Combos``) equals a plain loop over the combinations, on the failing
combinations of a bare prep (t = 2) with random detection patterns; the sample keeps the lowest order complete.
"""

from __future__ import annotations

import numpy as np

from mqt.qecc.circuit_synthesis.ft_search._core.combos import Combos
from mqt.qecc.circuit_synthesis.ft_search._core.faults import fault_table
from mqt.qecc.circuit_synthesis.ft_search._core.verify import (
    check_ft,
)

from .helpers import color_17_1_5, table_keys


def _setup():
    circ, state = color_17_1_5("gate")
    rep = check_ft(circ, state, 2, collect=True)
    tab = fault_table(circ)
    kx, kz, kc, _, _ = table_keys(tab, state, circ.detector_matrix())
    _, cls = np.unique(np.stack([kx, kz, kc], axis=1), axis=0, return_inverse=True)
    return rep, cls.ravel()


def _old(rep, cls, bm_full):
    combos = list(dict.fromkeys(tuple(int(cls[f]) for f in c) for c in rep.failing_reps))
    used = sorted({k for c in combos for k in c})
    members = {k: np.nonzero(cls == k)[0] for k in used}
    faults = sorted({int(f) for k in used for f in members[k]})
    fidx = {f: i for i, f in enumerate(faults)}
    bm = bm_full[faults]
    out = np.zeros((len(combos), bm.shape[1]), dtype=np.uint8)
    valid = np.ones(out.shape, dtype=bool)
    for ci, c in enumerate(combos):
        for k in c:
            rows = bm[[fidx[int(f)] for f in members[k]]]
            out[ci] ^= rows[0]
            valid[ci] &= (rows == rows[0]).all(axis=0)
    return faults, np.where(valid, out, 0)


def test_combine_equals_loop() -> None:
    """The vectorized coverage equals the loop over the combinations."""
    rep, cls = _setup()
    rng = np.random.default_rng(0)
    bm_full = rng.integers(0, 2, size=(len(cls), 40), dtype=np.uint8)
    faults, ref = _old(rep, cls, bm_full)
    cb = Combos(rep, cls)
    assert list(cb.faults) == faults
    assert len(cb) == len(ref)
    assert (cb.combine(bm_full[cb.faults]) == ref).all()


def test_sample_keeps_lowest_order() -> None:
    """A sample keeps every combination of the lowest order and scales the rest."""
    rep, cls = _setup()
    full = Combos(rep, cls)
    n1 = int((full.ns == np.min(full.ns)).sum())
    cb = Combos(rep, cls, max_combos=n1 + 50, seed=3)
    assert len(cb) == n1 + 50
    assert cb.total == len(full)
    assert (cb.ns[:n1] == np.min(full.ns)).all()
    assert np.isclose(cb.scale[n1:].mean(), (len(full) - n1) / 50)
