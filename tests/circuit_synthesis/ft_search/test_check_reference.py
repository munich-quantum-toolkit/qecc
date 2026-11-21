# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""The exact check against recorded results (verdict and failure counts) on search designs and bare preps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import stim

from mqt.qecc.circuit_synthesis.ft_search._core.circuit import Circuit
from mqt.qecc.circuit_synthesis.ft_search._core.verify import CSSState, check_ft

CASES = json.loads((Path(__file__).parent / "data" / "check_reference.json").read_text(encoding="utf-8"))
PARAMS = [(c, t) for c in CASES for t in c["results"]]


@pytest.mark.parametrize(("case", "t"), PARAMS, ids=[f"{c['name']}_t{t}" for c, t in PARAMS])
def test_check_matches_reference(case: dict, t: str) -> None:
    """The verdict and the failure counts per fault order equal the recorded ones; the early-exit check agrees."""
    sx, sz = np.array(case["sx"], dtype=np.uint8), np.array(case["sz"], dtype=np.uint8)
    state = CSSState(sx, sz)
    circ = Circuit.from_stim(stim.Circuit(case["stim"]), data=list(range(sx.shape[1])))
    want = case["results"][t]
    rep = check_ft(circ, state, int(t))
    assert rep.ok == want["ok"]
    assert {str(s): n for s, n in rep.failures.items()} == want["failures"]
    assert check_ft(circ, state, int(t), first_fail=True).ok == want["ok"]
