# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Bell reduction.

Bell reduction: every result of the beam search is noiseless-correct and FT, and each reduction saves one CX and one
measurement. On a known FT design with transversal blocks (whose ancilla encoders contain fresh |+>/|0> pairs).
"""

from __future__ import annotations

from mqt.qecc.circuit_synthesis.ft_search._core.bellred import (
    apply,
    candidates,
    normalize,
    prune_ancillas,
    reduce_front,
)
from mqt.qecc.circuit_synthesis.ft_search._core.circuit import Circuit, metrics
from mqt.qecc.circuit_synthesis.ft_search._core.verify import CSSState, check_ft, check_noiseless

from .helpers import RESULTS, color_17_1_5_state, load_circuit


def _recursive() -> tuple[Circuit, CSSState]:
    """An FT [[17,1,5]] color code |0> preparation with recursive blocks, and its state."""
    return load_circuit(RESULTS / "final" / "color_17_1_5_recursive_gate.stim", 17), color_17_1_5_state()


def _ft(c: Circuit, state: CSSState) -> bool:
    try:
        check_noiseless(c, state)
    except (AssertionError, ValueError):
        return False
    return check_ft(c, state, 2).ok


def test_single_reductions_keep_the_state() -> None:
    """Each single Bell reduction keeps the state and saves one CNOT and one measurement."""
    circ, state = _recursive()
    c0 = normalize(circ)
    check_noiseless(c0, state)
    cands = candidates(c0)
    assert cands
    for i, x in cands:
        r = apply(c0, i, x)
        if r is None:
            continue
        check_noiseless(r, state)  # the rewrite never changes the prepared state or breaks a detector
        assert metrics(r)["cnot"] == metrics(c0)["cnot"] - 1
        assert metrics(r)["meas"] == metrics(c0)["meas"] - 1


def test_reduce_front_results_are_ft_and_cheaper() -> None:
    """Every result of the beam search is FT and saves two operations per reduction."""
    circ, state = _recursive()
    assert _ft(circ, state)
    ops0 = metrics(circ)["ops"]
    found = reduce_front(circ, lambda c: _ft(c, state), beam=2, max_tries=60)
    assert found
    for c, n in found:
        assert _ft(c, state)
        assert metrics(c)["ops"] == ops0 - 2 * n


def test_prune_ancillas_keeps_ft() -> None:
    """Ancilla pruning keeps the circuit FT and removes ancillas only if it saves operations."""
    circ, state = _recursive()
    found = reduce_front(circ, lambda c: _ft(c, state), beam=2, max_tries=60)
    c, _ = found[0]
    pruned, removed = prune_ancillas(c, lambda x: _ft(x, state))
    assert _ft(pruned, state)
    assert metrics(pruned)["ops"] <= metrics(c)["ops"]
    assert bool(removed) == (metrics(pruned)["ops"] < metrics(c)["ops"])
