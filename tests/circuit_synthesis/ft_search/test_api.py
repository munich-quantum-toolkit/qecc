# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""The entry point ``synthesize_ft_prep`` on the Steane code: a short search returns a front of FT circuits."""

from __future__ import annotations

import itertools

import pytest
import stim

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import heuristic_prep_circuit
from mqt.qecc.circuit_synthesis.ft_search import synthesize_ft_prep
from mqt.qecc.circuit_synthesis.ft_search._core import search
from mqt.qecc.circuit_synthesis.ft_search._core.circuit import Circuit
from mqt.qecc.circuit_synthesis.ft_search._core.verify import check_ft, check_noiseless, state_of_code


@pytest.mark.parametrize("zero_state", [True, False])
def test_short_search(zero_state: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 15-second search with two workers finds a front of FT Steane circuits, also without numba's compiler.

    The front has one circuit per depth, each deeper one with fewer operations. The restart group and the re-scoring
    run every few seconds instead of every few minutes.
    """
    monkeypatch.setattr(search, "RESTART_EVERY", 2.0)
    monkeypatch.setattr(search, "RESCORE_EVERY", 1.0)
    code = CSSCode.from_code_name("Steane")
    front = synthesize_ft_prep(code, zero_state=zero_state, seconds=15, workers=2)
    assert front
    assert [f.depth for f in front] == sorted({f.depth for f in front})  # one circuit per depth, by depth
    assert all(a.ops > b.ops for a, b in itertools.pairwise(front))  # deeper only if fewer operations
    state = state_of_code(code, zero_state=zero_state)
    for f in front:
        insts = [x for x in f.circuit.flattened() if isinstance(x, stim.CircuitInstruction)]
        used = {t.value for inst in insts for t in inst.targets_copy() if t.is_qubit_target}
        assert used == set(range(f.circuit.num_qubits)) == set(range(f.qubits))  # ancillas numbered after the data
        assert f.cnots + f.measurements == f.ops
        c = Circuit.from_stim(f.circuit, data=list(range(code.n)))
        check_noiseless(c, state)
        assert check_ft(c, state, 1).ok


def test_t_needs_a_distance() -> None:
    """A code without a distance raises a ValueError."""
    code = CSSCode.from_code_name("Steane")
    code.distance = None  # ty: ignore[invalid-assignment]  (a code without a distance on purpose)
    with pytest.raises(ValueError, match="distance"):
        synthesize_ft_prep(code)


@pytest.mark.parametrize("t", [0, 4])
def test_t_outside_the_supported_range(t: int) -> None:
    """The search supports 1 to 3 faults."""
    with pytest.raises(ValueError, match="1 to 3"):
        synthesize_ft_prep(CSSCode.from_code_name("Steane"), t=t)


def test_unknown_synthesis_mode() -> None:
    """An unknown mode is reported before the search starts, also when start circuits are given."""
    code = CSSCode.from_code_name("Steane")
    prep = heuristic_prep_circuit(code).circ.to_stim_circuit()
    with pytest.raises(ValueError, match="unknown synthesis mode 'no_such_tool'"):
        synthesize_ft_prep(code, preps=[prep], synth=[("no_such_tool", 1)])


def test_start_circuit_must_prepare_the_state() -> None:
    """A start circuit for the wrong state is a ValueError that names the circuit."""
    code = CSSCode.from_code_name("Steane")
    with pytest.raises(ValueError, match="start circuit 0"):
        synthesize_ft_prep(code, preps=[stim.Circuit("R 0 1 2 3 4 5 6")])


def test_not_a_css_code() -> None:
    """Only CSS codes are accepted."""
    with pytest.raises(TypeError, match="CSSCode"):
        synthesize_ft_prep("Steane")  # ty: ignore[invalid-argument-type]  (a wrong type on purpose)


def test_ctrl_c_returns_the_front_so_far(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C while the search waits for its workers ends the search; nothing found yet gives an empty front."""

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(search, "wait", interrupt)
    with pytest.warns(UserWarning, match="no fault-tolerant circuit"):
        assert synthesize_ft_prep(CSSCode.from_code_name("Steane"), seconds=60, workers=1) == []
