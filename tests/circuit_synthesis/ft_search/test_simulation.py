# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Noisy simulation of a found circuit with MQT QECC's simulator (``simulation``)."""

from __future__ import annotations

import numpy as np

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis.ft_search._core.blocks import dual
from mqt.qecc.circuit_synthesis.ft_search._core.verify import CSSState, check_noiseless
from mqt.qecc.circuit_synthesis.ft_search.simulation import logical_error_rate
from mqt.qecc.circuit_synthesis.ft_search.synth import candidates


def _steane_zero() -> tuple:
    code = CSSCode.from_code_name("Steane")
    hx = np.asarray(code.Hx, dtype=np.uint8)
    state = CSSState(hx, dual(hx))  # logical zero: the uniform superposition over the X checks
    circ = candidates(hx, "greedy", 1)[0]
    check_noiseless(circ, state)
    return code, circ.to_stim()


def test_noiseless_prep_has_no_logical_errors() -> None:
    """A noiseless prep has no logical errors and keeps every shot."""
    code, circ = _steane_zero()
    ler, acc, errors, _ = logical_error_rate(circ, code, 0.0, shots=2000, min_errors=0)
    assert (ler, acc, errors) == (0.0, 1.0, 0)


def test_unverified_prep_has_logical_errors() -> None:
    """Without checks every shot is kept, and a single fault can cause a logical error."""
    code, circ = _steane_zero()
    ler, acc, errors, _ = logical_error_rate(circ, code, 0.01, shots=20000, min_errors=0)
    assert acc == 1.0  # ruff: ignore[float-equality-comparison]  (every shot is kept)
    assert errors > 0
    assert 0 < ler < 0.2


def test_max_shots_caps_the_sampling() -> None:
    """Sampling stops at ``max_shots`` when the requested errors do not occur."""
    code, circ = _steane_zero()
    *_, errors, shots = logical_error_rate(circ, code, 0.0, shots=1000, min_errors=5, max_shots=3000)
    assert (errors, shots) == (0, 3000)
