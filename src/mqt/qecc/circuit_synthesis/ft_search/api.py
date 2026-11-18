# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Entry point of the search: the front of fault-tolerant preparation circuits of a code's logical state."""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import stim

from ...codes.core.css_code import CSSCode
from ._core.circuit import Circuit
from ._core.config import Context, profile
from ._core.design import Design, make_node
from ._core.search import search
from ._core.verify import check_noiseless, check_size, state_of_code
from .synth import candidates, check_modes

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .synth import Synthesizer


@dataclass
class FrontCircuit:
    """A fault-tolerant preparation circuit on the front.

    Attributes:
        circuit: The circuit. The code's qubits come first, in the order of the code; the ancilla qubits follow.
            Each detector is the parity of one or more measurements and is 0 when no fault occurs. A shot is
            accepted if all detectors are 0.
        ops: Two-qubit gates plus measurements.
        depth: Two-qubit gate depth of the as-soon-as-possible schedule. Resets, single-qubit gates and measurements
            take no time step.
        cnots: Two-qubit gates.
        measurements: Measurements.
        qubits: Qubits, ancillas included.
    """

    circuit: stim.Circuit
    ops: int
    depth: int
    cnots: int
    measurements: int
    qubits: int


def _compact(circ: stim.Circuit, n: int) -> stim.Circuit:
    """Number the ancilla qubits of ``circ`` consecutively from ``n``; the code's qubits ``0..n-1`` keep theirs."""
    insts = [cast("stim.CircuitInstruction", x) for x in circ.flattened()]  # flattened(): no repeat blocks
    used = sorted({t.value for inst in insts for t in inst.targets_copy() if t.is_qubit_target})
    index = {q: q for q in range(n)} | {q: n + i for i, q in enumerate(q for q in used if q >= n)}
    out = stim.Circuit()
    for inst in insts:
        targets = [index[t.value] if t.is_qubit_target else t for t in inst.targets_copy()]
        out.append(inst.name, targets, inst.gate_args_copy())
    return out


def synthesize_ft_prep(
    code: CSSCode,
    *,
    t: int | None = None,
    zero_state: bool = True,
    preps: Sequence[stim.Circuit] | None = None,
    seconds: float = 600.0,
    workers: int | None = None,
    synth: Sequence[tuple[str, int]] | None = None,
    synthesizers: Mapping[str, Synthesizer] | None = None,
) -> list[FrontCircuit]:
    """Search for fault-tolerant preparation circuits of the logical zero (or plus) state of a CSS code.

    The search grows designs in a tree search over worker processes. A design is a preparation circuit with checks
    on it; a block of checks uses an ancilla state, which may need checks of its own. The search keeps the front of
    the fault-tolerant designs: for each depth the fewest operations, kept only if fewer than at every smaller depth.
    Each new front point is logged (``logging``, level INFO). Ctrl-C stops the search and returns the front so far.

    Args:
        code: The CSS code. For k > 1 logical qubits, all of them are prepared in the zero (or plus) state.
        t: Number of faults to tolerate, 1 to 3; by default ``(d - 1) // 2`` for the code's distance ``d``.
        zero_state: Whether to prepare the logical zero (else the plus) state.
        preps: Non-fault-tolerant preparation circuits to start from, on the code's qubits (gates R, RX, H, CX, CZ);
            by default one from each synthesis mode in ``synth``.
        seconds: Wall-clock time of the search; the call returns after it.
        workers: Worker processes (default: all cores, at most 61 on Windows).
        synth: Synthesis modes for the preparation circuits of the code state and of ancilla states, as
            (mode, circuits per state). Default: one circuit each from "qecc_heuristic", "greedy" and "random"
            (see :mod:`.synth`).
        synthesizers: Further synthesis modes by name: a picklable function ``f(h, k)`` that returns circuits, or a
            :class:`.synth.CommandSynthesizer` that runs an external tool.

    Returns:
        One circuit per front point, by depth. Empty if the search found no fault-tolerant circuit in time (a
        warning says so).

    Raises:
        TypeError: ``code`` is not a CSS code.
        ValueError: ``t`` is unknown or outside 1 to 3, a synthesis mode is unknown, a start circuit does not
            prepare the state, no start circuit was synthesized, or the code is too large for the exact check.
    """
    if not isinstance(code, CSSCode):
        msg = f"expected a CSSCode, got {type(code).__name__}"
        raise TypeError(msg)
    state = state_of_code(code, zero_state=zero_state)
    if t is None:
        if code.distance is None:
            msg = "the code's distance is unknown: give t"
            raise ValueError(msg)
        t = (code.distance - 1) // 2
    if not 1 <= t <= 3:
        msg = f"t = {t}: the search supports 1 to 3 faults"
        raise ValueError(msg)
    check_size(state)
    ctx = Context(synthesizers=dict(synthesizers or {}))
    if synth is not None:
        ctx.synth = list(synth)
    check_modes([mode for mode, _ in ctx.synth], ctx.synthesizers)
    n = code.n
    if preps:
        circuits = [Circuit.from_stim(stim.Circuit(str(p)), data=list(range(n))) for p in preps]
    else:
        circuits = [c for mode, k in ctx.synth for c in candidates(state.sx, mode, k, synthesizers=ctx.synthesizers)]
    if not circuits:
        msg = "no start circuit: the synthesis modes returned none"
        raise ValueError(msg)
    start = []
    for i, c in enumerate(circuits):
        try:
            check_noiseless(c, state)
        except AssertionError as e:
            msg = f"start circuit {i} does not prepare the logical {'zero' if zero_state else 'plus'} state: {e}"
            raise ValueError(msg) from e
        start.append(Design(make_node("D", c, state, origin=f"prep{i}"), t))
    if not workers:
        workers = os.cpu_count() or 1
        if sys.platform == "win32":
            workers = min(workers, 61)  # the largest process pool on Windows
    front = search(start, ctx, profile(t), workers=workers, time_limit=seconds)
    out: dict[tuple[int, int], FrontCircuit] = {}
    for s in front:  # several designs can share a front point: one of them is kept
        m = s["metrics"]
        point = (m["ops"], m["depth"])
        if point not in out:
            circ = _compact(stim.Circuit(s["stim"]), n)
            out[point] = FrontCircuit(circ, m["ops"], m["depth"], m["cnot"], m["meas"], circ.num_qubits)
    if not out:
        warnings.warn(
            f"no fault-tolerant circuit found in {seconds:.0f} s; give the search more time", UserWarning, stacklevel=2
        )
    return sorted(out.values(), key=lambda f: (f.depth, f.ops))
