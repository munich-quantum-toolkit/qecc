# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Candidate preparation circuits for a state with X-type generator ``h`` (the uniform superposition over rowspan(h)).

A synthesis mode is a name the search asks for (``Context.synth``). Built in:
  * "qecc_heuristic" / "qecc_gate": MQT QECC's heuristic synthesis, depth- or gate-optimizing (the default);
  * "qecc_depth": MQT QECC's SAT-based depth-optimal synthesis, the heuristic if it finds nothing;
  * "greedy": greedy column elimination (``guided.greedy_candidates``);
  * "random": greedy with random improving moves: non-optimal circuits with other fault sets.
Other tools plug in as further modes (``Context.synthesizers``): a picklable function ``f(h, k)`` that returns up to
``k`` circuits, or an external command (:class:`CommandSynthesizer`). Results are memoized per process.
"""

from __future__ import annotations

import functools
import os
import shlex
import subprocess  # ruff: ignore[suspicious-subprocess-import]  (runs the configured external synthesizer)
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import stim

from ...codes.core.css_code import CSSCode
from ..state_prep import depth_optimal_prep_circuit, heuristic_prep_circuit
from ._core import memo
from ._core.blocks import dual
from ._core.circuit import Circuit, Op
from ._core.guided import greedy_candidates
from ._core.verify import CSSState

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    Synthesizer = Callable[[np.ndarray, int], list[Circuit] | list[stim.Circuit]]

__all__ = ["CommandSynthesizer", "Synthesizer"]

QECC_TIMEOUT = 30  # seconds per SAT call of "qecc_depth"
QECC_ROLLOUT = 2  # rollout steps of the heuristics of MQT QECC


class CommandSynthesizer:
    """A synthesis mode that runs an external command.

    The command runs as ``<command> <dir> H <k> <mode>``. It reads the X-type generators ``h`` (key ``h``, uint8, one
    row per generator, column i = qubit i) from ``<dir>/H.npz`` and writes up to ``k`` stim circuits to
    ``<dir>/cand_H_<i>.stim``. Each circuit must prepare the state stabilized by the rows of ``h`` as X-type
    stabilizers from ``|0...0>``, with the gates R, RX, H, CX and CZ.
    """

    def __init__(self, command: str | Sequence[str], mode: str) -> None:
        """Set the command and the mode string passed to it.

        Args:
            command: The program and its first arguments, as a list or as one string split like a command line.
            mode: Passed to the command as its last argument.
        """
        if isinstance(command, str):
            command = [a.strip('"') for a in shlex.split(command, posix=os.name != "nt")]
        self.command, self.mode = list(command), mode

    def __call__(self, h: np.ndarray, k: int) -> list[Circuit]:
        """Up to ``k`` circuits from the external command.

        Raises:
            RuntimeError: The command failed; the message holds its error output.
        """
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(Path(tmp) / "H.npz", h=h)
            cmd = [*self.command, tmp, "H", str(k), self.mode]
            try:
                subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  (the configured command)
                    cmd, check=True, capture_output=True, text=True
                )
            except (subprocess.CalledProcessError, OSError) as e:
                err = getattr(e, "stderr", "") or str(e)
                msg = f"synthesis command {cmd[0]!r} failed: {err.strip()}"
                raise RuntimeError(msg) from e
            files = sorted(Path(tmp).glob("cand_H_*.stim"), key=lambda f: int(f.stem.rsplit("_", 1)[1]))[:k]
            return [Circuit.from_stim(stim.Circuit(f.read_text()), data=list(range(h.shape[1]))) for f in files]

    def __repr__(self) -> str:
        """The command and mode (part of the memo key of its results)."""
        return f"CommandSynthesizer({self.command!r}, {self.mode!r})"


def code_of_state(state: CSSState) -> CSSCode:
    """A CSS state as a code of MQT QECC with no logical qubits (its stabilizers fix the state completely)."""
    return CSSCode(
        Hx=np.asarray(state.sx, dtype=np.int8), Hz=np.asarray(state.sz, dtype=np.int8), x_distance=1, z_distance=1
    )


def _qecc_candidates(h: np.ndarray, k: int, *, mode: str) -> list[Circuit]:
    """Up to ``k`` circuits from the synthesis of MQT QECC on random column orders (the first: the given order)."""
    m, rng, out, seen = h.shape[1], np.random.default_rng(0), [], set()
    for i in range(k):
        perm = np.arange(m) if i == 0 else rng.permutation(m)
        g = h[:, perm]
        code = code_of_state(CSSState(g, dual(g)))
        with warnings.catch_warnings():
            # MQT QECC warns about 0 correctable errors of the distance-less ancilla code
            warnings.simplefilter("ignore")
            res = None
            if mode == "qecc_depth":
                res = depth_optimal_prep_circuit(
                    code, zero_state=True, min_depth=1, max_depth=m, min_timeout=2, max_timeout=QECC_TIMEOUT
                )
            if res is None:
                res = heuristic_prep_circuit(
                    code, optimize_depth=mode != "qecc_gate", state="all_zero", rollout=QECC_ROLLOUT
                )
        c = Circuit.from_stim(res.circ.to_stim_circuit(), data=list(range(m)))
        mapped = Circuit([Op(op.name, tuple(int(perm[q]) for q in op.qubits)) for op in c.ops], m, list(range(m)))
        if str(mapped.to_stim()) not in seen:
            seen.add(str(mapped.to_stim()))
            out.append(mapped)
    return out


BUILT_IN: dict[str, Synthesizer] = {
    mode: functools.partial(_qecc_candidates, mode=mode) for mode in ("qecc_depth", "qecc_heuristic", "qecc_gate")
}


MODES = ("greedy", "random", *BUILT_IN)


def check_modes(modes: Sequence[str], synthesizers: Mapping[str, Synthesizer] | None = None) -> None:
    """Raise ValueError for a synthesis mode that is neither built in nor in ``synthesizers``.

    Args:
        modes: Mode names.
        synthesizers: Modes added by the caller.

    Raises:
        ValueError: An unknown mode, with the list of known ones.
    """
    known = (*MODES, *(synthesizers or {}))
    for mode in modes:
        if mode not in known:
            raise ValueError(_unknown(mode, known))


def _unknown(mode: str, known: Sequence[str]) -> str:
    return f"unknown synthesis mode {mode!r}; known modes: {', '.join(known)}"


def _tool_key(synthesize: object) -> str:
    """Name of the tool behind a mode, so the memo tells two tools under one mode name apart."""
    if isinstance(synthesize, functools.partial):
        return f"{_tool_key(synthesize.func)}{synthesize.args!r}{sorted(synthesize.keywords.items())!r}"
    name = getattr(synthesize, "__qualname__", None)
    return f"{getattr(synthesize, '__module__', '')}.{name}" if name else repr(synthesize)


def candidates(
    h: np.ndarray,
    mode: str = "greedy",
    k: int = 4,
    *,
    seed: int = 0,
    synthesizers: Mapping[str, Synthesizer] | None = None,
) -> list[Circuit]:
    """Up to ``k`` preparation circuits of the state with X-type generators ``h``.

    Args:
        h: X-type generators of the state (rows over the qubits).
        mode: A built-in mode (:data:`MODES`) or one of ``synthesizers``.
        k: Maximum number of circuits.
        seed: Seed of the "greedy" and "random" modes.
        synthesizers: Modes added by the caller. A tool may return :class:`stim.Circuit` objects.

    Returns:
        The circuits, fewest CNOTs first.

    Raises:
        ValueError: An unknown mode.
    """
    h = np.asarray(h, dtype=np.uint8) % 2
    if mode == "greedy":
        return greedy_candidates(h, k, seed=seed)
    if mode == "random":  # deliberately non-optimal: random improving moves, two noise levels
        half = (k + 1) // 2
        return (
            greedy_candidates(h, half, seed=seed + 101, eps=0.3)
            + greedy_candidates(h, k - half, seed=seed + 202, eps=0.6)
        )[:k]
    synthesize = (synthesizers or {}).get(mode) or BUILT_IN.get(mode)
    if synthesize is None:
        raise ValueError(_unknown(mode, (*MODES, *(synthesizers or {}))))
    key = memo.digest("synth", h.tobytes(), h.shape, mode, _tool_key(synthesize), k)
    hit = cast("list[Circuit] | None", memo.get("synth", key))
    if hit is None:
        data = list(range(h.shape[1]))
        circs = [Circuit.from_stim(c, data=data) if isinstance(c, stim.Circuit) else c for c in synthesize(h, k)]
        hit = sorted(circs, key=lambda c: (sum(op.name == "CX" for op in c.ops), str(c.to_stim())))[:k]
        memo.put("synth", key, hit)
    return list(hit)
