# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Variable containers for Clifford exact synthesis encodings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import z3

    from .gate_operations import SymbolicGateOperation


@dataclass
class CliffordGateCountVars:
    """SAT variables for a Clifford gate-count encoding instance.

    Attributes:
        solver: Configured Z3 solver with all encoding constraints already added.
        gate_sel: Map from gate name to per-slot selection booleans.
            Exactly one entry across all gate names is True per slot (enforced
            by a PbEq constraint in the encoding).  The identity gate ``"ID"``
            is never present — it is a depth-only concept.
        alpha: Per-slot first-qubit index (target for single-qubit gates,
            control for two-qubit gates).
        beta: Per-slot second-qubit index (only meaningful for two-qubit gates).
        gate_set: Gate classes used during encoding, keyed by gate name.
            Needed by consumers (symmetry breaking, extraction) to determine
            gate properties such as ``IS_SELF_INVERSE`` and ``IS_TWO_QUBIT``.
    """

    solver: z3.Solver
    gate_sel: dict[str, list[z3.BoolRef]]
    alpha: list[z3.BitVecRef]
    beta: list[z3.BitVecRef]
    gate_set: dict[str, type[SymbolicGateOperation]]


@dataclass
class CliffordDepthVars:
    """SAT variables for a Clifford depth encoding instance.

    Attributes:
        solver: Configured Z3 solver with all encoding constraints already added.
        gate_vars: Map from gate name to ``[layer][idx]`` boolean arrays.
            The length of the inner list encodes the gate's index structure:

            - ``n`` entries → single-qubit gate; ``idx`` is the qubit index.
            - ``n*(n-1)`` entries → ordered two-qubit gate (CX-like); ``idx``
              encodes the ordered pair as
              ``ctrl*(n-1) + (tgt if tgt < ctrl else tgt-1)``.
            - ``n*(n-1)//2`` entries → symmetric two-qubit gate (CZ-like);
              ``idx`` encodes the unordered pair as
              ``i*(2n-i-1)//2 + (j-i-1)`` with ``i < j``.
        n: Number of qubits; needed by consumers to decode pair indices.
        gate_set: Gate classes used during encoding, keyed by gate name.
            Needed by consumers (symmetry breaking, extraction) to determine
            gate properties such as ``IS_SELF_INVERSE``.
    """

    solver: z3.Solver
    gate_vars: dict[str, list[list[z3.BoolRef]]]
    n: int
    gate_set: dict[str, type[SymbolicGateOperation]]
