# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Example: How to add a new gate to the exact synthesis framework.

This example shows how to add a CZ (controlled-Z) gate to the framework.
The process is:
1. Create a new gate class inheriting from SymbolicGateOperation
2. Implement the symbolic tableau transformations
3. Create a custom gate set dictionary including your new gate
4. Pass the gate set to the encoding function
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import z3

from mqt.qecc.circuit_synthesis.exact.encoding_gate_count import encode_clifford_gate_count
from mqt.qecc.circuit_synthesis.exact.gate_operations import (
    CNOTGate,
    HGate,
    IdentityGate,
    SymbolicGateOperation,
    get_standard_clifford_gate_set,
)
from mqt.qecc.codes.pauli import StabilizerTableau

if TYPE_CHECKING:
    import numpy as np


class CZGate(SymbolicGateOperation):
    """Controlled-Z gate operation.

    CZ gate is symmetric: it applies Z to the target if control is |1⟩,
    and Z to control if target is |1⟩.

    Tableau transformation:
    - Z[:,c] <- Z[:,c] ⊕ X[:,t]
    - Z[:,t] <- Z[:,t] ⊕ X[:,c]
    - X parts unchanged
    """

    IS_TWO_QUBIT: ClassVar[bool] = True
    IS_SYMMETRIC: ClassVar[bool] = True
    IS_SELF_INVERSE: ClassVar[bool] = True

    def __init__(self, control: int, target: int) -> None:
        """Initialize CZ gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.
        """
        self.control = control
        self.target = target

    @classmethod
    def from_qubits(cls, q1: int, q2: int, /) -> CZGate:
        """Instantiate CZ gate from qubit indices."""
        return cls(q1, q2)

    def add_clifford_tableau_transition(
        self,
        solver: z3.Solver,
        tableau_x_curr: np.ndarray,
        tableau_z_curr: np.ndarray,
        tableau_x_next: np.ndarray,
        tableau_z_next: np.ndarray,
    ) -> None:
        """Add CZ gate constraints for Clifford tableau."""
        num_rows = tableau_x_curr.shape[0]
        c = self.control
        t = self.target

        for row in range(num_rows):
            # X parts unchanged
            solver.add(tableau_x_next[row, c] == tableau_x_curr[row, c])
            solver.add(tableau_x_next[row, t] == tableau_x_curr[row, t])

            # Z[:,c] <- Z[:,c] ⊕ X[:,t]
            solver.add(tableau_z_next[row, c] == z3.Xor(tableau_z_curr[row, c], tableau_x_curr[row, t]))

            # Z[:,t] <- Z[:,t] ⊕ X[:,c]
            solver.add(tableau_z_next[row, t] == z3.Xor(tableau_z_curr[row, t], tableau_x_curr[row, c]))

    def add_css_matrix_transition(
        self,
        solver: z3.Solver,
        matrix_curr: np.ndarray,
        matrix_next: np.ndarray,
    ) -> None:
        """CZ gate not applicable to CSS CNOT-only encoding."""
        msg = "CZ gate cannot be applied in CSS CNOT-only encoding"
        raise NotImplementedError(msg)

    def to_stim_gate(self) -> tuple[str, list[int]]:
        """Convert to Stim gate representation."""
        return ("CZ", [self.control, self.target])

    def inverse_stim_gate(self) -> tuple[str, list[int]]:
        """CZ is self-inverse."""
        return ("CZ", [self.control, self.target])

    def qubits(self) -> set[int]:
        """Get qubits involved in this operation."""
        return {self.control, self.target}


def example_usage() -> None:
    """Show how to use a custom gate set in synthesis."""
    # Option 1: Start from standard gate set and add your gate
    custom_gate_set = get_standard_clifford_gate_set()
    custom_gate_set["CZ"] = CZGate

    # Option 2: Build a completely custom gate set
    minimal_gate_set: dict[str, type[SymbolicGateOperation]] = {
        "H": HGate,
        "CX": CNOTGate,
        "CZ": CZGate,
        "ID": IdentityGate,
    }

    # Now use the custom gate set in encoding
    target = StabilizerTableau.from_pauli_strings(["XX", "ZZ"])
    k = 0
    max_gates = 5

    # Pass the custom gate set to the encoding function
    encode_clifford_gate_count(target, k, max_gates, allow_qubit_permutation=True, gate_set=minimal_gate_set)


if __name__ == "__main__":
    example_usage()
