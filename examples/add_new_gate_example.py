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
3. Register the gate with the global registry
4. (Optional) Update encoding functions to use the new gate
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import z3

from mqt.qecc.circuit_synthesis.exact.gate_operations import (
    SymbolicGateOperation,
    get_gate_registry,
)

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

    def __init__(self, control: int, target: int) -> None:
        """Initialize CZ gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.
        """
        self.control = control
        self.target = target

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


# Register the new gate
registry = get_gate_registry()
registry.register_clifford_gate("CZ", CZGate)


def example_usage() -> None:
    """Show how to use the new CZ gate in synthesis."""
    # Now CZ is available for synthesis!
    # You can create instances:
    registry.create_gate("CZ", 0, 1, for_css=False)


    # The gate can now be used in encoding functions by updating
    # the gate selection logic to include CZ as an option


if __name__ == "__main__":
    example_usage()
