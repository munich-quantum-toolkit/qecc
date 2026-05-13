# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Gate operation abstractions for exact synthesis encoding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import z3

if TYPE_CHECKING:
    import numpy.typing as npt


class SymbolicGateOperation(ABC):
    """Abstract base class for gate operations in symbolic encoding."""

    @abstractmethod
    def add_clifford_tableau_transition(
        self,
        solver: z3.Solver,
        tableau_x_curr: npt.NDArray,
        tableau_z_curr: npt.NDArray,
        tableau_x_next: npt.NDArray,
        tableau_z_next: npt.NDArray,
    ) -> None:
        """Add constraints for this gate's effect on a Clifford tableau.

        Args:
            solver: Z3 solver instance.
            tableau_x_curr: Current X part of tableau.
            tableau_z_curr: Current Z part of tableau.
            tableau_x_next: Next X part of tableau.
            tableau_z_next: Next Z part of tableau.
        """

    @abstractmethod
    def add_css_matrix_transition(
        self,
        solver: z3.Solver,
        matrix_curr: npt.NDArray,
        matrix_next: npt.NDArray,
    ) -> None:
        """Add constraints for this gate's effect on a CSS check matrix.

        Args:
            solver: Z3 solver instance.
            matrix_curr: Current CSS matrix.
            matrix_next: Next CSS matrix.
        """

    @abstractmethod
    def to_stim_gate(self) -> tuple[str, list[int]]:
        """Convert to Stim gate representation.

        Returns:
            Tuple of (gate_name, qubit_targets).
        """

    @abstractmethod
    def inverse_stim_gate(self) -> tuple[str, list[int]]:
        """Get the inverse gate in Stim representation.

        Returns:
            Tuple of (gate_name, qubit_targets) for the inverse.
        """

    @abstractmethod
    def qubits(self) -> set[int]:
        """Get qubits involved in this operation.

        Returns:
            Set of qubit indices.
        """


class HGate(SymbolicGateOperation):
    """Hadamard gate operation."""

    def __init__(self, qubit: int) -> None:
        """Initialize H gate.

        Args:
            qubit: Target qubit index.
        """
        self.qubit = qubit

    def add_clifford_tableau_transition(
        self,
        solver: z3.Solver,
        tableau_x_curr: npt.NDArray,
        tableau_z_curr: npt.NDArray,
        tableau_x_next: npt.NDArray,
        tableau_z_next: npt.NDArray,
    ) -> None:
        """H gate: swap X and Z columns."""
        num_rows = tableau_x_curr.shape[0]
        q = self.qubit

        for row in range(num_rows):
            solver.add(tableau_x_next[row, q] == tableau_z_curr[row, q])
            solver.add(tableau_z_next[row, q] == tableau_x_curr[row, q])

    def add_css_matrix_transition(
        self,
        solver: z3.Solver,
        matrix_curr: npt.NDArray,
        matrix_next: npt.NDArray,
    ) -> None:
        """H gate not applicable to CSS encoding (requires full Clifford)."""
        msg = "H gate cannot be applied in CSS CNOT-only encoding"
        raise NotImplementedError(msg)

    def to_stim_gate(self) -> tuple[str, list[int]]:
        """Convert to Stim gate."""
        return ("H", [self.qubit])

    def inverse_stim_gate(self) -> tuple[str, list[int]]:
        """H is self-inverse."""
        return ("H", [self.qubit])

    def qubits(self) -> set[int]:
        """Get qubits involved."""
        return {self.qubit}


class SGate(SymbolicGateOperation):
    """S (phase) gate operation."""

    def __init__(self, qubit: int) -> None:
        """Initialize S gate.

        Args:
            qubit: Target qubit index.
        """
        self.qubit = qubit

    def add_clifford_tableau_transition(
        self,
        solver: z3.Solver,
        tableau_x_curr: npt.NDArray,
        tableau_z_curr: npt.NDArray,
        tableau_x_next: npt.NDArray,
        tableau_z_next: npt.NDArray,
    ) -> None:
        """S gate: Z <- Z ⊕ X."""
        num_rows = tableau_x_curr.shape[0]
        q = self.qubit

        for row in range(num_rows):
            solver.add(tableau_x_next[row, q] == tableau_x_curr[row, q])
            solver.add(tableau_z_next[row, q] == z3.Xor(tableau_z_curr[row, q], tableau_x_curr[row, q]))

    def add_css_matrix_transition(
        self,
        solver: z3.Solver,
        matrix_curr: npt.NDArray,
        matrix_next: npt.NDArray,
    ) -> None:
        """S gate not applicable to CSS encoding."""
        msg = "S gate cannot be applied in CSS CNOT-only encoding"
        raise NotImplementedError(msg)

    def to_stim_gate(self) -> tuple[str, list[int]]:
        """Convert to Stim gate."""
        return ("S", [self.qubit])

    def inverse_stim_gate(self) -> tuple[str, list[int]]:
        """Inverse is S_DAG."""
        return ("S_DAG", [self.qubit])

    def qubits(self) -> set[int]:
        """Get qubits involved."""
        return {self.qubit}


class CNOTGate(SymbolicGateOperation):
    """CNOT gate operation."""

    def __init__(self, control: int, target: int) -> None:
        """Initialize CNOT gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.
        """
        self.control = control
        self.target = target

    def add_clifford_tableau_transition(
        self,
        solver: z3.Solver,
        tableau_x_curr: npt.NDArray,
        tableau_z_curr: npt.NDArray,
        tableau_x_next: npt.NDArray,
        tableau_z_next: npt.NDArray,
    ) -> None:
        """CNOT gate: X[:,t] <- X[:,t] ⊕ X[:,c], Z[:,c] <- Z[:,c] ⊕ Z[:,t]."""
        num_rows = tableau_x_curr.shape[0]
        c = self.control
        t = self.target

        for row in range(num_rows):
            solver.add(tableau_x_next[row, c] == tableau_x_curr[row, c])
            solver.add(tableau_x_next[row, t] == z3.Xor(tableau_x_curr[row, t], tableau_x_curr[row, c]))
            solver.add(tableau_z_next[row, c] == z3.Xor(tableau_z_curr[row, c], tableau_z_curr[row, t]))
            solver.add(tableau_z_next[row, t] == tableau_z_curr[row, t])

    def add_css_matrix_transition(
        self,
        solver: z3.Solver,
        matrix_curr: npt.NDArray,
        matrix_next: npt.NDArray,
    ) -> None:
        """CNOT gate for CSS: M[:,t] <- M[:,t] ⊕ M[:,c]."""
        num_rows = matrix_curr.shape[0]
        c = self.control
        t = self.target

        for row in range(num_rows):
            solver.add(matrix_next[row, c] == matrix_curr[row, c])
            solver.add(matrix_next[row, t] == z3.Xor(matrix_curr[row, t], matrix_curr[row, c]))

    def to_stim_gate(self) -> tuple[str, list[int]]:
        """Convert to Stim gate."""
        return ("CX", [self.control, self.target])

    def inverse_stim_gate(self) -> tuple[str, list[int]]:
        """CNOT is self-inverse."""
        return ("CX", [self.control, self.target])

    def qubits(self) -> set[int]:
        """Get qubits involved."""
        return {self.control, self.target}


class IdentityGate(SymbolicGateOperation):
    """Identity (no-op) gate operation."""

    def __init__(self, qubit: int) -> None:
        """Initialize identity gate.

        Args:
            qubit: Qubit index (for bookkeeping in depth encoding).
        """
        self.qubit = qubit

    def add_clifford_tableau_transition(
        self,
        solver: z3.Solver,
        tableau_x_curr: npt.NDArray,
        tableau_z_curr: npt.NDArray,
        tableau_x_next: npt.NDArray,
        tableau_z_next: npt.NDArray,
    ) -> None:
        """Identity: no change."""
        num_rows = tableau_x_curr.shape[0]
        q = self.qubit

        for row in range(num_rows):
            solver.add(tableau_x_next[row, q] == tableau_x_curr[row, q])
            solver.add(tableau_z_next[row, q] == tableau_z_curr[row, q])

    def add_css_matrix_transition(
        self,
        solver: z3.Solver,
        matrix_curr: npt.NDArray,
        matrix_next: npt.NDArray,
    ) -> None:
        """Identity: no change."""
        num_rows = matrix_curr.shape[0]
        q = self.qubit

        for row in range(num_rows):
            solver.add(matrix_next[row, q] == matrix_curr[row, q])

    def to_stim_gate(self) -> tuple[str, list[int]]:
        """Convert to Stim gate."""
        return ("I", [self.qubit])

    def inverse_stim_gate(self) -> tuple[str, list[int]]:
        """Identity is self-inverse."""
        return ("I", [self.qubit])

    def qubits(self) -> set[int]:
        """Get qubits involved."""
        return {self.qubit}


class GateRegistry:
    """Registry of available gate operations for synthesis."""

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._clifford_gates: dict[str, type[SymbolicGateOperation]] = {}
        self._css_gates: dict[str, type[SymbolicGateOperation]] = {}

    def register_clifford_gate(self, name: str, gate_class: type[SymbolicGateOperation]) -> None:
        """Register a gate for Clifford synthesis.

        Args:
            name: Gate identifier (e.g., 'H', 'S', 'CX').
            gate_class: Gate operation class.
        """
        self._clifford_gates[name] = gate_class

    def register_css_gate(self, name: str, gate_class: type[SymbolicGateOperation]) -> None:
        """Register a gate for CSS synthesis.

        Args:
            name: Gate identifier (e.g., 'CX').
            gate_class: Gate operation class.
        """
        self._css_gates[name] = gate_class

    def get_clifford_gates(self) -> dict[str, type[SymbolicGateOperation]]:
        """Get all registered Clifford gates.

        Returns:
            Dictionary mapping gate names to gate classes.
        """
        return self._clifford_gates.copy()

    def get_css_gates(self) -> dict[str, type[SymbolicGateOperation]]:
        """Get all registered CSS gates.

        Returns:
            Dictionary mapping gate names to gate classes.
        """
        return self._css_gates.copy()

    def create_gate(self, name: str, *args: Any, for_css: bool = False) -> SymbolicGateOperation:
        """Create a gate instance.

        Args:
            name: Gate identifier.
            *args: Arguments for gate constructor.
            for_css: Whether this is for CSS encoding.

        Returns:
            Gate instance.

        Raises:
            KeyError: If gate not registered.
        """
        gates = self._css_gates if for_css else self._clifford_gates
        if name not in gates:
            msg = f"Gate '{name}' not registered for {'CSS' if for_css else 'Clifford'} synthesis"
            raise KeyError(msg)
        return gates[name](*args)


def get_standard_clifford_gate_set() -> dict[str, type[SymbolicGateOperation]]:
    """Get the standard Clifford gate set {H, S, CX, ID}.

    Returns:
        Dictionary mapping gate names to gate classes.
    """
    return {
        "H": HGate,
        "S": SGate,
        "CX": CNOTGate,
        "ID": IdentityGate,
    }


def get_standard_css_gate_set() -> dict[str, type[SymbolicGateOperation]]:
    """Get the standard CSS gate set {CX, ID}.

    Returns:
        Dictionary mapping gate names to gate classes.
    """
    return {
        "CX": CNOTGate,
        "ID": IdentityGate,
    }
