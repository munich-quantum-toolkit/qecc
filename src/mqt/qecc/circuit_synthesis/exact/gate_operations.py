# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Gate operation abstractions for exact synthesis encoding."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import z3

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class SymbolicGateOperation(ABC):
    """Abstract base class for gate operations in symbolic encoding.

    Class attributes (must be set on every concrete subclass):
        IS_TWO_QUBIT: True for two-qubit gates (CX, CZ, …).
        IS_SYMMETRIC: True when the gate treats both qubits symmetrically so
            that only unordered pairs need to be enumerated (e.g. CZ).
            Ignored for single-qubit gates.
        IS_SELF_INVERSE: True when applying the gate twice yields the identity
            in the binary stabilizer tableau (ignoring global phases).  This
            holds for H, CX, CZ, and also for S and SX: although S and SX have
            order 4 as unitaries, their binary tableau actions square to the
            identity (S: z↦x⊕z twice gives z; SX: x↦x⊕z twice gives x).
            Used by symmetry-breaking to prune adjacent identical gates.
    """

    IS_TWO_QUBIT: ClassVar[bool]
    IS_SYMMETRIC: ClassVar[bool]
    IS_SELF_INVERSE: ClassVar[bool]

    @classmethod
    @abstractmethod
    def from_qubits(cls, q1: int, q2: int, /) -> SymbolicGateOperation:
        """Instantiate the gate from up to two qubit indices.

        Single-qubit gates use *q1* and ignore *q2*.  Two-qubit gates use
        both.  This uniform interface allows extraction code to create gate
        instances without knowing the concrete arity.

        Args:
            q1: First (or only) qubit index.
            q2: Second qubit index (ignored for single-qubit gates).

        Returns:
            A concrete gate instance.
        """

    @abstractmethod
    def add_clifford_tableau_transition(
        self,
        solver: z3.Solver,
        tableau_x_curr: npt.NDArray[np.object_],
        tableau_z_curr: npt.NDArray[np.object_],
        tableau_x_next: npt.NDArray[np.object_],
        tableau_z_next: npt.NDArray[np.object_],
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
        matrix_curr: npt.NDArray[np.object_],
        matrix_next: npt.NDArray[np.object_],
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

    IS_TWO_QUBIT: ClassVar[bool] = False
    IS_SYMMETRIC: ClassVar[bool] = False
    IS_SELF_INVERSE: ClassVar[bool] = True

    def __init__(self, qubit: int) -> None:
        """Initialize H gate.

        Args:
            qubit: Target qubit index.
        """
        self.qubit = qubit

    @classmethod
    def from_qubits(cls, q1: int, _q2: int) -> HGate:
        """Instantiate H gate from qubit indices (_q2 ignored)."""
        return cls(q1)

    def add_clifford_tableau_transition(
        self,
        solver: z3.Solver,
        tableau_x_curr: npt.NDArray[np.object_],
        tableau_z_curr: npt.NDArray[np.object_],
        tableau_x_next: npt.NDArray[np.object_],
        tableau_z_next: npt.NDArray[np.object_],
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
        matrix_curr: npt.NDArray[np.object_],
        matrix_next: npt.NDArray[np.object_],
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

    IS_TWO_QUBIT: ClassVar[bool] = False
    IS_SYMMETRIC: ClassVar[bool] = False
    IS_SELF_INVERSE: ClassVar[bool] = True

    def __init__(self, qubit: int) -> None:
        """Initialize S gate.

        Args:
            qubit: Target qubit index.
        """
        self.qubit = qubit

    @classmethod
    def from_qubits(cls, q1: int, _q2: int) -> SGate:
        """Instantiate S gate from qubit indices (_q2 ignored)."""
        return cls(q1)

    def add_clifford_tableau_transition(
        self,
        solver: z3.Solver,
        tableau_x_curr: npt.NDArray[np.object_],
        tableau_z_curr: npt.NDArray[np.object_],
        tableau_x_next: npt.NDArray[np.object_],
        tableau_z_next: npt.NDArray[np.object_],
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
        matrix_curr: npt.NDArray[np.object_],
        matrix_next: npt.NDArray[np.object_],
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


class SqrtXGate(SymbolicGateOperation):
    """√X gate (SX = HSH) operation.

    Binary tableau action: X_out = X ⊕ Z, Z_out = Z.
    Self-inverse in the binary tableau: (SX)^2 = X as a unitary, but the
    binary tableau action squares to the identity (x⊕z)⊕z = x.
    """

    IS_TWO_QUBIT: ClassVar[bool] = False
    IS_SYMMETRIC: ClassVar[bool] = False
    IS_SELF_INVERSE: ClassVar[bool] = True

    def __init__(self, qubit: int) -> None:
        """Initialize SX gate.

        Args:
            qubit: Target qubit index.
        """
        self.qubit = qubit

    @classmethod
    def from_qubits(cls, q1: int, _q2: int) -> SqrtXGate:
        """Instantiate SX gate from qubit indices (_q2 ignored)."""
        return cls(q1)

    def add_clifford_tableau_transition(
        self,
        solver: z3.Solver,
        tableau_x_curr: npt.NDArray[np.object_],
        tableau_z_curr: npt.NDArray[np.object_],
        tableau_x_next: npt.NDArray[np.object_],
        tableau_z_next: npt.NDArray[np.object_],
    ) -> None:
        """SX gate: X_out = X ⊕ Z, Z_out = Z."""
        num_rows = tableau_x_curr.shape[0]
        q = self.qubit

        for row in range(num_rows):
            solver.add(tableau_x_next[row, q] == z3.Xor(tableau_x_curr[row, q], tableau_z_curr[row, q]))
            solver.add(tableau_z_next[row, q] == tableau_z_curr[row, q])

    def add_css_matrix_transition(
        self,
        solver: z3.Solver,
        matrix_curr: npt.NDArray[np.object_],
        matrix_next: npt.NDArray[np.object_],
    ) -> None:
        """SX gate not applicable to CSS encoding."""
        msg = "SX gate cannot be applied in CSS CNOT-only encoding"
        raise NotImplementedError(msg)

    def to_stim_gate(self) -> tuple[str, list[int]]:
        """Convert to Stim gate."""
        return ("SQRT_X", [self.qubit])

    def inverse_stim_gate(self) -> tuple[str, list[int]]:
        """Inverse is SQRT_X_DAG."""
        return ("SQRT_X_DAG", [self.qubit])

    def qubits(self) -> set[int]:
        """Get qubits involved."""
        return {self.qubit}


class CNOTGate(SymbolicGateOperation):
    """CNOT gate operation."""

    IS_TWO_QUBIT: ClassVar[bool] = True
    IS_SYMMETRIC: ClassVar[bool] = False
    IS_SELF_INVERSE: ClassVar[bool] = True

    def __init__(self, control: int, target: int) -> None:
        """Initialize CNOT gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.
        """
        self.control = control
        self.target = target

    @classmethod
    def from_qubits(cls, q1: int, q2: int) -> CNOTGate:
        """Instantiate CNOT gate from qubit indices."""
        return cls(q1, q2)

    def add_clifford_tableau_transition(
        self,
        solver: z3.Solver,
        tableau_x_curr: npt.NDArray[np.object_],
        tableau_z_curr: npt.NDArray[np.object_],
        tableau_x_next: npt.NDArray[np.object_],
        tableau_z_next: npt.NDArray[np.object_],
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
        matrix_curr: npt.NDArray[np.object_],
        matrix_next: npt.NDArray[np.object_],
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


class CZGate(SymbolicGateOperation):
    """CZ gate operation."""

    IS_TWO_QUBIT: ClassVar[bool] = True
    IS_SYMMETRIC: ClassVar[bool] = True
    IS_SELF_INVERSE: ClassVar[bool] = True

    def __init__(self, qubit1: int, qubit2: int) -> None:
        """Initialize CZ gate.

        Args:
            qubit1: First qubit index (canonical: qubit1 < qubit2).
            qubit2: Second qubit index.
        """
        self.qubit1 = qubit1
        self.qubit2 = qubit2

    @classmethod
    def from_qubits(cls, q1: int, q2: int) -> CZGate:
        """Instantiate CZ gate from qubit indices."""
        return cls(q1, q2)

    def add_clifford_tableau_transition(
        self,
        solver: z3.Solver,
        tableau_x_curr: npt.NDArray[np.object_],
        tableau_z_curr: npt.NDArray[np.object_],
        tableau_x_next: npt.NDArray[np.object_],
        tableau_z_next: npt.NDArray[np.object_],
    ) -> None:
        """CZ gate: Z[:,i] ^= X[:,j], Z[:,j] ^= X[:,i], X columns unchanged."""
        num_rows = tableau_x_curr.shape[0]
        i, j = self.qubit1, self.qubit2

        for row in range(num_rows):
            solver.add(tableau_x_next[row, i] == tableau_x_curr[row, i])
            solver.add(tableau_z_next[row, i] == z3.Xor(tableau_z_curr[row, i], tableau_x_curr[row, j]))
            solver.add(tableau_x_next[row, j] == tableau_x_curr[row, j])
            solver.add(tableau_z_next[row, j] == z3.Xor(tableau_z_curr[row, j], tableau_x_curr[row, i]))

    def add_css_matrix_transition(
        self,
        solver: z3.Solver,
        matrix_curr: npt.NDArray[np.object_],
        matrix_next: npt.NDArray[np.object_],
    ) -> None:
        """CZ gate not applicable to CSS encoding."""
        msg = "CZ gate cannot be applied in CSS CNOT-only encoding"
        raise NotImplementedError(msg)

    def to_stim_gate(self) -> tuple[str, list[int]]:
        """Convert to Stim gate."""
        return ("CZ", [self.qubit1, self.qubit2])

    def inverse_stim_gate(self) -> tuple[str, list[int]]:
        """CZ is self-inverse."""
        return ("CZ", [self.qubit1, self.qubit2])

    def qubits(self) -> set[int]:
        """Get qubits involved."""
        return {self.qubit1, self.qubit2}


class IdentityGate(SymbolicGateOperation):
    """Identity (no-op) gate operation."""

    IS_TWO_QUBIT: ClassVar[bool] = False
    IS_SYMMETRIC: ClassVar[bool] = False
    IS_SELF_INVERSE: ClassVar[bool] = True

    def __init__(self, qubit: int) -> None:
        """Initialize identity gate.

        Args:
            qubit: Qubit index (for bookkeeping in depth encoding).
        """
        self.qubit = qubit

    @classmethod
    def from_qubits(cls, q1: int, _q2: int) -> IdentityGate:
        """Instantiate identity gate from qubit indices (_q2 ignored)."""
        return cls(q1)

    def add_clifford_tableau_transition(
        self,
        solver: z3.Solver,
        tableau_x_curr: npt.NDArray[np.object_],
        tableau_z_curr: npt.NDArray[np.object_],
        tableau_x_next: npt.NDArray[np.object_],
        tableau_z_next: npt.NDArray[np.object_],
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
        matrix_curr: npt.NDArray[np.object_],
        matrix_next: npt.NDArray[np.object_],
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

    def create_gate(self, name: str, *args: int, for_css: bool = False) -> SymbolicGateOperation:
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


def get_clifford_sx_gate_set() -> dict[str, type[SymbolicGateOperation]]:
    """Get the Clifford gate set {H, SX, CX, ID}.

    Replaces S with the √X (SX = HSH) gate.

    Returns:
        Dictionary mapping gate names to gate classes.
    """
    return {
        "H": HGate,
        "SX": SqrtXGate,
        "CX": CNOTGate,
        "ID": IdentityGate,
    }


def get_clifford_cz_gate_set() -> dict[str, type[SymbolicGateOperation]]:
    """Get the extended Clifford gate set {H, S, CX, CZ, ID}.

    Returns:
        Dictionary mapping gate names to gate classes.
    """
    return {
        "H": HGate,
        "S": SGate,
        "CX": CNOTGate,
        "CZ": CZGate,
        "ID": IdentityGate,
    }


def get_clifford_extended_gate_set() -> dict[str, type[SymbolicGateOperation]]:
    """Get the full extended Clifford gate set {H, S, SX, CX, CZ, ID}.

    Combines CZ and SX extensions on top of the standard {H, S, CX} basis.

    Returns:
        Dictionary mapping gate names to gate classes.
    """
    return {
        "H": HGate,
        "S": SGate,
        "SX": SqrtXGate,
        "CX": CNOTGate,
        "CZ": CZGate,
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
