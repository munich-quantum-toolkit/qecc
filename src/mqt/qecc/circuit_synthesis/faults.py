# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Functionality for handling collections of circuit faults."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ldpc import mod2

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from .circuits import CNOTCircuit


class PureFaultSet:
    """Represents a collection of pure faults (X-type or Z-type) in a quantum circuit."""

    def __init__(self, num_qubits: int) -> None:
        """Initialize a PureFaultSet object.

        Args:
            num_qubits: The number of qubits in the circuit.
        """
        self.num_qubits = num_qubits
        self.faults = np.zeros((0, num_qubits), dtype=np.int8)  # Pure faults as binary vectors

    def add_fault(self, fault: np.ndarray) -> None:
        """Add a fault to the fault set.

        Args:
            fault: A 1D numpy array representing the fault. The array must have length ~num_qubits~.
        """
        fault = np.asarray(fault, dtype=np.int8)
        if fault.shape[0] != self.num_qubits:
            msg = f"Fault must have length {self.num_qubits}."
            raise ValueError(msg)
        self.faults = np.vstack([self.faults, fault])

    def combine(self, other: PureFaultSet) -> PureFaultSet:
        """Combine this fault set with another fault set.

        Args:
            other: Another PureFaultSet to combine with.

        Returns:
            A new PureFaultSet representing the combined faults.
        """
        if self.num_qubits != other.num_qubits:
            msg = "Fault sets must have the same number of qubits to combine."
            raise ValueError(msg)
        combined_faults = np.vstack([self.faults, other.faults])
        return PureFaultSet.from_fault_array(combined_faults)

    def to_array(self) -> npt.NDArray[np.int8]:
        """Convert the fault set to a numpy array.

        Returns:
            A 2D numpy array where each row represents a fault.
        """
        return self.faults

    @classmethod
    def from_fault_array(cls, array: npt.NDArray[np.int8]) -> PureFaultSet:
        """Create a PureFaultSet from a numpy array of faults.

        Returns:
            A PureFaultSet object containing the faults.
        """
        fault_set = cls(array.shape[1])
        fault_set.faults = np.unique(array, axis=0)
        return fault_set

    @classmethod
    def from_cnot_circuit(cls, circ: CNOTCircuit, kind: str = "X") -> PureFaultSet:
        """Generate a PureFaultSet from a CNOT circuit.

        Args:
            circ: The CNOT circuit to generate faults from.
            kind: The type of faults to generate ('X' or 'Z').

        Returns:
            A PureFaultSet containing the faults generated from the circuit.
        """
        assert kind.capitalize() in {"X", "Z"}, "Kind must be either 'X' or 'Z'."
        num_qubits = circ.num_qubits()
        qubit_faults = [[fault] for fault in np.eye(num_qubits, dtype=np.int8)]

        # iterate through circuit in reverse and combine faults
        for control, target in reversed(circ.cnots):
            ctrl, trgt = control, target
            if kind == "Z":
                ctrl, trgt = trgt, ctrl
            new_fault = qubit_faults[ctrl][-1] ^ qubit_faults[trgt][-1]
            qubit_faults[ctrl].append(new_fault)

        # Create the fault set
        return cls.from_fault_array(np.array([fault for faults in qubit_faults for fault in faults], dtype=np.int8))

    def remove_equivalent(self, stabs: npt.NDArray[np.int8]) -> None:
        """Remove faults belonging to the same coset with respect to the stabilizer group."""
        if stabs.shape[1] != self.num_qubits:
            msg = f"Stabilizer matrix must have {self.num_qubits} columns."
            raise ValueError(msg)
        if stabs.ndim != 2:
            msg = "Stabilizer matrix must be 2-dimensional."
            raise ValueError(msg)
        if self.faults.size == 0:
            return
        if stabs.shape[0] == 0:
            # If stabilizer matrix is empty, no faults can be removed
            return

        rref, _, _, pivots = mod2.row_echelon(stabs, full=True)
        # Reduce all faults to their coset representatives
        for i, fault in enumerate(self.faults):
            # Identify the indices of pivot columns where the fault has a 1
            active_pivots = [p for p in pivots if fault[p] == 1]
            if active_pivots:  # Ensure there are active pivots to reduce with
                self.faults[i] = fault ^ np.bitwise_xor.reduce(rref[active_pivots], axis=0)

        # remove all zero rows
        self.faults = self.faults[np.any(self.faults, axis=1)]
        self.faults = np.unique(self.faults, axis=0)

    def to_set(self) -> set[tuple[int, ...]]:
        """Convert the fault set to a set of tuples for easier comparison."""
        return set(map(tuple, self.faults))
