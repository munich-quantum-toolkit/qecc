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
import z3
from ldpc import mod2

from .synthesis_utils import symbolic_vector_add, symbolic_vector_eq, vars_to_stab

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
    def from_cnot_circuit(cls, circ: CNOTCircuit, kind: str = "X", reduce: bool = False) -> PureFaultSet:
        """Generate a PureFaultSet from a CNOT circuit.

        Args:
            circ: The CNOT circuit to generate faults from.
            kind: The type of faults to generate ('X' or 'Z').
            reduce: Reduce faults by stabilizers induced by the circuit.

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
        fs = cls.from_fault_array(np.array([fault for faults in qubit_faults for fault in faults], dtype=np.int8))
        if not reduce:
            return fs

        code = circ.get_code()
        stabs = code.Hx if kind == "X" else code.Hz

        fs.remove_equivalent(stabs)
        return fs

    def normalize(self, stabs: npt.NDArray[np.int8]) -> None:
        """Normalize the faults with respect to a stabilizer group.

        A fault is considered normalized if its entries in the pivot columns of the RREF of the stabilizer matrix are zero.

        Args:
            stabs: A 2D numpy array where each row is a stabilizer generator.
        """
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

    def remove_equivalent(self, stabs: npt.NDArray[np.int8]) -> None:
        """Remove faults belonging to the same coset with respect to the stabilizer group.

        Args:
            stabs: A 2D numpy array where each row is a stabilizer generator.
        """
        self.normalize(stabs)

        # remove all zero rows
        self.faults = self.faults[np.any(self.faults, axis=1)]
        self.faults = np.unique(self.faults, axis=0)

    def to_set(self) -> set[tuple[int, ...]]:
        """Convert the fault set to a set of tuples for easier comparison."""
        return set(map(tuple, self.faults))

    def faults_to_coset_leaders(self, generators: npt.NDArray[np.int8]) -> None:
        """Map all faults in the set to their coset leaders with respect to the stabilizer generators.

        This method modifies the fault set in place, replacing each fault with its coset leader.
        Warning: This might take a while.

        Args:
            generators: A 2D numpy array where each row is a stabilizer generator.
        """
        if generators.ndim != 2 or generators.shape[1] != self.num_qubits:
            msg = f"Generators must be a 2D array with {self.num_qubits} columns."
            raise ValueError(msg)

        self.faults = np.array([coset_leader(fault, generators) for fault in self.faults], dtype=np.int8)
        self.faults = np.unique(self.faults, axis=0)

    def filter_by_weight(self, w: int, stabs: npt.NDArray[np.int8]) -> None:
        """Filter faults by weight with respect to a stabilizer group.

        A fault is removed if its coset leader has weight lower than w.
        This operation also removes stabilizer equivalent errors and maps faults to their coset leaders.

        Args:
            w: Weight faults are filtered by.
            stabs: A 2D numpy array where each row is a stabilizer generator.
        """
        self.remove_equivalent(stabs)
        self.faults_to_coset_leaders(stabs)

        # filter remaining faults by weight
        weights = np.sum(self.faults, axis=1)
        mask = weights >= w
        self.faults = self.faults[mask]

    def __eq__(self, other: object) -> bool:
        """Check equality of two PureFaultSet objects.

        Two PureFaultSet objects are considered equal if they have the same number of qubits
        and contain the same faults. This check does not factor in stabilizer equivalence or coset leaders.

        Args:
            other: Another PureFaultSet object to compare with.

        Returns:
            True if both PureFaultSet objects are equal, False otherwise.
        """
        if not isinstance(other, PureFaultSet):
            return False
        return self.num_qubits == other.num_qubits and self.to_set() == other.to_set()

    def __hash__(self) -> int:
        """Return a hash of the PureFaultSet.

        Returns:
            An integer hash value.
        """
        return hash((self.num_qubits, tuple(map(tuple, self.faults))))

    def copy(self) -> PureFaultSet:
        """Create a copy of the PureFaultSet.

        Returns:
            A new PureFaultSet object with the same faults and number of qubits.
        """
        new_set = PureFaultSet(self.num_qubits)
        new_set.faults = np.copy(self.faults)
        return new_set

    def __repr__(self) -> str:
        """Return a string representation of the PureFaultSet."""
        return f"PureFaultSet(num_qubits={self.num_qubits}, faults={self.faults.tolist()})"


def coset_leader(fault: npt.NDArray[np.int8], generators: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    """Compute the coset leader of a fault given a set of stabilizer generators."""
    if len(generators) == 0:
        return fault
    s = z3.Optimize()
    leader = [z3.Bool(f"e_{i}") for i in range(len(fault))]
    coeff = [z3.Bool(f"c_{i}") for i in range(len(generators))]

    g = vars_to_stab(coeff, generators)

    s.add(symbolic_vector_eq(np.array(leader), symbolic_vector_add(fault.astype(bool), g)))
    s.minimize(z3.Sum(leader))

    s.check()  # always SAT
    m = s.model()
    return np.array([bool(m[leader[i]]) for i in range(len(fault))]).astype(int)


def fault_set_product(lhs: PureFaultSet, rhs: PureFaultSet) -> PureFaultSet:
    """Generate fault set by forming the product of all faults of two fault sets.

    Args:
        lhs: The first fault set.
        rhs: The second fault set.

    Returns:
        Fault set containing all products of faults of lhs and rhs.
    """
    if lhs.num_qubits != rhs.num_qubits:
        msg = "Fault sets must have the same number of qubits to combine."
        raise ValueError(msg)
    new_faults = (lhs.faults[:, np.newaxis, :] ^ rhs.faults).reshape(-1, lhs.num_qubits)
    return PureFaultSet.from_fault_array(new_faults)


def stabilizer_equivalent(lhs: PureFaultSet, rhs: PureFaultSet, stabs: npt.NDArray[np.int8] | None) -> bool:
    """Check if two fault sets are equivalent with respect to a stabilizer group.

    Args:
            lhs: The first fault set.
            rhs: The second fault set.
            stabs (optional): A 2D numpy array where each row is a stabilizer generator.

    Returns:
            True if the two fault sets are equivalent with respect to the stabilizer group, False otherwise.
    """
    if lhs.num_qubits != rhs.num_qubits:
        msg = "Fault sets must have the same number of qubits to compare."
        raise ValueError(msg)

    lhs_cpy = lhs.copy()
    rhs_cpy = rhs.copy()
    if stabs is not None:
        lhs_cpy.normalize(stabs)
        rhs_cpy.normalize(stabs)

    return lhs_cpy == rhs_cpy
