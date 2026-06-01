# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
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

from mqt.qecc.mod2 import row_echelon

from .synthesis_utils import symbolic_vector_add, symbolic_vector_eq, vars_to_stab

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterator

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

    def add_fault(self, fault: npt.NDArray[np.int8]) -> None:
        """Add a fault to the fault set.

        Args:
            fault: A 1D numpy array representing the fault. The array must have length ~num_qubits~.
        """
        fault = np.asarray(fault, dtype=np.int8)
        if fault.shape[0] != self.num_qubits:
            msg = f"Fault must have length {self.num_qubits}."
            raise ValueError(msg)
        self.faults = np.vstack([self.faults, fault])

    def add_faults(self, faults: npt.NDArray[np.int8]) -> None:
        """Add multiple faults to the fault set.

        Args:
            faults: A 2D numpy array representing a collection of faults.
        """
        self.faults = np.vstack((self.faults, faults))

    def combine(self, other: PureFaultSet, inplace: bool = False) -> PureFaultSet:
        """Combine this fault set with another fault set.

        Args:
            other: Another PureFaultSet to combine with.
            inplace: If True, modifies self.

        Returns:
            A new PureFaultSet representing the combined faults.
        """
        if self.num_qubits != other.num_qubits:
            msg = "Fault sets must have the same number of qubits to combine."
            raise ValueError(msg)
        combined_faults = np.vstack([self.faults, other.faults])

        if inplace:
            self.faults = combined_faults
            return self
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
        if array.ndim != 2:
            msg = "Input array must be 2-dimensional."
            raise ValueError(msg)
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

        rref, _, _, pivots = row_echelon(stabs, full=True)
        # Reduce all faults to their coset representatives
        for i, fault in enumerate(self.faults):
            # Identify the indices of pivot columns where the fault has a 1
            active_pivots = [pivots.index(p) for p in pivots if fault[p] == 1]
            if active_pivots:  # Ensure there are active pivots to reduce with
                self.faults[i] = fault ^ np.bitwise_xor.reduce(rref[active_pivots], axis=0)

    def remove_zero_rows(self) -> None:
        """Remove all zero rows from the fault set.

        This method modifies the fault set in place, removing any rows that are entirely zero.
        """
        self.faults = self.faults[np.any(self.faults, axis=1)]

    def remove_duplicates(self) -> None:
        """Remove duplicate faults from the fault set.

        This method modifies the fault set in place, ensuring that each fault is unique.
        """
        self.faults = np.unique(self.faults, axis=0)

    def remove_equivalent(self, stabs: npt.NDArray[np.int8]) -> None:
        """Remove faults belonging to the same coset with respect to the stabilizer group.

        Args:
            stabs: A 2D numpy array where each row is a stabilizer generator.
        """
        self.normalize(stabs)

        # remove all zero rows
        self.remove_zero_rows()
        self.remove_duplicates()

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

    def filter_by_weight_at_least(self, w: int, stabs: npt.NDArray[np.int8]) -> None:
        """Filter faults by weight with respect to a stabilizer group.

        A fault is removed if its coset leader has weight lower than w.
        This operation also removes stabilizer equivalent errors and maps faults to their coset leaders.

        Args:
            w: Weight faults are filtered by.
            stabs: A 2D numpy array where each row is a stabilizer generator.
        """
        self.remove_equivalent(stabs)
        self.faults_to_coset_leaders(stabs)

        if len(self.faults) == 0:
            return
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

    def __len__(self) -> int:
        """Return the number of faults in the PureFaultSet.

        Returns:
            The number of faults.
        """
        return int(self.faults.shape[0])

    def __getitem__(self, index: int) -> npt.NDArray[np.int8]:
        """Get a fault by index.

        Args:
            index: The index of the fault to retrieve.

        Returns:
            A 1D numpy array representing the fault.
        """
        return np.asarray(self.faults[index], dtype=np.int8)

    def __iter__(self) -> Iterator[npt.NDArray[np.int8]]:
        """Return an iterator over the faults in the PureFaultSet.

        Returns:
            An iterator over the faults.
        """
        return iter(self.faults)

    def all_faults_detected(self, stabs: npt.NDArray[np.int8]) -> bool:
        """Check whether all faults in the set are detected by the given stabilizers.

        Args:
            stabs: A 2D numpy array where each row is a stabilizer generator.

        Returns:
            True if every fault anticommutes with at least one generator, False otherwise
        """
        return bool(np.all(np.any(stabs @ self.faults.T % 2, axis=1)))

    def get_undetectable_faults_idx(self, stabs: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Return indices of faults that are not detectable by the given stabilizers.

        Args:
            stabs: A 2D numpy array where each row is a stabilizer generator.

        Returns:
            Indices of faults that commute with all generators.
        """
        return np.where(np.all(stabs @ self.faults.T % 2 == 0, axis=0))[0].astype(np.int8)

    def get_undetectable_faults(self, stabs: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Return faults that are not detectable by the given stabilizers.

        Args:
            stabs: A 2D numpy array where each row is a stabilizer generator.

        Returns:
            A 2D numpy array where each row is a fault that commutes with all generators.
        """
        return self.faults[self.get_undetectable_faults_idx(stabs)]

    def remove_undetectable_faults(self, stabs: npt.NDArray[np.int8]) -> None:
        """Remove all faults that are not detectable by the given stabilizers.

        Args:
            stabs: A 2D numpy array where each row is a stabilizer generator.
        """
        undetectable_indices = self.get_undetectable_faults_idx(stabs)
        self.faults = np.delete(self.faults, undetectable_indices, axis=0)

    def filter_faults(self, pred: Callable[[npt.NDArray[np.int8]], bool], inplace: bool = True) -> PureFaultSet:
        """Filter faults by removing faults for which the given predicate is False.

        This method modifies the fault set in place, removing faults that do not satisfy the predicate.

        Args:
            pred: A callable that takes a fault (1D numpy array) and returns True if the fault should be kept.
            inplace: If True, modifies the current fault set. If False, returns a new PureFaultSet with filtered faults.
        """
        filtered = np.array([fault for fault in self.faults if pred(fault)], dtype=np.int8)
        if filtered.size == 0:
            filtered = np.zeros((0, self.num_qubits), dtype=np.int8)

        if inplace:
            self.faults = filtered
            return self

        return PureFaultSet.from_fault_array(filtered)

    def permute_qubits(self, permutation: npt.NDArray[np.int8] | list[int], inplace: bool = True) -> PureFaultSet:
        """Permute the qubits in the fault set according to a given permutation.

        Args:
            permutation: A 1D numpy array or list representing the new order of qubits.
            inplace: If True, modifies the current fault set. If False, returns a new PureFaultSet with permuted faults.

        Returns:
            A new PureFaultSet with permuted faults if inplace is False.
        """
        if len(permutation) != self.num_qubits:
            msg = f"Permutation must have length {self.num_qubits}."
            raise ValueError(msg)

        permuted_faults = self.faults[:, permutation]
        if inplace:
            self.faults = permuted_faults
            return self

        return PureFaultSet.from_fault_array(permuted_faults)


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


def product_fault_set(lhs: PureFaultSet, rhs: PureFaultSet) -> PureFaultSet:
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

    lhs_copy = lhs.copy()
    rhs_copy = rhs.copy()
    if stabs is not None:
        lhs_copy.normalize(stabs)
        rhs_copy.normalize(stabs)

    return lhs_copy == rhs_copy


def t_distinct(fs1: PureFaultSet, fs2: PureFaultSet, t: int, stabs: npt.NDArray[np.int8] | None = None) -> bool:
    """Check if two fault sets are t-distinct.

    Two fault sets are t-distinct if there is no product of at most i faults from fs1 that is equivalent to a product of at most j faults in fs2 sucht that i+j<=t and the weight of either product is greater than i+j. If stabilizers are given the minimal weight is computed with respect to the stabilizer group generated by stabs.

    Args:
        fs1: The first fault set.
        fs2: The second fault set.
        t: The maximum number of faults to consider in the product.
        stabs: The stabilizer generators used to determine the minimal weight of a fault.

    Returns:
        True if the fault sets are t-distinct, False otherwise.
    """
    for i in range(1, t + 1):
        for j in range(1, t + 1 - i):
            fs1_prodc_vars = [
                z3.Bool(f"fs1_{i}") for i in range(len(fs1.faults))
            ]  # symbolic variables indicating if a fault is in the product
            fs2_prodc_vars = [z3.Bool(f"fs2_{i}") for i in range(len(fs2.faults))]
            p1 = vars_to_stab(fs1_prodc_vars, fs1.faults)
            p2 = vars_to_stab(fs2_prodc_vars, fs2.faults)
            s = z3.Solver()
            s.add(
                symbolic_vector_eq(p1, p2)  # check if the products are equivalent
            )
            s.add(z3.PbLe([(e, 1) for e in fs1_prodc_vars], i))
            s.add(
                z3.Or(fs1_prodc_vars)  # at least one fault from fs1 must be in the product
            )
            s.add(z3.PbLe([(e, 1) for e in fs2_prodc_vars], j))
            s.add(
                z3.Or(fs2_prodc_vars)  # at least one fault from fs2 must be in the product
            )
            # minimal weight of vector i greater than i+j
            if stabs is not None:
                stab_vars = [z3.Bool(f"stab_{k}") for k in range(stabs.shape[0])]
                stab_vec = vars_to_stab(stab_vars, stabs)
                coset = symbolic_vector_add(stab_vec, p1)
                # for all assignments to stab_vars, the coset element must have weight greater i+j
                s.add(z3.ForAll(stab_vars, z3.PbGe([(v, 1) for v in coset], i + j + 1)))

            if s.check() == z3.sat:
                # if the solver finds a solution, the fault sets are not t-distinct
                return False
    # if no solution was found, the fault sets are t-distinct
    return True


class XZFaultList:
    def __init__(self, num_qubits: int) -> None:
        """Initialise a XZFaultList object.

        Args:
            num_qubits (int): The number of qubits in the circuit
        """
        self.num_qubits = num_qubits
        self.faults = {
            "X": np.zeros((0, num_qubits), dtype=np.int8),
            "Z": np.zeros((0, num_qubits), dtype=np.int8),
        }

    def add_fault(self, faults: tuple[npt.NDArray[np.int8] | None, npt.NDArray[np.int8] | None]) -> None:
        """Add a single fault pair (X error, Z error) to the fault list.

        Args:
            faults: A tuple of (x_fault, z_fault) where each is a 1D numpy array.
                Each array must have length num_qubits.
                One of the faults may be set to None, which is treated as an all-zero fault.

        Raises:
            ValueError: If fault arrays don't have the correct length.
            ValueError: If both faults are None
        """
        assert len(faults) == 2, "Faults should be a tuple of x_fault and z_fault"

        x_fault, z_fault = faults
        if x_fault is None and z_fault is None:
            msg = "At least one fault must be provided."
            raise ValueError(msg)

        if x_fault is None:
            z_fault = np.asarray(z_fault, dtype=np.int8)
            x_fault = np.zeros(self.num_qubits, dtype=np.int8)
        elif z_fault is None:
            x_fault = np.asarray(x_fault, dtype=np.int8)
            z_fault = np.zeros(self.num_qubits, dtype=np.int8)
        else:
            x_fault = np.asarray(x_fault, dtype=np.int8)
            z_fault = np.asarray(z_fault, dtype=np.int8)

        if x_fault.shape[0] != self.num_qubits or z_fault.shape[0] != self.num_qubits:
            msg = f"Faults must have length {self.num_qubits}."
            raise ValueError(msg)

        self.faults["X"] = np.vstack([self.faults["X"], x_fault])
        self.faults["Z"] = np.vstack([self.faults["Z"], z_fault])

    def add_faults(self, faults: tuple[npt.NDArray[np.int8] | None, npt.NDArray[np.int8] | None]) -> None:
        """Add multiple fault pairs to the fault list.

        Args:
            faults: A tuple of (x_faults, z_faults) where each is a 2D numpy array.
                Each array should have num_qubits columns.
                One of the faults may be set to None, which is treated as an all-zero fault.

        Raises:
            ValueError: If fault arrays don't have the correct shape.
            ValueError: If both fault arrays are None
        """
        x_faults, z_faults = faults
        if x_faults is None and z_faults is None:
            msg = "At least one fault array must be provided."
            raise ValueError(msg)

        if x_faults is None:
            z_faults = np.asarray(z_faults, dtype=np.int8)
            x_faults = np.zeros_like(z_faults, dtype=np.int8)
        elif z_faults is None:
            x_faults = np.asarray(x_faults, dtype=np.int8)
            z_faults = np.zeros_like(x_faults, dtype=np.int8)
        else:
            x_faults = np.asarray(x_faults, dtype=np.int8)
            z_faults = np.asarray(z_faults, dtype=np.int8)

        if x_faults.shape[1] != self.num_qubits or z_faults.shape[1] != self.num_qubits:
            msg = f"Faults must have {self.num_qubits} columns."
            raise ValueError(msg)

        self.faults["X"] = np.vstack([self.faults["X"], x_faults])
        self.faults["Z"] = np.vstack([self.faults["Z"], z_faults])

    def copy(self) -> XZFaultList:
        """Create a copy of the XZFaultList.

        Returns:
            A new XZFaultList object with copied fault arrays.
        """
        new_list = XZFaultList(self.num_qubits)
        new_list.faults["X"] = np.copy(self.faults["X"])
        new_list.faults["Z"] = np.copy(self.faults["Z"])
        return new_list

    def __iter__(self):
        """Iterate over fault pairs in the list.

        Yields:
            Tuples of (x_fault, z_fault) for each row in the fault arrays.
        """
        for i in range(len(self.faults["X"])):
            yield (self.faults["X"][i], self.faults["Z"][i])

    def apply_cnot(self, control: int, target: int, inplace: bool = True) -> XZFaultList:
        """Apply a CNOT gate to the faults in the list.

        For X-type faults: target qubit is affected by control qubit (target ^= control).
        For Z-type faults: control qubit is affected by target qubit (control ^= target).

        Args:
            control: The index of the control qubit.
            target: The index of the target qubit.
            inplace: If True, modifies the current fault list. If False, returns a new XZFaultList.

        Returns:
            A new XZFaultList with updated faults if inplace is False, otherwise self.

        Raises:
            ValueError: If control or target indices are out of range or equal.
        """
        self.ensure_apply_valid_input(control, target)

        if inplace:
            # Apply CNOT directly to self.faults
            x_faults, z_faults = self.faults["X"], self.faults["Z"]
            ret = self
        else:
            # Create a new XZFaultList with copied faults
            new_list = XZFaultList(self.num_qubits)
            new_list.faults["X"] = np.copy(self.faults["X"])
            new_list.faults["Z"] = np.copy(self.faults["Z"])

            x_faults, z_faults = new_list.faults["X"], new_list.faults["Z"]
            ret = new_list

        # Apply CNOT
        x_faults[:, target] ^= x_faults[:, control]
        z_faults[:, control] ^= z_faults[:, target]

        return ret

    def apply_hadamard(self, qubit: int, inplace: bool = True) -> XZFaultList:
        """Apply a Hadamard gate to the faults in the list.

        A Hadamard gate swaps X and Z errors on the specified qubit.

        Args:
            qubit: The index of the qubit.
            inplace: If True, modifies the current fault list. If False, returns a new XZFaultList.

        Returns:
            A new XZFaultList with updated faults if inplace is False, otherwise self.

        Raises:
            ValueError: If qubit index is out of range.
        """
        self.ensure_apply_valid_input(qubit)

        if inplace:
            # Atomic swap using tuple assignment; use copies on RHS to avoid overlap
            self.faults["X"][:, qubit], self.faults["Z"][:, qubit] = (
                self.faults["Z"][:, qubit].copy(),
                self.faults["X"][:, qubit].copy(),
            )
            return self

        # Create a new XZFaultList with copied and swapped faults
        new_list = XZFaultList(self.num_qubits)
        new_list.faults["X"] = np.copy(self.faults["X"])
        new_list.faults["Z"] = np.copy(self.faults["Z"])

        # Atomic swap on the copies
        new_list.faults["X"][:, qubit], new_list.faults["Z"][:, qubit] = (
            new_list.faults["Z"][:, qubit].copy(),
            new_list.faults["X"][:, qubit].copy(),
        )

        return new_list

    def apply_reset(self, qubit: int, inplace: bool = True) -> XZFaultList:
        """Apply a reset operation to the faults in the list.

        A reset removes any accumulated X and Z errors on the specified qubit.

        Args:
            qubit: The index of the qubit.
            inplace: If True, modifies the current fault list. If False, returns a new XZFaultList.

        Returns:
            A new XZFaultList with updated faults if inplace is False, otherwise self.

        Raises:
            ValueError: If qubit index is out of range.
        """
        self.ensure_apply_valid_input(qubit)

        if inplace:
            self.faults["X"][:, qubit] = 0
            self.faults["Z"][:, qubit] = 0
            return self

        new_list = XZFaultList(self.num_qubits)
        new_list.faults["X"] = np.copy(self.faults["X"])
        new_list.faults["Z"] = np.copy(self.faults["Z"])
        new_list.faults["X"][:, qubit] = 0
        new_list.faults["Z"][:, qubit] = 0
        return new_list

    def apply_ccz(self, control1: int, control2: int, control3: int, inplace: bool = True) -> XZFaultList:
        """Apply a CCZ gate to the faults in the list.

        The propagation model is adversarial: any pair of X faults on two controls
        will induce a Z fault on the third control.
        We can do this also because the given circuit is assumed to be fault tolerant.

        Note: CCZ is symmetrical, thus there is no "target" per se

        Args:
            control1: The first control qubit.
            control2: The second control qubit.
            control3: The third control qubit.
            inplace: If True, modifies the current fault list. If False, returns a new XZFaultList.

        Returns:
            A new XZFaultList with updated faults if inplace is False, otherwise self.

        Raises:
            ValueError: If any control index is out of range.
            ValueError: If any control qubits are not distinct.
        """
        # Z faults just get propagated through
        # Only X faults are problematic

        # By right, the state of the qubits matter, which is why you can't simply propagate pauli gates through a CCZ gate.

        # Adversarial Fault Propagation for CCZ:
        # We do a simple logic, that every pair of X faults leads, in the worst case, to a Z fault on the other control. So we can just add all pairs of X faults as Z faults.
        # Z_i ^= (X_j & X_k) for all distinct i, j, k in {control1, control2, control3}

        self.ensure_apply_valid_input(control1, control2, control3)

        if inplace:
            x_faults, z_faults = self.faults["X"], self.faults["Z"]
            ret = self
        else:
            new_list = XZFaultList(self.num_qubits)
            new_list.faults["X"] = np.copy(self.faults["X"])
            new_list.faults["Z"] = np.copy(self.faults["Z"])
            x_faults, z_faults = new_list.faults["X"], new_list.faults["Z"]
            ret = new_list

        z_faults[:, control1] ^= x_faults[:, control2] & x_faults[:, control3]
        z_faults[:, control2] ^= x_faults[:, control1] & x_faults[:, control3]
        z_faults[:, control3] ^= x_faults[:, control1] & x_faults[:, control2]

        return ret

    def apply_ccx(self, control1: int, control2: int, target: int, inplace: bool = True) -> XZFaultList:
        """Apply a CCX (Toffoli) gate to the faults in the list, by applying a H_target x CCZ x H_target.

        Args:
            control1: The first control qubit.
            control2: The second control qubit.
            target: The target qubit.
            inplace: If True, modifies the current fault list. If False, returns a new XZFaultList.

        Returns:
            A new XZFaultList with updated faults if inplace is False, otherwise self.

        Raises:
            ValueError: If any qubit index is out of range.
            ValueError: If qubits are not distinct.
        """
        self.ensure_apply_valid_input(control1, control2, target)

        fault_list = self.apply_hadamard(target, inplace=inplace)
        fault_list.apply_ccz(control1, control2, target)
        fault_list.apply_hadamard(target)

        return fault_list

    def ensure_apply_valid_input(self, *qubits: int) -> bool:
        """Ensures that the input into apply_* functions are valid.

        Raises:
            ValueError: If any qubit index is out of range.
            ValueError: If qubits are not distinct.

        Returns:
            bool: True if everything is okay
        """
        n_q = len(qubits)
        if any(not 0 <= q < self.num_qubits for q in qubits):
            msg = f"Qubit {'indices' if n_q > 1 else 'index'} must be between 0 and {self.num_qubits - 1}."
            raise ValueError(msg)
        if n_q > 1 and len(set(qubits)) != n_q:
            msg = "All qubits must be different."
            raise ValueError(msg)

        return True

    def reduce_to_coset_leaders(
        self, generators: tuple[npt.NDArray[np.int8] | None, npt.NDArray[np.int8] | None], inplace: bool = True
    ) -> XZFaultList:
        """Reduce fault list to coset leaders using provided generators.

        Applies coset leader reduction to X and Z type faults independently using the
        corresponding generators. This is useful for reducing error syndromes to their
        canonical representatives in quantum error correction.

        Args:
            generators (Tuple[npt.NDArray[np.int8] | None, npt.NDArray[np.int8] | None]):
                Tuple of (x_generators, z_generators). Each should be a 2D numpy array with
                shape (num_generators, num_qubits) or None to skip reduction for that error type.
            inplace (bool, optional): If True, modify this fault list in place.
                If False, return a copy with reductions applied. Defaults to True.

        Raises:
            ValueError: If any generator array has incorrect dimensions (must be 2D with num_qubits columns).
            AssertionError: If generators tuple length is not 2.

        Returns:
            XZFaultList: The reduced fault list (self if inplace=True, otherwise a copy).
        """
        # Setting the corresponding generator to None means no reduction is done

        assert len(generators) == 2, "Generators should be a tuple of x_generators and z_generators"

        # use qecc_faults.coset_leader(single_fault, generators) for x and z
        ret = self if inplace else self.copy()

        for error_type, g in zip(ret.faults, generators, strict=False):
            # Ensure generators are numpy arrays (may be empty)
            g = None if g is None else np.asarray(g, dtype=np.int8)

            # Check sizes
            if g is not None and (g.ndim != 2 or g.shape[1] != self.num_qubits):
                msg = f"Generators must be a 2D array with {self.num_qubits} columns."
                raise ValueError(msg)

            if ret.faults[error_type].shape[0] > 0 and g is not None and g.size > 0:
                for i in range(ret.faults[error_type].shape[0]):
                    ret.faults[error_type][i] = np.asarray(coset_leader(ret.faults[error_type][i], g), dtype=np.int8)

        return ret

    def __repr__(self) -> str:
        repr_ = [object.__repr__(self), "X:", repr(self.faults["X"]), "Z:", repr(self.faults["Z"])]
        return "\n".join(repr_)
