# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods for performing Gaussian elimination on GUI2 and symplectic matrices."""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import stim

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from ..codes.pauli import StabilizerTableau

    
@dataclass
class EliminationConfig:
    """Configuration for elimination methods."""
    
    termination_criterion: Callable[[StabilizerTableau], bool]
    sorted_candidate_ops: Callable[[StabilizerTableau], list[tuple[TableauOperation, StabilizerTableau]]]


CheckMatrix = np.ndarray[np.int8]

class TableauOperation(ABC):
    """Represents an operation performed during tableau elimination."""

    @abstractmethod
    def apply(self, tableau: StabilizerTableau) -> StabilizerTableau:
        """Apply the operation to the given stabilizer tableau.

        Args:
            tableau (StabilizerTableau): The stabilizer tableau to apply the operation to.
        """

    @abstractmethod
    def apply_css(self, check_matrix: CheckMatrix) -> CheckMatrix:
        """Apply the operation to a CSS check matrix.

        Args:
            check_matrix (CheckMatrix): The CSS check matrix to apply the operation to.
        """

    @abstractmethod
    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit to append the operation to.
        """
        
        
    def to_stim_circuit(self) -> stim.Circuit:
        """Convert the operation to a Stim circuit representation.

        Returns:
            stim.Circuit: The Stim circuit representing the operation.
        """
        circuit = stim.Circuit()
        self.append_to_circuit(circuit)
        return circuit
        

TV2 = tuple[int, int, int, int]
class Transvection(TableauOperation):
    """Class representing a transvection operation on a stabilizer tableau."""
    
    def __init__(self, v: TV2,  i: int, j: int) -> None:
        """Initialize the transvection operation.

        Args:
            v: A tuple representing the transvection vector (v1, v2, v3, v4).
            i: The index of the first qubit.
            j: The index of the second qubit.
        """
        self.i = i
        self.j = j
        self.v = v

    def apply(self, tableau: StabilizerTableau) -> StabilizerTableau:
        """Apply the transvection operation to the given stabilizer tableau.

        This uses the logic from the `apply_tv2` function to apply a transvection.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
        """
        tab = tableau.tableau
        n = tableau.num_qubits
        i = self.i
        j = self.j
        cols_v = [i, j, i + n, j + n]
        cols_ov = [i + n, j + n, i, j]
        v_bits = np.array(self.v, dtype=np.int8)
        nz = np.flatnonzero(v_bits)  # which of the 4 components are 1

        c = np.zeros((2 * n,), dtype=np.int8)
        for k in nz:
            c ^= tab[:, cols_ov[k]]

        out = tab.copy()
        for k in nz:
            out[:, cols_v[k]] ^= c
        return out

    def apply_css(self, check_matrix: CheckMatrix) -> CheckMatrix:
        """Apply the operation to a CSS check matrix.

        Args:
            check_matrix (CheckMatrix): The CSS check matrix to apply the operation to.
        """
        msg = "Transvection operations are not defined for CSS check matrices."
        raise ValueError(msg)
        return check_matrix


    def all_two_qubit_transvections() -> list[TV2]:
        """Get all 9 possible two-qubit transvections.

        The 9 distinct 2-qubit transvections √(P_i P_j) (P∈{X,Y,Z} non-trivial) correspond to choosing (x,z) in {(1,0),(0,1),(1,1)} for each of the two qubits.
        """
        nontrivial = [(1, 0), (0, 1), (1, 1)]  # X, Z, Y in (x,z)
        out: list[TV2] = []
        for xi, zi in nontrivial:
            for xj, zj in nontrivial:
                out.append((xi, xj, zi, zj))
        return out
    
    @abstractmethod
    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit to append the operation to.
        """
        paulis = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
        i = self.i
        j = self.j
        xi, xj, zi, zj = self.v
        p_i = paulis[xi, zi]
        p_j = paulis[xj, zj]
        if p_i == "I" or p_j == "I":
            msg = f"Expected non-trivial Pauli on both qubits, got {p_i},{p_j}"
            raise ValueError(msg)

        basis_change = {"Z": [], "X": ["H"], "Y": ["S", "H"]}
        # Basis change: map Pi,Pj to Z on each qubit
        for g in basis_change[p_i]:
            circuit.append(g, [i])
        for g in basis_change[p_j]:
            circuit.append(g, [j])

        # Core: √(Z_i Z_j) == CZ(i,j) then S on i and j (up to global phase)
        circuit.append("CZ", [i, j])
        circuit.append("S", [i])
        circuit.append("S", [j])

        # Undo basis change
        for g in basis_change[p_j]:
            circuit.append(g, [j])
        for g in basis_change[p_j]:
            circuit.append(g, [i])


class CNOT(TableauOperation):
    """Class representing a CNOT operation on a stabilizer tableau."""

    def __init__(self, control: int, target: int) -> None:
        """Initialize the CNOT operation.

        Args:
            control: The index of the control qubit.
            target: The index of the target qubit.
        """
        self.control = control
        self.target = target

    def apply(self, tableau: StabilizerTableau) -> StabilizerTableau:
        """Apply the CNOT operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
        """
        tab = tableau.tableau
        c = self.control
        t = self.target

        out = tab.copy()
        x_part = out.get_x_part()
        z_part = out.get_z_part()
        x_part[:, t] ^= x_part[:, c]
        z_part[:, c] ^= z_part[:, t]
        return out

    def apply_css(self, check_matrix: CheckMatrix) -> CheckMatrix:
        """Apply the operation to a CSS check matrix.

        Args:
            check_matrix (CheckMatrix): The CSS check matrix to apply the operation to.
        """
        out = check_matrix.copy()
        out[:, self.target] ^= out[:, self.control]
        return out

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit to append the operation to.
        """
        circuit.append("CNOT", [self.control, self.target])

class EliminationSequence:
    """Class representing a sequence of tableau operations."""
    
    def __init__(self, operations: list[TableauOperation]) -> None:
        """Initialize the elimination sequence.

        Args:
            operations: A list of tableau operations.
        """
        self.operations = operations

    def to_circuit(self) -> stim.Circuit:
        """Convert the elimination sequence to a Stim circuit.

        Returns:
            stim.Circuit: The Stim circuit representing the elimination sequence.
        """
        circuit = stim.Circuit()
        for op in self.operations:
            op.append_to_circuit(circuit)
        return circuit
    

def eliminate(target_tableau: StabilizerTableau, config: EliminationConfig) -> tuple[list[TableauOperation], StabilizerTableau]:
    """Perform Gaussian elimination on the given stabilizer tableau.

    Args:
        target_tableau (StabilizerTableau): The stabilizer tableau to be eliminated.
        config (EliminationConfig): Configuration parameters for the elimination process.

    Returns:
        None: The function modifies the target_tableau in place.
    """
    tableau = target_tableau.copy()
    operations: list[TableauOperation] = []
    is_reduced = config.termination_criterion
    get_candidate_ops = config.sorted_candidate_ops
    
    while not is_reduced(target_tableau):
        candidate_ops = get_candidate_ops(target_tableau)
        if not candidate_ops:
            msg = "No more candidate operations available, but termination criterion not met."
            raise RuntimeError(msg)
        op = candidate_ops[0]
        tableau = op.apply(tableau)
        operations.append(op)
    return operations, tableau

    
def is_identity(tableau: StabilizerTableau) -> bool:
    """Check if the given stabilizer tableau is the identity tableau.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to check.

    Returns:
        bool: True if the tableau is the identity tableau, False otherwise.
    """
    n = tableau.n
    identity_matrix = np.eye(2 * n, dtype=np.int8)
    return np.array_equal(tableau.tableau.matrix, identity_matrix)

def score_symplectic(tableau: StabilizerTableau) -> tuple[tuple[int, ...], int]:
    """Score the given symplectic matrix using the default symplectic heuristic."""
    n = tableau.n
    
    def compute_r2_matrix(symplectic: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    # det(F_ij) = A_xx[i,j]*A_zz[i,j] XOR A_xz[i,j]*A_zx[i,j]
        a_xx = symplectic[:n, :n]
        a_xz = symplectic[:n, n:]
        a_zx = symplectic[n:, :n]
        a_zz = symplectic[n:, n:]
        det = (a_xx & a_zz) ^ (a_xz & a_zx)
        return det.astype(np.int8)


    def compute_r0_matrix(symplectic: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        a_xx = symplectic[:n, :n]
        a_xz = symplectic[:n, n:]
        a_zx = symplectic[n:, :n]
        a_zz = symplectic[n:, n:]
        zero = (a_xx == 0) & (a_xz == 0) & (a_zx == 0) & (a_zz == 0)
        return zero.astype(np.int8)
    def compute_r1_matrix_from_r2_r0(R2: npt.NDArray[np.int8], R0: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        return (1 ^ (R2 | R0)).astype(np.int8)
    
    def r1_r2(symplectic: npt.NDArray[np.int8]) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
        """Compute R1 and R2 matrices."""
        r2 = compute_r2_matrix(symplectic)
        r0 = compute_r0_matrix(symplectic)
        r1 = compute_r1_matrix_from_r2_r0(r2, r0)
        return r1, r2

    symplectic = tableau.tableau.matrix
    r1, r2 = r1_r2(symplectic)

    c1 = r1.sum(axis=0).astype(int)
    c2 = r2.sum(axis=0).astype(int)

    c1t = r1.sum(axis=1).astype(int)  # colSums(R1^T) = rowSums(R1)
    c2t = r2.sum(axis=1).astype(int)
    vec = np.concatenate([n * c2 + c1, n * c2t + c1t])
        # else:
        #     # fallback: approximate real weighting
        #     vec = np.concatenate([c2 + c1 / n, c2t + c1t / n])
    # else:
    #     vec = n * c2 + c1 if params.use_integer_weighting else (c2 + c1 / n)

    h_vec = tuple(sorted(int(x) for x in vec))
    
    h_scalar = int(r1.sum() + r2.sum())
    return h_vec, h_scalar

def get_candidate_transvections(
    tableau: StabilizerTableau,
) -> list[tuple(Transvection, StabilizerTableau)]:
    """Score all possible operations and return the top k scored operations.

    Args:
        tableau: The current symplectic matrix.
        transvections: List of all two-qubit transvections.
        pairs: List of qubit pairs to consider.
        params: Parameters for the greedy synthesis.
        k: Number of top scored operations to return.

    Returns:
        A list of the top k scored operations, each represented as a tuple of
        (operation, heuristic vector and scalar, resulting matrix).
    """
    n = tableau.n
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    transvections = Transvection.all_two_qubit_transvections()
    scores: list[tuple(Transvection, StabilizerTableau, list[int, ...])] = []    
    for i, j in pairs:
        for v in transvections:
            op = Transvection(v, i, j)
            tablea_op_applied = op.apply(tableau)
            h_vec, _ = score_symplectic(tablea_op_applied)
            scores.append(h_vec)
            

    # Sort scored operations by heuristic vector lexicographically
    scores.sort(key=operator.itemgetter(2))
    # Return the top k scored operations
    return [(tv, tab) for tv, tab, _ in scores]
