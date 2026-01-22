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

import ldpc.mod2.mod2_numpy as mod2
import numpy as np
import stim

from ..codes.pauli import StabilizerTableau

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt


@dataclass
class EliminationConfig:
    """Configuration for elimination methods."""

    termination_criterion: Callable[[StabilizerTableau], bool]
    sorted_candidate_ops: Callable[[StabilizerTableau], list[TableauOperation]]
    filters: list[OperationFilter] = None
    post_process_fn: Callable[
        [tuple[EliminationSequence, StabilizerTableau]], tuple[EliminationSequence, StabilizerTableau]
    ] = lambda x: x


CheckMatrix = np.ndarray[np.int8]


class TableauOperation(ABC):
    """Represents an operation performed during tableau elimination."""

    @abstractmethod
    def apply(self, tableau: StabilizerTableau, inplace: bool = False) -> StabilizerTableau:
        """Apply the operation to the given stabilizer tableau.

        Args:
            tableau (StabilizerTableau): The stabilizer tableau to apply the operation to.
            inplace (bool): If True, modifies the tableau in place. If False, returns a new tableau.

        Args:
            tableau (StabilizerTableau): The stabilizer tableau to apply the operation to.
        """

    @abstractmethod
    def apply_css(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:
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

    @abstractmethod
    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """


TV2 = tuple[int, int, int, int]


class Transvection(TableauOperation):
    """Class representing a transvection operation on a stabilizer tableau."""

    def __init__(self, v: TV2, i: int, j: int) -> None:
        """Initialize the transvection operation.

        Args:
            v: A tuple representing the transvection vector (v1, v2, v3, v4).
            i: The index of the first qubit.
            j: The index of the second qubit.
        """
        self.i = i
        self.j = j
        self.v = v

    def apply(self, tableau: StabilizerTableau, inplace: bool = False) -> StabilizerTableau:
        """Apply the transvection operation to the given stabilizer tableau.

        This uses the logic from the `apply_tv2` function to apply a transvection.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace (bool): If True, modifies the tableau in place. If False, returns a new tableau.
        """
        tab = tableau.tableau
        n = tableau.n
        i = self.i
        j = self.j
        cols_v = [i, j, i + n, j + n]
        cols_ov = [i + n, j + n, i, j]
        v_bits = np.array(self.v, dtype=np.int8)
        nz = np.flatnonzero(v_bits)  # which of the 4 components are 1

        c = np.zeros((2 * n,), dtype=np.int8)
        for k in nz:
            c ^= tab[:, cols_ov[k]]

        out = tab if inplace else tab.copy()

        for k in nz:
            out[:, cols_v[k]] ^= c
        return StabilizerTableau(out)  # TODO: handle phase?

    def apply_css(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:
        """Apply the operation to a CSS check matrix.

        Args:
            check_matrix (CheckMatrix): The CSS check matrix to apply the operation to.
            inplace (bool): If True, modifies the check matrix in place. If False, returns a new check matrix.
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

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """
        return {self.i, self.j}


class SingleQubitClifford(TableauOperation):
    """Class representing a single-qubit Clifford operation on a stabilizer tableau."""

    def __init__(self, qubit: int, clifford: str) -> None:
        """Initialize the single-qubit Clifford operation.

        Args:
            qubit: The index of the qubit.
            clifford: The Clifford operation to apply {H, S, HS, SH, HSH, I}.
        """
        self.qubit = qubit
        self.clifford = clifford

    def apply(self, tableau: StabilizerTableau, inplace: bool = False) -> StabilizerTableau:
        """Apply the single-qubit Clifford operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace (bool): If True, modifies the tableau in place. If False, returns a new tableau.
        """
        q = self.qubit

        out = tableau if inplace else tableau.copy()
        if self.clifford == "H":
            out.apply_h(q)
        elif self.clifford == "S":
            out.apply_s(q)
        elif self.clifford == "HS":
            out.apply_h(q)
            out.apply_s(q)
        elif self.clifford == "SH":
            out.apply_s(q)
            out.apply_h(q)
        elif self.clifford == "HSH":
            out.apply_h(q)
            out.apply_s(q)
            out.apply_h(q)
        elif self.clifford == "I":
            pass
        else:
            msg = f"Unsupported single-qubit Clifford operation: {self.clifford}"
            raise ValueError(msg)
        return out

    def apply_inverse(self, tableau: StabilizerTableau, inplace: bool = False) -> StabilizerTableau:
        """Apply the inverse of the single-qubit Clifford operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace: Whether to modify the tableau in place.

        Returns:
            StabilizerTableau: The resulting stabilizer tableau after applying the inverse operation.
        """
        q = self.qubit

        out = tableau if inplace else tableau.copy()
        if self.clifford == "H":
            out.apply_h(q)
        elif self.clifford == "S":
            out.apply_sdg(q)
        elif self.clifford == "HS":
            out.apply_sdg(q)
            out.apply_h(q)
        elif self.clifford == "SH":
            out.apply_h(q)
            out.apply_sdg(q)
        elif self.clifford == "HSH":
            out.apply_h(q)
            out.apply_sdg(q)
            out.apply_h(q)
        elif self.clifford == "I":
            pass
        else:
            msg = f"Unsupported single-qubit Clifford operation: {self.clifford}"
            raise ValueError(msg)
        return out

    def inverse(self) -> SingleQubitClifford:
        """Get the inverse of the single-qubit Clifford operation.

        Returns:
            SingleQubitClifford: The inverse single-qubit Clifford operation.
        """
        inverse_map = {
            "H": "H",
            "S": "SH",
            "HS": "SH",
            "SH": "HS",
            "HSH": "HSH",
            "I": "I",
        }
        if self.clifford not in inverse_map:
            msg = f"Unsupported single-qubit Clifford operation: {self.clifford}"
            raise ValueError(msg)
        return SingleQubitClifford(self.qubit, inverse_map[self.clifford])

    def available_cliffords() -> list[str]:
        """Get the list of available single-qubit Clifford operations."""
        return ["H", "S", "HS", "SH", "HSH", "I"]

    def apply_css(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:
        """Apply the operation to a CSS check matrix.

        Args:
            check_matrix (CheckMatrix): The CSS check matrix to apply the operation to.
        """
        msg = "Single-qubit Clifford operations are not defined for CSS check matrices."
        raise ValueError(msg)
        return check_matrix

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit to append the operation to.
        """
        circuit.append(self.clifford, [self.qubit])

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """
        return {self.qubit}

    @staticmethod
    def from_symplectic_block(block: npt.NDArray[np.int8], qubit: int) -> SingleQubitClifford:
        """Create a SingleQubitClifford from a symplectic block.

        Args:
            block: A 2x2 symplectic block representing the single-qubit Clifford operation.
            qubit: The index of the qubit.

        Returns:
            A single-qubit Clifford operation.
        """
        for name, mat in elems.items():
            if np.array_equal(block, mat):
                return SingleQubitClifford(qubit, name)
        msg = f"Unsupported single-qubit Clifford symplectic block:\n{block}"
        raise ValueError(msg)


class PauliOperation(TableauOperation):
    """Class representing a Pauli operation on a stabilizer tableau."""

    def __init__(self, qubit: int, pauli: str) -> None:
        """Initialize the Pauli operation.

        Args:
            qubit: The index of the qubit.
            pauli: The Pauli operation to apply {X, Y, Z}.
        """
        self.qubit = qubit
        self.pauli = pauli

    def apply(self, tableau: StabilizerTableau, inplace: bool = False) -> StabilizerTableau:
        """Apply the Pauli operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace (bool): If True, modifies the tableau in place. If False, returns a new tableau.
        """
        out = tableau if inplace else tableau.copy()
        if self.pauli == "X":
            out.apply_x(self.qubit)
        elif self.pauli == "Y":
            out.apply_y(self.qubit)
        elif self.pauli == "Z":
            out.apply_z(self.qubit)
        else:
            msg = f"Unsupported Pauli operation: {self.pauli}"
            raise ValueError(msg)
        return out

    def apply_css(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:
        """Apply the operation to a CSS check matrix.

        Args:
            check_matrix (CheckMatrix): The CSS check matrix to apply the operation to.
        """
        msg = "Pauli operations are not defined for CSS check matrices."
        raise ValueError(msg)
        return check_matrix

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit to append the operation to.
        """
        circuit.append(self.pauli, [self.qubit])

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """
        return {self.qubit}


def _matmul2(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return ((m1 @ m2) % 2).astype(np.int8)


identity = np.array([[1, 0], [0, 1]], dtype=np.int8)
hadamard = np.array([[0, 1], [1, 0]], dtype=np.int8)
phase = np.array([[1, 1], [0, 1]], dtype=np.int8)

# For right-multiplication, sequence "HS" means multiply by H then S => I*H*S
elems: dict[str, np.ndarray] = {
    "": identity,
    "H": hadamard,
    "S": phase,
    "HS": _matmul2(hadamard, phase),
    "SH": _matmul2(phase, hadamard),
    "HSH": _matmul2(_matmul2(hadamard, phase), hadamard),
}


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

    def apply(self, tableau: StabilizerTableau, inplace: bool = False) -> StabilizerTableau:
        """Apply the CNOT operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace (bool): If True, modifies the tableau in place. If False, returns a new tableau.
        """
        out = tableau if inplace else tableau.copy()
        out.apply_cx(self.control, self.target)
        return out

    def apply_css(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:
        """Apply the operation to a CSS check matrix.

        Args:
            check_matrix (CheckMatrix): The CSS check matrix to apply the operation to.
        """
        out = check_matrix if inplace else check_matrix.copy()
        out[:, self.target] ^= out[:, self.control]
        return out

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit to append the operation to.
        """
        circuit.append("CNOT", [self.control, self.target])

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """
        return {self.control, self.target}


class Swap(TableauOperation):
    """Class representing a SWAP operation on a stabilizer tableau."""

    def __init__(self, qubit_a: int, qubit_b: int) -> None:
        """Initialize the SWAP operation.

        Args:
            qubit_a: The index of the first qubit.
            qubit_b: The index of the second qubit.
        """
        self.qubit_a = qubit_a
        self.qubit_b = qubit_b

    def apply(self, tableau: StabilizerTableau, inplace: bool = False) -> StabilizerTableau:
        """Apply the SWAP operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace (bool): If True, modifies the tableau in place. If False, returns a new tableau.
        """
        out = tableau if inplace else tableau.copy()
        out.apply_swap(self.qubit_a, self.qubit_b)
        return out

    def apply_css(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:
        """Apply the operation to a CSS check matrix.

        Args:
            check_matrix (CheckMatrix): The CSS check matrix to apply the operation to.
        """
        out = check_matrix.copy()
        out[:, [self.qubit_a, self.qubit_b]] = out[:, [self.qubit_b, self.qubit_a]]
        return out

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit (stim.Circuit): The Stim circuit to append the operation to.
        """
        circuit.append("SWAP", [self.qubit_a, self.qubit_b])

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """
        return {self.qubit_a, self.qubit_b}


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


class OperationFilter(ABC):
    """Abstract base class for filtering tableau operations."""

    @abstractmethod
    def filter(self, operations: list[TableauOperation]) -> list[TableauOperation]:
        """Filter the given list of tableau operations.

        Args:
            operations: A list of tableau operations to filter.
            config: Configuration parameters for the elimination process.

        Returns:
            A filtered list of tableau operations.
        """

    @abstractmethod
    def update(self, op: TableauOperation) -> None:
        """Update the filter state with the given operation.

        Args:
            op: The tableau operation to update the filter with.
        """


class ParallelFilter(OperationFilter):
    """Context for elimination process."""

    def __init__(self) -> None:
        """Initialize the elimination context.

        Args:
            config: Configuration parameters for the elimination process.
        """
        self.blocked_qubits: set[int] = set()

    def filter(self, operations: list[TableauOperation]) -> list[TableauOperation]:
        """Filter the given list of tableau operations.

        Args:
            operations: A list of tableau operations to filter.

        Returns:
            A filtered list of tableau operations.
        """
        filtered_ops: list[TableauOperation] = [
            op for op in operations if not any(qubit in self.blocked_qubits for qubit in op.qubits())
        ]
        if not filtered_ops:
            self._reset()
            filtered_ops = operations
        return filtered_ops

    def update(self, op: TableauOperation) -> None:
        """Update the filter with the given operation.

        Args:
            op: The tableau operation to update the context with.
        """
        qubits_involved = op.qubits()
        self.blocked_qubits.update(qubits_involved)

    def _reset(self) -> None:
        """Unblock all qubits."""
        self.blocked_qubits.clear()


def eliminate(
    target_tableau: StabilizerTableau, config: EliminationConfig
) -> tuple[list[TableauOperation], StabilizerTableau]:
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

    while not is_reduced(tableau):
        candidate_ops: EliminationSequence = get_candidate_ops(tableau)
        filtered_ops = filter_candidates(candidate_ops, config)  # [:top_k]

        if not filtered_ops:
            msg = "No more candidate operations available, but termination criterion not met."
            raise RuntimeError(msg)

        op = filtered_ops[0]
        tableau = op.apply(tableau)
        operations.append(op)

        update_state(op, config)

    post_process_fn = config.post_process_fn
    operations, tableau = post_process_fn(operations, tableau)
    return operations, tableau


def update_state(op: TableauOperation, config: EliminationConfig) -> None:
    """Update the elimination state with the given operation.

    Args:
        op (TableauOperation): The tableau operation to update the state with.
        config (EliminationConfig): Configuration parameters for the elimination process.
    """
    if config.filters:
        for filter_ in config.filters:
            filter_.update(op)


def filter_candidates(ops: EliminationSequence, config: EliminationConfig) -> EliminationSequence:
    """Filter candidate operations using the filters defined in the elimination configuration.

    Args:
        ops (EliminationSequence): The list of candidate tableau operations.
        config (EliminationConfig): Configuration parameters for the elimination process.

    Returns:
        A filtered list of candidate tableau operations.
    """
    filtered_ops = ops
    for filter_ in config.filters:
        filtered_ops = filter_.filter(filtered_ops)
    return filtered_ops


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


def _compute_r2_matrix(symplectic: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    # det(F_ij) = A_xx[i,j]*A_zz[i,j] XOR A_xz[i,j]*A_zx[i,j]
    n = symplectic.shape[0] // 2
    a_xx = symplectic[:n, :n]
    a_xz = symplectic[:n, n:]
    a_zx = symplectic[n:, :n]
    a_zz = symplectic[n:, n:]
    det = (a_xx & a_zz) ^ (a_xz & a_zx)
    return det.astype(np.int8)


def _compute_r0_matrix(symplectic: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    n = symplectic.shape[0] // 2
    a_xx = symplectic[:n, :n]
    a_xz = symplectic[:n, n:]
    a_zx = symplectic[n:, :n]
    a_zz = symplectic[n:, n:]
    zero = (a_xx == 0) & (a_xz == 0) & (a_zx == 0) & (a_zz == 0)
    return zero.astype(np.int8)


def _compute_r1_matrix_from_r2_r0(R2: npt.NDArray[np.int8], R0: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    return (1 ^ (R2 | R0)).astype(np.int8)


def r1_r2(symplectic: npt.NDArray[np.int8]) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    """Compute R1 and R2 matrices."""
    r2 = _compute_r2_matrix(symplectic)
    r0 = _compute_r0_matrix(symplectic)
    r1 = _compute_r1_matrix_from_r2_r0(r2, r0)
    return r1, r2


def is_terminal_transvection(tableau: StabilizerTableau) -> bool:
    """Check if the given stabilizer tableau is in terminal form for transvection elimination.

    Args:
        tableau (StabilizerTableau): The stabilizer tableau to check.

    Returns:
        bool: True if the tableau is in terminal form, False otherwise.
    """
    r1, r2 = r1_r2(tableau.tableau.matrix)
    if np.any(r1):
        return False
    # permutation matrix: each row/col has exactly one 1
    if not np.all(r2.sum(axis=0) == 1):
        return False
    return np.all(r2.sum(axis=1) == 1)


def score_symplectic(tableau: StabilizerTableau) -> tuple[tuple[int, ...], int]:
    """Score the given symplectic matrix using the default symplectic heuristic."""
    n = tableau.n

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
) -> list[Transvection]:
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
    scores: list[tuple(Transvection, list[int, ...])] = []
    for i, j in pairs:
        for v in transvections:
            op = Transvection(v, i, j)
            tablea_op_applied = op.apply(tableau)
            h_vec, _ = score_symplectic(tablea_op_applied)
            scores.append((op, h_vec))

    scores.sort(key=operator.itemgetter(1))
    return [tv for tv, _ in scores]


def reduce_single_qubit_gates_and_swaps(
    tableau: StabilizerTableau,
) -> tuple[EliminationSequence, StabilizerTableau]:
    """Reduce a TERMINAL symplectic matrix U to identity using only SWAP/H/S by right-multiplication.

    Returns:
      ((swaps, one_qubit_ops), U_reduced)
    where U_reduced should be identity.
    """
    tableau_copy = tableau.copy()
    n = tableau.n

    perm, _blocks = _extract_perm_in_to_out_and_blocks(tableau_copy)

    # 2) Right-multiply by permutation inverse to bring blocks onto the diagonal
    inv = _perm_inverse(perm)
    swaps = _perm_to_swaps(inv)  # realize inv permutation
    for swap in swaps:
        tableau_copy = swap.apply(tableau_copy, inplace=True)

    # 3) Right-multiply by single-qubit Cliffords to bring each block to identity
    single_qubit_ops = []
    for q in range(n):
        f = tableau.symplectic_submatrix(q)
        op = SingleQubitClifford.from_symplectic_block(f, q)
        single_qubit_ops.append(op)
        tableau_copy = op.apply(tableau_copy, inplace=True)

    single_qubit_ops.extend(fix_tableau_signs_in_place(tableau_copy))
    return (swaps, single_qubit_ops), tableau_copy


def _extract_perm_in_to_out_and_blocks(tableau: StabilizerTableau) -> tuple[EliminationSequence, StabilizerTableau]:
    """Extract the permutation and corresponding 2×2 blocks from a terminal symplectic matrix.

    This function processes a terminal symplectic matrix `U` to determine the permutation
    of input qubits to output qubits and the associated 2×2 symplectic blocks.

    Args:
        U: A 2n×2n symplectic matrix in terminal form.

    Returns:
        A tuple containing:
        - perm: A 1D array where `perm[i]` gives the index `j` such that the determinant
          of the 2×2 block F_ij is 1 (indicating a valid symplectic transformation).
        - blocks: A list of 2×2 symplectic blocks corresponding to the permutation.
    """
    n = tableau.n
    symplectic = tableau.tableau.matrix
    r2 = _compute_r2_matrix(symplectic)

    perm = np.full(n, -1, dtype=int)
    blocks: list[np.ndarray] = [None] * n  # type: ignore

    for i in range(n):
        js = np.flatnonzero(r2[i])
        if len(js) != 1:
            msg = "Not terminal: R2 row is not one-hot."
            raise ValueError(msg)
        j = int(js[0])
        perm[i] = j
        blocks[i] = np.array(
            [
                [int(symplectic[i, j]), int(symplectic[i, j + n])],
                [int(symplectic[i + n, j]), int(symplectic[i + n, j + n])],
            ],
            dtype=np.int8,
        )

    if len(set(perm.tolist())) != n:
        msg = "Not terminal: R2 columns not one-hot."
        raise ValueError(msg)
    return perm, blocks


def _perm_inverse(perm_in_to_out: np.ndarray) -> np.ndarray:
    n = len(perm_in_to_out)
    inv = np.empty(n, dtype=int)
    for i, j in enumerate(perm_in_to_out):
        inv[int(j)] = i
    return inv


def _perm_to_swaps(perm_in_to_out: np.ndarray) -> list[Swap]:
    """Return a SWAP list that realizes perm_in_to_out when right-multiplying the symplectic matrix, i.e. permuting columns (wires). (Any decomposition is fine for the test.)."""
    perm = perm_in_to_out.copy().tolist()
    n = len(perm)
    swaps: list[Swap] = []
    pos = list(range(n))  # current label at position p

    # We want to permute columns so that wire i ends up at perm[i].
    # Do it via bubble-like swapping on positions.
    for i in range(n):
        target_pos = perm[i]
        cur_pos = pos.index(i)
        while cur_pos != target_pos:
            step = cur_pos + 1 if cur_pos < target_pos else cur_pos - 1
            swaps.append(Swap(cur_pos, step))
            # swap labels in pos
            pos[cur_pos], pos[step] = pos[step], pos[cur_pos]
            cur_pos = step

    # This is not minimal, but deterministic and fine for testing.
    # You can replace with cycle decomposition later.
    return swaps


def fix_tableau_signs_in_place(tableau: StabilizerTableau) -> EliminationSequence:
    """Determine Pauli corrections so that the tableau matches the desired sign bits.

    This function ensures that the tableau matches the target signs
    by appending the necessary Pauli corrections.
    """
    n = tableau.n
    # Extract the current signs from the tableau
    x_part = tableau.tableau.matrix[:, :n]
    z_part = tableau.tableau.matrix[:, n:]

    # Compute the corrections needed to match the target signs
    phase = tableau.phase.copy()

    if not np.any(phase):
        return []  # No corrections needed

    tableau_with_phase = np.hstack((x_part, z_part, np.array([phase]).T))
    ker = mod2.nullspace(tableau_with_phase)
    assert ker[-1, -1] == 1, "Last entry of kernel vector must be 1."
    correction_symplectic = ker[-1]
    zc = correction_symplectic[:n]
    xc = correction_symplectic[n:-1]
    ops = []
    for i, (xv, zv) in enumerate(zip(xc, zc, strict=False)):
        if xv == 1 and zv == 1:
            op = PauliOperation(i, "Y")
        elif xv == 1:
            op = PauliOperation(i, "X")
        elif zv == 1:
            op = PauliOperation(i, "Z")
        else:
            continue  # don't explicitly apply identity
        op.apply(tableau, inplace=True)

    return ops
