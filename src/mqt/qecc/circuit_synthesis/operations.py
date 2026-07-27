# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tableau operations used during elimination."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numba as nb
import numpy as np
import stim

from ..codes.core.pauli import PauliTableau

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..codes.core.pauli import CheckMatrix
    from .types import BinaryMatrix


class TableauOperation(ABC):
    """Represents an operation performed during tableau elimination."""

    def apply(self, tableau: BinaryMatrix, inplace: bool = False) -> BinaryMatrix:
        """Apply the operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace: If True, modifies the tableau in place. If False, returns a new tableau.
        """
        if hasattr(tableau, "is_x_type"):  # check with duck typing faster than isinstance
            return self.apply_check_matrix(tableau, inplace=inplace)  # ty: ignore[invalid-argument-type]
        return self.apply_stabilizer_tableau(tableau, inplace=inplace)

    @abstractmethod
    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit: The Stim circuit to append the operation to.
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

    def __repr__(self) -> str:
        """Return a string representation of the operation."""
        return f"{self.__class__.__name__}(qubits={self.qubits()})"

    @abstractmethod
    def apply_stabilizer_tableau(self, tableau: PauliTableau, inplace: bool = False) -> PauliTableau:
        """Apply the operation to a stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace: If True, modifies the tableau in place. If False, returns a new tableau.

        Returns:
            PauliTableau: The resulting stabilizer tableau after applying the operation.
        """

    @abstractmethod
    def apply_check_matrix(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:
        """Apply the operation to a CSS check matrix.

        Args:
            check_matrix: The CSS check matrix to apply the operation to.
            inplace: If True, modifies the check matrix in place. If False, returns a new check matrix.

        Returns:
            CheckMatrix: The resulting CSS check matrix after applying the operation.
        """

    @abstractmethod
    def __hash__(self) -> int:
        """Return a hash of the operation."""


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

    def apply_stabilizer_tableau_inplace(self, tableau: PauliTableau) -> None:
        """Apply the transvection operation to a stabilizer tableau."""
        n = tableau.n
        mat = tableau.tableau.data

        signs = tableau.signs()
        _apply_transvection_numba(
            mat[:, self.i], mat[:, self.i + n], mat[:, self.j], mat[:, self.j + n], signs, *self.v
        )
        tableau.phase = PauliTableau.phase_from_signs(mat, signs)

    def apply_stabilizer_tableau(self, tableau: PauliTableau, inplace: bool = False) -> PauliTableau:
        """Apply the transvection operation to a stabilizer tableau."""
        out = tableau if inplace else tableau.copy()

        n = out.n
        mat = out.tableau.data

        signs = out.signs()
        _apply_transvection_numba(
            mat[:, self.i], mat[:, self.i + n], mat[:, self.j], mat[:, self.j + n], signs, *self.v
        )
        out.phase = PauliTableau.phase_from_signs(mat, signs)

        return out

    def apply_check_matrix(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:
        """Apply the transvection operation to a CSS check matrix.

        Args:
            check_matrix: The CSS check matrix to apply the operation to.
            inplace: If True, modifies the check matrix in place. If False, returns a new check matrix.

        Returns:
            CheckMatrix: The resulting CSS check matrix after applying the operation.
        """
        msg = "Transvection operations are not implemented for CheckMatrix instances."
        raise NotImplementedError(msg)

    @staticmethod
    def all_two_qubit_transvections() -> list[TV2]:
        """Get all 9 possible two-qubit transvections.

        The 9 distinct 2-qubit transvections √(P_i P_j) (P∈{X,Y,Z} non-trivial)
        correspond to choosing (x,z) in {(1,0),(0,1),(1,1)} for each of the two qubits.

        Returns:
            List of all 9 transvection vectors as tuples (xi, xj, zi, zj).
        """
        nontrivial = [(1, 0), (0, 1), (1, 1)]
        out: list[TV2] = []
        for xi, zi in nontrivial:
            for xj, zj in nontrivial:
                out.append((xi, xj, zi, zj))
        return out

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit: The Stim circuit to append the operation to.
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

        basis_change = {"Z": [], "X": ["H"], "Y": ["S_DAG", "H"]}
        undo_basis_change = {"Z": [], "X": ["H"], "Y": ["H", "S"]}
        for g in basis_change[p_i]:
            circuit.append(g, [i])
        for g in basis_change[p_j]:
            circuit.append(g, [j])

        circuit.append("CZ", [i, j])
        circuit.append("S", [i])
        circuit.append("S", [j])

        for g in undo_basis_change[p_j]:
            circuit.append(g, [j])
        for g in undo_basis_change[p_i]:
            circuit.append(g, [i])

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """
        return {self.i, self.j}

    def __hash__(self) -> int:
        """Return a hash of the operation."""
        return hash((self.__class__, self.v, self.i, self.j))


def _matmul2(m1: npt.NDArray[np.int8], m2: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    return ((m1 @ m2) % 2).astype(np.int8)


identity = np.array([[1, 0], [0, 1]], dtype=np.int8)
hadamard = np.array([[0, 1], [1, 0]], dtype=np.int8)
phase = np.array([[1, 1], [0, 1]], dtype=np.int8)

elems: dict[str, npt.NDArray[np.int8]] = {
    "I": identity,
    "H": hadamard,
    "S": phase,
    "SH": _matmul2(hadamard, phase),
    "HS": _matmul2(phase, hadamard),
    "HSH": _matmul2(_matmul2(hadamard, phase), hadamard),
}


class SingleQubitClifford(TableauOperation):
    """Class representing a single-qubit Clifford operation on a stabilizer tableau."""

    def __init__(self, qubit: int, clifford: str) -> None:
        """Initialize the single-qubit Clifford operation.

        Args:
            qubit: The index of the qubit.
            clifford: The Clifford operation to apply {H, S, SDAG, HS, SH, HSH, SDAGH, HSDAG, HSDAGH, I}.
        """
        self.qubit = qubit
        self.clifford = clifford

    def apply_stabilizer_tableau(self, tableau: PauliTableau, inplace: bool = False) -> PauliTableau:
        """Apply the single-qubit Clifford operation to a stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace: If True, modifies the tableau in place. If False, returns a new tableau.

        Returns:
            PauliTableau: The resulting stabilizer tableau after applying the operation.
        """
        q = self.qubit

        out = tableau if inplace else tableau.copy()
        if self.clifford == "H":
            out.apply_h(q)
        elif self.clifford == "S":
            out.apply_s(q)
        elif self.clifford == "SDAG":
            out.apply_sdg(q)
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
        elif self.clifford == "SDAGH":
            out.apply_sdg(q)
            out.apply_h(q)
        elif self.clifford == "HSDAG":
            out.apply_h(q)
            out.apply_sdg(q)
        elif self.clifford == "HSDAGH":
            out.apply_h(q)
            out.apply_sdg(q)
            out.apply_h(q)
        elif self.clifford == "I":
            pass
        else:
            msg = f"Unsupported single-qubit Clifford operation: {self.clifford}"
            raise ValueError(msg)
        return out

    def __hash__(self) -> int:
        """Return a hash of the operation."""
        return hash((self.__class__, self.qubit, self.clifford))

    def apply_check_matrix(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:
        """Apply the single-qubit Clifford operation to a CSS check matrix.

        Args:
            check_matrix: The CSS check matrix to apply the operation to.
            inplace: If True, modifies the check matrix in place. If False, returns a new check matrix.

        Returns:
            CheckMatrix: The resulting CSS check matrix after applying the operation.
        """
        msg = "SingleQubitClifford operations are not implemented for CheckMatrix instances."
        raise NotImplementedError(msg)

    def apply_inverse(self, tableau: BinaryMatrix, inplace: bool = False) -> BinaryMatrix:
        """Apply the inverse of the single-qubit Clifford operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace: Whether to modify the tableau in place.

        Returns:
            BinaryMatrix: The resulting stabilizer tableau after applying the inverse operation.
        """
        if not isinstance(tableau, PauliTableau):
            msg = "SingleQubitClifford operations can only be applied to PauliTableau instances."
            raise TypeError(msg)
        q = self.qubit

        out = tableau if inplace else tableau.copy()
        if self.clifford == "H":
            out.apply_h(q)
        elif self.clifford == "S":
            out.apply_sdg(q)
        elif self.clifford == "SDAG":
            out.apply_s(q)
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
        elif self.clifford == "SDAGH":
            out.apply_h(q)
            out.apply_s(q)
        elif self.clifford == "HSDAG":
            out.apply_s(q)
            out.apply_h(q)
        elif self.clifford == "HSDAGH":
            out.apply_h(q)
            out.apply_s(q)
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
            "S": "SDAG",
            "SDAG": "S",
            "HS": "SDAGH",
            "SH": "HSDAG",
            "HSH": "HSDAGH",
            "SDAGH": "HS",
            "HSDAG": "SH",
            "HSDAGH": "HSH",
            "I": "I",
        }
        if self.clifford not in inverse_map:
            msg = f"Unsupported single-qubit Clifford operation: {self.clifford}"
            raise ValueError(msg)
        return SingleQubitClifford(self.qubit, inverse_map[self.clifford])

    @staticmethod
    def available_cliffords() -> list[str]:
        """Get the list of available single-qubit Clifford operations.

        Returns:
            List of Clifford operation names: H, S, SDAG, HS, SH, HSH, SDAGH, HSDAG, HSDAGH, I.
        """
        return ["H", "S", "SDAG", "HS", "SH", "HSH", "SDAGH", "HSDAG", "HSDAGH", "I"]

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit: The Stim circuit to append the operation to.
        """
        if self.clifford in {"H", "S", "I"}:
            circuit.append(self.clifford, [self.qubit])
        elif self.clifford == "SDAG":
            circuit.append("S_DAG", [self.qubit])
        elif self.clifford == "HS":
            circuit.append("H", [self.qubit])
            circuit.append("S", [self.qubit])
        elif self.clifford == "SH":
            circuit.append("S", [self.qubit])
            circuit.append("H", [self.qubit])
        elif self.clifford == "HSH":
            circuit.append("H", [self.qubit])
            circuit.append("S", [self.qubit])
            circuit.append("H", [self.qubit])
        elif self.clifford == "SDAGH":
            circuit.append("S_DAG", [self.qubit])
            circuit.append("H", [self.qubit])
        elif self.clifford == "HSDAG":
            circuit.append("H", [self.qubit])
            circuit.append("S_DAG", [self.qubit])
        elif self.clifford == "HSDAGH":
            circuit.append("H", [self.qubit])
            circuit.append("S_DAG", [self.qubit])
            circuit.append("H", [self.qubit])
        else:
            msg = f"Unsupported single-qubit Clifford operation: {self.clifford}"
            raise ValueError(msg)

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

    def apply_check_matrix(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:  # ruff:ignore[no-self-use]
        """Apply the Pauli operation to a CSS check matrix.

        Args:
            check_matrix: The CSS check matrix to apply the operation to.
            inplace: If True, modifies the check matrix in place. If False, returns a new check matrix.

        Returns:
            CheckMatrix: The resulting CSS check matrix after applying the operation.
        """
        return check_matrix if inplace else check_matrix.copy()  # Pauli operations do not change the check matrix

    def apply_stabilizer_tableau(self, tableau: PauliTableau, inplace: bool = False) -> PauliTableau:
        """Apply the Pauli operation to a stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace: If True, modifies the tableau in place. If False, returns a new tableau.

        Returns:
            PauliTableau: The resulting stabilizer tableau after applying the operation.
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

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit: The Stim circuit to append the operation to.
        """
        circuit.append(self.pauli, [self.qubit])

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """
        return {self.qubit}

    def __hash__(self) -> int:
        """Return a hash of the operation."""
        return hash((self.__class__, self.qubit, self.pauli))


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

    def apply(self, tableau: BinaryMatrix, inplace: bool = False) -> BinaryMatrix:
        """Apply the CNOT operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace: If True, modifies the tablau in place. If False, returns a new tableau.
        """
        if hasattr(tableau, "is_x_type"):  # check with duck typing faster than isinstance
            return self.apply_check_matrix(tableau, inplace=inplace)  # ty: ignore[invalid-argument-type]
        return self.apply_stabilizer_tableau(tableau, inplace=inplace)

    def apply_stabilizer_tableau(self, tableau: PauliTableau, inplace: bool = False) -> PauliTableau:
        """Apply the CNOT operation to a stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace: If True, modifies the tableau in place. If False, returns a new tableau.

        Returns:
            PauliTableau: The resulting stabilizer tableau after applying the operation.
        """
        out = tableau if inplace else tableau.copy()
        out.apply_cx(self.control, self.target)
        return out

    def apply_check_matrix(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:
        """Apply the operation to a CSS check matrix.

        Args:
            check_matrix: The CSS check matrix to apply the operation to.
            inplace: If True, modifies the check matrix in place. If False, returns a new check matrix.

        Returns:
            The resulting CSS check matrix after applying the operation.
        """
        out = check_matrix if inplace else check_matrix.copy()
        out.matrix[:, self.target] ^= out.matrix[:, self.control]
        return out

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit: The Stim circuit to append the operation to.
        """
        circuit.append("CNOT", [self.control, self.target])

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            set[int]: The set of qubit indices involved in the operation.
        """
        return {self.control, self.target}

    def __hash__(self) -> int:
        """Return a hash of the operation."""
        return hash((self.__class__, self.control, self.target))


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

    def apply(self, tableau: BinaryMatrix, inplace: bool = False) -> BinaryMatrix:
        """Apply the SWAP operation to the given stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace: If True, modifies the tableau in place. If False, returns a new tableau.
        """
        if hasattr(tableau, "is_x_type"):  # check with duck typing faster than isinstance
            return self.apply_check_matrix(tableau, inplace=inplace)  # ty: ignore[invalid-argument-type]
        return self.apply_stabilizer_tableau(tableau, inplace=inplace)

    def apply_stabilizer_tableau(self, tableau: PauliTableau, inplace: bool = False) -> PauliTableau:
        """Apply the SWAP operation to a stabilizer tableau.

        Args:
            tableau: The stabilizer tableau to apply the operation to.
            inplace: If True, modifies the tableau in place. If False, returns a new tableau.

        Returns:
            PauliTableau: The resulting stabilizer tableau after applying the operation.
        """
        out = tableau if inplace else tableau.copy()
        out.apply_swap(self.qubit_a, self.qubit_b)
        return out

    def apply_check_matrix(self, check_matrix: CheckMatrix, inplace: bool = False) -> CheckMatrix:
        """Apply the operation to a CSS check matrix.

        Args:
            check_matrix: The CSS check matrix to apply the operation to.
            inplace: If True, modifies the check matrix in place. If False, returns a new check matrix.
        """
        out = check_matrix if inplace else check_matrix.copy()
        out.matrix[:, [self.qubit_a, self.qubit_b]] = out.matrix[:, [self.qubit_b, self.qubit_a]]
        return out

    def append_to_circuit(self, circuit: stim.Circuit) -> None:
        """Append the operation to a Stim circuit.

        Args:
            circuit: The Stim circuit to append the operation to.
        """
        circuit.append("SWAP", [self.qubit_a, self.qubit_b])

    def qubits(self) -> set[int]:
        """Get the set of qubits involved in the operation.

        Returns:
            The set of qubit indices involved in the operation.
        """
        return {self.qubit_a, self.qubit_b}

    def __hash__(self) -> int:
        """Return a hash of the operation."""
        return hash((self.__class__, frozenset({self.qubit_a, self.qubit_b})))


@nb.jit(
    [
        nb.void(
            nb.int8[:],
            nb.int8[:],
            nb.int8[:],
            nb.int8[:],
            nb.int8[:],
            nb.int32,
            nb.int32,
            nb.int32,
            nb.int32,
        ),
        nb.void(
            nb.int64[:],
            nb.int64[:],
            nb.int64[:],
            nb.int64[:],
            nb.int8[:],
            nb.int32,
            nb.int32,
            nb.int32,
            nb.int32,
        ),
        nb.void(
            nb.int32[:],
            nb.int32[:],
            nb.int32[:],
            nb.int32[:],
            nb.int8[:],
            nb.int64,
            nb.int64,
            nb.int64,
            nb.int64,
        ),
    ],
    nopython=True,
    cache=True,
)  # type: ignore[untyped-decorator]
def _apply_transvection_numba(
    mat_i: np.ndarray,
    mat_i_n: np.ndarray,
    mat_j: np.ndarray,
    mat_j_n: np.ndarray,
    phase: np.ndarray,
    xi: int,
    xj: int,
    zi: int,
    zj: int,
) -> None:
    p_i = xi + 2 * zi
    p_j = xj + 2 * zj

    if p_i == 0 or p_j == 0:
        return

    basis_i = 1 if p_i == 1 else (2 if p_i == 3 else 0)
    basis_j = 1 if p_j == 1 else (2 if p_j == 3 else 0)
    undo_i = 1 if p_i == 1 else (3 if p_i == 3 else 0)
    undo_j = 1 if p_j == 1 else (3 if p_j == 3 else 0)

    if basis_i == 1:
        phase ^= mat_i * mat_i_n
        temp = mat_i.copy()
        mat_i[:] = mat_i_n
        mat_i_n[:] = temp
    elif basis_i == 2:
        mat_i_n ^= mat_i
        temp = mat_i.copy()
        mat_i[:] = mat_i_n
        mat_i_n[:] = temp

    if basis_j == 1:
        phase ^= mat_j * mat_j_n
        temp = mat_j.copy()
        mat_j[:] = mat_j_n
        mat_j_n[:] = temp
    elif basis_j == 2:
        mat_j_n ^= mat_j
        temp = mat_j.copy()
        mat_j[:] = mat_j_n
        mat_j_n[:] = temp

    phase ^= (mat_j * mat_j_n) ^ (mat_i * mat_i_n) ^ (mat_i * mat_j * (mat_j_n ^ mat_i_n))

    mat_i_n ^= mat_j ^ mat_i
    mat_j_n ^= mat_i ^ mat_j

    if undo_j == 1:
        phase ^= mat_j * mat_j_n
        temp = mat_j.copy()
        mat_j[:] = mat_j_n
        mat_j_n[:] = temp
    elif undo_j == 3:
        temp = mat_j.copy()
        mat_j[:] = mat_j_n
        mat_j_n[:] = temp
        mat_j_n ^= mat_j

    if undo_i == 1:
        phase ^= mat_i * mat_i_n
        temp = mat_i.copy()
        mat_i[:] = mat_i_n
        mat_i_n[:] = temp
    elif undo_i == 3:
        temp = mat_i.copy()
        mat_i[:] = mat_i_n
        mat_i_n[:] = temp
        mat_i_n ^= mat_i
    return
