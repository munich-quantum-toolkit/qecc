# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Class for working with representations of Pauli operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .symplectic import SymplecticMatrix, SymplecticVector

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy.typing as npt
    import stim


class Pauli:
    """Class representing an n-qubit Pauli operator."""

    def __init__(self, symplectic: SymplecticVector, phase: int = 0) -> None:
        """Create a new Pauli operator.

        Args:
            symplectic: A 2n x n binary matrix representing the symplectic form of the Pauli operator. The first n entries correspond to X operators, and the second n entries correspond to Z operators.
            phase: An integer 0 or 1 representing the phase of the Pauli operator (0 for +, 1 for -).
        """
        self.n = symplectic.n
        self.symplectic = symplectic
        self.phase = phase

    @classmethod
    def from_pauli_string(cls, p: str) -> Pauli:
        """Create a new Pauli operator from a Pauli string."""
        if not is_pauli_string(p):
            msg = f"Invalid Pauli string: {p}"
            raise InvalidPauliError(msg)
        pauli_start_index = 1 if p[0] in "+-" else 0
        x_part = np.array([c in "XY" for c in p[pauli_start_index:]]).astype(np.int8)
        z_part = np.array([c in "ZY" for c in p[pauli_start_index:]]).astype(np.int8)
        phase = int(p[0] == "-")
        return cls(SymplecticVector(np.concatenate((x_part, z_part))), phase)

    def commute(self, other: Pauli) -> bool:
        """Check if this Pauli operator commutes with another Pauli operator."""
        return self.symplectic @ other.symplectic == 0

    def anticommute(self, other: Pauli) -> bool:
        """Check if this Pauli operator anticommutes with another Pauli operator."""
        return not self.commute(other)

    def __mul__(self, other: Pauli) -> Pauli:
        """Multiply this Pauli operator by another Pauli operator."""
        if self.n != other.n:
            msg = "Pauli operators must have the same number of qubits."
            raise InvalidPauliError(msg)
        return Pauli(self.symplectic + other.symplectic, (self.phase + other.phase) % 2)

    def __repr__(self) -> str:
        """Return a string representation of the Pauli operator."""
        x_part = self.symplectic[: self.n]
        z_part = self.symplectic[self.n :]
        pauli = [
            "X" if x and not z else "Z" if z and not x else "Y" if x and z else "I"
            for x, z in zip(x_part, z_part, strict=False)
        ]
        return f"{'' if self.phase == 0 else '-'}" + "".join(pauli)

    def as_vector(self) -> npt.NDArray[np.int8]:
        """Convert the Pauli operator to a binary vector."""
        return np.concatenate((self.symplectic.vector, np.array([self.phase])))

    def __len__(self) -> int:
        """Return the number of qubits in the Pauli operator."""
        return int(self.n)

    def __getitem__(self, key: int) -> str:
        """Return the Pauli operator for a single qubit."""
        if key < 0 or key >= self.n:
            msg = "Index out of range."
            raise IndexError(msg)
        x = self.symplectic[key]
        z = self.symplectic[key + self.n]
        return "X" if x and not z else "Z" if z and not x else "Y" if x and z else "I"

    def x_part(self) -> npt.NDArray[np.int8]:
        """Return the X part of the Pauli operator."""
        return np.asarray(self.symplectic[: self.n], dtype=np.int8)

    def z_part(self) -> npt.NDArray[np.int8]:
        """Return the Z part of the Pauli operator."""
        return np.asarray(self.symplectic[self.n :], dtype=np.int8)

    def __eq__(self, other: object) -> bool:
        """Check if this Pauli operator is equal to another Pauli operator."""
        if not isinstance(other, Pauli):
            return False
        return self.symplectic == other.symplectic and self.phase == other.phase

    def __ne__(self, other: object) -> bool:
        """Check if this Pauli operator is not equal to another Pauli operator."""
        return not self == other

    def __neg__(self) -> Pauli:
        """Return the negation of this Pauli operator."""
        return Pauli(self.symplectic, 1 - self.phase)

    def __hash__(self) -> int:
        """Return a hash of the Pauli operator."""
        return hash((self.symplectic, self.phase))


class StabilizerTableau:
    """Class representing a stabilizer tableau."""

    def __init__(
        self, tableau: SymplecticMatrix | npt.NDArray[np.int8], phase: npt.NDArray[np.int8] | None = None
    ) -> None:
        """Create a new stabilizer tableau.

        Args:
            tableau: Symplectic matrix representing the stabilizer tableau.
            phase: An n x 1 binary vector representing the phase of the stabilizer tableau.
        """
        if isinstance(tableau, np.ndarray):
            self.tableau = SymplecticMatrix(tableau)
        else:
            self.tableau = tableau
        if phase is None:
            phase = np.zeros((self.tableau.shape[0]), dtype=np.int8)
        if self.tableau.shape[0] != phase.shape[0]:
            msg = "The number of rows in the tableau must match the number of phases."
            raise InvalidPauliError(msg)
        self.n = self.tableau.n
        self.n_rows = self.tableau.shape[0]
        self.phase = phase
        self.shape = (self.n_rows, self.n)

    @classmethod
    def from_stim_circuit(cls, circ: stim.Circuit) -> StabilizerTableau:
        """Create a StabilizerTableau from a stim.Circuit.

        Args:
            circ: A stim.Circuit object.

        Returns:
            A StabilizerTableau instance.
        """
        return cls.from_stim_tableau(circ.to_tableau())

    @classmethod
    def from_stim_tableau(cls, stim_tableau: stim.Tableau) -> StabilizerTableau:
        """Create a StabilizerTableau from a stim.Tableau.

        Args:
            stim_tableau: A stim.Tableau object.

        Returns:
            A StabilizerTableau instance.
        """
        x2x, x2z, z2x, z2z, x_signs, z_signs = stim_tableau.to_numpy(bit_packed=False)

        tableau_matrix = np.block([
            [x2x.astype(np.int8), x2z.astype(np.int8)],
            [z2x.astype(np.int8), z2z.astype(np.int8)],
        ]).astype(np.int8)

        phase = np.concatenate((x_signs.astype(np.int8), z_signs.astype(np.int8)))
        return cls(SymplecticMatrix(tableau_matrix), phase)

    @classmethod
    def from_pauli_strings(cls, pauli_strings: Sequence[str]) -> StabilizerTableau:
        """Create a new stabilizer tableau from a list of Pauli strings."""
        if len(pauli_strings) == 0:
            msg = "At least one Pauli string is required."
            raise InvalidPauliError(msg)

        paulis = [Pauli.from_pauli_string(p) for p in pauli_strings]
        return cls.from_paulis(paulis)

    @classmethod
    def from_paulis(cls, paulis: Sequence[Pauli]) -> StabilizerTableau:
        """Create a new stabilizer tableau from a list of Pauli operators."""
        if len(paulis) == 0:
            msg = "At least one Pauli operator is required."
            raise InvalidPauliError(msg)
        n = paulis[0].n
        if not all(p.n == n for p in paulis):
            msg = "All Pauli operators must have the same number of qubits."
            raise InvalidPauliError(msg)
        mat = SymplecticMatrix.zeros(len(paulis), n)
        phase = np.zeros((len(paulis)), dtype=np.int8)
        for i, p in enumerate(paulis):
            mat[i] = p.symplectic.vector
            phase[i] = p.phase
        return cls(mat, phase)

    @classmethod
    def empty(cls, n: int) -> StabilizerTableau:
        """Create a new empty stabilizer tableau."""
        return cls(SymplecticMatrix.empty(n), np.zeros(0, dtype=np.int8))

    @classmethod
    def identity(cls, n: int) -> StabilizerTableau:
        """Create a new identity stabilizer tableau."""
        return cls(SymplecticMatrix.identity(n), np.zeros(0, dtype=np.int8))

    @classmethod
    def from_matrix(cls, matrix: npt.NDArray[np.int8]) -> StabilizerTableau:
        """Create a StabilizerTableau from a symplectic matrix.

        Args:
            matrix: A 2n x 2n symplectic matrix representing the stabilizer tableau.

        Returns:
            A StabilizerTableau instance.
        """
        if matrix.shape[0] % 2 != 0 or matrix.shape[0] != matrix.shape[1]:
            msg = f"Expected a square 2n×2n symplectic matrix, got shape {matrix.shape}."
            raise ValueError(msg)

        matrix.shape[0] // 2
        symplectic_matrix = SymplecticMatrix(matrix)
        return cls(symplectic_matrix)

    def __eq__(self, other: object) -> bool:
        """Check if two stabilizer tableaus are equal."""
        if not isinstance(other, StabilizerTableau):
            return False
        return bool(self.tableau == other.tableau and np.all(self.phase == other.phase))

    def __ne__(self, other: object) -> bool:
        """Check if two stabilizer tableaus are not equal."""
        return not self == other

    def __len__(self) -> int:
        """Return the number of Paulis in the tableau."""
        return len(self.tableau)

    def all_commute(self, other: StabilizerTableau) -> bool:
        """Check if all Pauli operators in this stabilizer tableau commute with all Pauli operators in another stabilizer tableau."""
        return bool(np.all((self.tableau @ other.tableau).matrix == 0))

    def __getitem__(self, key: int) -> Pauli:
        """Get a Pauli operator from the stabilizer tableau."""
        return Pauli(SymplecticVector(self.tableau[key]), self.phase[key])

    def __hash__(self) -> int:
        """Compute the hash of the stabilizer tableau."""
        return hash((self.tableau, self.phase))

    def __iter__(self) -> Iterator[Pauli]:
        """Iterate over the Pauli operators in the stabilizer tableau."""
        for i in range(self.n_rows):
            yield self[i]

    def as_matrix(self) -> npt.NDArray[np.int8]:
        """Convert the stabilizer tableau to a binary matrix."""
        return np.hstack((self.tableau.matrix, self.phase[..., np.newaxis]))

    def apply_h(self, qubit) -> None:
        """Apply the Hadamard gate to the stabilizer tableau.

        Args:
            qubit: The index of the qubit to apply the Hadamard gate to.
        """
        self.phase += self.tableau[:, qubit] * self.tableau[:, qubit + self.n] % 2
        self.tableau[:, [qubit, qubit + self.n]] = self.tableau[:, [qubit + self.n, qubit]]

    def apply_cx(self, ctrl, tar) -> None:
        """Apply the CNOT gate to the stabilizer tableau.

        Args:
            ctrl: The index of the control qubit.
            tar: The index of the target qubit.
        """
        self.phase = (
            self.phase
            + self.tableau[:, ctrl]
            * self.tableau[:, tar + self.n]
            * (self.tableau[:, tar] + self.tableau[:, ctrl + self.n] * 1)
        ) % 2
        self.tableau[:, tar] = (self.tableau[:, tar] + self.tableau[:, ctrl]) % 2
        self.tableau[:, ctrl + self.n] = (self.tableau[:, tar + self.n] + self.tableau[:, ctrl + self.n]) % 2

    def apply_cz(self, ctrl, tar) -> None:
        """Apply the CZ gate to the stabilizer tableau.

        Args:
            ctrl: The index of the control qubit.
            tar: The index of the target qubit.
        """
        self.apply_h(tar)
        self.apply_cx(ctrl, tar)
        self.apply_h(tar)

    def apply_swap(self, q1, q2) -> None:
        """Apply the SWAP gate to the stabilizer tableau.

        Args:
            q1: The index of the first qubit.
            q2: The index of the second qubit.
        """
        self.apply_cx(q1, q2)
        self.apply_cx(q2, q1)
        self.apply_cx(q1, q2)

    def apply_s(self, qubit) -> None:
        """Apply the S gate to the stabilizer tableau.

        Args:
            qubit: The index of the qubit to apply the S gate to.
        """
        self.phase += self.tableau[:, qubit] * self.tableau[:, qubit + self.n] % 2
        self.tableau[:, qubit + self.n] = (self.tableau[:, qubit + self.n] + self.tableau[:, qubit]) % 2

    def apply_x(self, qubit) -> None:
        """Apply the X gate to the stabilizer tableau."""
        self.phase = (self.phase + self.tableau[:, qubit]) % 2

    def apply_z(self, qubit) -> None:
        """Apply the Z gate to the stabilizer tableau."""
        self.phase = (self.phase + self.tableau[:, qubit + self.n]) % 2

    def apply_y(self, qubit) -> None:
        """Apply the Y gate to the stabilizer tableau."""
        self.phase = self.phase + self.tableau[:, qubit] + self.tableau[:, qubit + self.n]

    def copy(self) -> StabilizerTableau:
        """Return a copy of the stabilizer tableau."""
        return StabilizerTableau(self.tableau.copy(), self.phase.copy())

    def to_pauli_list(self) -> list[Pauli]:
        """Return the tableau as a list of Paulis."""
        return [Pauli(SymplecticVector(self.tableau[i]), self.phase[i]) for i in range(self.n_rows)]

    def to_numpy(self) -> npt.NDArray[np.int8]:
        """Convert the stabilizer tableau to a NumPy array.

        Returns:
            A NumPy array where the first 2n columns represent the symplectic matrix
            and the last column represents the phase vector.
        """
        return np.hstack((self.tableau.matrix, self.phase[:, np.newaxis]))

    def is_css(self) -> bool:
        """Check if the stabilizer tableau is in CSS form."""
        x_part = self.tableau.matrix[:, : self.n]
        z_part = self.tableau.matrix[:, self.n :]
        return np.all(x_part[:, z_part.any(axis=0)] == 0) and np.all(z_part[:, x_part.any(axis=0)] == 0)

    def get_x_part(self) -> npt.NDArray[np.int8]:
        """Get the X part of the stabilizer tableau."""
        return self.tableau.matrix[:, : self.n]

    def get_z_part(self) -> npt.NDArray[np.int8]:
        """Get the Z part of the stabilizer tableau."""
        return self.tableau.matrix[:, self.n :]


def is_pauli_string(p: str) -> bool:
    """Check if a string is a valid Pauli string."""
    return (
        len(p) > 0
        and all(c in {"_", "I", "X", "Y", "Z"} for c in p[1:])
        and p[0] in {"+", "-", "I", "X", "Y", "Z", "_"}
    )


class InvalidPauliError(ValueError):
    """Exception raised when an invalid Pauli operator is encountered."""

    def __init__(self, message: str) -> None:
        """Create a new InvalidPauliError."""
        super().__init__(message)
