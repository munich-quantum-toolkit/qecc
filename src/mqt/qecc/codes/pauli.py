# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Class for working with representations of Pauli operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ldpc.mod2.mod2_numpy as mod2
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

    @classmethod
    def from_stim(cls, p: stim.PauliString) -> Pauli:
        """Create a new Pauli operator from Stim representation."""
        return cls.from_pauli_string(str(p))

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
        self.n_rows = int(self.tableau.shape[0])
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
        return cls(SymplecticMatrix.identity(n), np.zeros(2 * n, dtype=np.int8))

    @classmethod
    def from_matrix(cls, matrix: npt.NDArray[np.int8]) -> StabilizerTableau:
        """Create a StabilizerTableau from a symplectic matrix.

        Args:
            matrix: A 2n x 2n symplectic matrix representing the stabilizer tableau.

        Returns:
            A StabilizerTableau instance.
        """
        if matrix.shape[0] % 2 != 0 or matrix.shape[0] != matrix.shape[1]:
            msg = f"Expected a square 2nx2n symplectic matrix, got shape {matrix.shape}."
            raise ValueError(msg)

        matrix.shape[0] // 2
        symplectic_matrix = SymplecticMatrix(matrix)
        return cls(symplectic_matrix)

    @classmethod
    def from_check_matrix(cls, check_matrix: CheckMatrix) -> StabilizerTableau:
        """Create a StabilizerTableau from a CSS check matrix.

        Args:
            check_matrix: A CheckMatrix object representing the CSS check matrix.

        Returns:
            A StabilizerTableau instance.
        """
        if check_matrix.is_x_type():
            x_part = check_matrix.matrix
            z_part = np.zeros_like(x_part)
        elif check_matrix.is_z_type():
            z_part = check_matrix.matrix
            x_part = np.zeros_like(z_part)
        else:
            msg = "Check matrix must be of type 'X' or 'Z'."
            raise ValueError(msg)

        symplectic_matrix = SymplecticMatrix(np.hstack((x_part, z_part)))
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
        return hash((self.tableau, self.phase.tobytes()))

    def __iter__(self) -> Iterator[Pauli]:
        """Iterate over the Pauli operators in the stabilizer tableau."""
        for i in range(self.n_rows):
            yield self[i]

    def as_matrix(self) -> npt.NDArray[np.int8]:
        """Convert the stabilizer tableau to a binary matrix."""
        return np.hstack((self.tableau.matrix, self.phase[..., np.newaxis]))

    def apply_h(self, qubit: int) -> None:
        """Apply the Hadamard gate to the stabilizer tableau.

        Args:
            qubit: The index of the qubit to apply the Hadamard gate to.
        """
        self.phase ^= self.tableau[:, qubit] * self.tableau[:, qubit + self.n]
        self.tableau[:, [qubit, qubit + self.n]] = self.tableau[:, [qubit + self.n, qubit]]

    def apply_cx(self, ctrl: int, tar: int) -> None:
        """Apply the CNOT gate to the stabilizer tableau.

        Args:
            ctrl: The index of the control qubit.
            tar: The index of the target qubit.
        """
        self.phase ^= (
            self.tableau[:, ctrl]
            * self.tableau[:, tar + self.n]
            * (self.tableau[:, tar] ^ self.tableau[:, ctrl + self.n] ^ 1)
        )
        self.tableau[:, tar] = (self.tableau[:, tar] + self.tableau[:, ctrl]) % 2
        self.tableau[:, ctrl + self.n] = (self.tableau[:, tar + self.n] + self.tableau[:, ctrl + self.n]) % 2

    def apply_cz(self, ctrl: int, tar: int) -> None:
        """Apply the CZ gate to the stabilizer tableau.

        Args:
            ctrl: The index of the control qubit.
            tar: The index of the target qubit.
        """
        self.apply_h(tar)
        self.apply_cx(ctrl, tar)
        self.apply_h(tar)

    def apply_swap(self, q1: int, q2: int) -> None:
        """Apply the SWAP gate to the stabilizer tableau.

        Args:
            q1: The index of the first qubit.
            q2: The index of the second qubit.
        """
        self.apply_cx(q1, q2)
        self.apply_cx(q2, q1)
        self.apply_cx(q1, q2)

    def apply_s(self, qubit: int) -> None:
        """Apply the S gate to the stabilizer tableau.

        Args:
            qubit: The index of the qubit to apply the S gate to.
        """
        self.phase ^= self.tableau[:, qubit] * self.tableau[:, qubit + self.n]
        self.tableau[:, qubit + self.n] ^= self.tableau[:, qubit]

    def apply_sdg(self, qubit: int) -> None:
        """Apply the S† gate to the stabilizer tableau."""
        self.apply_s(qubit)
        self.apply_z(qubit)

    def apply_x(self, qubit: int) -> None:
        """Apply the X gate to the stabilizer tableau."""
        self.phase ^= self.tableau[:, qubit + self.n]

    def apply_z(self, qubit: int) -> None:
        """Apply the Z gate to the stabilizer tableau."""
        self.phase ^= self.tableau[:, qubit]

    def apply_y(self, qubit: int) -> None:
        """Apply the Y gate to the stabilizer tableau."""
        self.apply_x(qubit)
        self.apply_z(qubit)

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
        return bool(np.all(x_part[z_part.any(axis=1)] == 0) and np.all(z_part[x_part.any(axis=1)] == 0))

    def to_css(self) -> tuple[CheckMatrix, CheckMatrix]:
        """Convert the stabilizer tableau to CSS check matrices.

        Returns:
            A tuple containing the X and Z check matrices.
        """
        if not self.is_css():
            msg = "Stabilizer tableau is not in CSS form."
            raise InvalidPauliError(msg)
        x_part = self.get_x_part()
        z_part = self.get_z_part()
        x_checks: npt.NDArray[np.int8] = x_part[np.any(x_part, axis=1)]
        z_checks: npt.NDArray[np.int8] = z_part[np.any(z_part, axis=1)]
        return CheckMatrix(x_checks, "X"), CheckMatrix(z_checks, "Z")

    def get_x_part(self) -> npt.NDArray[np.int8]:
        """Get the X part of the stabilizer tableau."""
        return self.tableau.matrix[:, : self.n]

    def get_z_part(self) -> npt.NDArray[np.int8]:
        """Get the Z part of the stabilizer tableau."""
        return self.tableau.matrix[:, self.n :]

    def symplectic_submatrix(self, q: int) -> npt.NDArray[np.int8]:
        """Get the 2x2 symplectic submatrix for a given qubit.

        Args:
            q: The index of the qubit.

        Returns:
            A 2x2 NumPy array representing the symplectic submatrix for the given
            qubit.
        """
        return np.array(
            [
                [int(self.tableau[q, q]), int(self.tableau[q, q + self.n])],
                [int(self.tableau[q + self.n, q]), int(self.tableau[q + self.n, q + self.n])],
            ],
            dtype=np.int8,
        )

    def is_identity(self) -> bool:
        """Check if the stabilizer tableau is the identity.

        Returns:
            True if the stabilizer tableau is the identity, False otherwise.
        """
        return bool(
            np.array_equal(
                self.tableau.matrix,
                SymplecticMatrix.identity(self.n).matrix,
            )
            and np.all(self.phase == 0)
        )

    def __str__(self) -> str:
        """Return a string representation of the stabilizer tableau."""
        paulis = [str(self[i]) for i in range(self.n_rows)]
        return "\n".join(paulis)

    def __repr__(self) -> str:
        """Return a detailed string representation of the stabilizer tableau."""
        return (
            f"StabilizerTableau(n={self.n}, n_rows={self.n_rows}, tableau=\n{self.tableau.matrix},\nphase={self.phase})"
        )

    def num_rows(self) -> int:
        """Return the number of rows in the stabilizer tableau."""
        return self.n_rows

    def is_row(self, pauli: Pauli) -> bool:
        """Check if a given Pauli operator is a stabilizer of the tableau.

        Args:
            pauli: A Pauli operator to check.

        Returns:
            True if the Pauli operator is a stabilizer, False otherwise.
        """
        symplectic_vector = pauli.symplectic
        for i in range(self.n_rows):
            stab_vector = SymplecticVector(self.tableau[i])
            if symplectic_vector == stab_vector and pauli.phase == self.phase[i]:
                return True
        return False


def complete_stabilizer_tableau_with_destabilizers(
    stabilizers: StabilizerTableau, stab_rows: list[int] | None = None
) -> StabilizerTableau:
    """Given a tableau of stabilizers, complete it to a full tableau by adding destabilizers.

    Destabilizer d_i anticommutes with stabilizer s_i but commutes with all other stabilizers,
    destabilizers, and logical operators.

    Args:
        stabilizers: A tableau representing the stabilizers (and possibly some destabilizers) of the code.
        stab_rows: List of row indices that are stabilizers. If None, assumes all rows are stabilizers.
            Destabilizers will be added for each row specified in stab_rows.

    Returns:
        A tableau ordered as: destabilizers, logical X, stabilizers, logical Z.

    Raises:
        ValueError: If any row index in stab_rows is out of bounds or if valid destabilizers cannot be found.
    """
    n = stabilizers.n
    m_total = stabilizers.num_rows()

    if stab_rows is None:
        stab_rows = list(range(m_total))

    if not stab_rows:
        return stabilizers.copy()

    if max(stab_rows) >= m_total or min(stab_rows) < 0:
        msg = f"Row indices in stab_rows must be between 0 and {m_total - 1}."
        raise ValueError(msg)

    m = len(stab_rows)

    if m > n:
        msg = "Cannot have more stabilizers than qubits."
        raise ValueError(msg)

    stab_row_set = set(stab_rows)

    other_rows = [
        (i, stabilizers.tableau[i].copy(), stabilizers.phase[i]) for i in range(m_total) if i not in stab_row_set
    ]

    k = len(other_rows) // 2
    logical_x_rows = other_rows[:k]
    logical_z_rows = other_rows[k:]

    destabilizers: list[npt.NDArray[np.int8]] = []
    for idx, stab_row_idx in enumerate(stab_rows):
        stab_i = SymplecticVector(stabilizers.tableau[stab_row_idx])

        remaining_stab_indices = [stab_rows[j] for j in range(idx + 1, m)]

        d_i = _initialize_destabilizer_from_nullspace(stab_i, remaining_stab_indices, stabilizers)

        if d_i is None:
            msg = f"Could not find valid initial destabilizer for stabilizer at row {stab_row_idx}."
            raise ValueError(msg)

        for other_idx_pos in range(idx):
            s_j = SymplecticVector(stabilizers.tableau[stab_rows[other_idx_pos]])
            if d_i @ s_j == 1:
                d_j = SymplecticVector(destabilizers[other_idx_pos])
                d_i += d_j

        for other_idx_pos in range(idx):
            d_j = SymplecticVector(destabilizers[other_idx_pos])
            if d_i @ d_j == 1:
                s_j = SymplecticVector(stabilizers.tableau[stab_rows[other_idx_pos]])
                d_i += s_j

        for logical_idx, (_, logical_row, _) in enumerate(other_rows):
            log = SymplecticVector(logical_row)
            if d_i @ log == 1:
                if logical_idx < k:
                    l_z = SymplecticVector(other_rows[logical_idx + k][1])
                    d_i += l_z
                else:
                    l_x = SymplecticVector(other_rows[logical_idx - k][1])
                    d_i += l_x

        if d_i @ stab_i != 1:
            msg = f"Destabilizer construction failed for stabilizer at row {stab_row_idx}."
            raise ValueError(msg)

        destabilizers.append(d_i.vector.copy())

    if len(destabilizers) != m:
        msg = f"Could not find {m} valid destabilizers, only found {len(destabilizers)}."
        raise ValueError(msg)

    new_rows = []
    new_phases = []

    for _, row, phase in logical_x_rows:
        new_rows.append(row)
        new_phases.append(phase)

    for destab in destabilizers:
        new_rows.append(destab)
        new_phases.append(0)  # TODO: probably wrong, need to compute the correct phase for the destabilizer

    for _, row, phase in logical_z_rows:
        new_rows.append(row)
        new_phases.append(phase)

    for stab_row_idx in stab_rows:
        new_rows.append(stabilizers.tableau[stab_row_idx].copy())
        new_phases.append(stabilizers.phase[stab_row_idx])

    combined_matrix = np.vstack(new_rows)
    combined_phase = np.array(new_phases, dtype=np.int8)

    return StabilizerTableau(SymplecticMatrix(combined_matrix), combined_phase)


def _initialize_destabilizer_from_nullspace(
    stab_i: SymplecticVector,
    remaining_stab_indices: list[int],
    stabilizers: StabilizerTableau,
) -> SymplecticVector | None:
    """Initialize destabilizer from nullspace of remaining stabilizers."""
    n = stabilizers.n

    if not remaining_stab_indices:
        for q in range(n):
            for x, z in [(1, 0), (0, 1), (1, 1)]:
                candidate = SymplecticVector.zeros(n)
                candidate[q] = x
                candidate[q + n] = z
                if candidate @ stab_i == 1:
                    return candidate
        return None

    constraint_rows = []
    for stab_idx in remaining_stab_indices:
        s = stabilizers.tableau[stab_idx]
        s_x = s[:n]
        s_z = s[n:]
        constraint_row = np.concatenate([s_z, s_x])
        constraint_rows.append(constraint_row)

    constraint_matrix = np.vstack(constraint_rows)

    null = mod2.nullspace(constraint_matrix)

    if null.shape[0] == 0:
        return None

    for basis_vec in null:
        candidate = SymplecticVector(basis_vec)
        if candidate @ stab_i == 1:
            return candidate

    if null.shape[0] <= 12:
        for mask in range(1, 2 ** null.shape[0]):
            candidate_vec = np.zeros(2 * n, dtype=np.int8)
            for i in range(null.shape[0]):
                if mask & (1 << i):
                    candidate_vec = (candidate_vec + null[i]) % 2
            candidate = SymplecticVector(candidate_vec)
            if candidate @ stab_i == 1:
                return candidate

    return None


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


class CheckMatrix:
    """Type alias for CSS check matrices."""

    matrix: npt.NDArray[np.int8]
    type: str

    def __init__(self, matrix: npt.NDArray[np.int8], pauli_type: str) -> None:
        """Initialize the check matrix.

        Args:
            matrix: The binary check matrix.
            pauli_type: The type of the check matrix, either 'X' or 'Z'.
        """
        assert pauli_type in {"X", "Z"}, "Check matrix type must be either 'X' or 'Z'."
        self.matrix = matrix.copy()
        self.type = pauli_type

    def is_x_type(self) -> bool:
        """Check if the check matrix is of type 'X'."""
        return self.type == "X"

    def is_z_type(self) -> bool:
        """Check if the check matrix is of type 'Z'."""
        return self.type == "Z"

    def copy(self) -> CheckMatrix:
        """Create a copy of the check matrix."""
        return CheckMatrix(self.matrix.copy(), self.type)

    def is_identity(self) -> bool:
        """Check if the check matrix is an identity matrix."""
        n = self.matrix.shape[1]
        identity = np.eye(n, dtype=np.int8)
        return np.array_equal(self.matrix, identity)

    def __eq__(self, o: object) -> bool:
        """Check equality with another CheckMatrix."""
        if not isinstance(o, CheckMatrix):
            return NotImplemented
        return np.array_equal(self.matrix, o.matrix) and self.type == o.type

    def __hash__(self) -> int:
        """Compute the hash of the check matrix."""
        return hash((self.matrix.tobytes(), self.type))

    def num_qubits(self) -> int:
        """Get the number of qubits represented by the check matrix."""
        return int(self.matrix.shape[1])

    def num_rows(self) -> int:
        """Get the number of rows in the check matrix."""
        return int(self.matrix.shape[0])

    def __repr__(self) -> str:
        """Return a string representation of the check matrix."""
        return f"CheckMatrix(type={self.type}, matrix=\n{self.matrix})"
