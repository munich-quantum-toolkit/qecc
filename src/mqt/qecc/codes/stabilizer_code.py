# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Class for representing general stabilizer codes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ldpc import mod2

from .pauli import Pauli, StabilizerTableau

if TYPE_CHECKING:
    import numpy.typing as npt


class StabilizerCode:
    """A class for representing stabilizer codes."""

    def __init__(
        self,
        generators: StabilizerTableau | list[Pauli] | list[str],
        distance: int | None = None,
        z_logicals: StabilizerTableau | list[Pauli] | list[str] | None = None,
        x_logicals: StabilizerTableau | list[Pauli] | list[str] | None = None,
        n: int | None = None,
    ) -> None:
        """Initialize the code.

        Args:
            generators: The stabilizer generators of the code. We assume that stabilizers are ordered from left to right in ascending order of qubits.
            distance: The distance of the code.
            z_logicals: The logical Z-operators.
            x_logicals: The logical X-operators.
            n: The number of qubits in the code. If not given, it is inferred from the stabilizer generators.
        """
        self.generators = self.get_generators(generators, n)
        self.symplectic = self.generators.tableau.matrix

        if n is None:
            self.n = self.generators.n
        else:
            self.n = n

        if self.generators.n_rows != 0:
            self.k = self.n - mod2.rank(self.generators.as_matrix())
        else:
            self.k = self.n

        if distance is not None and distance <= 0:
            msg = "Distance must be a positive integer."
            raise InvalidStabilizerCodeError(msg)

        self.distance = 1 if distance is None else distance  # default distance is 1

        self.z_logicals = None
        self.x_logicals = None

        if z_logicals is not None:
            self.z_logicals = self.get_generators(z_logicals)
        if x_logicals is not None:
            self.x_logicals = self.get_generators(x_logicals)

        self._check_code_correct()

    def __hash__(self) -> int:
        """Compute a hash for the stabilizer code."""
        return hash(self.generators)

    def __eq__(self, other: object) -> bool:
        """Check if two stabilizer codes are equal."""
        if not isinstance(other, StabilizerCode):
            return NotImplemented
        rnk = mod2.rank(self.generators.as_matrix())
        return bool(
            rnk == mod2.rank(other.generators.as_matrix())
            and rnk == mod2.rank(np.vstack((self.generators.as_matrix(), other.generators.as_matrix())))
        )

    def get_syndrome(self, error: Pauli | str) -> npt.NDArray:
        """Compute the syndrome of the error.

        Args:
            error: The error as a pauli string or binary vector.
        """
        if isinstance(error, str):
            error = Pauli.from_pauli_string(error)
        return (self.generators.tableau @ error.symplectic).vector

    def stabs_as_pauli_strings(self) -> list[str]:
        """Return the stabilizers as Pauli strings."""
        return [str(p) for p in self.generators]

    def stabilizer_equivalent(self, p1: Pauli | str, p2: Pauli | str) -> bool:
        """Check if two Pauli strings are equivalent up to stabilizers of the code."""
        if isinstance(p1, str):
            p1 = Pauli.from_pauli_string(p1)
        if isinstance(p2, str):
            p2 = Pauli.from_pauli_string(p2)
        return bool(
            mod2.rank(np.vstack((self.generators.as_matrix(), p1.as_vector(), p2.as_vector())))
            == mod2.rank(np.vstack((self.generators.as_matrix(), p1.as_vector())))
        )

    def _check_code_correct(self) -> None:
        """Check if the code is correct. Throws an exception if not."""
        if self.z_logicals is not None or self.x_logicals is not None:
            if self.z_logicals is None:
                msg = "If logical X-operators are given, logical Z-operators must also be given."
                raise InvalidStabilizerCodeError(msg)
            if self.x_logicals is None:
                msg = "If logical Z-operators are given, logical X-operators must also be given."
                raise InvalidStabilizerCodeError(msg)

        if self.z_logicals is None:
            return

        if self.x_logicals is None:
            return

        if self.z_logicals.n != self.n:
            msg = "Logical operators must have the same number of qubits as the stabilizer generators."
            raise InvalidStabilizerCodeError(msg)

        if self.z_logicals.shape[0] > self.k:
            msg = "Number of logical Z-operators must be at most the number of logical qubits."
            raise InvalidStabilizerCodeError(msg)

        if self.x_logicals.n != self.n:
            msg = "Logical operators must have the same number of qubits as the stabilizer generators."
            raise InvalidStabilizerCodeError(msg)

        if self.x_logicals.shape[0] > self.k:
            msg = "Number of logical X-operators must be at most the number of logical qubits."
            raise InvalidStabilizerCodeError(msg)

        if not self.z_logicals.all_commute(self.generators):
            msg = "Logical Z-operators must anti-commute with the stabilizer generators."
            raise InvalidStabilizerCodeError(msg)
        if not self.x_logicals.all_commute(self.generators):
            msg = "Logical X-operators must commute with the stabilizer generators."
            raise InvalidStabilizerCodeError(msg)

        commutations = (self.z_logicals.tableau @ self.x_logicals.tableau).matrix
        if not np.all(np.sum(commutations, axis=1) == 1):
            msg = "Every logical X-operator must anti-commute with exactly one logical Z-operator."
            raise InvalidStabilizerCodeError(msg)

    @staticmethod
    def get_generators(
        generators: StabilizerTableau | list[Pauli] | list[str], n: int | None = None
    ) -> StabilizerTableau:
        """Get the stabilizer generators as a StabilizerTableau object.

        Args:
            generators: The stabilizer generators as a StabilizerTableau object, a list of Pauli objects, or a list of Pauli strings.
            n: The number of qubits in the code. Required if generators is an empty list.
        """
        if isinstance(generators, list):
            if len(generators) == 0:
                if n is None:
                    msg = "Number of qubits must be given if no generators are provided."
                    raise ValueError(msg)
                return StabilizerTableau.empty(n)
            if isinstance(generators[0], str):
                return StabilizerTableau.from_pauli_strings(generators)  # type: ignore[arg-type]
            if isinstance(generators[0], Pauli):
                return StabilizerTableau.from_paulis(generators)  # type: ignore[arg-type]
        return generators

    @classmethod
    def get_trivial_code(cls, n: int) -> StabilizerCode:
        """Get the trivial stabilizer code."""
        z_logicals = ["I" * i + "Z" + "I" * (n - i - 1) for i in range(n)]
        x_logicals = ["I" * i + "X" + "I" * (n - i - 1) for i in range(n)]
        return StabilizerCode([], distance=1, z_logicals=z_logicals, x_logicals=x_logicals, n=n)

    @classmethod
    def from_file(cls, file_path: str | Path) -> StabilizerCode:
        """Load a Stabilizer code from a file.

        The file should have one line per stabilizer generator as a string.

        For the 5-qubit perfect code, this would be:
        IXXZZ
        ZIXXZ
        ZZIXX
        XZZIX

        Args:
            file_path: The path to the file containing the code.

        Returns:
            StabilizerCode: The stabilizer code.
        """
        with Path(file_path).open(encoding="utf-8") as f:
            lines = f.readlines()
        generators = [line.strip() for line in lines]
        return cls(generators)


class InvalidStabilizerCodeError(ValueError):
    """Raised when the stabilizer code is invalid."""
