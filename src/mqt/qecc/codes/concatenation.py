# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Concatenated quantum codes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .css_code import CSSCode
from .pauli import Pauli
from .stabilizer_code import InvalidStabilizerCodeError, StabilizerCode
from .symplectic import SymplecticVector

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt


class ConcatenatedCode(StabilizerCode):
    """A concatenated quantum code."""

    def __init__(self, outer_code: StabilizerCode, inner_code: StabilizerCode | Sequence[StabilizerCode]) -> None:
        """Initialize a concatenated quantum code.

        Args:
            outer_code: The outer code.
            inner_code: The inner code. If a list of codes is provided, the qubits of the outer code are encoded by the different inner codes in the list.
        """
        self.outer_code = outer_code
        if isinstance(inner_code, list):
            self.inner_codes = inner_code
        else:
            self.inner_codes = [inner_code] * outer_code.n
        if not all(code.k == 1 for code in self.inner_codes):
            msg = "The inner codes must be stabilizer codes with a single logical qubit."
            raise InvalidStabilizerCodeError(msg)

        self.n = sum(code.n for code in self.inner_codes)
        generators = [self._outer_pauli_to_physical(p) for p in outer_code.generators]

        x_logicals = None
        z_logicals = None
        if outer_code.x_logicals is not None:
            x_logicals = [self._outer_pauli_to_physical(p) for p in outer_code.x_logicals]
        if outer_code.z_logicals is not None:
            z_logicals = [self._outer_pauli_to_physical(p) for p in outer_code.z_logicals]

        d = min(code.distance * outer_code.distance for code in self.inner_codes)
        StabilizerCode.__init__(self, generators, d, x_logicals, z_logicals)

    def __eq__(self, other: object) -> bool:
        """Check if two concatenated codes are equal."""
        if not isinstance(other, ConcatenatedCode):
            return NotImplemented
        return self.outer_code == other.outer_code and all(
            c1 == c2 for c1, c2 in zip(self.inner_codes, other.inner_codes)
        )

    def __hash__(self) -> int:
        """Compute the hash of the concatenated code."""
        return hash((self.outer_code, tuple(self.inner_codes)))

    def _outer_pauli_to_physical(self, p: Pauli) -> Pauli:
        """Convert a Pauli operator on the outer code to the operator on the concatenated code.

        Args:
            p: The Pauli operator.

        Returns:
            The Pauli operator on the physical qubits.
        """
        if len(p) != self.outer_code.n:
            msg = "The Pauli operator must have the same number of qubits as the outer code."
            raise InvalidStabilizerCodeError(msg)
        concatenated = SymplecticVector.zeros(self.n)
        phase = 0
        offset = 0
        for i in range(self.outer_code.n):
            c = self.inner_codes[i]
            new_offset = offset + c.n
            assert c.x_logicals is not None
            assert c.z_logicals is not None
            if p[i] == "X":
                concatenated[offset:new_offset] = c.x_logicals[0].x_part()
                concatenated[offset + self.n : new_offset + self.n] = c.x_logicals[0].z_part()
                phase += c.x_logicals[0].phase
            elif p[i] == "Z":
                concatenated[offset:new_offset] = c.z_logicals[0].x_part()
                concatenated[offset + self.n : new_offset + self.n] = c.z_logicals[0].z_part()
                phase += c.z_logicals[0].phase

            elif p[i] == "Y":
                concatenated[offset:new_offset] = c.x_logicals[0].x_part ^ c.z_logicals[0].x_part()
                concatenated[offset + self.n : new_offset + self.n] = c.x_logicals[0].z_part ^ c.z_logicals[0].z_part()
                phase += c.x_logicals[0].phase + c.z_logicals[0].phase

            offset = new_offset
        return Pauli(concatenated, phase)


# def _valid_logicals(lst: list[StabilizerTableau | None]) -> TypeGuard[list[StabilizerTableau]]:
#     return None not in lst


class ConcatenatedCSSCode(ConcatenatedCode, CSSCode):
    """A concatenated CSS code."""

    def __init__(self, outer_code: CSSCode, inner_codes: CSSCode | Sequence[CSSCode]) -> None:
        """Initialize a concatenated CSS code.

        Args:
            outer_code: The outer code.
            inner_codes: The inner code. If a list of codes is provided, the qubits of the outer code are encoded by the different inner codes in the list.
        """
        # self.outer_code = outer_code
        if isinstance(inner_codes, CSSCode):
            inner_codes = [inner_codes] * outer_code.n

        if not all(code.k == 1 for code in inner_codes):
            msg = "The inner codes must be CSS codes with a single logical qubit."
            raise InvalidStabilizerCodeError(msg)

        ConcatenatedCode.__init__(self, outer_code, inner_codes)
        hx = np.array([self._outer_checks_to_physical(check, "X") for check in outer_code.Hx], dtype=np.int8)
        hz = np.array([self._outer_checks_to_physical(check, "Z") for check in outer_code.Hz], dtype=np.int8)
        d = min(code.distance * outer_code.distance for code in inner_codes)
        CSSCode.__init__(self, hx, hz, d)

    def _outer_checks_to_physical(self, check: npt.NDArray[np.int8], operator: str) -> npt.NDArray[np.int8]:
        """Convert a check operator on the outer code to the operator on the concatenated code.

        Args:
            check: The check operator.
            operator: The type of operator to be converted. Either 'X' or 'Z'.

        Returns:
            The check operator on the physical qubits.
        """
        if check.shape[0] != self.outer_code.n:
            msg = "The check operator must have the same number of qubits as the outer code."
            raise InvalidStabilizerCodeError(msg)
        concatenated = np.zeros((self.n), dtype=np.int8)
        offset = 0
        for i in range(self.outer_code.n):
            c = self.inner_codes[i]
            new_offset = offset + c.n
            if check[i] == 1:
                logical = c.Lx if operator == "X" else c.Lz
                concatenated[offset:new_offset] = logical
            offset = new_offset
        return concatenated
