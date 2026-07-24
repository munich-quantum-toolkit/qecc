# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Class for representing quantum error correction codes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mqt.qecc import mod2

from ...mod2 import are_in_same_coset, is_in_row_space
from .pauli import CheckMatrix, PauliTableau
from .stabilizer_code import StabilizerCode

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt


class CSSCode(StabilizerCode):
    """A class for representing CSS codes."""

    def __init__(
        self,
        Hx: npt.NDArray[np.int8] | None = None,  # noqa: N803
        Hz: npt.NDArray[np.int8] | None = None,  # noqa: N803
        distance: int | None = None,
        x_distance: int | None = None,
        z_distance: int | None = None,
        n: int | None = None,
        Lx: npt.NDArray[np.int8] | None = None,  # noqa: N803
        Lz: npt.NDArray[np.int8] | None = None,  # noqa: N803
    ) -> None:
        """Initialize the code."""
        if Hx is None and Hz is None:
            if n is None:
                msg = "If no check matrices are provided, the code size must be specified."
                raise InvalidCSSCodeError(msg)
            hx = np.zeros((0, n), dtype=np.int8)
            hz = np.zeros((0, n), dtype=np.int8)
        else:
            self._check_valid_check_matrices(Hx, Hz)
            if Hx is not None:
                inferred_n = Hx.shape[1]
                hx = Hx
                hz = Hz if Hz is not None else np.zeros((0, inferred_n), dtype=np.int8)
            else:
                assert Hz is not None
                inferred_n = Hz.shape[1]
                hx = np.zeros((0, inferred_n), dtype=np.int8)
                hz = Hz

        num_qubits = hx.shape[1]
        if n is not None and n != num_qubits:
            msg = f"Given code size n={n} does not match check-matrix width {num_qubits}."
            raise InvalidCSSCodeError(msg)
        num_logicals = num_qubits - mod2.rank(hx) - mod2.rank(hz)

        if (Lx is None) != (Lz is None):
            msg = "Both Lx and Lz must be provided together or both must be None."
            raise InvalidCSSCodeError(msg)

        if Lx is not None and Lz is not None:
            lx = Lx.copy()
            lz = Lz.copy()
        else:
            lx = CSSCode._compute_logical(hz, hx)
            lz = CSSCode._compute_logical(hx, hz)
            CSSCode._normalize_logicals(lx, lz, num_logicals)

        if len(lx) == 0:
            lx = np.zeros((0, num_qubits), dtype=np.int8)
        if len(lz) == 0:
            lz = np.zeros((0, num_qubits), dtype=np.int8)

        generators = _tableau_from_css_checks(hx, hz)
        x_logicals = PauliTableau.from_check_matrix(CheckMatrix(lx, "X"))
        z_logicals = PauliTableau.from_check_matrix(CheckMatrix(lz, "Z"))

        self._num_x_checks = hx.shape[0]
        super().__init__(
            generators,
            distance,
            x_logicals=x_logicals,
            z_logicals=z_logicals,
            n=num_qubits,
        )

        self.x_distance = x_distance if x_distance is not None else self.distance
        self.z_distance = z_distance if z_distance is not None else self.distance

        if self.x_distance < self.distance or self.z_distance < self.distance:
            msg = "The x and z distances must be greater than or equal to the distance"
            raise InvalidCSSCodeError(msg)

    @property
    def Hx(self) -> npt.NDArray[np.int8]:  # noqa: N802
        """The X-check matrix as a view into the stabilizer generators."""
        return self.generators.symplectic[: self._num_x_checks, : self.n]

    @property
    def Hz(self) -> npt.NDArray[np.int8]:  # noqa: N802
        """The Z-check matrix as a view into the stabilizer generators."""
        return self.generators.symplectic[self._num_x_checks :, self.n :]

    @property
    def Lx(self) -> npt.NDArray[np.int8]:  # noqa: N802
        """The logical X matrix as a view into the logical X operators."""
        return self.x_logicals.get_x_part()

    @property
    def Lz(self) -> npt.NDArray[np.int8]:  # noqa: N802
        """The logical Z matrix as a view into the logical Z operators."""
        return self.z_logicals.get_z_part()

    def x_checks_as_pauli_strings(self) -> list[str]:
        """Return the x checks as Pauli strings."""
        return ["".join("X" if bit == 1 else "I" for bit in row) for row in self.Hx]

    def z_checks_as_pauli_strings(self) -> list[str]:
        """Return the z checks as Pauli strings."""
        return ["".join("Z" if bit == 1 else "I" for bit in row) for row in self.Hz]

    def x_logicals_as_pauli_strings(self) -> list[str]:
        """Return the x logicals as a Pauli strings."""
        return ["".join("X" if bit == 1 else "I" for bit in row) for row in self.Lx]

    def z_logicals_as_pauli_strings(self) -> list[str]:
        """Return the z logicals as Pauli strings."""
        return ["".join("Z" if bit == 1 else "I" for bit in row) for row in self.Lz]

    @staticmethod
    def _compute_logical(m1: npt.NDArray[np.int8], m2: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Compute the logical matrix L."""
        ker_m1 = mod2.nullspace(m1)  # compute the kernel basis of m1
        im_m2_transp = mod2.row_basis(m2)  # compute the image basis of m2
        log_stack = np.vstack([im_m2_transp, ker_m1], dtype=np.int8)
        pivots = mod2.row_echelon(log_stack.T)[3]
        log_op_indices = [i for i in range(im_m2_transp.shape[0], log_stack.shape[0]) if i in pivots]
        return log_stack[log_op_indices]

    def get_x_syndrome(self, error: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Compute the x syndrome of the error."""
        return self.Hx @ error % 2

    def get_z_syndrome(self, error: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Compute the z syndrome of the error."""
        return self.Hz @ error % 2

    def check_if_logical_x_error(self, residual: npt.NDArray[np.int8]) -> bool:
        """Check if the residual X error acts as a logical operator (anticommutes with some Z logical)."""
        return bool((self.Lz @ residual % 2 == 1).any())

    def check_if_x_stabilizer(self, pauli: npt.NDArray[np.int8]) -> bool:
        """Check if the X-type Pauli (given by its support) is an X stabilizer."""
        return is_in_row_space(pauli, self.Hx)

    def check_if_logical_z_error(self, residual: npt.NDArray[np.int8]) -> bool:
        """Check if the residual Z error acts as a logical operator (anticommutes with some X logical)."""
        return bool((self.Lx @ residual % 2 == 1).any())

    def check_if_z_stabilizer(self, pauli: npt.NDArray[np.int8]) -> bool:
        """Check if the Z-type Pauli (given by its support) is a Z stabilizer."""
        return is_in_row_space(pauli, self.Hz)

    def stabilizer_eq_x_error(self, error_1: npt.NDArray[np.int8], error_2: npt.NDArray[np.int8]) -> bool:
        """Check if two X errors are in the same coset of the X stabilizers."""
        return are_in_same_coset(error_1, error_2, self.Hx)

    def stabilizer_eq_z_error(self, error_1: npt.NDArray[np.int8], error_2: npt.NDArray[np.int8]) -> bool:
        """Check if two Z errors are in the same coset of the Z stabilizers."""
        return are_in_same_coset(error_1, error_2, self.Hz)

    def is_self_dual(self) -> bool:
        """Check if the code is self-dual."""
        return bool(
            self.Hx.shape[0] == self.Hz.shape[0] and mod2.rank(self.Hx) == mod2.rank(np.vstack([self.Hx, self.Hz]))
        )

    @staticmethod
    def _check_valid_check_matrices(Hx: npt.NDArray[np.int8] | None, Hz: npt.NDArray[np.int8] | None) -> None:  # noqa: N803
        """Check if the code is a valid CSS code."""
        if Hx is not None and Hz is not None:
            if Hx.shape[1] != Hz.shape[1]:
                msg = "Check matrices must have the same number of columns"
                raise InvalidCSSCodeError(msg)
            if np.any(Hx @ Hz.T % 2 != 0):
                msg = "The check matrices must be orthogonal"
                raise InvalidCSSCodeError(msg)

    @classmethod
    def get_trivial_code(cls, n: int) -> CSSCode:
        """Return the trivial code."""
        return CSSCode(None, None, 1, n=n)

    @staticmethod
    def from_code_name(code_name: str) -> CSSCode:
        r"""Return CSSCode object for a known code.

        The following codes are supported:
        - [[7, 1, 3]] Steane (\"Steane\")
        - [[15, 1, 3]] tetrahedral code (\"Tetrahedral\")
        - [[9, 1, 3]] Shore code (\"Shor\")
        - [[12, 2, 4]] Carbon Code (\"Carbon\")
        - [[23, 1, 7]] golay code (\"Golay\")

        Args:
            code_name: The name of the code.
        """
        prefix = (Path(__file__) / "../../instances/").resolve()
        paths = {
            "steane": prefix / "steane/",
            "tetrahedral": prefix / "tetrahedral/",
            "shor": prefix / "shor/",
            "golay": prefix / "golay/",
            "carbon": prefix / "carbon/",
        }

        distances = {
            "steane": (3, 3),
            "tetrahedral": (7, 3),
            "shor": (3, 3),
            "golay": (7, 7),
            "carbon": (4, 4),
        }  # X, Z distances

        code_name = code_name.lower()
        if code_name not in paths:
            msg = f"Unknown code name: {code_name}"
            raise InvalidCSSCodeError(msg)

        hx = np.load(paths[code_name] / "hx.npy")
        hz = np.load(paths[code_name] / "hz.npy")
        x_distance, z_distance = distances[code_name]
        distance = min(x_distance, z_distance)
        return CSSCode(hx, hz, distance, x_distance=x_distance, z_distance=z_distance)

    @classmethod
    def from_file(cls, file_path: str | Path) -> CSSCode:
        """Load a CSS code from a file.

        The file can contain either:
        1. Pauli string format - X and Z stabilizers as Pauli strings
        2. Binary matrix format - X stabilizers followed by empty line, then Z stabilizers

        For Pauli string format (Steane code example):
        XIIXXXI
        IXIIXXX
        IIXXIXX
        ZIIZZZI
        IZIIZZZ
        IIZZIZZ

        For binary matrix format (space-separated):
        1 0 0 1 1 1 0
        0 1 0 0 1 1 1
        0 0 1 1 0 1 1

        1 0 0 1 1 1 0
        0 1 0 0 1 1 1
        0 0 1 1 0 1 1

        For list notation format:
        [[1,0,0,1,1,1,0],
         [0,1,0,0,1,1,1],
         [0,0,1,1,0,1,1]]

        [[1,0,0,1,1,1,0],
         [0,1,0,0,1,1,1],
         [0,0,1,1,0,1,1]]

        For numpy array notation:
        [[1 0 0 1 1 1 0]
         [0 1 0 0 1 1 1]
         [0 0 1 1 0 1 1]]

        [[1 0 0 1 1 1 0]
         [0 1 0 0 1 1 1]
         [0 0 1 1 0 1 1]]

        Args:
            file_path: The path to the file containing the code.

        Returns:
            CSSCode: The CSS code.
        """
        content = Path(file_path).read_text(encoding="utf-8").strip()

        if not content:
            msg = "File is empty"
            raise InvalidCSSCodeError(msg)

        if _is_css_binary_matrix_format(content):
            return _load_css_from_binary_matrix(content)

        lines = content.split("\n")
        stabilizers = [line.strip() for line in lines if line.strip()]

        x_stabs = []
        z_stabs = []

        for stab in stabilizers:
            if "X" in stab:
                x_stabs.append([1 if c == "X" else 0 for c in stab])
            elif "Z" in stab:
                z_stabs.append([1 if c == "Z" else 0 for c in stab])
            else:
                msg = f"Invalid stabilizer: {stab}"
                raise InvalidCSSCodeError(msg)

        x_stabs_array = np.array(x_stabs, dtype=np.int8) if x_stabs else None
        z_stabs_array = np.array(z_stabs, dtype=np.int8) if z_stabs else None

        return CSSCode(x_stabs_array, z_stabs_array)

    @staticmethod
    def _normalize_logicals(lx: npt.NDArray[np.int8], lz: npt.NDArray[np.int8], num_logicals: int) -> None:
        """Normalize the logical operators.

        The basis of logical operators computed by `_compute_logical` do not necessarily
        represent single-logical qubit operators but might be product operators.
        This method normalizes the logical operators such that each logical qubit
        is represented by a pair of logical X and Z operators that anti-commute
        only with each other, i.e. Lx[i]@Lz[j].T = 1 if i == j else 0.
        Assumes that `_compute_logical` has already been called to compute ``lx`` and ``lz``.
        """
        assert lx.shape[0] == lz.shape[0], "Number of X and Z logicals must be the same."

        for i in range(num_logicals):
            xl = lx[i]
            anticommute_x = np.where(xl @ lz.T % 2)[0]
            first = anticommute_x[0]
            zl = lz[first]
            anticommute_z = np.where(zl @ lx.T % 2)[0]
            for j in anticommute_x[1:]:
                lz[j] ^= zl

            for j in anticommute_z:
                if j != i:
                    lx[j] ^= xl
            lz[[i, first]] = lz[[first, i]]

    def set_x_logicals(self, logicals: npt.NDArray[np.int8]) -> None:
        """Set all X logical operators."""
        if logicals.shape[0] != self.k:
            msg = f"Number of logicals {logicals.shape[0]} does not match k={self.k}"
            raise InvalidCSSCodeError(msg)

        commutes = (self.Hz @ logicals.T) % 2
        if np.any(commutes != 0):
            msg = "Logical operators must commute with the Z stabilizers"
            raise InvalidCSSCodeError(msg)

        self.x_logicals = PauliTableau.from_check_matrix(CheckMatrix(logicals, pauli_type="X"))

    def set_z_logicals(self, logicals: npt.NDArray[np.int8]) -> None:
        """Set all Z logical operators."""
        if logicals.shape[0] != self.k:
            msg = f"Number of logicals {logicals.shape[0]} does not match k={self.k}"
            raise InvalidCSSCodeError(msg)

        commutes = (self.Hx @ logicals.T) % 2
        if np.any(commutes != 0):
            msg = "Logical operators must commute with the X stabilizers"
            raise InvalidCSSCodeError(msg)

        self.z_logicals = PauliTableau.from_check_matrix(CheckMatrix(logicals, pauli_type="Z"))


def _tableau_from_css_checks(hx: npt.NDArray[np.int8], hz: npt.NDArray[np.int8]) -> PauliTableau:
    """Combine CSS check matrices into a Pauli tableau."""
    x_rows = np.hstack((hx, np.zeros_like(hx)))
    z_rows = np.hstack((np.zeros_like(hz), hz))
    return PauliTableau(np.vstack((x_rows, z_rows)))


def _is_css_binary_matrix_format(content: str) -> bool:
    """Check if the content appears to be CSS binary matrix format.

    Args:
        content: The file content to check.

    Returns:
        True if the content looks like a CSS binary matrix format, False otherwise.
    """
    content = content.strip()

    if not content:
        return False

    if content.startswith("[["):
        sections = content.split("]]")
        return len(sections) >= 3

    lines = [line.strip() for line in content.split("\n") if line.strip()]
    if not lines:
        return False

    first_line = lines[0].strip()
    if not first_line:
        return False

    tokens = [t.strip() for t in first_line.split(",")] if "," in first_line else first_line.split()

    if len(tokens) < 2:
        return False

    return all(token in {"0", "1"} for token in tokens[:5])


def _load_css_from_binary_matrix(content: str) -> CSSCode:
    """Load a CSS code from binary matrix format.

    Args:
        content: The file content containing the binary matrices.

    Returns:
        CSSCode: The CSS code.
    """
    content = content.strip()

    if content.startswith("[["):
        return _load_css_from_list_notation(content)

    lines = content.split("\n")
    sections = []
    current_section: list[str] = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if current_section:
                sections.append(current_section)
                current_section = []
        else:
            current_section.append(line_stripped)

    if current_section:
        sections.append(current_section)

    if len(sections) < 1:
        msg = "No valid binary matrix data found in file"
        raise InvalidCSSCodeError(msg)

    if len(sections) > 2:
        msg = "Too many sections in file. Expected at most 2 (X stabilizers and Z stabilizers)"
        raise InvalidCSSCodeError(msg)

    hx = _parse_binary_matrix_section(sections[0]) if len(sections) >= 1 else None
    hz = _parse_binary_matrix_section(sections[1]) if len(sections) >= 2 else None

    return CSSCode(hx, hz)


def _load_css_from_list_notation(content: str) -> CSSCode:
    """Load a CSS code from list notation format.

    Args:
        content: The file content in list notation format.

    Returns:
        CSSCode: The CSS code.
    """
    sections = content.split("]]")
    matrices = []

    for section in sections:
        section_stripped = section.strip()
        if not section_stripped:
            continue

        section_stripped = section_stripped.lstrip("[").strip()
        if not section_stripped:
            continue

        rows = []
        for line in section_stripped.split("\n"):
            line_stripped = line.strip().lstrip("[").rstrip(",").rstrip("]")
            if not line_stripped:
                continue

            # Handle both comma-separated and space-separated values
            tokens = (
                [t.strip() for t in line_stripped.split(",") if t.strip()]
                if "," in line_stripped
                else line_stripped.split()
            )

            if not tokens:
                continue

            for token in tokens:
                if token not in {"0", "1"}:
                    msg = f"Invalid token '{token}' in binary matrix (expected '0' or '1'): {line.strip()}"
                    raise InvalidCSSCodeError(msg)

            row = [int(t) for t in tokens]
            rows.append(row)

        if rows:
            matrices.append(np.array(rows, dtype=np.int8))

    if len(matrices) < 1:
        msg = "No valid binary matrix data found in file"
        raise InvalidCSSCodeError(msg)

    if len(matrices) > 2:
        msg = "Too many matrices in file. Expected at most 2 (X stabilizers and Z stabilizers)"
        raise InvalidCSSCodeError(msg)

    hx = matrices[0] if len(matrices) >= 1 else None
    hz = matrices[1] if len(matrices) >= 2 else None

    return CSSCode(hx, hz)


def _parse_binary_matrix_section(lines: list[str]) -> npt.NDArray[np.int8]:
    """Parse a section of lines into a binary matrix.

    Args:
        lines: List of lines containing binary matrix data.

    Returns:
        Binary matrix as numpy array.
    """
    rows = []
    for line in lines:
        tokens = [t.strip() for t in line.split(",") if t.strip()] if "," in line else line.split()

        if not tokens:
            continue

        for token in tokens:
            if token not in {"0", "1"}:
                msg = f"Invalid token '{token}' in binary matrix (expected '0' or '1'): {line.strip()}"
                raise InvalidCSSCodeError(msg)

        row = [int(t) for t in tokens]
        rows.append(row)

    if not rows:
        msg = "No valid binary matrix data found in section"
        raise InvalidCSSCodeError(msg)

    return np.array(rows, dtype=np.int8)


class InvalidCSSCodeError(ValueError):
    """Raised when the CSS code is invalid."""
