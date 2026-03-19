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

import ldpc.mod2.mod2_numpy as mod2
import numpy as np

from .pauli import CheckMatrix, StabilizerTableau
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
            self.Hx = np.zeros((0, n), dtype=np.int8)
            self.Hz = np.zeros((0, n), dtype=np.int8)
            self.Lx = np.eye(n, dtype=np.int8)
            self.Lz = np.eye(n, dtype=np.int8)
            triv = StabilizerCode.get_trivial_code(n)
            super().__init__(triv.generators, triv.distance, triv.x_logicals, triv.z_logicals)
            return

        self._check_valid_check_matrices(Hx, Hz)

        if Hx is None:
            assert Hz is not None
            self.n = Hz.shape[1]
            self.Hx = np.zeros((0, self.n), dtype=np.int8)
        else:
            self.Hx = Hx
        if Hz is None:
            assert Hx is not None
            self.n = Hx.shape[1]
            self.Hz = np.zeros((0, self.n), dtype=np.int8)
        else:
            self.Hz = Hz

        z_padding = np.zeros(self.Hx.shape, dtype=np.int8)
        x_padding = np.zeros(self.Hz.shape, dtype=np.int8)

        x_padded = np.hstack([self.Hx, z_padding])
        z_padded = np.hstack([x_padding, self.Hz])
        phases = np.zeros((x_padded.shape[0] + z_padded.shape[0]), dtype=np.int8)
        super().__init__(StabilizerTableau(np.vstack((x_padded, z_padded)), phases), distance)

        self.x_distance = x_distance if x_distance is not None else self.distance
        self.z_distance = z_distance if z_distance is not None else self.distance

        if self.x_distance < self.distance or self.z_distance < self.distance:
            msg = "The x and z distances must be greater than or equal to the distance"
            raise InvalidCSSCodeError(msg)

        if Lx is not None:
            self.Lx = Lx
        else:
            self.Lx = CSSCode._compute_logical(self.Hz, self.Hx)
        if Lz is not None:
            self.Lz = Lz
        else:
            self.Lz = CSSCode._compute_logical(self.Hx, self.Hz)

        if Lx is None and Lz is None:
            self._normalize_logicals()

        if len(self.Lx) == 0:
            self.Lx = np.zeros((0, self.n), dtype=np.int8)
        if len(self.Lz) == 0:
            self.Lz = np.zeros((0, self.n), dtype=np.int8)

        self.x_logicals = StabilizerTableau.from_check_matrix(CheckMatrix(self.Lx, pauli_type="X"))
        self.z_logicals = StabilizerTableau.from_check_matrix(CheckMatrix(self.Lz, pauli_type="Z"))

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
        log_stack = np.vstack([im_m2_transp, ker_m1])
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
        """Check if the residual is a logical error."""
        return bool((self.Lz @ residual % 2 == 1).any())

    def check_if_x_stabilizer(self, pauli: npt.NDArray[np.int8]) -> bool:
        """Check if the Pauli is a stabilizer."""
        return bool(mod2.rank(np.vstack((self.Hx, pauli))) == mod2.rank(self.Hx))

    def check_if_logical_z_error(self, residual: npt.NDArray[np.int8]) -> bool:
        """Check if the residual is a logical error."""
        return (self.Hx.shape[0] != 0) and bool((self.Lx @ residual % 2 == 1).any())

    def check_if_z_stabilizer(self, pauli: npt.NDArray[np.int8]) -> bool:
        """Check if the Pauli is a stabilizer."""
        return (self.Hz.shape[0] != 0) and bool(mod2.rank(np.vstack((self.Hz, pauli))) == mod2.rank(self.Hz))

    def stabilizer_eq_x_error(self, error_1: npt.NDArray[np.int8], error_2: npt.NDArray[np.int8]) -> bool:
        """Check if two X errors are in the same coset."""
        if self.Hx.shape[0] == 0:
            return bool(np.array_equal(error_1, error_2))
        m1 = np.vstack([self.Hx, error_1])
        m2 = np.vstack([self.Hx, error_2])
        m3 = np.vstack([self.Hx, error_1, error_2])
        return bool(mod2.rank(m1) == mod2.rank(m2) == mod2.rank(m3))

    def stabilizer_eq_z_error(self, error_1: npt.NDArray[np.int8], error_2: npt.NDArray[np.int8]) -> bool:
        """Check if two Z errors are in the same coset."""
        if self.Hz.shape[0] == 0:
            return bool(np.array_equal(error_1, error_2))
        m1 = np.vstack([self.Hz, error_1])
        m2 = np.vstack([self.Hz, error_2])
        m3 = np.vstack([self.Hz, error_1, error_2])
        return bool(mod2.rank(m1) == mod2.rank(m2) == mod2.rank(m3))

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
    def from_code_name(code_name: str, distance: int | None = None) -> CSSCode:
        r"""Return CSSCode object for a known code.

        The following codes are supported:
        - [[7, 1, 3]] Steane (\"Steane\")
        - [[15, 1, 3]] tetrahedral code (\"Tetrahedral\")
        - [[9, 1, 3]] Shore code (\"Shor\")
        - [[12, 2, 4]] Carbon Code (\"Carbon\")
        - [[9, 1, 3]] rotated surface code (\"Surface, 3\"), also default when no distance is given
        - [[25, 1, 5]] rotated surface code (\"Surface, 5\")
        - [[15, 7, 3]] Hamming code (\"Hamming\")
        - [[23, 1, 7]] golay code (\"Golay\")

        Args:
            code_name: The name of the code.
            distance: The distance of the code.
        """
        prefix = (Path(__file__) / "../").resolve()
        paths = {
            "steane": prefix / "steane/",
            "tetrahedral": prefix / "tetrahedral/",
            "shor": prefix / "shor/",
            "surface_3": prefix / "rotated_surface_d3/",
            "surface_5": prefix / "rotated_surface_d5/",
            "golay": prefix / "golay/",
            "carbon": prefix / "carbon/",
            "hamming": prefix / "hamming_15/",
        }

        distances = {
            "steane": (3, 3),
            "tetrahedral": (7, 3),
            "shor": (3, 3),
            "golay": (7, 7),
            "surface_3": (3, 3),
            "surface_5": (5, 5),
            "carbon": (4, 4),
            "hamming": (3, 3),
        }  # X, Z distances

        code_name = code_name.lower()
        if code_name == "surface":
            if distance is None:
                distance = 3
            code_name += f"_{distance}"

        if code_name in paths:
            hx = np.load(paths[code_name] / "hx.npy")
            hz = np.load(paths[code_name] / "hz.npy")

            if code_name in distances:
                x_distance, z_distance = distances[code_name]
                distance = min(x_distance, z_distance)
                return CSSCode(hx, hz, distance, x_distance=x_distance, z_distance=z_distance)

            if distance is None:
                msg = f"Distance is not specified for {code_name}"
                raise InvalidCSSCodeError(msg)
        msg = f"Unknown code name: {code_name}"
        raise InvalidCSSCodeError(msg)

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

    def _normalize_logicals(self) -> None:
        """Normalize the logical operators.

        The basis of logical operators computed by `_compute_logical` do not necessarily
        represent single-logical qubit operators but might be product operators.
        This method normalizes the logical operators such that each logical qubit
        is represented by a pair of logical X and Z operators that anti-commute
        only with each other, i.e. Lx[i]@Lz[j].T = 1 if i == j else 0.
        Assumes that `_compute_logical` has already been called to compute `Lx` and `Lz`.
        """
        k = self.Lx.shape[0]
        assert k == self.Lz.shape[0], "Number of X and Z logicals must be the same."

        for i in range(self.k):
            xl = self.Lx[i]
            anticommute_x = np.where(xl @ self.Lz.T % 2)[0]
            first = anticommute_x[0]
            zl = self.Lz[first]
            anticommute_z = np.where(zl @ self.Lx.T % 2)[0]
            for j in anticommute_x[1:]:
                self.Lz[j] ^= zl

            for j in anticommute_z:
                if j != i:
                    self.Lx[j] ^= xl
            self.Lz[[i, first]] = self.Lz[[first, i]]

    def set_x_logicals(self, logicals: npt.NDArray[np.int8]) -> None:
        """Set all X logical operators."""
        if logicals.shape[0] != self.k:
            msg = f"Number of logicals {logicals.shape[0]} does not match k={self.k}"
            raise InvalidCSSCodeError(msg)

        commutes = (self.Hz @ logicals.T) % 2
        if np.any(commutes != 0):
            msg = "Logical operators must commute with the Z stabilizers"
            raise InvalidCSSCodeError(msg)

        self.Lx = logicals.copy()
        self.x_logicals = StabilizerTableau.from_check_matrix(CheckMatrix(self.Lx, pauli_type="X"))

    def set_z_logicals(self, logicals: npt.NDArray[np.int8]) -> None:
        """Set all Z logical operators."""
        if logicals.shape[0] != self.k:
            msg = f"Number of logicals {logicals.shape[0]} does not match k={self.k}"
            raise InvalidCSSCodeError(msg)

        commutes = (self.Hx @ logicals.T) % 2
        if np.any(commutes != 0):
            msg = "Logical operators must commute with the X stabilizers"
            raise InvalidCSSCodeError(msg)

        self.Lz = logicals.copy()
        self.z_logicals = StabilizerTableau.from_check_matrix(CheckMatrix(self.Lz, pauli_type="Z"))


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

            row = [int(t) for t in tokens if t in {"0", "1"}]
            if row:
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

        row = [int(t) for t in tokens if t in {"0", "1"}]
        if row:
            rows.append(row)

    if not rows:
        msg = "No valid binary matrix data found in section"
        raise InvalidCSSCodeError(msg)

    return np.array(rows, dtype=np.int8)


class InvalidCSSCodeError(ValueError):
    """Raised when the CSS code is invalid."""
