# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
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

from mqt.qecc.mod2 import is_in_row_space, nullspace, rank

from .pauli import Pauli, PauliTableau
from .symplectic import symplectic_product

if TYPE_CHECKING:
    import numpy.typing as npt


class StabilizerCode:
    """A class for representing stabilizer codes."""

    def __init__(
        self,
        generators: PauliTableau | list[Pauli] | list[str],
        distance: int | None = None,
        z_logicals: PauliTableau | list[Pauli] | list[str] | None = None,
        x_logicals: PauliTableau | list[Pauli] | list[str] | None = None,
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
        # check for non-trivial stabilizers before computing log. operators for a more informative error message
        self._check_nontrivial_stabilizer()

        if n is None:
            self.n = self.generators.n
        else:
            self.n = n

        if self.generators.n_rows != 0:
            self.k = self.n - rank(self.generators.symplectic)
        else:
            self.k = self.n

        if distance is not None and distance <= 0:
            msg = "Distance must be a positive integer."
            raise InvalidStabilizerCodeError(msg)

        self.distance = 1 if distance is None else distance  # default distance is 1
        if z_logicals is None and x_logicals is None:
            l_x, l_z = self.compute_logical_ops()
            self.x_logicals = self.get_generators(l_x, self.n)
            self.z_logicals = self.get_generators(l_z, self.n)
        elif z_logicals is not None and x_logicals is not None:
            self.z_logicals = self.get_generators(z_logicals, self.n)
            self.x_logicals = self.get_generators(x_logicals, self.n)
        else:
            msg = "Both z_logicals and x_logicals must be provided together or both must be None."
            raise InvalidStabilizerCodeError(msg)

        self._check_logicals_correct()

    @property
    def symplectic(self) -> npt.NDArray[np.int8]:
        """The binary symplectic matrix of the stabilizer generators."""
        return self.generators.symplectic

    def __hash__(self) -> int:
        """Compute a hash for the stabilizer code."""
        return hash((self.generators, tuple(self.z_logicals), tuple(self.x_logicals)))

    def __eq__(self, other: object) -> bool:
        """Check if two stabilizer codes are equal.

        Two codes are equal if their generator matrices, logical X matrices,
        and logical Z matrices are identical (element-wise comparison).
        For semantic equivalence (same stabilizer group and logical basis),
        use the is_equivalent method instead.
        """
        if not isinstance(other, StabilizerCode):
            return NotImplemented
        return (
            np.array_equal(self.generators.as_matrix(), other.generators.as_matrix())
            and np.array_equal(self.x_logicals.as_matrix(), other.x_logicals.as_matrix())
            and np.array_equal(self.z_logicals.as_matrix(), other.z_logicals.as_matrix())
        )

    def equal_stabilizer_group(self, other: StabilizerCode) -> bool:
        """Check if two stabilizer codes have the same stabilizer group.

        The comparison is over the symplectic supports of the generators;
        generator signs are ignored. To include signs, there has to be a sign-aware matrix multiplication for PauliTableaus.
        """
        if self.n != other.n:
            return False

        self_matrix = self.generators.symplectic
        other_matrix = other.generators.symplectic
        stabs_rnk = rank(self_matrix)

        return bool(stabs_rnk == rank(other_matrix) and stabs_rnk == rank(np.vstack((self_matrix, other_matrix))))

    def equal_logical_basis(self, other: StabilizerCode) -> bool:
        """Check if two stabilizer codes have the same logical basis."""
        if len(self.z_logicals) != len(other.z_logicals) or len(self.x_logicals) != len(other.x_logicals):
            return False

        for lz_other in other.z_logicals:
            if not self.is_z_logical(lz_other):
                return False

        return all(self.is_x_logical(lx_other) for lx_other in other.x_logicals)

    def is_equivalent(self, other: StabilizerCode) -> bool:
        """Check if two stabilizer codes are equivalent.

        Two codes are equivalent if they have the same stabilizer group
        and the same logical basis (up to stabilizer equivalence).
        This is a semantic comparison, unlike __eq__ which checks
        for exact matrix equality.
        """
        return self.equal_stabilizer_group(other) and self.equal_logical_basis(other)

    def is_z_logical(self, p: Pauli | str) -> bool:
        """Check if a given Pauli string is a logical Z operator of the code."""
        if isinstance(p, str):
            p = Pauli.from_pauli_string(p)

        return any(self.stabilizer_equivalent(p, logical) for logical in self.z_logicals)

    def is_x_logical(self, p: Pauli | str) -> bool:
        """Check if a given Pauli string is a logical X operator of the code."""
        if isinstance(p, str):
            p = Pauli.from_pauli_string(p)

        return any(self.stabilizer_equivalent(p, logical) for logical in self.x_logicals)

    def get_logical_mapping(self, other: StabilizerCode) -> list[int] | None:
        """Get a mapping of logical qubits from this code to another code, if it exists.

        Returns:
            A list of integers where the i-th element is the index of the logical qubit in the other code that corresponds to the i-th logical qubit in this code. Returns None if no such mapping exists.
        """
        if self.k != other.k:
            return None

        if not self.equal_logical_basis(other):
            return None

        available_indices = set(range(len(other.z_logicals)))
        mapping = []

        for lz_self, lx_self in zip(self.z_logicals, self.x_logicals, strict=False):
            found_match = False
            for idx in available_indices.copy():
                lz_other = other.z_logicals[idx]
                lx_other = other.x_logicals[idx]
                if self.stabilizer_equivalent(lz_self, lz_other) and self.stabilizer_equivalent(lx_self, lx_other):
                    mapping.append(idx)
                    available_indices.remove(idx)
                    found_match = True
                    break
            if not found_match:
                return None

        return mapping

    def is_logical(self, p: Pauli | str) -> bool:
        """Check if a given Pauli string is a logical operator of the code."""
        return self.is_z_logical(p) or self.is_x_logical(p)

    def get_syndrome(self, error: Pauli | str) -> npt.NDArray[np.int8]:
        """Compute the syndrome of the error.

        Args:
            error: The error as a pauli string or binary vector.
        """
        if isinstance(error, str):
            error = Pauli.from_pauli_string(error)
        return np.asarray(symplectic_product(self.generators.symplectic, error.symplectic.vector), dtype=np.int8)

    def stabs_as_pauli_strings(self) -> list[str]:
        """Return the stabilizers as Pauli strings."""
        return [str(p) for p in self.generators]

    def stabilizer_equivalent(self, p1: Pauli | str, p2: Pauli | str) -> bool:
        """Check if two Pauli strings are equivalent up to stabilizers of the code.

        The comparison is over binary symplectic supports; phases are ignored.
        """
        if isinstance(p1, str):
            p1 = Pauli.from_pauli_string(p1)
        if isinstance(p2, str):
            p2 = Pauli.from_pauli_string(p2)
        difference = (p1.symplectic.vector + p2.symplectic.vector) % 2
        return is_in_row_space(difference, self.generators.symplectic)

    def to_tableau(self) -> PauliTableau:
        """Convert the stabilizer code to a tableau with logicals and stabilizers.

        Returns:
            PauliTableau: A tableau with rows ordered as:
                - X logical operators
                - Z logical operators
                - Stabilizer generators
        """
        x_log_matrix = self.x_logicals.tableau.matrix
        z_log_matrix = self.z_logicals.tableau.matrix
        stab_matrix = self.generators.tableau.matrix

        combined_matrix = np.vstack([x_log_matrix, z_log_matrix, stab_matrix])

        x_log_phases = self.x_logicals.phase
        z_log_phases = self.z_logicals.phase
        stab_phases = self.generators.phase

        combined_phases = np.concatenate([x_log_phases, z_log_phases, stab_phases])

        return PauliTableau(combined_matrix, combined_phases)

    def _check_nontrivial_stabilizer(self) -> None:
        """Check if the stabilizer generators are non-trivial. Throws an exception if not."""
        stabilizer_commutations = symplectic_product(self.generators.symplectic, self.generators.symplectic)
        if not np.all(stabilizer_commutations == 0):
            msg = "Stabilizer generators must commute with each other."
            raise InvalidStabilizerCodeError(msg)

    def _check_logicals_correct(self) -> None:
        """Check if the code is correct. Throws an exception if not."""
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

        if self.z_logicals.shape[0] != self.x_logicals.shape[0]:
            msg = "Number of logical Z-operators must be equal to the number of logical X-operators."
            raise InvalidStabilizerCodeError(msg)

        if not self.z_logicals.all_commute(self.generators):
            msg = "Logical Z-operators must commute with the stabilizer generators."
            raise InvalidStabilizerCodeError(msg)
        if not self.x_logicals.all_commute(self.generators):
            msg = "Logical X-operators must commute with the stabilizer generators."
            raise InvalidStabilizerCodeError(msg)

        commutations = symplectic_product(self.z_logicals.symplectic, self.x_logicals.symplectic)
        if not np.array_equal(commutations, np.eye(self.z_logicals.shape[0], dtype=np.int8)):
            msg = "Every logical X-operator must anti-commute with exactly one logical Z-operator and vice versa."
            raise InvalidStabilizerCodeError(msg)

    @staticmethod
    def get_generators(
        generators: PauliTableau | list[Pauli] | list[str],
        n: int | None = None,
    ) -> PauliTableau:
        """Get the stabilizer generators as a PauliTableau object.

        Args:
            generators: The stabilizer generators as a PauliTableau object, a list of Pauli objects, or a list of Pauli strings.
            n: The number of qubits in the code. Required if generators is an empty list.
        """
        if isinstance(generators, list):
            if len(generators) == 0:
                if n is None:
                    msg = "Number of qubits must be given if no generators are provided."
                    raise ValueError(msg)
                return PauliTableau.empty(n)
            if isinstance(generators[0], str):
                return PauliTableau.from_pauli_strings(generators)  # ty: ignore[invalid-argument-type]
            if isinstance(generators[0], Pauli):
                return PauliTableau.from_paulis(generators)  # ty: ignore[invalid-argument-type]
        assert isinstance(generators, PauliTableau)
        return generators

    @classmethod
    def get_trivial_code(cls, n: int) -> StabilizerCode:
        """Get the trivial stabilizer code."""
        z_logicals: list[str] = ["I" * i + "Z" + "I" * (n - i - 1) for i in range(n)]
        x_logicals: list[str] = ["I" * i + "X" + "I" * (n - i - 1) for i in range(n)]
        return StabilizerCode([], distance=1, z_logicals=z_logicals, x_logicals=x_logicals, n=n)

    def compute_logical_ops(self) -> tuple[PauliTableau, PauliTableau]:
        """Compute logical Z/X operators from stabilizer generators.

        Finds a basis of the centralizer modulo the stabilizers and canonicalizes it
        into k symplectic pairs (Z̄_i, X̄_i).

        Returns:
            The tuple (L_x, L_z) of logical X and Z operators as PauliTableau objects, ordered such that rows i represent X̄_i and Z̄_i.
        """
        if self.generators.n_rows == 0:
            z_logicals = PauliTableau.from_pauli_strings([
                "I" * i + "Z" + "I" * (self.n - i - 1) for i in range(self.n)
            ])
            x_logicals = PauliTableau.from_pauli_strings([
                "I" * i + "X" + "I" * (self.n - i - 1) for i in range(self.n)
            ])
            return x_logicals, z_logicals

        n = self.n
        mat = self.generators.symplectic

        identity = np.eye(n, dtype=np.int8)
        z0 = np.zeros((n, n), dtype=np.int8)
        lamb = np.block([[z0, identity], [identity, z0]])

        mat_times_lamb = (mat @ lamb) % 2
        ns = nullspace(mat_times_lamb).astype(np.int8)

        if ns.size == 0:
            if self.k > 0:
                msg = f"Cannot compute logical operators: nullspace is empty (ns.size=0) but code has k={self.k} logical qubits. This would result in empty z_logicals and x_logicals, creating an inconsistent code state."
                raise InvalidStabilizerCodeError(msg)
            return PauliTableau.empty(n), PauliTableau.empty(n)

        base = mat.copy()
        base_rank = rank(base)
        target = 2 * self.k

        logical_basis: list[np.ndarray] = []
        for v in ns:
            if len(logical_basis) == target:
                break
            test = np.vstack((base, v))
            if rank(test) > base_rank:
                logical_basis.append(v.copy())
                base = test
                base_rank += 1

        logical_basis_arr = np.array(logical_basis, dtype=np.int8)

        logs = logical_basis_arr.copy()
        vecs = [logs[i].copy() for i in range(logs.shape[0])]
        zs: list[np.ndarray] = []
        xs: list[np.ndarray] = []

        def ortho_against_pairs(a: np.ndarray) -> np.ndarray:
            a = a.copy()
            for zp, xp in zip(zs, xs, strict=False):
                if symplectic_product(a, xp):
                    a ^= zp
                if symplectic_product(a, zp):
                    a ^= xp
            return a

        used: set[int] = set()
        k = self.k
        i = 0
        while len(zs) < k and i < len(vecs):
            if i in used:
                i += 1
                continue
            v = ortho_against_pairs(vecs[i])
            maybe_j = None
            for t in range(len(vecs)):
                if t == i or t in used:
                    continue
                w_cand = ortho_against_pairs(vecs[t])
                if symplectic_product(v, w_cand) == 1:
                    maybe_j = t
                    w = w_cand
                    break
            if maybe_j is None:
                i += 1
                continue
            j = maybe_j

            v = ortho_against_pairs(v)
            w = ortho_against_pairs(w)
            if symplectic_product(v, w) != 1:
                i += 1
                continue

            for t in range(len(vecs)):
                if t in {i, j} or t in used:
                    continue
                u = vecs[t]
                if symplectic_product(u, w):
                    u ^= v
                if symplectic_product(u, v):
                    u ^= w
                vecs[t] = u

            zs.append(v)
            xs.append(w)
            used.add(i)
            used.add(j)
            i += 1

        if len(zs) != k or len(xs) != k:
            msg = f"Failed to derive {k} logical X/Z pairs from the stabilizer generators."
            raise InvalidStabilizerCodeError(msg)

        z_mat = np.vstack(zs).astype(np.int8) if k > 0 else np.zeros((0, 2 * n), dtype=np.int8)
        x_mat = np.vstack(xs).astype(np.int8) if k > 0 else np.zeros((0, 2 * n), dtype=np.int8)
        return PauliTableau(x_mat, np.zeros((k,), dtype=np.int8)), PauliTableau(z_mat, np.zeros((k,), dtype=np.int8))

    @classmethod
    def from_file(cls, file_path: str | Path) -> StabilizerCode:
        r"""Load a Stabilizer code from a file.

        The file can contain either:
        1. One line per stabilizer generator as a Pauli string (e.g., "IXXZZ")
        2. A binary symplectic matrix (2nxm) where m is the number of stabilizers

        For the 5-qubit perfect code in Pauli string format:
        IXXZZ
        ZIXXZ
        ZZIXX
        XZZIX

        For binary symplectic matrix format, any of these are accepted:
        - Space-separated: "1 0 0 0 0 ..."
        - Comma-separated: "1,0,0,0,0,..."
        - List notation: "[[1,0,0,...],[0,1,0,...]]"
        - NumPy array notation: "[[1 0 0 ...]\n [0 1 0 ...]]"

        Args:
            file_path: The path to the file containing the code.

        Returns:
            StabilizerCode: The stabilizer code.
        """
        content = Path(file_path).read_text(encoding="utf-8").strip()

        if not content:
            msg = "File is empty"
            raise InvalidStabilizerCodeError(msg)

        if _is_binary_matrix_format(content):
            return _load_from_binary_matrix(content)

        lines = content.split("\n")
        generators = [line.strip() for line in lines if line.strip()]
        return cls(generators)

    def is_stabilizer(self, p: Pauli | str) -> bool:
        """Check if a given Pauli string is a stabilizer of the code."""
        if isinstance(p, str):
            p = Pauli.from_pauli_string(p)
        return self.stabilizer_equivalent(p, Pauli.from_pauli_string("I" * self.n))

    def __repr__(self) -> str:
        """Return a string representation of the code."""
        return f"StabilizerCode(n={self.n}, k={self.k}, distance={self.distance})"

    def __str__(self) -> str:
        """Return a human-readable string representation of the code."""
        lines = [f"Stabilizer Code: n={self.n}, k={self.k}, distance={self.distance}"]
        lines.append("Stabilizer Generators:")
        lines.extend(f"  {gen}" for gen in self.generators)
        lines.append("Logical Z operators:")
        lines.extend(f"  {z}" for z in self.z_logicals)
        lines.append("Logical X operators:")
        lines.extend(f"  {x}" for x in self.x_logicals)
        return "\n".join(lines)


def _is_binary_matrix_format(content: str) -> bool:
    """Check if the content appears to be a binary matrix format.

    Args:
        content: The file content to check.

    Returns:
        True if the content looks like a binary matrix, False otherwise.
    """
    content = content.strip()

    if not content:
        return False

    if content.startswith(("[[", "[")):
        return True

    lines = content.split("\n")
    non_empty_lines = [line.strip() for line in lines if line.strip()]

    if not non_empty_lines:
        return False

    first_line = non_empty_lines[0]
    first_tokens = [t.strip() for t in first_line.split(",")] if "," in first_line else first_line.split()

    if len(first_tokens) < 2:
        return False

    if not all(token in {"0", "1"} for token in first_tokens):
        return False

    if len(non_empty_lines) > 1:
        second_line = non_empty_lines[1]
        second_tokens = [t.strip() for t in second_line.split(",")] if "," in second_line else second_line.split()

        if len(second_tokens) != len(first_tokens):
            return False

        if not all(token in {"0", "1"} for token in second_tokens):
            return False

    return True


def _parse_bracketed_rows(content: str) -> list[str]:
    """Parse bracketed content into individual row strings.

    Args:
        content: The bracketed content (e.g., "[[1,0],[0,1]]" or nested format).

    Returns:
        List of row strings, each representing one row of the matrix.
    """
    depth = 0
    current_row: list[str] = []
    rows = []

    for char in content:
        if char == "[":
            depth += 1
            if depth == 2:
                current_row = []
        elif char == "]":
            depth -= 1
            if depth == 1 and current_row:
                rows.append("".join(current_row))
                current_row = []
        elif depth == 2:
            current_row.append(char)

    return rows


def _load_from_binary_matrix(content: str) -> StabilizerCode:
    """Load a stabilizer code from binary symplectic matrix format.

    Args:
        content: The file content containing the binary matrix.

    Returns:
        StabilizerCode: The stabilizer code.
    """
    content = content.strip()

    if content.startswith(("[[", "[")):
        row_strings = _parse_bracketed_rows(content)
    else:
        row_strings = [line.strip() for line in content.split("\n") if line.strip()]

    rows = []
    for line_idx, line in enumerate(row_strings):
        tokens = [t.strip() for t in line.split(",") if t.strip()] if "," in line else line.split()

        for token in tokens:
            if token not in {"0", "1"}:
                msg = f"Invalid token '{token}' in binary matrix at line {line_idx + 1}: expected '0' or '1'"
                raise InvalidStabilizerCodeError(msg)

        row = [int(t) for t in tokens]
        if row:
            rows.append(row)

    if not rows:
        msg = "No valid binary matrix data found in file"
        raise InvalidStabilizerCodeError(msg)

    row_lengths = {len(row) for row in rows}
    if len(row_lengths) != 1:
        msg = "Binary matrix rows must all have the same length."
        raise InvalidStabilizerCodeError(msg)

    matrix = np.array(rows, dtype=np.int8)

    if matrix.shape[1] % 2 != 0:
        msg = f"Binary matrix must have an even number of columns (2n), got {matrix.shape[1]}"
        raise InvalidStabilizerCodeError(msg)

    n = matrix.shape[1] // 2
    m = matrix.shape[0]

    generators = []
    for row_idx in range(m):
        x_part = matrix[row_idx, :n]
        z_part = matrix[row_idx, n:]

        pauli_str = ""
        for i in range(n):
            x_val = x_part[i]
            z_val = z_part[i]

            if x_val == 0 and z_val == 0:
                pauli_str += "I"
            elif x_val == 1 and z_val == 0:
                pauli_str += "X"
            elif x_val == 0 and z_val == 1:
                pauli_str += "Z"
            elif x_val == 1 and z_val == 1:
                pauli_str += "Y"

        generators.append(pauli_str)

    return StabilizerCode(generators)


class InvalidStabilizerCodeError(ValueError):
    """Raised when the stabilizer code is invalid."""
