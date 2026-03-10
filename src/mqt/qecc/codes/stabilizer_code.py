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
from ldpc.mod2.mod2_numpy import nullspace, rank

from .pauli import Pauli, StabilizerTableau

if TYPE_CHECKING:
    import numpy.typing as npt

    from mqt.qecc.codes.symplectic import SymplecticVector


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
            self.k = self.n - rank(self.generators.as_matrix())
        else:
            self.k = self.n

        if distance is not None and distance <= 0:
            msg = "Distance must be a positive integer."
            raise InvalidStabilizerCodeError(msg)

        self.distance = 1 if distance is None else distance  # default distance is 1
        self.compute_logical_ops()
        if z_logicals is not None:
            self.z_logicals = self.get_generators(z_logicals)
        if x_logicals is not None:
            self.x_logicals = self.get_generators(x_logicals)

        self._check_code_correct()

    def __hash__(self) -> int:
        """Compute a hash for the stabilizer code."""
        return hash((self.generators, tuple(self.z_logicals), tuple(self.x_logicals)))

    def __eq__(self, other: object) -> bool:
        """Check if two stabilizer codes are equal."""
        if not isinstance(other, StabilizerCode):
            return NotImplemented
        self_matrix = self.generators.as_matrix()
        other_matrix = other.generators.as_matrix()
        stabs_rnk = rank(self_matrix)

        if not (stabs_rnk == rank(other_matrix) and stabs_rnk == rank(np.vstack((self_matrix, other_matrix)))):
            return False

        if len(self.z_logicals) != len(other.z_logicals) or len(self.x_logicals) != len(other.x_logicals):
            return False

        for lz_self, lz_other in zip(self.z_logicals, other.z_logicals, strict=False):
            if not self.stabilizer_equivalent(lz_self, lz_other):
                return False

        for lx_self, lx_other in zip(self.x_logicals, other.x_logicals, strict=False):
            if not self.stabilizer_equivalent(lx_self, lx_other):
                return False

        return True

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
        vector: SymplecticVector = self.generators.tableau @ error.symplectic
        return vector.vector

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
            rank(np.vstack((self.generators.as_matrix(), p1.as_vector(), p2.as_vector())))
            == rank(np.vstack((self.generators.as_matrix(), p1.as_vector())))
        )

    def to_tableau(self) -> StabilizerTableau:
        """Convert the stabilizer code to a tableau with logicals and stabilizers.

        Returns:
            StabilizerTableau: A tableau with rows ordered as:
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

        return StabilizerTableau(combined_matrix, combined_phases)

    def _check_code_correct(self) -> None:
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

        stabilizer_commutations = (self.generators.tableau @ self.generators.tableau).matrix
        if not np.all(stabilizer_commutations == 0):
            msg = "Stabilizer generators must commute with each other."
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

    def compute_logical_ops(self) -> None:
        """Compute logical Z/X operators from stabilizer generators.

        Finds a basis of the centralizer modulo the stabilizers and canonicalizes it
        into k symplectic pairs (Z̄_i, X̄_i). Stores results in self.z_logicals/self.x_logicals.
        """
        if self.generators.n_rows == 0:
            self.z_logicals = StabilizerTableau.from_pauli_strings([
                "I" * i + "Z" + "I" * (self.n - i - 1) for i in range(self.n)
            ])
            self.x_logicals = StabilizerTableau.from_pauli_strings([
                "I" * i + "X" + "I" * (self.n - i - 1) for i in range(self.n)
            ])
            return

        n = self.n
        mat = self.generators.tableau.matrix
        r = mat.shape[0]

        identity = np.eye(n, dtype=np.int8)
        z0 = np.zeros((n, n), dtype=np.int8)
        lamb = np.block([[z0, identity], [identity, z0]])

        mat_times_lamb = (mat @ lamb) % 2
        ns = nullspace(mat_times_lamb).astype(np.int8)

        if ns.size == 0:
            self.z_logicals = StabilizerTableau.empty(n)
            self.x_logicals = StabilizerTableau.empty(n)
            return

        def mod2_rank(mat: np.ndarray) -> int:
            return int(rank(mat % 2))

        base = mat.copy()
        base_rank = mod2_rank(base)
        target = 2 * (n - r)

        logical_basis: list[np.ndarray] = []
        for v in ns:
            if len(logical_basis) == target:
                break
            test = np.vstack((base, v))
            if mod2_rank(test) > base_rank:
                logical_basis.append(v.copy())
                base = test
                base_rank += 1

        logical_basis_arr = np.array(logical_basis, dtype=np.int8)

        if logical_basis_arr.shape[0] != target:
            rows = ns.shape[0]
            extended_basis = base.copy()
            extended_basis_rank = base_rank
            for i in range(rows):
                for j in range(i + 1, rows):
                    v = (ns[i] ^ ns[j]) % 2
                    if len(logical_basis_arr) == target:
                        break
                    test = np.vstack((extended_basis, v))
                    if mod2_rank(test) > extended_basis_rank:
                        logical_basis_arr = np.vstack((logical_basis_arr, v.copy))
                        extended_basis = test
                        extended_basis_rank += 1
                if len(logical_basis_arr) == target:
                    break
            logical_basis_arr = np.array(logical_basis_arr, dtype=np.int8)

        def symp(u: np.ndarray, v: np.ndarray) -> int:
            return int((u[:n] @ v[n:] + u[n:] @ v[:n]) % 2)

        logs = logical_basis_arr.copy()
        vecs = [logs[i].copy() for i in range(logs.shape[0])]
        zs: list[np.ndarray] = []
        xs: list[np.ndarray] = []

        def ortho_against_pairs(a: np.ndarray) -> np.ndarray:
            a = a.copy()
            for zp, xp in zip(zs, xs, strict=False):
                if symp(a, xp):
                    a ^= zp
                if symp(a, zp):
                    a ^= xp
            return a

        used: set[int] = set()
        k = n - r
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
                if symp(v, w_cand) == 1:
                    maybe_j = t
                    w = w_cand
                    break
            if maybe_j is None:
                i += 1
                continue
            j = maybe_j

            v = ortho_against_pairs(v)
            w = ortho_against_pairs(w)
            if symp(v, w) != 1:
                i += 1
                continue

            for t in range(len(vecs)):
                if t in {i, j} or t in used:
                    continue
                u = vecs[t]
                if symp(u, w):
                    u ^= v
                if symp(u, v):
                    u ^= w
                vecs[t] = u

            zs.append(v)
            xs.append(w)
            used.add(i)
            used.add(j)
            i += 1

        if len(zs) != k or len(xs) != k:
            self.z_logicals = StabilizerTableau.empty(n)
            self.x_logicals = StabilizerTableau.empty(n)
            return

        z_mat = np.vstack(zs).astype(np.int8) if k > 0 else np.zeros((0, 2 * n), dtype=np.int8)
        x_mat = np.vstack(xs).astype(np.int8) if k > 0 else np.zeros((0, 2 * n), dtype=np.int8)
        self.z_logicals = StabilizerTableau(z_mat, np.zeros((k,), dtype=np.int8))
        self.x_logicals = StabilizerTableau(x_mat, np.zeros((k,), dtype=np.int8))

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
    if not lines:
        return False

    first_line = lines[0].strip()

    if not first_line:
        return False

    tokens = [t.strip() for t in first_line.split(",")] if "," in first_line else first_line.split()

    if len(tokens) < 2:
        return False

    return all(token in {"0", "1"} for token in tokens[:5])


def _load_from_binary_matrix(content: str) -> StabilizerCode:
    """Load a stabilizer code from binary symplectic matrix format.

    Args:
        content: The file content containing the binary matrix.

    Returns:
        StabilizerCode: The stabilizer code.
    """
    content = content.strip()

    if content.startswith(("[[", "[")):
        content = content.replace("[", "").replace("]", "")

    lines = [line.strip() for line in content.split("\n") if line.strip()]

    rows = []
    for line in lines:
        tokens = [t.strip() for t in line.split(",") if t.strip()] if "," in line else line.split()

        row = [int(t) for t in tokens if t in {"0", "1"}]
        if row:
            rows.append(row)

    if not rows:
        msg = "No valid binary matrix data found in file"
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
