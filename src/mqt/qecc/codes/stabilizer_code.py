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
        self.compute_logical()
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

    def compute_logical(self) -> None:
        """Compute logical Z/X operators from stabilizer generators.

        Finds a basis of the centralizer modulo the stabilizers and canonicalizes it
        into k symplectic pairs (Z̄_i, X̄_i). Stores results in self.z_logicals/self.x_logicals.
        """
        # Trivial code (no stabilizers): identity logicals
        if self.generators.n_rows == 0:
            self.z_logicals = StabilizerTableau.from_pauli_strings(
                ["I"*i + "Z" + "I"*(self.n - i - 1) for i in range(self.n)]
            )
            self.x_logicals = StabilizerTableau.from_pauli_strings(
                ["I"*i + "X" + "I"*(self.n - i - 1) for i in range(self.n)]
            )
            return

        import numpy as np  # local import for safety

        n = self.n
        M = self.generators.tableau.matrix  # (r x 2n) over GF(2)
        r = M.shape[0]

        # Symplectic form Λ = [[0,I],[I,0]]
        I = np.eye(n, dtype=np.int8)
        Z0 = np.zeros((n, n), dtype=np.int8)
        Lambda = np.block([[Z0, I], [I, Z0]])

        # Centralizer: nullspace of M Λ
        B = (M @ Lambda) % 2
        C = mod2.nullspace(B).astype(np.int8)  # rows span centralizer

        if C.size == 0:
            self.z_logicals = StabilizerTableau.empty(n)
            self.x_logicals = StabilizerTableau.empty(n)
            return

        def mod2_rank(A: np.ndarray) -> int:
            return int(mod2.rank(A % 2))

        # Build basis of the quotient C / span(M) with size 2k = 2(n - r)
        base = M.copy()
        base_rank = mod2_rank(base)
        target = 2 * (n - r)

        logical_basis: list[np.ndarray] = []
        for v in C:
            if len(logical_basis) == target:
                break
            test = np.vstack((base, v))
            if mod2_rank(test) > base_rank:
                logical_basis.append(v.copy())
                base = test
                base_rank += 1

        logical_basis = np.array(logical_basis, dtype=np.int8)

        # Fallback: try simple pairwise sums to complete to 2k (rarely needed)
        if logical_basis.shape[0] != target:
            rows = C.shape[0]
            used = base.copy()
            used_rank = base_rank
            for i in range(rows):
                for j in range(i + 1, rows):
                    v = (C[i] ^ C[j]) % 2
                    if len(logical_basis) == target:
                        break
                    test = np.vstack((used, v))
                    if mod2_rank(test) > used_rank:
                        logical_basis.append(v.copy())
                        used = test
                        used_rank += 1
                if len(logical_basis) == target:
                    break
            logical_basis = np.array(logical_basis, dtype=np.int8)

        def symp(u: np.ndarray, v: np.ndarray) -> int:
            return int((u[:n] @ v[n:] + u[n:] @ v[:n]) % 2)

        # Symplectic Gram–Schmidt to obtain k pairs (Z_i, X_i)
        L = logical_basis.copy()
        vecs = [L[i].copy() for i in range(L.shape[0])]
        Zs: list[np.ndarray] = []
        Xs: list[np.ndarray] = []

        def ortho_against_pairs(a: np.ndarray) -> np.ndarray:
            a = a.copy()
            for Zp, Xp in zip(Zs, Xs):
                if symp(a, Xp):
                    a ^= Zp
                if symp(a, Zp):
                    a ^= Xp
            return a

        used: set[int] = set()
        k = n - r
        i = 0
        while len(Zs) < k and i < len(vecs):
            if i in used:
                i += 1
                continue
            v = ortho_against_pairs(vecs[i])
            # Find partner w with symp(v, w) = 1
            j = None
            for t in range(len(vecs)):
                if t == i or t in used:
                    continue
                w_cand = ortho_against_pairs(vecs[t])
                if symp(v, w_cand) == 1:
                    j = t
                    w = w_cand
                    break
            if j is None:
                i += 1
                continue

            v = ortho_against_pairs(v)
            w = ortho_against_pairs(w)
            if symp(v, w) != 1:
                i += 1
                continue

            # Clean remaining vectors to commute with the new pair
            for t in range(len(vecs)):
                if t == i or t == j or t in used:
                    continue
                u = vecs[t]
                if symp(u, w):
                    u ^= v
                if symp(u, v):
                    u ^= w
                vecs[t] = u

            Zs.append(v)
            Xs.append(w)
            used.add(i)
            used.add(j)
            i += 1

        if len(Zs) != k or len(Xs) != k:
            self.z_logicals = StabilizerTableau.empty(n)
            self.x_logicals = StabilizerTableau.empty(n)
            return

        z_mat = np.vstack(Zs).astype(np.int8) if k > 0 else np.zeros((0, 2 * n), dtype=np.int8)
        x_mat = np.vstack(Xs).astype(np.int8) if k > 0 else np.zeros((0, 2 * n), dtype=np.int8)
        self.z_logicals = StabilizerTableau(z_mat, np.zeros((k,), dtype=np.int8))
        self.x_logicals = StabilizerTableau(x_mat, np.zeros((k,), dtype=np.int8))


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
