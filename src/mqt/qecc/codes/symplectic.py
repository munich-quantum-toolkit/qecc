# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Classes and Methods for working with symplectic vector spaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# from ldpc import mod2

if TYPE_CHECKING:
    from typing import Any

    import numpy.typing as npt


class SymplecticVector:
    """Symplectic Vector Class."""

    def __init__(self, vector: npt.NDArray[np.int8]) -> None:
        """Initialize the Symplectic Vector."""
        self.vector = vector
        self.n = vector.shape[0] // 2

    @classmethod
    def zeros(cls, n: int) -> SymplecticVector:
        """Create a zero vector of length n."""
        return cls(np.zeros(2 * n, dtype=np.int8))

    @classmethod
    def ones(cls, n: int) -> SymplecticVector:
        """Create a ones vector of length n."""
        return cls(np.ones(2 * n, dtype=np.int8))

    def __add__(self, other: SymplecticVector) -> SymplecticVector:
        """Add two symplectic vectors."""
        return SymplecticVector((self.vector + other.vector) % 2)

    def __sub__(self, other: SymplecticVector) -> SymplecticVector:
        """Subtract two symplectic vectors."""
        return SymplecticVector((self.vector - other.vector) % 2)

    def __neg__(self) -> SymplecticVector:
        """Negate the vector."""
        return SymplecticVector(-self.vector)

    def __matmul__(self, other: SymplecticVector) -> int:
        """Compute the symplectic inner product."""
        assert self.n == other.n, "Vectors must be of the same length."
        return int(
            (self.vector[: self.n] @ other.vector[self.n :] - self.vector[self.n :] @ other.vector[: self.n]) % 2
        )

    def __getitem__(self, key: int | slice) -> Any:  # noqa: ANN401
        """Get the value of the vector at index key."""
        return self.vector[key]

    def __setitem__(self, key: int | slice, value: int) -> None:
        """Set the value of the vector at index key."""
        self.vector[key] = value

    def __eq__(self, other: object) -> bool:
        """Check if two vectors are equal."""
        if not isinstance(other, SymplecticVector):
            return False
        return np.array_equal(self.vector, other.vector)

    def __ne__(self, other: object) -> bool:
        """Check if two vectors are not equal."""
        return not self == other

    def __hash__(self) -> int:
        """Return the hash of the vector."""
        return hash(self.vector.to_bytes())


class SymplecticMatrix:
    """Symplectic Matrix Class."""

    def __init__(self, matrix: npt.NDArray[np.int8]) -> None:
        """Initialize the Symplectic Matrix."""
        assert matrix.ndim == 2, "Matrix must be 2D."
        self.matrix = matrix
        self.n = matrix.shape[1] // 2
        self.shape = matrix.shape

    def transpose(self) -> SymplecticMatrix:
        """Return the transpose of the matrix."""
        return SymplecticMatrix(self.matrix.T)

    @classmethod
    def zeros(cls, n_rows: int, n: int) -> SymplecticMatrix:
        """Create a zero matrix of size n."""
        return cls(np.zeros((n_rows, 2 * n), dtype=np.int8))

    @classmethod
    def identity(cls, n: int) -> SymplecticMatrix:
        """Create the identity matrix of size n."""
        return cls(
            np.block([
                [np.zeros((n, n), dtype=np.int8), np.eye(n, dtype=np.int8)],
                [np.eye(n, dtype=np.int8), np.zeros((n, n), dtype=np.int8)],
            ])
        )

    @classmethod
    def empty(cls, n: int) -> SymplecticMatrix:
        """Create an empty matrix of size n."""
        return cls(np.empty((0, 2 * n), dtype=np.int8))

    def __add__(self, other: SymplecticMatrix) -> SymplecticMatrix:
        """Add two symplectic matrices."""
        return SymplecticMatrix((self.matrix + other.matrix) % 2)

    def __sub__(self, other: SymplecticMatrix) -> SymplecticMatrix:
        """Subtract two symplectic matrices."""
        return SymplecticMatrix((self.matrix - other.matrix) % 2)

    def __matmul__(self, other: SymplecticMatrix | SymplecticVector) -> Any:  # noqa: ANN401
        """Compute the symplectic product of two matrices."""
        assert self.n == other.n, "Matrices must be of the same size."
        n = self.n
        if isinstance(other, SymplecticVector):
            return SymplecticVector((self.matrix[:, :n] @ other[n:] + self.matrix[:, n:] @ other[:n]) % 2)
        m1 = self.matrix
        m2 = other.matrix
        return SymplecticMatrix(((m1[:, :n] @ m2[:, n:].T) + (m1[:, n:] @ m2[:, :n].T)) % 2)

    def __getitem__(self, key: tuple[int, int] | int | slice) -> Any:  # noqa: ANN401
        """Get the value of the matrix at index key."""
        return self.matrix[key]

    def __setitem__(self, key: tuple[int, int] | int | slice, value: npt.NDArray[np.int8]) -> None:
        """Set the value of the matrix at index key."""
        self.matrix[key] = value

    def __repr__(self) -> str:
        """Return the string representation of the matrix."""
        return str(self.matrix.__repr__())

    def __iter__(self) -> npt.NDArray[np.int8]:
        """Iterate over the rows of the matrix."""
        return self.matrix.__iter__()

    def __eq__(self, other: object) -> bool:
        """Check if two matrices are equal."""
        if not isinstance(other, SymplecticMatrix):
            return False
        return np.array_equal(self.matrix, other.matrix)

    def __ne__(self, other: object) -> bool:
        """Check if two matrices are not equal."""
        return not self == other

    def __hash__(self) -> int:
        """Return the hash of the matrix."""
        return hash(self.matrix.tobytes())

    def __len__(self) -> int:
        """Return the number of rows in the matrix."""
        return len(self.matrix)
