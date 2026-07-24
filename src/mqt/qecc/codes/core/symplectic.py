# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Classes and Methods for working with symplectic vector spaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    import numpy.typing as npt


def symplectic_product(
    lhs: npt.NDArray[np.int8],
    rhs: npt.NDArray[np.int8],
) -> np.int8 | npt.NDArray[np.int8]:
    """Compute the binary symplectic product of vectors or row matrices.

    Both operands are interpreted in ``[X | Z]`` form, i.e. the first half of
    each row is the X part and the second half is the Z part.

    Args:
        lhs: A vector of length ``2n`` or a matrix whose rows have length ``2n``.
        rhs: A vector of length ``2n`` or a matrix whose rows have length ``2n``.

    Returns:
        A scalar for vector/vector input, a vector for vector/matrix or
        matrix/vector input, and a matrix for matrix/matrix input.

    Raises:
        ValueError: If an operand is not one- or two-dimensional, has an odd or
            zero width, or the operands represent different numbers of qubits.
    """
    widths = []
    for name, operand in (("lhs", lhs), ("rhs", rhs)):
        if operand.ndim not in {1, 2}:
            msg = f"{name} must be one- or two-dimensional, got {operand.ndim} dimensions."
            raise ValueError(msg)
        width = operand.shape[-1]
        if width == 0 or width % 2 != 0:
            msg = f"{name} must have nonzero even width, got shape {operand.shape}."
            raise ValueError(msg)
        widths.append(width)
    if widths[0] != widths[1]:
        msg = f"Operands must represent the same number of qubits, got widths {widths[0]} and {widths[1]}."
        raise ValueError(msg)
    n = widths[0] // 2

    lhs_x, lhs_z = lhs[..., :n], lhs[..., n:]
    rhs_x, rhs_z = rhs[..., :n], rhs[..., n:]
    if rhs.ndim == 2:
        rhs_x, rhs_z = rhs_x.T, rhs_z.T
    product = (lhs_x @ rhs_z + lhs_z @ rhs_x) % 2
    if lhs.ndim == 1 and rhs.ndim == 1:
        return np.int8(product)
    return product.astype(np.int8)


class SymplecticVector:
    """Symplectic Vector Class."""

    def __init__(self, vector: npt.NDArray[np.int8]) -> None:
        """Initialize the Symplectic Vector."""
        assert vector.ndim == 1, "Vector must be 1D."
        assert vector.shape[0] % 2 == 0, "Vector must have even length."
        self.data = vector
        self.n = vector.shape[0] // 2

    def copy(self) -> SymplecticVector:
        """Return a copy of the vector."""
        return SymplecticVector(self.data.copy())

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
        return SymplecticVector((self.data + other.data) % 2)

    def __sub__(self, other: SymplecticVector) -> SymplecticVector:
        """Subtract two symplectic vectors."""
        return SymplecticVector((self.data - other.data) % 2)

    def __neg__(self) -> SymplecticVector:
        """Negate the vector."""
        return SymplecticVector(-self.data)

    def __matmul__(self, other: SymplecticVector) -> int:
        """Compute the symplectic inner product."""
        return int(symplectic_product(self.data, other.data))

    def __getitem__(self, key: int | slice) -> int | npt.NDArray[np.int8]:
        """Get the value of the vector at index key."""
        return self.data[key]

    def __setitem__(self, key: int | slice, value: int | npt.NDArray[np.int8]) -> None:
        """Set the value of the vector at index key."""
        self.data[key] = value

    def __eq__(self, other: object) -> bool:
        """Check if two vectors are equal."""
        if not isinstance(other, SymplecticVector):
            return False
        return np.array_equal(self.data, other.data)

    def __ne__(self, other: object) -> bool:
        """Check if two vectors are not equal."""
        return not self == other

    def __hash__(self) -> int:
        """Return the hash of the vector."""
        return hash(self.data.tobytes())

    def __repr__(self) -> str:
        """Return the string representation of the vector."""
        return str(self.data.__repr__())

    def __len__(self) -> int:
        """Return the length of the vector."""
        return len(self.data)


class SymplecticMatrix:
    """Symplectic Matrix Class."""

    def __init__(self, matrix: npt.NDArray[np.int8]) -> None:
        """Initialize the Symplectic Matrix."""
        assert matrix.ndim == 2, "Matrix must be 2D."
        assert matrix.shape[1] % 2 == 0, "Matrix must have even width."
        self.data = matrix
        self.n = matrix.shape[1] // 2
        self.shape = matrix.shape

    def copy(self) -> SymplecticMatrix:
        """Return a copy of the matrix."""
        return SymplecticMatrix(self.data.copy())

    @classmethod
    def zeros(cls, n_rows: int, n: int) -> SymplecticMatrix:
        """Create a zero matrix of size n."""
        return cls(np.zeros((n_rows, 2 * n), dtype=np.int8))

    @classmethod
    def identity(cls, n: int) -> SymplecticMatrix:
        """Create the identity matrix of size n."""
        return cls(np.eye(2 * n, dtype=np.int8))

    @classmethod
    def symplectic_identity(cls, n: int) -> SymplecticMatrix:
        """Create the identity matrix of size n."""
        mat = np.zeros((2 * n, 2 * n), dtype=np.int8)
        mat[:n, n:] = np.eye(n, dtype=np.int8)
        mat[n:, :n] = np.eye(n, dtype=np.int8)
        return cls(mat)

    @classmethod
    def empty(cls, n: int) -> SymplecticMatrix:
        """Create an empty matrix of size n."""
        return cls(np.empty((0, 2 * n), dtype=np.int8))

    def __add__(self, other: SymplecticMatrix) -> SymplecticMatrix:
        """Add two symplectic matrices."""
        return SymplecticMatrix((self.data + other.data) % 2)

    def __sub__(self, other: SymplecticMatrix) -> SymplecticMatrix:
        """Subtract two symplectic matrices."""
        return SymplecticMatrix((self.data - other.data) % 2)

    def __matmul__(self, other: SymplecticMatrix | SymplecticVector) -> npt.NDArray[np.int8]:
        """Compute the symplectic product with another matrix or vector."""
        return np.asarray(symplectic_product(self.data, other.data), dtype=np.int8)

    @overload
    def __getitem__(self, key: int) -> npt.NDArray[np.int8]: ...

    @overload
    def __getitem__(self, key: slice) -> npt.NDArray[np.int8]: ...

    @overload
    def __getitem__(self, key: tuple[int, int]) -> np.int8: ...

    @overload
    def __getitem__(self, key: tuple[slice, int]) -> npt.NDArray[np.int8]: ...

    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> npt.NDArray[np.int8]: ...

    @overload
    def __getitem__(self, key: tuple[slice, list[int]]) -> npt.NDArray[np.int8]: ...

    def __getitem__(
        self,
        key: int | slice | tuple[int, int] | tuple[slice, int] | tuple[slice, slice] | tuple[slice, list[int]],
    ) -> Any:
        """Get the value of the matrix at index key."""
        return self.data[key]

    def __setitem__(
        self,
        key: int | slice | tuple[int, int] | tuple[slice, int] | tuple[slice, slice] | tuple[slice, list[int]],
        value: npt.NDArray[np.int8] | np.int8,
    ) -> None:
        """Set the value of the matrix at index key."""
        self.data[key] = value

    def __repr__(self) -> str:
        """Return the string representation of the matrix."""
        return str(self.data.__repr__())

    def __iter__(self) -> Iterator[npt.NDArray[np.int8]]:
        """Iterate over the rows of the matrix."""
        return self.data.__iter__()

    def __eq__(self, other: object) -> bool:
        """Check if two matrices are equal."""
        if not isinstance(other, SymplecticMatrix):
            return False
        return np.array_equal(self.data, other.data)

    def __ne__(self, other: object) -> bool:
        """Check if two matrices are not equal."""
        return not self == other

    def __hash__(self) -> int:
        """Return the hash of the matrix."""
        return hash(self.data.tobytes())

    def __len__(self) -> int:
        """Return the number of rows in the matrix."""
        return len(self.data)
