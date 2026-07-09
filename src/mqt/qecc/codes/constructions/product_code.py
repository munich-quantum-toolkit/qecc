# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Package for code construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as scs
from scipy import sparse
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from numpy.typing import NDArray


def is_all_zeros(array: NDArray[np.int32]) -> bool:
    """Check if array is all zeros."""
    return not np.any(array)


def sparse_all_zeros(mat: csr_matrix) -> bool:
    """Check if sparse matrix is all zeros."""
    mat.data %= 2
    return bool(mat.sum() == 0)


def run_sparse_checks_scipy(d_1: csr_matrix, d_2: csr_matrix, d_3: csr_matrix, d_4: csr_matrix) -> None:
    """Run checks on the boundary maps."""
    if not (sparse_all_zeros(d_1 @ d_2) and sparse_all_zeros(d_2 @ d_3) and sparse_all_zeros(d_3 @ d_4)):
        msg = "Error generating 4D code, boundary maps do not square to zero"
        raise RuntimeError(msg)


def run_checks_scipy(
    d_1: NDArray[np.int32], d_2: NDArray[np.int32], d_3: NDArray[np.int32], d_4: NDArray[np.int32]
) -> None:
    """Run checks on the boundary maps."""
    sd_1 = scs.csr_matrix(d_1)
    sd_2 = scs.csr_matrix(d_2)
    sd_3 = scs.csr_matrix(d_3)
    sd_4 = scs.csr_matrix(d_4)

    if not (
        is_all_zeros((sd_1 * sd_2).todense() % 2)
        and is_all_zeros((sd_2 * sd_3).todense() % 2)
        and is_all_zeros((sd_3 * sd_4).todense() % 2)
    ):
        msg = "Error generating 4D code, boundary maps do not square to zero"
        raise RuntimeError(msg)


def generate_4d_product_code(
    a_1: NDArray[np.int32], a_2: NDArray[np.int32], a_3: NDArray[np.int32], p: NDArray[np.int32], checks: bool = True
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """Generate 4D product code."""
    r, c = p.shape
    id_r: NDArray[np.int32] = np.identity(r, dtype=np.int32)
    id_c: NDArray[np.int32] = np.identity(c, dtype=np.int32)
    id_n0: NDArray[np.int32] = np.identity(a_1.shape[0], dtype=np.int32)
    id_n1: NDArray[np.int32] = np.identity(a_2.shape[0], dtype=np.int32)
    id_n2: NDArray[np.int32] = np.identity(a_3.shape[0], dtype=np.int32)
    id_n3: NDArray[np.int32] = np.identity(a_3.shape[1], dtype=np.int32)

    d_1: NDArray[np.int32] = np.hstack((np.kron(a_1, id_r), np.kron(id_n0, p))).astype(np.int32)

    x = np.hstack((np.kron(a_2, id_r), np.kron(id_n1, p)))
    y = np.kron(a_1, id_c)
    dims = (y.shape[0], x.shape[1] - y.shape[1])
    z = np.hstack((np.zeros(dims, dtype=np.int32), y))
    d_2 = np.vstack((x, z))

    x = np.hstack((np.kron(a_3, id_r), np.kron(id_n2, p)))
    y = np.kron(a_2, id_c)
    dims = (y.shape[0], x.shape[1] - y.shape[1])
    z = np.hstack((np.zeros(dims, dtype=np.int32), y))
    d_3: NDArray[np.int32] = np.vstack((x, z)).astype(np.int32)

    d_4: NDArray[np.int32] = np.vstack((np.kron(id_n3, p), np.kron(a_3, id_c)))

    if checks:
        run_checks_scipy(d_1, d_2, d_3, d_4)

    return d_1, d_2, d_3, d_4


def generate_3d_product_code(
    a_1: NDArray[np.int32], a_2: NDArray[np.int32], p: NDArray[np.int32]
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
    """Generate 3D product code."""
    r, c = p.shape
    id_r: NDArray[np.int32] = np.identity(r, dtype=np.int32)
    id_c: NDArray[np.int32] = np.identity(c, dtype=np.int32)
    id_n0: NDArray[np.int32] = np.identity(a_1.shape[0], dtype=np.int32)
    id_n1: NDArray[np.int32] = np.identity(a_2.shape[0], dtype=np.int32)
    id_n2: NDArray[np.int32] = np.identity(a_2.shape[1], dtype=np.int32)

    d_1: NDArray[np.int32] = np.hstack((np.kron(a_1, id_r), np.kron(id_n0, p)))

    x = np.hstack((np.kron(a_2, id_r), np.kron(id_n1, p)))
    y = np.kron(a_1, id_c)
    dims = (y.shape[0], x.shape[1] - y.shape[1])
    z = np.hstack((np.zeros(dims, dtype=np.int32), y))
    d_2: NDArray[np.int32] = np.vstack((x, z))

    d_3: NDArray[np.int32] = np.vstack((np.kron(id_n2, p), np.kron(a_2, id_c)))

    if not (is_all_zeros(d_1 @ d_2 % 2) and is_all_zeros(d_2 @ d_3 % 2)):
        msg = "Error generating 3D code, boundary maps do not square to zero"
        raise RuntimeError(msg)

    return d_1, d_2, d_3


def generate_sparse_4d_product_code(
    a_1: csr_matrix,
    a_2: csr_matrix,
    a_3: csr_matrix,
    p: csr_matrix,
    checks: bool = True,
) -> tuple[csr_matrix, csr_matrix, csr_matrix, csr_matrix]:
    """Generate 4D HGP code."""
    r, c = p.shape

    id_r = sparse.identity(r, dtype=int)
    id_c = sparse.identity(c, dtype=int)
    id_n0 = sparse.identity(a_1.shape[0], dtype=int)
    id_n1 = sparse.identity(a_2.shape[0], dtype=int)
    id_n2 = sparse.identity(a_3.shape[0], dtype=int)
    id_n3 = sparse.identity(a_3.shape[1], dtype=int)

    d_1 = sparse.hstack((sparse.kron(a_1, id_r), sparse.kron(id_n0, p)))

    x = sparse.hstack((sparse.kron(a_2, id_r), sparse.kron(id_n1, p)))
    y = sparse.kron(a_1, id_c)
    dims = (y.shape[0], x.shape[1] - y.shape[1])
    nmat = csr_matrix(np.zeros(dims))
    z = sparse.hstack((nmat, y))
    d_2 = sparse.vstack((x, z))

    x = sparse.hstack((sparse.kron(a_3, id_r), sparse.kron(id_n2, p)))
    y = sparse.kron(a_2, id_c)
    dims = (y.shape[0], x.shape[1] - y.shape[1])
    mat = csr_matrix(np.zeros(dims))
    z = sparse.hstack([mat, y])
    d_3 = sparse.vstack((x, z))

    d_4 = sparse.vstack((sparse.kron(id_n3, p), sparse.kron(a_3, id_c)))

    if checks:
        run_sparse_checks_scipy(d_1, d_2, d_3, d_4)

    return d_1, d_2, d_3, d_4


def generate_sparse_3d_product_code(
    a_1: csr_matrix, a_2: csr_matrix, p: csr_matrix
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Generate 3D HGP code."""
    r, c = p.shape

    id_r = sparse.identity(r, dtype=int)
    id_c = sparse.identity(c, dtype=int)
    id_n0 = sparse.identity(a_1.shape[0], dtype=int)
    id_n1 = sparse.identity(a_2.shape[0], dtype=int)
    id_n2 = sparse.identity(a_2.shape[1], dtype=int)

    d_1 = sparse.hstack((sparse.kron(a_1, id_r), sparse.kron(id_n0, p)))

    x = sparse.hstack((sparse.kron(a_2, id_r), sparse.kron(id_n1, p)))
    y = sparse.kron(a_1, id_c)
    dims = (y.shape[0], x.shape[1] - y.shape[1])
    z = sparse.hstack((csr_matrix(np.zeros(dims), dtype=int), y))
    d_2 = sparse.vstack((x, z))

    d_3 = sparse.vstack((sparse.kron(id_n2, p), sparse.kron(a_2, id_c)))

    if not (sparse_all_zeros(d_1 @ d_2) and sparse_all_zeros(d_2 @ d_3)):
        msg = "Error generating 3D code, boundary maps do not square to zero"
        raise RuntimeError(msg)

    return d_1, d_2, d_3  # mx, hx, hzT # hx, hzT, mzT
