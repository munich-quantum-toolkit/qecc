# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sparse code constructor for 3D and 4D HGP codes."""

from __future__ import annotations

import json
import subprocess  # noqa: S404
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.io as sio
from bposd.hgp import hgp
from ldpc import mod2
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix

from . import code_constructor

if TYPE_CHECKING:
    from numpy.typing import NDArray


def is_all_zeros(array: NDArray[np.int_]) -> bool:
    """Check if array is all zeros."""
    return not np.any(array)


def sparse_all_zeros(mat: csr_matrix) -> bool:
    """Check if sparse matrix is all zeros."""
    mat.data %= 2
    return bool(mat.sum() == 0)


def run_checks_scipy(d_1: csr_matrix, d_2: csr_matrix, d_3: csr_matrix, d_4: csr_matrix) -> None:
    """Run checks on the boundary maps."""
    if not (sparse_all_zeros(d_1 @ d_2) and sparse_all_zeros(d_2 @ d_3) and sparse_all_zeros(d_3 @ d_4)):
        msg = "Error generating 4D code, boundary maps do not square to zero"
        raise RuntimeError(msg)


def generate_4d_product_code(
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
        run_checks_scipy(d_1, d_2, d_3, d_4)

    return d_1, d_2, d_3, d_4


def generate_3d_product_code(
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


def create_outpath(codename: str) -> str:
    """Create output path for code."""
    path = f"/codes/generated_codes/{codename}/"
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_code(
    hx: csr_matrix,
    hz: csr_matrix,
    mz: csr_matrix,
    codename: str,
    lx: csr_matrix = None,
    lz: csr_matrix = None,
) -> None:
    """Save code to file."""
    path = create_outpath(codename)

    matrices: list[csr_matrix] = [hx, hz, mz, lx, lz]
    names = ["hx", "hz", "mz", "lx", "lz"]
    for mat, name in zip(matrices, names):
        if mat is not None:
            path_str = path + name
            try:
                np.savetxt(path_str + ".txt", mat.todense(), fmt="%i")
            except Exception:
                np.savetxt(path_str + ".txt", mat, fmt="%i")
            sio.mmwrite(
                path_str + ".mtx",
                coo_matrix(mat),
                comment="Field: GF(2)",
                field="integer",
            )


def run_compute_distances(codename: str) -> None:
    """Run compute distances bash script."""
    path = "/codes/generated_codes/" + codename
    subprocess.run(["bash", "compute_distances_3D.sh", path], check=False)  # noqa: S603, S607


def _compute_distances(hx: NDArray[np.int32], hz: NDArray[np.int32], codename: str) -> None:
    run_compute_distances(codename)
    _, n = hx.shape
    code_k = n - mod2.rank(hx) - mod2.rank(hz)
    with Path(f"/codes/generated_codes/{codename}/info.txt").open(encoding="utf-8") as f:
        code_dict: dict[str, Any] = dict(
            line[: line.rfind("#")].split(" = ") for line in f if not line.startswith("#") and line.strip()
        )

    code_dict["n"] = n
    code_dict["k"] = code_k
    code_dict["dX"] = int(code_dict["dX"])
    code_dict["dZ"] = int(code_dict["dZ"])

    with Path(f"/codes/generated_codes/{codename}/code_params.txt").open("w", encoding="utf-8") as file:
        file.write(json.dumps(code_dict))


def _store_code_params(hx: csr_matrix, hz: csr_matrix, codename: str) -> None:
    """Store code parameters in file."""
    code_dict = {}
    hx, hz = hx.todense(), hz.todense()
    _m, n = hx.shape
    code_k = n - mod2.rank(hx) - mod2.rank(hz)
    code_dict["n"] = n
    code_dict["k"] = code_k
    with Path(f"/codes/generated_codes/{codename}/code_params.txt").open("w", encoding="utf-8") as file:
        file.write(json.dumps(code_dict))


def create_code(
    constructor: str,
    seed_codes: list[csr_matrix],
    codename: str,
    compute_distance: bool = False,
    compute_logicals: bool = False,
) -> None:
    """Create code."""
    # Construct initial 2 dim code
    if constructor == "hgp":
        code = hgp(seed_codes[0], seed_codes[1])
    else:
        msg = f"No constructor specified or the specified constructor {constructor} not implemented."
        raise ValueError(msg)

    # Extend to 3D HGP
    a1 = sparse.csr_matrix(code.hx)
    a2 = sparse.csr_matrix(code.hz.T)
    res = generate_3d_product_code(a1, a2, sparse.csr_matrix(seed_codes[2]))
    hx, hz_t, mz_t = res

    hz = hz_t.transpose()
    mz = mz_t.transpose()
    if compute_logicals:
        lx, lz = code_constructor._compute_logicals(hx.todense(), hz.todense())  # noqa: SLF001
        save_code(hx=hx, hz=hz, mz=mz, codename=codename, lx=lx, lz=lz)

    if compute_distance:
        _compute_distances(hx.todense(), hz.todense(), codename)

    else:
        _store_code_params(hx.todense(), hz.todense(), codename)
