# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Package for code construction."""

from __future__ import annotations

import json
import subprocess  # noqa: S404
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ldpc.code_util
import numpy as np
import scipy.io as sio
import scipy.sparse as scs
from bposd.hgp import hgp
from ldpc import mod2
from scipy import sparse

if TYPE_CHECKING:
    from numpy.typing import NDArray


def is_all_zeros(array: NDArray[np.int32]) -> bool:
    """Check if array is all zeros."""
    return not np.any(array)


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


def create_outpath(codename: str) -> str:
    """Create output path for code files."""
    path = f"generated_codes/{codename}/"
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_code(
    hx: NDArray[np.int32],
    hz: NDArray[np.int32],
    mx: NDArray[np.int32],
    mz: NDArray[np.int32],
    codename: str,
    lx: NDArray[np.int32] | None,
    lz: NDArray[np.int32] | None,
) -> None:
    """Save code to files."""
    path = create_outpath(codename)
    ms = [hx, hz, mx, mz, lx, lz] if lx is not None and lz is not None else [hx, hz, mx, mz]
    names: list[str] = ["hx", "hz", "mx", "mz", "lx", "lz"]
    for mat, name in zip(ms, names):
        if mat is not None:
            np.savetxt(path + name + ".txt", mat, fmt="%i")
            sio.mmwrite(
                path + name + ".mtx",
                sparse.coo_matrix(mat),
                comment="Field: GF(2)",
            )


def run_compute_distances(codename: str) -> None:
    """Run compute distances bash script."""
    path = "generated_codes/" + codename
    subprocess.run(["bash", "compute_distances.sh", path], check=False)  # noqa: S603, S607


def _compute_distances(hx: NDArray[np.int32], hz: NDArray[np.int32], codename: str) -> None:
    run_compute_distances(codename)
    code_dict: Any = {}
    _m, n = hx.shape
    code_k = n - mod2.rank(hx) - mod2.rank(hz)
    with Path(f"generated_codes/{codename}/info.txt").open(encoding="utf-8") as f:
        code_dict = dict(
            line[: line.rfind("#")].split(" = ") for line in f if not line.startswith("#") and line.strip()
        )

    code_dict["n"] = n
    code_dict["k"] = code_k
    code_dict["dX"] = int(code_dict["dX"])
    code_dict["dZ"] = int(code_dict["dZ"])
    code_dict["dMX"] = int(code_dict["dMX"])
    code_dict["dMZ"] = int(code_dict["dMZ"])

    with Path(f"generated_codes/{codename}/code_params.txt").open("w", encoding="utf-8") as file:
        file.write(json.dumps(code_dict))


def _compute_logicals(hx: NDArray[np.int32], hz: NDArray[np.int32]) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    def compute_lz(hx: NDArray[np.int32], hz: NDArray[np.int32]) -> NDArray[np.int32]:
        # lz logical operators
        # lz\in ker{hx} AND \notin Im(Hz.T)

        ker_hx = mod2.nullspace(hx)  # compute the kernel basis of hx
        im_hz_t = mod2.row_basis(hz)  # compute the image basis of hz.T

        # in the below we row reduce to find vectors in kx that are not in the image of hz.T.
        log_stack = np.vstack([im_hz_t, ker_hx])
        pivots = mod2.row_echelon(log_stack.T)[3]
        log_op_indices = [i for i in range(im_hz_t.shape[0], log_stack.shape[0]) if i in pivots]
        return log_stack[log_op_indices]

    lx = compute_lz(hz, hx)
    lz = compute_lz(hx, hz)
    return lx, lz


def create_code(
    constructor: str,
    seed_codes: list[NDArray[np.int32]],
    codename: str,
    compute_distance: bool = False,
    compute_logicals: bool = False,
    checks: bool = False,
) -> None:
    """Create 4D code."""
    # Construct initial 2 dim code
    if constructor == "hgp":
        code = hgp(seed_codes[0], seed_codes[1])
    else:
        msg = f"No constructor specified or the specified constructor {constructor} not implemented."
        raise ValueError(msg)

    # Extend to 3D HGP
    a1 = code.hx
    a2 = code.hz.T
    res = generate_3d_product_code(a1, a2, seed_codes[2])

    # Build 4D HGP code
    mx, hx, hz_t, mz_t = generate_4d_product_code(*res, seed_codes[3], checks=checks)

    hz = hz_t.T
    mz = mz_t.T

    # Perform checks
    if np.any(hz_t @ mz_t % 2) or np.any(hx @ hz_t % 2) or np.any(mx @ hx % 2):
        msg = "err"
        raise RuntimeError(msg)
    save_code(hx, hz, mx, mz, codename, lx=None, lz=None)

    if compute_logicals:
        lx, lz = _compute_logicals(hx, hz)
        save_code(hx, hz, mx, mz, codename, lx=lx, lz=lz)

    else:
        save_code(hx, hz, mx, mz, codename, lx=None, lz=None)

    if compute_distance:
        _compute_distances(hx, hz, codename)


if __name__ == "__main__":
    for d in range(3, 8):
        seed_codes = [
            ldpc.codes.ring_code(d),
            ldpc.codes.ring_code(d),
            ldpc.codes.ring_code(d),
            ldpc.codes.ring_code(d),
        ]

        constructor = "hgp"
        codename = f"4D_toric_{d:d}"
        compute_distance = False
        compute_logicals = True
        create_code(
            constructor,
            seed_codes,
            codename,
            compute_distance,
            compute_logicals,
        )
