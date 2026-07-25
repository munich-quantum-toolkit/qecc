# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Package for code construction."""

from __future__ import annotations

import json
import subprocess  # ruff:ignore[suspicious-subprocess-import]
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ldpc.codes
import numpy as np
import scipy.io as sio
from bposd.hgp import hgp
from scipy import sparse

from mqt.qecc import mod2

from ...codes.constructions.hypergraph_product_code import generate_3d_product_code, generate_4d_product_code

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    for mat, name in zip(ms, names, strict=False):
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
    subprocess.run(["bash", "compute_distances.sh", path], check=False)  # ruff:ignore[subprocess-without-shell-equals-true, start-process-with-partial-path]


def _compute_distances(hx: NDArray[np.int32], hz: NDArray[np.int32], codename: str) -> None:
    run_compute_distances(codename)
    code_dict: dict[str, Any] = {}
    _, n = hx.shape
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

    Path(f"generated_codes/{codename}/code_params.txt").write_text(json.dumps(code_dict), encoding="utf-8")


def _compute_logicals(hx: NDArray[np.int32], hz: NDArray[np.int32]) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    def compute_lz(hx: NDArray[np.int32], hz: NDArray[np.int32]) -> NDArray[np.int32]:
        # lz logical operators
        # lz\in ker{hx} AND \notin Im(Hz.T)

        ker_hx = mod2.nullspace(hx)  # compute the kernel basis of hx
        im_hz_t = mod2.row_basis(hz)  # compute the image basis of hz.T

        # in the below we row reduce to find vectors in kx that are not in the image of hz.T.
        log_stack = np.vstack([im_hz_t, ker_hx], dtype=np.int32)
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
            [code.toarray() for code in seed_codes],
            codename,
            compute_distance,
            compute_logicals,
        )
