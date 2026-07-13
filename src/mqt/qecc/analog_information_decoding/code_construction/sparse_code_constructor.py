# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
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
from scipy import sparse
from scipy.sparse import coo_matrix, csr_matrix

from mqt.qecc.mod2 import rank

from ...codes.constructions.hypergraph_product_code import generate_sparse_3d_product_code
from . import code_constructor

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    lx: csr_matrix | None = None,
    lz: csr_matrix | None = None,
) -> None:
    """Save code to file."""
    path = create_outpath(codename)

    matrices: list[csr_matrix | None] = [hx, hz, mz, lx, lz]
    names = ["hx", "hz", "mz", "lx", "lz"]
    for mat, name in zip(matrices, names, strict=False):
        if mat is not None:
            path_str = path + name
            try:
                np.savetxt(path_str + ".txt", mat.todense(), fmt="%i")
            except ValueError:
                np.savetxt(path_str + ".txt", mat.toarray(), fmt="%i")
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
    code_k = n - rank(hx) - rank(hz)
    with Path(f"/codes/generated_codes/{codename}/info.txt").open(encoding="utf-8") as f:
        code_dict: dict[str, Any] = dict(
            line[: line.rfind("#")].split(" = ") for line in f if not line.startswith("#") and line.strip()
        )

    code_dict["n"] = n
    code_dict["k"] = code_k
    code_dict["dX"] = int(code_dict["dX"])
    code_dict["dZ"] = int(code_dict["dZ"])

    Path(f"/codes/generated_codes/{codename}/code_params.txt").write_text(json.dumps(code_dict), encoding="utf-8")


def _store_code_params(hx: csr_matrix, hz: csr_matrix, codename: str) -> None:
    """Store code parameters in file."""
    code_dict = {}
    hx, hz = hx.todense(), hz.todense()
    _m, n = hx.shape
    code_k = n - rank(hx) - rank(hz)
    code_dict["n"] = n
    code_dict["k"] = code_k
    Path(f"/codes/generated_codes/{codename}/code_params.txt").write_text(json.dumps(code_dict), encoding="utf-8")


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
    res = generate_sparse_3d_product_code(a1, a2, sparse.csr_matrix(seed_codes[2]))
    hx, hz_t, mz_t = res

    hz = hz_t.transpose()
    mz = mz_t.transpose()
    if compute_logicals:
        lx, lz = code_constructor._compute_logicals(hx.todense(), hz.todense())  # noqa: SLF001
        save_code(hx=hx, hz=hz, mz=mz, codename=codename, lx=csr_matrix(lx), lz=csr_matrix(lz))

    if compute_distance:
        _compute_distances(hx.todense(), hz.todense(), codename)

    else:
        _store_code_params(hx.todense(), hz.todense(), codename)
