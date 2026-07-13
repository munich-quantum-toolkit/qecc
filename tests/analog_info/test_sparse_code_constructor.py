# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the sparse code constructor's use of the centralized HGP builder."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.sparse import csr_matrix

# `sparse_code_constructor` imports `bposd`, whose source contains invalid escape
# sequences that raise a SyntaxWarning on Python >= 3.12. Under the project's
# ``filterwarnings = ["error"]`` config this would turn the import into a
# collection error, so suppress it while importing the module under test.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import mqt.qecc.analog_information_decoding.code_construction.sparse_code_constructor as scc

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_create_code_builds_valid_css_code_via_sparse_3d(monkeypatch: pytest.MonkeyPatch) -> None:
    """`create_code` must route through `generate_sparse_3d_product_code` and yield a valid CSS code.

    This pins down the wiring after the HGP builders were centralized into
    `codes.constructions.hypergraph_product_code`: `create_code` extends the
    seed hypergraph-product code to a 3D product code and treats the first two
    returned boundary maps as the (Hx, Hz) of a CSS code. A valid CSS code
    requires ``Hx @ Hz.T == 0`` over GF(2).

    The bposd hypergraph product and the file/subprocess side effects are
    stubbed out so the test is deterministic and touches neither external
    solvers nor the filesystem.
    """
    # Stand in for the bposd hypergraph product with a fixed valid 2-complex
    # (Hx @ Hz.T = 0 mod 2), so `create_code` gets well-formed boundary maps.
    hx_seed = np.array([[1, 1]], dtype=np.int32)
    hz_seed = np.array([[1, 1], [1, 1]], dtype=np.int32)
    monkeypatch.setattr(scc, "hgp", lambda _h1, _h2: SimpleNamespace(hx=hx_seed, hz=hz_seed))

    captured: dict[str, NDArray[np.int32]] = {}

    def _capture(hx: NDArray[np.int32], hz: NDArray[np.int32], _codename: str) -> None:
        captured["hx"] = np.asarray(hx)
        captured["hz"] = np.asarray(hz)

    # Redirect the terminal write step so nothing is persisted to disk.
    monkeypatch.setattr(scc, "_store_code_params", _capture)

    p = csr_matrix(np.array([[1, 1]], dtype=np.int32))
    # seed_codes[0:2] are consumed by the stubbed `hgp`; seed_codes[2] (`p`) drives the
    # 3D extension. Pass them all as csr_matrix to match the `list[csr_matrix]` signature.
    scc.create_code("hgp", [csr_matrix(hx_seed), csr_matrix(hz_seed), p], "unit_test_code")

    hx, hz = captured["hx"], captured["hz"]
    assert hx.size > 0
    assert hx.shape[1] == hz.shape[1]
    # The extended code must be a valid CSS code.
    assert not np.any((hx @ hz.T) % 2)


def test_create_code_rejects_unknown_constructor() -> None:
    """An unsupported constructor name must raise a clear error."""
    p = csr_matrix(np.array([[1, 1]], dtype=np.int32))
    with pytest.raises(ValueError, match="not implemented"):
        scc.create_code("not_a_constructor", [p, p, p], "unit_test_code")
