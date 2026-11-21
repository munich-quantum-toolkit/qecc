# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Ancilla blocks: measure a set of commuting checks of one type jointly through a prepared ancilla state.

Z block (catches data X errors) with Z-type checks M on support S: the ancilla on |S| qubits is the uniform
superposition over ker(M|S); transversal CNOTs data -> ancilla, then the ancilla is measured in Z. Each element of
rowspace(M|S) is a free detector. Its ancilla sends Z errors back onto the data. X blocks are the dual (CNOTs ancilla
-> data, measured in X, back type X). With M the full stabilizer both ancillas are the logical state (Steane-type).

Split blocks (Huang & Brown, arXiv:2012.15403): ancilla qubit a couples to the data qubit Γ(a) (``gamma``); H̃
(``htilde``, rows over the ancilla qubits) says which outcomes add up per check, so the data checks are H = H̃ Γ^T.
Grouped blocks: ``groups`` gives each ancilla qubit a set of data qubits, coupled one after another; its errors reach
its whole group, so the ancilla target weighs them by their weight on the data (``verify.WeightedTarget``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mqt.qecc import mod2

from .verify import gf2_row_basis


def dual(g: np.ndarray) -> np.ndarray:
    """Basis of the orthogonal complement of rowspan(g) (rows over the same columns)."""
    g = np.asarray(g, dtype=np.int64).reshape(-1, np.shape(g)[-1]) % 2
    if not g.any():
        return np.eye(g.shape[1], dtype=np.uint8)
    return np.asarray(mod2.nullspace(g), dtype=np.uint8).reshape(-1, g.shape[1])


@dataclass
class BlockSpec:
    """Which checks a block measures. ``kind`` "Z": Z-type checks (detect X errors); "X": X-type.

    ``gamma`` / ``htilde`` (split blocks): data qubit of each ancilla qubit and the ancilla check matrix, with
    ``checks`` = htilde Γ^T. None: one ancilla qubit per support qubit.
    """

    kind: str
    checks: np.ndarray  # rows over all data qubits
    pos: int | None = None  # None: coupled after the prep; else before base op ``pos``
    gamma: tuple[int, ...] | None = None
    htilde: np.ndarray | None = None
    groups: tuple[tuple[int, ...], ...] | None = None  # grouped blocks: data qubits of each ancilla qubit, in order

    def __post_init__(self) -> None:
        """Normalize ``gamma``, ``htilde`` and ``groups`` and check that they give ``checks``."""
        if self.groups is not None:
            self.groups = tuple(tuple(int(q) for q in g) for g in self.groups)
            if all(len(g) == 1 for g in self.groups):  # singletons: the transversal (split) form
                self.gamma, self.groups = tuple(g[0] for g in self.groups), None
            else:
                ht = np.asarray(self.htilde, dtype=np.uint8) % 2
                self.htilde = ht
                assert np.array_equal(self.gamma_matrix() @ ht.T % 2, (np.asarray(self.checks) % 2).T), (
                    "checks != htilde Γ^T"
                )
                return
        if self.gamma is not None:
            self.gamma = tuple(int(q) for q in self.gamma)
            ht = np.asarray(self.htilde, dtype=np.uint8) % 2
            self.htilde = ht
            img = np.zeros_like(self.checks, dtype=np.uint8)
            for a, q in enumerate(self.gamma):
                img[:, q] ^= ht[:, a]
            assert np.array_equal(img, np.asarray(self.checks) % 2), "checks != htilde Γ^T"

    @property
    def split(self) -> bool:
        """Whether the block has an explicit ancilla map (split or grouped)."""
        return self.gamma is not None or self.groups is not None

    @property
    def grouped(self) -> bool:
        """Whether some ancilla qubit of the block couples to several data qubits."""
        return self.groups is not None

    def anc_groups(self) -> list[tuple[int, ...]]:
        """Data qubits of each ancilla qubit (one each unless grouped)."""
        if self.groups is not None:
            return list(self.groups)
        return [(q,) for q in self.anc_data]

    def gamma_matrix(self) -> np.ndarray:
        """Γ: data qubits (all of them) x ancilla qubits."""
        n = np.asarray(self.checks).shape[1]
        gs = self.anc_groups()
        g = np.zeros((n, len(gs)), dtype=np.uint8)
        for a, grp in enumerate(gs):
            for q in grp:
                g[q, a] ^= 1
        return g

    @property
    def support(self) -> list[int]:
        """Data qubits that the block's checks act on."""
        return [int(q) for q in np.nonzero(self.checks.any(axis=0))[0]]

    @property
    def anc_data(self) -> list[int]:
        """Data qubit of each ancilla qubit (Γ); transversal blocks only."""
        if self.groups is not None:
            msg = "grouped block: use anc_groups"
            raise ValueError(msg)
        return list(self.gamma) if self.gamma is not None else self.support

    @property
    def n_anc(self) -> int:
        """Number of ancilla qubits of the block."""
        return len(self.groups) if self.groups is not None else len(self.anc_data)

    def restricted(self) -> np.ndarray:
        """The ancilla check matrix H̃ (rows over the ancilla qubits)."""
        if self.gamma is not None or self.groups is not None:
            assert self.htilde is not None, "a split block has htilde (``__post_init__``)"
            return self.htilde
        return self.checks[:, self.support] % 2

    def ancilla_generator(self) -> np.ndarray:
        """Rows spanning the code the ancilla is a uniform superposition over (the prep target, 'hx')."""
        m = self.restricted()
        return dual(m) if self.kind == "Z" else gf2_row_basis(m)

    def harmless_back(self) -> np.ndarray:
        """Ancilla errors of the back-propagating type that act on the data like a measured check.

        rowspace(H̃) and, for split blocks, pairs of ancilla qubits on the same data qubit. The local target judges back
        errors modulo these.
        """
        m = self.restricted()
        if self.groups is not None:  # every ancilla error whose image on the data is a measured check
            return dual(self.reduced_gamma())
        if self.gamma is None:
            return m
        pairs = []
        first: dict[int, int] = {}
        for a, q in enumerate(self.gamma):
            if q in first:
                v = np.zeros(self.n_anc, dtype=np.uint8)
                v[[first[q], a]] = 1
                pairs.append(v)
            else:
                first[q] = a
        return np.vstack([m, *pairs]) if pairs else m

    def reduced_gamma(self) -> np.ndarray:
        """Γ on the support, each column reduced modulo the measured checks.

        An ancilla error e acts on the data like a measured check iff reduced_gamma() e = 0.
        """
        sup = self.support
        g = self.gamma_matrix()[sup]  # support x anc
        h = gf2_row_basis(np.asarray(self.checks, dtype=np.uint8)[:, sup] % 2)
        cols = g.T.copy()  # anc x support: data image of each ancilla qubit
        for r in h:
            p = int(np.nonzero(r)[0][0])
            cols[cols[:, p] == 1] ^= r
        return cols.T % 2
