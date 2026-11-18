# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Failing fault combinations grouped by fault class, and their coverage by candidate checks.

A fault class holds the faults with the same key. A candidate catches a combination only if it detects it for every
member of each class involved: per class the detection pattern of its first member, valid where all members agree; a
combination's pattern is the XOR over its classes (a class used twice cancels). Used by the greedy (``greedy_ft``) and
the promotion ILP (``gauge.promote``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .verify import FTReport


class Combos:
    """The failing combinations of an ``FTReport`` as combinations of fault classes."""

    def __init__(self, rep: FTReport, cls: np.ndarray, *, max_combos: int | None = None, seed: int = 0) -> None:
        """Group the collected failures of ``rep`` by fault class.

        Args:
            rep: An ``FTReport`` with collected failures.
            cls: Class id per elementary fault.
            max_combos: Keep at most this many combinations: all of the lowest order, then a seeded sample of the rest
                (``scale``: the weight that keeps sums over the sample unbiased).
            seed: Seed of the sample.
        """
        cls = np.asarray(cls, dtype=np.int64)
        arrs = rep.rep_arrays
        parts, ns = [], []
        width = 3
        for a in arrs:
            if not len(a):
                continue
            c = cls[a]
            _, idx = np.unique(c, axis=0, return_index=True)
            c = c[np.sort(idx)]  # distinct, in order of first occurrence
            pad = np.full((len(c), width), -1, dtype=np.int64)
            pad[:, : c.shape[1]] = c
            parts.append(pad)
            ns.append(np.full(len(c), c.shape[1], dtype=np.int64))
        combos = np.vstack(parts) if parts else np.zeros((0, width), dtype=np.int64)
        ns = np.concatenate(ns) if ns else np.zeros(0, dtype=np.int64)
        self.total = len(combos)
        scale = np.ones(len(combos))
        if max_combos is not None and len(combos) > max_combos:
            lo = np.min(ns)
            keep = np.nonzero(ns == lo)[0][:max_combos]
            rest = np.nonzero(ns != lo)[0]
            k = max_combos - len(keep)
            pick = (
                np.sort(np.random.default_rng(seed).choice(len(rest), size=min(k, len(rest)), replace=False))
                if k > 0 and len(rest)
                else np.zeros(0, dtype=np.int64)
            )
            sel = np.concatenate([keep, rest[pick]])
            scale = np.concatenate([np.ones(len(keep)), np.full(len(pick), len(rest) / max(len(pick), 1))])
            combos, ns = combos[sel], ns[sel]
        self.combos, self.ns, self.scale = combos, ns, scale
        used = np.unique(combos[combos >= 0])
        self.faults = np.nonzero(np.isin(cls, used))[0]
        fcls = np.searchsorted(used, cls[self.faults])
        self._idx = np.where(combos >= 0, np.searchsorted(used, np.maximum(combos, 0)), len(used))
        self._order = np.argsort(fcls, kind="stable")
        counts = np.bincount(fcls, minlength=len(used))
        self._counts = counts
        self._starts = np.cumsum(counts) - counts

    def __len__(self) -> int:
        """Number of combinations."""
        return len(self.combos)

    def combine(self, bm: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        """Coverage (combinations x candidates) from the detection matrix ``bm`` (``self.faults`` x candidates).

        ``rows``: only these combinations.
        """
        bm = np.asarray(bm).astype(np.uint8, copy=False) & 1
        n_c = bm.shape[1]
        idx = self._idx if rows is None else self._idx[np.asarray(rows, dtype=np.int64)]
        if not len(idx):
            return np.zeros((0, n_c), dtype=np.uint8)
        srt = bm[self._order]
        first = srt[self._starts]
        bad = np.maximum.reduceat(srt ^ np.repeat(first, self._counts, axis=0), self._starts, axis=0)
        v = np.vstack([first, np.zeros((1, n_c), dtype=np.uint8)])
        k = np.vstack([bad == 0, np.ones((1, n_c), dtype=bool)])
        out, ok = v[idx[:, 0]].copy(), k[idx[:, 0]].copy()
        for c in range(1, idx.shape[1]):
            out ^= v[idx[:, c]]
            ok &= k[idx[:, c]]
        return np.where(ok, out, 0).astype(np.uint8)
