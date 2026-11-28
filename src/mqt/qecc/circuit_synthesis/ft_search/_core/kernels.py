# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compiled kernels of the exact check (``verify.check_ft``): fault keys and the enumeration of fault sets.

A set of keys is *undetected* if the XOR of its signatures is 0 and *bad* if the XOR of its X (or Z) syndromes has
coset-leader weight above its size. s = 1: keys with signature 0; s = 2: pairs inside one signature bucket; s = 3:
index triples i <= j <= k with sig(k) = sig(i) ^ sig(j) (a fault may be used twice).
"""

from __future__ import annotations

import numpy as np
from numba import njit

_ONE = np.uint64(1)


@njit(cache=True)
def _flip(m: np.ndarray, i: int, k: int) -> None:
    m[i, k >> 6] ^= _ONE << np.uint64(k & 63)


@njit(cache=True)
def _inject(x: np.ndarray, z: np.ndarray, q: int, p: int, k: int) -> None:
    """Pauli ``p`` (1 X, 2 Y, 3 Z) on qubit ``q`` as fault ``k``."""
    if p <= 2:
        _flip(x, q, k)
    if p >= 2:
        _flip(z, q, k)


@njit(cache=True)
def _propagate(names: np.ndarray, q0: np.ndarray, q1: np.ndarray, n_qubits: int) -> tuple:
    """Push every elementary fault (order of ``faults._fault_list``) to the end of the circuit.

    Returns X and Z frames per qubit and flips per record, as bit sets over the faults. ``names``: ``faults._OPCODE``.
    """
    n_faults, n_meas = 0, 0
    for c in names:
        n_faults += 1 if c in {5, 6} else 4 if c in {7, 8} else 15 if c in {3, 4} else 3
        n_meas += 1 if 5 <= c <= 8 else 0
    words = max((n_faults + 63) // 64, 1)
    x = np.zeros((n_qubits, words), dtype=np.uint64)
    z = np.zeros((n_qubits, words), dtype=np.uint64)
    rec = np.zeros((max(n_meas, 1), words), dtype=np.uint64)
    k, r = 0, 0
    for i in range(len(names)):
        c, a, b = names[i], q0[i], q1[i]
        live = (k + 63) >> 6  # words holding faults injected so far (the later ones are still zero)
        if c == 3:
            for w in range(live):
                x[b, w] ^= x[a, w]
                z[a, w] ^= z[b, w]
        elif c == 4:
            for w in range(live):
                z[a, w] ^= x[b, w]
                z[b, w] ^= x[a, w]
        elif c == 2:
            for w in range(live):
                x[a, w], z[a, w] = z[a, w], x[a, w]
        elif c <= 1:
            for w in range(live):
                x[a, w] = 0
                z[a, w] = 0
        elif c <= 8:
            src = x if c in {5, 7} else z
            for w in range(live):
                rec[r, w] ^= src[a, w]
            if c >= 7:
                for w in range(live):
                    x[a, w] = 0
                    z[a, w] = 0
        if 5 <= c <= 8:
            _flip(rec, r, k)
            k += 1
            r += 1
        if c in {3, 4}:
            for pa in range(4):
                for pb in range(4):
                    if pa or pb:
                        if pa:
                            _inject(x, z, a, pa, k)
                        if pb:
                            _inject(x, z, b, pb, k)
                        k += 1
        elif c not in {5, 6}:
            for p in range(1, 4):
                _inject(x, z, a, p, k)
                k += 1
    return x, z, rec, n_faults


@njit(cache=True)
def _keys_from(cols: np.ndarray, idx: np.ndarray, ptr: np.ndarray, n_faults: int) -> np.ndarray:
    """Per-fault keys: bit j of key k = bit k of the XOR of the columns ``idx[ptr[j]:ptr[j + 1]]``."""
    out = np.zeros(n_faults, dtype=np.uint64)
    acc = np.zeros(cols.shape[1], dtype=np.uint64)
    for j in range(len(ptr) - 1):
        acc[:] = 0
        for c in idx[ptr[j] : ptr[j + 1]]:
            for w in range(len(acc)):
                acc[w] ^= cols[c, w]
        bit = _ONE << np.uint64(j)
        for k in range(n_faults):
            if (acc[k >> 6] >> np.uint64(k & 63)) & _ONE:
                out[k] |= bit
    return out


@njit(cache=True)
def fault_keys(
    names: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    n_qubits: int,
    sz_idx: np.ndarray,
    sz_ptr: np.ndarray,
    sx_idx: np.ndarray,
    sx_ptr: np.ndarray,
    key_idx: np.ndarray,
    key_ptr: np.ndarray,
) -> tuple:
    """Keys (X syndrome, Z syndrome, detector signature) of every elementary fault.

    Each group is given as (index, pointer) arrays: the qubits (records) each key bit XORs.
    """
    x, z, rec, n_faults = _propagate(names, q0, q1, n_qubits)
    return (
        _keys_from(x, sz_idx, sz_ptr, n_faults),
        _keys_from(z, sx_idx, sx_ptr, n_faults),
        _keys_from(rec, key_idx, key_ptr, n_faults),
    )


@njit(cache=True)
def _weight(dense: bool, keys: np.ndarray, vals: np.ndarray, s: np.uint64, above: np.int8) -> np.int8:
    """Coset-leader weight of syndrome ``s``: ``vals[s]`` (dense table), or looked up in the sorted ``keys``."""
    v = np.int64(s)
    if dense:
        return vals[v]
    i = np.searchsorted(keys, v)
    return vals[i] if i < len(keys) and keys[i] == v else above


@njit(cache=True)
def _grow(a: np.ndarray) -> np.ndarray:
    b = np.empty(2 * len(a), dtype=a.dtype)
    b[: len(a)] = a
    return b


@njit(cache=True)
def _enumerate_faults(
    ux: np.ndarray,
    uz: np.ndarray,
    uc: np.ndarray,
    counts: np.ndarray,
    x_dense: bool,
    x_keys: np.ndarray,
    x_vals: np.ndarray,
    z_dense: bool,
    z_keys: np.ndarray,
    z_vals: np.ndarray,
    t: int,
    collect: bool,
    first_fail: bool,
    n_anchor: int,
    counts_old: np.ndarray,
) -> tuple:
    """The undetected bad fault sets of size at most ``t`` (at most 3).

    Returns failures and raw failures per size, then (only with ``collect``) the failing X and Z syndromes and the key
    indices of each set for sizes 1, 2, 3, flattened. ``first_fail`` stops at the first bad set. Only sets with a key
    index below ``n_anchor`` count, each raw count less the product of ``counts_old`` (``verify.check_added``: the sets
    with a new fault).
    """
    nk = len(ux)
    above = np.int8(t + 1)
    fails, raw = np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)
    fx, fz = np.empty(1024, dtype=np.uint64), np.empty(1024, dtype=np.uint64)
    r1, r2, r3 = np.empty(1024, dtype=np.int64), np.empty(1024, dtype=np.int64), np.empty(1024, dtype=np.int64)
    nf, n1, n2, n3 = 0, 0, 0, 0
    done = False
    for k in range(min(nk, n_anchor)):
        if uc[k] == 0 and (
            _weight(x_dense, x_keys, x_vals, ux[k], above) > 1 or _weight(z_dense, z_keys, z_vals, uz[k], above) > 1
        ):
            fails[0] += 1
            raw[0] += counts[k] - counts_old[k]
            if collect:
                if nf == len(fx):
                    fx, fz = _grow(fx), _grow(fz)
                if n1 == len(r1):
                    r1 = _grow(r1)
                fx[nf], fz[nf], r1[n1] = ux[k], uz[k], k
                nf += 1
                n1 += 1
            done = first_fail
            if done:
                break
    if t >= 2 and not done:
        order = np.argsort(uc, kind="mergesort")  # by signature, members in increasing key index
        sorted_c = uc[order]
        new = np.ones(nk, dtype=np.bool_)
        new[1:] = sorted_c[1:] != sorted_c[:-1]
        start = np.append(np.nonzero(new)[0], nk)
        sig = sorted_c[start[:-1]]
        for g in range(len(sig)):
            for p in range(start[g], start[g + 1]):
                a = order[p]
                if a >= n_anchor:  # members in increasing key order
                    break
                for q in range(p + 1, start[g + 1]):
                    b = order[q]
                    ex, ez = ux[a] ^ ux[b], uz[a] ^ uz[b]
                    if (
                        _weight(x_dense, x_keys, x_vals, ex, above) > 2
                        or _weight(z_dense, z_keys, z_vals, ez, above) > 2
                    ):
                        fails[1] += 1
                        raw[1] += counts[a] * counts[b] - counts_old[a] * counts_old[b]
                        if collect:
                            if nf == len(fx):
                                fx, fz = _grow(fx), _grow(fz)
                            if n2 + 2 > len(r2):
                                r2 = _grow(r2)
                            fx[nf], fz[nf], r2[n2], r2[n2 + 1] = ex, ez, a, b
                            nf += 1
                            n2 += 2
                        done = first_fail
                        if done:
                            break
                if done:
                    break
            if done:
                break
        if t >= 3 and not done:
            # by pairs of signature groups (ga <= gb, the cancelling group gc >= gb: each multiset of groups once),
            # members in key order within a group; a triple, a pair or a whole group triple is skipped when the sum of
            # its classes' leader weights is <= 3 for both types (leader weight is subadditive, so it cannot fail)
            lwx, lwz = np.empty(nk, dtype=np.int8), np.empty(nk, dtype=np.int8)
            for k in range(nk):
                lwx[k] = _weight(x_dense, x_keys, x_vals, ux[k], above)
                lwz[k] = _weight(z_dense, z_keys, z_vals, uz[k], above)
            ng = len(sig)
            gmx, gmz = np.zeros(ng, dtype=np.int8), np.zeros(ng, dtype=np.int8)
            gmin = np.empty(ng, dtype=np.int64)
            for g in range(ng):
                gmin[g] = order[start[g]]
                for p in range(start[g], start[g + 1]):
                    gmx[g] = max(gmx[g], lwx[order[p]])
                    gmz[g] = max(gmz[g], lwz[order[p]])
            for ga in range(ng):
                for gb in range(ga, ng):
                    target = sig[ga] ^ sig[gb]
                    gc = np.searchsorted(sig, target)
                    if gc >= ng or sig[gc] != target or gc < gb:
                        continue
                    if gmx[ga] + gmx[gb] + gmx[gc] <= 3 and gmz[ga] + gmz[gb] + gmz[gc] <= 3:
                        continue
                    if min(gmin[ga], gmin[gb], gmin[gc]) >= n_anchor:
                        continue
                    for pa in range(start[ga], start[ga + 1]):
                        a = order[pa]
                        for pb in range(pa if gb == ga else start[gb], start[gb + 1]):
                            b = order[pb]
                            sx_, sz_ = lwx[a] + lwx[b], lwz[a] + lwz[b]
                            if sx_ + gmx[gc] <= 3 and sz_ + gmz[gc] <= 3:
                                continue
                            xab, zab = ux[a] ^ ux[b], uz[a] ^ uz[b]
                            cab, oab = counts[a] * counts[b], counts_old[a] * counts_old[b]
                            for pc in range(pb if gc == gb else start[gc], start[gc + 1]):
                                c = order[pc]
                                if a >= n_anchor and b >= n_anchor and c >= n_anchor:
                                    continue
                                if sx_ + lwx[c] <= 3 and sz_ + lwz[c] <= 3:
                                    continue
                                ex, ez = xab ^ ux[c], zab ^ uz[c]
                                if (
                                    _weight(x_dense, x_keys, x_vals, ex, above) > 3
                                    or _weight(z_dense, z_keys, z_vals, ez, above) > 3
                                ):
                                    fails[2] += 1
                                    raw[2] += cab * counts[c] - oab * counts_old[c]
                                    if collect:
                                        if nf == len(fx):
                                            fx, fz = _grow(fx), _grow(fz)
                                        if n3 + 3 > len(r3):
                                            r3 = _grow(r3)
                                        i, j, k = min(a, b, c), 0, max(a, b, c)
                                        j = a + b + c - i - k
                                        fx[nf], fz[nf], r3[n3], r3[n3 + 1], r3[n3 + 2] = ex, ez, i, j, k
                                        nf += 1
                                        n3 += 3
                                    done = first_fail
                                    if done:
                                        break
                            if done:
                                break
                        if done:
                            break
                    if done:
                        break
                if done:
                    break
    return fails, raw, fx[:nf], fz[:nf], r1[:n1], r2[:n2], r3[:n3]


def enumerate_faults(
    ux: np.ndarray,
    uz: np.ndarray,
    uc: np.ndarray,
    counts: np.ndarray,
    x_dense: bool,
    x_keys: np.ndarray,
    x_vals: np.ndarray,
    z_dense: bool,
    z_keys: np.ndarray,
    z_vals: np.ndarray,
    t: int,
    collect: bool,
    first_fail: bool,
    n_anchor: int,
    counts_old: np.ndarray,
) -> tuple:
    """``_enumerate_faults`` with its triples sorted (i <= j <= k, lexicographic): the samples of the lists rely on it."""
    fails, raw, fx, fz, r1, r2, r3 = _enumerate_faults(
        ux,
        uz,
        uc,
        counts,
        x_dense,
        x_keys,
        x_vals,
        z_dense,
        z_keys,
        z_vals,
        t,
        collect,
        first_fail,
        n_anchor,
        counts_old,
    )
    m = len(r3) // 3
    if m > 1:
        rows = r3.reshape(-1, 3)
        nk = np.int64(len(ux))  # i, j, k < nk: one int64 key per triple (nk^3 < 2^63 up to ~2e6 key classes)
        perm = np.argsort((rows[:, 0] * nk + rows[:, 1]) * nk + rows[:, 2])
        r3 = np.ascontiguousarray(rows[perm]).reshape(-1)
        lo = len(fx) - m
        fx, fz = fx.copy(), fz.copy()
        fx[lo:], fz[lo:] = fx[lo:][perm], fz[lo:][perm]
    return fails, raw, fx, fz, r1, r2, r3


def warm() -> None:
    """Compile the kernels or load them from numba's disk cache (before the search starts its workers)."""
    i, b = np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    p = np.array([0, 1], dtype=np.int64)
    kx, kz, kc = fault_keys(np.zeros(1, dtype=np.uint8), i, i - 1, 1, i, p, i, p, b, p[:1])
    v = np.zeros(1, dtype=np.int8)
    u = np.zeros(1, dtype=np.uint64)
    c = np.ones(1, dtype=np.int64)
    z = np.zeros(1, dtype=np.int64)
    enumerate_faults(u, u, u, c, True, b, v, True, b, v, 1, False, False, 1, z)
    enumerate_faults(u, u, u, c, False, i, v, False, i, v, 1, True, True, 1, z)
    del kx, kz, kc
