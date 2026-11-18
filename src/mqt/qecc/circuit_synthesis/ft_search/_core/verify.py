# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exhaustive fault-tolerance check for postselected CSS state preparation.

A circuit is fault tolerant to t faults if every set of s <= t elementary faults either flips a detector (the shot is
rejected) or leaves output errors E_x, E_z with min wt(E * g) <= s over the stabilizers g of the state of the other
type. X and Z parts are judged separately (as in Peham et al.).

Acceptability depends only on syndromes, so each fault reduces to a key (X syndrome, Z syndrome, detector signature)
and a fault set to the XOR of its keys; a table gives the coset-leader weight of each syndrome. A fault may be used
twice: such a set equals a smaller one judged with a looser threshold, so it adds no false failure. More than 64
detectors are compressed by a random linear map; a collision can only make a detected set look undetected, so the check
stays conservative. The key computation and the enumeration are compiled (``kernels``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, overload

import numpy as np
import stim

from mqt.qecc import mod2

from . import memo
from .circuit import MEASUREMENTS, Circuit
from .faults import _fault_list, _first_fault, _op_arrays
from .kernels import _propagate, enumerate_faults, fault_keys

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ....codes.core.css_code import CSSCode


@dataclass
class CSSState:
    """Stabilizer of a CSS state on the data qubits.

    Attributes:
        sx: X-type generators (rows over data qubits). Detect Z errors.
        sz: Z-type generators, including Z-type logicals the state is stabilized by. Detect X errors.
    """

    sx: np.ndarray
    sz: np.ndarray

    @classmethod
    def zero_state(cls, hx: np.ndarray, hz: np.ndarray, lz: np.ndarray) -> CSSState:
        """|0...0>_L of a CSS code: X stabilizers Hx; Z stabilizers Hz plus the Z logicals."""
        return cls(np.asarray(hx, dtype=np.uint8) % 2, np.vstack([hz, lz]).astype(np.uint8) % 2)

    @classmethod
    def plus_state(cls, hx: np.ndarray, hz: np.ndarray, lx: np.ndarray) -> CSSState:
        """|+...+>_L of a CSS code."""
        return cls(np.vstack([hx, lx]).astype(np.uint8) % 2, np.asarray(hz, dtype=np.uint8) % 2)


def state_of_code(code: CSSCode, *, zero_state: bool = True) -> CSSState:
    """The logical zero (or plus) state of all logical qubits of ``code`` as the search's target state."""
    u8 = lambda a: np.asarray(a, dtype=np.uint8) % 2
    return (
        CSSState.zero_state(u8(code.Hx), u8(code.Hz), u8(code.Lz))
        if zero_state
        else CSSState.plus_state(u8(code.Hx), u8(code.Hz), u8(code.Lx))
    )


def gf2_row_basis(m: np.ndarray) -> np.ndarray:
    """Linearly independent rows spanning the row space of ``m`` over GF(2), in reduced row echelon form."""
    a = np.ascontiguousarray(np.asarray(m, dtype=np.uint8) % 2)
    key = memo.digest(a.tobytes(), a.shape)
    hit = memo.get("basis", key)
    if hit is None:
        reduced, rank, _, _ = mod2.row_echelon(a.astype(np.int64), full=True)
        hit = np.asarray(reduced[:rank], dtype=np.uint8)
        memo.put("basis", key, hit, limit=256)
    return np.array(hit, dtype=np.uint8)


def _pack(bits: np.ndarray) -> np.ndarray:
    """Pack rows of <= 64 bits into uint64."""
    bits = np.asarray(bits, dtype=np.uint64)
    if bits.shape[1] == 0:
        return np.zeros(bits.shape[0], dtype=np.uint64)
    weights = np.left_shift(np.uint64(1), np.arange(bits.shape[1], dtype=np.uint64))
    return (bits * weights).sum(axis=1, dtype=np.uint64)


def _leader_levels(h: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted syndromes of all errors of weight at most ``t`` and their minimum weight.

    Breadth-first: level w = (level w - 1 XOR one column) minus everything seen.
    """
    cols = np.unique(_pack(h.T).astype(np.int64))
    keys, vals = [np.zeros(1, dtype=np.int64)], [np.zeros(1, dtype=np.int8)]
    seen = keys[0]
    front = keys[0]
    for w in range(1, t + 1):
        step = max(1, (1 << 24) // max(len(cols), 1))
        cand = np.unique(
            np.concatenate([(front[i : i + step, None] ^ cols[None, :]).ravel() for i in range(0, len(front), step)])
        )
        new = cand[~np.isin(cand, seen, assume_unique=True)]
        if not len(new):
            break
        keys.append(new)
        vals.append(np.full(len(new), w, dtype=np.int8))
        seen = np.union1d(seen, new)
        front = new
    k, v = np.concatenate(keys), np.concatenate(vals)
    o = np.argsort(k)
    return k[o], v[o]


class SparseLeader:
    """``leader_weights`` for many rows: the syndromes of weight <= t, sorted; any other syndrome weighs t + 1.

    Indexed like the dense table, by an int or an integer array.
    """

    def __init__(self, h: np.ndarray, t: int) -> None:
        """Build the table for the rows ``h`` and fault distance ``t``."""
        self.keys, self.vals = _leader_levels(h, t)
        self.t = t

    @overload
    def __getitem__(self, idx: int) -> int: ...
    @overload
    def __getitem__(self, idx: np.ndarray) -> np.ndarray: ...
    def __getitem__(self, idx: int | np.ndarray) -> int | np.ndarray:
        """Leader weight of a syndrome (an int) or of each syndrome in an integer array."""
        a = np.asarray(idx, dtype=np.int64)
        pos = np.minimum(np.searchsorted(self.keys, a), len(self.keys) - 1)
        out = np.where(self.keys[pos] == a, self.vals[pos], np.int8(self.t + 1)).astype(np.int8)
        return out if a.ndim else int(out)


def leader_weights(h: np.ndarray, t: int) -> np.ndarray | SparseLeader:
    """Minimum weight of an error with each syndrome w.r.t. independent rows ``h``; ``t + 1`` if above t.

    A dense table up to 20 rows, else a ``SparseLeader``. Memoized: callers must not modify the table.
    """
    h = np.ascontiguousarray(np.asarray(h, dtype=np.uint8) % 2)
    key = memo.digest(h.tobytes(), h.shape, t)
    hit = memo.get("leader", key)
    if isinstance(hit, (np.ndarray, SparseLeader)):
        return hit
    r, _n = h.shape
    if r > 20:
        out = SparseLeader(h, t)
    else:
        out = np.full(1 << r, t + 1, dtype=np.int8)
        k, v = _leader_levels(h, t)
        out[k] = v
    memo.put("leader", key, out, limit=64)
    return out


@dataclass
class WeightedTarget(CSSState):
    """Ancilla target of a grouped block (``BlockSpec.groups``).

    Errors of the type that propagates back onto the data (``back``: "Z" judged by the ``sx`` rows, "X" by ``sz``)
    weigh the minimum weight of ``gamma`` e modulo the measured checks ``hd`` (``gamma``: support x ancilla qubits). A
    single fault on an ancilla qubit coupled to two data qubits can be a weight-2 data error.
    """

    gamma: np.ndarray
    hd: np.ndarray
    back: str = "Z"


def target_key(state: CSSState) -> list[object]:
    """The parts of a memo key that identify a target: its rows and, for a ``WeightedTarget``, its weights."""
    items: list[object] = [state.sx.tobytes(), state.sx.shape, state.sz.tobytes(), state.sz.shape]
    if isinstance(state, WeightedTarget) and state.gamma is not None:
        gamma, hd = state.gamma, state.hd
        assert hd is not None, "a weighted target has both gamma and hd"
        items += ["weighted", state.back, gamma.tobytes(), gamma.shape, hd.tobytes(), hd.shape]
    return items


def weighted_leader(basis: np.ndarray, gamma: np.ndarray, hd: np.ndarray, t: int) -> np.ndarray | SparseLeader:
    """``leader_weights`` for ancilla errors e (syndromes w.r.t. ``basis``), weighed by the data errors they cause.

    The weight of e is min |gamma e + h| over h in rowspace(hd). Syndromes that no data error of weight <= t reaches weigh t + 1.
    """
    basis = np.ascontiguousarray(np.asarray(basis, dtype=np.uint8) % 2)
    gamma = np.asarray(gamma, dtype=np.uint8) % 2
    hd = np.asarray(hd, dtype=np.uint8).reshape(-1, gamma.shape[0]) % 2
    key = memo.digest(basis.tobytes(), basis.shape, gamma.tobytes(), gamma.shape, hd.tobytes(), hd.shape, t)
    hit = memo.get("weighted", key)
    if isinstance(hit, (np.ndarray, SparseLeader)):
        return hit
    s, na = gamma.shape
    # columns of [gamma | hd^T] as s-bit integers; elimination keeps, per pivot, which columns add up to it
    cols = [int(sum(int(gamma[i, a]) << i for i in range(s))) for a in range(na)]
    cols += [int(sum(int(row[i]) << i for i in range(s))) for row in hd]
    piv: dict[int, tuple[int, int]] = {}  # lowest bit -> (vector, column combination)
    for j, col in enumerate(cols):
        v, comb = col, 1 << j
        while v:
            lb = v & -v
            if lb not in piv:
                piv[lb] = (v, comb)
                break
            pv, pc = piv[lb]
            v, comb = v ^ pv, comb ^ pc
    rows = [int(sum(int(b[a]) << a for a in range(na))) for b in basis]
    amask = (1 << na) - 1
    best: dict[int, int] = {}
    for w in range(t + 1):
        for sup in itertools.combinations(range(s), w):
            v = sum(1 << i for i in sup)
            comb = 0
            while v:
                lb = v & -v
                if lb not in piv:
                    break
                pv, pc = piv[lb]
                v, comb = v ^ pv, comb ^ pc
            if v:
                continue  # not reachable from the ancilla
            e = comb & amask
            k = sum(((r & e).bit_count() & 1) << i for i, r in enumerate(rows))
            best.setdefault(k, w)
    r = basis.shape[0]
    if r > 20:
        ks = np.array(sorted(best), dtype=np.int64)
        out = SparseLeader.__new__(SparseLeader)
        out.keys, out.vals, out.t = ks, np.array([best[k] for k in ks], dtype=np.int8), t
    else:
        out = np.full(1 << r, t + 1, dtype=np.int8)
        for k, w in best.items():
            out[k] = w
    memo.put("weighted", key, out, limit=64)
    return out


def state_leader(state: CSSState, err: str, t: int) -> np.ndarray | SparseLeader:
    """Leader-weight table for errors of type ``err`` ("X": judged by the ``sz`` rows, "Z": by ``sx``) of a target."""
    basis = gf2_row_basis(state.sz if err == "X" else state.sx)
    if isinstance(state, WeightedTarget) and state.back == err:
        return weighted_leader(basis, state.gamma, state.hd, t)
    return leader_weights(basis, t)


def weighted_failures(raw: dict, t: int, base: float = 1000.0) -> float:
    """Failing fault sets (``raw``: s -> count) weighted so that lower orders dominate; 0 iff FT."""
    return float(sum(v * base ** (t - s) for s, v in raw.items()))


@dataclass
class FTReport:
    """Outcome of :func:`check_ft`."""

    t: int
    num_faults: int
    num_keys: int
    failures: dict[int, int] = field(default_factory=dict)  # s -> number of failing key combinations
    raw_failures: dict[int, int] = field(default_factory=dict)  # s -> failing sets of elementary faults
    witnesses: list[list[str]] = field(default_factory=list)
    # with collect=True: data X / Z syndromes (w.r.t. gf2_row_basis(sz) / gf2_row_basis(sx)) of every failing set
    failing_x: list[int] = field(default_factory=list)
    failing_z: list[int] = field(default_factory=list)
    failing_reps: list[tuple[int, ...]] | _Reps = field(default_factory=list)  # representative elementary fault indices
    rep_arrays: list = field(default_factory=list)  # the same as (m_s x s) arrays per s (fast; ``combos``)

    @property
    def ok(self) -> bool:
        """Whether no fault set fails."""
        return not any(self.failures.values())


def _rows(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(index, pointer) arrays of the nonzero columns of each row of ``b``."""
    r, c = np.nonzero(b)
    return c.astype(np.int64), np.searchsorted(r, np.arange(b.shape[0] + 1)).astype(np.int64)


def _fault_keys(circ: Circuit, state: CSSState, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Key (X syndrome, Z syndrome, detector signature) of every elementary fault, in the order of ``_fault_list``.

    More than 64 detectors are compressed by a random linear map (seeded) to 64 bits.
    """
    sz = gf2_row_basis(state.sz)
    sx = gf2_row_basis(state.sx)
    names, q0, q1 = _op_arrays(circ)
    d = np.array(circ.data, dtype=np.int64)
    sz_idx, sz_ptr = _rows(sz)
    sx_idx, sx_ptr = _rows(sx)
    det = circ.detector_matrix().astype(np.int64)
    if det.shape[0] > 64:
        proj = np.random.default_rng(seed).integers(0, 2, size=(det.shape[0], 64), dtype=np.int64)
        det = ((det.T @ proj) % 2).T
    key_idx, key_ptr = _rows(det)
    return fault_keys(names, q0, q1, circ.num_qubits, d[sz_idx], sz_ptr, d[sx_idx], sx_ptr, key_idx, key_ptr)


def _group_keys(kx: np.ndarray, kz: np.ndarray, kc: np.ndarray) -> tuple[np.ndarray, ...]:
    """Fault classes (faults with the same key).

    Returns the distinct keys in lexicographic order, the first fault and size of each class, and each fault's class.
    """
    order = np.lexsort((kc, kz, kx))
    a, b, c = kx[order], kz[order], kc[order]
    new = np.ones(len(order), dtype=bool)
    new[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1]) | (c[1:] != c[:-1])
    starts = np.nonzero(new)[0]
    counts = np.diff(np.append(starts, len(order)))
    cls = np.empty(len(order), dtype=np.int64)
    cls[order] = np.cumsum(new) - 1
    return a[starts], b[starts], c[starts], order[starts], counts, cls


def fault_classes(circ: Circuit, state: CSSState) -> np.ndarray:
    """Class index of each elementary fault of ``circ``: faults in one class have the same key."""
    return _group_keys(*_fault_keys(circ, state, 0))[-1]


MAX_ROWS = 62  # syndromes and stabilizer masks are packed into one 64-bit integer (int64 masks in ``stabs``)


def check_size(state: CSSState) -> None:
    """Raise ValueError if the state has more than ``MAX_ROWS`` independent X or Z stabilizers."""
    for name, rows in (("X", state.sx), ("Z", state.sz)):
        r = gf2_row_basis(rows).shape[0] if np.size(rows) else 0
        if r > MAX_ROWS:
            msg = f"the state has {r} independent {name}-type stabilizers; the exact check supports at most {MAX_ROWS}"
            raise ValueError(msg)


def check_ft(
    circ: Circuit,
    state: CSSState,
    t: int,
    *,
    max_witnesses: int = 0,
    seed: int = 0,
    collect: bool = False,
    first_fail: bool = False,
) -> FTReport:
    """Check fault tolerance of a postselected CSS state preparation up to ``t`` faults.

    Args:
        circ: The circuit.
        state: The state it prepares.
        t: Number of faults, 1 to 3.
        max_witnesses: Failing fault sets to describe in ``FTReport.witnesses`` (default: none).
        seed: Seed of the detector compression.
        collect: Also record every failing fault set and its data syndromes.
        first_fail: Stop at the first failing set (``ok`` exact, the counts partial).

    Returns:
        The report.

    Raises:
        ValueError: ``t`` is above 3, or the state has too many stabilizers (``check_size``).
    """
    if t > 3:
        msg = "the exact check supports t = 1 to 3"
        raise ValueError(msg)
    check_size(state)
    kx, kz, kc = _fault_keys(circ, state, seed)
    ux, uz, uc, rep, counts, _ = _group_keys(kx, kz, kc)
    report = FTReport(t=t, num_faults=len(kx), num_keys=len(ux))
    wx, wz = state_leader(state, "X", t), state_leader(state, "Z", t)
    tables = (*_leader_arrays(wx), *_leader_arrays(wz))
    want = collect or max_witnesses > 0
    stop = first_fail and not collect and not max_witnesses
    keys = (np.ascontiguousarray(ux), np.ascontiguousarray(uz), np.ascontiguousarray(uc))
    cnt = np.ascontiguousarray(counts, dtype=np.int64)
    fails, raw, fx, fz, r1, r2, r3 = enumerate_faults(
        *keys, cnt, *tables, t, want, stop, len(ux), np.zeros(len(ux), dtype=np.int64)
    )
    for s in range(1, t + 1):
        report.failures[s] = int(fails[s - 1])
        report.raw_failures[s] = int(raw[s - 1])
    reps = [r1.reshape(-1, 1), r2.reshape(-1, 2), r3.reshape(-1, 3)][:t]
    if max_witnesses and any(len(c) for c in reps):
        fl = _fault_list(circ)
        for combos in reps:
            for combo in combos:
                if len(report.witnesses) >= max_witnesses:
                    break
                out = []
                for k in combo:
                    i, lab = fl[rep[int(k)]]
                    op = circ.ops[i]
                    out.append(f"{lab}@{op.name}{op.qubits}#{i}")
                report.witnesses.append(out)
    if collect:
        report.failing_x = fx.astype(np.int64).tolist()
        report.failing_z = fz.astype(np.int64).tolist()
        report.rep_arrays = [rep[a.astype(np.int64)] for a in reps]
        report.failing_reps = _Reps(report.rep_arrays)
    return report


def _leader_arrays(w: np.ndarray | SparseLeader) -> tuple[bool, np.ndarray, np.ndarray]:
    """A leader table as the kernels take it: (dense?, keys, values); a dense table has no keys."""
    if isinstance(w, np.ndarray):
        return True, np.zeros(0, dtype=np.int64), w
    return False, w.keys, w.vals


def _even_sets(rows: np.ndarray, h: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> int:
    """Sum of the Walsh-Hadamard terms of ``check_added`` over the key sets ``rows`` and the characters."""
    s = rows.shape[1]
    if s == 1:
        return int(h[rows[:, 0]].sum())
    a = rows[:, 0]
    if s == 2:
        same = rows[:, 0] == rows[:, 1]
        d = h[a[~same]] * h[rows[~same, 1]]
        e = h[a[same]] ** 2 + x2[a[same]]
        return int(d.sum()) + int(e.sum()) // 2
    b, c = rows[:, 1], rows[:, 2]
    ab, bc = a == b, b == c
    m = ~ab & ~bc
    total = int((h[a[m]] * h[b[m]] * h[c[m]]).sum())
    m = ab & ~bc
    total += int(((h[a[m]] ** 2 + x2[a[m]]) * h[c[m]]).sum()) // 2
    m = ~ab & bc
    total += int((h[a[m]] * (h[b[m]] ** 2 + x2[b[m]])).sum()) // 2
    m = ab & bc
    total += int((h[a[m]] ** 3 + 3 * h[a[m]] * x2[a[m]] + 2 * x3[a[m]]).sum()) // 6
    return total


def check_added(circ: Circuit, state: CSSState, t: int, old: Circuit, old_report: FTReport) -> dict[str, dict] | None:
    """The failures of ``circ``, which is ``old`` with checks or flags added, from the failure list of ``old``.

    Ops are matched by ``Circuit.tags``. The new ops must leave the data errors and records of ``old``'s faults as they
    were, and every new detector must consist of new records (true for checks and flags outside blocks). Uses ``old``'s
    complete failure list (``check_ft(old, collect=True)``) and enumerates only the fault sets with a new fault.

    A failing set of ``old`` stays undetected iff the new detector bits of its members cancel. A key class of ``old``
    splits by its members' bit patterns; the raw count of the undetected part is the mean over the characters c of the
    product of h_c = sum_p n_p (-1)^(c.p) (Walsh-Hadamard). A class used two or three times counts its new keys as a
    multiset, by Burnside: (h^2 + x_2) / 2 and (h^3 + 3 h x_2 + 2 x_3) / 6 with x_k = sum_p n_p^k (-1)^(c.(k p)).
    With n_p replaced by [n_p > 0] the same counts the new key sets.

    Returns:
        {"raw": s -> failing fault sets, "added": s -> failing key sets with a new fault, "failures": s -> failing key
        sets (a key with both old and new faults may count twice)}, or None where the method does not apply.
    """
    if circ.tags is None or old.tags is None:
        return None
    if len(set(circ.tags)) != len(circ.tags) or len(set(old.tags)) != len(old.tags):
        return None
    pos_old = {tg: i for i, tg in enumerate(old.tags)}
    new_ops = np.array([tg not in pos_old for tg in circ.tags])
    meas = [i for i, op in enumerate(circ.ops) if op.name in MEASUREMENTS]
    new_rec = {r for r, i in enumerate(meas) if new_ops[i]}
    dets = [list(d) for d in circ.detectors] if circ.detectors is not None else [[r] for r in range(len(meas))]
    new_dets = [d for d in dets if any(r in new_rec for r in d)]
    if (
        any(r not in new_rec for d in new_dets for r in d)
        or len(dets) - len(new_dets) != old.detector_matrix().shape[0]
    ):
        return None
    k = len(new_dets)
    if k == 0 or k > 6:
        return None
    # new detector bits of every fault of circ, then of old's faults (same op, same fault within the op)
    names, q0, q1 = _op_arrays(circ)
    _, _, rec, n_faults = _propagate(names, q0, q1, circ.num_qubits)
    shifts = np.arange(n_faults, dtype=np.uint64)
    words, offs = shifts >> np.uint64(6), shifts & np.uint64(63)
    pattern = np.zeros(n_faults, dtype=np.int64)
    for b, d in enumerate(new_dets):
        par = np.zeros(n_faults, dtype=np.uint64)
        for r in d:
            par ^= (rec[r][words.astype(np.int64)] >> offs) & np.uint64(1)
        pattern |= par.astype(np.int64) << b
    f_new, f_old = _first_fault(circ), _first_fault(old)
    op_old = np.array([pos_old.get(tg, -1) for tg in circ.tags], dtype=np.int64)
    old_ops = np.nonzero(~new_ops)[0]
    if len(old_ops) != len(old.ops):
        return None
    pat_old = np.zeros(int(f_old[-1]), dtype=np.int64)
    for i in old_ops:  # the old op's faults, in the same order
        j = op_old[i]
        pat_old[f_old[j] : f_old[j + 1]] = pattern[f_new[i] : f_new[i + 1]]
    # the old failing key sets that stay undetected (raw count, exact)
    cls = fault_classes(old, state)
    n_cls = int(cls.max()) + 1 if len(cls) else 0
    npat = 1 << k
    n_p = np.zeros((n_cls, npat), dtype=np.int64)
    np.add.at(n_p, (cls, pat_old), 1)
    sign = 1 - 2 * np.array([[(c & p).bit_count() & 1 for p in range(npat)] for c in range(npat)])
    h = n_p @ sign.T  # (classes x characters): sum_p n_p chi_c(p)
    x2 = (n_p**2).sum(axis=1)[:, None]  # a pattern taken twice adds up to 0: chi = 1
    x3 = (n_p**3) @ sign.T
    ind = (n_p > 0).astype(np.int64)  # new keys: an old class with a pattern some member has
    hi = ind @ sign.T
    xi = ind.sum(axis=1)[:, None]
    raw: dict[int, int] = {}
    kept: dict[int, int] = {}
    for s_, arr in enumerate(old_report.rep_arrays, start=1):
        if not len(arr):
            raw[s_] = kept[s_] = 0
            continue
        rows = np.sort(cls[arr], axis=1)
        raw[s_] = _even_sets(rows, h, x2, x3) // npat
        kept[s_] = _even_sets(rows, hi, xi, hi) // npat
    # the sets with a new fault: keys holding new faults first, then the enumeration from them only
    is_new = np.zeros(n_faults, dtype=bool)
    for i in np.nonzero(new_ops)[0]:
        is_new[f_new[i] : f_new[i + 1]] = True
    ux, uz, uc, _, counts, cid = _group_keys(*_fault_keys(circ, state, 0))
    n_new = np.bincount(cid, weights=is_new, minlength=len(ux)).astype(np.int64)
    rank = np.argsort(n_new == 0, kind="stable")  # anchors (keys with a new fault) first
    ux, uz, uc = ux[rank], uz[rank], uc[rank]
    counts = counts[rank].astype(np.int64)
    old_counts = counts - n_new[rank]
    n_anchor = int((n_new > 0).sum())
    wx, wz = state_leader(state, "X", t), state_leader(state, "Z", t)
    fails, raw_new, *_ = enumerate_faults(
        np.ascontiguousarray(ux),
        np.ascontiguousarray(uz),
        np.ascontiguousarray(uc),
        np.ascontiguousarray(counts),
        *_leader_arrays(wx),
        *_leader_arrays(wz),
        t,
        False,
        False,
        n_anchor,
        np.ascontiguousarray(old_counts),
    )
    added = {s_: int(fails[s_ - 1]) for s_ in range(1, t + 1)}
    total = {s_: raw.get(s_, 0) + int(raw_new[s_ - 1]) for s_ in range(1, t + 1)}
    failures = {s_: kept.get(s_, 0) + added[s_] for s_ in range(1, t + 1)}
    return {"raw": total, "added": added, "failures": failures}


class _Reps:
    """``failing_reps`` as a lazy sequence of tuples (millions at t = 3: built only if iterated)."""

    def __init__(self, arrays: list[np.ndarray]) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return sum(len(a) for a in self.arrays)

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        for a in self.arrays:
            for i in range(0, len(a), 4096):  # in chunks: one list of all rows can take GBs
                for row in a[i : i + 4096].tolist():
                    yield tuple(row)

    def __eq__(self, other: object) -> bool:
        return list(self) == list(other)  # ty: ignore[invalid-argument-type]  (compared with other sequences only)

    __hash__ = None  # a mutable sequence compared by value: not hashable


def check_noiseless(circ: Circuit, state: CSSState) -> None:
    """Assert that every detector (else every measurement) is deterministic 0 and the data end in ``state``."""
    if circ.detectors is not None:
        circ.to_stim().detector_error_model()  # raises on non-deterministic detectors
    sim = stim.TableauSimulator()
    sim.set_num_qubits(circ.num_qubits)
    for i, op in enumerate(circ.ops):
        if op.name in MEASUREMENTS and circ.detectors is None:
            (q,) = op.qubits
            exp = sim.peek_z(q) if op.name in {"M", "MR"} else sim.peek_x(q)
            if exp != 1:
                msg = f"measurement #{i} {op} not deterministic 0 (expectation {exp})"
                raise AssertionError(msg)
        sim.do(stim.Circuit(f"{op.name} " + " ".join(map(str, op.qubits))))
    if circ.detectors is not None:
        rec = np.array(sim.current_measurement_record(), dtype=np.uint8)
        par = (circ.detector_matrix().astype(np.int64) @ rec) % 2
        if par.any():
            msg = f"detectors {np.nonzero(par)[0].tolist()} have value 1"
            raise AssertionError(msg)
    n = circ.num_qubits
    for rows, p in ((state.sx, "X"), (state.sz, "Z")):
        for row in rows:
            ps = stim.PauliString(n)
            for k, v in enumerate(row):
                if v:
                    ps[circ.data[k]] = p
            if sim.peek_observable_expectation(ps) != 1:
                msg = f"state not stabilized by {ps}"
                raise AssertionError(msg)
