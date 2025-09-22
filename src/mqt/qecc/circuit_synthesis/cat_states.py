# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods for preparing cat states and running experiments on them."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from functools import cache
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import stim
import z3

from .circuit_utils import relabel_qubits
from .noise import CircuitLevelNoise

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt


def cat_state_balanced_tree(w: int) -> stim.Circuit:
    """Build preparation circuit as perfect, balanced binary tree. Only works if w is a power of two.

    Circuit will be built over qubits start_idx, ..., start_idx+w
    Args:
        w: number of qubits of the cat state, assumed to be a power of two
        p: noise parameter
        start_idx: lowest index of qubit appearing in the circuit.

    Returns:
        noisy stim circuit preparing the cat state.
    """
    # Check if w is a power of two
    if (w & (w - 1)) != 0 or w == 0:
        msg = "w must be a power of two."
        raise ValueError(msg)

    circ = stim.Circuit()
    circ.append_operation("H", [0])

    def build_circ_rec(begin: int, end: int) -> None:
        if begin + 1 >= end:
            return
        mid = (begin + end) // 2
        circ.append_operation("CX", [begin, mid])
        build_circ_rec(begin, mid)
        build_circ_rec(mid, end)

    build_circ_rec(0, w)
    return circ


def cat_state_line(w: int) -> stim.Circuit:
    """Build preparation circuit only using cnots along a line.

    Circuit will be built over qubits start_idx, ..., start_idx+w
    Args:
        w: number of qubits of the cat state
        p: noise parameter
        start_idx: lowest index of qubit appearing in the circuit.

    Returns:
        noisy stim circuit preparing the cat state
    """
    circ = stim.Circuit()
    circ.append_operation("H", [0])
    for i in reversed(range(1, w)):
        circ.append("CX", [0, i])
    return circ


class CatStatePreparationExperiment:
    """Cat-state prep with post-selection, allowing ancilla size w2 ≤ data size w1.

    Qubit layout in the combined circuit:
      data:   0 .. w1-1
      ancilla: w1 .. w1+w2-1

    Transversal CX copies X from data -> ancilla on the first w2 data qubits:
      pairs: (data[i], ancilla[w1 + permutation[i]]) for i=0..w2-1
    """


class CatStatePreparationExperiment:
    """Cat-state prep with post-selection, allowing ancilla size w2 ≤ data size w1.

    Layout:
      data:    0 .. w1-1
      ancilla: w1 .. w1+w2-1

    Wiring (one parallel layer):
      pairs: (controls[i], w1 + permutation[i])  for i=0..w2-1
    """

    def __init__(
        self,
        circ1: stim.Circuit,  # data-prep circuit, size w1
        circ2: stim.Circuit,  # ancilla-prep circuit, size w2 (can be < w1)
        permutation: Sequence[int] | None = None,  # perm over 0..w2-1 (ancilla targets)
        controls: Sequence[int] | None = None,  # length-w2 list of data controls (subset of 0..w1-1)
    ) -> None:
        w1 = circ1.num_qubits
        w2 = circ2.num_qubits
        if w1 < 1 or w2 < 1:
            msg = "Both circuits must have at least one qubit."
            raise ValueError(msg)
        if w2 > w1:
            msg = "Ancilla (w2) must be ≤ data (w1)."
            raise ValueError(msg)

        self.w1 = w1
        self.w2 = w2
        self.total_qubits = w1 + w2

        # Defaults
        if controls is None:
            controls = list(range(w2))  # first w2 data qubits
        if permutation is None:
            permutation = list(range(w2))  # identity on ancilla

        # Build combined circuit:
        comb = stim.Circuit()
        comb += circ1
        comb += relabel_qubits(circ2, w1)  # ancilla shifted to [w1..w1+w2-1]

        # Wiring
        pairs = build_transversal_pairs(controls, permutation, w1=w1, w2=w2)
        append_transversal_cnot_pairs(comb, pairs)

        # Measure ancilla now (post-selection later in sampling)
        comb.append_operation("MR", list(range(w1, w1 + w2)))

        self.circ = comb

    # ---------- noisy variant ----------

    def _get_noisy_circ(self, p: float) -> stim.Circuit:
        """Return a noisy version of the combined circuit."""
        return CircuitLevelNoise(p, p, p, p).apply(self.circ)

    # ---------- sampling / stats ----------

    def sample_cat_state(
        self, p: float, n_samples: int = 1024, batch_size: int | None = None
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Run with circuit-level noise, post-select on ancilla ∈ {0^w2, 1^w2},
        and histogram the symmetric error weight on data.

        Returns:
            acceptance_rate, acceptance_rate_error,
            error_rates (length floor(w1/2)+1), error_rates_error
        """
        circ = self._get_noisy_circ(p)
        # Final, *noise-free* measurement of data qubits (like your previous code)
        circ.append("TICK")
        circ.append("MR", list(range(self.w1)))

        if batch_size is None:
            batch_size = n_samples
        if n_samples > 1e7:
            batch_size = int(1e7)

        total_samples = 0
        total_accepted = 0

        # histogram over symmetric data error weights (0..floor(w1/2))
        max_sym_w = self.w1 // 2
        hist_total = np.zeros(max_sym_w + 1, dtype=int)

        # number of recorded bits per shot = w2 (ancilla MR) + w1 (data MR) = total_qubits
        n_batches = int(np.ceil(n_samples / batch_size))
        for _ in range(n_batches):
            this_batch = min(batch_size, n_samples - total_samples)
            sampler = circ.compile_sampler()
            res = sampler.sample(this_batch).astype(int)  # shape: [this_batch, w2 + w1]
            total_samples += this_batch

            anc = res[:, : self.w2]  # ancilla measurements first
            data = res[:, self.w2 : self.w2 + self.w1]  # then data

            # post-select: ancilla is all-0 or all-1
            ok_rows = np.where(np.logical_or(np.all(anc == 0, axis=1), np.all(anc == 1, axis=1)))[0]
            if ok_rows.size == 0:
                continue

            data_ok = data[ok_rows, :]
            total_accepted += data_ok.shape[0]

            # symmetric data weight
            wts = data_ok.sum(axis=1)
            sym_wts = np.minimum(wts, self.w1 - wts).astype(int)

            # accumulate histogram
            hist, _ = np.histogram(sym_wts, bins=np.arange(max_sym_w + 2))
            hist_total += hist

        acceptance_rate = total_accepted / max(total_samples, 1)
        acceptance_rate_error = np.sqrt(acceptance_rate * max(1 - acceptance_rate, 0) / max(total_samples, 1))

        error_rates = hist_total / max(total_samples, 1)
        error_rates_error = np.sqrt(error_rates * np.maximum(1 - error_rates, 0) / max(total_samples, 1))

        return acceptance_rate, acceptance_rate_error, error_rates, error_rates_error

    # ---------- plotting ----------

    def plot_one_p(
        self, p: float, n_samples: int = 1024, batch_size: int | None = None, ax: plt.Axes | None = None
    ) -> None:
        ra, ra_err, hist, hist_err = self.sample_cat_state(p, n_samples, batch_size)
        x = np.arange(self.w1 // 2 + 1)
        if ax is None:
            _fig, ax = plt.subplots()

        cmap = plt.cm.plasma
        colors = cmap(np.linspace(0, 1, len(x)))

        bar_width = 0.8
        for xi, yi, err, color in zip(x, hist, hist_err, colors):
            ax.bar(
                xi,
                yi,
                width=bar_width,
                color=color,
                alpha=0.8,
                edgecolor="black",
                hatch="//",
                label=f"Error count {xi}" if xi == 0 else "",
            )
            ax.errorbar(xi, yi, yerr=err, fmt="none", capsize=5, color="black", linewidth=1.5)

        ax.set_xlabel("Number of data-qubit errors (symmetric)")
        ax.set_ylabel("Probability")
        ax.set_xticks(x)
        ax.set_yscale("log")
        ax.margins(0.2, 0.2)
        plt.title(f"Cat prep: w1={self.w1}, w2={self.w2}, p={p:.3f}. Acceptance = {ra:.3f} ± {ra_err:.3f}")
        plt.show()

    # ---------- sweep ----------

    def cat_prep_experiment(
        self, ps: list[float], shots_per_p: int | list[int]
    ) -> tuple[list[float], list[float], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        if isinstance(shots_per_p, list):
            assert len(shots_per_p) == len(ps)
        else:
            shots_per_p = [shots_per_p for _ in range(len(ps))]

        hists = None
        hists_err = None
        ras = []
        ra_errs = []
        for p, n_shots in zip(ps, shots_per_p):
            ra, ra_err, hist, hist_err = self.sample_cat_state(p, n_shots, batch_size=100000)
            ras.append(ra)
            ra_errs.append(ra_err)
            if hists is None:
                hists = hist
                hists_err = hist_err
            else:
                hists = np.vstack((hists, hist))
                hists_err = np.vstack((hists_err, hist_err))
        return ras, ra_errs, hists, hists_err


def transversal_cnot(
    circ1: stim.Circuit, circ2: stim.Circuit, permutation: list[int] | npt.NDArray[int] | None = None
) -> stim.Circuit:
    """Perform a transversal CNOT from circ1 to circ2."""
    # this function assumes that circ1 acts on the first w qubits and circ2 on the second w qubits
    w = circ1.num_qubits
    if permutation is None:
        permutation = list(range(w))
    circ = circ1 + circ2
    # make permuted transversal cnot
    for i in range(w):
        circ.append_operation("CX", [i, permutation[i] + w])
    return circ


def append_transversal_cnot_pairs(circ: stim.Circuit, pairs: Sequence[tuple[int, int]]) -> None:
    """Append a (possibly parallel) layer of CX using disjoint pairs."""
    if not pairs:
        return
    flat = []
    for c, t in pairs:
        flat.extend([c, t])
    circ.append_operation("CX", flat)


def build_transversal_pairs(
    controls: Sequence[int],  # length = w2, subset of 0..w1-1
    perm_targets: Sequence[int],  # permutation of 0..w2-1
    w1: int,  # data size
    w2: int,  # ancilla size
) -> list[tuple[int, int]]:
    """Returns list of (control, target) indices for a single parallel CX layer:
    (controls[i], w1 + perm_targets[i])  for i=0..w2-1.
    """
    if len(controls) != w2:
        msg = f"len(controls) must equal w2; got {len(controls)} vs {w2}"
        raise ValueError(msg)
    if sorted(set(controls)) != sorted(controls):
        # we require a set of distinct controls, order matters
        msg = "controls must be a list of distinct data-qubit indices"
        raise ValueError(msg)
    if not all(0 <= c < w1 for c in controls):
        msg = "controls indices must be in 0..w1-1"
        raise ValueError(msg)
    perm_targets = list(perm_targets)
    if sorted(perm_targets) != list(range(w2)):
        msg = "perm_targets must be a permutation of 0..w2-1"
        raise ValueError(msg)

    return [(controls[i], w1 + perm_targets[i]) for i in range(w2)]


def binary_tree_fault_gens(w: int, include_full: bool = False):
    assert w > 0
    assert (w & (w - 1)) == 0
    gens = []
    m = w.bit_length() - 1
    max_k = m if include_full else m - 1
    for k in range(max_k + 1):
        L = 1 << k
        gens.extend(((1 << L) - 1) << start for start in range(0, w, L))
    return gens


# =========================
# Bit helpers
# =========================


def apply_perm_mask(mask: int, P: list[int]) -> int:
    """Forward permutation: image bits go to positions P[i]."""
    out = 0
    for i, j in enumerate(P):
        if (mask >> i) & 1:
            out |= 1 << j
    return out


def bitcount(x: int) -> int:
    return x.bit_count()


def bits_of_mask(mask: int, w: int) -> list[int]:
    """Return [0/1]*w (LSB at index 0)."""
    return [(mask >> j) & 1 for j in range(w)]


def support_bits(mask: int, w: int) -> list[int]:
    return [i for i in range(w) if (mask >> i) & 1]


def ones_indices(mask: int, w: int) -> list[int]:
    return [j for j in range(w) if (mask >> j) & 1]


# =========================
# Pruned balanced tree circuits & generators
# =========================


def cat_state_pruned_balanced_circuit(w: int):
    """Prepare GHZ_w using a 2^m template with descending strides and
    prune any CX whose target >= w.

    Order is coarse->fine (stride 2^(m-1) down to 1), so every control
    has already joined the GHZ spine when it fans out.
    """
    if stim is None:
        msg = "stim is required for circuit extraction. Install `stim` or use --structure pruned_tree."
        raise RuntimeError(msg)

    if w <= 0:
        msg = "w must be >= 1"
        raise ValueError(msg)
    circ = stim.Circuit()
    circ.append_operation("H", [0])

    if w == 1:
        return circ

    m = math.ceil(math.log2(w))
    W = 1 << m  # template width

    # Descending strides: 2^(m-1), 2^(m-2), ..., 1
    for stride in (1 << k for k in range(m - 1, -1, -1)):
        step = 2 * stride
        for j in range(0, W, step):
            c = j
            t = j + stride
            if c < w and t < w:
                circ.append_operation("CX", [c, t])
    return circ


def _cx_forward(mask: int, c: int, t: int) -> int:
    if (mask >> c) & 1:
        mask ^= 1 << t
    return mask


def fault_gens_from_circuit(circ, include_full: bool = False) -> list[int]:
    """Single-fault outputs for a GHZ fanout circuit:
      • all singletons (X injected at a leaf at any time before it's touched),
      • for each CX pair (c,t) in sequence, inject X on c *just before that CX*
        and propagate through the remaining pairs.
    Stim may group many disjoint CX pairs in one op; we must flatten them.
    """
    if stim is None:
        msg = "stim is required for circuit extraction."
        raise RuntimeError(msg)

    w = circ.num_qubits
    ops: list[tuple[int, int]] = []
    for op in circ:
        if op.name != "CX":
            continue
        tgts = op.targets_copy()
        assert len(tgts) % 2 == 0
        for k in range(0, len(tgts), 2):
            c = tgts[k].value
            t = tgts[k + 1].value
            if c < w and t < w:
                ops.append((c, t))

    ALL = (1 << w) - 1
    gens = set()

    # all singletons
    gens.update(1 << q for q in range(w))

    # inject on control just before each CX, then propagate forward
    for idx, (c0, _) in enumerate(ops):
        mask = 1 << c0
        for c, t in ops[idx:]:
            mask = _cx_forward(mask, c, t)
        if not include_full and mask == ALL:
            continue
        gens.add(mask)

    return sorted(gens)


def pruned_tree_fault_gens(w: int, include_full: bool = False) -> list[int]:
    """Masked dyadic intervals: produces exactly the generator set you'd
    get from the pruned balanced tree schedule.
    """
    assert w > 0
    mask_all = (1 << w) - 1
    m = w.bit_length() - 1  # floor(log2 w)
    gens = []
    for k in range(m + 1):
        L = 1 << k
        for start in range(0, w, L):
            gen = (((1 << L) - 1) << start) & mask_all
            gens.append(gen)
    # dedupe
    gens = list(dict.fromkeys(gens))
    if not include_full:
        gens = [g for g in gens if g != mask_all]
    return gens


# =========================
# Degree sets & bad catalog
# =========================


@cache
def degree_sets_by_h(gens_key: tuple[int, ...], t: int):
    """Return [S_h] for h=0..t where S_h is the set of XORs of exactly h generators."""
    gens = list(gens_key)
    S = [set() for _ in range(t + 1)]
    S[0].add(0)
    for g in gens:
        for h in range(t, 0, -1):
            for s in list(S[h - 1]):
                S[h].add(s ^ g)
    return S


def as_key(gens: list[int]) -> tuple[int, ...]:
    return tuple(sorted(set(gens)))


def build_bad_catalog_cached(gens1: list[int], gens2: list[int], t: int):
    """Precompute:
      - bad_sets: dict x2 -> set of forbidden images
      - x2_list: list of all degree-<=t masks from circuit 2 to test
      - w: width
    Uses degree_sets_by_h(...) only once per (gens,t) pair (and it's cached).
    """
    w = max((g.bit_length() for g in gens1 + gens2), default=0)
    ALL = (1 << w) - 1

    S1_by_h = degree_sets_by_h(as_key(gens1), t)
    S2_by_h = degree_sets_by_h(as_key(gens2), t)

    # bucket S1 by weight for fast lookups
    S1_by_h_by_wt = []
    for h in range(t + 1):
        buckets = defaultdict(list)
        for m in S1_by_h[h]:
            buckets[bitcount(m)].append(m)
        S1_by_h_by_wt.append(buckets)

    bad_sets: dict[int, set] = defaultdict(set)
    x2_list: list[int] = []
    for h2 in range(1, t + 1):
        for x2 in S2_by_h[h2]:
            s = bitcount(x2)
            if s == 0:
                continue
            x2_list.append(x2)
            for h1 in range(1, t - h2 + 1):
                # equality case
                for b in S1_by_h_by_wt[h1].get(s, []):
                    if min(s, w - s) > (h1 + h2):
                        bad_sets[x2].add(b)
                # complement case
                for b in S1_by_h_by_wt[h1].get(w - s, []):
                    if min(w - s, s) > (h1 + h2):
                        bad_sets[x2].add(ALL ^ b)
    return bad_sets, x2_list, w


def find_violation_from_catalog(P: list[int], bad_sets: dict[int, set], x2_list: list[int]) -> tuple[int, int] | None:
    """Return (x2, image) if a forbidden image occurs, else None."""
    for x2 in x2_list:
        y = apply_perm_mask(x2, P)
        if y in bad_sets.get(x2, ()):
            return (x2, y)
    return None


def t_distinct_cat_exact_permutation_with_catalog(P: list[int], bad_sets: dict[int, set], x2_list: list[int]) -> bool:
    return find_violation_from_catalog(P, bad_sets, x2_list) is None


# =========================
# Catalog-guided local repair
# =========================


def find_perm_local_search(
    gens1: list[int],
    gens2: list[int],
    w: int,
    t: int,
    seed=1,
    restarts=16,
    max_iters=500000,
    init_perm: list[int] | None = None,
):
    rng = random.Random(seed)
    bad_sets, x2_list, w2 = build_bad_catalog_cached(gens1, gens2, t)
    assert w == w2
    all_cols = list(range(w))

    def setup_perm():
        if init_perm is not None:
            P = init_perm[:]
        else:
            P = list(range(w))
            rng.shuffle(P)
        inv = [0] * w
        for i, j in enumerate(P):
            inv[j] = i
        return P, inv

    for rs in range(restarts):
        P, invP = setup_perm()
        it = 0
        while it < max_iters:
            it += 1
            vio = find_violation_from_catalog(P, bad_sets, x2_list)
            if vio is None:
                return P, {"status": "sat", "iters": it, "restarts": rs, "bad_x2": len(bad_sets)}
            x2, y = vio
            S = support_bits(x2, w)
            T = set(ones_indices(y, w))
            comp_cols = [c for c in all_cols if c not in T]
            success = False
            for _ in range(64):
                i = rng.choice(S)
                c = rng.choice(comp_cols)
                k = invP[c]  # k ∉ S
                old_i, old_k = P[i], P[k]
                new_y = (y ^ (1 << old_i)) | (1 << c)  # (T - {old_i}) ∪ {c}
                if new_y not in bad_sets.get(x2, ()):
                    P[i], P[k] = P[k], P[i]
                    invP[old_i], invP[old_k] = invP[old_k], invP[old_i]
                    success = True
                    break
            if not success:
                i1, i2 = rng.sample(range(w), 2)
                j1, j2 = P[i1], P[i2]
                P[i1], P[i2] = j2, j1
                invP[j1], invP[j2] = i2, i1
    return None, {"status": "unknown", "iters": max_iters, "restarts": restarts}


# =========================
# SAT-guided local repair
# =========================


def parity_xor(xs):
    if z3 is None:
        msg = "z3 is required for --method sat."
        raise RuntimeError(msg)
    if not xs:
        return z3.BoolVal(False)
    acc = xs[0]
    for t in xs[1:]:
        acc = z3.Xor(acc, t)
    return acc


def mask_weight_bools(bits):
    if z3 is None:
        msg = "z3 is required for --method sat."
        raise RuntimeError(msg)
    return z3.Sum([z3.If(b, 1, 0) for b in bits])


@cache
def _cached_gen_bits(gens_key: tuple[int, ...], w: int):
    gens = list(gens_key)
    return [list(bits_of_mask(g, w)) for g in gens]


def _key_for_gens(gens: list[int]) -> tuple[int, ...]:
    return tuple(sorted(set(gens)))


def sat_find_counterexample_for_perm_from_gens(
    P: list[int],
    w: int,
    t: int,
    gens1: list[int],
    gens2: list[int],
    seed=1,
):
    if z3 is None:
        msg = "z3 is required for --method sat."
        raise RuntimeError(msg)

    z3.set_param("sat.random_seed", seed)

    g1_key = _key_for_gens(gens1)
    g2_key = _key_for_gens(gens2)
    G1_bits = _cached_gen_bits(g1_key, w)
    G2_bits = _cached_gen_bits(g2_key, w)
    n1 = len(G1_bits)
    n2 = len(G2_bits)

    invP = [0] * w
    for i, j in enumerate(P):
        invP[j] = i

    splits = [(h1, h2) for h2 in range(1, t + 1) for h1 in range(1, t - h2 + 1)]
    random.Random(seed).shuffle(splits)

    for h1, h2 in splits:
        Sbase = z3.Solver()

        u = [z3.Bool(f"u_{h1}_{i}") for i in range(n1)]  # circuit A
        v = [z3.Bool(f"v_{h2}_{j}") for j in range(n2)]  # circuit B

        Sbase.add(z3.PbEq([(u[i], 1) for i in range(n1)], h1))
        Sbase.add(z3.PbEq([(v[j], 1) for j in range(n2)], h2))

        x1_bits = []
        for j in range(w):
            terms1 = [u[i] for i in range(n1) if G1_bits[i][j]]
            x1_bits.append(parity_xor(terms1))

        y_bits = []
        for j in range(w):
            r = invP[j]
            terms2 = [v[k] for k in range(n2) if G2_bits[k][r]]
            y_bits.append(parity_xor(terms2))

        s = mask_weight_bools(y_bits)
        Sbase.add(s >= (h1 + h2 + 1))
        Sbase.add(s <= (w - (h1 + h2 + 1)))

        # Case A: y == x1
        Sa = z3.Solver()
        Sa.add(Sbase.assertions())
        for j in range(w):
            Sa.add(y_bits[j] == x1_bits[j])
        if Sa.check() == z3.sat:
            M = Sa.model()
            x2_mask = 0
            for j in range(n2):
                if z3.is_true(M[v[j]]):
                    x2_mask ^= gens2[j]
            y_mask = apply_perm_mask(x2_mask, P)
            return {"h1": h1, "h2": h2, "x2_mask": x2_mask, "y_mask": y_mask, "complement": False}

        # Case B: y == ~x1
        Sb = z3.Solver()
        Sb.add(Sbase.assertions())
        for j in range(w):
            Sb.add(y_bits[j] == z3.Not(x1_bits[j]))
        if Sb.check() == z3.sat:
            M = Sb.model()
            x2_mask = 0
            for j in range(n2):
                if z3.is_true(M[v[j]]):
                    x2_mask ^= gens2[j]
            y_mask = apply_perm_mask(x2_mask, P)
            return {"h1": h1, "h2": h2, "x2_mask": x2_mask, "y_mask": y_mask, "complement": True}

    return None  # no split found ⇒ permutation is t-fault-tolerant


def repair_once_by_swap(P: list[int], w: int, witness, rng=None, tries=128) -> bool:
    if rng is None:
        rng = random.Random()
    x2 = witness["x2_mask"]
    y = witness["y_mask"]
    S = [i for i in range(w) if (x2 >> i) & 1]
    T = {j for j in range(w) if (y >> j) & 1}

    inv = [0] * w
    for i, j in enumerate(P):
        inv[j] = i

    outside = [c for c in range(w) if c not in T]
    if not S or not outside:
        return False

    for _ in range(tries):
        i = rng.choice(S)
        c = rng.choice(outside)
        k = inv[c]  # k ∉ S
        P[i], P[k] = P[k], P[i]
        return True
    return False


def sat_guided_local_repair_from_gens(
    gens1: list[int],
    gens2: list[int],
    w: int,
    t: int,
    P_init: list[int] | None = None,
    max_iter: int = 1000,
    seed: int = 0,
):
    rng = random.Random(seed)
    if P_init is None:
        P = list(range(w))
        rng.shuffle(P)
    else:
        P = P_init[:]

    it = 0
    while it < max_iter:
        it += 1
        wit = sat_find_counterexample_for_perm_from_gens(P, w, t, gens1=gens1, gens2=gens2, seed=rng.randrange(1 << 30))
        if wit is None:
            return True, P, {"iters": it}
        ok = repair_once_by_swap(P, w, wit, rng=rng)
        if not ok:
            i1, i2 = rng.sample(range(w), 2)
            P[i1], P[i2] = P[i2], P[i1]
    return False, P, {"iters": it}


def _support_bits(mask: int, w: int) -> list[int]:
    out = []
    i = 0
    m = mask
    while m:
        if m & 1:
            out.append(i)
        m >>= 1
        i += 1
    return out


def cegar_permutation_sat(
    gens1: list[int],
    gens2: list[int],
    w: int,
    t: int,
    seed: int = 1,
    symmetry_fix: bool = True,
    max_rounds: int = 200000,
    batch_clauses: int = 1,
):
    """CEGAR over the permutation polytope:
      - Vars X_{i,j} are Bool, rows/cols one-hot.
      - Iterate: get model P; if no violation, SAT -> return P.
        Otherwise add blocking clause forbidding this exact assignment on the violating support S:
            not(AND_{i in S} X_{i, P[i]})
        (Optionally add a few clauses per round with batch_clauses>1.).

    Returns:
      (perm, stats) on SAT,
      (None, {'status':'unsat',...}) on UNSAT,
      (None, {'status':'unknown',...}) if round limit hit.
    """
    rng = random.Random(seed)
    # Precompute catalog once
    bad_sets, x2_list, w2 = build_bad_catalog_cached(gens1, gens2, t)
    assert w == w2

    # Z3 variables
    X = [[z3.Bool(f"X_{i}_{j}") for j in range(w)] for i in range(w)]
    S = z3.Solver()

    # one-hot rows and cols
    for i in range(w):
        S.add(z3.PbEq([(X[i][j], 1) for j in range(w)], 1))
    for j in range(w):
        S.add(z3.PbEq([(X[i][j], 1) for i in range(w)], 1))

    # Symmetry breaking
    if symmetry_fix and w > 0:
        S.add(X[0][0])

    def model_to_perm(M) -> list[int]:
        P = [-1] * w
        for i in range(w):
            for j in range(w):
                if z3.is_true(M[X[i][j]]):
                    P[i] = j
                    break
        return P

    rounds = 0
    added_clauses = 0
    while True:
        rounds += 1
        if rounds > max_rounds:
            return None, {"status": "unknown", "rounds": rounds - 1, "added_clauses": added_clauses}

        chk = S.check()
        if chk != z3.sat:
            return None, {"status": "unsat", "rounds": rounds - 1, "added_clauses": added_clauses}

        P = model_to_perm(S.model())
        # Check for a violation under current P
        vio = find_violation_from_catalog(P, bad_sets, x2_list)
        if vio is None:
            return P, {"status": "sat", "rounds": rounds - 1, "added_clauses": added_clauses}

        # Learn blocking clause(s)
        # Always block the current witness; optionally add a few more from random scan
        blocked = 0
        for _ in range(max(1, batch_clauses)):
            if _ == 0:
                x2, y = vio
            else:
                # Try to find another violation quickly by random probing
                # (keeps clause learning aggressive without re-solving)
                x2 = rng.choice(x2_list)
                y = apply_perm_mask(x2, P)
                if y not in bad_sets.get(x2, ()):
                    continue

            S_set = _support_bits(x2, w)  # rows in support
            # Under model P, each row i maps to column P[i]
            list = [z3.Not(X[i][P[i]]) for i in S_set]
            S.add(z3.Or(list))
            added_clauses += 1
            blocked += 1

        # On next loop, solver will return a different permutation (or UNSAT)


def _ft_w_4_cat_state() -> tuple[stim.Circuit, list[list[int]]]:
    circ = stim.Circuit()
    circ.append("RX", [4])
    circ.append("R", [0, 1, 2, 3])
    circ.append("CX", [4, 0])
    circ.append("CX", [0, 1])
    circ.append("CX", [1, 2])
    circ.append("CX", [2, 3])
    circ.append("CX", [3, 4])
    circ.append("MR", [4])
    return circ, [([4], [0, 1, 2, 3])]


def recursive_fuse_cat_state(w: int, t: int) -> tuple[stim.Circuit, list[list[int]]]:
    def _recurse(w1: int, w2: int) -> tuple[stim.Circuit, list[tuple[list[int], list[int]]]]:
        if w <= 0:
            msg = "w must be >= 1"
            raise ValueError(msg)

        if w1 < 4:
            c1, measurements_1 = cat_state_pruned_balanced_circuit(w1), []
        elif w1 == 4:
            c1, measurements_1 = _ft_w_4_cat_state()
        else:
            c1, measurements_1 = _recurse((w1 + 1) // 2, w1 // 2)

        if w2 < 4:
            c2, measurements_2 = cat_state_pruned_balanced_circuit(w2), []
        elif w2 == 4:
            c2, measurements_2 = _ft_w_4_cat_state()
        else:
            c2, measurements_2 = _recurse((w2 + 1) // 2, w2 // 2)

        # combine circuits
        circ = stim.Circuit()
        # map measurements to the end (assume measurements are at the end of each circuit)
        if w1 >= 4:
            m1 = {i: i for i in range(w1)} | {i: i + w2 for i in range(w1, c1.num_qubits)}
            circ += relabel_qubits(c1, m1)
            # remap measurement indices according to m1, m2
            measurements_1 = [
                ([m1[anc] for anc in ancillas], [m1[data] for data in data_qubits])
                for ancillas, data_qubits in measurements_1
            ]
        else:
            circ += c1

        if w2 >= 4:
            m2 = {i: i + w1 for i in range(w2)} | {i: i + c1.num_qubits + w2 for i in range(w2, c2.num_qubits)}
            circ += relabel_qubits(c2, m2)
            measurements_2 = [
                ([m2[anc] for anc in ancillas], [m2[data] for data in data_qubits])
                for ancillas, data_qubits in measurements_2
            ]
        else:
            circ += relabel_qubits(c2, w1)

        # interleaf measurements one by one
        measurements = []
        for i in range(max(len(measurements_1), len(measurements_2))):
            if i < len(measurements_1):
                measurements.append(measurements_1[i])
            if i < len(measurements_2):
                measurements.append(measurements_2[i])

        # add further measurements
        n_meas = min(t, w1, w2)
        # measure ZZ operator between n_meas pairs of data qubits
        new_measurements = []
        for i in range(n_meas):
            anc = circ.num_qubits
            circ.append("R", [anc])
            circ.append("CX", [i, anc])
            circ.append("CX", [i + w1, anc])
            circ.append("MR", [anc])
            new_measurements.append(anc)

        data_to_flip = list(range(w1)) if w1 < w2 else list(range(w1, w1 + w2))
        measurements.append((new_measurements, data_to_flip))

        return circ, measurements

    if w < 4:
        return cat_state_pruned_balanced_circuit(w), [([], list(range(w)))]

    if w == 4:
        return _ft_w_4_cat_state()

    return _recurse((w + 1) // 2, w // 2)


def _ancilla_controls_map(circ: stim.Circuit) -> dict[int, List[int]]:
    """For each ancilla qubit measured via an 'R ... CX ... MR' block,
    collect the list of *controls* that hit it as a CX target between its R and MR.
    """
    active_blocks: dict[int, List[int]] = {}  # anc -> list of controls
    anc_controls: dict[int, List[int]] = {}

    for op in circ:
        name = op.name
        tgts = op.targets_copy()

        if name == "R":
            # Start a fresh block for each reset target
            for t in tgts:
                q = t.value
                active_blocks[q] = []

        elif name == "CX":
            # For each pair (c, t): if that t is an active ancilla, record c
            assert len(tgts) % 2 == 0
            for k in range(0, len(tgts), 2):
                c = tgts[k].value
                t = tgts[k + 1].value
                if t in active_blocks:
                    active_blocks[t].append(c)

        elif name == "MR":
            # Close the block(s): finalize the control lists
            for t in tgts:
                q = t.value
                if q in active_blocks:
                    anc_controls[q] = active_blocks[q]
                    del active_blocks[q]

        else:
            # ignore other ops
            pass

    # Any still-active blocks (missing MR) are ignored.
    return anc_controls


# ---- helper scans over the circuit ----


def _build_meas_index_map(circ: stim.Circuit) -> dict[int, int]:
    """Map measured qubit -> column index in sampler output, for all MR ops before the final data MR."""
    m = {}
    col = 0
    for op in circ:
        if op.name == "MR":
            for t in op.targets_copy():
                m[t.value] = col
                col += 1
    return m


def _build_anc_controls(circ: stim.Circuit) -> dict[int, list[int]]:
    """For every target t of a CX(c,t), remember c as a control of t."""
    ctrl = defaultdict(list)
    for op in circ:
        if op.name == "CX":
            tgts = op.targets_copy()
            assert len(tgts) % 2 == 0
            for k in range(0, len(tgts), 2):
                c = tgts[k].value
                t = tgts[k + 1].value
                ctrl[t].append(c)
    return ctrl


def _rx_prepared_qubits(circ: stim.Circuit) -> set[int]:
    """Set of qubits that are prepared with RX (|+>) at some point (used to tag 4-qubit base ancillas)."""
    s = set()
    for op in circ:
        if op.name == "RX":
            s.update(t.value for t in op.targets_copy())
    return s


# ---- the simulator ----


def simulate_recursive_cat_construction(
    w: int,
    t: int,
    p: float,
    n_samples: int = 1024,
    batch_size: int | None = None,
    add_final_measure: bool = True,
):
    """Simulate the recursive fusion scheme returned by `recursive_fuse_cat_state(w,t)`.

    Post-selection per step:
      * If the step has exactly one ancilla prepared with RX  -> accept iff corrected ancilla == 1.
      * Otherwise                                             -> accept iff all corrected ancillas agree (all 0s or all 1s).
    If accepted, apply a UNIFORM X frame update to every qubit in `data_qubits` for that step
    IFF the (single or common) corrected ancilla bit is 1.

    Returns:
        acceptance_rate, acceptance_rate_error,
        error_rates (len floor(w/2)+1), error_rates_error
    """
    # 1) Build the fusion circuit + step metadata
    circ_base, measurements = recursive_fuse_cat_state(w, t)

    # 2) Scan ancilla bookkeeping on the *base* circuit (before adding final data MR)
    meas_index_of_qubit = _build_meas_index_map(circ_base)  # ancilla MR columns
    anc_controls = _build_anc_controls(circ_base)  # for parity correction
    rx_qubits = _rx_prepared_qubits(circ_base)  # base-4 ancillas have RX

    # 3) Optionally add a final, noise-free MR of all data qubits (to form histograms)
    circ_base = CircuitLevelNoise(p, p, p, p).apply(circ_base)
    if add_final_measure:
        circ_run = stim.Circuit()
        circ_run += circ_base
        circ_run.append("TICK")
        circ_run.append("MR", list(range(w)))  # measure data at the end
        data_cols_start = len(meas_index_of_qubit)  # data bits are at the end
    else:
        circ_run = circ_base
        data_cols_start = None  # no data MR

    # 4) (Optional) add circuit-level noise if you have a wrapper; otherwise run as-is.
    # If you have a helper like CircuitLevelNoise, you can uncomment:
    # circ_noisy = CircuitLevelNoise(p, p, p, p).apply(circ_run)
    circ_noisy = circ_run

    # 5) Sampling
    if batch_size is None:
        batch_size = n_samples
    if n_samples > 10_000_000:
        batch_size = min(batch_size, 10_000_000)

    sampler = circ_noisy.compile_sampler()
    total_samples = 0
    total_accepted = 0

    max_sym = w // 2
    hist_total = np.zeros(max_sym + 1, dtype=int)

    # 6) We maintain a running acceptance mask & frame corrections over DATA qubits
    #    NOTE: corrections are only meaningful on data qubits 0..w-1; we never flip ancillas physically.
    while total_samples < n_samples:
        this_batch = min(batch_size, n_samples - total_samples)
        raw = sampler.sample(this_batch).astype(np.uint8)
        total_samples += this_batch

        # Split columns: [ancilla MR ...] then optionally [final data MR ...]
        if add_final_measure:
            anc_bits_all = raw[:, :data_cols_start]
            data_bits_all = raw[:, data_cols_start : data_cols_start + w]
        else:
            anc_bits_all = raw
            data_bits_all = None

        remaining = np.ones(this_batch, dtype=bool)
        corrections = np.zeros((this_batch, w), dtype=np.uint8)  # frame on data qubits

        for i in range(this_batch):
            {q: anc_bits_all[i, c] for q, c in meas_index_of_qubit.items()}

        # 7) Iterate fusion steps
        for ancillas, data_qubits in measurements:
            if len(ancillas) == 0:
                # No checks: nothing to post-select; also no uniform flip here.
                continue

            # Columns in ancilla measurement matrix for these ancillas
            try:
                anc_cols = [meas_index_of_qubit[a] for a in ancillas]
            except KeyError:
                # Defensive: if a listed ancilla wasn't measured in circ_base, reject all
                remaining[:] = False
                break

            # Parity flips from the current frame: sum of corrections on each ancilla's controls
            flips = np.zeros((this_batch, len(ancillas)), dtype=np.uint8)
            for j, a in enumerate(ancillas):
                ctrls = anc_controls.get(a, [])
                if ctrls:
                    flips[:, j] = (corrections[:, ctrls].sum(axis=1) & 1).astype(np.uint8)

            # Corrected ancilla outcomes on currently remaining rows
            obs = anc_bits_all[remaining][:, anc_cols] ^ flips[remaining]

            # Acceptance rule for this step
            # Special "force-1" if it's a single ancilla prepared with RX (the ft-4 gadget)
            force_zero = len(ancillas) == 1 and ancillas[0] in rx_qubits

            if force_zero:
                # Accept iff corrected bit == 0
                agree = obs[:, 0] == 0
                step_bit = np.zeros_like(obs[:, 0], dtype=np.uint8)  # the only acceptable bit is 0
            else:
                # Accept iff all corrected ancillas agree (all 0 or all 1)
                s = obs.sum(axis=1)
                agree = (s == 0) | (s == obs.shape[1])
                # use the common bit for the uniform flip (0: no flip; 1: flip)
                step_bit = obs[:, 0]  # any column is fine since they agree

            if not agree.any():
                remaining[:] = False
                break

            # Update remaining-rows mask
            idx_rem = np.where(remaining)[0]
            keep_mask = np.zeros_like(remaining)
            keep_mask[idx_rem[agree]] = True
            remaining &= keep_mask
            if not remaining.any():
                break

            # Uniform frame update on data_qubits for rows where step_bit == 1
            if data_qubits and not force_zero:
                rows_to_flip = idx_rem[agree & (step_bit == 1)]
                if rows_to_flip.size:
                    corrections[rows_to_flip[:, None], np.asarray(data_qubits, dtype=int)] ^= 1

        # 8) Collect accepted rows; optionally build error histogram
        idx_acc = np.where(remaining)[0]
        total_accepted += idx_acc.size

        if add_final_measure and idx_acc.size:
            # Apply final frame to the data MR bits to get error pattern
            dat = data_bits_all[idx_acc, :].copy()
            dat ^= corrections[idx_acc, :]
            wts = dat.sum(axis=1)
            sym_wts = np.minimum(wts, w - wts).astype(int)
            hist, _ = np.histogram(sym_wts, bins=np.arange(max_sym + 2))
            hist_total += hist

    # 9) Stats
    acceptance_rate = total_accepted / max(total_samples, 1)
    acceptance_rate_error = np.sqrt(acceptance_rate * max(1 - acceptance_rate, 0) / max(total_samples, 1))

    if add_final_measure:
        error_rates = hist_total / max(total_samples, 1)
        error_rates_error = np.sqrt(error_rates * np.maximum(1 - error_rates, 0) / max(total_samples, 1))
    else:
        # If no final data MR was added, return zeros for the histogram
        error_rates = np.zeros(max_sym + 1, dtype=float)
        error_rates_error = np.zeros_like(error_rates)

    return acceptance_rate, acceptance_rate_error, error_rates, error_rates_error
