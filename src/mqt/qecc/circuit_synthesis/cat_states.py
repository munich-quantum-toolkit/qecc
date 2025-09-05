# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods for preparing cat states and running experiments on them."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import stim

from .circuit_utils import relabel_qubits
from .noise import CircuitLevelNoise

if TYPE_CHECKING:
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
    """Class for running cat state preparation experiments based on post-selection.

    One way to initialize cat states is to prepare two copies, connect them with a transversal CNOT, measure the ancilla qubits and post-select on the results.
    The performance of this method depends very much on the circuits and how the cnots are connected.
    """

    def __init__(
        self, circ1: stim.Circuit, circ2: stim.Circuit, permutation: list[int] | npt.NDArray[int] | None = None
    ) -> None:
        """Initialize the experiment with the two halves of the cat state preparation circuit.

        Args:
            circ1: The first half of the cat state preparation circuit preparing the data qubits. Qubits are assumed to be from 0 to n_qubits-1.
            circ2: The second half of the cat state preparation circuit preparing the ancilla states. Qubits are assumed to be from 0 to n_qubits-1.
            permutation: The permutation to apply to the transversal CNOTs connecting the two halves.
        """
        assert circ1.num_qubits == circ2.num_qubits, "The two circuits must have the same number of qubits."
        self.w = circ1.num_qubits
        self.circ = transversal_cnot(circ1, relabel_qubits(circ2, self.w), permutation)
        self.circ.append("MR", range(self.w, self.w * 2))

    def _get_noisy_circ(self, p: float) -> stim.Circuit:
        """Return a noisy version of the cat state preparation circuit.

        Args:
            p: The noise parameter.

        Returns:
            The noisy cat state preparation circuit.
        """
        return CircuitLevelNoise(p, p, p, p).apply(self.circ)

    def sample_cat_state(
        self, p: float, n_samples: int = 1024, batch_size: int | None = None
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Sample the circuit under circuit-level noise in batches and accumulate statistics.

        Noise statistics are sample by running the circuit and post-selecting on the ancilla qubits. If the ancilla state is not in the all 0 or all 1 state, the sample is discarded. For the samples which are not rejected, the number of errors on the data qubits is counted and a histogram is built.

        Args:
            p: noise parameter.
            n_samples: The total number of samples to collect.
            batch_size: The number of samples to collect in each batch.
                If None, the batch size is equal to n_samples.

        Returns:
            acceptance_rate: The fraction of samples that were accepted.
            acceptance_rate_error: The statistical error on the acceptance rate.
            error_rates: The histogram of error rates.
            error_rates_error: The statistical error on the error rates.
        """
        circ = self._get_noisy_circ(p)
        circ.append("TICK")
        circ.append("MR", range(self.w))  # no noise on final measurement

        if batch_size is None:
            batch_size = n_samples

        if n_samples > 1e7:
            batch_size = int(1e7)

        total_samples = 0
        total_accepted = 0
        w = circ.num_qubits // 2
        # Prepare an array for histogram counts.
        # Using bins defined by range(w//2 + 1) produces w//2 bins.
        hist_total = np.zeros(w // 2 + 1, dtype=int)

        # Determine how many batches you need.
        n_batches = int(np.ceil(n_samples / batch_size))

        for _ in range(n_batches):
            current_batch = min(batch_size, n_samples - total_samples)

            sampler = circ.compile_sampler()
            res = sampler.sample(current_batch).astype(int)
            total_samples += current_batch

            # Process the ancilla measurements to determine accepted events.
            anc = res[:, :w]
            filtered = np.where(np.logical_or(np.all(anc == 1, axis=1), np.all(anc == 0, axis=1)))[0]
            state = res[filtered, w:]
            total_accepted += state.shape[0]

            # Only update if some accepted events are present in the batch.
            if state.shape[0] > 0:
                error_weights = np.min(np.vstack((state.sum(axis=1), w - state.sum(axis=1))), axis=0)
                hist, _ = np.histogram(error_weights, bins=range(w // 2 + 2))
                hist_total += hist

        # Compute overall acceptance rate and its binomial error.
        acceptance_rate = total_accepted / total_samples
        acceptance_rate_error = np.sqrt(acceptance_rate * (1 - acceptance_rate) / total_samples)

        # Compute overall histogram error rates and their errors.
        error_rates = hist_total / total_samples
        error_rates_error = np.sqrt(error_rates * (1 - error_rates) / total_samples)

        return acceptance_rate, acceptance_rate_error, error_rates, error_rates_error

    def plot_one_p(
        self, p: float, n_samples: int = 1024, batch_size: int | None = None, ax: plt.Axes | None = None
    ) -> None:
        """Plot histogram showing probabilities that a certain number of errors occurred in a cat state preparation experiment with a given physical error rate.

        Args:
            p: physical error rate for the experiment.
            n_samples: number of samples to take.
            batch_size: number of samples to take in each batch.
            ax: matplotlib axis to plot on.

        Returns:
            None
        """
        ra, ra_err, hist, hist_err = self.sample_cat_state(p, n_samples, batch_size)
        w = self.w
        x = np.arange(w // 2 + 1)
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

        ax.set_xlabel("Number of errors")
        ax.set_ylabel("Probability")
        ax.set_xticks(x)
        ax.set_yscale("log")
        ax.margins(0.2, 0.2)
        plt.title(f"Error distribution for w = {self.w}, p = {p:.2f}. Acceptance rate = {ra:.2f} +/- {ra_err:.2f}")
        plt.show()

    def cat_prep_experiment(
        self, ps: list[float], shots_per_p: int | list[int]
    ) -> tuple[list[float], list[float], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Run a series of cat state preparation experiments.

        Args:
            ps: The noise parameters to use.
            shots_per_p: The number of shots to take for each noise parameter.
                If an integer, the same number of shots is used for all noise parameters.
                If a list, the number of shots is taken from the list for each noise parameter.
            perm: The permutation

        Returns:
            ras: The acceptance rates for each noise parameter.
            ra_errs: The statistical errors on the acceptance rates.
            hists: The histograms of error rates for each noise parameter.
            hists_err: The statistical errors on the histograms
        """
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


def degree_sets_from_gens(gens: list[int], t: int):
    S = [set() for _ in range(t + 1)]
    S[0].add(0)
    for g in gens:
        for h in range(t, 0, -1):
            for s in list(S[h - 1]):
                S[h].add(s ^ g)
    return S


def bitcount(x: int) -> int:
    return x.bit_count()


def support_bits(mask: int, w: int) -> list[int]:
    return [i for i in range(w) if (mask >> i) & 1]


def ones_indices(mask: int, w: int) -> list[int]:
    return [j for j in range(w) if (mask >> j) & 1]


def apply_perm_mask(mask: int, P: list[int]) -> int:
    """Forward permutation P: image bits go to new positions P[i]."""
    out = 0
    for i, j in enumerate(P):
        if (mask >> i) & 1:
            out |= 1 << j
    return out


# ---------- strict t-distinct check (cross-circuit), returns a witness ----------


def find_violation_strict(gens1, gens2_img, t: int):
    """Return (h1,h2,x2, image_mask, target_mask, w) or None."""
    w = max((g.bit_length() for g in gens1 + gens2_img), default=0)
    all_ones = (1 << w) - 1
    S1_by_h = degree_sets_from_gens(gens1, t)
    S2_by_h = degree_sets_from_gens(gens2_img, t)

    for h2 in range(1, t + 1):
        for h1 in range(1, t - h2 + 1):
            S1 = S1_by_h[h1]
            for x2 in S2_by_h[h2]:
                y = x2
                wtmin = min(bitcount(y), bitcount(all_ones ^ y))
                if wtmin <= h1 + h2:
                    continue
                if y in S1:
                    return (h1, h2, x2, y, y, w)
                yc = all_ones ^ y
                if yc in S1:
                    return (h1, h2, x2, y, yc, w)
    return None


def t_distinct_cat_exact_permutation(gens1, gens2, t: int, P: list[int]) -> bool:
    gens2_img = [apply_perm_mask(g, P) for g in gens2]
    return find_violation_strict(gens1, gens2_img, t) is None


# ---------- build the "bad images" catalog exactly (as in SAT one-shot) ----------


def build_bad_catalog(gens1, gens2, t: int):
    """For each degree h2 pattern x2 from circuit 2 (before permutation),
    compute the set of *image masks* that are forbidden: bad_images[x2] = {b1,b2,...}.
    """
    w = max(g.bit_length() for g in gens1 + gens2)
    all_ones = (1 << w) - 1

    S1_by_h = degree_sets_from_gens(gens1, t)
    S2_by_h = degree_sets_from_gens(gens2, t)

    # bucket S1_by_h by weight for fast selection
    S1_by_h_by_wt = []
    for h in range(t + 1):
        buckets = defaultdict(list)
        for m in S1_by_h[h]:
            buckets[bitcount(m)].append(m)
        S1_by_h_by_wt.append(buckets)

    bad_images = defaultdict(set)  # x2 -> set of forbidden image masks
    x2_list = []
    for h2 in range(1, t + 1):
        for x2 in S2_by_h[h2]:
            s = bitcount(x2)
            if s == 0:
                continue
            x2_list.append(x2)
            for h1 in range(1, t - h2 + 1):
                # equality case: S1 masks with same weight s
                for b in S1_by_h_by_wt[h1].get(s, []):
                    if min(s, w - s) > (h1 + h2):
                        bad_images[x2].add(b)
                # complement case: need wt(b) == w-s, add ~b as bad image
                for b in S1_by_h_by_wt[h1].get(w - s, []):
                    if min(w - s, s) > (h1 + h2):
                        bad_images[x2].add(all_ones ^ b)
    return bad_images, x2_list, w


# ---------- witness-guided local repair (Moser–Tardos–style) ----------


def find_perm_local_search(w: int, t: int, seed=1, restarts=16, max_iters=500000):
    """Build gens for two balanced 2-ary trees of size w and search a permutation P
    so that no x2 maps to a forbidden image (strict criterion up to t).
    Returns (P or None, stats).
    """
    rng = random.Random(seed)
    gens1 = binary_tree_fault_gens(w, include_full=False)
    gens2 = binary_tree_fault_gens(w, include_full=False)

    bad_images, x2_list, w2 = build_bad_catalog(gens1, gens2, t)
    assert w == w2
    all_cols = list(range(w))

    def random_perm():
        P = list(range(w))
        rng.shuffle(P)
        inv = [0] * w
        for i, j in enumerate(P):
            inv[j] = i
        return P, inv

    def image_mask_of_x2(x2, P):
        return apply_perm_mask(x2, P)

    # fast membership check
    bad_sets = {x2: set(bs) for x2, bs in bad_images.items()}

    for rs in range(restarts):
        P, invP = random_perm()
        it = 0
        while it < max_iters:
            it += 1
            # scan for a violation (sampled scan speeds it up; do full scan if you prefer)
            found = None
            for x2 in x2_list:
                y = image_mask_of_x2(x2, P)
                if y in bad_sets.get(x2, ()):
                    found = (x2, y)
                    break
            if not found:
                return P, {"status": "sat", "iters": it, "restarts": rs, "bad_x2": len(bad_sets)}

            x2, y = found
            S = support_bits(x2, w)
            T = set(ones_indices(y, w))
            comp_cols = [c for c in all_cols if c not in T]
            # Try a few smart swaps to kill the event without creating another for the same x2
            success = False
            attempt_budget = 64
            while attempt_budget > 0:
                attempt_budget -= 1
                i = rng.choice(S)
                c = rng.choice(comp_cols)
                k = invP[c]  # row currently mapped to column c; since c∉T, k∉S
                # simulate swap (i <-> k) and test x2's new image
                old_j_i, old_j_k = P[i], P[k]
                # new image set is (T - {old_j_i}) ∪ {c}
                new_y = (y ^ (1 << old_j_i)) | (1 << c)
                if new_y not in bad_sets.get(x2, ()):
                    # accept swap
                    P[i], P[k] = P[k], P[i]
                    invP[old_j_i], invP[old_j_k] = invP[old_j_k], invP[old_j_i]
                    success = True
                    break
            if not success:
                # random perturbation: swap two random rows (keeps P a permutation)
                i1, i2 = rng.sample(range(w), 2)
                j1, j2 = P[i1], P[i2]
                P[i1], P[i2] = j2, j1
                invP[j1], invP[j2] = i2, i1
        # restart
    return None, {"status": "unknown", "iters": max_iters, "restarts": restarts}
