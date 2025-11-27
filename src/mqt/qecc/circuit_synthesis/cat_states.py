# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods for preparing cat states and running experiments on them."""

from __future__ import annotations

import math
from collections import defaultdict
from functools import cache
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import stim
import z3

from .circuit_utils import compact_stim_circuit, relabel_qubits
from .noise import CircuitLevelNoise

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

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
        """Initialize the cat state experiment.

        Args:
            circ1: data-prep circuit, size w1
            circ2: ancilla-prep circuit, size w2 (can be < w1)
            permutation: perm over 0..w2-1 (ancilla targets)
            controls: length-w2 list of data controls (subset of 0..w1-1)
        """
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

        comb.append_operation("MR", list(range(w1, w1 + w2)))

        self.circ = compact_stim_circuit(comb)

    def _get_noisy_circ(self, p: float) -> stim.Circuit:
        """Return a noisy version of the combined circuit."""
        return CircuitLevelNoise(p, p, p, p).apply(self.circ)

    def sample_cat_state(
        self, p: float, n_samples: int = 1024, batch_size: int | None = None
    ) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Run with circuit-level noise, post-select on ancilla ∈ {0^w2, 1^w2}, and histogram the symmetric error weight on data.

        Returns:
            acceptance_rate, acceptance_rate_error,
            error_rates (length floor(w1/2)+1), error_rates_error
        """
        circ = self._get_noisy_circ(p)
        # Final, *noise-free* measurement of data qubits
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

            wts = data_ok.sum(axis=1)
            sym_wts = np.minimum(wts, self.w1 - wts).astype(int)

            hist, _ = np.histogram(sym_wts, bins=np.arange(max_sym_w + 2))
            hist_total += hist

        acceptance_rate = total_accepted / max(total_samples, 1)
        acceptance_rate_error = np.sqrt(acceptance_rate * max(1 - acceptance_rate, 0) / max(total_samples, 1))

        error_rates = hist_total / max(total_accepted, 1)
        error_rates_error = np.sqrt(error_rates * np.maximum(1 - error_rates, 0) / max(total_accepted, 1))

        return acceptance_rate, acceptance_rate_error, error_rates, error_rates_error

    def plot_one_p(
        self, p: float, n_samples: int = 1024, batch_size: int | None = None, ax: plt.Axes | None = None
    ) -> None:
        """Plot the distribution of residual error weights on the data cat state for the given physical error rate."""
        ra, ra_err, hist, hist_err = self.sample_cat_state(p, n_samples, batch_size)
        x = np.arange(self.w1 // 2 + 1)
        if ax is None:
            _fig, ax = plt.subplots()

        cmap = plt.cm.plasma
        colors = cmap(np.linspace(0, 1, len(x)))

        bar_width = 0.8
        for xi, yi, err, color in zip(x, hist, hist_err, colors, strict=False):
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

        ax.set_xlabel("Weight of residual error")
        ax.set_ylabel("Probability")
        ax.set_xticks(x)
        ax.set_yscale("log")
        ax.margins(0.2, 0.2)
        plt.title(f"Cat prep: w1={self.w1}, w2={self.w2}, p={p:.3f}. Acceptance = {ra:.3f} ± {ra_err:.3f}")
        plt.show()

    def cat_prep_experiment(
        self, ps: list[float], shots_per_p: int | list[int]
    ) -> tuple[list[float], list[float], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Simulate post-selection based cat state preparation for various physical error rates using a circuit-level depolarizing noise.

        Args:
            ps: list of physical error rates.
            shots_per_p: number of shots for each simulation or list if number of shots should depend on the simulated physical error rate.

        Returns:
            A tuple containing
            - List of acceptance rates
            - List of estimated error bars of the acceptance rate
            - List of lists of error rates of length w/2. The i-th entry is the fraction of shots with a residual error of weight i on the data.
            - List of lists of error bars for the residual errors
        """
        if isinstance(shots_per_p, list):
            assert len(shots_per_p) == len(ps)
        else:
            shots_per_p = [shots_per_p for _ in range(len(ps))]

        hists = None
        hists_err = None
        ras = []
        ra_errs = []
        for p, n_shots in zip(ps, shots_per_p, strict=False):
            ra, ra_err, hist, hist_err = self.sample_cat_state(p, n_shots, batch_size=100000)
            ras.append(ra)
            ra_errs.append(ra_err)
            if hists is None:
                hists = hist
                hists_err = hist_err
            else:
                hists = np.vstack((hists, hist))
                assert hists_err is not None
                hists_err = np.vstack((hists_err, hist_err))
        assert hists is not None
        assert hists_err is not None
        return ras, ra_errs, hists, hists_err


def append_transversal_cnot_pairs(circ: stim.Circuit, pairs: Sequence[tuple[int, int]]) -> None:
    """Append a layer of CX using disjoint pairs.

    This function assumes that circ1 acts on the first w qubits and circ2 on the second w qubits
    """
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
    """Returns list of (control, target) indices for a single parallel CX layer (controls[i], w1 + perm_targets[i])  for i=0..w2-1."""
    if len(controls) != w2:
        msg = f"len(controls) must equal w2; got {len(controls)} vs {w2}"
        raise ValueError(msg)
    if sorted(set(controls)) != sorted(controls):
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


def cat_state_pruned_balanced_circuit(w: int) -> stim.Circuit:
    """Prepare GHZ_w in log-depth with a balanced tree circuit.

    If w is not a power of two, CNOTs are pruned from the full balanced tree until the circuit acts on w qubits.

    Args:
            w: number of qubits of the cat state
    Returns:
            stim circuit preparing the cat state
    """
    if w <= 0:
        msg = "w must be >= 1"
        raise ValueError(msg)
    circ = stim.Circuit()
    circ.append_operation("H", [0])

    if w == 1:
        return circ

    m = math.ceil(math.log2(w))
    next_power_two = 1 << m

    for stride in (1 << k for k in range(m - 1, -1, -1)):
        step = 2 * stride
        for j in range(0, next_power_two, step):
            c = j
            t = j + stride
            if c < w and t < w:
                circ.append_operation("CX", [c, t])
    return circ


def _propagate_forward(error: int, c: int, t: int) -> int:
    if (error >> c) & 1:
        error ^= 1 << t
    return error


def fault_gens_from_circuit(circ: stim.Circuit) -> list[int]:
    """Propagated single-qubit errors in a cat state preparation circuit.

      Includes all singletons ,
      For each CX pair (c,t) in sequence, inject X on c *just before that CX* and propagate through the remaining pairs.

    Args:
        circ: stim circuit consisting of CX gates only
        include_full: whether to include the full-weight generator

    Returns:
        sorted list of int bit strings representing the propagated errors
    """
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

    all_ones = (1 << w) - 1
    gens: set[int] = set()

    # all singletons
    gens.update(1 << q for q in range(w))

    # inject on control just before each CX, then propagate forward
    for idx, (c0, _) in enumerate(ops):
        mask = 1 << c0
        for c, t in ops[idx:]:
            mask = _propagate_forward(mask, c, t)
        if mask == all_ones:  # error on all qubits is trivial
            continue
        gens.add(mask)

    return sorted(gens)


def _ft_w_4_cat_state() -> tuple[stim.Circuit, list[tuple[list[int], list[int]]]]:
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


def recursive_fuse_cat_state(w: int, t: int) -> tuple[stim.Circuit, list[tuple[list[int], list[int]]]]:
    """Construct t-FT cat state prep circuit from arXiv:2506.17181."""

    def _recurse(w1: int, w2: int) -> tuple[stim.Circuit, list[tuple[list[int], list[int]]]]:
        if w <= 0:
            msg = "w must be >= 1"
            raise ValueError(msg)

        if w1 < 4:
            c1 = cat_state_pruned_balanced_circuit(w1)
            measurements_1: list[tuple[list[int], list[int]]] = []
        elif w1 == 4:
            c1, measurements_1 = _ft_w_4_cat_state()
        else:
            c1, measurements_1 = _recurse((w1 + 1) // 2, w1 // 2)

        if w2 < 4:
            c2 = cat_state_pruned_balanced_circuit(w2)
            measurements_2: list[tuple[list[int], list[int]]] = []
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


def _ancilla_controls_map(circ: stim.Circuit) -> dict[int, list[int]]:
    """For each ancilla qubit measured via an 'R ... CX ... MR' block, collect the list of *controls* that hit it as a CX target between its R and MR."""
    active_blocks: dict[int, list[int]] = {}  # anc -> list of controls
    anc_controls: dict[int, list[int]] = {}

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
    s: set[int] = set()
    for op in circ:
        if op.name == "RX":
            s.update(t.value for t in op.targets_copy())
    return s


def simulate_recursive_cat_construction(
    w: int,
    t: int,
    p: float,
    n_samples: int = 1024,
    batch_size: int | None = None,
) -> tuple[float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Simulate the recursive fusion scheme returned by `recursive_fuse_cat_state(w,t)`.

    Post-selection per step:
      * If the step has exactly one ancilla prepared with RX  -> accept iff corrected ancilla == 1.
      * Otherwise                                             -> accept iff all corrected ancillas agree (all 0s or all 1s).
    If accepted, apply a UNIFORM X frame update to every qubit in `data_qubits` for that step
    IFF the (single or common) corrected ancilla bit is 1.

    Args:
        w: cat state size
        t: fault distance
        p: physical error rate
        n_samples: number of shots
        batch_size: number of shots per batch

    Returns:
        A tuple containing
        - Acceptance rate
        - Estimated error bars of the acceptance rate
        - A list of error rates of length w/2. The i-th entry is the fraction of shots with a residual error of weight i on the data.
        - A list of error bars for the residual errors
    """
    circ_base, measurements = recursive_fuse_cat_state(w, t)

    meas_index_of_qubit = _build_meas_index_map(circ_base)  # ancilla MR columns
    anc_controls = _build_anc_controls(circ_base)  # for parity correction
    rx_qubits = _rx_prepared_qubits(circ_base)  # base-4 ancillas have RX

    circ_base = CircuitLevelNoise(p, p, p, p).apply(circ_base)
    circ_run = stim.Circuit()
    circ_run += circ_base
    circ_run.append("TICK")
    circ_run.append("MR", list(range(w)))  # measure data at the end
    data_cols_start = len(meas_index_of_qubit)  # data bits are at the end

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

    while total_samples < n_samples:
        this_batch = min(batch_size, n_samples - total_samples)
        raw = sampler.sample(this_batch).astype(np.uint8)
        total_samples += this_batch

        anc_bits_all = raw[:, :data_cols_start]
        data_bits_all = raw[:, data_cols_start : data_cols_start + w]

        remaining = np.ones(this_batch, dtype=bool)
        corrections = np.zeros((this_batch, w), dtype=np.uint8)  # frame on data qubits

        for i in range(this_batch):
            {q: anc_bits_all[i, c] for q, c in meas_index_of_qubit.items()}

        for ancillas, data_qubits in measurements:
            if len(ancillas) == 0:
                # No checks: nothing to post-select; also no uniform flip here.
                continue

            try:
                anc_cols = [meas_index_of_qubit[a] for a in ancillas]
            except KeyError:
                remaining[:] = False
                break

            flips = np.zeros((this_batch, len(ancillas)), dtype=np.uint8)
            for j, a in enumerate(ancillas):
                ctrls = anc_controls.get(a, [])
                if ctrls:
                    flips[:, j] = (corrections[:, ctrls].sum(axis=1) & 1).astype(np.uint8)

            # Corrected ancilla outcomes on currently remaining rows
            obs = anc_bits_all[remaining][:, anc_cols] ^ flips[remaining]

            # Acceptance rule for this step
            # Special "force-1" if it's a single ancilla prepared with RX (4-qubit state)
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
                step_bit = obs[:, 0]

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

        idx_acc = np.where(remaining)[0]
        total_accepted += idx_acc.size

        if idx_acc.size:
            dat = data_bits_all[idx_acc, :].copy()
            dat ^= corrections[idx_acc, :]
            wts = dat.sum(axis=1)
            sym_wts = np.minimum(wts, w - wts).astype(int)
            hist, _ = np.histogram(sym_wts, bins=np.arange(max_sym + 2))
            hist_total += hist

    acceptance_rate = total_accepted / max(total_samples, 1)
    acceptance_rate_error = np.sqrt(acceptance_rate * max(1 - acceptance_rate, 0) / max(total_samples, 1))

    error_rates = hist_total / max(total_samples, 1)
    error_rates_error = np.sqrt(error_rates * np.maximum(1 - error_rates, 0) / max(total_samples, 1))

    return acceptance_rate, acceptance_rate_error, error_rates, error_rates_error


def _degree_sets_exact(gens: list[int], t: int) -> list[set[int]]:
    @cache
    def _degree_sets_exact_cached(gens_key: tuple[int, ...], t: int) -> list[set[int]]:
        gens = list(gens_key)
        fault_sets: list[set[int]] = [set() for _ in range(t + 1)]
        fault_sets[0].add(0)
        for g in gens:
            for h in range(t, 0, -1):
                for s in list(fault_sets[h - 1]):
                    fault_sets[h].add(s ^ g)
        return fault_sets

    key = tuple(sorted(set(gens)))
    return _degree_sets_exact_cached(key, t)


def propagate_and_permute_error(error: int, controls: list[int], perm: list[int]) -> int:
    """Propagate error through the controls of a partial transversal CNOT and permute according to target permutation.

    Args:
        error: int bit string encoding the error
        controls: controls of the transversal CNOT
        perm: permutation of the targets

    Returns:
        int bit string representing the error that actually propagates
    """
    out = 0
    for j, q in enumerate(controls):
        if (error >> q) & 1:
            out |= 1 << perm[j]
    return out


def propagate_error_transversal(error: int, controls: list[int]) -> int:
    """Propagate error through the controls of a partial transversal CNOT.

    Args:
        error: int bit string encoding the error
        controls: controls of the transversal CNOT

    Returns:
        int bit string representing the error that actually propagates
    """
    out = 0
    for j, q in enumerate(controls):
        if (error >> q) & 1:
            out |= 1 << j
    return out


def permute_error(error: int, perm: list[int], w: int) -> int:
    """Permute error according to the given permutation.

    Args:
        error: int bit string representing the error
        perm: permutation on [0..w-1] qubits
        w: number of qubits

    Returns:
        the permuted error
    """
    out = 0
    for i in range(w):
        if (error >> i) & 1:
            out |= 1 << perm[i]
    return out


def check_ft_partial_cnot(
    gens1: list[int], w1: int, gens2: list[int], w2: int, controls: list[int], perm: list[int], t: int
) -> tuple[bool, dict[str, int | list[int]] | None]:
    """Check whether the CNOT defined by the given selection of control qubits and permutation is FT-t for the given fault set generators.

    Args:
        gens1: fault set generators of data cat state
        w1: number of qubits of data cat state
        gens2: fault set generators of ancilla cat state
        w2: number of qubits of ancilla cat state
        controls: list of integers denoting qubits acting as controls in the data cat state
        perm: permutation of [0..w2-1] defining how targets are permuted
        t: fault distance

    Returns:
        Boolean indicating whether the given CNOT is FT-t, along with a counterexample if False
    """
    assert len(controls) == w2
    assert len(perm) == w2
    fss1 = _degree_sets_exact(gens1, t)
    fss2 = _degree_sets_exact(gens2, t)
    all_ones = (1 << w2) - 1

    for h1 in range(1, t + 1):
        for err in fss1[h1]:
            wt1 = err.bit_count()
            sym1 = min(wt1, w1 - wt1)
            for h2 in range(t - h1 + 1):  # include h2=0
                if sym1 <= h1 + h2:
                    continue
                err_projected = propagate_error_transversal(err, controls)
                err_permuted = permute_error(err_projected, perm, w2)
                if err_permuted in fss2[h2] or ((all_ones ^ err_permuted) in fss2[h2]):
                    return False, {
                        "h1": h1,
                        "h2": h2,
                        "err": err,
                        "sym_w1": sym1,
                        "err_projected": err_projected,
                        "err_permuted": err_permuted,
                        "w1": w1,
                        "w2": w2,
                        "controls": controls,
                        "perm": perm,
                    }
    return True, None


def search_ft_cnot_cegar(
    gens1: list[int],
    w1: int,
    gens2: list[int],
    w2: int,
    t: int,
    max_rounds: int = 10000,
) -> tuple[list[int] | None, list[int] | None, dict[str, str | int]]:
    r"""Use CEGAR approach to find an ft-t partial transversal CNOT.

    Args:
        gens1: fault set generators of data cat state
        w1: number of qubits of data cat state
        gens2: fault set generators of ancilla cat state
        w2: number of qubits of ancilla cat state
        t: target fault distance
        add_symmetry_breakers (default True): whether to encode symmetry breakers
        max_rounds (default 10000): number of CEGAR iterations to try

    Returns:
       list of controls, list of permutation and search information dict {\"sat\":..., \"rounds\"}
    """
    all_ones = (1 << w2) - 1
    s = z3.Solver()

    # ---- Variables ----
    ctrl = [z3.Bool(f"ctrl_{q}") for q in range(w1)]
    s.add(z3.PbEq([(q, 1) for q in ctrl], w2))  # exactly w2 qubits must be controls

    trgt = [z3.Int(f"trgt_{q}") for q in range(w1)]
    for v in trgt:
        s.add(z3.And(v >= 0, v < w2))

    # different controls must map to different targets
    for u in range(w1):
        for v in range(u + 1, w1):
            s.add(z3.Implies(z3.And(ctrl[u], ctrl[v]), trgt[u] != trgt[v]))

    # ---- BitVec helpers to build y(x1) directly from (ctrl, trgt) ----
    one_bv = z3.BitVecVal(1, w2)
    zero_bv = z3.BitVecVal(0, w2)

    def permuted_error_symbolic(err: int) -> z3.BitVecRef:
        """Build BitVector error under permutation."""
        permuted = zero_bv
        m = err
        q = 0
        while m:
            if m & 1:
                permuted |= z3.If(ctrl[q], (one_bv << z3.Int2BV(trgt[q], w2)), zero_bv)
            m >>= 1
            q += 1
        return permuted

    def find_violation() -> tuple[bool, dict[str, Any]]:
        """Check if found solution is ft-t."""
        mdl = s.model()
        chosen = [q for q in range(w1) if mdl.evaluate(ctrl[q])]
        chosen.sort()
        perm = [mdl.evaluate(trgt[q]).as_long() for q in chosen]  # length w2
        ok, wit = check_ft_partial_cnot(gens1, w1, gens2, w2, chosen, perm, t)
        if ok:
            return (True, {"controls": chosen, "perm": perm})
        return (False, {"controls": chosen, "perm": perm, "witness": wit})

    # ---- CEGAR loop ----
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        if s.check() != z3.sat:
            return None, None, {"status": "unsat", "rounds": rounds}
        is_ft, info = find_violation()
        if is_ft:
            return info["controls"], info["perm"], {"status": "sat", "rounds": rounds}

        wit = info["witness"]
        err = wit["err"]
        err_symbolic = permuted_error_symbolic(err)

        chosen = info["controls"]
        perm = info["perm"]

        err_projected = propagate_and_permute_error(err, chosen, perm)

        s.add(err_symbolic != z3.BitVecVal(err_projected, w2))
        s.add(err_symbolic != z3.BitVecVal(all_ones ^ err_projected, w2))

    return None, None, {"status": "unknown", "rounds": rounds}
