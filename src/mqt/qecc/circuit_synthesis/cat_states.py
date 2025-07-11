# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods for preparing cat states and running experiments on them."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import stim
from qiskit import QuantumCircuit

from .circuit_utils import qiskit_to_stim_circuit, relabel_qubits
from .noise import CircuitLevelNoiseNoIdling

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
        self.circ = qiskit_to_stim_circuit(QuantumCircuit.from_qasm_str(self.circ.to_qasm(open_qasm_version=2)))

    def _get_noisy_circ(self, p: float) -> stim.Circuit:
        """Return a noisy version of the cat state preparation circuit.

        Args:
            p: The noise parameter.

        Returns:
            The noisy cat state preparation circuit.
        """
        return CircuitLevelNoiseNoIdling(p).apply(self.circ)

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
