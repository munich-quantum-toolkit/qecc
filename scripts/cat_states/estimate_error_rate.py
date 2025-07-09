# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Estimate Cat State Preparation Logical Error Rate."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from mqt.qecc.circuit_synthesis.cat_states import CatStatePreparationExperiment, cat_state_balanced_tree, cat_state_line

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

# permutations used in experiments for tree-structured cat state preparation
permutations = {4: None, 8: [7, 1, 2, 3, 4, 5, 6, 0], 16: [0, 1, 6, 10, 13, 3, 5, 15, 2, 8, 11, 14, 4, 9, 12, 7]}


def tree(w: int) -> tuple[QuantumCircuit, list[int]]:
    """Create cat state preparation circuits for cat state preparation circuits with tree topology."""
    if w not in {4, 8, 16}:
        msg = "Weight must be 4, 8 or 16"
        raise ValueError(msg)

    circ = cat_state_balanced_tree(w)
    perm = permutations[w]
    return circ, perm


def line(w: int) -> tuple[QuantumCircuit, list[int]]:
    """Create cat state preparation circuits for cat state preparation circuits with linear topology."""
    circ = cat_state_line(w)
    perm = list(range(w))
    for i in range(1, w // 2, 2):
        perm[i], perm[w // 2 + i] = perm[w // 2 + i], perm[i]
    return circ, perm


def main() -> None:
    """Run error rate estimation for cat state preparation circuits."""
    parser = argparse.ArgumentParser(description="Estimate logical error rate for cat state preparation circuits")

    parser.add_argument("-p", "--p_error", type=float, help="Physical error rate")
    parser.add_argument("-w", "--weight", type=int, help="Weight of the cat state. Can be 4, 8 or 16")
    parser.add_argument(
        "-n", "--n_samples", type=int, default=int(100e9), help="Number of samples to estimate logical error rate"
    )
    parser.add_argument("-l", "--linear", type=bool, help="Whether to use linear or tree cat state preparation circuit")

    args = parser.parse_args()

    if args.linear:
        circ, perm = line(args.weight)
    else:
        circ, perm = tree(args.weight)
    p = args.p_error
    n = args.n_samples
    experiment = CatStatePreparationExperiment(circ, circ, perm)
    ra, ra_error, error_rates, error_rates_error = experiment.sample_cat_state(p, n)

    print(";".join([str(ra), str(ra_error), str(error_rates), str(error_rates_error)]))


if __name__ == "__main__":
    main()
