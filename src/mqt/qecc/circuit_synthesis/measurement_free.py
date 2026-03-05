# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit synthesis for measurement-free fault-tolerant quantum computing."""

from __future__ import annotations

import z3


def correction_network(
    n_data: int, n_flips: int, correction_map: dict[tuple[int, ...], list[int]]
) -> dict[tuple[int, ...], list[int]]:
    """Compute the correction network of MCS gates for a mapping of flipped qubits to corrections.

    Args:
        n_data: number of data qubits.
        n_flips: number of qubits which are used as controls in the MCX.
        correction_map: dictionary mapping flipped qubit configurations to wanted corrections. "Flip qubits" ("data qubits") are assumed to be indexed from 0 to n_flips-1 (n_data-1). Empty dictionary means no correction network.
    """
    solver = z3.Solver()
    # variables
    flip_corrections = [
        [z3.Bool(f"fc_{i}_{j}") for j in range(n_data)] for i in range(2**n_flips)
    ]  # variable assigning bit patterns to corrections
    flip_vars = [z3.Bool(f"f_{i}") for i in range(n_flips)]

    # go through all possible states of the flip qubits
    final_correction = [False for i in range(n_data)]

    # Encode action of the entire correction network
    for flips in correction_map:
        flip = sum(2**q for q in flips)
        flip_bits = [(flip >> bit) & 1 for bit in reversed(range(n_flips))]
        flip_pattern_subset = z3.And([flip_vars[i] if bool(flip_bits[i]) else True for i in range(n_flips)])

        for i in range(n_data):
            final_correction[i] = z3.Xor(z3.And(flip_pattern_subset, flip_corrections[flip][i]), final_correction[i])

    # Encode that bit patterns should lead to the corrections given in correction_map
    implications = []
    for flips, correction in correction_map.items():
        flip = sum(2**q for q in flips)
        flip_bits = [(flip >> bit) & 1 for bit in reversed(range(n_flips))]

        is_flip_pattern_exact = z3.And([flip_vars[i] == bool(flip_bits[i]) for i in range(n_flips)])

        correction_vec = [i in correction for i in range(n_data)]
        implications.append(
            z3.Implies(is_flip_pattern_exact, z3.And([final_correction[i] == correction_vec[i] for i in range(n_data)]))
        )

    solver.add(z3.ForAll(flip_vars, z3.And(implications)))  # TODO: get read of forall quantifier

    if solver.check() != z3.sat:
        return {}

    m = solver.model()

    network = {}
    for i in range(1, len(flip_corrections)):
        flips = tuple(q for q in range(n_flips) if (i >> q) & 1 == 1)
        correction = [q for q in range(n_data) if m[flip_corrections[i][q]]]
        network[flips] = correction

    return network
