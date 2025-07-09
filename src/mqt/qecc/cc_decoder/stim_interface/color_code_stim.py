# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generate stim circuit for the 2D color code."""

from __future__ import annotations

import itertools as it
from typing import TYPE_CHECKING, Any

import numpy as np
import stim

if TYPE_CHECKING:
    from numpy.typing import NDArray


def neighbors(perm: NDArray[int]) -> NDArray[NDArray[NDArray[int]]]:
    """Return the neighbors of a lattice point in the 2D color code."""
    node_sw = np.array((perm[0] + 1, perm[1], perm[2] - 1))
    node_se = np.array((perm[0], perm[1] + 1, perm[2] - 1))
    node_e = np.array((perm[0] - 1, perm[1] + 1, perm[2]))
    node_ne = np.array((perm[0] - 1, perm[1], perm[2] + 1))
    node_nw = np.array((perm[0], perm[1] - 1, perm[2] + 1))
    node_w = np.array((perm[0] + 1, perm[1] - 1, perm[2]))
    return np.asarray((node_sw, node_se, node_e, node_ne, node_nw, node_w))


def gen_pcm_and_logical(distance: int) -> tuple[NDArray[bool], set[int]]:
    """Generate the parity check matrix and logical operator for the 2D color code."""
    lattice_points_to_qubit_index, ancilla_qubit_to_lattice_points = {}, {}
    qubit_count, ancilla_qubit_count = 0, 0
    logical_operator = set()
    t = (distance - 1) // 2

    for comb in it.product(range(3 * t + 1), repeat=3):
        if sum(comb) == 3 * t:
            if (comb[1] - comb[0]) % 3 == 1:
                ancilla_qubit_to_lattice_points[ancilla_qubit_count] = comb
                ancilla_qubit_count += 1
            else:
                lattice_points_to_qubit_index[comb] = qubit_count
                if comb[2] == 0:
                    logical_operator.add(qubit_count)
                qubit_count += 1

    parity_check_matrix = np.zeros((ancilla_qubit_count, qubit_count), dtype=bool)

    for ancilla_qubit, lattice_point in ancilla_qubit_to_lattice_points.items():
        for neighbor in neighbors(lattice_point):
            neighbour_tpl = tuple(neighbor)
            if neighbour_tpl in lattice_points_to_qubit_index:
                qubit = lattice_points_to_qubit_index[neighbour_tpl]
                parity_check_matrix[ancilla_qubit, qubit] = True
    return (parity_check_matrix, logical_operator)


def add_checks_one_round(pcm: NDArray[int], circuit: Any, detectors: bool, error_probability: float) -> Any:  # noqa: ANN401
    """Add one round of checks to the circuit."""
    for check in pcm:
        if error_probability == 0:
            mpp_x_instruction = "MPP "
            mpp_z_instruction = "MPP "
        else:
            mpp_x_instruction = f"MPP({error_probability}) "
            mpp_z_instruction = f"MPP({error_probability}) "
        for q in np.where(check)[0]:
            mpp_x_instruction += "X" + str(q) + "*"
            mpp_z_instruction += "Z" + str(q) + "*"
        #        circuit.append_from_stim_program_text(mpp_x_instruction[:-1])
        circuit.append_from_stim_program_text(mpp_z_instruction[:-1])
    if detectors is True:
        for q in range(len(pcm)):
            circuit.append(
                #                "DETECTOR", [stim.target_rec(-1*q-1), stim.target_rec(-1*q-2*len(pcm)-1)])
                "DETECTOR",
                [stim.target_rec(-1 * q - 1), stim.target_rec(-1 * q - len(pcm) - 1)],
            )
    return circuit


def gen_stim_circuit_memory_experiment(
    pcm: NDArray[int], logical_operator: NDArray[int], distance: int, error_probability: float
) -> Any:  # noqa: ANN401
    """Generate a stim circuit for a memory experiment on the 2D color code."""
    data_qubits = range(len(pcm[0]))
    circuit = stim.Circuit()
    circuit.append("R", data_qubits)

    # initialization
    circuit = add_checks_one_round(pcm, circuit, False, 0)

    # rounds of QEC
    for _i in range(distance):
        circuit.append("X_ERROR", data_qubits, error_probability)
        #        circuit.append("DEPOLARIZE1", data_qubits, error_probability)
        circuit = add_checks_one_round(pcm, circuit, True, error_probability)

    # logical measurement

    circuit = add_checks_one_round(pcm, circuit, True, 0)

    log_measurement_instruction = "MPP "
    for q in logical_operator:
        log_measurement_instruction += "Z" + str(q) + "*"
    circuit.append_from_stim_program_text(log_measurement_instruction[:-1])

    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], (0))

    return circuit
