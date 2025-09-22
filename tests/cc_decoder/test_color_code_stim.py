# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""tests for the color code stim decoder main functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import stim
from numpy import array_equal

if TYPE_CHECKING:
    from numpy.typing import NDArray
from mqt.qecc.cc_decoder.stim_interface.color_code_stim import (
    add_checks_one_round,
    gen_pcm_and_logical,
    gen_stim_circuit_memory_experiment,
    neighbors,
)


@pytest.fixture
def hamming_code() -> NDArray[bool]:
    """Return hamming code parity check matrix."""
    return np.array([
        [True, True, False, True, True, False, False],
        [False, True, True, False, True, True, False],
        [False, False, False, True, True, True, True],
    ])


def test_gen_pcm_and_logical(hamming_code: NDArray[bool]) -> None:
    """Test parity check matrix and logical matrix generation."""
    distance = 3
    expected_logicals = {2, 5, 6}

    pcm, logicals = gen_pcm_and_logical(distance)

    assert array_equal(hamming_code, pcm)
    assert expected_logicals == logicals


def test_neighbours() -> None:
    """Test neighbour computation for color code grid."""
    input_perm = np.array([1, 2, 3])
    expected = np.array([[2, 2, 2], [1, 3, 2], [0, 3, 3], [0, 2, 4], [1, 1, 4], [2, 1, 3]])
    assert array_equal(expected, neighbors(input_perm))


def test_add_checks_one_round(hamming_code: NDArray[bool]) -> None:
    """Test stim circuit generation for one stabilizer round."""
    expected_circuit = stim.Circuit()
    circuit = stim.Circuit()
    expected_circuit.append_from_stim_program_text("MPP Z0*Z1*Z3*Z4")
    expected_circuit.append_from_stim_program_text("MPP Z1*Z2*Z4*Z5")
    expected_circuit.append_from_stim_program_text("MPP Z3*Z4*Z5*Z6")
    assert expected_circuit == add_checks_one_round(hamming_code, circuit, False, 0)


def test_gen_stim_memory_experiment(hamming_code: NDArray[bool]) -> None:
    """Test generation of stim circuit for a memory experiment."""
    expected_circuit = stim.Circuit()
    stim.Circuit()
    expected_circuit.append_from_stim_program_text("R 0 1 2 3 4 5 6")
    expected_circuit.append_from_stim_program_text("MPP Z0*Z1*Z3*Z4 Z1*Z2*Z4*Z5 Z3*Z4*Z5*Z6")
    expected_circuit.append_from_stim_program_text("X_ERROR(0.5) 0 1 2 3 4 5 6")
    expected_circuit.append_from_stim_program_text("MPP(0.5) Z0*Z1*Z3*Z4 Z1*Z2*Z4*Z5 Z3*Z4*Z5*Z6")
    expected_circuit.append_from_stim_program_text("DETECTOR rec[-1] rec[-4]")
    expected_circuit.append_from_stim_program_text("DETECTOR rec[-2] rec[-5]")
    expected_circuit.append_from_stim_program_text("DETECTOR rec[-3] rec[-6]")
    expected_circuit.append_from_stim_program_text("X_ERROR(0.5) 0 1 2 3 4 5 6")
    expected_circuit.append_from_stim_program_text("MPP(0.5) Z0*Z1*Z3*Z4 Z1*Z2*Z4*Z5 Z3*Z4*Z5*Z6")
    expected_circuit.append_from_stim_program_text("DETECTOR rec[-1] rec[-4]")
    expected_circuit.append_from_stim_program_text("DETECTOR rec[-2] rec[-5]")
    expected_circuit.append_from_stim_program_text("DETECTOR rec[-3] rec[-6]")
    expected_circuit.append_from_stim_program_text("X_ERROR(0.5) 0 1 2 3 4 5 6")
    expected_circuit.append_from_stim_program_text("MPP(0.5) Z0*Z1*Z3*Z4 Z1*Z2*Z4*Z5 Z3*Z4*Z5*Z6")
    expected_circuit.append_from_stim_program_text("DETECTOR rec[-1] rec[-4]")
    expected_circuit.append_from_stim_program_text("DETECTOR rec[-2] rec[-5]")
    expected_circuit.append_from_stim_program_text("DETECTOR rec[-3] rec[-6]")
    expected_circuit.append_from_stim_program_text("MPP Z0*Z1*Z3*Z4 Z1*Z2*Z4*Z5 Z3*Z4*Z5*Z6")
    expected_circuit.append_from_stim_program_text("DETECTOR rec[-1] rec[-4]")
    expected_circuit.append_from_stim_program_text("DETECTOR rec[-2] rec[-5]")
    expected_circuit.append_from_stim_program_text("DETECTOR rec[-3] rec[-6]")
    expected_circuit.append_from_stim_program_text("MPP Z2*Z5*Z6")
    expected_circuit.append_from_stim_program_text("OBSERVABLE_INCLUDE(0) rec[-1]")
    logical = np.array([2, 5, 6])
    assert expected_circuit == gen_stim_circuit_memory_experiment(hamming_code, logical, 3, 0.5)
