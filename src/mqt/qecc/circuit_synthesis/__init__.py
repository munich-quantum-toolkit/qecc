# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods and utilities for synthesizing fault-tolerant circuits and gadgets."""

from __future__ import annotations

from .cat_states import CatStatePreparationExperiment, cat_state_balanced_tree, cat_state_line
from .circuit_utils import qiskit_to_stim_circuit
from .encoding import depth_optimal_encoding_circuit, gate_optimal_encoding_circuit, heuristic_encoding_circuit
from .simulation import LutDecoder, NoisyNDFTStatePrepSimulator
from .state_prep import (
    StatePrepCircuit,
    depth_optimal_prep_circuit,
    gate_optimal_prep_circuit,
    gate_optimal_verification_circuit,
    gate_optimal_verification_stabilizers,
    heuristic_prep_circuit,
    heuristic_verification_circuit,
    heuristic_verification_stabilizers,
    naive_verification_circuit,
)
from .state_prep_det import DeterministicVerification, DeterministicVerificationHelper

__all__ = [
    "CatStatePreparationExperiment",
    "DeterministicVerification",
    "DeterministicVerificationHelper",
    "LutDecoder",
    "NoisyNDFTStatePrepSimulator",
    "StatePrepCircuit",
    "cat_state_balanced_tree",
    "cat_state_line",
    "depth_optimal_encoding_circuit",
    "depth_optimal_prep_circuit",
    "gate_optimal_encoding_circuit",
    "gate_optimal_prep_circuit",
    "gate_optimal_verification_circuit",
    "gate_optimal_verification_stabilizers",
    "heuristic_encoding_circuit",
    "heuristic_prep_circuit",
    "heuristic_verification_circuit",
    "heuristic_verification_stabilizers",
    "naive_verification_circuit",
    "qiskit_to_stim_circuit",
]
