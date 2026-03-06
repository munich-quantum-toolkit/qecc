# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Methods and utilities for synthesizing fault-tolerant circuits and gadgets."""

from __future__ import annotations

from .cat_states import CatStatePreparationExperiment, cat_state_balanced_tree, cat_state_line
from .circuit_utils import qiskit_to_stim_circuit
from .circuits import CliffordIsometry, CNOTCircuit
from .encoding import (
    cnot_encoding_circuit,
    depth_optimal_encoding_circuit,
    depth_optimal_encoding_circuit_non_css,
    gate_optimal_encoding_circuit,
    gottesman_encoding_circuit,
    synthesize_encoding_circuit,
)
from .noise import CircuitLevelNoiseIdlingParallel, CircuitLevelNoiseIdlingSequential
from .simulation import LutDecoder, SteaneNDFTStatePrepSimulator, VerificationNDFTStatePrepSimulator
from .state_prep import (
    FaultyStatePrepCircuit,
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
from .synthesis import (
    SynthesisConfig,
    synthesize_cnot,
    synthesize_non_css,
)

__all__ = [
    "CNOTCircuit",
    "CatStatePreparationExperiment",
    "CircuitLevelNoiseIdlingParallel",
    "CircuitLevelNoiseIdlingSequential",
    "CliffordIsometry",
    "DeterministicVerification",
    "DeterministicVerificationHelper",
    "FaultyStatePrepCircuit",
    "LutDecoder",
    "SteaneNDFTStatePrepSimulator",
    "SynthesisConfig",
    "VerificationNDFTStatePrepSimulator",
    "cat_state_balanced_tree",
    "cat_state_line",
    "cnot_encoding_circuit",
    "depth_optimal_encoding_circuit",
    "depth_optimal_encoding_circuit_non_css",
    "depth_optimal_prep_circuit",
    "gate_optimal_encoding_circuit",
    "gate_optimal_prep_circuit",
    "gate_optimal_verification_circuit",
    "gate_optimal_verification_stabilizers",
    "gottesman_encoding_circuit",
    "heuristic_prep_circuit",
    "heuristic_verification_circuit",
    "heuristic_verification_stabilizers",
    "naive_verification_circuit",
    "qiskit_to_stim_circuit",
    "synthesize_cnot",
    "synthesize_encoding_circuit",
    "synthesize_non_css",
]
