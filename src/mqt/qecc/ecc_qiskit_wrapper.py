# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Qiskit wrapper for the ECC Framework."""

from __future__ import annotations

import argparse
import locale
import pathlib
from typing import TYPE_CHECKING

from qiskit.qasm2 import dump, load, loads
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    amplitude_damping_error,
    depolarizing_error,
    pauli_error,
)

from . import apply_ecc

if TYPE_CHECKING:  # pragma: no cover
    from qiskit.result import Result


def compose_error(error: QuantumError, new_error: QuantumError) -> QuantumError:
    """Compose two quantum errors."""
    return new_error if error is None else error.compose(new_error)


def create_noise_model(n_model: str, p_error: float) -> NoiseModel:
    """Create a noise model for a given error rate and error model."""
    # Create an empty noise model
    noise_model = NoiseModel()
    error = None
    for char in n_model:
        if char == "B":
            # Add a bit flip error channel
            new_error = pauli_error([("X", p_error), ("I", 1 - p_error)])
            error = compose_error(error, new_error)
        elif char == "D":
            # Add a depolarization error channel
            new_error = depolarizing_error(p_error, 1)
            error = compose_error(error, new_error)
        elif char == "A":
            # simple amplitude damping error channel which mimics energy loss to the environment (T1-error)
            new_error = amplitude_damping_error(p_error)
            error = compose_error(error, new_error)
        elif char == "P":
            # Add a phase flip error channel
            new_error = pauli_error([("Z", p_error), ("I", 1 - p_error)])
            error = compose_error(error, new_error)

        else:
            raise ValueError("Unknown error type in noise model: " + char)

    assert error is not None

    # single qubit and multi qubit operations are treated the same at the moment
    noise_model.add_all_qubit_quantum_error(
        error, ["u1", "u2", "u3", "h", "id", "t", "tdg", "sdg", "rx", "ry", "rz", "s"]
    )
    noise_model.add_all_qubit_quantum_error(error.tensor(error), ["cx", "swap"])
    noise_model.add_all_qubit_quantum_error(error.tensor(error).tensor(error), ["cswap"])

    return noise_model


def print_simulation_results(result: Result, n_shots: int, threshold_probability: float = 0) -> None:
    """Print the simulation results."""
    printed_results = 0
    summarized_counts: dict[str, int] = {}
    result_counts = result.get_counts()
    for result_id in result_counts:
        sub_result = result_id.split(" ")[-1]
        if sub_result not in summarized_counts:
            summarized_counts[sub_result] = 0
        summarized_counts[sub_result] += result_counts[result_id]

    for result_id in sorted(summarized_counts.keys()):
        # Print all results > threshold_probability
        if summarized_counts[result_id] / n_shots > threshold_probability or printed_results == 0:
            str(result_id)
            printed_results += 1
            if printed_results == 1000:
                break


def main() -> None:
    """Run main function of the Qiskit wrapper for the ECC Framework."""
    parser = argparse.ArgumentParser(description="Qiskit wrapper for the ECC Framework")
    parser.add_argument(
        "-m",
        type=str,
        default="D",
        help="Define the error_channels (e.g., -m APD), available errors channels are amplitude "
        'damping (A), phase flip (P), bit flip (B), and depolarization (D) (Default="D")',
    )
    parser.add_argument(
        "-p",
        type=float,
        default=0.001,
        help="Set the noise probability (Default=0.001)",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=2000,
        help="Set the number of shots for the simulation (Default=2000)",
    )
    parser.add_argument("-s", type=int, default=0, help="Set a seed (Default=0)")
    parser.add_argument("-f", type=str, required=True, help="Path to a OpenQASM file")
    parser.add_argument(
        "-e",
        type=str,
        required=False,
        default=None,
        help="Export circuit with applied ECC as OpenQASM circuit instead of "
        'simulating it (e.g., -e "/path/to/new/openqasm_file") (Default=None)',
    )
    parser.add_argument(
        "-fs",
        type=str,
        default="stabilizer",
        help='Specify a simulator (Default="stabilizer", which is fast but does not support "non-Clifford gates"',
    )
    parser.add_argument(
        "-ecc",
        type=str,
        default="Q7Steane",
        help='Specify an ECC to be applied to the circuit. Currently available are "none", "Q3Shor", "Q5Laflamme", '
        '"Q7Steane", "Q9Shor", "Q9Surface", and "Q18Surface" (Default=Q7Steane)',
    )
    parser.add_argument(
        "-fq",
        type=int,
        default=100,
        help="Specify after how many qubit usages error correction is applied to it (Default=100)",
    )

    args = parser.parse_args()
    assert args is not None
    error_channels = args.m
    error_probability = args.p
    number_of_shots = args.n
    seed = args.s
    open_qasm_file = args.f

    forced_simulator = None if args.fs.lower() == "none" else args.fs

    ecc = None if args.ecc.lower() == "none" else args.ecc

    ecc_frequency = args.fq
    ecc_export_filename = args.e
    if forced_simulator is not None and "stabilizer" in forced_simulator and "A" in error_channels:
        pass

    # Creating the noise model
    if error_probability > 0:
        noise_model = create_noise_model(n_model=error_channels, p_error=error_probability)
    else:
        noise_model = NoiseModel()

    circ = load(open_qasm_file)

    if not any(gate.operation.name == "measure" for gate in circ.data):
        circ.measure_all()

    # Initializing the quantum circuit
    if ecc is not None:
        # Applying error correction to the circuit
        result = apply_ecc(circ, ecc, ecc_frequency)
        circ = loads(result["circ"])

    if ecc_export_filename is not None:
        with pathlib.Path(ecc_export_filename).open("w", encoding=locale.getpreferredencoding(False)) as f:
            dump(circ, f)
        return

    # Setting the simulator backend to the requested one
    simulator_backend = AerSimulator(method=forced_simulator, noise_model=noise_model)

    job_result = simulator_backend.run(circ, shots=number_of_shots, seed_simulator=seed).result()

    if job_result.status != "COMPLETED":
        raise RuntimeError("Simulation exited with status: " + str(job_result.status))

    print_simulation_results(job_result, number_of_shots)
