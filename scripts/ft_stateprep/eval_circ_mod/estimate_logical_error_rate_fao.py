# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Estimate logical error rates for CSS state preparation circuits."""

from __future__ import annotations

import argparse
from pathlib import Path

import stim

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis.circuit_utils import relabel_qubits
from mqt.qecc.circuit_synthesis.noise import CircuitLevelNoiseIdlingParallel
from mqt.qecc.circuit_synthesis.simulation import VerificationNDFTStatePrepSimulator

AVAILABLE_CODES: list[str] = [
    "cc_17_1_5",
    "cc_20_2_6",
    "cc_25_1_5",
    "cc_31_1_7",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Estimate logical error rate for a given code.")

    parser.add_argument(
        "code",
        type=str,
        help="Code name. Available: " + ", ".join(AVAILABLE_CODES),
    )
    parser.add_argument("-p", "--p_error", type=float, required=True, help="Physical error rate p.")
    parser.add_argument(
        "-p_idle_factor",
        "--p_idle_factor",
        type=float,
        default=0.01,
        help="Multiplier for idle error probability.",
    )
    parser.add_argument("--zero_state", default=True, action="store_true", help="Prepare logical |0>.")
    parser.add_argument(
        "--plus_state",
        default=False,
        dest="zero_state",
        action="store_false",
        help="Prepare logical |+> instead.",
    )
    parser.add_argument("--x_errors", default=True, action="store_true", help="Compute X-error LER.")
    parser.add_argument(
        "--z_errors",
        default=False,
        dest="x_errors",
        action="store_false",
        help="Compute Z-error LER.",
    )
    parser.add_argument(
        "-n",
        "--n_errors",
        type=int,
        default=500,
        help="Minimum number of errors for Monte Carlo estimator.",
    )
    parser.add_argument(
        "--check-matrix-path",
        type=Path,
        default=Path(__file__).resolve().parent / "check_matrix" / "flag_at_origin",
    )
    parser.add_argument(
        "--circuits-path",
        type=Path,
        default=Path(__file__).resolve().parent / "circuits" / "flag_at_origin",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core simulation with remapping (your new requirement)
# ---------------------------------------------------------------------------
def run_simulation(
    code_name: str,
    p: float,
    p_idle_factor: float,
    min_errors: int,
    zero_state: bool,
    x_errors: bool,
    check_matrix_path: Path,
    circuits_path: Path,
) -> tuple[float, float, int, int]:
    """Runs the simulation for a given code and returns (p_L, acceptance, errors, shots)."""
    # Load code
    code_file = check_matrix_path / f"{code_name}.txt"
    if not code_file.exists():
        msg = f"Code file not found: {code_file}"
        raise FileNotFoundError(msg)
    code = CSSCode.from_file(str(code_file))

    # Load circuit
    circ_file = circuits_path / f"{code_name}.stim"
    if not circ_file.exists():
        msg = f"Circuit file not found: {circ_file}"
        raise FileNotFoundError(msg)

    circuit = stim.Circuit.from_file(str(circ_file))

    # Build noise model
    noise = CircuitLevelNoiseIdlingParallel(
        p,
        p,
        p * 2 / 3,  # same as before
        p,
        p * p_idle_factor,
    )

    # ----------------------------------------------------------------------
    # NEW: Apply noise & remap qubits (your colleague's fix)
    # ----------------------------------------------------------------------
    noisy = noise.apply(circuit)

    n_measured = noisy.num_measurements
    n_code = code.n

    # mapping: measurement qubits go to the end; data qubits go to front
    mapping = {i: i + n_code for i in range(n_measured)} | {
        i: i - n_measured for i in range(n_measured, n_measured + n_code)
    }

    circuit_relabelled = relabel_qubits(circuit, mapping)

    # Create simulator
    sim = VerificationNDFTStatePrepSimulator(
        state_prep_circ=circuit_relabelled,
        code=code,
        zero_state=zero_state,
    )

    # Run Monte-Carlo
    if x_errors:
        # returns: (p_L, acceptance_rate, num_errors, shots)
        result = sim.logical_error_rate(noise=noise, min_errors=min_errors)
    else:
        result = sim.secondary_logical_error_rate(noise=noise, p=p, min_errors=min_errors)

    return (
        float(result[0]),
        float(result[1]),
        int(result[2]),
        int(result[3]),
    )


# ---------------------------------------------------------------------------
# Entry point (CSV output for bash)
# ---------------------------------------------------------------------------
def main() -> None:
    """Main function to parse arguments and run the simulation."""
    args = parse_args()

    p_logical, acceptance, errors, shots = run_simulation(
        code_name=args.code,
        p=args.p_error,
        p_idle_factor=args.p_idle_factor,
        min_errors=args.n_errors,
        zero_state=args.zero_state,
        x_errors=args.x_errors,
        check_matrix_path=args.check_matrix_path,
        circuits_path=args.circuits_path,
    )

    # CSV-formatted output: p_L, acceptance, errors, shots
    print(f"{p_logical},{acceptance},{errors},{shots}")


if __name__ == "__main__":
    main()
