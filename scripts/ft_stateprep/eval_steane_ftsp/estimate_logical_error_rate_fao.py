# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Estimate logical error rates for CSS state preparation circuits (flag-at-origin comparison)."""

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
        "--shots",
        type=int,
        default=100000,
        help="Total shots. With the min-errors estimator this is the HARD CAP (stop at whichever "
        "comes first: n_errors or shots). Pass a large value for low p.",
    )
    parser.add_argument("--shots_per_batch", type=int, default=100000, help="Shots per sampling batch.")
    parser.add_argument(
        "--fixed-shots",
        dest="fixed_shots",
        default=False,
        action="store_true",
        help="Run exactly --shots shots and report whatever results (disables the min-errors early stop).",
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
# Core simulation with remapping + reset fix
# ---------------------------------------------------------------------------
def run_simulation(
    code_name: str,
    p: float,
    p_idle_factor: float,
    min_errors: int,
    shots: int,
    shots_per_batch: int,
    fixed_shots: bool,
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

    # Build noise model (identical parameterization to the main pipeline: p_meas=(2/3)p, p_init=p)
    noise = CircuitLevelNoiseIdlingParallel(
        p,
        p,
        p * 2 / 3,
        p,
        p * p_idle_factor,
    )

    # ----------------------------------------------------------------------
    # Apply noise & remap qubits (colleague's fix): measurement qubits go to the
    # end; data qubits go to the front so the simulator sees data at [0, code.n).
    # ----------------------------------------------------------------------
    noisy = noise.apply(circuit)

    n_measured = noisy.num_measurements
    n_code = code.n

    mapping = {i: i + n_code for i in range(n_measured)} | {
        i: i - n_measured for i in range(n_measured, n_measured + n_code)
    }

    circuit_relabelled = relabel_qubits(circuit, mapping)

    # ----------------------------------------------------------------------
    # RESET FIX (fair comparison): the .stim circuits contain NO reset operations
    # at all -- every qubit (data AND flag/ancilla) relies on implicit |0>, so the
    # noise model injects no initialization/state-prep error anywhere. Prepend a
    # reset on ALL qubits so each carries DEPOLARIZE1(p_init), matching the main
    # pipeline where every physical qubit is effectively reset. (Data-only would
    # leave the flag/ancilla qubits noise-free and bias this method's LER downward.)
    # ----------------------------------------------------------------------
    reset_prefix = stim.Circuit()
    reset_prefix.append("R", list(range(circuit_relabelled.num_qubits)))
    circuit_relabelled = reset_prefix + circuit_relabelled

    # Create simulator
    sim = VerificationNDFTStatePrepSimulator(
        state_prep_circ=circuit_relabelled,
        code=code,
        zero_state=zero_state,
    )

    # Run Monte-Carlo. at_least_min_errors=True -> stop at min_errors OR the shots cap,
    # whichever first (requires the `while i <= total_batches:` library edit).
    at_least_min_errors = not fixed_shots
    if x_errors:
        result = sim.logical_error_rate(
            noise=noise,
            shots=shots,
            shots_per_batch=shots_per_batch,
            at_least_min_errors=at_least_min_errors,
            min_errors=min_errors,
        )
    else:
        result = sim.secondary_logical_error_rate(
            noise=noise,
            p=p,
            shots=shots,
            shots_per_batch=shots_per_batch,
            at_least_min_errors=at_least_min_errors,
            min_errors=min_errors,
        )

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
        shots=args.shots,
        shots_per_batch=args.shots_per_batch,
        fixed_shots=args.fixed_shots,
        zero_state=args.zero_state,
        x_errors=args.x_errors,
        check_matrix_path=args.check_matrix_path,
        circuits_path=args.circuits_path,
    )

    # CSV-formatted output: p_L, acceptance, errors, shots
    print(f"{p_logical},{acceptance},{errors},{shots}")


if __name__ == "__main__":
    main()
