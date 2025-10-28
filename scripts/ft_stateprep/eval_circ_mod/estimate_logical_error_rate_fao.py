# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Estimate logical error rate for CSS state preparation circuits for a given code and physical error rate."""

from __future__ import annotations

import argparse
from pathlib import Path

import stim

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis.noise import CircuitLevelNoiseIdlingParallel
from mqt.qecc.circuit_synthesis.simulation import VerificationNDFTStatePrepSimulator


def main() -> None:
    """Run the logical error rate estimation for a given code and physical error rate."""
    available_codes = ["cc_17_1_5", "cc_20_2_6", "cc_25_1_5", "cc_31_1_7"]
    parser = argparse.ArgumentParser(description="Estimate logical error rate for CSS state preparation circuits")
    parser.add_argument(
        "code",
        type=str,
        help="Code for which to estimate logical error rate. Available codes: " + ", ".join(available_codes),
    )
    parser.add_argument("-p", "--p_error", type=float, help="Physical error rate")
    parser.add_argument("-p_idle_factor", "--p_idle_factor", type=float, default=0.01, help="Idling error rate")
    parser.add_argument("--zero_state", default=True, action="store_true", help="Synthesize logical |0> state.")
    parser.add_argument(
        "--plus_state", default=False, dest="zero_state", action="store_false", help="Synthesize logical |+> state."
    )
    parser.add_argument("--x_errors", default=True, action="store_true", help="Calculate error rates for X-errors")
    parser.add_argument(
        "--z_errors", default=False, dest="x_errors", action="store_false", help="Calculate error rates for Z errors"
    )
    parser.add_argument("-n", "--n_errors", type=int, default=500, help="Number of errors to sample")

    args = parser.parse_args()
    code_name = args.code
    check_matrix_path = (Path("__file__") / "../check_matrix/flag_at_origin/").resolve()
    if not check_matrix_path.exists():
        msg = "Check matrix path does not exist."
        raise ValueError(msg)
    code = CSSCode.from_file(check_matrix_path / (code_name + ".txt"))

    prefix = (Path(__file__) / "../circuits/flag_at_origin/" / (code_name + ".stim")).resolve()

    with Path(prefix).open(encoding="utf-8") as f:
        circuit_text = f.read()

    stim_circuit = stim.Circuit(circuit_text)

    sim = VerificationNDFTStatePrepSimulator(
        state_prep_circ=stim_circuit,
        code=code,
        zero_state=args.zero_state,
    )
    p = args.p_error
    noise = CircuitLevelNoiseIdlingParallel(p, p, p * 2 / 3, p, p * args.p_idle_factor)
    if args.x_errors:
        print(f"Starting X error rate estimation with {args.n_errors} Errors...")
        res = sim.logical_error_rate(noise=noise, min_errors=args.n_errors)
    else:
        res = sim.secondary_logical_error_rate(noise=noise, p=p, min_errors=args.n_errors)

    print(",".join([str(x) for x in res]))


if __name__ == "__main__":
    main()
