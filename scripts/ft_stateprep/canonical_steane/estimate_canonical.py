# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Estimate logical error rate for CSS state preparation circuits for a given code and physical error rate."""

from __future__ import annotations

import argparse

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import SteaneNDFTStatePrepSimulator, heuristic_prep_circuit
from mqt.qecc.circuit_synthesis.state_prep import canonical_steane_type_prep_circuits, random_steane_type_prep_circuits
from mqt.qecc.codes import HexagonalColorCode, SquareOctagonColorCode

codes = {"golay": CSSCode.from_code_name("golay"), "hex": HexagonalColorCode(7), "sqoct": SquareOctagonColorCode(7)}


def main() -> None:
    """Run the logical error rate estimation for a given code and physical error rate."""
    parser = argparse.ArgumentParser(description="Estimate logical error rate for CSS state preparation circuits")
    parser.add_argument(
        "code",
        type=str,
        help="Code for which to estimate logical error rate. Available codes: " + ", ".join(codes),
    )
    parser.add_argument("-p", "--p_error", type=float, help="Physical error rate")
    parser.add_argument("-n", "--n_errors", type=int, default=500, help="Number of errors to sample")
    parser.add_argument(
        "--perm",
        type=str,
        default="canonical",
        help="Permutation for state preparation. Can be 'canonical' or 'random'",
    )
    parser.add_argument(
        "-e", "--error_type", type=str, default="X", help="Type of error to use for simulation. Can be 'X' or 'Z'."
    )

    args = parser.parse_args()
    code_name = args.code
    if code_name in codes:
        code = codes[code_name]
    else:
        raise ValueError("Code " + code_name + " not available. Available codes: " + ", ".join(codes))

    if args.perm == "canonical":
        qc1, qc2, qc3, qc4 = canonical_steane_type_prep_circuits(code)
        perm = "canonical"
    elif args.perm == "random":
        qc1, qc2, qc3, qc4, perm = random_steane_type_prep_circuits(code)
    else:
        raise ValueError("Permutation " + args.perm + " not available. Available permutations: 'canonical, random'")

    sim = SteaneNDFTStatePrepSimulator(
        qc1, qc2, code, qc3, qc4, check_circuit=heuristic_prep_circuit(code, zero_state=False).circ
    )
    sim.set_p(args.p_error, args.p_error * 0.01)

    if args.error_type == "X":
        p_l, r_a, num_logical_errors, total_shots, p_l_error, r_a_error = sim.logical_error_rate(
            min_errors=args.n_errors
        )
    elif args.error_type == "Z":
        p_l, r_a, num_logical_errors, total_shots, p_l_error, r_a_error = sim.secondary_logical_error_rate(
            min_errors=args.n_errors
        )
    else:
        msg = "Error type should be either X or Z."
        raise ValueError(msg)

    print(
        ";".join([
            str(x) for x in [args.p_error, p_l, r_a, num_logical_errors, total_shots, p_l_error, r_a_error, perm]
        ])
    )


if __name__ == "__main__":
    main()
