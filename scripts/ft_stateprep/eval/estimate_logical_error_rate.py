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

from qiskit import QuantumCircuit

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis import (
    NoisyNDFTStatePrepSimulator,
    gate_optimal_prep_circuit,
    gate_optimal_verification_circuit,
    heuristic_prep_circuit,
    heuristic_verification_circuit,
    naive_verification_circuit,
)
from mqt.qecc.codes import HexagonalColorCode, SquareOctagonColorCode


def main() -> None:
    """Run the logical error rate estimation for a given code and physical error rate."""
    available_codes = ["steane", "tetrahedral", "shor", "surface", "cc_4_8_8", "cc_6_6_6", "hamming", "carbon"]
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
    parser.add_argument("-n", "--n_errors", type=int, default=500, help="Number of errors to sample")
    parser.add_argument("--exact_circ", default=False, action="store_true", help="Use exact synthesis")
    parser.add_argument("--heuristic_circ", dest="exact_circ", action="store_false", help="Use heuristic synthesis")
    parser.add_argument("--exact_ver", default=True, action="store_true", help="Use exact verification")
    parser.add_argument("--heuristic_ver", dest="exact_ver", action="store_false", help="Use heuristic verification")
    parser.add_argument("--naive_ver", default=False, action="store_true", help="Use naive verification")
    parser.add_argument("--no_ver", action="store_true", help="Use no verification")
    parser.add_argument(
        "-d", "--distance", type=int, default=3, help="Code Distance (only required for surface and color codes)"
    )
    parser.add_argument("--no_parallel_gates", default=False, action="store_true")

    args = parser.parse_args()
    code_name = args.code
    if "surface" in code_name:
        d = args.distance
        code = CSSCode.from_code_name("surface", d)
        code_name = f"rotated_surface_d{d}"
    elif "cc_4_8_8" in code_name:
        d = 5
        code = SquareOctagonColorCode(d)
    elif "cc_6_6_6" in code_name:
        d = 5
        code = HexagonalColorCode(d)
    elif code_name in available_codes:
        code = CSSCode.from_code_name(code_name)
    else:
        raise ValueError("Code " + code_name + " not available. Available codes: " + ", ".join(available_codes))

    prefix = (Path(__file__) / "../circuits/").resolve()
    sp_circ_name = "opt" if args.exact_circ else "heuristic"
    ver_circ_name = ("opt" if args.exact_ver else "heuristic") if not args.naive_ver else "naive"

    state_name = "zero" if args.zero_state else "plus"
    ft_name = "non_ft" if args.no_ver else "ft"
    circ_file = f"{state_name}_{ft_name}_{sp_circ_name}_{ver_circ_name}.qasm"

    # check if file exists
    if not (prefix / code_name / circ_file).exists():
        # create circuit
        circ = None
        if args.exact_circ:
            circ = gate_optimal_prep_circuit(code, zero_state=args.zero_state, max_timeout=600)
        else:
            circ = heuristic_prep_circuit(code, zero_state=args.zero_state)

        assert circ is not None
        if args.naive_ver:
            qc = naive_verification_circuit(circ)
        elif args.exact_ver:
            qc = gate_optimal_verification_circuit(circ, max_timeout=600)
        else:
            qc = heuristic_verification_circuit(circ)
    else:
        # load circuit from file
        qc = QuantumCircuit.from_qasm_file(prefix / code_name / circ_file)

    sim = NoisyNDFTStatePrepSimulator(
        qc,
        code=code,
        p=args.p_error,
        p_idle=args.p_idle_factor * args.p_error,
        zero_state=args.zero_state,
        parallel_gates=not args.no_parallel_gates,
    )
    res = sim.logical_error_rate(min_errors=args.n_errors)
    print(",".join([str(x) for x in res]))


if __name__ == "__main__":
    main()
