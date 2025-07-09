# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Estimate logical error rate for d=7 square-octagon color code zero state preparation circuit for a given physical error rate."""

from __future__ import annotations

import argparse
import pickle  # noqa: S403
from pathlib import Path

from qiskit import QuantumCircuit

from mqt.qecc.circuit_synthesis import (
    NoisyNDFTStatePrepSimulator,
)
from mqt.qecc.codes import SquareOctagonColorCode


def main() -> None:
    """Run the simulation."""
    parser = argparse.ArgumentParser(description="Estimate logical error rate for CSS state preparation circuits")
    parser.add_argument("-p", "--p_error", type=float, help="Physical error rate")
    parser.add_argument("-p_idle_factor", "--p_idle_factor", type=float, default=0.01, help="Idling error rate")
    parser.add_argument("--naive_ver", default=False, action="store_true", help="Use naive verification")
    parser.add_argument("-n", "--n_errors", type=int, default=500, help="Number of errors to sample")

    args = parser.parse_args()
    code = SquareOctagonColorCode(7)
    prefix = (Path(__file__) / "../circuits/cc_4_8_8_d7/").resolve()
    circ_name = "zero_ft_heuristic_mixed.qasm" if not args.naive_ver else "zero_ft_naive.qasm"
    qc = QuantumCircuit.from_qasm_file(prefix / circ_name)

    lut_path = (Path(__file__) / "../luts/decoder_488_7.pickle").resolve()
    with lut_path.open("rb") as f:
        lut = pickle.load(f)  # noqa: S301
    sim = NoisyNDFTStatePrepSimulator(qc, code, decoder=lut, p=args.p_error, p_idle=args.p_idle_factor * args.p_error)
    res = sim.logical_error_rate(min_errors=args.n_errors)
    print(",".join([str(x) for x in [args.p_error, *res]]))


if __name__ == "__main__":
    main()
