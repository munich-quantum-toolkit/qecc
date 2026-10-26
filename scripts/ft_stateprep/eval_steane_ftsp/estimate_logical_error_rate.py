# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: S301, S403

"""Estimate logical error rate for CSS state preparation circuits for a given code and physical error rate."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from qiskit import QuantumCircuit

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis.noise import CircuitLevelNoiseIdlingParallel
from mqt.qecc.circuit_synthesis.simulation import SteaneNDFTStatePrepSimulator
from mqt.qecc.codes import HexagonalColorCode, RotatedSurfaceCode, SquareOctagonColorCode


def load_circuit_with_resets(path: Path) -> QuantumCircuit:
    """Load a state-preparation circuit from a QASM file and prepend a reset on every qubit.

    The synthesized QASM circuits initialize their qubits implicitly (``qreg q[n];``), so the
    stim circuit produced from them contains no explicit reset (``R``) operations. The
    circuit-level noise model attaches the initialization/state-preparation error
    (``DEPOLARIZE1(p_init)``) *only* to reset operations, which means that without explicit
    resets the state-preparation error is silently dropped from every noisy circuit.

    Prepending a reset on all qubits fixes this: after conversion to stim each ``reset`` becomes
    an ``R`` instruction, which the noise model then follows with ``DEPOLARIZE1(p_init)``. The
    reset lands at the very start of the block (before the encoding gates), which is the correct
    location for an initialization error.
    """
    qc = QuantumCircuit.from_qasm_file(path)
    prep = QuantumCircuit(*qc.qregs, *qc.cregs, name=qc.name)
    prep.reset(range(qc.num_qubits))
    return prep.compose(qc)


def main() -> None:
    """Run the logical error rate estimation for a given code and physical error rate."""
    available_codes = ["eve_20_2_6"]
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
    parser.add_argument(
        "-d", "--distance", type=int, default=3, help="Code Distance (only required for surface and color codes)"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=100000,
        help="Total number of shots. Used as a hard cap when --fixed-shots is set.",
    )
    parser.add_argument("--shots_per_batch", type=int, default=100000, help="Number of shots per sampling batch.")
    parser.add_argument(
        "--fixed-shots",
        dest="fixed_shots",
        default=False,
        action="store_true",
        help=(
            "Run exactly --shots shots and report whatever logical error rate results, instead of "
            "sampling until --n_errors logical errors have been found (which is unbounded for low p)."
        ),
    )

    args = parser.parse_args()
    code_name = args.code
    decoder = None
    if "surface" in code_name:
        d = args.distance
        code = RotatedSurfaceCode(d)
        code_name = f"rotated_surface_d{d}"
    elif code_name == "cc_4_8_8_d7":
        d = 7
        code = SquareOctagonColorCode(d)
        lut_path = (Path("__file__") / "../../eval/luts/decoder_488_7.pickle").resolve()
        if lut_path.exists():
            with lut_path.open("rb") as f:
                decoder = pickle.load(f)
        else:
            msg = "LUT file not found."
            raise ValueError(msg)
    elif code_name == "cc_6_6_6_d7":
        d = 7
        code = HexagonalColorCode(d)
    elif code_name == "cc_4_8_8_d5":
        d = 5
        code = SquareOctagonColorCode(d)
    elif code_name == "cc_6_6_6_d5":
        d = 5
        code = HexagonalColorCode(d)
    elif code_name in available_codes:
        prefix = (Path(__file__) / "../check_matrix/").resolve()
        matrix_file = prefix / (code_name + ".txt")
        code = CSSCode.from_file(matrix_file)
    else:
        raise ValueError("Code " + code_name + " not available. Available codes: " + ", ".join(available_codes))

    prefix = (Path(__file__) / "../circuits/").resolve()
    circ_file_core = f"{code_name}_heuristic_"

    circuits = []
    # load circuit from file (with explicit resets so that init noise is applied, see above)
    for id_ in [0, 1, 2, 3]:
        circ_file = circ_file_core + str(id_)
        path = prefix / code_name / (circ_file + ".qasm")
        circuits.append(load_circuit_with_resets(path))

    sim = SteaneNDFTStatePrepSimulator(
        circ1=circuits[0],
        circ2=circuits[1],
        code=code,
        circ3=circuits[2],
        circ4=circuits[3],
        decoder=decoder,
    )
    p = args.p_error
    noise = CircuitLevelNoiseIdlingParallel(p, p, p * 2 / 3, p, p * args.p_idle_factor)
    # at_least_min_errors=True  -> sample until n_errors logical errors are found (unbounded time)
    # at_least_min_errors=False -> run exactly --shots shots and report whatever rate results
    at_least_min_errors = not args.fixed_shots
    if args.x_errors:
        res = sim.logical_error_rate(
            noise=noise,
            shots=args.shots,
            shots_per_batch=args.shots_per_batch,
            at_least_min_errors=at_least_min_errors,
            min_errors=args.n_errors,
        )
    else:
        res = sim.secondary_logical_error_rate(
            noise=noise,
            p=p,
            shots=args.shots,
            shots_per_batch=args.shots_per_batch,
            at_least_min_errors=at_least_min_errors,
            min_errors=args.n_errors,
        )

    print(",".join([str(x) for x in res]))


if __name__ == "__main__":
    main()
