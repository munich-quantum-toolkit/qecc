# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Simulate existing QASM circuits and record results to CSV.

Each circuit is loaded from a folder structure like:
    circuits/{n}/{n}_{seed}.qasm

Example:
    python simulate_circuits.py --n 128 --input_dir circuits --output_csv results_128.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import TYPE_CHECKING

from qiskit.qasm2 import loads

from mqt.qecc.circuit_compilation import CodeSwitchGraph, count_code_switches

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def append_to_csv(csv_path: Path, row: dict) -> None:
    """Append a row to a CSV file, creating it with headers if it doesn't exist."""
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def already_done(csv_path: Path, seed: int) -> bool:
    """Return True if the CSV contains a row whose 'seed' column equals the given seed."""
    if not csv_path.exists():
        return False

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # safe integer comparison
            if int(row["seed"]) == seed:
                return True
    return False


def run_trial(qc: QuantumCircuit, n_qubits: int, depth: int, seed: int, probs_type: str) -> dict:
    """Run a single trial comparing naive and min-cut code switch counting."""
    t0_lookahead = time.time()
    naive = count_code_switches(qc)[0]
    t1_lookahead = time.time()

    builder = CodeSwitchGraph()
    builder.build_from_qiskit(qc, one_way_transversal_cnot=True)

    t0_mincut = time.time()
    switches_mc, _, _, _ = builder.compute_min_cut()
    t1_mincut = time.time()

    return {
        "n_qubits": n_qubits,
        "layer_per_qubit": depth,
        "seed": seed,
        "gate_probs_type": probs_type,
        "naive": naive,
        "mincut": switches_mc,
        "abs_saving": naive - switches_mc,
        "rel_saving": (naive - switches_mc) / naive if naive > 0 else None,
        "t_naive": t1_lookahead - t0_lookahead,
        "t_mincut": t1_mincut - t0_mincut,
    }


def main() -> None:
    """Parse arguments and run simulation trials."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--qasm_path", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--probs_type", type=str, default="default")
    args = parser.parse_args()

    qc = loads(args.qasm_path.read_text())
    depth = 2 * args.n

    if already_done(args.output_csv, args.seed):
        print(f"Seed {args.seed} already done, skipping.")
        return

    result = run_trial(qc, args.n, depth, args.seed, args.probs_type)
    append_to_csv(args.output_csv, result)


if __name__ == "__main__":
    main()
