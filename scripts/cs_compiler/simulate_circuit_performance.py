# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

r"""Simulate existing QASM circuits and record results to CSV.

Each circuit is loaded from a QASM file path given via ``--qasm_path``, e.g.
    circuits_performance_benchmarking/{n}/{n}_{seed}.qasm

Example:
    python simulate_circuit_performance.py \\
        --qasm_path circuits_performance_benchmarking/128/128_0.qasm \\
        --n 128 --seed 0 --output_csv results_performance_benchmarking/results_128.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import TYPE_CHECKING

from qiskit.qasm2 import loads

from mqt.qecc.circuit_compilation import MinimalCodeSwitchingCompiler, naive_switching

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def append_to_csv(csv_path: Path, row: dict) -> None:
    """Append a row to a CSV file, creating it with headers if it doesn't exist."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
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
            if int(row["seed"]) == seed:
                return True
    return False


def run_trial(qc: QuantumCircuit, n_qubits: int, depth: int, seed: int, probs_type: str) -> dict:
    """Run a single trial comparing naive and min-cut code switch counting."""
    t0_naive = time.time()
    naive = naive_switching(qc)[0]
    t1_naive = time.time()

    source_code = {"H", "CX"}
    sink_code = {"T", "CX"}
    builder = MinimalCodeSwitchingCompiler(source_code, sink_code)
    builder.build_from_qiskit(qc, one_way_gates={"CX"})

    t0_mincut_solver = time.time()
    switches_mc, _, _, _ = builder.compute_min_cut()
    t1_mincut_solver = time.time()

    return {
        "n_qubits": n_qubits,
        "layer_per_qubit": depth,
        "seed": seed,
        "gate_probs_type": probs_type,
        "naive": naive,
        "mincut": switches_mc,
        "abs_saving": naive - switches_mc,
        "rel_saving": (naive - switches_mc) / naive if naive > 0 else None,
        "t_naive": t1_naive - t0_naive,
        "t_mincut_solver": t1_mincut_solver - t0_mincut_solver,
    }


def main() -> None:
    """Parse arguments and run simulation trials."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--qasm_path", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--distr_type", type=str, default="even")
    args = parser.parse_args()

    qc = loads(args.qasm_path.read_text())
    depth = 2 * args.n

    if already_done(args.output_csv, args.seed):
        print(f"Seed {args.seed} already done, skipping.")
        return

    result = run_trial(qc, args.n, depth, args.seed, args.distr_type)
    append_to_csv(args.output_csv, result)


if __name__ == "__main__":
    main()
