# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generate and save random universal circuits as QASM 2.0 files.

This script creates a set of random circuits for a given number of qubits `n`
and saves them under `output_dir/n/` as QASM 2.0 files named `{n}_{seed}.qasm`.

Example:
    python generate_random_circuits.py --n 128 --num_circuits 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qiskit.qasm2 import dumps

# Import your circuit generator (adjust this import!)
from mqt.qecc.circuit_compilation import random_universal_circuit


def generate_circuits(n: int, num_circuits: int, output_dir: Path, gate_distr_type: str = "even") -> None:
    """Generate and save random universal circuits.

    Args:
        n: Number of qubits.
        num_circuits: Number of circuits to generate.
        output_dir: Base directory to store generated circuits.
        gate_distr_type: Type of gate distribution to use ('even', 'ht_heavy', 'cx_heavy').
    """
    depth = 2 * n
    folder = output_dir / gate_distr_type / str(n)
    folder.mkdir(parents=True, exist_ok=True)

    gate_probs_options = {
        "even": {"h": 0.15, "t": 0.15, "cx": 0.15, "id": 0.55},
        "ht_heavy": {"h": 0.2, "t": 0.2, "cx": 0.05, "id": 0.55},
        "cx_heavy": {"h": 0.1, "t": 0.1, "cx": 0.3, "id": 0.5},
    }

    print(f"Generating {num_circuits} {gate_distr_type} circuits for n={n}, depth={depth}...")

    for seed in range(num_circuits):
        qc = random_universal_circuit(
            num_qubits=n, depth=depth, seed=seed, gate_probs=gate_probs_options[gate_distr_type]
        )
        filename = folder / f"{n}_{seed}.qasm"

        with filename.open("w", encoding="utf-8") as f:
            f.write(dumps(qc))

        if seed % 50 == 0:
            print(f"  → Generated {seed}/{num_circuits}")

    print(f"✅ Finished generating circuits for n={n}. Saved in: {folder}")


def main() -> None:
    """Parse arguments and trigger circuit generation."""
    parser = argparse.ArgumentParser(description="Generate random universal circuits.")
    parser.add_argument("--n", type=int, required=True, help="Number of qubits.")
    parser.add_argument("--num_circuits", type=int, default=400, help="Number of circuits to generate.")
    parser.add_argument(
        "--distr_type", type=str, default="even", help="Gate distribution type: 'even', 'ht_heavy', or 'cx_heavy'."
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("circuits_performance_benchmarking"), help="Base output directory."
    )
    args = parser.parse_args()

    generate_circuits(args.n, args.num_circuits, args.output_dir, args.distr_type)


if __name__ == "__main__":
    main()
