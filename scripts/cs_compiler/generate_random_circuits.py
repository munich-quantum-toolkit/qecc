# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
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

import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm2 import dumps


def random_universal_circuit(
    num_qubits: int, depth: int, gate_probs: dict[str, float] | None = None, seed: int | None = None
) -> QuantumCircuit:
    """Generate a random universal quantum circuit using H, T, CNOT, and ID gates.

    Each depth layer assigns one operation per qubit (unless it's part of a CNOT).
    Avoids consecutive identical non-ID single-qubit gates (even across ID layers).

    Args:
        num_qubits (int): Number of qubits in the circuit.
        depth (int): Number of layers.
        gate_probs (dict): Probabilities for each gate, e.g. {"h": 0.3, "t": 0.3, "cx": 0.2, "id": 0.2}. All gates (h, t, cx, id) must be present.
        seed (int, optional): RNG seed for reproducibility.

    Returns:
        QuantumCircuit: Randomly generated circuit.
    """
    if gate_probs is None:
        gate_probs = {"h": 0.15, "t": 0.15, "cx": 0.15, "id": 0.55}

    # Normalize probabilities
    total = sum(gate_probs.values())
    gate_probs = {k: v / total for k, v in gate_probs.items()}

    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(num_qubits)

    gates = list(gate_probs.keys())
    probs = list(gate_probs.values())

    # Track last gate and last non-id single-qubit gate
    last_gate = ["id"] * num_qubits
    last_non_id_gate = ["id"] * num_qubits

    for _ in range(depth):
        available_qubits = set(range(num_qubits))

        for q in available_qubits.copy():
            if q not in available_qubits:
                continue  # already consumed by a CNOT

            # Draw a gate with back-to-back restrictions
            while True:
                gate = rng.choice(gates, p=probs)

                # Always allow id
                if gate == "id":
                    break

                # For single-qubit gates, avoid repeating last non-id gate
                if gate in {"h", "t"} and gate != last_non_id_gate[q]:
                    break

                # For CX, handled separately
                if gate == "cx":
                    break

            # Two-qubit gate handling
            if gate == "cx":
                others = list(available_qubits - {q})
                if not others:
                    fallback_gates = [g for g in ["h", "t", "id"] if g == "id" or g != last_non_id_gate[q]]
                    fallback_probs = [gate_probs[g] for g in fallback_gates]
                    fallback_total = sum(fallback_probs)
                    gate = rng.choice(
                        fallback_gates,
                        p=[p / fallback_total for p in fallback_probs],
                    )
                else:
                    target = rng.choice(others)
                    if rng.random() < 0.5:
                        circuit.cx(q, target)
                    else:
                        circuit.cx(target, q)
                    available_qubits.discard(q)
                    available_qubits.discard(target)
                    last_gate[q] = last_gate[target] = "cx"
                    # don't update last_non_id_gate (CX isn't a single-qubit gate)
                    continue

            # Apply single-qubit gate
            if gate == "h":
                circuit.h(q)
            elif gate == "t":
                circuit.t(q)
            elif gate == "id":
                pass  # do nothing

            available_qubits.discard(q)
            last_gate[q] = gate
            if gate != "id":
                last_non_id_gate[q] = gate

    return circuit


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

    if gate_distr_type not in gate_probs_options:
        msg = f"Invalid gate_distr_type '{gate_distr_type}'. Must be one of: {list(gate_probs_options.keys())}"
        raise ValueError(msg)

    print(f"Generating {num_circuits} {gate_distr_type} circuits for n={n}, depth={depth}...")

    skipped = 0
    generated = 0

    for seed in range(num_circuits):
        filename = folder / f"{n}_{seed}.qasm"

        if filename.exists():
            print(f"  → Skipping existing circuit: {filename.name}")
            skipped += 1
            continue
        qc = random_universal_circuit(
            num_qubits=n, depth=depth, seed=seed, gate_probs=gate_probs_options[gate_distr_type]
        )

        with filename.open("w", encoding="utf-8") as f:
            f.write(dumps(qc))

        generated += 1

        if (seed + 1) % 50 == 0 or seed == 0:
            print(f"  → Generated {seed + 1}/{num_circuits}")

    print(f"🟢 Finished: {generated} new circuits generated, {skipped} skipped.")
    print(f"Saved in: {folder}")


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
