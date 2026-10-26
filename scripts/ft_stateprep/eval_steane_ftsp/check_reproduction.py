# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check that the synthesis recipes in recipes.py reproduce the published circuits.

Usage (from this directory):

    python check_reproduction.py --reference <zenodo_steane_ftsp/reference> [--codes a,b] [--synth-code both] [--no-ft]

Steps:
    1. Integrity: verify ``SHA256SUMS`` of the reference snapshot (abort on mismatch).
    2. Circuits: rebuild C1-C4 of every code and compare with ``circuits/ours/<code>/<code>_heuristic_{i}.qasm``:
       ``exact`` = same ordered CNOT list, ``dag`` = same DAG (only gates on disjoint qubits reordered),
       ``h`` = same Hadamard qubits. A circuit passes if ``dag`` and ``h`` hold.
    3. FT: the reference circuits of each code have the mutually disjoint fault sets the protocol needs.

The exit code is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import logging
import subprocess  # noqa: S404
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import qiskit
import stim
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from recipes import CODES, FIND_4_CODES, RECIPES

import mqt.qecc
from mqt.qecc.circuit_synthesis import CNOTCircuit, FaultyStatePrepCircuit
from mqt.qecc.circuit_synthesis.synthesis_utils import check_mutually_disjointness_spcs

Gate = tuple[str, tuple[int, ...]]


def check_integrity(reference: Path) -> list[str]:
    """Return the files of the reference snapshot whose SHA-256 does not match ``SHA256SUMS``."""
    bad = []
    for line in (reference / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        path = reference / name.lstrip("*")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    return bad


def gates(qc: QuantumCircuit) -> list[Gate]:
    """Return the ordered list of gates of a circuit, ignoring registers and barriers."""
    return [
        (inst.operation.name, tuple(qc.find_bit(q).index for q in inst.qubits))
        for inst in qc.data
        if inst.operation.name != "barrier"
    ]


def from_gates(num_qubits: int, gate_list: list[Gate]) -> QuantumCircuit:
    """Build a register-normalized circuit from a gate list (supports h and cx only)."""
    qc = QuantumCircuit(num_qubits)
    for name, qubits in gate_list:
        getattr(qc, name)(*qubits)
    return qc


def compare(generated: CNOTCircuit, reference: QuantumCircuit) -> dict[str, object]:
    """Compare a synthesized circuit with a reference circuit.

    ``exact`` uses the CNOT list of the synthesized circuit itself, because ``to_qiskit_circuit`` re-layers the CNOTs.
    """
    g, r = gates(generated.to_qiskit_circuit()), gates(reference)
    unsupported = {name for name, _ in g + r} - {"h", "cx"}
    n = generated.num_qubits()
    dag = (
        n == reference.num_qubits
        and not unsupported
        and circuit_to_dag(from_gates(n, g)) == circuit_to_dag(from_gates(n, r))
    )
    h_gen = {q[0] for name, q in g if name == "h"}
    h_ref = {q[0] for name, q in r if name == "h"}
    return {
        "exact": [tuple(c) for c in generated.cnots] == [q for name, q in r if name == "cx"],
        "dag": dag,
        "h": h_gen == h_ref,
        "n_cx": (sum(name == "cx" for name, _ in g), sum(name == "cx" for name, _ in r)),
        "unsupported": unsupported,
        "pass": bool(dag and h_gen == h_ref),
    }


def reference_qasm(reference: Path, code: str, i: int) -> Path:
    """Return the path of reference circuit C{i+1} of ``code``."""
    return reference / "circuits" / "ours" / code / f"{code}_heuristic_{i}.qasm"


def check_circuits(reference: Path, code: str, synth_code: str | None) -> list[dict[str, object]]:
    """Rebuild C1-C4 of ``code`` and compare them with the reference circuits."""
    recipe = RECIPES[code]
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", UserWarning)
        spcs = recipe(synth_code) if synth_code is not None else recipe()
    return [
        compare(spc.circ, QuantumCircuit.from_qasm_file(reference_qasm(reference, code, i)))
        for i, spc in enumerate(spcs)
    ]


def _disjoint(a: FaultyStatePrepCircuit, b: FaultyStatePrepCircuit, x_error: bool) -> bool:
    return (
        len(check_mutually_disjointness_spcs([a], [b], x_error=x_error))
        == len(check_mutually_disjointness_spcs([b], [a], x_error=x_error))
        == 2
    )


def check_ft(reference: Path, code: str) -> tuple[bool, list[str]]:
    """Check the fault-set disjointness of the four reference circuits (``check_4`` of the notebooks).

    C1/C2 and C3/C4 must have disjoint X fault sets; C3 and C4 must each have Z fault sets disjoint from C1 and C2.
    """
    css_code = CODES[code]()
    spcs = []
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", UserWarning)
        for i in range(4):
            qc = QuantumCircuit.from_qasm_file(reference_qasm(reference, code, i))
            circ = CNOTCircuit.from_qiskit_circuit(qc, init_all=True)
            spcs.append(FaultyStatePrepCircuit(circ, css_code.x_distance // 2, css_code.z_distance // 2, code=css_code))

        pairs = [(0, 1, True), (0, 2, False), (1, 2, False), (0, 3, False), (1, 3, False), (2, 3, True)]
        failed = [f"C{i + 1}/C{j + 1} {'X' if x else 'Z'}" for i, j, x in pairs if not _disjoint(spcs[i], spcs[j], x)]
    return not failed, failed


def environment() -> str:
    """Describe the software state (git HEAD of this repository and package versions)."""
    repo = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        cmd = ["git", "-C", str(repo), *args]
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()  # noqa: S603

    head = git("rev-parse", "--short", "HEAD") or "- (no git checkout)"
    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "src")
    return (
        f"mqt-qecc {mqt.qecc.__version__} | git HEAD {head}{' (src/ modified)' if dirty else ''} | "
        f"python {sys.version.split()[0]} | numpy {np.__version__} | qiskit {qiskit.__version__} | stim {stim.__version__}"
    )


def fmt(res: dict[str, object]) -> str:
    """Format a comparison result as a table cell."""
    status = "PASS" if res["pass"] else "FAIL"
    detail = "exact" if res["exact"] else ("dag" if res["dag"] else "diff")
    if not res["h"]:
        detail += ",H!"
    n_gen, n_ref = res["n_cx"]  # type: ignore[misc]
    cx = f"{n_gen}" if n_gen == n_ref else f"{n_gen}/{n_ref}"
    return f"{status} {detail} {cx}cx"


def main() -> int:
    """Run the checks and print the result table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True, help="path to zenodo_steane_ftsp/reference")
    parser.add_argument("--codes", default=",".join(RECIPES), help="comma-separated codes (default: all)")
    parser.add_argument(
        "--synth-code",
        choices=["original", "c1", "both"],
        default="original",
        help="code passed to the synthesis of C2-C4 for the find_4 codes (default: original)",
    )
    parser.add_argument("--no-ft", action="store_true", help="skip the fault-set disjointness check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    logging.getLogger("mqt").setLevel(logging.WARNING)

    reference = args.reference.resolve()
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    unknown = [c for c in codes if c not in RECIPES]
    if unknown:
        parser.error(f"unknown codes: {', '.join(unknown)}")

    print(environment(), flush=True)
    bad = check_integrity(reference)
    if bad:
        print(f"Reference integrity FAILED for: {', '.join(bad)}", flush=True)
        return 2
    print(f"Reference integrity OK ({reference})\n", flush=True)

    header = f"{'code':<32} {'C1':<18} {'C2':<18} {'C3':<18} {'C4':<18} {'FT':<6} {'time':>7}"
    print(header)
    print("-" * len(header), flush=True)
    all_ok = True
    for code in codes:
        variants: list[str | None] = [None]
        if code in FIND_4_CODES:
            variants = ["original", "c1"] if args.synth_code == "both" else [args.synth_code]
        ft_cell = "-"
        if not args.no_ft:
            ft_ok, ft_failed = check_ft(reference, code)
            ft_cell = "PASS" if ft_ok else "FAIL"
            all_ok &= ft_ok
        for variant in variants:
            label = code if variant is None else f"{code} [{variant}]"
            start = time.perf_counter()
            try:
                results = check_circuits(reference, code, variant)
                cells = [fmt(r) for r in results]
                ok = all(bool(r["pass"]) for r in results)
            except Exception as exc:
                cells = [f"ERROR {type(exc).__name__}: {exc}"]
                ok = False
            all_ok &= ok
            elapsed = time.perf_counter() - start
            print(f"{label:<32} " + " ".join(f"{c:<18}" for c in cells) + f" {ft_cell:<6} {elapsed:>6.1f}s", flush=True)
        if not args.no_ft and not ft_ok:
            print(f"    FT failures: {', '.join(ft_failed)}", flush=True)

    print(
        "\nPASS = DAG-equivalent with matching H qubits; 'exact' = identical CNOT order; 'a/b cx' = generated/reference"
    )
    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
