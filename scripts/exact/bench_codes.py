#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

r"""Benchmark exact synthesis on standard quantum error-correcting codes.

For each code two circuits are produced:
  - gate-count-optimal encoding circuit
  - depth-optimal encoding circuit, then TQ-count minimised at that depth

Non-CSS codes (CLIFFORD_ISOMETRY / STABILIZER_STATE) use the extended gate set
{H, S, SX, CX, CZ}.  CSS codes use the standard {CX} set.
Symmetry breaking and exponential-backoff search are always enabled.

Circuits and metrics (gate count, depth, two-qubit gate count, optimality,
total search time) are written as JSONL records — one record per
(code, objective) pair — to stdout and optionally to a file.

Usage:
    python scripts/exact/bench_codes.py [options]

Options:
    --timeout     Per-bound SAT-solver timeout in seconds (default: 86400 = 24 h)
    --codes       Comma-separated list of code names to benchmark (default: all)
    --output      Write JSONL records to FILE in addition to stdout
    --nprocesses  Number of parallel worker processes (default: 1)

Parallel execution (all codes at once):
    python scripts/exact/bench_codes.py --timeout 86400 --nprocesses 8 \
        --output results.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import numpy as np

from mqt.qecc.circuit_synthesis import heuristic_prep_circuit, synthesize_encoding_circuit
from mqt.qecc.circuit_synthesis.circuits import CNOTCircuit
from mqt.qecc.circuit_synthesis.exact.gate_operations import get_clifford_extended_gate_set
from mqt.qecc.circuit_synthesis.exact.search import synthesize_exact
from mqt.qecc.circuit_synthesis.exact.types import Objective, SynthesisStatus, TargetKind
from mqt.qecc.codes import CSSCode, StabilizerCode
from mqt.qecc.codes.pauli import CheckMatrix, StabilizerTableau

if TYPE_CHECKING:
    from mqt.qecc.circuit_synthesis.circuits import CliffordIsometry
    from mqt.qecc.circuit_synthesis.exact.types import SynthesisResult

DEFAULT_TIMEOUT = 86_400  # 24 hours
DEFAULT_NPROCESSES = 1

_CSS_KINDS = {TargetKind.CSS_ISOMETRY, TargetKind.CSS_STATE}


# ---------------------------------------------------------------------------
# Code definitions
# ---------------------------------------------------------------------------


def _build_codes() -> list[dict[str, Any]]:
    """Return the list of benchmark codes with their synthesis parameters."""
    steane = CSSCode.from_code_name("Steane")
    surface = CSSCode.from_code_name("Surface", distance=3)
    hamming = CSSCode.from_code_name("Hamming")
    tetrahedral = CSSCode.from_code_name("Tetrahedral")
    carbon = CSSCode.from_code_name("Carbon")
    css_832 = CSSCode(
        Hx=np.array([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.int8),
        Hz=np.array(
            [
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 1, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 1, 0, 1, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
            ],
            dtype=np.int8,
        ),
        distance=2,
    )
    vasmer_kubica = CSSCode(
        Hx=np.array(
            [
                [1, 1, 1, 1, 0, 0, 0, 1, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
                [0, 0, 1, 1, 0, 1, 1, 0, 0, 1],
            ],
            dtype=np.int8,
        ),
        Hz=np.array(
            [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.int8,
        ),
        distance=2,
    )

    iceberg_codes = [
        CSSCode(Hx=np.ones((1, 2 * n), dtype=np.int8), Hz=np.ones((1, 2 * n), dtype=np.int8), distance=2)
        for n in (2, 3, 4)
    ]

    css_codes = [
        ("steane", "[[7,1,3]] Steane", steane),
        ("surface", "[[9,1,3]] Surface (d=3)", surface),
        ("hamming", "[[15,7,3]] Hamming", hamming),
        ("tetrahedral", "[[15,1,3]] Tetrahedral", tetrahedral),
        ("carbon", "[[12,2,4]] Carbon", carbon),
        ("css_832", "[[8,3,2]] CSS", css_832),
        ("vasmer_kubica", "[[10,1,2]] Vasmer-Kubica", vasmer_kubica),
        ("iceberg_422", "[[4,2,2]] Iceberg", iceberg_codes[0]),
        ("iceberg_642", "[[6,4,2]] Iceberg", iceberg_codes[1]),
        ("iceberg_862", "[[8,6,2]] Iceberg", iceberg_codes[2]),
    ]

    codes: list[dict[str, Any]] = []

    for name, display, css in css_codes:
        checks = CheckMatrix(css.Hx, pauli_type="X")
        x_logicals = CheckMatrix(css.Lx, pauli_type="X")
        codes.append({
            "name": name,
            "display": display,
            "target_kind": TargetKind.CSS_ISOMETRY,
            "target": checks,
            "x_logicals": x_logicals,
            "z_logicals": None,
            "stabilizer_code": css,
        })

    # [[5,1,3]] five-qubit code
    five_qubit_stabs = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    five_qubit_xl = ["XXXXX"]
    five_qubit_zl = ["ZZZZZ"]
    five_qubit = StabilizerCode(five_qubit_stabs, x_logicals=five_qubit_xl, z_logicals=five_qubit_zl)
    codes.append({
        "name": "five_qubit",
        "display": "[[5,1,3]] Five-qubit",
        "target_kind": TargetKind.CLIFFORD_ISOMETRY,
        "target": StabilizerTableau.from_pauli_strings(five_qubit_stabs),
        "x_logicals": StabilizerTableau.from_pauli_strings(five_qubit_xl),
        "z_logicals": StabilizerTableau.from_pauli_strings(five_qubit_zl),
        "stabilizer_code": five_qubit,
    })

    # [[8,3,3]] Gottesman code  (arXiv:quant-ph/9705052)
    gottesman_8_stabs = ["XXXXXXXX", "ZZZZZZZZ", "IXIXYZYZ", "IXZYIXZY", "IYXZXZIY"]
    gottesman_8_xl = ["XXIIIZIZ", "XIXZIIZI", "XIIZXZII"]
    gottesman_8_zl = ["IZIZIZIZ", "IIZZIIZZ", "IIIIZZZZ"]
    gottesman_8 = StabilizerCode(
        gottesman_8_stabs,
        x_logicals=gottesman_8_xl,
        z_logicals=gottesman_8_zl,
    )
    codes.append({
        "name": "gottesman_8",
        "display": "[[8,3,3]] Gottesman",
        "target_kind": TargetKind.CLIFFORD_ISOMETRY,
        "target": StabilizerTableau.from_pauli_strings(gottesman_8_stabs),
        "x_logicals": StabilizerTableau.from_pauli_strings(gottesman_8_xl),
        "z_logicals": StabilizerTableau.from_pauli_strings(gottesman_8_zl),
        "stabilizer_code": gottesman_8,
    })

    # State preparation
    for name, display, css in css_codes:
        for suffix, label, pauli_type, matrix in [
            ("_zero", "|0⟩_L", "X", css.Hx),
            ("_plus", "|+⟩_L", "Z", css.Hz),
        ]:
            codes.append({
                "name": f"{name}{suffix}",
                "display": f"{display} {label}",
                "target_kind": TargetKind.CSS_STATE,
                "target": CheckMatrix(matrix, pauli_type=pauli_type),
                "x_logicals": None,
                "z_logicals": None,
                "stabilizer_code": css,
                "zero_state": suffix == "_zero",
            })

    for suffix, label, logicals in [
        ("_zero", "|0⟩_L", ["ZZZZZ"]),
        ("_plus", "|+⟩_L", ["XXXXX"]),
    ]:
        codes.append({
            "name": f"five_qubit{suffix}",
            "display": f"[[5,1,3]] Five-qubit {label}",
            "target_kind": TargetKind.STABILIZER_STATE,
            "target": StabilizerTableau.from_pauli_strings(five_qubit_stabs + logicals),
            "x_logicals": None,
            "z_logicals": None,
            "stabilizer_code": five_qubit,
        })

    for suffix, label, logicals in [
        ("_zero", "|0⟩_L", gottesman_8_zl),
        ("_plus", "|+⟩_L", gottesman_8_xl),
    ]:
        codes.append({
            "name": f"gottesman_8{suffix}",
            "display": f"[[8,3,3]] Gottesman {label}",
            "target_kind": TargetKind.STABILIZER_STATE,
            "target": StabilizerTableau.from_pauli_strings(gottesman_8_stabs + logicals),
            "x_logicals": None,
            "z_logicals": None,
            "stabilizer_code": gottesman_8,
        })

    return codes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _heuristic_bounds(code_spec: dict[str, Any]) -> tuple[int, int]:
    """Return (gate_count_bound, depth_bound) from heuristic synthesis."""
    target_kind = code_spec["target_kind"]
    sc = code_spec["stabilizer_code"]

    if target_kind == TargetKind.CSS_STATE:
        zero_state = code_spec["zero_state"]
        gc_circ = heuristic_prep_circuit(sc, optimize_depth=False, zero_state=zero_state)
        d_circ = heuristic_prep_circuit(sc, optimize_depth=True, zero_state=zero_state)
        return gc_circ.circ.num_cnots(), d_circ.circ.depth()

    if target_kind == TargetKind.STABILIZER_STATE:
        n = sc.n
        return n * (n - 1), 2 * n

    heuristic = synthesize_encoding_circuit(sc)
    if isinstance(heuristic, CNOTCircuit):
        return heuristic.num_cnots(), heuristic.depth()
    # CliffordIsometry: count H/S/CX from heuristic (standard gate set)
    count = 0
    for inst in heuristic.to_stim_circuit():
        if inst.name in {"H", "S", "S_DAG"}:
            count += len(inst.targets_copy())
        elif inst.name == "CX":
            count += len(inst.targets_copy()) // 2
    return count, heuristic.depth()


def _count_two_qubit_gates(circuit: CliffordIsometry | CNOTCircuit) -> int:
    """Count two-qubit (CX, CZ) gates in a synthesized circuit."""
    count = 0
    for inst in circuit.to_stim_circuit():
        if inst.name in {"CX", "CZ"}:
            count += len(inst.targets_copy()) // 2
    return count


def _circuit_to_str(circuit: CliffordIsometry | CNOTCircuit) -> str:
    """Serialize a circuit to a Stim circuit string."""
    return str(circuit.to_stim_circuit())


def _make_record(
    code_spec: dict[str, Any],
    objective: Objective,
    gate_set_name: str,
    heuristic_bound: int,
    result: SynthesisResult,
    runtime: float,
    tq_proven_optimal: bool | None = None,
) -> dict[str, Any]:
    circuit = result.circuit if result.status == SynthesisStatus.SUCCESS else None
    tq = _count_two_qubit_gates(circuit) if circuit is not None else None
    return {
        "code": code_spec["name"],
        "display": code_spec["display"],
        "objective": objective.value,
        "gate_set": gate_set_name,
        "heuristic_bound": heuristic_bound,
        "gate_count": result.gate_count,
        "depth": result.depth,
        "two_qubit_gates": tq,
        "proven_optimal": result.proven_optimal,
        "tq_proven_optimal": tq_proven_optimal,
        "status": result.status.value,
        "runtime": round(runtime, 3),
        "circuit": _circuit_to_str(circuit) if circuit is not None else None,
    }


# ---------------------------------------------------------------------------
# Per-code synthesis worker
# ---------------------------------------------------------------------------


def _synthesize_one_code(code_name: str, timeout: int) -> list[dict[str, Any]]:
    """Synthesize gate-count-optimal and depth-optimal circuits for one code.

    Designed to run in a worker process. Rebuilds the code spec internally so
    that no non-picklable objects are passed across process boundaries.

    Args:
        code_name: Name key from _build_codes().
        timeout: Per-bound SAT-solver timeout in seconds.

    Returns:
        List of two record dicts: one for gate-count, one for depth objective.
    """
    all_codes = {c["name"]: c for c in _build_codes()}
    code_spec = all_codes[code_name]

    is_clifford = code_spec["target_kind"] not in _CSS_KINDS
    gate_set = get_clifford_extended_gate_set() if is_clifford else None
    gate_set_name = "extended" if is_clifford else "standard_css"

    gc_bound, d_bound = _heuristic_bounds(code_spec)

    common_kwargs: dict[str, Any] = {
        "target": code_spec["target"],
        "target_kind": code_spec["target_kind"],
        "x_logicals": code_spec["x_logicals"],
        "z_logicals": code_spec["z_logicals"],
        "use_symmetry_breaking": True,
        "gate_set": gate_set,
        "use_exponential_backoff": True,
        "min_timeout": 1,
        "timeout": timeout,
        "verify": True,
    }

    records: list[dict[str, Any]] = []

    # --- Gate-count optimization ---
    t0 = time.monotonic()
    gc_result = synthesize_exact(
        **common_kwargs,
        objective=Objective.GATE_COUNT,
        lower_bound=0,
        upper_bound=gc_bound,
    )
    gc_time = time.monotonic() - t0
    records.append(_make_record(code_spec, Objective.GATE_COUNT, gate_set_name, gc_bound, gc_result, gc_time))

    # --- Depth optimization ---
    t0 = time.monotonic()
    d_result = synthesize_exact(
        **common_kwargs,
        objective=Objective.DEPTH,
        lower_bound=0,
        upper_bound=d_bound,
    )
    d_time = time.monotonic() - t0

    if d_result.status != SynthesisStatus.SUCCESS or d_result.depth is None:
        records.append(_make_record(code_spec, Objective.DEPTH, gate_set_name, d_bound, d_result, d_time))
        return records

    # --- Secondary phase: minimize TQ count at fixed optimal depth ---
    d_star = d_result.depth
    best_d_result = d_result
    tq_count = _count_two_qubit_gates(d_result.circuit)
    tq_proven_optimal = False

    for max_tq in range(tq_count - 1, -1, -1):
        t_step = time.monotonic()
        tq_result = synthesize_exact(
            target=code_spec["target"],
            target_kind=code_spec["target_kind"],
            x_logicals=code_spec["x_logicals"],
            z_logicals=code_spec["z_logicals"],
            objective=Objective.DEPTH,
            lower_bound=d_star,
            upper_bound=d_star,
            use_symmetry_breaking=True,
            gate_set=gate_set,
            max_two_qubit_gates=max_tq,
            use_exponential_backoff=False,
            timeout=timeout,
            verify=True,
        )
        d_time += time.monotonic() - t_step

        if tq_result.status == SynthesisStatus.SUCCESS:
            best_d_result = tq_result
        elif tq_result.status == SynthesisStatus.UNSAT:
            tq_proven_optimal = True
            break
        else:
            break

    records.append(
        _make_record(
            code_spec,
            Objective.DEPTH,
            gate_set_name,
            d_bound,
            best_d_result,
            d_time,
            tq_proven_optimal=tq_proven_optimal,
        )
    )
    return records


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------


def print_table(records: list[dict[str, Any]]) -> None:
    """Print a human-readable summary table."""
    hdr = (
        f"{'Code':<24}  {'Obj':<12}  {'GateSet':<12}  "
        f"{'Heuristic':>9}  {'Gates':>6}  {'TQ':>4}  {'Depth':>5}  "
        f"{'Opt':>3}  {'TQ_Opt':>6}  {'Time(s)':>8}"
    )
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    for r in records:
        gc = str(r["gate_count"]) if r["gate_count"] is not None else "—"
        tq = str(r["two_qubit_gates"]) if r["two_qubit_gates"] is not None else "—"
        d = str(r["depth"]) if r["depth"] is not None else "—"
        opt = "yes" if r["proven_optimal"] else ("?" if r["status"] == "success" else "—")
        tq_opt_val = r.get("tq_proven_optimal")
        tq_opt = "yes" if tq_opt_val else ("?" if tq_opt_val is False and r["status"] == "success" else "—")
        print(
            f"{r['display']:<24}  {r['objective']:<12}  {r['gate_set']:<12}  "
            f"{r['heuristic_bound']:>9}  {gc:>6}  {tq:>4}  {d:>5}  "
            f"{opt:>3}  {tq_opt:>6}  {r['runtime']:>8.1f}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-bound SAT-solver timeout in seconds (default: %(default)s = 24 h)",
    )
    parser.add_argument(
        "--codes",
        default=None,
        help="Comma-separated list of code names to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write JSONL records to this file (in addition to stdout)",
    )
    parser.add_argument(
        "--nprocesses",
        type=int,
        default=DEFAULT_NPROCESSES,
        dest="nprocesses",
        help="Number of parallel worker processes (default: %(default)s)",
    )
    args = parser.parse_args()

    all_codes = _build_codes()

    if args.codes:
        wanted = {c.strip() for c in args.codes.split(",")}
        all_codes = [c for c in all_codes if c["name"] in wanted]
        if not all_codes:
            print(f"No matching codes for: {args.codes}", file=sys.stderr)
            sys.exit(1)

    code_names = [c["name"] for c in all_codes]

    out_path = pathlib.Path(args.output) if args.output else None
    out_file = out_path.open("w", encoding="utf-8") if out_path else None

    all_records: list[dict[str, Any]] = []

    try:
        with ProcessPoolExecutor(max_workers=args.nprocesses) as executor:
            futures = {executor.submit(_synthesize_one_code, name, args.timeout): name for name in code_names}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    recs = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"ERROR [{name}]: {exc}", file=sys.stderr)
                    continue

                for rec in recs:
                    line = json.dumps(rec)
                    print(line, flush=True)
                    if out_file is not None:
                        out_file.write(line + "\n")
                        out_file.flush()

                all_records.extend(recs)
                print(
                    f"  done [{name}]  "
                    f"gc={recs[0]['gate_count']}  depth={recs[1]['depth'] if len(recs) > 1 else '?'}  "
                    f"tq={recs[1]['two_qubit_gates'] if len(recs) > 1 else '?'}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        if out_file is not None:
            out_file.close()

    print("\n", file=sys.stderr)
    print_table(all_records)


if __name__ == "__main__":
    main()
