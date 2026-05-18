#!/usr/bin/env python3
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

r"""Benchmark exact synthesis on random Clifford circuits.

Each invocation handles one or more (n, seed) pairs and emits JSONL records to
stdout — one record per (kind x objective x sym_break) combination.

Run a single job (GNU-parallel compatible):
    python scripts/exact/bench_random.py --n 3 --seed 42

Run 10 samples for a single qubit count:
    python scripts/exact/bench_random.py --n 3 --num-samples 10

Run a full sweep over multiple qubit counts:
    python scripts/exact/bench_random.py --qubit-sizes 2 3 4 5 --num-samples 10

Run with exponential-backoff search (records tagged search=backoff):
    python scripts/exact/bench_random.py --qubit-sizes 2 3 4 --num-samples 10 \
        --exponential-backoff --min-timeout 1

Parallelise individual (n, seed) pairs with GNU parallel:
    parallel -j8 \\
        "python scripts/exact/bench_random.py --n {1} --seed {2}" \\
        ::: 2 3 4 5 ::: {0..9} >> results.jsonl 2>errors.log

Aggregate a completed JSONL file (supports mixed fixed/backoff records):
    python scripts/exact/bench_random.py --aggregate results.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
from qiskit.quantum_info import random_clifford

from mqt.qecc.circuit_synthesis.exact.gate_operations import (
    get_clifford_extended_gate_set,
    get_standard_clifford_gate_set,
)
from mqt.qecc.circuit_synthesis.exact.search import synthesize_exact
from mqt.qecc.circuit_synthesis.exact.types import Objective, TargetKind
from mqt.qecc.codes.pauli import StabilizerTableau

_OBJECTIVES = [Objective.GATE_COUNT, Objective.DEPTH]
_SYM_BREAKS = [False, True]

# Qubit sizes to benchmark (when --n is not specified, used only in aggregate mode)
DEFAULT_QUBIT_SIZES = [2, 3, 4, 5]
DEFAULT_NUM_SAMPLES = 10
DEFAULT_TIMEOUT = 3600  # 1 hour per SAT-solver call
DEFAULT_UPPER_BOUND = 50

_GATE_SETS = {
    "standard": get_standard_clifford_gate_set,  # {H, S, CX, ID}
    "extended": get_clifford_extended_gate_set,  # {H, S, SX, CX, CZ, ID}
}


def make_targets(
    n: int, seed: int
) -> tuple[StabilizerTableau, StabilizerTableau, StabilizerTableau, StabilizerTableau]:
    """Build synthesis targets from a uniformly random n-qubit Clifford.

    Uses ``qiskit.quantum_info.random_clifford`` which samples uniformly from
    the Clifford group via the Bravyi-Maslov algorithm (arXiv:2003.09412).

    The same Clifford is used for both target kinds:

    - Unitary:  full Clifford map (X- and Z-images as logical operators,
                no stabilizer rows → CLIFFORD_UNITARY target)
    - State:    stabilizer state prepared by applying the Clifford to |0…0⟩
                (Z-images only; X-images are the destabilizers and are dropped
                → STABILIZER_STATE target)

    The qiskit ``Clifford`` tableau stores:
    - ``destab_x/z``:  X- and Z-parts of U(X_i) per qubit  → our x_logicals
    - ``stab_x/z``:    X- and Z-parts of U(Z_i) per qubit  → our z_logicals (= state stabilizers)
    - phases:          False = +1, True = -1

    Args:
        n: Number of qubits.
        seed: RNG seed for reproducibility.

    Returns:
        (stabs_empty, x_logicals, z_logicals, state_stabs)
    """
    cliff = random_clifford(n, seed=seed)

    x_log_matrix = np.hstack([cliff.destab_x.astype(np.int8), cliff.destab_z.astype(np.int8)])
    x_log_phase = cliff.destab_phase.astype(np.int8)

    z_log_matrix = np.hstack([cliff.stab_x.astype(np.int8), cliff.stab_z.astype(np.int8)])
    z_log_phase = cliff.stab_phase.astype(np.int8)

    x_logicals = StabilizerTableau(x_log_matrix, x_log_phase)
    z_logicals = StabilizerTableau(z_log_matrix, z_log_phase)
    stabs_empty = StabilizerTableau.empty(n)
    state_stabs = StabilizerTableau(z_log_matrix.copy(), z_log_phase.copy())

    return stabs_empty, x_logicals, z_logicals, state_stabs


_SINGLE_QUBIT_GATES = {"H", "S", "S_DAG", "SQRT_X", "SQRT_X_DAG"}
_TWO_QUBIT_GATES = {"CX", "CZ"}
_NON_PAULI = _SINGLE_QUBIT_GATES | _TWO_QUBIT_GATES


def _gate_count_from_circuit(result: object) -> int | None:
    """Count non-Pauli Clifford gates in the synthesized circuit."""
    circuit = getattr(result, "circuit", None)
    if circuit is None:
        return None
    count = 0
    for inst in circuit.to_stim_circuit(with_resets=False):
        if inst.name in _SINGLE_QUBIT_GATES:
            count += len(inst.targets_copy())
        elif inst.name in _TWO_QUBIT_GATES:
            count += len(inst.targets_copy()) // 2
    return count


def _depth_from_circuit(result: object) -> int | None:
    """Compute ASAP-scheduled depth of non-Pauli Clifford gates."""
    circuit = getattr(result, "circuit", None)
    if circuit is None:
        return None
    qubit_layer: dict[int, int] = {}
    for inst in circuit.to_stim_circuit(with_resets=False):
        if inst.name not in _NON_PAULI:
            continue
        for grp in inst.target_groups():
            qubits = [t.qubit_value for t in grp]
            layer = max((qubit_layer.get(q, 0) for q in qubits), default=0) + 1
            for q in qubits:
                qubit_layer[q] = layer
    return max(qubit_layer.values(), default=0)


def run_one(
    n: int,
    seed: int,
    kind: str,
    target: StabilizerTableau,
    x_logicals: StabilizerTableau | None,
    z_logicals: StabilizerTableau | None,
    objective: Objective,
    sym_break: bool,
    timeout: int,
    upper_bound: int,
    gate_set_name: str = "standard",
    use_exponential_backoff: bool = False,
    min_timeout: int = 1,
) -> dict:
    """Run a single synthesis call and return a result record."""
    target_kind = TargetKind.CLIFFORD_UNITARY if kind == "unitary" else TargetKind.STABILIZER_STATE

    t0 = time.monotonic()
    result = synthesize_exact(
        target=target,
        target_kind=target_kind,
        objective=objective,
        x_logicals=x_logicals,
        z_logicals=z_logicals,
        lower_bound=0,
        upper_bound=upper_bound,
        use_symmetry_breaking=sym_break,
        timeout=timeout,
        verify=True,
        gate_set=_GATE_SETS[gate_set_name](),
        use_exponential_backoff=use_exponential_backoff,
        min_timeout=min_timeout,
    )
    elapsed = time.monotonic() - t0

    return {
        "n": n,
        "seed": seed,
        "kind": kind,
        "objective": objective.value,
        "sym_break": sym_break,
        "gate_set": gate_set_name,
        "search": "backoff" if use_exponential_backoff else "fixed",
        "status": result.status.value,
        "gate_count": _gate_count_from_circuit(result),
        "depth": _depth_from_circuit(result),
        "proven_optimal": result.proven_optimal,
        "verified": result.verified,
        "runtime": round(elapsed, 3),
    }


def run_one_pair(
    n: int,
    seed: int,
    timeout: int,
    upper_bound: int,
    gate_set_name: str = "standard",
    use_exponential_backoff: bool = False,
    min_timeout: int = 1,
) -> None:
    """Run all synthesis combinations for one (n, seed) pair, printing JSONL."""
    stabs_empty, x_logicals, z_logicals, state_stabs = make_targets(n, seed)

    combinations = [
        ("unitary", stabs_empty, x_logicals, z_logicals),
        ("state", state_stabs, None, None),
    ]

    for kind, target, xl, zl in combinations:
        for obj in _OBJECTIVES:
            for sym_break in _SYM_BREAKS:
                record = run_one(
                    n=n,
                    seed=seed,
                    kind=kind,
                    target=target,
                    x_logicals=xl,
                    z_logicals=zl,
                    objective=obj,
                    sym_break=sym_break,
                    timeout=timeout,
                    upper_bound=upper_bound,
                    gate_set_name=gate_set_name,
                    use_exponential_backoff=use_exponential_backoff,
                    min_timeout=min_timeout,
                )
                print(json.dumps(record), flush=True)


def run_job(args: argparse.Namespace) -> None:
    """Run all synthesis combinations for the (n, seed) pairs implied by args."""
    use_eb = args.exponential_backoff
    min_to = args.min_timeout
    gs_name = args.gate_set

    if args.seed is not None:
        # Single explicit (n, seed) pair — GNU-parallel mode.
        run_one_pair(args.n, args.seed, args.timeout, args.upper_bound, gs_name, use_eb, min_to)
        return

    qubit_sizes = args.qubit_sizes or [args.n]
    num_samples = args.num_samples

    for n in qubit_sizes:
        for seed in range(num_samples):
            print(f"=== n={n} seed={seed} ===", file=sys.stderr, flush=True)
            run_one_pair(n, seed, args.timeout, args.upper_bound, gs_name, use_eb, min_to)


def _print_aggregate_rows(
    groups: dict[tuple, list[dict]],
    n: int,
    kind: str,
    obj: str,
    sym_break: bool,
    gate_set_names: list[str],
    search_strategies: list[str],
) -> None:
    for gs in gate_set_names:
        for search in search_strategies:
            _print_aggregate_row(groups, n, kind, obj, sym_break, gs, search)


def _print_aggregate_row(
    groups: dict[tuple, list[dict]],
    n: int,
    kind: str,
    obj: str,
    sym_break: bool,
    gs: str,
    search: str,
) -> None:
    key = (n, kind, obj, sym_break, gs, search)
    if key not in groups:
        return
    recs = groups[key]
    successes = [r for r in recs if r["status"] == "success"]
    ok_str = f"{len(successes)}/{len(recs)}"
    n_optimal = sum(1 for r in successes if r.get("proven_optimal"))
    opt_str = f"{n_optimal}/{len(successes)}" if successes else "—"
    avg_rt = sum(r["runtime"] for r in recs) / len(recs)
    if obj == "gate_count":
        vals = [r["gate_count"] for r in successes if r["gate_count"] is not None]
        avg_gates = f"{sum(vals) / len(vals):.2f}" if vals else "—"
        avg_depth = "—"
    else:
        vals = [r["depth"] for r in successes if r["depth"] is not None]
        avg_depth = f"{sum(vals) / len(vals):.2f}" if vals else "—"
        avg_gates = "—"
    sym_str = "yes" if sym_break else "no"
    print(
        f"{n:>2}  {kind:8}  {obj:12}  {sym_str:4}  {gs:8}  {search:7}  "
        f"{ok_str:>5}  {opt_str:>5}  {avg_rt:>10.2f}  {avg_gates:>8}  {avg_depth:>7}"
    )


def aggregate(path: str) -> None:
    """Read a JSONL file and print a summary table.

    Supports mixed files containing both ``search=fixed`` (legacy) and
    ``search=backoff`` records.  Groups are shown sorted by search strategy so
    the two can be compared side-by-side.
    """
    records = []
    with pathlib.Path(path).open(encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if stripped:
                records.append(json.loads(stripped))

    if not records:
        print("No records found.")
        return

    # Normalise old records that pre-date the search / gate_set fields.
    for r in records:
        r.setdefault("search", "fixed")
        r.setdefault("gate_set", "standard")
        r.setdefault("proven_optimal", False)

    # Group by (n, kind, objective, sym_break, gate_set, search)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = (r["n"], r["kind"], r["objective"], r["sym_break"], r["gate_set"], r["search"])
        groups[key].append(r)

    gate_set_names = sorted({r["gate_set"] for r in records})
    search_strategies = sorted({r["search"] for r in records})

    # Print table
    header = (
        f"{'n':>2}  {'kind':8}  {'obj':12}  {'sym':4}  {'gate_set':8}  {'search':7}  "
        f"{'ok':>5}  {'opt':>5}  {'runtime':>10}  {'gates':>8}  {'depth':>7}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for n in sorted({r["n"] for r in records}):
        for kind in ("unitary", "state"):
            for obj in ("gate_count", "depth"):
                for sym_break in (False, True):
                    _print_aggregate_rows(groups, n, kind, obj, sym_break, gate_set_names, search_strategies)
        print()  # blank line between n groups

    print(sep)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--n", type=int, help="Number of qubits (use with --seed or --num-samples)")
    mode.add_argument(
        "--qubit-sizes",
        type=int,
        nargs="+",
        metavar="N",
        help="Space-separated list of qubit counts to sweep (e.g. --qubit-sizes 2 3 4 5)",
    )
    mode.add_argument("--aggregate", metavar="FILE", help="Aggregate JSONL results and print table")

    parser.add_argument(
        "--gate-set",
        choices=list(_GATE_SETS),
        default="standard",
        dest="gate_set",
        help=("Gate set to use for synthesis (default: standard = {H, S, CX}; extended = {H, S, SX, CX, CZ})"),
    )
    parser.add_argument("--seed", type=int, default=None, help="Fixed RNG seed (single-job mode; requires --n)")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of random samples per qubit count (default: %(default)s); seeds 0..N-1",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-bound SAT-solver timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--upper-bound",
        type=int,
        default=DEFAULT_UPPER_BOUND,
        help="Upper bound on gate count / depth to search (default: %(default)s)",
    )
    parser.add_argument(
        "--exponential-backoff",
        action="store_true",
        default=False,
        help=(
            "Use exponential-backoff search instead of the default fixed-timeout scan. "
            "Starts at --min-timeout seconds per bound and doubles after each pass, "
            "up to --timeout. Records are tagged search=backoff."
        ),
    )
    parser.add_argument(
        "--min-timeout",
        type=int,
        default=1,
        dest="min_timeout",
        help="Starting per-bound timeout in seconds for exponential-backoff mode (default: %(default)s)",
    )

    args = parser.parse_args()

    if args.aggregate:
        aggregate(args.aggregate)
        return

    if args.seed is not None and args.n is None:
        parser.error("--seed requires --n")

    run_job(args)


if __name__ == "__main__":
    main()
