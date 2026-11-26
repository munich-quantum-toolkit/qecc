# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Estimate the logical error rate and acceptance rate of the circuits in ``index.csv`` by Monte Carlo sampling.

Usage: ``python simulate.py [--p 1e-3] [--min-errors 100] [--max-shots 100000000] [--code color_17_1_5]``
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import stim

from mqt.qecc import CSSCode
from mqt.qecc.circuit_synthesis.ft_search.simulation import logical_error_rate

HERE = Path(__file__).resolve().parent


def load_code(name: str) -> CSSCode:
    """The code of a circuit directory, from ``codes/<name>.json``."""
    c = json.loads((HERE / "codes" / f"{name}.json").read_text(encoding="utf-8"))
    m = {k: np.array(c[k], dtype=np.int8) for k in ("Hx", "Hz", "Lx", "Lz")}
    return CSSCode(Hx=m["Hx"], Hz=m["Hz"], Lx=m["Lx"], Lz=m["Lz"], distance=c["d"])


def main() -> None:
    """Simulate the circuits of the index and print one CSV row per circuit."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p", type=float, default=1e-3, help="physical error rate")
    ap.add_argument("--min-errors", type=int, default=100, help="logical errors to sample per circuit")
    ap.add_argument("--max-shots", type=int, default=10**8, help="shots per circuit at most")
    ap.add_argument("--code", default=None, help="only the circuits of this code")
    a = ap.parse_args()
    rows = list(csv.DictReader((HERE / "index.csv").open(encoding="utf-8")))
    print("code,ops,depth,acceptance,logical_error_rate,errors,shots", flush=True)
    for r in rows:
        if a.code and r["code"] != a.code:
            continue
        code = load_code(r["code"])
        circ = stim.Circuit((HERE / r["file"]).read_text(encoding="utf-8"))
        ler, acc, errors, shots = logical_error_rate(
            circ, code, a.p, zero_state=r["state"] == "zero", min_errors=a.min_errors, max_shots=a.max_shots
        )
        print(f"{r['code']},{r['ops']},{r['depth']},{acc:.4f},{ler:.3e},{errors},{shots}", flush=True)


if __name__ == "__main__":
    main()
