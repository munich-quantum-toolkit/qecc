# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Simulate the decoding performance of the UF heuristic decoder."""

from __future__ import annotations

from pathlib import Path

from mqt import qecc

code = qecc.Code(
    "../../examples/lp_(4,8)-[[1024,18,nan]]_hx.txt",
    "../../examples/lp_(4,8)-[[1024,18,nan]]_hz.txt",
)
code.K = 18
outpath = "./dp-sims-bindings.out"

runs_per_p = 1
curr_p = 0.00001
max_per = 0.00003
step_size = 0.00001
nr_failed_runs = 0
code_k = code.K

with Path(outpath).open("w", encoding="utf-8") as outfile:
    while curr_p < max_per:
        nr_failed_runs = 0
        for _ in range(runs_per_p):
            decoder = qecc.UFHeuristic()
            decoder.set_code(code)
            err = qecc.sample_iid_pauli_err(code.N, curr_p)
            decoder.decode(code.get_x_syndrome(err))
            result = decoder.result.estimate
            if not code.is_stabilizer(result):
                nr_failed_runs += 1
        frac_fails = nr_failed_runs / runs_per_p
        wer = frac_fails / code_k
        res = str(curr_p) + ":" + str(wer) + "\n"
        print(res)
        outfile.write(res)
        curr_p += step_size
