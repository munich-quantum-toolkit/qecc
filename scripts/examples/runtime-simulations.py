# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compute average runtimes for the UF heuristic on toric codes."""

from __future__ import annotations

from pathlib import Path

from mqt import qecc

codes = [
    "../../examples/toricCodes/toric_(nan,nan)-[[8,2,2]]_hz.txt",
    "../../examples/toricCodes/toric_(nan,nan)-[[18,2,3]]_hz.txt",
    "../../examples/toricCodes/toric_(nan,nan)-[[32,2,4]]_hz.txt",
    "../../examples/toricCodes/toric_(nan,nan)-[[50,2,5]]_hz.txt",
    "../../examples/toricCodes/toric_(nan,nan)-[[72,2,6]]_hz.txt",
    "../../examples/toricCodes/toric_(nan,nan)-[[98,2,7]]_hz.txt",
    "../../examples/toricCodes/toric_(nan,nan)-[[128,2,8]]_hz.txt",
    "../../examples/toricCodes/toric_(nan,nan)-[[162,2,9]]_hz.txt",
    "../../examples/toricCodes/toric_(nan,nan)-[[200,2,10]]_hz.txt",
]
outpath: str = "./rt-sims-bindings.out"

nr_samples: int = 1
nr_runs: int = 1
per: float = 0.01

with Path(outpath).open("w", encoding="utf-8") as outfile:
    for code_path in codes:
        sample_sum = 0.0
        for _ in range(nr_samples):
            runs_sum = 0.0
            for _ in range(nr_runs):
                code = qecc.Code(code_path)
                err = qecc.sample_iid_pauli_err(code.N, per)
                decoder = qecc.UFHeuristic()
                decoder.set_code(code)
                syndr = code.get_x_syndrome(err)
                decoder.decode(syndr)
                time = decoder.result.decoding_time
                runs_sum += time
            sample_sum += runs_sum
        outp = code_path + ":" + str(sample_sum / nr_samples)
        print(outp)
        outfile.write(outp + "\n")
        outfile.flush()
