# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Generates circuits for benchmark runs."""

import json
import pathlib

from mqt.qecc.cococo import circuit_construction, dag_helper

q = 120
j = 8
d = 320
num_gates = j * d
reps = 20

circuits = []
for _ in range(reps):
    dag, pairs = circuit_construction.create_random_sequential_circuit_dag(j, q, num_gates)
    circuits.append(pairs)

    # check whether the number of layers you want
    layers = dag_helper.count_cx_gates_per_layer(dag)
    print("Layers circuit", layers)
    print("Number of layers", len(layers))


path = f"true_seq_circs_j{j}_q{q}_numgates{num_gates}d{d}_x{reps}.json"

with pathlib.Path(path).open("w", encoding="utf-8") as f:
    json.dump(circuits, f)
