#!/bin/bash
# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

declare -a n_values=("128" "256" "512")
base_dir="circuits_performance_benchmarking"
results_dir="results_performance_benchmarking"
mkdir -p "$results_dir"

export base_dir
export results_dir

run_and_simulate() {
    local n=$1
    local seed=$2

    local qasm_path="${base_dir}/${n}/${n}_${seed}.qasm"
    local csv_path="${results_dir}/results_${n}.csv"

    python simulate_circuit_performance.py \
        --qasm_path "$qasm_path" \
        --n "$n" \
        --seed "$seed" \
        --output_csv "$csv_path"
}

export -f run_and_simulate

# parallelize circuits, not n's
for n in "${n_values[@]}"; do
    seq 0 4 | parallel --jobs 16 run_and_simulate "$n" {}
done
