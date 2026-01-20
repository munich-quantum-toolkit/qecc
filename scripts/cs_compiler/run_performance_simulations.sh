#!/bin/bash
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# declare -a n_values=("64" "128" "256" "512")
declare -a n_values=()
for ((i=64; i<=1024; i+=64)); do
    n_values+=("$i")
done
declare -a distr_types=("even" "cx_heavy")
base_dir="circuits_performance_benchmarking"
results_dir="results_performance_benchmarking"
mkdir -p "$results_dir"

export base_dir
export results_dir

run_and_simulate() {
    local n=$1
    local distr_type=$2
    local seed=$3

    local qasm_path="${base_dir}/${distr_type}/${n}/${n}_${seed}.qasm"
    local csv_path="${results_dir}/${distr_type}/results_${n}.csv"

    python simulate_circuit_performance.py \
        --qasm_path "$qasm_path" \
        --n "$n" \
        --seed "$seed" \
        --output_csv "$csv_path" \
        --distr_type "$distr_type"
}

export -f run_and_simulate

# parallelize circuits, not n's
parallel --jobs 5 run_and_simulate ::: "${n_values[@]}" ::: "${distr_types[@]}" ::: $(seq 0 99)
