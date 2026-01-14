#!/bin/bash
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -euo pipefail
# Generate random universal circuits for different system sizes in parallel.

declare -a n_values=()
for ((i=64; i<=1024; i+=64)); do
    n_values+=("$i")
done
declare -a distr_types=("even" "cx_heavy")
num_circuits=100
export num_circuits
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SCRIPT_DIR

run_and_generate() {
    local n=$1
    local distr_type=$2
    python "${SCRIPT_DIR}/generate_random_circuits.py" --n "$n" --num_circuits "$num_circuits" --distr_type "$distr_type"
}

export -f run_and_generate

# Run 3 jobs in parallel (adjust --jobs/-j according to your server)
parallel --jobs 3 run_and_generate ::: "${n_values[@]}" ::: "${distr_types[@]}"
