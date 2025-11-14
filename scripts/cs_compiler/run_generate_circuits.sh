#!/bin/bash
# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright ...
# SPDX-License-Identifier: MIT
#
# Generate random universal circuits for different system sizes in parallel.

declare -a n_values=("64" "128" "256" "512")
declare -a distr_types=("even" "ht_heavy" "cx_heavy")
num_circuits=10
export num_circuits

run_and_generate() {
    local n=$1
    local distr_type=$2
    python generate_random_circuits.py --n "$n" --num_circuits "$num_circuits" --distr_type "$distr_type"
}

export -f run_and_generate

# Run 3 jobs in parallel (adjust --jobs/-j according to your server)
parallel --jobs 3 run_and_generate ::: "${n_values[@]}" ::: "${distr_types[@]}"
