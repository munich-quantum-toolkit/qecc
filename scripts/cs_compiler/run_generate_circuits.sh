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

declare -a n_values=("128" "256" "512")
num_circuits=1
export num_circuits

run_and_generate() {
    local n=$1
    python generate_random_circuits.py --n "$n" --num_circuits "$num_circuits"
}

export -f run_and_generate

# Run 3 jobs in parallel (adjust --jobs/-j according to your server)
parallel --jobs 3 run_and_generate ::: ${n_values[@]}
