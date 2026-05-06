#!/bin/bash
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

declare -a p=("0.001" "0.0015" "0.002" "0.0025" "0.003" "0.0035" "0.004" "0.0045" "0.005" "0.0055" "0.006" "0.0065" "0.007" "0.0075" "0.008" "0.0085" "0.009" "0.0095" "0.01")

echo "p;p_l;r_a;num_logical_errors;total_shots;p_l_error;r_a_error" > "$1.csv"

run_and_write() {
    local res=$(python estimate_canonical.py ${@:2:$#-2} "-p" "${@: -1}")
    local line="${@: -1};${res}"
    (flock -e 200 echo $line >> "$1.csv") 200>lock
}

export -f run_and_write

parallel --load 16 --link run_and_write $@ ::: ${p[@]}
