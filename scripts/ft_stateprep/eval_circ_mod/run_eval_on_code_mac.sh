#!/bin/bash
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Description: Estimate logical error rates for given code in parallel. Results are saved in a csv file described by the first argument. All other arguments are passed to the python script.

# declare -a p=("0.01" "0.009" "0.007" "0.005" "0.003" "0.001" "0.0009" "0.0007")
# declare -a p=("0.003" "0.0025" "0.0035" "0.002" "0.004" "0.0015" "0.0045" "0.001" "0.005" "0.0055" "0.006" "0.0065" "0.007" "0.0075" "0.008" "0.0085" "0.009" "0.0095" "0.01")
declare -a p=("0.001")

echo "p p_l acceptance errors runs" > "$1.csv"

run_and_write() {
    local res=$(python estimate_logical_error_rate.py ${@:2:$#-2} "-p" "${@: -1}")
    local line="${@: -1} ${res}"
    (flock -e 200 echo $line >> "$1.csv") 200>lock
}

export -f run_and_write

parallel -j 6 --link run_and_write $@ ::: ${p[@]}
