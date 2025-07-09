#!/bin/bash
# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Description: Estimate logical error rates for given code in parallel. Results are saved in a csv file described by the first argument. All other arguments are passed to the python script.

declare -a p=("0.00005" "0.00006" "0.00007" "0.00008" "0.00009" "0.0001" "0.0002" "0.0003" "0.0004" "0.0005" "0.0006" "0.0007" "0.0008" "0.0009" "0.001" "0.002" "0.003" "0.004" "0.005" "0.006" "0.007" "0.008" "0.009" "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.2" "0.3" "0.4" "0.5")

echo "p p_l acceptance errors runs" > "$1.csv"

run_and_write() {
    local res=$(python estimate_logical_error_rate.py ${@:2:$#-2} "-p" "${@: -1}")
    local line="${@: -1} ${res}"
    (flock -e 200 echo $line >> "$1.csv") 200>lock
}

export -f run_and_write

parallel --load 16 --link run_and_write $@ ::: ${p[@]}
