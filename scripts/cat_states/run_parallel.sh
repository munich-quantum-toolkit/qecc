#!/bin/bash
# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

declare -a p=("0.001" "0.002" "0.003" "0.004" "0.005" "0.006" "0.007" "0.008" "0.009" "0.01" "0.02" "0.03" "0.04" "0.05" "0.06" "0.07" "0.08" "0.09" "0.1" "0.2")

echo "p;rar;errors;errors_errors" > "$1.csv"

run_and_write() {
    local res=$(python estimate_error_rate.py ${@:2:$#-2} "-p" "${@: -1}")
    local line="${@: -1} ${res}"
    (flock -e 200 echo $line >> "$1.csv") 200>lock
}

export -f run_and_write

parallel --load 2 --link run_and_write $@ ::: ${p[@]}
