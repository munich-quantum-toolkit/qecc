#!/bin/bash
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
#
# Collect the per-task result lines written by submit_eval.sbatch into CSVs.
# Produces one CSV per basis that exists: <code>_x.csv and/or <code>_z.csv.
#
#   Usage:   ./gather_results.sh <code_name>
#   Example: ./gather_results.sh cc_6_6_6_d5

set -euo pipefail

CODE="${1:?usage: ./gather_results.sh <code_name>}"

shopt -s nullglob
found_any=0

for basis in x z; do
    lines=(results/${CODE}_${basis}_p*.line)
    if [ ${#lines[@]} -eq 0 ]; then
        continue
    fi
    found_any=1
    out="${CODE}_${basis}.csv"
    {
        echo "p p_l acceptance errors runs"
        cat "${lines[@]}" | sort -g   # sort numerically by the leading p value
    } > "$out"
    echo "== ${out} (${#lines[@]} rows) =="
    column -t "$out"
    echo
done

if [ "$found_any" -eq 0 ]; then
    echo "No result files results/${CODE}_{x,z}_p*.line found. Did the array job finish?" >&2
    exit 1
fi
