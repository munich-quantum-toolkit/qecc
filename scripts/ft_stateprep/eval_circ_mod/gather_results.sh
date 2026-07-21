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
# Collect the per-task result lines written by submit_eval.sbatch into one CSV.
#
#   Usage:   ./gather_results.sh <code_name> [output_csv]
#   Example: ./gather_results.sh cc_6_6_6_d5
#            ./gather_results.sh cc_6_6_6_d5 cc_6_6_6_d5.csv

set -euo pipefail

CODE="${1:?usage: ./gather_results.sh <code_name> [output_csv]}"
OUT="${2:-${CODE}.csv}"

shopt -s nullglob
lines=(results/${CODE}_p*.line)
if [ ${#lines[@]} -eq 0 ]; then
    echo "No result files results/${CODE}_p*.line found. Did the array job finish?" >&2
    exit 1
fi

{
    echo "p p_l acceptance errors runs"
    cat "${lines[@]}" | sort -g   # sort numerically by the leading p value
} > "$OUT"

echo "Wrote ${#lines[@]} rows to ${OUT}:"
column -t "$OUT"
