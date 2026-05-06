#!/bin/bash
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 OUTPUT_PREFIX [extra python flags...]"
  exit 1
fi

# First argument = output CSV file name (without .csv)
OUT="$1.csv"
shift 1   # remove output file name; remaining args are forwarded to python

# Join remaining args into a single exported string
EXTRA_ARGS_STR="$*"
export EXTRA_ARGS_STR

# arrays of jobs
CODES=("cc_17_1_5" "cc_20_2_6" "cc_25_1_5" "cc_31_1_7")
P_VALUES=("0.001")

echo "code,p,p_logical,acceptance,errors,shots" > "$OUT"

run_and_write() {
  # This function will be run by GNU parallel in a new shell.
  # Reconstruct EXTRA_ARGS array locally from the exported string.
  read -r -a EXTRA_ARR <<< "$EXTRA_ARGS_STR"

  local code="$1"
  local p="$2"

  # call python with reconstructed array (handles multiple flags)
  local result
  result=$(python3 estimate_logical_error_rate_fao.py "$code" -p "$p" "${EXTRA_ARR[@]}")

  local line="$code,$p,$result"

  # append atomically
  ( flock -e 200; echo "$line" >> "$OUT" ) 200>lockfile
}

export -f run_and_write
export OUT

# Run parallel: Cartesian product of CODES x P_VALUES
parallel --jobs 2 run_and_write ::: "${CODES[@]}" ::: "${P_VALUES[@]}"
