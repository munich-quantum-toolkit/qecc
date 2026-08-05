#!/bin/bash
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
#
# SLURM array over the 4 flag-at-origin comparison codes, single p=0.001.
# One array task per code. Writes results/<code>_<x|z>_p0.001.line (reuses gather_results.sh).
#
#   Usage:  sbatch --clusters=serial --partition=serial_long --qos=cm4_serial_long \
#                  --array=0-3 submit_eval_fao.sbatch            # X errors
#           ... --array=0-3 submit_eval_fao.sbatch --z_errors    # Z errors

#SBATCH -J qecc_fao
#SBATCH -D ./
#SBATCH -o logs/fao_%A_%a.out
#SBATCH -e logs/fao_%A_%a.err
#SBATCH --clusters=serial
#SBATCH --partition=serial_long
#SBATCH --qos=cm4_serial_long
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --array=0-3
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --mail-type=NONE

module load slurm_setup
set -euo pipefail

EXTRA_ARGS=("$@")     # typically empty (X) or just --z_errors

# One code per array index (0..3)
CODES=(cc_17_1_5 cc_20_2_6 cc_25_1_5 cc_31_1_7)
# min-errors targets matched to the main pipeline per distance/basis:
#   cc_17_1_5=d5  cc_20_2_6=d6  cc_25_1_5=d5  cc_31_1_7=d7
NX=(500 500 500 100)     # X target per code
NZ=(250 250 250 50)      # Z target per code
CAP=5000000000           # shot cap (backstop; small codes finish well within 7 days)
P=0.001

CODE="${CODES[$SLURM_ARRAY_TASK_ID]}"

# Detect basis from extra args and pick the matching -n target.
BASIS="x"
for a in "${EXTRA_ARGS[@]}"; do
    if [ "$a" == "--z_errors" ]; then BASIS="z"; fi
done
if [ "$BASIS" == "z" ]; then
    NERR="${NZ[$SLURM_ARRAY_TASK_ID]}"
else
    NERR="${NX[$SLURM_ARRAY_TASK_ID]}"
fi

source "$HOME/qecc/.venv/bin/activate"
mkdir -p results logs

echo "[$(date '+%F %T')] host=$(hostname) code=${CODE} basis=${BASIS} p=${P} n=${NERR}"

RES="$(python estimate_logical_error_rate_fao.py "${CODE}" -p "${P}" -n "${NERR}" --shots "${CAP}" "${EXTRA_ARGS[@]}")"

echo "${P} ${RES}" > "results/${CODE}_${BASIS}_p${P}.line"
echo "[$(date '+%F %T')] done: ${CODE} ${BASIS} ${P} ${RES}"
