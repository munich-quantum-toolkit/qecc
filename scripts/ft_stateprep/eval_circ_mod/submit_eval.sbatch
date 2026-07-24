#!/bin/bash
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
#
# SLURM job-array replacement for run_eval_on_code.sh (GNU parallel).
# Each array task computes the logical error rate for ONE physical error rate p.
#
#   Usage:   sbatch submit_eval.sbatch <code_name> [extra args for the python script]
#   Example: sbatch submit_eval.sbatch cc_6_6_6_d5 -d 5
#            sbatch submit_eval.sbatch cc_6_6_6_d5 -d 5 --z_errors
#
# IMPORTANT: keep the --array range in sync with the length of `p_values` below
#            (0-indexed).  3 values -> --array=0-2 ; 18 values -> --array=0-17.

#SBATCH -J qecc_eval
#SBATCH -D ./                          # working dir = the directory you submit from
#SBATCH -o logs/qecc_%A_%a.out         # %A = array-job id, %a = array-task id
#SBATCH -e logs/qecc_%A_%a.err
#SBATCH --clusters=serial              # LRZ: 1-few-core jobs go to the 'serial' cluster
#SBATCH --partition=serial_std         # serial_std (<=48h). Use serial_long for longer runs.
#SBATCH --cpus-per-task=1              # each python run is single-threaded
#SBATCH --mem=4G                       # bump to 8-16G for the d7 color code + LUT decoder
#SBATCH --time=24:00:00                # backstop only; the real terminator is --fixed-shots (serial allows up to 48h)
#SBATCH --array=0-17                   # <-- one task per p-value; match p_values length (18 values -> 0-17)
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --mail-type=NONE               # leave NONE unless you add a *valid* --mail-user
# #SBATCH --mail-user=erik.example@tum.de   # invalid address => LRZ revokes submission rights

module load slurm_setup

set -euo pipefail

# ---------------------------------------------------------------- configuration
CODE="${1:?usage: sbatch submit_eval.sbatch <code_name> [extra python args]}"
shift
EXTRA_ARGS=("$@")

# The physical error rates to sweep. Index into this with SLURM_ARRAY_TASK_ID.
# Index reference:  0=0.5  1=0.3  2=0.1  3=0.09  4=0.07  5=0.05  6=0.03  7=0.01
#                   8=0.009  9=0.007  10=0.005  11=0.003  12=0.001  13=0.0009
#                   14=0.0007  15=0.0005  16=0.0003  17=0.0001
# Full sweep -> --array=0-17.  Single point p=0.001 -> --array=12.  Calibration -> --array=0.
p_values=(0.5 0.3 0.1 0.09 0.07 0.05 0.03 0.01 0.009 0.007 0.005 0.003 0.001 0.0009 0.0007 0.0005 0.0003 0.0001)

P="${p_values[$SLURM_ARRAY_TASK_ID]}"

# Detect the error basis from the extra args so X and Z runs write to DIFFERENT files.
# (No effort on your side: pass --z_errors as usual and the filename tag flips to "z".)
BASIS="x"
for a in "${EXTRA_ARGS[@]}"; do
    if [ "$a" == "--z_errors" ]; then BASIS="z"; fi
done

# ---------------------------------------------------------------- environment
# venv created once on a LOGIN node (see setup instructions). Set this to wherever
# yours lives -- e.g. "$HOME/qecc/.venv" if you used `uv venv` inside the repo.
VENV="$HOME/qecc/.venv"
source "${VENV}/bin/activate"
# If you needed a `module load gcc` to COMPILE the C++ extension, load the same here:
# module load gcc

# ---------------------------------------------------------------- run
mkdir -p results logs

echo "[$(date '+%F %T')] host=$(hostname) code=${CODE} basis=${BASIS} p=${P} task=${SLURM_ARRAY_TASK_ID}"

RES="$(python estimate_logical_error_rate.py "${CODE}" -p "${P}" "${EXTRA_ARGS[@]}")"

# One line per task: "<p> <p_l,acceptance,errors,runs>"  (same format as before, no flock needed)
# Filename carries the basis tag (x/z) so the X and Z sweeps never overwrite each other.
echo "${P} ${RES}" > "results/${CODE}_${BASIS}_p${P}.line"

echo "[$(date '+%F %T')] done: basis=${BASIS} ${P} ${RES}"
