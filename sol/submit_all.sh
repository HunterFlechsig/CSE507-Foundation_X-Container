#!/usr/bin/env bash
set -euo pipefail

# Submit the seven CANDID-PTX Foundation X+ experiments.
# Usage: ./sol/submit_all.sh [seed]

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SEED="${1:-42}"
SBATCH="$ROOT/sol/run_experiment.sbatch"

experiments=(
  candidptx_cls
  candidptx_loc
  candidptx_seg
  candidptx_cls_loc
  candidptx_cls_seg
  candidptx_loc_seg
  candidptx_cls_loc_seg
)

mkdir -p "/scratch/${USER}/foundationx_runs/logs"

for experiment in "${experiments[@]}"; do
  echo "Submitting $experiment seed $SEED"
  sbatch --job-name="fx_${experiment}" "$SBATCH" "$experiment" "$SEED"
done
