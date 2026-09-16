#!/bin/bash
#SBATCH --job-name=fx_candidptx
#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 12
#SBATCH --mem=80G
#SBATCH -p general
#SBATCH -q class
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hunterflechsig@gmail.com
#SBATCH --export=NONE
#SBATCH -o /scratch/%u/foundationx_runs/logs/%x_%j.out
#SBATCH -e /scratch/%u/foundationx_runs/logs/%x_%j.err

# Foundation X+ CANDID-PTX experiment on SOL (Mamba env, no container).
#
#   sbatch sol/run_experiment.sbatch candidptx_cls 42
#   sbatch sol/run_experiment.sbatch candidptx_cls 42 resume
#   sbatch --time=0-04:00:00 --job-name=fx_smoke sol/run_experiment.sbatch candidptx_cls 42 smoke

set -euo pipefail

EXPERIMENT="${1:?Usage: sbatch sol/run_experiment.sbatch <experiment> <seed> [start|resume|smoke]}"
SEED="${2:?Usage: sbatch sol/run_experiment.sbatch <experiment> <seed> [start|resume|smoke]}"
MODE="${3:-start}"

REPO_ROOT="${REPO_ROOT:-$HOME/CSE507-Foundation_X-Container}"
if [[ ! -d "$REPO_ROOT/Foundation_X+" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# shellcheck source=sol/env.sh
source "$REPO_ROOT/sol/env.sh"

mkdir -p "/scratch/${USER}/foundationx_runs/logs"
cd "$REPO_ROOT"

export SCRATCH="/scratch/${USER}"
export OUTPUT_ROOT="${SCRATCH}/foundationx_runs"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-24}"
export FX_ENV

if [[ "$MODE" == "smoke" ]]; then
  export TOTAL_EPOCHS=2
  export BATCH_SIZE=8
  export NUM_WORKERS=4
fi

module purge
module load mamba/latest
if ! module load "$CUDA_MODULE" 2>/dev/null; then
  echo "Could not load $CUDA_MODULE; trying documented CUDA 12.6 module."
  module load cuda-12.6.1-gcc-12.1.0
fi
# SOL docs: source activate, never mamba activate.
source activate "$FX_ENV"

echo "========================================="
echo "Host       : $(hostname)"
echo "Job ID     : ${SLURM_JOB_ID:-none}"
echo "Experiment : ${EXPERIMENT}"
echo "Seed       : ${SEED}"
echo "Mode       : ${MODE}"
echo "Repo       : ${REPO_ROOT}"
echo "Python     : $(command -v python)"
echo "Env        : ${FX_ENV}"
echo "Date       : $(date)"
echo "========================================="
nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [[ "$MODE" == "resume" ]]; then
  ./experiment_scripts/resume_experiment.sh "$EXPERIMENT" "$SEED"
else
  ./experiment_scripts/start_experiment.sh "$EXPERIMENT" "$SEED"
fi
