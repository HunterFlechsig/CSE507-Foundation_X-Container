#!/usr/bin/env bash
set -euo pipefail

# Create the Foundation X+ Mamba env on ASU SOL the way Research Computing
# documents it:
#   - interactive compute node (not a login node)
#   - module load mamba/latest
#   - source activate (never mamba/conda activate)
#   - pip only after the env is active; never pip --user
#   - PyTorch via pip, as SOL's PyTorch recipe requires
#
# Usage on SOL:
#   interactive -t 120 -p htc
#   cd ~/CSE507-Foundation_X-Container
#   ./sol/setup_env.sh
#
# Refs:
#   https://asurc.atlassian.net/wiki/spaces/RC/pages/1905328428
#   https://asurc.atlassian.net/wiki/spaces/RC/pages/1938423809
#   https://asurc.atlassian.net/wiki/spaces/RC/pages/2554363964

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=sol/env.sh
source "$ROOT/sol/env.sh"

if [[ "$(hostname)" == *login* && "${FORCE:-}" != "1" ]]; then
  echo "Do not install packages on a login node." >&2
  echo "Start a compute session first:" >&2
  echo "  interactive -t 120 -p htc" >&2
  echo "Then rerun: $0" >&2
  echo "Override with FORCE=1 only if you know you are already on a compute node." >&2
  exit 1
fi

module purge
module load mamba/latest

load_cuda() {
  local candidate
  local candidates=(
    "${CUDA_MODULE}"
    cuda-11.8.0-gcc-12.1.0
    cuda-11.7.0-gcc-12.1.0
    cuda-11.6.2-gcc-12.1.0
    cuda-12.6.1-gcc-12.1.0
  )
  for candidate in "${candidates[@]}"; do
    if module load "$candidate" 2>/dev/null; then
      CUDA_MODULE="$candidate"
      echo "Loaded CUDA module: $CUDA_MODULE"
      return 0
    fi
  done
  echo "Could not load a CUDA module. Run: module spider cuda" >&2
  return 1
}

load_cuda

SCRATCH="${SCRATCH:-/scratch/${USER}}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SCRATCH/conda-pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/pip-cache}"
export TMPDIR="${TMPDIR:-$SCRATCH/tmp}"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$(dirname "$FX_ENV")"

echo "Creating Mamba env at $FX_ENV"
if [[ ! -x "$FX_ENV/bin/python" ]]; then
  mamba create -y -p "$FX_ENV" -c conda-forge \
    python=3.10 pip setuptools=69.5.1 wheel ninja cython
else
  echo "Env already exists; installing/updating packages."
fi

# SOL docs: never `mamba activate`; use source activate with the env path.
source activate "$FX_ENV"

python -m pip install --no-cache-dir 'pip==24.0' 'setuptools==69.5.1' 'wheel==0.43.0'
python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu117 \
  torch==1.13.1 torchvision==0.14.1
python -m pip install --no-cache-dir -r "$ROOT/sol/requirements.txt"
python -m pip install --no-cache-dir 'setuptools==69.5.1'

export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
echo "CUDA_HOME=$CUDA_HOME"
python -c "import pkg_resources, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

cd "$ROOT/Foundation_X+/models/dino/ops"
python setup.py build install
rm -rf build
python -c "import MultiScaleDeformableAttention as m; print('MSDA', m.__file__)"

echo
echo "Env ready: $FX_ENV"
echo "Activate later with:"
echo "  module load mamba/latest"
echo "  source activate $FX_ENV"
echo "Then run an experiment:"
echo "  ./experiment_scripts/start_experiment.sh candidptx_cls 42"
echo "Or submit: sbatch sol/run_experiment.sbatch candidptx_cls 42"
