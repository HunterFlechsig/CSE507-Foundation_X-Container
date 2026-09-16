#!/usr/bin/env bash
# Shared SOL paths. Source this from setup and launch scripts.
# Docs: use `source activate`, never `mamba activate` / `conda activate`.

FX_ENV="${FX_ENV:-/scratch/${USER}/envs/foundationx}"
CUDA_MODULE="${CUDA_MODULE:-cuda-11.8.0-gcc-12.1.0}"
