# CSE 507 Foundation X+ on SOL

This repository is a class-oriented fork of [Foundation X](https://github.com/JLianglab/Foundation_X). Training code is in `Foundation_X+/`.

ASU Research Computing wants **Mamba environments**, not a self-installed conda/venv and not a custom Apptainer image for this workflow. Follow their Python docs:

- Do not install packages on login nodes. Use `interactive` first.
- `module load mamba/latest`
- `source activate` (never `mamba activate` / `conda activate`)
- `pip` only after the env is active; never `pip --user`
- PyTorch is installed with `pip` inside that env

## One-time setup on SOL

```bash
ssh hflechsi@sol.asu.edu
cd ~/CSE507-Foundation_X-Container   # or your clone path

interactive -t 120 -p htc
./sol/setup_env.sh
```

That creates `/scratch/$USER/envs/foundationx` (scratch, not `$HOME`, so it does not fill your home quota), installs Torch 1.13.1+cu117 with pip, then compiles `MultiScaleDeformableAttention`.

If `module load cuda-11.8.0-gcc-12.1.0` fails, run `module spider cuda` and rerun with:

```bash
CUDA_MODULE=cuda-11.6.2-gcc-12.1.0 ./sol/setup_env.sh
```

## Run experiments

All seven CANDID-PTX-only Foundation X+ runs use `--cyclictask`. Training uses only the requested heads. Evaluation always includes classification, localization, and segmentation so you can compare focused vs unfocused performance.

| ID | Experiment name | Trained tasks |
| --- | --- | --- |
| a | `candidptx_cls` | `candidptxCLS` |
| b | `candidptx_loc` | `candidptxLOC` |
| c | `candidptx_seg` | `candidptxSEG` |
| d | `candidptx_cls_loc` | `candidptxCLS` + `candidptxLOC` |
| e | `candidptx_cls_seg` | `candidptxCLS` + `candidptxSEG` |
| f | `candidptx_loc_seg` | `candidptxLOC` + `candidptxSEG` |
| g | `candidptx_cls_loc_seg` | `candidptxCLS` + `candidptxLOC` + `candidptxSEG` |

Each run is 50 epochs (`--total_epochs 51` because the training loop is `range(1, total_epochs)`). `--saveAllModel` writes a checkpoint after every trained task: `ckpt_E<epoch>_TH<head>.pth`. CANDID-PTX heads are `TH9` (CLS), `TH10` (LOC), and `TH11` (SEG).

```bash
# one experiment
sbatch sol/run_experiment.sbatch candidptx_cls 42

# resume from the latest checkpoint
sbatch sol/run_experiment.sbatch candidptx_cls 42 resume

# short smoke test
sbatch --time=0-04:00:00 --job-name=fx_smoke sol/run_experiment.sbatch candidptx_cls 42 smoke

# all seven
./sol/submit_all.sh 42
```

Or on an allocated GPU node after `source activate /scratch/$USER/envs/foundationx`:

```bash
./experiment_scripts/start_experiment.sh candidptx_cls 42
./experiment_scripts/resume_experiment.sh candidptx_cls 42
```

Outputs go to `/scratch/$USER/foundationx_runs/<experiment>_seed_<seed>/`.

Data paths default to the shared lab copies:

- images: `/scratch/jliang12/data/CANDID-PTX/`
- splits: `/data/jliang12/nuislam/data_files_splits/candid_ptx/`
- Ark backbone (if present): `/data/jliang12/dongaoma/Ark_models/...ep200.pth.tar`

If the Ark backbone is missing, the launcher falls back to ImageNet init.

## Analysis

```bash
module load mamba/latest
source activate /scratch/$USER/envs/foundationx
python plot/plot_synergy.py \
  /scratch/$USER/foundationx_runs/candidptx_cls_seed_42 \
  /scratch/$USER/foundationx_runs/candidptx_loc_seed_42 \
  /scratch/$USER/foundationx_runs/candidptx_seg_seed_42 \
  /scratch/$USER/foundationx_runs/candidptx_cls_loc_seed_42 \
  /scratch/$USER/foundationx_runs/candidptx_cls_seg_seed_42 \
  /scratch/$USER/foundationx_runs/candidptx_loc_seg_seed_42 \
  /scratch/$USER/foundationx_runs/candidptx_cls_loc_seg_seed_42 \
  --output plot/candidptx_synergy.png
```

Solid lines are focused (trained on that task). Dashed lines are unfocused (evaluated but not trained).

## Original papers and code

- Foundation X: `Foundation_X/`
- Foundation X+: `Foundation_X+/`
- Upstream: https://github.com/JLianglab/Foundation_X
