#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_SCRIPTS_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$EXPERIMENT_SCRIPTS_ROOT/.." && pwd)

# Always evaluate all three CANDID-PTX tasks so focused vs unfocused
# performance can be compared after every trained task.
CANDID_EVAL_TAGS="TESTcls_candidptxCLS_TESTloc_candidptxLOC_TESTseg_candidptxSEG"

declare -A FX_TRAIN_TAGS=(
  [candidptx_cls]="candidptxCLS"
  [candidptx_loc]="candidptxLOC"
  [candidptx_seg]="candidptxSEG"
  [candidptx_cls_loc]="candidptxCLS_candidptxLOC"
  [candidptx_cls_seg]="candidptxCLS_candidptxSEG"
  [candidptx_loc_seg]="candidptxLOC_candidptxSEG"
  [candidptx_cls_loc_seg]="candidptxCLS_candidptxLOC_candidptxSEG"
)

fx_known_experiments() {
  printf '%s\n' "${!FX_TRAIN_TAGS[@]}" | sort
}

fx_cyclictask() {
  local experiment_name="$1"
  local train_tags="${FX_TRAIN_TAGS[$experiment_name]:-}"
  if [[ -z "$train_tags" ]]; then
    echo "Unknown experiment: $experiment_name" >&2
    echo "Expected one of:" >&2
    fx_known_experiments >&2
    return 2
  fi
  printf '%s_%s\n' "$train_tags" "$CANDID_EVAL_TAGS"
}

fx_python() {
  # shellcheck source=sol/env.sh
  source "$REPO_ROOT/sol/env.sh"
  if [[ ! -x "$FX_ENV/bin/python" ]]; then
    echo "Foundation X+ env not found at $FX_ENV" >&2
    echo "On SOL, create it from a compute node:" >&2
    echo "  interactive -t 120 -p htc" >&2
    echo "  ./sol/setup_env.sh" >&2
    return 1
  fi
  printf '%s\n' "$FX_ENV/bin/python"
}

fx_latest_checkpoint() {
  local output_dir="$1"
  local latest=""
  local candidate
  shopt -s nullglob
  for candidate in "$output_dir"/ckpt_E*_TH*.pth; do
    latest="$candidate"
  done
  shopt -u nullglob
  if [[ -z "$latest" ]]; then
    return 1
  fi
  # Prefer the numerically latest epoch, not directory order.
  python3 - "$output_dir" <<'PY'
import glob, os, re, sys
root = sys.argv[1]
pattern = re.compile(r"ckpt_E(\d+)_TH(\d+)\.pth$")
best = None
best_key = (-1, -1)
for path in glob.glob(os.path.join(root, "ckpt_E*_TH*.pth")):
    match = pattern.search(os.path.basename(path))
    if not match:
        continue
    key = (int(match.group(1)), int(match.group(2)))
    if key > best_key:
        best_key = key
        best = path
if best is None:
    sys.exit(1)
print(best)
PY
}

fx_launch_experiment() {
  local experiment_name="$1"
  local seed="$2"
  local resume="$3"
  shift 3 || true

  if [[ ! "$seed" =~ ^-?[0-9]+$ ]]; then
    echo "Seed must be an integer: $seed" >&2
    return 2
  fi

  local python_bin
  python_bin=$(fx_python)

  local cyclictask
  cyclictask=$(fx_cyclictask "$experiment_name")

  local scratch="${SCRATCH:-/scratch/${USER:-$LOGNAME}}"
  local output_root="${OUTPUT_ROOT:-$scratch/foundationx_runs}"
  local output_dir="${output_root}/${experiment_name}_seed_${seed}"
  mkdir -p "$output_dir"

  # Epoch loop is range(start_epoch=1, total_epochs), so 51 yields epochs 1-50.
  local total_epochs="${TOTAL_EPOCHS:-51}"
  local batch_size="${BATCH_SIZE:-24}"
  local num_workers="${NUM_WORKERS:-8}"
  local imgsize="${IMGSIZE:-224}"
  local backbone="${BACKBONEMODEL:-Swin-B}"
  local opt="${OPT:-adamw}"
  local server="${SERVER:-SOL}"
  local config="${CONFIGFILE:-config/DINO/DINO_4scale_swinBASE.py}"
  local dataset_file="${DATASETFILE:-foundation6Ark6_datasets}"
  local coco_path="${COCO_PATH:-/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/}"
  local backbone_dir="${BACKBONE_DIR:-/data/jliang12/dongaoma/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar}"
  local init="${INIT:-ark}"

  if [[ ! -f "$backbone_dir" ]]; then
    echo "Ark backbone not found at $backbone_dir; falling back to ImageNet init." >&2
    init="imagenet22k"
    backbone_dir=""
  fi

  local training_args=(
    --taskcomponent foundation_x5_pretraining
    --train
    --numClasses 1
    --dataset_file "$dataset_file"
    --classification_dataset "$dataset_file"
    --num_workers "$num_workers"
    --coco_path "$coco_path"
    --weight-decay 0.0001
    --output_dir "$output_dir"
    -c "$config"
    --imgsize "$imgsize"
    --backbonemodel "$backbone"
    --init "$init"
    --total_epochs "$total_epochs"
    --batch_size "$batch_size"
    --opt "$opt"
    --seed "$seed"
    --finetune_ignore label_enc.weight class_embed
    --lr_backbone "${LR_BACKBONE:-1e-5}"
    --lr_locEnc "${LR_LOCENC:-1e-4}"
    --lr_locDec "${LR_LOCDEC:-1e-4}"
    --lr_segmentor "${LR_SEGMENTOR:-1e-4}"
    --cyclictask "$cyclictask"
    --serverC "$server"
    --modelEMA True_Epoch
    --lockrelease
    --saveAllModel
    --options dn_scalar=100 embed_init_tgt=TRUE
    dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False
    dn_box_noise_scale=1.0
  )

  if [[ -n "$backbone_dir" ]]; then
    training_args+=(--backbone_dir "$backbone_dir")
  fi

  if [[ "$resume" == true ]]; then
    local ckpt
    if ! ckpt=$(fx_latest_checkpoint "$output_dir"); then
      echo "No checkpoint found to resume in $output_dir" >&2
      return 1
    fi
    echo "Resuming from $ckpt"
    training_args+=(--resume "$ckpt")
  fi

  echo "Experiment : $experiment_name"
  echo "Seed       : $seed"
  echo "Python     : $python_bin"
  echo "Cyclic task: $cyclictask"
  echo "Output     : $output_dir"

  cd "$REPO_ROOT/Foundation_X+"
  exec "$python_bin" main_Consolidated.py "${training_args[@]}" "$@"
}
