#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate eeg2image

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="data/b2i_data"
EPOCHS=210
TEST_IMAGE_COUNT=50000
DATASET_BATCH_SIZE=64
GENERATE_BATCH_SIZE=64
SPLITS=10

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

is_training_running() {
  local name="$1"
  pgrep -af "python train_gan.py.*--output_dir experiments/ablations/${name}( |$)" >/dev/null
}

train_experiment() {
  local name="$1"
  local use_diffaug="$2"
  local use_mode_loss="$3"
  local output_dir="experiments/ablations/${name}"
  local ckpt_index="${output_dir}/ckpt/ckpt-210.index"

  mkdir -p "$output_dir"

  while [ ! -f "$ckpt_index" ]; do
    if is_training_running "$name"; then
      log "${name}: detected active training process, waiting for ckpt-210"
      sleep 120
      continue
    fi

    log "${name}: starting/resuming training"
    if ! python train_gan.py \
      --data_root "$DATA_ROOT" \
      --output_dir "$output_dir" \
      --epochs "$EPOCHS" \
      --use_diffaug "$use_diffaug" \
      --use_mode_loss "$use_mode_loss" \
      >> "${output_dir}/train.out" 2>&1; then
      log "${name}: training command exited with failure, retrying in 30s"
      sleep 30
      continue
    fi
  done

  log "${name}: ckpt-210 is ready"
}

evaluate_experiment() {
  local name="$1"
  local output_dir="experiments/ablations/${name}"
  local eval_dir="experiments/ablations/${name}_eval"
  local ckpt_path="${output_dir}/ckpt/ckpt-210"
  local result_json="${eval_dir}/inception_score.json"

  if [ -f "$result_json" ]; then
    log "${name}: evaluation already exists, skipping"
    return
  fi

  mkdir -p "$eval_dir"

  log "${name}: generating 50k images from ckpt-210"
  python generate_from_ckpt.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$eval_dir" \
    --gan_ckpt_dir "${output_dir}/ckpt" \
    --gan_ckpt_path "$ckpt_path" \
    --test_image_count "$TEST_IMAGE_COUNT" \
    --dataset_batch_size "$DATASET_BATCH_SIZE" \
    --generate_batch_size "$GENERATE_BATCH_SIZE" \
    >> "${eval_dir}/generate.out" 2>&1

  log "${name}: evaluating inception score"
  python evaluate_is.py \
    --image_dir "${eval_dir}/images" \
    --output_path "$result_json" \
    --splits "$SPLITS" \
    >> "${eval_dir}/evaluate.out" 2>&1

  log "${name}: evaluation finished -> ${result_json}"
}

main() {
  log "Starting ablation suite on GPU ${CUDA_VISIBLE_DEVICES}"

  train_experiment baseline false false
  evaluate_experiment baseline

  train_experiment mode_only false true
  evaluate_experiment mode_only

  train_experiment diffaug_only true false
  evaluate_experiment diffaug_only

  train_experiment full true true
  evaluate_experiment full

  log "Ablation suite completed"
}

main "$@"
