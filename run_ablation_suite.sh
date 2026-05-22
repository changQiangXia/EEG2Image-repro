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
EXPERIMENT_ROOT="experiments/backbone_compare"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

is_training_running() {
  local name="$1"
  pgrep -af "python train_gan.py.*--output_dir ${EXPERIMENT_ROOT}/${name}( |$)" >/dev/null
}

train_experiment() {
  local name="$1"
  local gan_variant="$2"
  local output_dir="${EXPERIMENT_ROOT}/${name}"
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
      --gan_variant "$gan_variant" \
      --use_diffaug false \
      --use_mode_loss false \
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
  local gan_variant="$2"
  local output_dir="${EXPERIMENT_ROOT}/${name}"
  local eval_dir="${EXPERIMENT_ROOT}/${name}_eval"
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
    --gan_variant "$gan_variant" \
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
  log "Starting backbone comparison on GPU ${CUDA_VISIBLE_DEVICES}"

  train_experiment simple_gan simple_gan
  evaluate_experiment simple_gan simple_gan

  train_experiment dcgan dcgan
  evaluate_experiment dcgan dcgan

  log "Backbone comparison completed"
}

main "$@"
