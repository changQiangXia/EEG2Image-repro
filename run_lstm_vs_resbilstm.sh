#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

DEFAULT_DATA_ROOT="${REPO_ROOT}/data/b2i_data"
DEFAULT_OFFICIAL_LSTM_CKPT="${REPO_ROOT}/lstm_kmean/experiments/best_ckpt/ckpt-2420"

if [[ ! -d "${DEFAULT_DATA_ROOT}" && -d /root/autodl-tmp/data/b2i_data ]]; then
  DEFAULT_DATA_ROOT="/root/autodl-tmp/data/b2i_data"
fi

if [[ ! -f "${DEFAULT_OFFICIAL_LSTM_CKPT}.index" && -f /root/autodl-tmp/lstm_kmean/experiments/best_ckpt/ckpt-2420.index ]]; then
  DEFAULT_OFFICIAL_LSTM_CKPT="/root/autodl-tmp/lstm_kmean/experiments/best_ckpt/ckpt-2420"
fi

DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
OFFICIAL_LSTM_CKPT="${OFFICIAL_LSTM_CKPT:-${DEFAULT_OFFICIAL_LSTM_CKPT}}"
SEEDS="${SEEDS:-45 123 2026}"

OFFICIAL_GAN_DIR="experiments/formal_probe_lstm_featl2_gan_e10"
RESBILSTM_FEATURE_DIR="experiments/formal_resbilstm_lstm_frozen_feature_e20_lr1e4_v2"
RESBILSTM_GAN_DIR="experiments/formal_resbilstm_lstm_frozen_v2_featl2_gan_e10_ckptevery"

python lstm_kmean/train.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${RESBILSTM_FEATURE_DIR}" \
  --encoder_variant resbilstm_lstm \
  --epochs 20 \
  --batch_size 256 \
  --feature_dim 128 \
  --learning_rate 0.0001 \
  --checkpoint_every_epochs 5 \
  --max_to_keep 5000 \
  --seed 45 \
  --warmstart_lstm_ckpt "${OFFICIAL_LSTM_CKPT}" \
  --freeze_warmstart_lstm true

python train_gan.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OFFICIAL_GAN_DIR}" \
  --triplet_ckpt_path "${OFFICIAL_LSTM_CKPT}" \
  --encoder_variant lstm \
  --condition_source feat_l2norm \
  --epochs 10 \
  --batch_size 128 \
  --checkpoint_every_epochs 10 \
  --use_diffaug false \
  --use_mode_loss false \
  --seed 45

python train_gan.py \
  --data_root "${DATA_ROOT}" \
  --output_dir "${RESBILSTM_GAN_DIR}" \
  --triplet_ckpt_dir "${RESBILSTM_FEATURE_DIR}/ckpt" \
  --triplet_ckpt_path "${RESBILSTM_FEATURE_DIR}/ckpt/ckpt-17" \
  --encoder_variant resbilstm_lstm \
  --condition_source feat_l2norm \
  --epochs 10 \
  --batch_size 128 \
  --checkpoint_every_epochs 1 \
  --max_to_keep 20 \
  --use_diffaug false \
  --use_mode_loss false \
  --seed 45

for seed in ${SEEDS}; do
  official_eval_dir="experiments/formal_probe_lstm_featl2_eval10k_seed${seed}"
  resbilstm_eval_dir="experiments/formal_resbilstm_lstm_frozen_v2_ckpt10_eval10k_seed${seed}"

  if [[ "${seed}" == "45" ]]; then
    official_eval_dir="experiments/formal_probe_lstm_featl2_eval10k"
    resbilstm_eval_dir="experiments/formal_resbilstm_lstm_frozen_v2_ckpt10_eval10k"
  fi

  python generate_from_ckpt.py \
    --data_root "${DATA_ROOT}" \
    --output_dir "${official_eval_dir}" \
    --triplet_ckpt_path "${OFFICIAL_LSTM_CKPT}" \
    --encoder_variant lstm \
    --condition_source feat_l2norm \
    --gan_ckpt_dir "${OFFICIAL_GAN_DIR}/ckpt" \
    --gan_ckpt_path "${OFFICIAL_GAN_DIR}/ckpt/ckpt-10" \
    --test_image_count 10000 \
    --dataset_batch_size 64 \
    --generate_batch_size 64 \
    --seed "${seed}"

  python evaluate_is.py \
    --image_dir "${official_eval_dir}/images" \
    --output_path "${official_eval_dir}/inception_score.json" \
    --splits 10 \
    --batch_size 32

  python generate_from_ckpt.py \
    --data_root "${DATA_ROOT}" \
    --output_dir "${resbilstm_eval_dir}" \
    --triplet_ckpt_dir "${RESBILSTM_FEATURE_DIR}/ckpt" \
    --triplet_ckpt_path "${RESBILSTM_FEATURE_DIR}/ckpt/ckpt-17" \
    --encoder_variant resbilstm_lstm \
    --condition_source feat_l2norm \
    --gan_ckpt_dir "${RESBILSTM_GAN_DIR}/ckpt" \
    --gan_ckpt_path "${RESBILSTM_GAN_DIR}/ckpt/ckpt-10" \
    --test_image_count 10000 \
    --dataset_batch_size 64 \
    --generate_batch_size 64 \
    --seed "${seed}"

  python evaluate_is.py \
    --image_dir "${resbilstm_eval_dir}/images" \
    --output_path "${resbilstm_eval_dir}/inception_score.json" \
    --splits 10 \
    --batch_size 32
done

python - <<'PY'
import json
from pathlib import Path

pairs = [
    ("45", Path("experiments/formal_probe_lstm_featl2_eval10k/inception_score.json"), Path("experiments/formal_resbilstm_lstm_frozen_v2_ckpt10_eval10k/inception_score.json")),
    ("123", Path("experiments/formal_probe_lstm_featl2_eval10k_seed123/inception_score.json"), Path("experiments/formal_resbilstm_lstm_frozen_v2_ckpt10_eval10k_seed123/inception_score.json")),
    ("2026", Path("experiments/formal_probe_lstm_featl2_eval10k_seed2026/inception_score.json"), Path("experiments/formal_resbilstm_lstm_frozen_v2_ckpt10_eval10k_seed2026/inception_score.json")),
]

base_scores = []
candidate_scores = []
for seed, base_path, candidate_path in pairs:
    if not base_path.exists() or not candidate_path.exists():
        continue
    base = json.loads(base_path.read_text())["inception_score_mean"]
    candidate = json.loads(candidate_path.read_text())["inception_score_mean"]
    base_scores.append(base)
    candidate_scores.append(candidate)
    print(f"seed {seed}: official_lstm={base:.6f}, resbilstm_lstm={candidate:.6f}, gap={candidate - base:+.6f}")

if base_scores and len(base_scores) == len(candidate_scores):
    base_mean = sum(base_scores) / len(base_scores)
    candidate_mean = sum(candidate_scores) / len(candidate_scores)
    print(f"mean: official_lstm={base_mean:.6f}, resbilstm_lstm={candidate_mean:.6f}, gap={candidate_mean - base_mean:+.6f}")
PY
