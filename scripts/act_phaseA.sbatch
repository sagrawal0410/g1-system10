#!/usr/bin/env bash
#SBATCH --job-name=actA-rollout
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=01:00:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
# Phase A: (1) mint a near-zero-step ACT proxy (R0_base_ACT_handsout) using the
# SAME recipe as the trained hands-out ACT (bB_launch_act.sh) but steps=10, then
# (2) open-loop rollout both the trained ACT (RA_handsout) and the proxy to npz.
# Slurm assigns the GPU -> do NOT set CUDA_VISIBLE_DEVICES.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
export HF_HUB_OFFLINE=0

DATA_ROOT=/lambdafs/shaurya/g1_sonic_system1/data/g1_act_lerobot
RA_CKPT=/lambdafs/shaurya/g1_sonic_system1/outputs/act_baseline_latent/checkpoints/020000/pretrained_model
PROXY_OUT=/lambdafs/shaurya/g1_sonic_system1/outputs/act_base_proxy_handsout
ROLLOUT=$HOME/g1_sonic_system1/scripts/act_rollout.py

echo "[actA] node=$(hostname) job=${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=index,name,memory.used --format=csv || true

# ---- 1. Mint the near-zero-step proxy (skip if already present) ----
PROXY_CKPT=$(ls -d "$PROXY_OUT"/checkpoints/[0-9]*/pretrained_model 2>/dev/null | sort | tail -1 || true)
if [ -z "${PROXY_CKPT:-}" ] || [ ! -f "${PROXY_CKPT:-}/model.safetensors" ]; then
    echo "[actA] Minting near-zero-step ACT proxy at $PROXY_OUT (steps=10) ..."
    rm -rf "$PROXY_OUT"
    lerobot-train \
      --dataset.repo_id=shaurya/g1_sonic_act \
      --dataset.root="$DATA_ROOT" \
      --dataset.video_backend=pyav \
      --dataset.episodes='[1,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21]' \
      --policy.type=act \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --policy.discrete_latent=true \
      --policy.latent_block='[0,64]' \
      --policy.latent_num_classes=32 \
      --policy.latent_grid_scale=16.0 \
      --policy.chunk_size=40 \
      --policy.n_action_steps=40 \
      --policy.vision_backbone=resnet18 \
      --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
      --output_dir="$PROXY_OUT" \
      --job_name=act-base-proxy-handsout \
      --batch_size=8 \
      --steps=10 \
      --save_freq=10 \
      --log_freq=1 \
      --num_workers=8 \
      --env_eval_freq=0 \
      --wandb.enable=false
    PROXY_CKPT=$(ls -d "$PROXY_OUT"/checkpoints/[0-9]*/pretrained_model 2>/dev/null | sort | tail -1 || true)
else
    echo "[actA] Proxy checkpoint already present: $PROXY_CKPT (skipping mint)"
fi

if [ -z "${PROXY_CKPT:-}" ] || [ ! -f "${PROXY_CKPT}/model.safetensors" ]; then
    echo "[actA] ERROR: proxy checkpoint not found after minting." >&2
    ls -R "$PROXY_OUT/checkpoints" >&2 || true
    exit 1
fi
echo "[actA] proxy checkpoint = $PROXY_CKPT"

# ---- 2. Rollouts -> npz ----
echo "[actA] Rollout RA_handsout (trained ACT) ..."
python "$ROLLOUT" --label RA_handsout --checkpoint "$RA_CKPT"

echo "[actA] Rollout R0_base_ACT_handsout (proxy) ..."
python "$ROLLOUT" --label R0_base_ACT_handsout --checkpoint "$PROXY_CKPT"

echo "[actA] DONE. npz in /lambdafs/shaurya/g1_sonic_system1/results/eval_npz/"
ls -la /lambdafs/shaurya/g1_sonic_system1/results/eval_npz/ | grep -E 'RA_handsout|R0_base_ACT_handsout' || true
