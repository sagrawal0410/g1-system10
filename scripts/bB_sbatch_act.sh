#!/usr/bin/env bash
#SBATCH --job-name=act-baseline-latent
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=02:00:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
# Phase B ACT fallback (Slurm). Use only if the interactive gpu-003 run dies
# before completing. Resumes from the latest checkpoint if one exists (progress
# preserved), else starts fresh. Slurm assigns the GPU -> do NOT set
# CUDA_VISIBLE_DEVICES.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export WANDB_PROJECT=g1-sonic-vla

ORIG_OUT=/lambdafs/shaurya/g1_sonic_system1/outputs/act_baseline_latent
LAST_CKPT_CFG="$ORIG_OUT/checkpoints/last/pretrained_model/train_config.json"

if [ -f "$LAST_CKPT_CFG" ]; then
    echo "[sbatch] Resuming from existing checkpoint config: $LAST_CKPT_CFG"
    exec lerobot-train --config_path="$LAST_CKPT_CFG" --resume=true
fi

echo "[sbatch] No checkpoint found; starting a FRESH run to a new output dir."
FRESH_OUT=/lambdafs/shaurya/g1_sonic_system1/outputs/act_baseline_latent_slurm
exec lerobot-train \
  --dataset.repo_id=shaurya/g1_sonic_act \
  --dataset.root=/lambdafs/shaurya/g1_sonic_system1/data/g1_act_lerobot \
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
  --output_dir="$FRESH_OUT" \
  --job_name=act-baseline-latent \
  --batch_size=8 \
  --steps=50000 \
  --save_freq=5000 \
  --log_freq=200 \
  --num_workers=8 \
  --env_eval_freq=0 \
  --wandb.enable=true \
  --wandb.project=g1-sonic-vla
