#!/usr/bin/env bash
# Phase B: launch LeRobot ACT (discrete-FSQ-latent head) on the converted
# g1_act_lerobot dataset. Runs on liquid-gpu-003 GPU6. CPU-heavy pyav video
# decode -> several dataloader workers.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

export CUDA_VISIBLE_DEVICES=6
export WANDB_PROJECT=g1-sonic-vla
# Keep any HF hub calls offline-friendly; dataset + backbone are already local/cached.
export HF_HUB_OFFLINE=0

OUT=/lambdafs/shaurya/g1_sonic_system1/outputs/act_baseline_latent
mkdir -p /lambdafs/shaurya/g1_sonic_system1/outputs

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
  --output_dir="$OUT" \
  --job_name=act-baseline-latent \
  --batch_size=8 \
  --steps=50000 \
  --save_freq=5000 \
  --log_freq=200 \
  --num_workers=8 \
  --env_eval_freq=0 \
  --wandb.enable=true \
  --wandb.project=g1-sonic-vla
