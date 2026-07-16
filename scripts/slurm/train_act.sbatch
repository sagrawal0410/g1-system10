#!/usr/bin/env bash
#SBATCH --job-name=act-train
#SBATCH --partition=defq
#SBATCH --account=liquidai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --time=02:00:00
#SBATCH --output=/lambdafs/shaurya/g1_sonic_system1/logs/%x-%j.out
set -euo pipefail
# hands-out: defaults | hands-in: REPO_ID=shaurya/g1_sonic_act_handsin DATA=.../g1_act_lerobot_handsB OUT=.../act_handsin_taskcond EXP=act-handsin-taskcond MASK=$DATA/hand_mask_table.json
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export WANDB_PROJECT=g1-sonic-vla
export HF_HUB_OFFLINE=0
REPO_ID="${REPO_ID:-shaurya/g1_sonic_act}"
DATA="${DATA:-/lambdafs/shaurya/g1_sonic_system1/data/g1_act_lerobot}"
OUT="${OUT:-/lambdafs/shaurya/g1_sonic_system1/outputs/act_baseline_latent}"
EXP="${EXP:-act-baseline-latent}"
MASK="${MASK:-}"
LAST_CKPT_CFG="$OUT/checkpoints/last/pretrained_model/train_config.json"
if [ -f "$LAST_CKPT_CFG" ]; then exec lerobot-train --config_path="$LAST_CKPT_CFG" --resume=true; fi
MASK_ARGS=()
if [ -n "$MASK" ]; then MASK_ARGS=(--policy.hand_active_mask=true --policy.hand_mask_table="$MASK"); fi
exec lerobot-train \
  --dataset.repo_id="$REPO_ID" --dataset.root="$DATA" --dataset.video_backend=pyav \
  --dataset.episodes='[1,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21]' \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.discrete_latent=true --policy.latent_block='[0,64]' \
  --policy.latent_num_classes=32 --policy.latent_grid_scale=16.0 "${MASK_ARGS[@]}" \
  --policy.chunk_size=40 --policy.n_action_steps=40 --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --output_dir="$OUT" --job_name="$EXP" --batch_size=8 --steps=50000 \
  --save_freq=5000 --log_freq=200 --num_workers=8 --env_eval_freq=0 \
  --wandb.enable=true --wandb.project=g1-sonic-vla
