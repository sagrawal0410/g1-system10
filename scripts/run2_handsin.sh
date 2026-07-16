#!/usr/bin/env bash
# RUN 2 — GR00T-B "hands in, normalized + task-conditioned" SFT launch.
# 3-GPU DDP on liquid-gpu-003 GPUs 3,4,5 (distinct from Run 1's 0,1,2).
# DISTINCT master_port (29501) so it doesn't collide with Run 1's 29500 —
# both co-reside on one node; a shared port would hang DDP rendezvous.
#
# Usage: bash run2_handsin.sh <OPTION_B_DATASET_PATH>
set -euo pipefail

DATASET_PATH="${1:?Usage: run2_handsin.sh <option-B dataset path>}"

export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate groot
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export ACTION_LAYOUT_PATH=/home/shaurya/g1_sonic_system1/results/action_layout.json

export CUDA_VISIBLE_DEVICES=3,4,5
export NUM_GPUS=3
export MASTER_PORT=29501
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export MAX_STEPS="${MAX_STEPS:-20000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export USE_WANDB=1
export WANDB_PROJECT=g1-sonic-vla
export WANDB__SERVICE_WAIT=300

cd ~/g1_sonic_system1/repos/Isaac-GR00T

exec bash ~/g1_sonic_system1/scripts/finetune_with_block_loss.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag UNITREE_G1_SONIC \
    --output-dir /lambdafs/shaurya/g1_sonic_system1/outputs/grootN17-sft-handsin-taskcond \
    --experiment-name grootN17-sft-handsin-taskcond \
    --wandb-project g1-sonic-vla
