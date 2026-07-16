#!/usr/bin/env bash
# RUN 1 — GR00T-A "hands out" SFT launch (3-GPU DDP on liquid-gpu-003 GPUs 0,1,2).
set -euo pipefail

export HF_HUB_CACHE=/lambdafs/shaurya/hf_cache/hub
source ~/miniconda3/etc/profile.d/conda.sh
conda activate groot
# torchcodec (video decode) needs conda's ffmpeg-7 libs on the loader path;
# the compute node has no system ffmpeg, so this is REQUIRED or the dataloader dies.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# per-block (latent/left_hand/right_hand) wandb loss logging reads the real layout:
export ACTION_LAYOUT_PATH=/home/shaurya/g1_sonic_system1/results/action_layout.json

export CUDA_VISIBLE_DEVICES=0,1,2
export NUM_GPUS=3
export MASTER_PORT=29500
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
    --dataset-path /lambdafs/shaurya/g1_sonic_system1/data/g1_encoded_sonic \
    --embodiment-tag UNITREE_G1_SONIC \
    --output-dir /lambdafs/shaurya/g1_sonic_system1/outputs/grootN17-sft-handsout \
    --experiment-name grootN17-sft-handsout \
    --wandb-project g1-sonic-vla
